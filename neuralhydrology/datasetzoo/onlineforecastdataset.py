# import re
# from collections import defaultdict
from typing import List, Dict, Union
from functools import reduce
from pathlib import Path
# from pandas.tseries import frequencies
from pandas.tseries.frequencies import to_offset
from numba import NumbaPendingDeprecationWarning
from numba import njit, prange
# from ruamel.yaml import YAML
# from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import xarray as xr
import torch
import logging
import warnings
import sys
import pickle
import os

# from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.datasetzoo.genericdataset import GenericDataset
from neuralhydrology.datautils import utils
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import NoTrainDataError, NoEvaluationDataError
# from neuralhydrology.utils import samplingutils


LOGGER = logging.getLogger(__name__)


class OnlineForecastDataset(GenericDataset):
    """Online forecast dataset class for operational forecasting with mixed temporal indexing.
    
    This dataset handles operational forecast data where:
    - Hindcast variables are indexed by (basin, time) only - loaded from local CSV files
    - Forecast variables are indexed by (basin, time, lead_time) - loaded from online NOAA GEFS sources
    - Supports probabilistic forecasts with quartiles (q25, q50, q75)

    The dataset loads data from two sources:
    1. Historical data from local CSV files in data/harz/timeseries/hydromet_timeseries_{basin}.csv
    2. Forecast data from online NOAA GEFS 35-day forecast dataset with full pipeline:
       - Loads ensemble forecasts from NOAA GEFS zarr store
       - Computes ensemble quartiles (25th, 50th, 75th percentiles)
       - Interpolates to hourly resolution for first 240 hours
       - Extracts data for specific basin centroids

    Data structure:
    - Hindcast variables: shape (basin, time) - historical observations
    - Forecast variables: shape (basin, time, lead_time) - forecast ensemble quartiles
    - Mixed indexing allows operational forecasting workflows

    Supported basins: DE1, DE2, DE3, DE4, DE5 (Harz reservoir catchments)

    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool
        Defines if the dataset is used for training or evaluating.
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame.
    id_to_int : Dict[str, int], optional
        Basin id to integer mapping for one-hot encoding.
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        Feature scaling parameters.
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xr.DataArray]] = {}):
    
        # Initialize forecast-specific attributes that are filled in the data loading functions
        self._x_h = {}
        self._x_f = {}
    
        super().__init__(cfg=cfg,
                         is_train=is_train,
                         period=period,
                         basin=basin,
                         additional_features=additional_features,
                         id_to_int=id_to_int,
                         scaler=scaler)

    def _load_basin_data(self, basin: str, columns: list) -> pd.DataFrame:
        """Load input and output data for a specific basin."""
        # This method should not be used directly - instead use _load_hindcast_data or _load_forecast_data
        raise NotImplementedError("Use _load_hindcast_data or _load_forecast_data instead")
    
    def _load_hindcast_data(self, basin: str, columns: list) -> pd.DataFrame:
        """Load historical/hindcast data for a specific basin from local CSV files."""
        return load_local_historical_data_for_basin(
            data_dir=self.cfg.data_dir,
            basin=basin,
            columns=columns,
            start_date=getattr(self.cfg, 'forecast_start_date', None),
            end_date=getattr(self.cfg, 'forecast_end_date', None)
        )
    
    def _load_forecast_data(self, basin: str, columns: list) -> pd.DataFrame:
        """Load forecast data for a specific basin from online sources."""
        return load_online_forecast_data_for_basin(
            data_dir=self.cfg.data_dir,
            basin=basin,
            columns=columns,
            start_date=getattr(self.cfg, 'forecast_start_date', None),
            end_date=getattr(self.cfg, 'forecast_end_date', None)
        )

    def _initialize_frequency_configuration(self):
        """Checks and extracts configuration values for 'use_frequency', 'seq_length', and 'predict_last_n'"""

        self.seq_len = self.cfg.seq_length
        self._forecast_seq_len = self.cfg.forecast_seq_length
        self._predict_last_n = self.cfg.predict_last_n
        self._forecast_offset = self.cfg.forecast_offset

        # NOTE this dataset does not currently support multiple timestep frequencies. Instead 
        # we populate use_frequencies with the native frequency of the input data. 
        if self.cfg.use_frequencies:
            LOGGER.warning('Multiple timestep frequencies are not supported by this dataset: '
                           'defaulting to native frequency of input data')
        self.frequencies = []

        if not self.frequencies:
            if not isinstance(self.seq_len, int) or not isinstance(self._predict_last_n, int):
                raise ValueError('seq_length and predict_last_n must be integers')
            self.seq_len = [self.seq_len]
            self._forecast_seq_len = [self._forecast_seq_len]
            self._predict_last_n = [self._predict_last_n]

    def _load_or_create_xarray_dataset(self) -> xr.Dataset:
        # if no netCDF file is passed, data set is created from raw basin files and online sources
        if (self.cfg.train_data_file is None) or (not self.is_train):
            
            if not self._disable_pbar:
                LOGGER.info("Loading basin data into xarray data set using direct xarray approach.")
            
            # Load forecast data as xarray dataset (similar to operational notebook approach)
            forecast_ds = self._load_forecast_xarray_data()
            
            # Load historical data as xarray dataset 
            historical_ds = self._load_historical_xarray_data()
            
            # Merge forecast and historical datasets
            if forecast_ds is not None and historical_ds is not None:
                LOGGER.info("Merging forecast and historical datasets...")
                
                # Handle coordinate alignment - rename init_time to time in forecast data if needed
                if 'init_time' in forecast_ds.dims:
                    forecast_ds = forecast_ds.rename({'init_time': 'time'})
                
                # Find merge period: start from forecast data begin, end at historical data end
                forecast_start = forecast_ds['time'].min().values
                forecast_end = forecast_ds['time'].max().values
                historical_start = historical_ds['time'].min().values
                historical_end = historical_ds['time'].max().values
                
                # Use your specified logic: forecast start to historical end
                merge_start = forecast_start
                merge_end = historical_end
                
                # Validate that the merge period makes sense
                if merge_start > merge_end:
                    LOGGER.warning(f"Forecast starts ({merge_start}) after historical ends ({merge_end})")
                    LOGGER.warning("Using intersection period instead")
                    merge_start = max(forecast_start, historical_start)
                    merge_end = min(forecast_end, historical_end)
                
                LOGGER.info(f"Merge period: {merge_start} to {merge_end}")
                LOGGER.info(f"  Based on: forecast start to historical end logic")
                
                # Align both datasets to the merge period  
                forecast_aligned = forecast_ds.sel(time=slice(merge_start, merge_end))
                historical_aligned = historical_ds.sel(time=slice(merge_start, merge_end))
                
                LOGGER.info(f"Aligned forecast data: {len(forecast_aligned.time)} time steps")
                LOGGER.info(f"Aligned historical data: {len(historical_aligned.time)} time steps")
                
                # Merge the aligned datasets
                xr_dataset = xr.merge([historical_aligned, forecast_aligned], compat='override')
                
            elif forecast_ds is not None:
                # Only forecast data available
                if 'init_time' in forecast_ds.dims:
                    forecast_ds = forecast_ds.rename({'init_time': 'time'})
                xr_dataset = forecast_ds
                
            elif historical_ds is not None:
                # Only historical data available  
                xr_dataset = historical_ds
                
            else:
                # No data available
                if self.is_train:
                    raise NoTrainDataError
                else:
                    raise NoEvaluationDataError

            
            # Apply temporal slicing, warmup, and frequency handling
            xr_dataset = self._apply_temporal_processing(xr_dataset)
            
            # Save training data if requested
            if self.is_train and self.cfg.save_train_data:
                self._save_xarray_dataset(xr_dataset)

        else:
            # Reload previously-saved training data
            with self.cfg.train_data_file.open("rb") as fp:
                d = pickle.load(fp)
            xr_dataset = xr.Dataset.from_dict(d)
            if not self.frequencies:
                native_frequency = utils.infer_frequency(xr_dataset["time"].values)
                self.frequencies = [native_frequency]

        return xr_dataset

    def _load_forecast_xarray_data(self) -> xr.Dataset:
        """Load forecast data directly as xarray dataset following operational notebook approach."""
        try:
            # Import the functions from neuralhydrology datautils
            from neuralhydrology.datautils.fetch_basin_forecasts import (
                load_basin_centroids,
                fetch_forecasts_for_basins,
                interpolate_to_hourly,
            )
            
            LOGGER.info("Loading forecast data using operational pipeline...")
            
            # Check if any of the configured time periods fall before NOAA GEFS archive availability
            self._validate_forecast_archive_availability()
            
            # Load basin centroids (similar to operational notebook)
            basin_centroids_file = self.cfg.data_dir / "basin_centroids" / "basin_centroids.csv"
            if not basin_centroids_file.exists():
                LOGGER.warning(f"Basin centroids file not found: {basin_centroids_file}")
                return None
                
            centroids = load_basin_centroids(basin_centroids_file)
            
            # Filter centroids to only include basins we're working with
            centroids = centroids[centroids['basin_name'].isin(self.basins)]
            
            # Load NOAA GEFS forecast dataset
            LOGGER.info("Connecting to NOAA GEFS dataset...")
            ds = xr.open_zarr(
                "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr?email=optional@email.com", 
                decode_timedelta=True
            )
            
            # Extract forecasts for basin centroids
            basin_forecasts = fetch_forecasts_for_basins(ds, centroids)
            
            # Compute forecast quartiles as separate variables
            basin_forecasts_quartiles = self._compute_forecast_quartiles_as_variables(basin_forecasts)
            
            # Interpolate to hourly for first 240 hours (10 days)
            basin_forecasts_hourly = interpolate_to_hourly(basin_forecasts_quartiles, max_hours=240)
            
            # Filter to only include forecast_inputs from config
            forecast_vars = [var for var in basin_forecasts_hourly.data_vars if var in self.cfg.forecast_inputs]
            if forecast_vars:
                basin_forecasts_hourly = basin_forecasts_hourly[forecast_vars]
            
            LOGGER.info(f"Loaded forecast data with variables: {list(basin_forecasts_hourly.data_vars)}")
            return basin_forecasts_hourly
            
        except ImportError:
            LOGGER.warning("Could not import forecast data functions. Forecast data will not be loaded.")
            return None
        except Exception as e:
            LOGGER.warning(f"Error loading forecast data: {e}")
            return None
            
    def _load_historical_xarray_data(self) -> xr.Dataset:
        """Load historical data directly as xarray dataset."""
        try:
            LOGGER.info("Loading historical data from local CSV files...")
            
            # Get date range from config or use defaults
            start_date = getattr(self.cfg, 'train_start_date', '2000-01-01')
            end_date = getattr(self.cfg, 'train_end_date', '2023-12-31')
            
            # Convert to datetime if needed
            if hasattr(start_date, 'strftime'):
                start_date = start_date.strftime('%Y-%m-%d')
            if hasattr(end_date, 'strftime'):
                end_date = end_date.strftime('%Y-%m-%d')
                
            basin_data = {}
            basin_names = []
            
            # Load data for each basin
            for basin in self.basins:
                basin_names.append(basin)
                
                # Construct file path
                csv_file = self.cfg.data_dir / "timeseries" / f"hydromet_timeseries_{basin}.csv"
                
                if not csv_file.exists():
                    LOGGER.warning(f"File not found for basin {basin}: {csv_file}")
                    continue
                    
                LOGGER.info(f"Loading data for basin {basin} from {csv_file}")
                
                # Read CSV file
                df = pd.read_csv(csv_file)
                df['date'] = pd.to_datetime(df['date'])
                
                # Filter to requested date range
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
                df_filtered = df[mask].copy()
                
                if df_filtered.empty:
                    LOGGER.warning(f"No data in date range for basin {basin}")
                    continue
                    
                # Set date as index and sort
                df_filtered = df_filtered.set_index('date').sort_index()
                
                # Filter to only include hindcast_inputs and target_variables from config
                available_cols = df_filtered.columns.tolist()
                wanted_cols = self.cfg.hindcast_inputs + self.cfg.target_variables
                keep_cols = [col for col in wanted_cols if col in available_cols]
                
                if keep_cols:
                    df_filtered = df_filtered[keep_cols]
                    basin_data[basin] = df_filtered
                    LOGGER.info(f"Loaded {len(df_filtered)} records for basin {basin} with variables: {keep_cols}")
                else:
                    LOGGER.warning(f"No requested variables found for basin {basin}")
            
            if not basin_data:
                LOGGER.warning("No historical data loaded for any basin")
                return None
            
            # Convert to xarray Dataset
            all_times = set()
            for df in basin_data.values():
                all_times.update(df.index)
            all_times = sorted(list(all_times))
            
            # Get all variable names
            sample_df = next(iter(basin_data.values()))
            variable_names = sample_df.columns.tolist()
            
            # Create data arrays for each variable
            data_vars = {}
            for var_name in variable_names:
                # Create 2D array: (basin, time)
                data_array = np.full((len(basin_names), len(all_times)), np.nan)
                
                for i, basin_name in enumerate(basin_names):
                    if basin_name in basin_data:
                        df = basin_data[basin_name]
                        if var_name in df.columns:
                            # Reindex to match all_times, filling missing with NaN
                            var_series = df[var_name].reindex(all_times)
                            data_array[i, :] = var_series.values
                
                data_vars[var_name] = (['basin', 'time'], data_array)
            
            # Create coordinates
            coords = {
                'basin': basin_names,
                'time': all_times
            }
            
            # Create Dataset
            historical_ds = xr.Dataset(data_vars, coords=coords)
            
            # Add attributes
            historical_ds.attrs['source'] = 'Local CSV files'
            historical_ds.attrs['date_range'] = f"{start_date} to {end_date}"
            
            LOGGER.info(f"Created historical dataset with {len(basin_names)} basins, {len(all_times)} time steps, and variables: {variable_names}")
            return historical_ds
            
        except Exception as e:
            LOGGER.warning(f"Error loading historical data: {e}")
            return None
            
    def _compute_forecast_quartiles_as_variables(self, forecast_ds, quartiles=[0.25, 0.5, 0.75]):
        """Compute quartiles from ensemble forecast data as separate variables."""
        LOGGER.info(f"Computing quartiles {quartiles} from ensemble forecasts as separate variables...")
        
        # Define quartile suffixes
        quartile_suffixes = {
            0.25: '_q25',
            0.5: '_q50', 
            0.75: '_q75'
        }
        
        # Create new dataset with quartile variables
        new_data_vars = {}
        
        # Process each data variable
        for var_name in forecast_ds.data_vars:
            var_data = forecast_ds[var_name]
            
            # Compute quartiles for this variable
            var_quartiles = var_data.quantile(quartiles, dim='ensemble_member')
            
            # Create separate variables for each quartile
            for i, q in enumerate(quartiles):
                suffix = quartile_suffixes.get(q, f'_q{int(q*100)}')
                new_var_name = f"{var_name}{suffix}"
                
                # Extract the quartile data (remove the quantile dimension)
                quartile_data = var_quartiles.isel(quantile=i).drop('quantile')
                
                # Add to new data variables
                new_data_vars[new_var_name] = quartile_data
        
        # Create new dataset with the same coordinates (excluding ensemble_member)
        coords_to_keep = {k: v for k, v in forecast_ds.coords.items() if 'ensemble_member' not in v.dims}
        
        # Create the new dataset
        quartile_ds = xr.Dataset(
            data_vars=new_data_vars,
            coords=coords_to_keep,
            attrs=forecast_ds.attrs.copy()
        )
        
        # Update attributes
        quartile_ds.attrs['quartile_processing'] = f'Computed quartiles {quartiles} as separate variables'
        quartile_ds.attrs['original_ensemble_members'] = len(forecast_ds.ensemble_member)
        
        return quartile_ds
        
    def _apply_temporal_processing(self, xr_dataset: xr.Dataset) -> xr.Dataset:
        """Apply temporal slicing, warmup periods, and frequency handling to the dataset."""
        
        # Infer native frequency from time coordinate
        time_coord = 'time' if 'time' in xr_dataset.dims else 'date'
        native_frequency = utils.infer_frequency(xr_dataset[time_coord].values)
        if not self.frequencies:
            self.frequencies = [native_frequency]
        
        # Get temporal periods for processing
        processed_datasets = []
        
        for basin in self.basins:
            if basin not in xr_dataset.basin.values:
                continue
                
            basin_ds = xr_dataset.sel(basin=basin)
            
            # Get start and end dates for this basin
            start_dates = self.start_and_end_dates[basin]["start_dates"]
            end_dates = [
                date + pd.Timedelta(days=1, seconds=-1) for date in self.start_and_end_dates[basin]["end_dates"]
            ]
            
            # Calculate warmup offsets
            offsets = [(self.seq_len[i] - max(self._predict_last_n[i], self._forecast_seq_len[i])) * to_offset(freq)
                       for i, freq in enumerate(self.frequencies)]
            
            basin_slices = []
            
            # Process each temporal slice
            for start_date, end_date in zip(start_dates, end_dates):
                # Add warmup period
                warmup_start_date = min(start_date - offset for offset in offsets)
                
                # Select time range
                time_slice = basin_ds.sel({time_coord: slice(warmup_start_date, end_date)})
                
                # Set targets in warmup period to NaN
                for target_var in self.cfg.target_variables:
                    if target_var in time_slice.data_vars:
                        warmup_mask = time_slice[time_coord] < start_date
                        time_slice[target_var] = time_slice[target_var].where(~warmup_mask, np.nan)
                
                basin_slices.append(time_slice)
            
            if basin_slices:
                # Concatenate temporal slices
                basin_combined = xr.concat(basin_slices, dim=time_coord)
                basin_combined = basin_combined.assign_coords(basin=basin)
                processed_datasets.append(basin_combined)
        
        if processed_datasets:
            # Combine all basins
            final_dataset = xr.concat(processed_datasets, dim="basin")
            return final_dataset
        else:
            return xr_dataset

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        basin, indices = self.lookup_table[item]

        sample = {}
        for freq, seq_len, forecast_seq_len, idx in zip(self.frequencies, self.seq_len, self._forecast_seq_len, indices):
            # if there's just one frequency, don't use suffixes.
            freq_suffix = '' if len(self.frequencies) == 1 else f'_{freq}'
            
            # Calculate indices for hindcast and forecast periods
            hindcast_start_idx = idx + self._forecast_offset + forecast_seq_len - seq_len
            hindcast_end_idx = idx + self._forecast_offset
            forecast_start_idx = idx
            global_end_idx = idx + self._forecast_offset + forecast_seq_len 

            sample[f'x_h{freq_suffix}'] = self._x_h[basin][freq][hindcast_start_idx:hindcast_end_idx]
            sample[f'x_f{freq_suffix}'] = self._x_f[basin][freq][forecast_start_idx]
            sample[f'y{freq_suffix}'] = self._y[basin][freq][hindcast_start_idx:global_end_idx]
            sample[f'date{freq_suffix}'] = self._dates[basin][freq][hindcast_start_idx:global_end_idx]

            # Handle static inputs
            static_inputs = []
            if self._attributes:
                static_inputs.append(self._attributes[basin])
            if self._x_s:
                static_inputs.append(self._x_s[basin][freq][idx])
            if static_inputs:
                sample[f'x_s{freq_suffix}'] = torch.cat(static_inputs, dim=-1)

            if self.cfg.timestep_counter:
                sample[f'x_h{freq_suffix}'] = torch.concatenate([sample[f'x_h{freq_suffix}'], self.hindcast_counter], dim=-1)
                sample[f'x_f{freq_suffix}'] = torch.concatenate([sample[f'x_f{freq_suffix}'], self.forecast_counter], dim=-1)

        if self._per_basin_target_stds:
            sample['per_basin_target_stds'] = self._per_basin_target_stds[basin]
        if self.id_to_int:
            sample['x_one_hot'] = torch.nn.functional.one_hot(torch.tensor(self.id_to_int[basin]),
                                                              num_classes=len(self.id_to_int)).to(torch.float32)

        return sample
    
    def _create_lookup_table(self, xr_dataset: xr.Dataset):
        lookup = []
        if not self._disable_pbar:
            LOGGER.info("Create lookup table and convert to pytorch tensor")

        # Split data into forecast and hindcast components
        xr_fcst = xr_dataset[self.cfg.forecast_inputs]
        xr_hcst = xr_dataset[[var for var in xr_dataset.variables if var not in self.cfg.forecast_inputs]]
        
        # Drop lead_time dimension from hindcast data if it exists
        if 'lead_time' in xr_hcst.dims:
            xr_hcst = xr_hcst.drop_dims('lead_time')

        # Use 'time' as the time dimension name (instead of 'date')
        time_dim = 'time' if 'time' in xr_hcst.dims else 'date'

        basins_without_samples = []
        basin_coordinates = xr_hcst["basin"].values.tolist()
        
        for basin in tqdm(basin_coordinates, file=sys.stdout, disable=self._disable_pbar):
            x_h, x_f, x_s, y, dates = {}, {}, {}, {}, {}
            frequency_maps = {}
            lowest_freq = utils.sort_frequencies(self.frequencies)[0]

            # Convert to pandas DataFrame for easier handling
            df_hcst_native = xr_hcst.sel(basin=basin, drop=True).to_dataframe()
            df_fcst_native = xr_fcst.sel(basin=basin, drop=True).to_dataframe()

            for freq in self.frequencies:
                # No resampling - use native resolution
                df_hcst_resampled = df_hcst_native
                df_fcst_resampled = df_fcst_native
            
                # Extract hindcast inputs
                x_h[freq] = df_hcst_resampled[self.cfg.hindcast_inputs].values
                
                # Extract forecast inputs - handle MultiIndex properly
                if isinstance(df_fcst_resampled.index, pd.MultiIndex):
                    # Convert MultiIndex DataFrame to 3D array: (time, lead_time, variables)
                    fcst_xr = df_fcst_resampled[self.cfg.forecast_inputs].to_xarray()
                    x_f[freq] = fcst_xr.to_array().transpose('time', 'lead_time', 'variable').values
                else:
                    x_f[freq] = df_fcst_resampled[self.cfg.forecast_inputs].values
                
                # Extract targets from hindcast data
                y[freq] = df_hcst_resampled[self.cfg.target_variables].values
                
                # Store dates
                dates[freq] = df_hcst_resampled.index.to_numpy()

                # Create frequency mapping
                frequency_factor = int(utils.get_frequency_factor(lowest_freq, freq))
                if len(df_hcst_resampled) % frequency_factor != 0:
                    raise ValueError(f'The length of the dataframe at frequency {freq} is {len(df_hcst_resampled)} '
                                     f'(including warmup), which is not a multiple of {frequency_factor}.')
                frequency_maps[freq] = np.arange(len(df_hcst_resampled) // frequency_factor) \
                                       * frequency_factor + (frequency_factor - 1)

            # Store period start for inference
            if not self.is_train:
                self.period_starts[basin] = pd.to_datetime(xr_hcst.sel(basin=basin)[time_dim].values[0])

            # Validate samples
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

                flag = validate_samples(
                    x_h=[x_h[freq] for freq in self.frequencies] if self.is_train else None,
                    x_f=[x_f[freq] for freq in self.frequencies] if self.is_train else None,
                    y=[y[freq] for freq in self.frequencies] if self.is_train else None,
                    seq_length=self.seq_len,
                    forecast_seq_length=self._forecast_seq_len,
                    forecast_offset=self._forecast_offset,
                    predict_last_n=self._predict_last_n,
                    frequency_maps=[frequency_maps[freq] for freq in self.frequencies]
                )

            valid_samples = np.argwhere(flag == 1)
            for f in valid_samples:
                lookup.append((basin, [frequency_maps[freq][int(f)] for freq in self.frequencies]))

            # Store data for basins with valid samples
            if valid_samples.size > 0:
                if not self.cfg.hindcast_inputs:
                    raise ValueError('Hindcast inputs must be provided if forecast inputs are provided.')
                self._x_h[basin] = {freq: torch.from_numpy(_x_h.astype(np.float32)) for freq, _x_h in x_h.items()}
                self._x_f[basin] = {freq: torch.from_numpy(_x_f.astype(np.float32)) for freq, _x_f in x_f.items()}
                self._y[basin] = {freq: torch.from_numpy(_y.astype(np.float32)) for freq, _y in y.items()}
                self._dates[basin] = dates
            else:
                basins_without_samples.append(basin)

        if basins_without_samples:
            LOGGER.info(f"These basins do not have a single valid sample in the {self.period} period: {basins_without_samples}")
        
        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}
        self.num_samples = len(self.lookup_table)
        
        if self.num_samples == 0:
            if self.is_train:
                raise NoTrainDataError
            else:
                raise NoEvaluationDataError

    def _validate_forecast_archive_availability(self):
        """Validate that training, validation, and test periods don't extend before NOAA GEFS archive availability.
        
        The NOAA GEFS forecast archive is only available from 2000-01-01T00:00:00 onwards.
        This method checks all configured time periods and raises an error if any period
        starts before this date.
        
        Raises
        ------
        ValueError
            If any time period starts before 2000-01-01
        """
        # NOAA GEFS forecast archive availability starts from this date
        archive_start_date = pd.to_datetime('2000-01-01')
        
        # Get all configured time periods
        time_periods = []
        
        # Check training period
        if hasattr(self.cfg, 'train_start_date') and self.cfg.train_start_date:
            train_start = pd.to_datetime(self.cfg.train_start_date)
            time_periods.append(('training', train_start))
        
        # Check validation period
        if hasattr(self.cfg, 'validation_start_date') and self.cfg.validation_start_date:
            validation_start = pd.to_datetime(self.cfg.validation_start_date)
            time_periods.append(('validation', validation_start))
        
        # Check test period
        if hasattr(self.cfg, 'test_start_date') and self.cfg.test_start_date:
            test_start = pd.to_datetime(self.cfg.test_start_date)
            time_periods.append(('test', test_start))
        
        # Check each period
        invalid_periods = []
        for period_name, period_start in time_periods:
            if period_start < archive_start_date:
                invalid_periods.append((period_name, period_start))
        
        if invalid_periods:
            error_msg = (
                f"NOAA GEFS forecast archive is only available from {archive_start_date.strftime('%Y-%m-%d')} onwards. "
                f"The following time periods extend before this date and cannot be used:\n"
            )
            for period_name, period_start in invalid_periods:
                error_msg += f"  - {period_name} period starts at {period_start.strftime('%Y-%m-%d')}\n"
            
            error_msg += (
                f"\nPlease adjust your configuration to use dates from {archive_start_date.strftime('%Y-%m-%d')} onwards "
                f"or disable forecast data loading for earlier periods."
            )
            
            LOGGER.error(error_msg)
            raise ValueError(error_msg)

def load_local_historical_data_for_basin(data_dir: Path, basin: str, columns: list, 
                                        start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Load historical data from local CSV files for a specific basin.
    
    Parameters
    ----------
    data_dir : Path
        Path to the data directory containing harz data
    basin : str
        The basin identifier (e.g., 'DE1', 'DE2', etc.)
    columns : list
        List of variable names to extract
    start_date : str, optional
        Start date for data loading (YYYY-MM-DD format)
    end_date : str, optional
        End date for data loading (YYYY-MM-DD format)
        
    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame containing historical data
    """
    timeseries_dir = data_dir / "harz" / "timeseries"
    csv_file = timeseries_dir / f"hydromet_timeseries_{basin}.csv"
    
    if not csv_file.exists():
        LOGGER.warning(f"Historical data file not found for basin {basin}: {csv_file}")
        return pd.DataFrame(index=pd.DatetimeIndex([], name='time'))
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter to date range if specified
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
            df = df[mask]
        
        # Set date as index and rename to 'time' for consistency
        df = df.set_index('date').sort_index()
        df.index.name = 'time'
        
        # Filter to requested variables if specified
        if columns:
            available_vars = [var for var in columns if var in df.columns]
            if available_vars:
                df = df[available_vars]
            else:
                LOGGER.warning(f"No requested variables found in historical data for basin {basin}")
                return pd.DataFrame(index=pd.DatetimeIndex([], name='time'))
        
        return df
        
    except Exception as e:
        LOGGER.error(f"Error loading historical data for basin {basin}: {e}")
        return pd.DataFrame(index=pd.DatetimeIndex([], name='time'))


def load_online_forecast_data_for_basin(data_dir: Path, basin: str, columns: list,
                                       start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Load forecast data from online NOAA GEFS dataset for a specific basin.
    
    This function implements the full NOAA GEFS forecast loading pipeline:
    1. Load basin centroid coordinates
    2. Fetch forecasts from NOAA GEFS zarr dataset
    3. Compute ensemble quartiles (q25, q50, q75)
    4. Interpolate to hourly resolution
    5. Return as MultiIndex DataFrame
    
    Parameters
    ----------
    data_dir : Path
        Path to the data directory (used to find basin metadata)
    basin : str
        The basin identifier (e.g., 'DE1', 'DE2', etc.)
    columns : list
        List of forecast variable names to extract
    start_date : str, optional
        Start date for data loading (YYYY-MM-DD format)
    end_date : str, optional
        End date for data loading (YYYY-MM-DD format)
        
    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame with (time, lead_time) index containing forecast data
    """
    try:
        # Import required modules
        from neuralhydrology.datautils.fetch_basin_forecasts import (
            load_basin_centroids, fetch_forecasts_for_basins, interpolate_to_hourly
        )
        
        # Load basin centroids
        basin_centroids_file = data_dir / "harz" / "basin_centroids" / "basin_centroids.csv"
        if not basin_centroids_file.exists():
            LOGGER.error(f"Basin centroids file not found: {basin_centroids_file}")
            return _create_empty_forecast_dataframe()
        
        centroids = load_basin_centroids(str(basin_centroids_file))
        
        # Map basin names - the centroids file uses different naming convention
        basin_name_mapping = {
            'DE1': 'innerste_reservoir_catchment_Basin_0',
            'DE2': 'oker_reservoir_catchment_Basin_0', 
            'DE3': 'ecker_reservoir_catchment_Basin_0',
            'DE4': 'soese_reservoir_catchment_Basin_0',
            'DE5': 'grane_reservoir_catchment_Basin_0'
        }
        
        if basin not in basin_name_mapping:
            LOGGER.warning(f"Basin {basin} not found in basin mapping. Available basins: {list(basin_name_mapping.keys())}")
            return _create_empty_forecast_dataframe()
        
        # Filter to specific basin
        mapped_basin_name = basin_name_mapping[basin]
        basin_centroid = centroids[centroids['basin_name'] == mapped_basin_name]
        
        if basin_centroid.empty:
            LOGGER.warning(f"Centroid not found for mapped basin {mapped_basin_name}")
            return _create_empty_forecast_dataframe()
        
        # Load NOAA GEFS dataset
        LOGGER.info(f"Loading NOAA GEFS data for basin {basin}...")
        zarr_url = "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr?email=optional@email.com"
        ds = xr.open_zarr(zarr_url, decode_timedelta=True)
        LOGGER.info(f"NOAA GEFS dataset loaded successfully with dimensions: {dict(ds.dims)}")
        
        # Filter to date range if specified
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            ds = ds.sel(init_time=slice(start_dt, end_dt))
            LOGGER.info(f"Filtered to date range: {start_date} to {end_date}")
        
        # Extract forecasts for the basin
        LOGGER.info(f"Extracting forecasts for basin {basin} (mapped: {mapped_basin_name})...")
        basin_forecasts = fetch_forecasts_for_basins(ds, basin_centroid, init_time=None)
        LOGGER.info(f"Extracted forecasts with dimensions: {dict(basin_forecasts.dims)}")
        
        # Update basin coordinate to use DE1-DE5 naming for consistency
        basin_forecasts = basin_forecasts.assign_coords(basin=[basin])
        
        # Compute quartiles as separate variables
        LOGGER.info(f"Computing forecast quartiles (25th, 50th, 75th percentiles)...")
        forecast_quartiles = compute_forecast_quartiles_as_variables(basin_forecasts)
        LOGGER.info(f"Created {len(forecast_quartiles.data_vars)} quartile variables")
        
        # Interpolate to hourly resolution for first 240 hours (10 days)
        LOGGER.info(f"Interpolating to hourly resolution for first 240 hours...")
        forecast_hourly = interpolate_to_hourly(forecast_quartiles, max_hours=240)
        LOGGER.info(f"Interpolated to {len(forecast_hourly.lead_time)} hourly lead time steps")
        
        # Convert to DataFrame with MultiIndex
        LOGGER.info(f"Converting to DataFrame with MultiIndex (time, lead_time)...")
        df = _convert_forecast_to_dataframe(forecast_hourly, basin, columns)
        
        LOGGER.info(f"✅ Successfully loaded forecast data for basin {basin}")
        LOGGER.info(f"   Final DataFrame shape: {df.shape}")
        LOGGER.info(f"   Available variables: {list(df.columns)}")
        if not df.empty:
            LOGGER.info(f"   Time range: {df.index.get_level_values('time').min()} to {df.index.get_level_values('time').max()}")
            LOGGER.info(f"   Lead time range: {df.index.get_level_values('lead_time').min()} to {df.index.get_level_values('lead_time').max()} hours")
        
        return df
        
    except Exception as e:
        LOGGER.error(f"Error loading forecast data for basin {basin}: {e}")
        return _create_empty_forecast_dataframe()


def compute_forecast_quartiles_as_variables(forecast_ds: xr.Dataset, 
                                          quartiles: list = [0.25, 0.5, 0.75]) -> xr.Dataset:
    """Compute quartiles from ensemble forecast data as separate variables.
    
    Parameters
    ----------
    forecast_ds : xr.Dataset
        Dataset with ensemble forecasts containing 'ensemble_member' dimension
    quartiles : list, optional
        List of quantiles to compute (default: [0.25, 0.5, 0.75])
        
    Returns
    -------
    xr.Dataset
        Dataset with quartile variables (e.g., temperature_2m_q25, temperature_2m_q50, etc.)
    """
    quartile_suffixes = {0.25: '_q25', 0.5: '_q50', 0.75: '_q75'}
    new_data_vars = {}
    
    for var_name in forecast_ds.data_vars:
        if 'ensemble_member' not in forecast_ds[var_name].dims:
            # Variable doesn't have ensemble dimension, keep as is
            new_data_vars[var_name] = forecast_ds[var_name]
            continue
            
        var_data = forecast_ds[var_name]
        var_quartiles = var_data.quantile(quartiles, dim='ensemble_member')
        
        for i, q in enumerate(quartiles):
            suffix = quartile_suffixes.get(q, f'_q{int(q*100)}')
            new_var_name = f"{var_name}{suffix}"
            quartile_data = var_quartiles.isel(quantile=i).drop('quantile')
            new_data_vars[new_var_name] = quartile_data
    
    # Keep coordinates that don't have ensemble_member dimension
    coords_to_keep = {k: v for k, v in forecast_ds.coords.items() 
                     if 'ensemble_member' not in v.dims}
    
    return xr.Dataset(new_data_vars, coords=coords_to_keep, attrs=forecast_ds.attrs)


def _convert_forecast_to_dataframe(forecast_ds: xr.Dataset, basin_name: str, columns: list) -> pd.DataFrame:
    """Convert forecast xarray Dataset to pandas DataFrame with MultiIndex.
    
    Parameters
    ----------
    forecast_ds : xr.Dataset
        Forecast dataset with time and lead_time dimensions
    basin_name : str
        Name of the basin
    columns : list
        List of variable columns to include
        
    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame with (time, lead_time) index
    """
    # Select data for the specific basin
    if 'basin' in forecast_ds.dims:
        basin_data = forecast_ds.sel(basin=basin_name)
    else:
        basin_data = forecast_ds
    
    # Convert to DataFrame 
    df = basin_data.to_dataframe()
    
    # Ensure we have the right index structure
    if 'init_time' in df.index.names:
        df = df.reset_index()
        df = df.rename(columns={'init_time': 'time'})
        df = df.set_index(['time', 'lead_time'])
    elif df.index.names != ['time', 'lead_time']:
        # Try to reconstruct proper MultiIndex
        df = df.reset_index()
        time_cols = [col for col in df.columns if 'time' in col.lower()]
        lead_cols = [col for col in df.columns if 'lead' in col.lower()]
        
        if time_cols and lead_cols:
            df = df.set_index([time_cols[0], lead_cols[0]])
            df.index.names = ['time', 'lead_time']
    
    # Filter to requested columns
    if columns:
        available_vars = [var for var in columns if var in df.columns]
        if available_vars:
            df = df[available_vars]
        else:
            LOGGER.warning(f"No requested variables found in forecast data. Available: {list(df.columns)}")
            return _create_empty_forecast_dataframe()
    
    return df


def _create_empty_forecast_dataframe() -> pd.DataFrame:
    """Create an empty DataFrame with proper MultiIndex structure for forecasts."""
    multi_index = pd.MultiIndex.from_product(
        [pd.DatetimeIndex([]), pd.Index([])], 
        names=['time', 'lead_time']
    )
    return pd.DataFrame(index=multi_index)


# Use the same validation function from the original ForecastDataset
@njit()
def validate_samples(x_h: List[np.ndarray], 
                     x_f: List[np.ndarray],
                     y: List[np.ndarray], 
                     seq_length: List[int],
                     forecast_seq_length: List[int],
                     forecast_offset: int,
                     predict_last_n: List[int], 
                     frequency_maps: List[np.ndarray]) -> np.ndarray:
    """Checks for invalid samples due to NaN or insufficient sequence length.

    Parameters
    ----------
    x_h : List[np.ndarray]
        List of dynamic hindcast input data; one entry per frequency
    x_f : List[np.ndarray]
        List of dynamic forecast input data; one entry per frequency
    y : List[np.ndarray]
        List of target values; one entry per frequency
    seq_length : List[int]
        List of sequence lengths; one entry per frequency
    forecast_seq_length : List[int]
        List of forecast sequence lengths; one entry per frequency
    forecast_offset : int
        Number of timesteps between hindcast end and forecast start
    predict_last_n: List[int]
        List of predict_last_n; one entry per frequency
    frequency_maps : List[np.ndarray]
        List of arrays mapping lowest-frequency samples to their corresponding last sample in each frequency

    Returns
    -------
    np.ndarray 
        Array has a value of 1 for valid samples and a value of 0 for invalid samples.
    """
    n_samples = len(frequency_maps[0])
    flag = np.ones(n_samples)
    
    for i in range(len(frequency_maps)):
        for j in prange(n_samples):
            hindcast_seq_length = seq_length[i] - forecast_seq_length[i] 
            last_sample_of_freq = frequency_maps[i][j]
            
            # Check sufficient history
            if last_sample_of_freq < hindcast_seq_length:
                flag[j] = 0
                continue

            # Check sufficient future data
            if (last_sample_of_freq + forecast_offset + forecast_seq_length[i]) > n_samples:
                flag[j] = 0
                continue 

            # Check for NaN in hindcast inputs
            if x_h is not None:
                _x_h = x_h[i][last_sample_of_freq - hindcast_seq_length + 1:last_sample_of_freq + 1]
                if np.any(np.isnan(_x_h)):
                    flag[j] = 0
                    continue

            # Check for NaN in forecast inputs
            if x_f is not None:
                _x_f = x_f[i][last_sample_of_freq]
                if np.any(np.isnan(_x_f)):
                    flag[j] = 0
                    continue

            # Check for all-NaN targets
            if y is not None:
                _y = y[i][last_sample_of_freq - predict_last_n[i] + 1:last_sample_of_freq + 1]
                if np.prod(np.array(_y.shape)) > 0 and np.all(np.isnan(_y)):
                    flag[j] = 0
                    continue

    return flag
