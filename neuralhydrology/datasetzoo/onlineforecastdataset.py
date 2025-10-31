from pathlib import Path
import logging
import pickle
import shutil
import sys
import time
import warnings
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import torch
import xarray as xr
from numba import NumbaPendingDeprecationWarning, njit, prange
from pandas.tseries.frequencies import to_offset
from tqdm import tqdm

from neuralhydrology.datasetzoo.genericdataset import GenericDataset
from neuralhydrology.datautils import utils
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import NoEvaluationDataError, NoTrainDataError


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
    
        # Validate data availability before initializing
        self._validate_data_availability(cfg)

        super().__init__(cfg=cfg,
                         is_train=is_train,
                         period=period,
                         basin=basin,
                         additional_features=additional_features,
                         id_to_int=id_to_int,
                         scaler=scaler)

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
        train_data_file: Optional[Path] = getattr(self.cfg, 'train_data_file', None)

        cached_dataset = self._load_cached_dataset(train_data_file)
        if cached_dataset is not None:
                return cached_dataset

        if train_data_file is None or not self.is_train:
            dataset = self._build_dataset_from_sources()
            if self.is_train and self.cfg.save_train_data:
                self._save_dataset_cache(dataset)
            return dataset

        dataset = self._load_pickled_dataset(Path(train_data_file))
        if not self.frequencies:
            native_frequency = utils.infer_frequency(dataset["time"].values)
            self.frequencies = [native_frequency]
        return dataset

    def _build_dataset_from_sources(self) -> xr.Dataset:
        if not self._disable_pbar:
            LOGGER.info("Loading basin data into xarray data set using direct xarray approach.")

        forecast_ds = self._load_forecast_xarray_data()
        historical_ds = self._load_historical_xarray_data()

        if forecast_ds is None or historical_ds is None:
            if self.is_train:
                raise NoTrainDataError
            raise NoEvaluationDataError

        if 'init_time' in forecast_ds.dims:
            forecast_ds = forecast_ds.rename({'init_time': 'time'})

        if 'time' not in forecast_ds.dims or 'time' not in historical_ds.dims:
            raise ValueError('Both forecast and historical datasets must provide a time dimension.')

        if not self.frequencies:
            inferred_freq = utils.infer_frequency(historical_ds['time'].values)
            if inferred_freq is None:
                raise ValueError('Could not infer native frequency from historical dataset.')
            self.frequencies = [inferred_freq]

        offsets = [(self.seq_len[i] - max(self._predict_last_n[i], self._forecast_seq_len[i])) * to_offset(freq)
                   for i, freq in enumerate(self.frequencies)]
        retry_attempts, retry_delay = self._get_retry_config()

        basin_datasets = []
        basins_without_data: List[str] = []

        for basin in self.basins:
            if 'basin' not in historical_ds.coords or basin not in historical_ds['basin'].values:
                LOGGER.warning("Historical data not available for basin %s - skipping.", basin)
                basins_without_data.append(basin)
                continue
            if 'basin' not in forecast_ds.coords or basin not in forecast_ds['basin'].values:
                LOGGER.warning("Forecast data not available for basin %s - skipping.", basin)
                basins_without_data.append(basin)
                continue
            if basin not in self.start_and_end_dates:
                LOGGER.warning("No temporal configuration found for basin %s - skipping.", basin)
                basins_without_data.append(basin)
                continue

            hist_basin = historical_ds.sel(basin=basin, drop=False)
            fcst_basin = forecast_ds.sel(basin=basin, drop=False)

            native_frequency = utils.infer_frequency(hist_basin['time'].values) or self.frequencies[0]

            start_dates = self.start_and_end_dates[basin]["start_dates"]
            end_dates = [date + pd.Timedelta(days=1, seconds=-1)
                         for date in self.start_and_end_dates[basin]["end_dates"]]

            hist_slices = []
            fcst_slices = []

            for start_date, end_date in zip(start_dates, end_dates):
                warmup_start = min(start_date - offset for offset in offsets)

                full_hist_index = pd.date_range(start=warmup_start,
                                                end=end_date,
                                                freq=native_frequency)
                period_index = pd.date_range(start=start_date,
                                             end=end_date,
                                             freq=native_frequency)

                hist_slice = hist_basin.sel(time=slice(warmup_start, end_date)).reindex({'time': full_hist_index})
                fcst_slice = self._materialize_forecast_slice(
                    fcst_basin=fcst_basin,
                    basin=basin,
                    start_date=start_date,
                    end_date=end_date,
                    period_index=period_index,
                    attempts=retry_attempts,
                    delay=retry_delay,
                )

                warmup_mask = hist_slice['time'] < np.datetime64(start_date)
                for target_var in self.cfg.target_variables:
                    if target_var in hist_slice.data_vars:
                        hist_slice[target_var] = hist_slice[target_var].where(~warmup_mask, np.nan)

                hist_slices.append(hist_slice)
                fcst_slices.append(fcst_slice)

            if not hist_slices or not fcst_slices:
                LOGGER.warning("No temporal slices created for basin %s - skipping.", basin)
                basins_without_data.append(basin)
                continue

            hist_combined = xr.concat(hist_slices, dim='time').sortby('time')
            fcst_combined = xr.concat(fcst_slices, dim='time').sortby('time')

            hist_index = pd.Index(hist_combined['time'].values)
            if hist_index.duplicated().any():
                hist_combined = hist_combined.isel(time=~hist_index.duplicated(keep='last'))

            fcst_index = pd.Index(fcst_combined['time'].values)
            if fcst_index.duplicated().any():
                fcst_combined = fcst_combined.isel(time=~fcst_index.duplicated(keep='last'))

            basin_datasets.append(xr.merge([hist_combined, fcst_combined], compat='override'))

        if basins_without_data:
            LOGGER.info("Skipped basins without complete data: %s", sorted(set(basins_without_data)))

        if not basin_datasets:
            if self.is_train:
                raise NoTrainDataError
            raise NoEvaluationDataError

        return xr.concat(basin_datasets, dim='basin')

    def _load_cached_dataset(self, train_data_file: Optional[Path]) -> Optional[xr.Dataset]:
        for cache_path in self._cache_candidates(train_data_file):
            if not cache_path.exists() or not (cache_path.is_dir() or cache_path.suffix == '.zarr'):
                continue

            LOGGER.info("Loading cached dataset from %s", cache_path)
        dataset = xr.open_zarr(store=cache_path)
        if not self.frequencies:
            native_frequency = utils.infer_frequency(dataset["time"].values)
            self.frequencies = [native_frequency]
        if getattr(self.cfg, "plot_cached_dataset_preview", False):
            self._plot_cached_dataset(dataset)
        return dataset

        return None

    def _determine_cache_path(self) -> Path:
        candidates = self._cache_candidates(getattr(self.cfg, 'train_data_file', None))
        return next((path for path in candidates if path is not None), self.cfg.train_dir / 'train_data.zarr')

    def _save_dataset_cache(self, dataset: xr.Dataset) -> None:
        cache_path = self._determine_cache_path()
        if cache_path.exists():
            LOGGER.info("Removing existing cache at %s", cache_path)
            if cache_path.is_dir():
                shutil.rmtree(cache_path)
            else:
                cache_path.unlink()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Saving materialized dataset to %s", cache_path)
        dataset.chunk({'basin': max(1, len(self.basins))}).to_zarr(store=cache_path, mode='w')

    def _cache_candidates(self, train_data_file: Optional[Path]) -> List[Path]:
        train_dir = getattr(self.cfg, 'train_dir', None)
        if train_dir is None:
            return []

        train_dir_path = Path(train_dir)
        return [train_dir_path / 'train_data.zarr']

    def _plot_cached_dataset(self, dataset: xr.Dataset) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            LOGGER.warning("Matplotlib not available; cannot render cached dataset preview.")
            return

        if not self.basins:
            LOGGER.warning("No basins available; skipping cached dataset preview plot.")
            return

        basin = self.basins[0]

        hindcast_var = next((name for name, da in dataset.data_vars.items() if 'lead_time' not in da.dims), None)
        forecast_var = next((name for name, da in dataset.data_vars.items() if 'lead_time' in da.dims), None)

        if hindcast_var is None or forecast_var is None:
            LOGGER.warning("Could not find suitable variables for cached dataset preview plot.")
            return

        try:
            hind_da = dataset[hindcast_var].sel(basin=basin).load()
            forecast_da = dataset[forecast_var].sel(basin=basin).load()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to extract data for cached dataset preview plot: %s", exc)
            return

        if 'time' not in hind_da.dims:
            LOGGER.warning("Hindcast variable %s does not have a time dimension; skipping preview plot.", hindcast_var)
            return

        time_index = pd.to_datetime(hind_da['time'].values)
        if time_index.size == 0:
            LOGGER.warning("Cached dataset preview: empty time coordinate.")
            return

        # Use most recent week of historical data for readability
        hist_window = min(time_index.size, 7 * 24)
        hind_series = hind_da.isel(time=slice(-hist_window, None)).to_pandas()

        latest_time = time_index[-1]
        try:
            forecast_slice = forecast_da.sel(time=latest_time)
        except Exception:
            forecast_slice = forecast_da.isel(time=-1)
        forecast_slice = forecast_slice.load()

        if 'lead_time' not in forecast_slice.dims:
            LOGGER.warning("Forecast variable %s is missing lead_time dimension; skipping preview plot.", forecast_var)
            return

        lead_times = pd.to_timedelta(forecast_slice['lead_time'].values)
        forecast_dates = pd.to_datetime(latest_time) + lead_times
        forecast_values = forecast_slice.values

        fig, (ax_hist, ax_fc) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

        ax_hist.plot(hind_series.index, hind_series.values, color='black', lw=1.5, label=f"{hindcast_var} (historical)")
        ax_hist.axvline(pd.to_datetime(latest_time), color='royalblue', lw=1, label='Forecast init')
        ax_hist.set_ylabel(hindcast_var)
        ax_hist.legend()
        ax_hist.grid(alpha=0.2)

        ax_fc.plot(forecast_dates, forecast_values, color='crimson', lw=1.5, label=f"{forecast_var}")
        ax_fc.set_ylabel(forecast_var)
        ax_fc.set_xlabel('Date')
        ax_fc.legend()
        ax_fc.grid(alpha=0.2)

        fig.suptitle(f"Cached dataset preview – basin {basin}")
        fig.autofmt_xdate(rotation=15)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def _load_pickled_dataset(self, train_data_file: Path) -> xr.Dataset:
        with train_data_file.open("rb") as fp:
            dataset_dict = pickle.load(fp)
        return xr.Dataset.from_dict(dataset_dict)

    def _get_retry_config(self) -> Tuple[int, float]:
        attempts = getattr(self.cfg, 'forecast_load_retries', 3) or 1
        delay = getattr(self.cfg, 'forecast_load_retry_delay', 5) or 0
        return max(1, int(attempts)), max(0.0, float(delay))

    def _materialize_forecast_slice(self,
                                    fcst_basin: xr.Dataset,
                                    basin: str,
                                    start_date: pd.Timestamp,
                                    end_date: pd.Timestamp,
                                    period_index: pd.DatetimeIndex,
                                    attempts: int,
                                    delay: float) -> xr.Dataset:
        last_exception: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                fcst_slice = fcst_basin.sel(time=slice(start_date, end_date))
                fcst_slice = fcst_slice.reindex({'time': period_index})
                return fcst_slice.load()
            except Exception as exc:  # noqa: BLE001 - log and retry specific fetch errors
                last_exception = exc
                LOGGER.warning(
                    "Attempt %s/%s failed to materialize forecast data for basin %s between %s and %s: %s",
                    attempt,
                    attempts,
                    basin,
                    start_date,
                    end_date,
                    exc,
                )
                if attempt < attempts and delay > 0:
                    time.sleep(delay)

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Forecast materialization failed without raising an exception.")

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
            
            # Extract base variable names from config (strip quartile suffixes)
            base_vars_needed = set()
            for var in self.cfg.forecast_inputs:
                # Remove quartile suffixes (_q25, _q50, _q75) to get base variable name
                base_var = var.replace('_q25', '').replace('_q50', '').replace('_q75', '')
                base_vars_needed.add(base_var)
            
            # Filter remote dataset to only needed base variables BEFORE fetching
            available_base_vars = [v for v in base_vars_needed if v in ds.data_vars]
            if not available_base_vars:
                raise ValueError(f"None of the base forecast variables {base_vars_needed} found in NOAA dataset. "
                               f"Available: {list(ds.data_vars)[:20]}...")  # Show first 20
            
            LOGGER.info(f"Filtering NOAA dataset to base variables: {available_base_vars}")
            ds = ds[available_base_vars]
            
            # Extract forecasts for basin centroids (now only for needed variables)
            basin_forecasts = fetch_forecasts_for_basins(ds, centroids)
            
            # Compute forecast quartiles as separate variables (adds _q25, _q50, _q75 suffixes)
            basin_forecasts_quartiles = self._compute_forecast_quartiles_as_variables(basin_forecasts)
            
            # Interpolate to hourly for first 240 hours (10 days)
            basin_forecasts_hourly = interpolate_to_hourly(basin_forecasts_quartiles, max_hours=240)
            
            # Final filter to exactly match config (in case interpolation changed variable names)
            forecast_vars = [var for var in basin_forecasts_hourly.data_vars if var in self.cfg.forecast_inputs]
            if not forecast_vars:
                raise ValueError(f"After processing, none of cfg.forecast_inputs {self.cfg.forecast_inputs} found. "
                               f"Available after quartile computation: {list(basin_forecasts_hourly.data_vars)}")
            
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
            LOGGER.info("Loading all available historical data from local CSV files...")
            
            # Load ALL available data - don't constrain by period dates
            # The warmup calculation will handle temporal slicing later
            
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
                    
                LOGGER.info(f"Loading all available data for basin {basin} from {csv_file}")
                
                # Read CSV file - NO DATE FILTERING
                df = pd.read_csv(csv_file)
                df['date'] = pd.to_datetime(df['date'])
                
                # Set date as index and sort
                df = df.set_index('date').sort_index()
                
                # Filter to only include hindcast_inputs and target_variables from config
                available_cols = df.columns.tolist()
                wanted_cols = self.cfg.hindcast_inputs + self.cfg.target_variables
                keep_cols = [col for col in wanted_cols if col in available_cols]
                
                if keep_cols:
                    df = df[keep_cols]
                    basin_data[basin] = df
                    LOGGER.info(f"Loaded {len(df)} records for basin {basin} covering {df.index.min().date()} to {df.index.max().date()}")
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
            
            
            LOGGER.info(f"Created historical dataset with {len(basin_names)} basins, {len(all_times)} time steps, and variables: {variable_names}")
            return historical_ds
            
        except Exception as e:
            LOGGER.warning(f"Error loading historical data: {e}")
            return None
            
    def _compute_forecast_quartiles_as_variables(self,
                                                 forecast_ds: xr.Dataset,
                                                 quartiles: Tuple[float, ...] = (0.25, 0.5, 0.75)) -> xr.Dataset:
        """Compute requested ensemble quartiles as standalone forecast variables."""
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

        forecast_vars = [var for var in self.cfg.forecast_inputs if var in xr_dataset.data_vars]
        if not forecast_vars:
            raise ValueError('Configured cfg.forecast_inputs are missing from the merged dataset.')

        xr_fcst = xr_dataset[forecast_vars]

        hindcast_vars = [var for var in xr_dataset.data_vars if var not in forecast_vars]
        if not hindcast_vars:
            raise ValueError('Dataset does not contain any hindcast or target variables after removing cfg.forecast_inputs.')

        xr_hcst = xr_dataset[hindcast_vars]

        if 'lead_time' in xr_hcst.dims:
            xr_hcst = xr_hcst.drop_dims('lead_time')

        time_dim = 'time' if 'time' in xr_hcst.dims else 'date'

        basins_without_samples = []
        basin_coordinates = xr_hcst['basin'].values.tolist()

        for basin in tqdm(basin_coordinates, file=sys.stdout, disable=self._disable_pbar):
            x_h, x_f, x_s, y, dates = {}, {}, {}, {}, {}
            frequency_maps = {}
            lowest_freq = utils.sort_frequencies(self.frequencies)[0]

            basin_hcst = xr_hcst.sel(basin=basin, drop=True)
            basin_fcst = xr_fcst.sel(basin=basin, drop=True)

            if time_dim not in basin_hcst.dims:
                LOGGER.warning(f"Time dimension not found for basin {basin} - skipping.")
                basins_without_samples.append(basin)
                continue

            for freq in self.frequencies:
                available_hindcast = [var for var in self.cfg.hindcast_inputs if var in basin_hcst.data_vars]
                if not available_hindcast:
                    raise ValueError(f'No hindcast inputs available for basin {basin}. Check cfg.hindcast_inputs.')

                hc_inputs = basin_hcst[available_hindcast]
                x_h_array = hc_inputs.to_array('variable').transpose(time_dim, 'variable').values
                x_h[freq] = x_h_array

                available_forecast = [var for var in self.cfg.forecast_inputs if var in basin_fcst.data_vars]
                if not available_forecast:
                    raise ValueError(f'No forecast inputs available for basin {basin}. Check cfg.forecast_inputs.')

                fc_inputs = basin_fcst[available_forecast]
                fc_array = fc_inputs.to_array('variable')
                if 'lead_time' in fc_array.dims:
                    fc_array = fc_array.transpose(time_dim, 'lead_time', 'variable').values
                else:
                    # Keep forecast tensors three-dimensional even without lead_time information.
                    fc_array = fc_array.transpose(time_dim, 'variable').values[:, np.newaxis, :]
                x_f[freq] = fc_array

                available_targets = [var for var in self.cfg.target_variables if var in basin_hcst.data_vars]
                if not available_targets:
                    raise ValueError(f'No target variables available for basin {basin}. Check cfg.target_variables.')

                target_da = basin_hcst[available_targets]
                y_array = target_da.to_array('variable').transpose(time_dim, 'variable').values
                y[freq] = y_array

                dates[freq] = basin_hcst[time_dim].values

                n_time = x_h[freq].shape[0]
                frequency_factor = int(utils.get_frequency_factor(lowest_freq, freq))
                if frequency_factor <= 0:
                    raise ValueError(f'Invalid frequency factor {frequency_factor} derived from {freq} relative to {lowest_freq}.')
                if n_time % frequency_factor != 0:
                    raise ValueError(f'The length of the sequence at frequency {freq} is {n_time} '
                                     f'(including warmup), which is not a multiple of {frequency_factor}.')
                frequency_maps[freq] = np.arange(n_time // frequency_factor) * frequency_factor + (frequency_factor - 1)

            if not self.is_train:
                self.period_starts[basin] = pd.to_datetime(basin_hcst[time_dim].values[0])

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

    def _validate_data_availability(self, cfg: Config):
        """Validate that configured time periods are within available data ranges.
        
        This method checks that all configured train, validation, and test periods
        fall within the available data ranges for both historical and forecast data:
        
        - Historical data: available until 2024-04-30 23:00:00
        - Forecast data: available from 2020-10-01 00:00:00 onwards
        
        For periods that require both historical and forecast data, both constraints
        must be satisfied.
        
        Parameters
        ----------
        cfg : Config
            The run configuration containing time period definitions
        
        Raises
        ------
        ValueError
            If any configured period extends outside available data ranges
        """
        # Define data availability constraints
        historical_data_end = pd.to_datetime('2024-04-30 23:00:00')
        forecast_data_start = pd.to_datetime('2020-10-01 00:00:00')
        
        LOGGER.info("Validating data availability against configured time periods...")
        LOGGER.info(f"  Historical data available until: {historical_data_end}")
        LOGGER.info(f"  Forecast data available from: {forecast_data_start}")
        
        # Collect all configured periods
        periods_to_check = []
        
        # Check training period
        if hasattr(cfg, 'train_start_date') and hasattr(cfg, 'train_end_date'):
            if cfg.train_start_date and cfg.train_end_date:
                train_start = pd.to_datetime(cfg.train_start_date, format='%d/%m/%Y')
                train_end = pd.to_datetime(cfg.train_end_date, format='%d/%m/%Y')
                periods_to_check.append(('train', train_start, train_end))
        
        # Check validation period  
        if hasattr(cfg, 'validation_start_date') and hasattr(cfg, 'validation_end_date'):
            if cfg.validation_start_date and cfg.validation_end_date:
                val_start = pd.to_datetime(cfg.validation_start_date, format='%d/%m/%Y')
                val_end = pd.to_datetime(cfg.validation_end_date, format='%d/%m/%Y')
                periods_to_check.append(('validation', val_start, val_end))
        
        # Check test period
        if hasattr(cfg, 'test_start_date') and hasattr(cfg, 'test_end_date'):
            if cfg.test_start_date and cfg.test_end_date:
                test_start = pd.to_datetime(cfg.test_start_date, format='%d/%m/%Y')
                test_end = pd.to_datetime(cfg.test_end_date, format='%d/%m/%Y')
                periods_to_check.append(('test', test_start, test_end))
        
        # Validate each period
        errors = []
        warnings = []
        
        for period_name, period_start, period_end in periods_to_check:
            LOGGER.info(f"  Checking {period_name} period: {period_start.date()} to {period_end.date()}")
            
            # Check if period requires forecast data (overlaps with forecast availability)
            needs_forecast_data = period_end >= forecast_data_start
            
            # Check if period requires historical data (starts before or overlaps with historical availability)
            needs_historical_data = period_start <= historical_data_end
            
            # Validate forecast data availability
            if needs_forecast_data and period_start < forecast_data_start:
                gap_days = (forecast_data_start - period_start).days
                errors.append(
                    f"{period_name.capitalize()} period starts {gap_days} days before forecast data becomes available. "
                    f"Period starts: {period_start.date()}, forecast data starts: {forecast_data_start.date()}"
                )
            
            # Validate historical data availability  
            if needs_historical_data and period_end > historical_data_end:
                gap_days = (period_end - historical_data_end).days
                errors.append(
                    f"{period_name.capitalize()} period extends {gap_days} days beyond available historical data. "
                    f"Period ends: {period_end.date()}, historical data ends: {historical_data_end.date()}"
                )
            
            # Check for periods that fall entirely outside available data ranges
            if period_end < forecast_data_start and period_start > historical_data_end:
                errors.append(
                    f"{period_name.capitalize()} period falls entirely outside available data ranges. "
                    f"Period: {period_start.date()} to {period_end.date()}"
                )
            
            # For OnlineForecastDataset, periods without forecast data are considered errors
            # since this dataset is specifically designed for forecast operations
            if period_end < forecast_data_start and needs_historical_data:
                gap_days = (forecast_data_start - period_end).days
                errors.append(
                    f"{period_name.capitalize()} period ends {gap_days} days before forecast data becomes available. "
                    f"OnlineForecastDataset requires forecast data availability. "
                    f"Period ends: {period_end.date()}, forecast data starts: {forecast_data_start.date()}"
                )
            
            # Also error for periods that start after historical data ends (no historical context)
            if period_start > historical_data_end:
                gap_days = (period_start - historical_data_end).days
                errors.append(
                    f"{period_name.capitalize()} period starts {gap_days} days after historical data ends. "
                    f"No historical context available for model training/evaluation. "
                    f"Period starts: {period_start.date()}, historical data ends: {historical_data_end.date()}"
                )
        
        # Log warnings
        for warning in warnings:
            LOGGER.warning(f"⚠️  {warning}")
        
        # Raise errors if any validation failed
        if errors:
            error_msg = "Data availability validation failed:\n\n"
            for i, error in enumerate(errors, 1):
                error_msg += f"{i}. {error}\n"
            
            error_msg += f"\nAvailable data ranges:\n"
            error_msg += f"  • Historical data: up to {historical_data_end.date()} {historical_data_end.time()}\n"
            error_msg += f"  • Forecast data: from {forecast_data_start.date()} {forecast_data_start.time()} onwards\n\n"
            error_msg += f"Suggested fixes:\n"
            error_msg += f"  • Adjust configured date ranges to fall within available data periods\n"
            error_msg += f"  • Update data sources to extend availability ranges"
            
            raise ValueError(error_msg)
        
        LOGGER.info("✅ Data availability validation passed - all periods fall within available data ranges")

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
