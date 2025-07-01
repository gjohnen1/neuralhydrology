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
    - Forecast variables are indexed by (basin, time, lead_time) - loaded from online sources
    - Supports probabilistic forecasts with quartiles (q25, q50, q75)

    The dataset loads data from two sources:
    1. Historical data from local CSV files in data/harz/timeseries/hydromet_timeseries_{basin}.csv
    2. Forecast data from online NOAA GEFS sources (or placeholder data for testing)

    Data structure:
    - Hindcast variables: shape (basin, time) - historical observations
    - Forecast variables: shape (basin, time, lead_time) - forecast ensemble quartiles
    - Mixed indexing allows operational forecasting workflows

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
    
        super(GenericDataset, self).__init__(cfg=cfg,
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
        # if no netCDF file is passed, data set is created from raw basin files
        if (self.cfg.train_data_file is None) or (not self.is_train):
            data_list = []
            
            # Separate hindcast and forecast variables
            hcst_keep_cols = self.cfg.target_variables + self.cfg.hindcast_inputs
            fcst_keep_cols = self.cfg.forecast_inputs 

            if not self._disable_pbar:
                LOGGER.info("Loading basin data into xarray data set.")
            for basin in tqdm(self.basins, disable=self._disable_pbar, file=sys.stdout):

                # Load hindcast data (time-indexed only) and forecast data (time + lead_time indexed) separately
                df_hcst = self._load_hindcast_data(basin, hcst_keep_cols)
                df_fcst = self._load_forecast_data(basin, fcst_keep_cols) 

                # Handle MultiIndex for forecast data
                if isinstance(df_fcst.index, pd.MultiIndex):
                    # Ensure proper ordering of MultiIndex
                    if df_fcst.index.names != ['time', 'lead_time']:
                        df_fcst = df_fcst.reset_index()
                        df_fcst = df_fcst.set_index(['time', 'lead_time'])
                    lead_times = df_fcst.index.unique(level='lead_time')
                else:
                    # If not MultiIndex, assume it's already properly structured
                    lead_times = None

                # add columns from dataframes passed as additional data files
                if self.additional_features:
                    df_hcst = pd.concat([df_hcst, *[d[basin] for d in self.additional_features if basin in d]], axis=1)
                    df_fcst = pd.concat([df_fcst, *[d[basin] for d in self.additional_features if basin in d]], axis=1)
                
                # if target variables are missing for basin, add empty column to still allow predictions to be made
                if not self.is_train:
                    df_hcst = self._add_missing_targets(df_hcst)

                # check if any feature should be duplicated
                df_hcst = self._duplicate_features(df_hcst)
                df_fcst = self._duplicate_features(df_fcst)

                # check if a shifted copy of a feature should be added
                df_fcst = self._add_lagged_features(df_fcst)
                df_hcst = self._add_lagged_features(df_hcst)

                # Get start and end dates for this basin
                start_dates = self.start_and_end_dates[basin]["start_dates"]
                end_dates = [
                    date + pd.Timedelta(days=1, seconds=-1) for date in self.start_and_end_dates[basin]["end_dates"]
                ]

                # infer native frequency from hindcast data 
                native_frequency = utils.infer_frequency(df_hcst.index)
                if not self.frequencies:
                    self.frequencies = [native_frequency]

                # Assert that the used frequencies are lower or equal than the native frequency
                try:
                    freq_vs_native = [utils.compare_frequencies(freq, native_frequency) for freq in self.frequencies]
                except ValueError:
                    LOGGER.warning('Cannot compare provided frequencies with native frequency. '
                                   'Make sure the frequencies are not higher than the native frequency.')
                    freq_vs_native = []
                if any(comparison > 1 for comparison in freq_vs_native):
                    raise ValueError(f'Frequency is higher than native data frequency {native_frequency}.')

                # Calculate warmup offsets
                offsets = [(self.seq_len[i] - max(self._predict_last_n[i], self._forecast_seq_len[i])) * to_offset(freq)
                           for i, freq in enumerate(self.frequencies)]
                
                hcst_basin_data_list = []
                fcst_basin_data_list = []
                
                # create xarray data set for each period slice of the specific basin
                for i, (start_date, end_date) in enumerate(zip(start_dates, end_dates)):
                    # Check frequency alignment
                    if not all(to_offset(freq).is_on_offset(start_date) for freq in self.frequencies):
                        LOGGER.warning(f'Start date {start_date} is not aligned with frequency. '
                                       'This might cause issues with datetime indexing.')

                    # Add warmup period for hindcast data
                    warmup_start_date = min(start_date - offset for offset in offsets)
                    df_hcst_sub = df_hcst[warmup_start_date:end_date]

                    # For forecast data, select based on initialization time (first level of MultiIndex)
                    if isinstance(df_fcst.index, pd.MultiIndex):
                        # Select forecast data based on initialization times within the period
                        idx = pd.IndexSlice
                        df_fcst_sub = df_fcst.loc[idx[start_date:end_date, :], :]
                    else:
                        df_fcst_sub = df_fcst[start_date:end_date]

                    # Reindex hindcast data to ensure complete time series
                    full_range = pd.date_range(start=warmup_start_date, end=end_date, freq=native_frequency)
                    df_hcst_sub = df_hcst_sub.reindex(pd.DatetimeIndex(full_range, name=df_hcst_sub.index.name))
                    
                    # For forecast data, ensure complete time x lead_time coverage if MultiIndex
                    if isinstance(df_fcst.index, pd.MultiIndex) and lead_times is not None:
                        time_range = pd.date_range(start=start_date, end=end_date, freq=native_frequency)
                        full_forecast_index = pd.MultiIndex.from_product(
                            [time_range, lead_times], 
                            names=['time', 'lead_time']
                        )
                        df_fcst_sub = df_fcst_sub.reindex(full_forecast_index)

                    # Set targets before period start to NaN (for warmup period)
                    df_hcst_sub.loc[df_hcst_sub.index < start_date, self.cfg.target_variables] = np.nan

                    hcst_basin_data_list.append(df_hcst_sub)
                    fcst_basin_data_list.append(df_fcst_sub)

                if not hcst_basin_data_list:
                    # Skip basin in case no start and end dates were defined
                    continue

                # Concatenate all time slices
                df_hcst = pd.concat(hcst_basin_data_list, axis=0)
                df_fcst = pd.concat(fcst_basin_data_list, axis=0)

                # Handle duplicate indices in hindcast data (due to overlapping warmup periods)
                df_non_duplicated = df_hcst[~df_hcst.index.duplicated(keep=False)]
                df_duplicated = df_hcst[df_hcst.index.duplicated(keep=False)]
                filtered_duplicates = []
                
                if len(df_duplicated) > 0:
                    for _, grp in df_duplicated.groupby(df_duplicated.index):
                        mask = ~grp[self.cfg.target_variables].isna().any(axis=1)
                        if not mask.any():
                            # If all duplicates have NaN targets, keep the first
                            filtered_duplicates.append(grp.head(1))
                        else:
                            # Keep the first row with non-NaN targets
                            filtered_duplicates.append(grp[mask].head(1))

                if filtered_duplicates:
                    df_filtered_duplicates = pd.concat(filtered_duplicates, axis=0)
                    df_hcst = pd.concat([df_non_duplicated, df_filtered_duplicates], axis=0)
                else:
                    df_hcst = df_non_duplicated

                # Sort and ensure complete time coverage for hindcast data
                df_hcst = df_hcst.sort_index(axis=0, ascending=True)
                if len(df_hcst) > 0:
                    df_hcst = df_hcst.reindex(
                        pd.DatetimeIndex(
                            data=pd.date_range(df_hcst.index[0], df_hcst.index[-1], freq=native_frequency),
                            name=df_hcst.index.name
                        )
                    )
                
                # Convert to xarray Dataset and add basin coordinate
                xr_hcst = xr.Dataset.from_dataframe(df_hcst.astype(np.float32))
                xr_fcst = xr.Dataset.from_dataframe(df_fcst.astype(np.float32))

                # Merge datasets - this ensures both have consistent temporal coverage
                xr = xr_fcst.merge(xr_hcst) 
                xr = xr.assign_coords({'basin': basin})
                data_list.append(xr)

            if not data_list:
                # If no period for no basin has defined timeslices, raise error
                if self.is_train:
                    raise NoTrainDataError
                else:
                    raise NoEvaluationDataError

            # Create one large dataset with basin dimension
            xr_dataset = xr.concat(data_list, dim="basin")
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
        # For now, return a simplified structure for testing
        # TODO: Implement full online forecast loading from NOAA GEFS
        
        # Create empty MultiIndex DataFrame with proper structure
        time_range = pd.date_range(
            start=start_date or '2023-01-01', 
            end=end_date or '2023-12-31', 
            freq='H'
        )[:100]  # Limit for testing
        
        lead_times = range(1, 241)  # 240 hour forecast horizon
        
        multi_index = pd.MultiIndex.from_product(
            [time_range, lead_times], 
            names=['time', 'lead_time']
        )
        
        # Create DataFrame with requested columns filled with NaN for now
        data = {}
        for col in columns:
            data[col] = np.full(len(multi_index), np.nan, dtype=np.float32)
        
        df = pd.DataFrame(data, index=multi_index)
        
        LOGGER.info(f"Created placeholder forecast data for basin {basin} with shape {df.shape}")
        return df
        
    except Exception as e:
        LOGGER.error(f"Error loading forecast data for basin {basin}: {e}")
        # Return empty MultiIndex DataFrame
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
