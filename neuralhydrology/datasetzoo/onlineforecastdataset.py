"""Dataset class for loading basin forecast and historical data from online sources."""

import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Union, Optional

import numpy as np
import pandas as pd
import torch
import xarray as xr
from numba import NumbaPendingDeprecationWarning
from numba import njit, prange
from pandas.tseries.frequencies import to_offset
from tqdm import tqdm

from neuralhydrology.datasetzoo.genericdataset import GenericDataset
from neuralhydrology.datautils import utils
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import NoTrainDataError, NoEvaluationDataError

LOGGER = logging.getLogger(__name__)


class OnlineForecastDataset(GenericDataset):
    """Dataset for loading basin forecast and historical data from online sources or NetCDF files.
    
    This dataset is designed to work with combined forecast-historical datasets like those
    created by the operational_harz pipeline. It expects data structured as:
    - Hindcast features: indexed by (basin, time) - historical/observational data
    - Forecast features: indexed by (basin, time, lead_time) - forecast data with lead times
    
    The dataset can either:
    1. Load pre-processed NetCDF files containing this combined structure
    2. Work with xarray datasets passed directly (for online workflows)
    
    It provides the same interface as ForecastDataset but handles the multi-dimensional
    forecast structure with lead_time dimensions.
    
    Parameters
    ----------
    cfg : Config
        The run configuration containing paths and variable specifications.
    is_train : bool
        Defines if the dataset is used for training or evaluating.
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded.
    basin : str, optional
        If passed, the data for only this basin will be loaded.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame.
    id_to_int : Dict[str, int], optional
        Basin ID to integer mapping for one-hot encoding.
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        Scaling parameters for features.
    online_dataset : xr.Dataset, optional
        Pre-loaded xarray dataset (for online workflows that bypass file loading).
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xr.DataArray]] = {},
                 online_dataset: Optional[xr.Dataset] = None):
        
        # Store the online dataset if provided
        self.online_dataset = online_dataset
        
        # Initialize the parent class with BaseDataset constructor to get the same structure as ForecastDataset
        super(GenericDataset, self).__init__(cfg=cfg,
                                             is_train=is_train,
                                             period=period,
                                             basin=basin,
                                             additional_features=additional_features,
                                             id_to_int=id_to_int,
                                             scaler=scaler)
        
        # Initialize forecast-specific configuration
        self._initialize_frequency_configuration()
        
        # Load xarray dataset (either from files or use provided online dataset)
        self.xr = self._load_or_create_xarray_dataset()
        
        # Create lookup table and pytorch tensors
        self._create_lookup_table(self.xr)

    def _load_basin_data(self, basin: str, columns: list) -> pd.DataFrame:
        """Load input and output data from NetCDF files, similar to ForecastDataset."""
        # Load timeseries data using the same pattern as ForecastDataset
        df = load_timeseries(data_dir=self.cfg.data_dir, 
                           time_series_data_sub_dir=self.cfg.time_series_data_sub_dir, 
                           basin=basin, 
                           columns=columns)
        return df

    def _initialize_frequency_configuration(self):
        """Initialize frequency configuration for forecast mode."""
        self.seq_len = self.cfg.seq_length
        self._forecast_seq_len = getattr(self.cfg, 'forecast_seq_length', 24)
        self._predict_last_n = self.cfg.predict_last_n
        self._forecast_offset = getattr(self.cfg, 'forecast_offset', 0)
        
        # Handle frequencies
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
        """Load or create xarray dataset with same structure as ForecastDataset."""
        
        # If an online dataset was provided, use it directly
        if self.online_dataset is not None:
            LOGGER.info("Using provided online dataset")
            return self._process_online_dataset(self.online_dataset)
            
        # if no netCDF file is passed, data set is created from raw basin files
        if (self.cfg.train_data_file is None) or (not self.is_train):
            data_list = []
            
            # Ensure we have required config variables for forecast mode
            if not hasattr(self.cfg, 'hindcast_inputs') or not hasattr(self.cfg, 'forecast_inputs'):
                raise ValueError("Both 'hindcast_inputs' and 'forecast_inputs' must be specified in config for OnlineForecastDataset")
            
            # Define columns to keep for hindcast and forecast data
            hcst_keep_cols = self.cfg.target_variables + self.cfg.hindcast_inputs
            fcst_keep_cols = self.cfg.forecast_inputs 

            if not self._disable_pbar:
                LOGGER.info("Loading basin data into xarray data set.")
            for basin in tqdm(self.basins, disable=self._disable_pbar, file=sys.stdout):

                df_hcst = self._load_basin_data(basin, hcst_keep_cols)
                df_fcst = self._load_basin_data(basin, fcst_keep_cols) 

                # Handle different index structures
                if isinstance(df_fcst.index, pd.MultiIndex):
                    # Already has the correct MultiIndex structure (date, lead_time)
                    if df_fcst.index.names != ['date', 'lead_time']:
                        # Rename indices if needed
                        df_fcst.index.names = ['date', 'lead_time']
                    lead_times = df_fcst.index.unique(level='lead_time')
                else:
                    # Need to handle case where forecast data doesn't have MultiIndex
                    # This might happen if the data was flattened or has different structure
                    raise ValueError(f"Forecast data for basin {basin} should have MultiIndex (date, lead_time)")

                # add columns from dataframes passed as additional data files
                if self.additional_features:
                    df_hcst = pd.concat([df_hcst, *[d[basin] for d in self.additional_features if basin in d]], axis=1)
                    df_fcst = pd.concat([df_fcst, *[d[basin] for d in self.additional_features if basin in d]], axis=1)
                    
                # if target variables are missing for basin, add empty column to still allow predictions to be made
                if not self.is_train:
                    # target variables held in hcst dataset
                    df_hcst = self._add_missing_targets(df_hcst)

                # check if any feature should be duplicated
                df_hcst = self._duplicate_features(df_hcst)
                df_fcst = self._duplicate_features(df_fcst)

                # check if a shifted copy of a feature should be added
                df_fcst = self._add_lagged_features(df_fcst)
                df_hcst = self._add_lagged_features(df_hcst)

                # Make end_date the last second of the specified day, such that the
                # dataset will include all hours of the last day, not just 00:00.
                start_dates = self.start_and_end_dates[basin]["start_dates"]
                end_dates = [
                    date + pd.Timedelta(days=1, seconds=-1) for date in self.start_and_end_dates[basin]["end_dates"]
                ]

                # infer native frequency from hindcast data 
                native_frequency = utils.infer_frequency(df_hcst.index)
                if not self.frequencies:
                    self.frequencies = [native_frequency]  # use df's native resolution by default

                # Assert that the used frequencies are lower or equal than the native frequency
                try:
                    freq_vs_native = [utils.compare_frequencies(freq, native_frequency) for freq in self.frequencies]
                except ValueError:
                    LOGGER.warning('Cannot compare provided frequencies with native frequency. '
                                   'Make sure the frequencies are not higher than the native frequency.')
                    freq_vs_native = []
                if any(comparison > 1 for comparison in freq_vs_native):
                    raise ValueError(f'Frequency is higher than native data frequency {native_frequency}.')

                # used to get the maximum warmup-offset across all frequencies
                offsets = [(self.seq_len[i] - max(self._predict_last_n[i], self._forecast_seq_len[i])) * to_offset(freq)
                           for i, freq in enumerate(self.frequencies)]
                
                hcst_basin_data_list = []
                fcst_basin_data_list = []
                
                # create xarray data set for each period slice of the specific basin
                for i, (start_date, end_date) in enumerate(zip(start_dates, end_dates)):
                    # if the start date is not aligned with the frequency, the resulting datetime indices will be off
                    if not all(to_offset(freq).is_on_offset(start_date) for freq in self.frequencies):
                        misaligned = [freq for freq in self.frequencies if not to_offset(freq).is_on_offset(start_date)]
                        raise ValueError(f'start date {start_date} is not aligned with frequencies {misaligned}.')

                    # add warmup period
                    warmup_start_date = min(start_date - offset for offset in offsets)
                    df_hcst_sub = df_hcst[warmup_start_date:end_date]

                    # `df_fcst_sub` has a multiindex
                    idx = pd.IndexSlice
                    df_fcst_sub = df_fcst.loc[idx[start_date:end_date, :], :]

                    # make sure the df covers the full date range, filling gaps with NaNs
                    full_range = pd.date_range(start=warmup_start_date, end=end_date, freq=native_frequency)
                    df_hcst_sub = df_hcst_sub.reindex(pd.DatetimeIndex(full_range, name=df_hcst_sub.index.name))
                    
                    # Select the rows between the start_date and end_date
                    full_range = pd.date_range(start=start_date, end=end_date, freq=native_frequency)
                    df_fcst_sub = df_fcst_sub.reindex(pd.MultiIndex.from_product([full_range, lead_times], names=['date', 'lead_time']))

                    # Set all targets before period start to NaN
                    if self.cfg.target_variables:
                        target_cols = [col for col in self.cfg.target_variables if col in df_hcst_sub.columns]
                        if target_cols:
                            df_hcst_sub.loc[df_hcst_sub.index < start_date, target_cols] = np.nan

                    hcst_basin_data_list.append(df_hcst_sub)
                    fcst_basin_data_list.append(df_fcst_sub)

                if not hcst_basin_data_list:
                    # Skip basin in case no start and end dates where defined.
                    continue

                # In case of multiple time slices per basin, stack the time slices in the time dimension.
                df_hcst = pd.concat(hcst_basin_data_list, axis=0)
                df_fcst = pd.concat(fcst_basin_data_list, axis=0)

                # Handle duplicated indices due to overlaps between warmup and training periods
                df_non_duplicated = df_hcst[~df_hcst.index.duplicated(keep=False)]
                df_duplicated = df_hcst[df_hcst.index.duplicated(keep=False)]
                filtered_duplicates = []
                
                if not df_duplicated.empty:
                    for _, grp in df_duplicated.groupby(level=0):  # Group by date level
                        if self.cfg.target_variables:
                            target_cols = [col for col in self.cfg.target_variables if col in grp.columns]
                            if target_cols:
                                mask = ~grp[target_cols].isna().any(axis=1)
                                if not mask.any():
                                    filtered_duplicates.append(grp.head(1))
                                else:
                                    filtered_duplicates.append(grp[mask].head(1))
                            else:
                                filtered_duplicates.append(grp.head(1))
                        else:
                            filtered_duplicates.append(grp.head(1))

                if filtered_duplicates:
                    df_filtered_duplicates = pd.concat(filtered_duplicates, axis=0)
                    df_hcst = pd.concat([df_non_duplicated, df_filtered_duplicates], axis=0)
                else:
                    df_hcst = df_non_duplicated

                # Sort by DatetimeIndex and reindex to fill gaps with NaNs.
                df_hcst = df_hcst.sort_index(axis=0, ascending=True)
                df_hcst = df_hcst.reindex(
                    pd.DatetimeIndex(data=pd.date_range(df_hcst.index[0], df_hcst.index[-1], freq=native_frequency),
                                     name=df_hcst.index.name))
                
                # Convert to xarray Dataset and add basin string as additional coordinate
                xr_fcst = xr.Dataset.from_dataframe(df_fcst.astype(np.float32))
                xr_hcst = xr.Dataset.from_dataframe(df_hcst.astype(np.float32))

                # merging xarray datasets has the convenient side-effect that both forecast and hindcast data will have the same temporal extent
                xr_merged = xr_fcst.merge(xr_hcst) 
                xr_merged = xr_merged.assign_coords({'basin': basin})
                data_list.append(xr_merged)

            if not data_list:
                # If no period for no basin has defined timeslices, raise error.
                if self.is_train:
                    raise NoTrainDataError
                else:
                    raise NoEvaluationDataError

            # create one large dataset that has two coordinates: datetime and basin
            xr_final = xr.concat(data_list, dim="basin")
            if self.is_train and self.cfg.save_train_data:
                self._save_xarray_dataset(xr_final)

        else:
            # Otherwise we can reload previously-saved training data
            with self.cfg.train_data_file.open("rb") as fp:
                d = pickle.load(fp)
            xr_final = xr.Dataset.from_dict(d)
            if not self.frequencies:
                native_frequency = utils.infer_frequency(xr_final["date"].values)
                self.frequencies = [native_frequency]

        return xr_final

    def _process_online_dataset(self, online_ds: xr.Dataset) -> xr.Dataset:
        """Process an online dataset to match expected structure."""
        
        # Validate that the dataset has the expected structure
        if 'basin' not in online_ds.dims:
            raise ValueError("Online dataset must have 'basin' dimension")
        if 'time' not in online_ds.dims:
            raise ValueError("Online dataset must have 'time' dimension") 
            
        # Check if we have forecast variables (should have lead_time dimension)
        forecast_vars = [var for var in online_ds.data_vars if 'lead_time' in online_ds[var].dims]
        hindcast_vars = [var for var in online_ds.data_vars if 'lead_time' not in online_ds[var].dims and var != 'basin']
        
        LOGGER.info(f"Found {len(forecast_vars)} forecast variables and {len(hindcast_vars)} hindcast variables")
        
        # Infer frequency from time coordinate
        if not self.frequencies:
            native_frequency = utils.infer_frequency(online_ds["time"].values)
            self.frequencies = [native_frequency]
            LOGGER.info(f"Inferred frequency: {native_frequency}")
        
        # Filter basins if needed
        if self.basins:
            # Only keep basins that are in both the dataset and our basin list
            available_basins = [b for b in self.basins if b in online_ds.basin.values]
            if available_basins:
                online_ds = online_ds.sel(basin=available_basins)
                LOGGER.info(f"Filtered to {len(available_basins)} basins: {available_basins}")
            else:
                raise ValueError(f"None of the requested basins {self.basins} found in online dataset")
        
        return online_ds

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        basin, indices = self.lookup_table[item]

        sample = {}
        for freq, seq_len, forecast_seq_len, idx in zip(self.frequencies, self.seq_len, self._forecast_seq_len, indices):
            # if there's just one frequency, don't use suffixes.
            freq_suffix = '' if len(self.frequencies) == 1 else f'_{freq}'
            
            # NOTE idx is the index of the forecast issue time
            # hence, idx + self._forecast_offset is the index of the first forecast
            hindcast_start_idx = idx + self._forecast_offset + forecast_seq_len - seq_len
            hindcast_end_idx = idx + self._forecast_offset # slice-end is excluding - we take values up, but not including, the issue time
            # `forecast_start_idx` equals `idx` because in the case of forecasts, the 
            # time dimension refers to the initialization time and NOT the time of the 
            # forecast itself, which instead is indexed by lead time. 
            forecast_start_idx = idx
            global_end_idx = idx + self._forecast_offset + forecast_seq_len 

            sample[f'x_h{freq_suffix}'] = self._x_h[basin][freq][hindcast_start_idx:hindcast_end_idx]
            # x_f is 3D: [time, lead_time, features], so we get [lead_time, features] for the forecast time
            sample[f'x_f{freq_suffix}'] = self._x_f[basin][freq][forecast_start_idx]  # shape: [lead_time, features]
            sample[f'y{freq_suffix}'] = self._y[basin][freq][hindcast_start_idx:global_end_idx]
            sample[f'date{freq_suffix}'] = self._dates[basin][freq][hindcast_start_idx:global_end_idx]

            # check for static inputs
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
    
    def _create_lookup_table(self, xr: xr.Dataset):
        lookup = []
        if not self._disable_pbar:
            LOGGER.info("Create lookup table and convert to pytorch tensor")

        # Split data into forecast and hindcast components
        forecast_vars = [var for var in self.cfg.forecast_inputs if var in xr.variables]
        hindcast_vars = [var for var in xr.variables if var not in self.cfg.forecast_inputs and var not in ['basin']]
        
        if forecast_vars:
            xr_fcst = xr[forecast_vars]
        else:
            xr_fcst = None
            
        if hindcast_vars:
            xr_hcst = xr[hindcast_vars]
            # Drop lead_time dimension from hindcast data if it exists
            if 'lead_time' in xr_hcst.dims:
                xr_hcst = xr_hcst.drop_dims('lead_time')
        else:
            xr_hcst = None

        # list to collect basins ids of basins without a single training sample
        basins_without_samples = []
        basin_coordinates = xr["basin"].values.tolist()
        for basin in tqdm(basin_coordinates, file=sys.stdout, disable=self._disable_pbar):
            # store data of each frequency as numpy array of shape [time steps, features] and dates as numpy array of
            # shape (time steps,)
            x_h, x_f, x_s, y, dates = {}, {}, {}, {}, {}

            # keys: frequencies, values: array mapping each lowest-frequency
            # sample to its corresponding sample in this frequency
            frequency_maps = {}
            lowest_freq = utils.sort_frequencies(self.frequencies)[0]

            # converting from xarray to pandas DataFrame because resampling is much faster in pandas.
            if xr_hcst is not None:
                df_hcst_native = xr_hcst.sel(basin=basin, drop=True).to_dataframe()
            else:
                df_hcst_native = pd.DataFrame()
                
            if xr_fcst is not None:
                df_fcst_native = xr_fcst.sel(basin=basin, drop=True).to_dataframe()
            else:
                df_fcst_native = pd.DataFrame()

            for freq in self.frequencies:
                # multiple frequencies are not supported so we don't do any resampling here
                df_hcst_resampled = df_hcst_native
                df_fcst_resampled = df_fcst_native
            
                # pull all of the data that needs to be validated
                if df_hcst_resampled.empty or not self.cfg.hindcast_inputs:
                    x_h[freq] = np.array([]).reshape(0, 0)  # Empty array
                else:
                    available_hindcast = [col for col in self.cfg.hindcast_inputs if col in df_hcst_resampled.columns]
                    if available_hindcast:
                        x_h[freq] = df_hcst_resampled[available_hindcast].values
                    else:
                        x_h[freq] = np.array([]).reshape(len(df_hcst_resampled), 0)
                        
                if df_fcst_resampled.empty or not self.cfg.forecast_inputs:
                    x_f[freq] = np.array([]).reshape(0, 0, 0)  # Empty 3D array
                else:
                    available_forecast = [col for col in self.cfg.forecast_inputs if col in df_fcst_resampled.columns]
                    if available_forecast:
                        # cast multiindex dataframe to three-dimensional array
                        x_f[freq] = df_fcst_resampled[available_forecast].to_xarray().to_array().transpose('date', 'lead_time', 'variable').values
                    else:
                        # Create empty 3D array with correct shape
                        dates = df_fcst_resampled.index.get_level_values('date').unique()
                        lead_times = df_fcst_resampled.index.get_level_values('lead_time').unique()
                        x_f[freq] = np.array([]).reshape(len(dates), len(lead_times), 0)
                        
                if df_hcst_resampled.empty or not self.cfg.target_variables:
                    y[freq] = np.array([]).reshape(0, 0)  # Empty array
                else:
                    available_targets = [col for col in self.cfg.target_variables if col in df_hcst_resampled.columns]
                    if available_targets:
                        y[freq] = df_hcst_resampled[available_targets].values
                    else:
                        y[freq] = np.array([]).reshape(len(df_hcst_resampled), 0)
                
                # Add dates of the (resampled) data to the dates dict
                if not df_hcst_resampled.empty:
                    dates[freq] = df_hcst_resampled.index.to_numpy()
                else:
                    dates[freq] = np.array([], dtype='datetime64[ns]')

                # number of frequency steps in one lowest-frequency step
                frequency_factor = int(utils.get_frequency_factor(lowest_freq, freq))
                # array position i is the last entry of this frequency that belongs to the lowest-frequency sample i.
                if len(df_hcst_resampled) == 0:
                    frequency_maps[freq] = np.array([])
                elif len(df_hcst_resampled) % frequency_factor != 0:
                    raise ValueError(f'The length of the dataframe at frequency {freq} is {len(df_hcst_resampled)} '
                                     f'(including warmup), which is not a multiple of {frequency_factor} (i.e., the '
                                     f'factor between the lowest frequency {lowest_freq} and the frequency {freq}. '
                                     f'To fix this, adjust the {self.period} start or end date such that the period '
                                     f'(including warmup) has a length that is divisible by {frequency_factor}.')
                else:
                    frequency_maps[freq] = np.arange(len(df_hcst_resampled) // frequency_factor) \
                                           * frequency_factor + (frequency_factor - 1)


            # store first date of sequence to be able to restore dates during inference
            if not self.is_train:
                if xr_hcst is not None:
                    self.period_starts[basin] = pd.to_datetime(xr_hcst.sel(basin=basin)["date"].values[0])
                elif xr_fcst is not None:
                    self.period_starts[basin] = pd.to_datetime(xr_fcst.sel(basin=basin)["date"].values[0])
                else:
                    self.period_starts[basin] = pd.to_datetime(xr.sel(basin=basin)["date"].values[0])

            # we can ignore the deprecation warning about lists because we don't use the passed lists
            # after the validate_samples call. The alternative numba.typed.Lists is still experimental.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

                # checks inputs and outputs for each sequence. valid: flag = 1, invalid: flag = 0
                # manually unroll the dicts into lists to make sure the order of frequencies is consistent.
                # during inference, we want all samples with sufficient history (even if input is NaN), so
                # we pass x_d, x_s, y as None.
                if all(len(frequency_maps[freq]) > 0 for freq in self.frequencies):
                    flag = validate_samples(x_h=[x_h[freq] for freq in self.frequencies] if self.is_train else None,
                                            x_f=[x_f[freq] for freq in self.frequencies] if self.is_train else None,
                                            y=[y[freq] for freq in self.frequencies] if self.is_train else None,
                                            seq_length=self.seq_len,
                                            forecast_seq_length=self._forecast_seq_len,
                                            forecast_offset=self._forecast_offset,
                                            predict_last_n=self._predict_last_n,
                                            frequency_maps=[frequency_maps[freq] for freq in self.frequencies])
                else:
                    # No data available for this basin
                    flag = np.array([])

            valid_samples = np.argwhere(flag == 1) if len(flag) > 0 else np.array([])
            self.valid_samples = valid_samples
            if len(valid_samples) > 0:
                for f in valid_samples:
                    # store pointer to basin and the sample's index in each frequency
                    lookup.append((basin, [frequency_maps[freq][int(f)] for freq in self.frequencies]))

            self.lookup = lookup 
            # only store data if this basin has at least one valid sample in the given period
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
            LOGGER.info(
                f"These basins do not have a single valid sample in the {self.period} period: {basins_without_samples}")
        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}
        self.num_samples = len(self.lookup_table)
        self.valid_samples = valid_samples
        if self.num_samples == 0:
            if self.is_train:
                raise NoTrainDataError
            else:
                raise NoEvaluationDataError

    def load_local_historical_data(timeseries_dir: Path, centroids: pd.DataFrame, start_date: str, end_date: str) -> xr.Dataset:
        """Load local historical data from CSV files for multiple basins.
    
        This function loads historical timeseries data from local CSV files stored in the timeseries directory.
        It's designed to work with the Harz dataset structure where each basin has its own CSV file.
    
        Parameters
        ----------
        timeseries_dir : Path
            Path to the directory containing the timeseries CSV files (e.g., data/harz/timeseries)
        centroids : pd.DataFrame
            DataFrame containing basin information with 'basin_name' column
        start_date : str
            Start date for the historical data in format 'YYYY-MM-DD'
        end_date : str 
            End date for the historical data in format 'YYYY-MM-DD'
        
        Returns
        -------
        xr.Dataset
            xarray Dataset containing historical data indexed by 'basin' and 'time'
        
        Raises
        ------
        FileNotFoundError
            If the timeseries directory doesn't exist or basin files are missing
        """
        if not timeseries_dir.exists():
            raise FileNotFoundError(f"Timeseries directory not found: {timeseries_dir}")
    
        basin_data_list = []
    
        for _, row in centroids.iterrows():
            basin_name = row['basin_name']
        
            # Look for the basin's CSV file
            basin_file = timeseries_dir / f"hydromet_timeseries_{basin_name}.csv"
        
            if not basin_file.exists():
                LOGGER.warning(f"No timeseries file found for basin {basin_name} at {basin_file}")
                continue
            
            try:
                # Load the CSV file
                df = pd.read_csv(basin_file, index_col='date', parse_dates=['date'])
            
                # Filter by date range
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                df = df.loc[start_dt:end_dt]
            
                if df.empty:
                    LOGGER.warning(f"No data in date range {start_date} to {end_date} for basin {basin_name}")
                    continue
            
                # Convert to xarray Dataset
                data_vars = {}
                for col in df.columns:
                    # Add '_hist' suffix to distinguish from forecast variables
                    var_name = f"{col}_hist" if not col.endswith('_hist') else col
                    data_vars[var_name] = (('time',), df[col].astype(np.float32).values)
            
                basin_ds = xr.Dataset(
                    data_vars,
                    coords={
                        'time': df.index.values,
                        'basin': basin_name
                    }
                )
                basin_ds = basin_ds.expand_dims('basin')
            
                # Add metadata
                basin_ds.attrs['source'] = 'local_csv_files'
                basin_ds.attrs['basin_file'] = str(basin_file)
                basin_ds.attrs['date_range'] = f"{start_date} to {end_date}"
            
                basin_data_list.append(basin_ds)
            
            except Exception as e:
                LOGGER.error(f"Error loading data for basin {basin_name}: {e}")
                continue
    
        if not basin_data_list:
            raise RuntimeError("No historical data could be loaded for any basin")
    
        # Combine all basin datasets
        try:
            combined_ds = xr.concat(basin_data_list, dim="basin")
            LOGGER.info(f"Successfully loaded historical data for {len(basin_data_list)} basins")
            return combined_ds
        except Exception as e:
            raise RuntimeError(f"Error combining historical datasets: {e}")


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
    predict_last_n: List[int]
        List of predict_last_n; one entry per frequency
    frequency_maps : List[np.ndarray]
        List of arrays mapping lowest-frequency samples to their corresponding last sample in each frequency;
        one list entry per frequency.

    Returns
    -------
    np.ndarray 
        Array has a value of 1 for valid samples and a value of 0 for invalid samples.
    """

    # number of samples is number of lowest-frequency samples (all maps have this length)
    n_samples = len(frequency_maps[0])
    if n_samples == 0:
        return np.array([])

    # 1 denotes valid sample, 0 denotes invalid sample
    flag = np.ones(n_samples)
    for i in range(len(frequency_maps)):  # iterate through frequencies
        for j in prange(n_samples):  # iterate through lowest-frequency samples

            # The length of the hindcast period is the total sequence length minus the length of the forecast sequence
            hindcast_seq_length = seq_length[i] - forecast_seq_length[i] 

            # find the last sample in this frequency that belongs to the lowest-frequency step j
            last_sample_of_freq = frequency_maps[i][j]
            # check whether there is sufficient data available to create a valid sequence (regardless of NaN etc, which are checked in the following sections)
            if last_sample_of_freq < (hindcast_seq_length - 1):
                flag[j] = 0  # too early for this frequency's seq_length (not enough history)
                continue

            # add forecast_offset here, because it determines how many timesteps ahead we're going to be predicting (remember forecast_offset is the number of timesteps between initialization and the first forecast)
            if x_h is not None and (last_sample_of_freq + forecast_offset + forecast_seq_length[i]) > len(x_h[i]):
                flag[j] = 0
                continue 
            if x_f is not None and (last_sample_of_freq + forecast_offset) >= len(x_f[i]):
                flag[j] = 0
                continue 

            # any NaN in the hindcast inputs makes the sample invalid
            if x_h is not None:
                # NOTE hindcast stops the day before the forecast starts, so don't need to slice end
                _x_h = x_h[i][last_sample_of_freq - hindcast_seq_length + 1:last_sample_of_freq + 1]
                if np.any(np.isnan(_x_h)):
                    flag[j] = 0
                    continue

            # any NaN in the forecast inputs make the sample invalid 
            if x_f is not None:
                # x_f is 3D: [time, lead_time, features]
                _x_f = x_f[i][last_sample_of_freq]  # shape: [lead_time, features]
                if np.any(np.isnan(_x_f)):
                    flag[j] = 0
                    continue

            # all-NaN in the targets makes the sample invalid
            if y is not None:
                _y = y[i][last_sample_of_freq - predict_last_n[i] + 1:last_sample_of_freq + 1]
                if np.prod(np.array(_y.shape)) > 0 and np.all(np.isnan(_y)):
                    flag[j] = 0
                    continue

    return flag

def load_timeseries(data_dir: Path, time_series_data_sub_dir: str, basin: str, columns: list) -> pd.DataFrame:
    """Load time series data from CSV files into pandas DataFrame.
    
    This function is adapted for the OnlineForecastDataset to load historical data from CSV files
    rather than NetCDF files, following the Harz dataset structure.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory. This folder must contain a folder called 'timeseries' containing the time series
        data for each basin as CSV files named 'hydromet_timeseries_{basin}.csv'.
    time_series_data_sub_dir : str
        Subdirectory within timeseries containing the data files (can be None for direct access).
    basin : str
        The basin identifier.
    columns : list
        List of column names to load from the CSV file.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame containing the time series data as stored in the CSV file.

    Raises
    ------
    FileNotFoundError
        If no CSV file exists for the specified basin.
    """
    timeseries_dir = data_dir / "timeseries"
    
    # Allow time series data from different members
    if time_series_data_sub_dir is not None:
        timeseries_dir = timeseries_dir / time_series_data_sub_dir

    # Look for CSV file following Harz naming convention
    csv_file = timeseries_dir / f"hydromet_timeseries_{basin}.csv"
    
    if not csv_file.exists():
        raise FileNotFoundError(f"No CSV file found for basin {basin} at {csv_file}")

    # Load CSV file
    df = pd.read_csv(csv_file, index_col='date', parse_dates=['date'])
    
    # Filter columns if specified
    if columns:
        available_columns = [col for col in columns if col in df.columns]
        if available_columns:
            df = df[available_columns]
        else:
            LOGGER.warning(f"None of the requested columns {columns} found in {csv_file}")
            # Return empty DataFrame with correct index
            df = pd.DataFrame(index=df.index)
    
    return df