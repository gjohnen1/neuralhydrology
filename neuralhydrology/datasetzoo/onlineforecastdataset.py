from pathlib import Path
import logging
import pickle
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import torch
import xarray as xr
from pandas.tseries.frequencies import to_offset
from ruamel.yaml import YAML
from tqdm import tqdm

from neuralhydrology.datasetzoo.genericdataset import GenericDataset
from neuralhydrology.datautils import utils
from neuralhydrology.datautils.fetch_basin_forecasts import (
    load_basin_centroids,
    fetch_forecasts_for_basins,
    interpolate_to_hourly,
)
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import NoEvaluationDataError, NoTrainDataError


LOGGER = logging.getLogger(__name__)


@dataclass
class _DataAvailability:
    historical_start: Optional[pd.Timestamp] = None
    historical_end: Optional[pd.Timestamp] = None
    forecast_start: Optional[pd.Timestamp] = None
    forecast_end: Optional[pd.Timestamp] = None

    def update_from_dataset(self, dataset: Optional[xr.Dataset], dim: str, kind: str) -> None:
        if dataset is None or dim not in dataset.coords:
            return
        coord_values = dataset[dim].values
        if coord_values.size == 0:
            return
        start = pd.to_datetime(coord_values[0])
        end = pd.to_datetime(coord_values[-1])
        if kind == 'historical':
            if self.historical_start is None or start < self.historical_start:
                self.historical_start = start
            if self.historical_end is None or end > self.historical_end:
                self.historical_end = end
        elif kind == 'forecast':
            if self.forecast_start is None or start < self.forecast_start:
                self.forecast_start = start
            if self.forecast_end is None or end > self.forecast_end:
                self.forecast_end = end

    def update_from_attrs(self, attrs: Dict[str, str]) -> None:
        mapping = {
            'cache_hist_start': 'historical_start',
            'cache_hist_end': 'historical_end',
            'historical_data_end': 'historical_end',
            'cache_issue_start': 'forecast_start',
            'cache_issue_end': 'forecast_end',
            'forecast_data_start': 'forecast_start',
        }
        for attr_key, field in mapping.items():
            value = attrs.get(attr_key)
            if value is None:
                continue
            timestamp = pd.to_datetime(value)
            current = getattr(self, field)
            if field.endswith('start'):
                if current is None or timestamp < current:
                    setattr(self, field, timestamp)
            else:
                if current is None or timestamp > current:
                    setattr(self, field, timestamp)

    def to_attrs(self) -> Dict[str, str]:
        attrs: Dict[str, str] = {}
        if self.historical_start is not None:
            attrs['cache_hist_start'] = str(self.historical_start)
        if self.historical_end is not None:
            attrs['cache_hist_end'] = str(self.historical_end)
            attrs['historical_data_end'] = str(self.historical_end)
        if self.forecast_start is not None:
            attrs['cache_issue_start'] = str(self.forecast_start)
            attrs['forecast_data_start'] = str(self.forecast_start)
        if self.forecast_end is not None:
            attrs['cache_issue_end'] = str(self.forecast_end)
        return attrs


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

    CACHE_VERSION = "full-span-v2"

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xr.DataArray]] = {}):
    
        self._x_h: Dict[str, Dict[str, torch.Tensor]] = {}
        self._x_f: Dict[str, Dict[str, torch.Tensor]] = {}
        self._issue_times: Dict[str, Dict[str, np.ndarray]] = {}
        self._availability = _DataAvailability()

        scaler = self._ensure_scaler(period=period, scaler=scaler, cfg=cfg)

        super().__init__(cfg=cfg,
                         is_train=is_train,
                         period=period,
                         basin=basin,
                         additional_features=additional_features,
                         id_to_int=id_to_int,
                         scaler=scaler)

    def _load_attributes(self) -> pd.DataFrame:
        """Load catchment attributes from *_attributes.csv files in data_dir."""
        attributes_path = self.cfg.data_dir

        if not attributes_path.exists():
            raise FileNotFoundError(f"Attribute folder not found at {attributes_path}")

        txt_files = list(attributes_path.glob('*_attributes.csv'))
        
        if not txt_files:
             return pd.DataFrame()

        # Read-in attributes into one big dataframe
        dfs = []
        for txt_file in txt_files:
            df_temp = pd.read_csv(txt_file, sep=',', header=0, dtype={'gauge_id': str})
            df_temp = df_temp.set_index('gauge_id')
            dfs.append(df_temp)

        df = pd.concat(dfs, axis=1)

        if self.basins:
            if any(b not in df.index for b in self.basins):
                raise ValueError('Some basins are missing static attributes.')
            df = df.loc[self.basins]

        return df

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
        basin_datasets = []
        
        # Ensure cache directory exists
        cache_dir = self.cfg.data_dir / "zarr_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        for basin in self.basins:
            cache_path = cache_dir / f"{basin}.zarr"
            
            if cache_path.exists():
                LOGGER.info(f"Loading cached dataset for basin {basin} from {cache_path}")
                try:
                    ds = xr.open_zarr(store=cache_path, decode_timedelta=True)
                    # Check version
                    if ds.attrs.get('onlineforecast_cache_version') == self.CACHE_VERSION:
                        basin_datasets.append(ds)
                        continue
                    else:
                        LOGGER.info(f"Cache version mismatch for basin {basin}. Rebuilding.")
                        ds.close()
                        shutil.rmtree(cache_path)
                except Exception as e:
                    LOGGER.warning(f"Failed to load cache for basin {basin}: {e}. Rebuilding.")
                    if cache_path.exists():
                        shutil.rmtree(cache_path)

            # Build and cache if not loaded
            ds = self._build_and_cache_basin_dataset(basin, cache_path)
            basin_datasets.append(ds)

        if not basin_datasets:
            if self.is_train:
                raise NoTrainDataError
            raise NoEvaluationDataError

        # Merge all basin datasets
        merged = xr.concat(basin_datasets, dim='basin')
        
        # Ensure frequencies are set (crucial if loading from cache)
        if not self.frequencies:
            inferred_freq = utils.infer_frequency(merged['time'].values)
            self.frequencies = [inferred_freq]
            LOGGER.info(f"Inferred frequency from dataset: {inferred_freq}")
        
        # Update availability from the merged dataset
        self._update_data_availability(merged_ds=merged)
        self._validate_data_availability(self.cfg)
        
        return merged

    def _build_and_cache_basin_dataset(self, basin: str, cache_path: Path) -> xr.Dataset:
        LOGGER.info(f"Building dataset for basin {basin}...")
        
        # Load raw data for this specific basin
        # We pass [basin] to fetch only what is needed
        historical_ds = self._load_historical_xarray_data(basins=[basin])
        forecast_ds = self._load_forecast_xarray_data(basins=[basin])

        if historical_ds is None or forecast_ds is None:
            raise ValueError(f"Failed to load raw data for basin {basin}")

        # Standardize dimensions
        if 'init_time' in forecast_ds.dims and 'issue_time' not in forecast_ds.dims:
            forecast_ds = forecast_ds.rename({'init_time': 'issue_time'})
        if 'time' in forecast_ds.dims and 'issue_time' not in forecast_ds.dims:
            forecast_ds = forecast_ds.rename({'time': 'issue_time'})

        # Sort
        forecast_ds = forecast_ds.sortby('issue_time')
        historical_ds = historical_ds.sortby('time')

        # Determine dynamic range (Forecast archive start -> Historic data end)
        # We want the full overlap where we have data
        hist_start = pd.to_datetime(historical_ds['time'].values[0])
        hist_end = pd.to_datetime(historical_ds['time'].values[-1])
        fcst_start = pd.to_datetime(forecast_ds['issue_time'].values[0])
        fcst_end = pd.to_datetime(forecast_ds['issue_time'].values[-1])
        
        # Let's implement the user's specific slicing request:
        slice_start = fcst_start
        slice_end = hist_end
        
        LOGGER.info(f"Slicing basin {basin} from {slice_start} (Forecast Start) to {slice_end} (Historic End)")
        
        # Slice Forecasts
        # We keep forecasts that issue within this range
        forecast_ds = forecast_ds.sel(issue_time=slice(slice_start, slice_end))
        
        # Slice History
        # We need history to cover the warmup period before the first forecast.
        # Calculating max warmup
        if not self.frequencies:
            inferred_freq = utils.infer_frequency(historical_ds['time'].values)
            self.frequencies = [inferred_freq]
            
        reference_ts = pd.Timestamp('2000-01-01')
        warmup_offsets = []
        for i, freq in enumerate(self.frequencies):
            # Default to 0 if not set yet (first run)
            seq_len = self.seq_len[i] if isinstance(self.seq_len, list) else self.seq_len
            fcst_len = self._forecast_seq_len[i] if isinstance(self._forecast_seq_len, list) else self._forecast_seq_len
            pred_last = self._predict_last_n[i] if isinstance(self._predict_last_n, list) else self._predict_last_n
            
            forecast_horizon = max(pred_last, fcst_len)
            offset = (seq_len - forecast_horizon) * to_offset(freq)
            warmup_offsets.append(reference_ts + offset - reference_ts)
        max_warmup = max(warmup_offsets) if warmup_offsets else pd.Timedelta(0)
        
        hist_start_needed = slice_start - max_warmup
        historical_ds = historical_ds.sel(time=slice(hist_start_needed, slice_end))

        # Merge
        # We use compat='override' to merge different coordinates (time vs issue_time)
        merged = xr.merge([historical_ds, forecast_ds], compat='override')
        
        # Add attributes
        merged.attrs['onlineforecast_cache_version'] = self.CACHE_VERSION
        merged.attrs['basin'] = basin
        
        # Save to Zarr with retries
        LOGGER.info(f"Saving cache for basin {basin} to {cache_path}")
        # Ensure basin coord is string
        if 'basin' in merged.coords:
            merged['basin'] = merged['basin'].astype(str)

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                merged.to_zarr(store=cache_path, mode='w')
                break
            except Exception as e:
                if attempt == max_retries:
                    LOGGER.error(f"Failed to save Zarr cache for basin {basin} after {max_retries} attempts.")
                    raise e
                LOGGER.warning(f"Attempt {attempt}/{max_retries} to save Zarr cache failed: {e}. Retrying in 5s...")
                time.sleep(5)
        
        # Re-open from cache to ensure we use the local copy and sever ties to remote
        merged.close()
        LOGGER.info(f"Reloading basin {basin} from newly created cache at {cache_path}")
        ds_cached = xr.open_zarr(store=cache_path, decode_timedelta=True)
        
        return ds_cached

    def _ensure_scaler(self,
                       period: str,
                       scaler: Dict[str, Union[pd.Series, xr.DataArray]],
                       cfg: Config) -> Dict[str, Union[pd.Series, xr.DataArray]]:
        if period not in ['validation', 'test'] or scaler:
            return scaler

        train_dir = getattr(cfg, 'train_dir', None)
        if train_dir is None:
            raise ValueError("cfg.train_dir must be set to automatically load the scaler for validation/test periods")

        try:
            return self._load_scaler(Path(train_dir))
        except FileNotFoundError as exc:  # noqa: B904
            raise ValueError("Scaler not provided and automatic loading from cfg.train_dir failed.") from exc

    def _load_scaler(self, train_dir: Path) -> Dict[str, Union[pd.Series, xr.Dataset]]:
        try:
            scaler = utils.load_scaler(train_dir)
            LOGGER.info("Loaded scaler from %s/train_data/train_data_scaler.yml", train_dir)
            return scaler
        except FileNotFoundError:
            pass

        yaml_path = train_dir / "train_data_scaler.yml"
        pickle_path = train_dir / "train_data_scaler.p"

        if yaml_path.exists():
            LOGGER.info("Loaded scaler from %s", yaml_path)
            with yaml_path.open("r") as fp:
                yaml_loader = YAML(typ="safe")
                scaler_dump = yaml_loader.load(fp)

            return self._deserialize_scaler_dict(scaler_dump)

        if pickle_path.exists():
            LOGGER.info("Loaded scaler from %s", pickle_path)
            with pickle_path.open('rb') as fp:
                return pickle.load(fp)

        raise FileNotFoundError(f"No scaler file found under {train_dir} (checked train_data folder and direct files).")

    @staticmethod
    def _deserialize_scaler_dict(scaler_dump: Dict[str, Dict]) -> Dict[str, Union[pd.Series, xr.Dataset]]:
        scaler: Dict[str, Union[pd.Series, xr.Dataset]] = {}
        for key, value in scaler_dump.items():
            if key in ["attribute_means", "attribute_stds", "camels_attr_means", "camels_attr_stds"]:
                scaler[key] = pd.Series(value)
            elif key in ["xarray_feature_scale", "xarray_feature_center"]:
                scaler[key] = xr.Dataset.from_dict(value).astype(np.float32)
        return scaler

    def _update_data_availability(self,
                                  historical_ds: Optional[xr.Dataset] = None,
                                  forecast_ds: Optional[xr.Dataset] = None,
                                  merged_ds: Optional[xr.Dataset] = None) -> None:
        if historical_ds is not None:
            self._availability.update_from_dataset(historical_ds, 'time', 'historical')
        if forecast_ds is not None:
            self._availability.update_from_dataset(forecast_ds, 'issue_time', 'forecast')

        if merged_ds is None:
            return

        self._availability.update_from_dataset(merged_ds, 'time', 'historical')
        issue_dim = 'issue_time' if 'issue_time' in merged_ds.coords else (
            'init_time' if 'init_time' in merged_ds.coords else None)
        if issue_dim is not None:
            self._availability.update_from_dataset(merged_ds, issue_dim, 'forecast')

        self._availability.update_from_attrs(merged_ds.attrs)

    def _filter_issue_times_for_period(self, basin: str, issue_times: np.ndarray) -> np.ndarray:
        if issue_times.size == 0:
            return issue_times

        start_dates = self.start_and_end_dates.get(basin, {}).get('start_dates', [])
        end_dates = self.start_and_end_dates.get(basin, {}).get('end_dates', [])
        start_dates = [pd.to_datetime(date) for date in start_dates]
        end_dates = [pd.to_datetime(date) for date in end_dates]

        issue_index = pd.to_datetime(issue_times)
        if start_dates and end_dates:
            adjusted_end_dates = [end_date + pd.Timedelta(days=1, seconds=-1) for end_date in end_dates]
            mask = np.zeros(issue_index.size, dtype=bool)
            for start_date, end_date in zip(start_dates, adjusted_end_dates):
                mask |= (issue_index >= start_date) & (issue_index <= end_date)
        else:
            mask = np.ones(issue_index.size, dtype=bool)

        forecast_start = self._availability.forecast_start
        forecast_end = self._availability.forecast_end
        if forecast_start is not None:
            mask &= issue_index >= forecast_start
        if forecast_end is not None:
            mask &= issue_index <= forecast_end

        return issue_index[mask].to_numpy(dtype='datetime64[ns]')

    def _load_forecast_xarray_data(self, basins: List[str] = None) -> xr.Dataset:
        """Load forecast data directly as xarray dataset following operational notebook approach."""
        target_basins = basins if basins is not None else self.basins
        
        LOGGER.info(f"Loading forecast data for basins: {target_basins}")
        
        # Load basin centroids (similar to operational notebook)
        basin_centroids_file = self.cfg.data_dir / "basin_centroids" / "basin_centroids.csv"
        if not basin_centroids_file.exists():
            LOGGER.warning(f"Basin centroids file not found: {basin_centroids_file}")
            return None
            
        centroids = load_basin_centroids(basin_centroids_file)
        
        # Filter centroids to only include basins we're working with
        centroids = centroids[centroids['basin_name'].isin(target_basins)]
        
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
        
        # Interpolate to hourly. Use the configured forecast sequence length.
        max_hours = max(self._forecast_seq_len)
        basin_forecasts_hourly = interpolate_to_hourly(basin_forecasts_quartiles, max_hours=max_hours)
        
        # Final filter to exactly match config (in case interpolation changed variable names)
        forecast_vars = [var for var in basin_forecasts_hourly.data_vars if var in self.cfg.forecast_inputs]
        if not forecast_vars:
            raise ValueError(f"After processing, none of cfg.forecast_inputs {self.cfg.forecast_inputs} found. "
                           f"Available after quartile computation: {list(basin_forecasts_hourly.data_vars)}")
        
        basin_forecasts_hourly = basin_forecasts_hourly[forecast_vars]
        
        LOGGER.info(f"Loaded forecast data with variables: {list(basin_forecasts_hourly.data_vars)}")
        return basin_forecasts_hourly
            
    def _load_historical_xarray_data(self, basins: List[str] = None) -> xr.Dataset:
        """Load historical data directly as xarray dataset."""
        target_basins = basins if basins is not None else self.basins
        
        LOGGER.info(f"Loading historical data for basins: {target_basins}")
        
        basin_datasets = []
        
        for basin in target_basins:
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
                # Convert to xarray immediately per basin
                ds = xr.Dataset.from_dataframe(df)
                # Rename date to time to match expected convention
                if 'date' in ds.dims:
                    ds = ds.rename({'date': 'time'})
                ds = ds.expand_dims(basin=[basin])
                basin_datasets.append(ds)
                LOGGER.info(f"Loaded {len(df)} records for basin {basin} covering {df.index.min().date()} to {df.index.max().date()}")
            else:
                LOGGER.warning(f"No requested variables found for basin {basin}")
        
        if not basin_datasets:
            LOGGER.warning("No historical data loaded for any basin")
            return None
        
        # Combine efficiently. 
        LOGGER.info("Merging basin datasets...")
        # This aligns all datasets along the time dimension (union of times) and concatenates along basin
        historical_ds = xr.concat(basin_datasets, dim='basin')
        
        LOGGER.info(f"Created historical dataset with {len(basin_datasets)} basins.")
        return historical_ds
            
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
        for freq, seq_len, forecast_seq_len, pointer in zip(self.frequencies, self.seq_len, self._forecast_seq_len, indices):
            # if there's just one frequency, don't use suffixes.
            freq_suffix = '' if len(self.frequencies) == 1 else f'_{freq}'
            hindcast_idx = pointer['hindcast_idx']
            forecast_idx = pointer['forecast_idx']
            issue_time = pointer.get('issue_time')

            hindcast_history = seq_len - forecast_seq_len
            # We want the hindcast to include the issue_time, so we shift the window by +1
            # hindcast_idx is the index of issue_time. Slicing [start:end] excludes end.
            # So to include issue_time, end must be hindcast_idx + 1.
            hindcast_start_idx = hindcast_idx + self._forecast_offset - hindcast_history + 1
            hindcast_end_idx = hindcast_idx + self._forecast_offset + 1
            global_end_idx = hindcast_idx + self._forecast_offset + forecast_seq_len + 1

            # Load from mmap (numpy) and convert to tensor
            x_h = self._x_h[basin][freq][hindcast_start_idx:hindcast_end_idx]
            sample[f'x_h{freq_suffix}'] = torch.from_numpy(x_h)
            
            x_f = self._x_f[basin][freq][forecast_idx]
            sample[f'x_f{freq_suffix}'] = torch.from_numpy(x_f)
            
            # Create dictionaries for InputLayer compatibility
            # InputLayer expects x_d_hindcast and x_d_forecast as dictionaries of tensors (seq_len, 1)
            sample[f'x_d_hindcast{freq_suffix}'] = {
                k: sample[f'x_h{freq_suffix}'][:, i].unsqueeze(-1) 
                for i, k in enumerate(self.cfg.hindcast_inputs)
            }
            sample[f'x_d_forecast{freq_suffix}'] = {
                k: sample[f'x_f{freq_suffix}'][:, i].unsqueeze(-1)
                for i, k in enumerate(self.cfg.forecast_inputs)
            }

            y = self._y[basin][freq][hindcast_start_idx:global_end_idx]
            sample[f'y{freq_suffix}'] = torch.from_numpy(y)
            
            sample[f'date{freq_suffix}'] = self._dates[basin][freq][hindcast_start_idx:global_end_idx]
            if issue_time is not None:
                sample[f'date_issue{freq_suffix}'] = issue_time

            # Handle static inputs
            static_inputs = []
            if self._attributes:
                static_inputs.append(self._attributes[basin])
            if self._x_s:
                static_inputs.append(self._x_s[basin][freq][hindcast_idx])
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
        lookup: List[Tuple[str, List[Dict[str, Union[int, np.datetime64]]]]] = []
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
        issue_dim = 'issue_time' if 'issue_time' in xr_fcst.dims else ('time' if 'time' in xr_fcst.dims else 'date')

        basins_without_samples: List[str] = []
        
        # Filter basin_coordinates to only include basins requested in self.basins
        # This ensures we only process and train on the basins specified in the config/basins.txt
        # even if the cache contains more basins.
        available_basins = set(xr_hcst['basin'].values.tolist())
        requested_basins = set(self.basins)
        basin_coordinates = list(available_basins.intersection(requested_basins))
        
        if not basin_coordinates:
             raise ValueError(f"None of the requested basins {self.basins} found in the dataset.")

        for basin in tqdm(basin_coordinates, file=sys.stdout, disable=self._disable_pbar):
            basin_hcst = xr_hcst.sel(basin=basin, drop=True)
            
            # Optimization: Determine time range for this basin and period to slice BEFORE loading
            # This prevents loading the entire history into RAM when we only need a small slice (e.g. test period)
            start_dates = self.start_and_end_dates.get(basin, {}).get('start_dates', [])
            end_dates = self.start_and_end_dates.get(basin, {}).get('end_dates', [])
            
            basin_fcst = xr_fcst.sel(basin=basin, drop=True)
            
            if start_dates and end_dates:
                # Convert to timestamps and find global min/max for this basin
                min_time = pd.to_datetime(min(start_dates))
                max_time = pd.to_datetime(max(end_dates)) + pd.Timedelta(days=1) # Add buffer for end of day
                
                # Slice the Zarr array lazily
                basin_fcst = basin_fcst.sel({issue_dim: slice(min_time, max_time)})
            
            # Now load the (potentially much smaller) slice into memory
            basin_fcst = basin_fcst.load()

            if time_dim not in basin_hcst.dims:
                LOGGER.warning("Time dimension not found for basin %s - skipping.", basin)
                basins_without_samples.append(basin)
                continue

            if issue_dim not in basin_fcst.dims:
                LOGGER.warning("Issue-time dimension not found for basin %s - skipping.", basin)
                basins_without_samples.append(basin)
                continue

            hindcast_df = basin_hcst.to_dataframe().reset_index().set_index(time_dim).sort_index()
            if hindcast_df.empty:
                LOGGER.warning("Hindcast data empty for basin %s - skipping.", basin)
                basins_without_samples.append(basin)
                continue

            filtered_issue_times = self._filter_issue_times_for_period(basin, basin_fcst[issue_dim].values)
            if filtered_issue_times.size == 0:
                LOGGER.warning("No forecast issue times within configured period for basin %s - skipping.", basin)
                basins_without_samples.append(basin)
                continue

            basin_fcst = basin_fcst.sel({issue_dim: filtered_issue_times})

            forecast_df = basin_fcst.to_dataframe().reset_index().set_index([issue_dim, 'lead_time']).sort_index()
            if forecast_df.empty:
                LOGGER.warning("Forecast data empty for basin %s - skipping.", basin)
                basins_without_samples.append(basin)
                continue

            issue_times = forecast_df.index.get_level_values(issue_dim).unique()
            if issue_times.empty:
                LOGGER.warning("No forecast issue times available for basin %s - skipping.", basin)
                basins_without_samples.append(basin)
                continue

            # Verify all configured features are present
            missing_hindcast = set(self.cfg.hindcast_inputs) - set(hindcast_df.columns)
            if missing_hindcast:
                raise ValueError(f'Missing hindcast inputs for basin {basin}: {missing_hindcast}')
            available_hindcast = self.cfg.hindcast_inputs

            missing_targets = set(self.cfg.target_variables) - set(hindcast_df.columns)
            if missing_targets:
                raise ValueError(f'Missing target variables for basin {basin}: {missing_targets}')
            available_targets = self.cfg.target_variables

            missing_forecast = set(self.cfg.forecast_inputs) - set(forecast_df.columns)
            if missing_forecast:
                raise ValueError(f'Missing forecast inputs for basin {basin}: {missing_forecast}')
            available_forecast = self.cfg.forecast_inputs

            hindcast_matrix = hindcast_df[available_hindcast].to_numpy(dtype=np.float32)
            target_matrix = hindcast_df[available_targets].to_numpy(dtype=np.float32)
            date_values = hindcast_df.index.to_numpy()

            fc_inputs = basin_fcst[available_forecast]
            # NOAA GEFS precipitation rates are averaged since the previous step, so the first
            # lead-time in each issue can be NaN after interpolation; fill forward to retain samples.
            if 'lead_time' in fc_inputs.dims:
                fc_inputs = fc_inputs.bfill(dim='lead_time')
            
            # Manual stacking to avoid xarray.to_array() reshaping errors and ensure correct feature order
            fc_tensor_list = []
            for var in available_forecast:
                da = fc_inputs[var]
                if 'lead_time' in da.dims:
                    da = da.transpose(issue_dim, 'lead_time')
                    fc_tensor_list.append(da.values)
                else:
                    da = da.transpose(issue_dim)
                    fc_tensor_list.append(da.values[:, np.newaxis])
            
            fc_tensor = np.stack(fc_tensor_list, axis=-1).astype(np.float32)

            required_len = max(self._forecast_seq_len)
            if fc_tensor.shape[1] < required_len:
                LOGGER.warning(f"Forecast tensor length ({fc_tensor.shape[1]}) is shorter than required forecast_seq_length ({required_len}) for basin {basin} - skipping.")
                basins_without_samples.append(basin)
                continue

            hindcast_index = pd.Index(hindcast_df.index)
            issue_time_values = issue_times.to_numpy()
            hindcast_positions = hindcast_index.get_indexer(issue_times)

            self._x_h.setdefault(basin, {})
            self._x_f.setdefault(basin, {})
            self._y.setdefault(basin, {})
            self._dates.setdefault(basin, {})
            self._issue_times.setdefault(basin, {})

            validity_masks: List[np.ndarray] = []
            
            for freq_idx, freq in enumerate(self.frequencies):
                hindcast_history = self.seq_len[freq_idx] - self._forecast_seq_len[freq_idx]
                if hindcast_history <= 0:
                    raise ValueError('seq_length must exceed forecast_seq_length to provide hindcast context.')

                validity = np.zeros(len(issue_times), dtype=bool)

                for candidate_idx, anchor_idx in enumerate(hindcast_positions):
                    if anchor_idx < 0:
                        continue

                    # Shifted by +1 to include issue_time in hindcast
                    hindcast_start = anchor_idx + self._forecast_offset - hindcast_history + 1
                    hindcast_end = anchor_idx + self._forecast_offset + 1
                    forecast_end = anchor_idx + self._forecast_offset + self._forecast_seq_len[freq_idx] + 1

                    if hindcast_start < 0:
                        continue
                    if forecast_end > target_matrix.shape[0]:
                        continue

                    if self.is_train:
                        hindcast_window = hindcast_matrix[hindcast_start:hindcast_end]
                        if np.any(np.isnan(hindcast_window)):
                            continue

                        forecast_window = fc_tensor[candidate_idx, :self._forecast_seq_len[freq_idx]]
                        if np.any(np.isnan(forecast_window)):
                            continue

                        target_window = target_matrix[hindcast_start:forecast_end]
                        predict_last_n = self._predict_last_n[freq_idx]
                        if predict_last_n > 0:
                            tail = target_window[-predict_last_n:]
                            if tail.size > 0 and np.all(np.isnan(tail)):
                                continue

                    validity[candidate_idx] = True

                validity_masks.append(validity)

            if not validity_masks:
                basins_without_samples.append(basin)
                continue

            combined_validity = np.logical_and.reduce(validity_masks)
            # The last forecast_seq_len issue times usually remain False because we cannot
            # form a full target horizon once we move past the available observation window.
            valid_indices = np.where(combined_validity)[0]

            if valid_indices.size == 0:
                basins_without_samples.append(basin)
                continue

            for freq in self.frequencies:
                # Define cache paths
                cache_dir = self.cfg.train_dir / "binary_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                h_file = cache_dir / f"{basin}_{freq}_{self.period}_x_h.npy"
                f_file = cache_dir / f"{basin}_{freq}_{self.period}_x_f.npy"
                y_file = cache_dir / f"{basin}_{freq}_{self.period}_y.npy"
                
                # Save and memmap
                np.save(h_file, hindcast_matrix)
                np.save(f_file, fc_tensor)
                np.save(y_file, target_matrix)
                
                self._x_h[basin][freq] = np.load(h_file, mmap_mode='r')
                self._x_f[basin][freq] = np.load(f_file, mmap_mode='r')
                self._y[basin][freq] = np.load(y_file, mmap_mode='r')
                
                self._dates[basin][freq] = date_values
                self._issue_times[basin][freq] = issue_time_values

            if not self.is_train:
                start_dates = self.start_and_end_dates.get(basin, {}).get('start_dates', [])
                if start_dates:
                    self.period_starts[basin] = pd.to_datetime(start_dates[0])
                else:
                    self.period_starts[basin] = pd.to_datetime(date_values[0])

            for idx in valid_indices:
                pointers = []
                for freq in self.frequencies:
                    pointers.append({
                        'hindcast_idx': int(hindcast_positions[idx]),
                        'forecast_idx': int(idx),
                        'issue_time': issue_time_values[idx],
                    })
                lookup.append((basin, pointers))

        if basins_without_samples:
            LOGGER.info("These basins do not have a single valid sample in the %s period: %s",
                        self.period, basins_without_samples)

        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}
        self.num_samples = len(self.lookup_table)

        if self.num_samples == 0:
            if self.is_train:
                raise NoTrainDataError
            raise NoEvaluationDataError

    def _validate_data_availability(self, cfg: Config):
        """Validate that configured time periods are within available data ranges.
        
        This method checks that all configured train, validation, and test periods
        fall within the available data ranges for both historical and forecast data
        that were detected while loading the datasets.

        Parameters
        ----------
        cfg : Config
            The run configuration containing time period definitions
        
        Raises
        ------
        ValueError
            If any configured period extends outside available data ranges
        """
        availability = self._availability
        historical_end = availability.historical_end
        forecast_start = availability.forecast_start
        historical_start = availability.historical_start
        forecast_end = availability.forecast_end

        if historical_end is None or forecast_start is None:
            LOGGER.warning("Data availability bounds are unknown. Skipping validation.")
            return

        LOGGER.info("Validating data availability against configured time periods...")
        LOGGER.info(f"  Historical coverage: {historical_start} to {historical_end}")
        LOGGER.info(f"  Forecast coverage: {forecast_start} to {forecast_end}")
        
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
            needs_forecast_data = forecast_start is not None and period_end >= forecast_start
            
            # Check if period requires historical data (starts before or overlaps with historical availability)
            needs_historical_data = historical_end is not None and period_start <= historical_end
            
            # Validate forecast data availability
            if needs_forecast_data and period_start < forecast_start:
                gap_days = (forecast_start - period_start).days
                errors.append(
                    f"{period_name.capitalize()} period starts {gap_days} days before forecast data becomes available. "
                    f"Period starts: {period_start.date()}, forecast data starts: {forecast_start.date()}"
                )
            
            # Validate historical data availability  
            if needs_historical_data and period_end > historical_end:
                gap_days = (period_end - historical_end).days
                errors.append(
                    f"{period_name.capitalize()} period extends {gap_days} days beyond available historical data. "
                    f"Period ends: {period_end.date()}, historical data ends: {historical_end.date()}"
                )
            
            # Check for periods that fall entirely outside available data ranges
            if (forecast_start is not None and period_end < forecast_start) and \
               (historical_end is not None and period_start > historical_end):
                errors.append(
                    f"{period_name.capitalize()} period falls entirely outside available data ranges. "
                    f"Period: {period_start.date()} to {period_end.date()}"
                )
            
            # For OnlineForecastDataset, periods without forecast data are considered errors
            # since this dataset is specifically designed for forecast operations
            if forecast_start is not None and period_end < forecast_start and needs_historical_data:
                gap_days = (forecast_start - period_end).days
                errors.append(
                    f"{period_name.capitalize()} period ends {gap_days} days before forecast data becomes available. "
                    f"OnlineForecastDataset requires forecast data availability. "
                    f"Period ends: {period_end.date()}, forecast data starts: {forecast_start.date()}"
                )
            
            # Also error for periods that start after historical data ends (no historical context)
            if historical_end is not None and period_start > historical_end:
                gap_days = (period_start - historical_end).days
                errors.append(
                    f"{period_name.capitalize()} period starts {gap_days} days after historical data ends. "
                    f"No historical context available for model training/evaluation. "
                    f"Period starts: {period_start.date()}, historical data ends: {historical_end.date()}"
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
            error_msg += f"  • Historical data: up to {historical_end.date()} {historical_end.time()}\n"
            error_msg += f"  • Forecast data: from {forecast_start.date()} {forecast_start.time()} onwards\n\n"
            error_msg += f"Suggested fixes:\n"
            error_msg += f"  • Adjust configured date ranges to fall within available data periods\n"
            error_msg += f"  • Update data sources to extend availability ranges"
            
            raise ValueError(error_msg)
        
        LOGGER.info("✅ Data availability validation passed")



