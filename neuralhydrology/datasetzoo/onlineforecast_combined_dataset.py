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


# ICON-D2 Horizon is 48 hours
ICOND2_HORIZON = 48

# Basin mapping for ICON-D2
BASIN_TO_CATCHMENT = {
    'DE1': 'innerste',
    'DE2': 'oker',
    'DE3': 'ecker',
    'DE4': 'soese',
    'DE5': 'grane',
}


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


class CombinedForecastDataset(GenericDataset):
    """Combined forecast dataset class for operational forecasting with mixed temporal indexing.
    
    This dataset handles operational forecast data from two sources:
    1. ICON-D2: Short-range 48h forecasts (high resolution)
    2. GEFS: Extended 10-day forecasts

    Historical/Hindcast data is loaded from local CSV files.
    Forecast variables are indexed by (basin, time, lead_time).

    ICON-D2 data is sourced from local NetCDF files in data/harz/icond2/.
    GEFS data is sourced from online NOAA GEFS zarr store.

    ICON-D2 variables are suffixed with `_icond2` (e.g. temperature_mean_icond2).
    ICON-D2 data is zero-padded for lead times beyong 48h.
    An `icond2_available` feature is added as a binary mask (1 for h1-48, 0 for h49+).

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

    CACHE_VERSION = "icond2-gefs-v1"

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

        # Collect all required variables for cache validation
        required_hindcast_vars = set(self.cfg.hindcast_inputs + self.cfg.target_variables)
        required_forecast_vars = set(self.cfg.forecast_inputs)
        
        for basin in self.basins:
            cache_path = cache_dir / f"{basin}_combined.zarr"
            
            if cache_path.exists():
                LOGGER.info(f"Loading cached dataset for basin {basin} from {cache_path}")
                try:
                    ds = xr.open_zarr(store=cache_path, decode_timedelta=True)
                    # Check version
                    if ds.attrs.get('onlineforecast_cache_version') != self.CACHE_VERSION:
                        LOGGER.info(f"Cache version mismatch for basin {basin}. Rebuilding.")
                        ds.close()
                        shutil.rmtree(cache_path)
                    else:
                        # Check if all required variables are present in the cache
                        cached_vars = set(ds.data_vars)
                        missing_hindcast = required_hindcast_vars - cached_vars
                        missing_forecast = required_forecast_vars - cached_vars
                        
                        if missing_hindcast or missing_forecast:
                            LOGGER.info(f"Cache for basin {basin} is missing required variables. Rebuilding.")
                            if missing_hindcast:
                                LOGGER.info(f"  Missing hindcast/target vars: {missing_hindcast}")
                            if missing_forecast:
                                LOGGER.info(f"  Missing forecast vars: {missing_forecast}")
                            ds.close()
                            shutil.rmtree(cache_path)
                        else:
                            basin_datasets.append(ds)
                            continue
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

        # Determine dynamic range
        hist_start = pd.to_datetime(historical_ds['time'].values[0])
        hist_end = pd.to_datetime(historical_ds['time'].values[-1])
        fcst_start = pd.to_datetime(forecast_ds['issue_time'].values[0])
        
        slice_start = fcst_start
        slice_end = hist_end
        
        LOGGER.info(f"Slicing basin {basin} from {slice_start} (Forecast Start) to {slice_end} (Historic End)")
        
        # Slice Forecasts
        forecast_ds = forecast_ds.sel(issue_time=slice(slice_start, slice_end))
        
        # Slice History
        if not self.frequencies:
            inferred_freq = utils.infer_frequency(historical_ds['time'].values)
            self.frequencies = [inferred_freq]
            
        reference_ts = pd.Timestamp('2000-01-01')
        warmup_offsets = []
        for i, freq in enumerate(self.frequencies):
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
        merged = xr.merge([historical_ds, forecast_ds], compat='override')
        
        # Add attributes
        merged.attrs['onlineforecast_cache_version'] = self.CACHE_VERSION
        merged.attrs['basin'] = basin
        
        # IMPORTANT: Load all data into memory before saving to zarr.
        # This decouples the remote data fetch from the local save operation,
        # avoiding failures when the remote server (NOAA GEFS) is intermittently unavailable.
        LOGGER.info(f"Loading data into memory for basin {basin} before caching...")
        merged = merged.compute()
        
        # Save to Zarr with retries
        LOGGER.info(f"Saving cache for basin {basin} to {cache_path}")
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
        except FileNotFoundError as exc:
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
        """Load and merge ICON-D2 and GEFS forecast data."""
        target_basins = basins if basins is not None else self.basins
        
        # 1. Load ICON-D2
        icond2_ds = self._load_icond2_forecasts(target_basins)
        
        # 2. Load GEFS (NOAA standard approach)
        gefs_ds = self._load_gefs_forecasts(target_basins)
        
        # 3. Merge sources (parallel features with different valid horizons)
        # Note: We align on common issue times.
        merged_ds = self._merge_forecast_sources(icond2_ds, gefs_ds)
        
        return merged_ds

    def _load_icond2_forecasts(self, basins: List[str]) -> xr.Dataset:
        """Load ICON-D2 deterministic and ensemble forecasts from local NetCDF."""
        datasets = []
        icond2_dir = self.cfg.data_dir / "icond2"
        
        # Identify variables to load based on forecast_inputs (those ending in _icond2)
        icond2_vars = [v.replace('_icond2', '') for v in self.cfg.forecast_inputs if v.endswith('_icond2') and not '_q' in v]
        # Variables with quartiles (assuming standard naming convention e.g. _icond2_q50)
        icond2_quartile_vars = [v.replace('_icond2', '').split('_q')[0] for v in self.cfg.forecast_inputs if '_icond2_q' in v]
        
        # Unique base variables needed
        base_vars_needed = set(icond2_vars + icond2_quartile_vars)
        if not base_vars_needed:
            LOGGER.warning("No ICON-D2 variables found in forecast_inputs (looking for *_icond2 suffixes).")
            return None

        for basin in basins:
            catchment = BASIN_TO_CATCHMENT.get(basin)
            if not catchment:
                LOGGER.warning(f"No catchment mapping for basin {basin}. Skipping ICON-D2.")
                continue
                
            LOGGER.info(f"Loading ICON-D2 for basin {basin} (catchment: {catchment})...")
            
            # 1. Load Deterministic
            # Path: forecasts/icond2_deterministic/CAMELS_DE_1h_deterministic_met_forecast_gregor_{catchment}.nc
            det_path = icond2_dir / "forecasts" / "icond2_deterministic" / f"CAMELS_DE_1h_deterministic_met_forecast_gregor_{catchment}.nc"
            
            basin_ds_parts = []
            
            if det_path.exists():
                det_ds = xr.open_dataset(det_path, decode_timedelta=True)
                # Filter to 00Z only
                det_ds = det_ds.sel(init_time=det_ds.init_time.dt.hour == 0)
                
                # Standardize dimensions
                if 'init_time' in det_ds.dims:
                    det_ds = det_ds.rename({'init_time': 'issue_time'})
                if 'gauge_id' in det_ds.coords:
                    det_ds = det_ds.drop_vars('gauge_id')
                
                # Only keep needed variables that exist in the file
                available_vars = set(det_ds.data_vars)
                vars_to_keep = [v for v in base_vars_needed if v in available_vars]
                
                if not vars_to_keep:
                    LOGGER.warning(f"None of the required variables {base_vars_needed} found in {det_path}")
                else:
                    # Rename variables to have _icond2 suffix
                    rename_map = {v: f"{v}_icond2" for v in vars_to_keep}
                    det_ds = det_ds[vars_to_keep].rename(rename_map)
                    basin_ds_parts.append(det_ds)
            else:
                LOGGER.warning(f"ICON-D2 Deterministic file not found: {det_path}")

            # 2. Load Ensemble (Precipitation only usually)
            # Path: forecasts/icond2_ensemble/CAMELS_DE_1h_ensemble_met_forecast_gregor_{catchment}.nc
            ens_path = icond2_dir / "forecasts" / "icond2_ensemble" / f"CAMELS_DE_1h_ensemble_met_forecast_gregor_{catchment}.nc"
            
            if ens_path.exists():
                ens_ds = xr.open_dataset(ens_path, decode_timedelta=True)
                # Filter to 00Z only
                ens_ds = ens_ds.sel(init_time=ens_ds.init_time.dt.hour == 0)
                
                # Only keep variables that are needed for quartiles
                available_vars = set(ens_ds.data_vars)
                # icond2_quartile_vars contains base names like 'precipitation_mean'
                # Use set() to avoid processing duplicates (q25, q50, q75 all map to same base var)
                vars_for_quartiles = list(set(v for v in icond2_quartile_vars if v in available_vars))
                
                if not vars_for_quartiles:
                    LOGGER.warning(f"None of the required ensemble variables {set(icond2_quartile_vars)} found in {ens_path}")
                else:
                    ens_ds = ens_ds[vars_for_quartiles]
                    
                    # Compute quartiles
                    ens_quartiles = self._compute_forecast_quartiles_as_variables(ens_ds, suffix_base="_icond2")
                    
                    # Standardize dimensions
                    if 'init_time' in ens_quartiles.dims:
                        ens_quartiles = ens_quartiles.rename({'init_time': 'issue_time'})
                    if 'gauge_id' in ens_quartiles.coords:
                        ens_quartiles = ens_quartiles.drop_vars('gauge_id')
                    
                    basin_ds_parts.append(ens_quartiles)
            else:
                LOGGER.warning(f"ICON-D2 Ensemble file not found: {ens_path}")

            if basin_ds_parts:
                merged = xr.merge(basin_ds_parts)
                # Add basin/lead_time coordinate management if needed
                merged = merged.assign_coords(basin=basin)
                # Ensure lead_time is integer hours if not already
                if pd.api.types.is_timedelta64_dtype(merged.lead_time):
                    # Convert to hours integer
                    merged = merged.assign_coords(lead_time=(merged.lead_time / pd.Timedelta('1h')).astype(int))
                
                datasets.append(merged)
        
        if not datasets:
            return None
            
        return xr.concat(datasets, dim='basin')

    def _load_gefs_forecasts(self, basins: List[str]) -> xr.Dataset:
        """Load GEFS forecasts with retry logic and fallback to existing cache."""
        # First, try to load from existing OnlineForecastDataset cache
        cache_dir = self.cfg.data_dir / "zarr_cache"
        gefs_inputs = [v for v in self.cfg.forecast_inputs if '_gefs' in v]
        
        # Map from config names (with _gefs) to cache names (without _gefs)
        # e.g., 'temperature_2m_gefs_q50' -> 'temperature_2m_q50'
        config_to_cache_name = {}
        cache_to_config_name = {}
        for v in gefs_inputs:
            cache_name = v.replace('_gefs', '')
            config_to_cache_name[v] = cache_name
            cache_to_config_name[cache_name] = v
        
        # Check if we can load from existing basin caches
        cached_datasets = []
        for basin in basins:
            basin_cache = cache_dir / f"{basin}.zarr"
            if basin_cache.exists():
                try:
                    cached_ds = xr.open_zarr(store=basin_cache, decode_timedelta=True)
                    
                    # Check if all GEFS variables are present (try both naming conventions)
                    available_direct = [v for v in gefs_inputs if v in cached_ds.data_vars]
                    available_mapped = [v for v in gefs_inputs if config_to_cache_name.get(v) in cached_ds.data_vars]
                    
                    if len(available_direct) == len(gefs_inputs):
                        # Direct match (new naming with _gefs suffix)
                        LOGGER.info(f"Loading GEFS data for basin {basin} from existing cache: {basin_cache}")
                        cached_datasets.append(cached_ds[gefs_inputs])
                    elif len(available_mapped) == len(gefs_inputs):
                        # Mapped match (old naming without _gefs suffix) - rename variables
                        LOGGER.info(f"Loading GEFS data for basin {basin} from existing cache (renaming vars): {basin_cache}")
                        cache_vars = [config_to_cache_name[v] for v in gefs_inputs]
                        subset = cached_ds[cache_vars]
                        # Rename to match config expectations
                        rename_map = {cache_to_config_name[old]: old for old in cache_vars}
                        rename_map = {v: k for k, v in rename_map.items()}  # Invert: cache_name -> config_name
                        subset = subset.rename({config_to_cache_name[v]: v for v in gefs_inputs})
                        cached_datasets.append(subset)
                    else:
                        LOGGER.info(f"Cache {basin_cache} missing some GEFS vars, will fetch from NOAA")
                        LOGGER.info(f"  Expected: {gefs_inputs}")
                        LOGGER.info(f"  Available (direct): {available_direct}")
                        LOGGER.info(f"  Available (mapped): {[config_to_cache_name.get(v) for v in available_mapped]}")
                        cached_ds.close()
                except Exception as e:
                    LOGGER.warning(f"Could not load cache {basin_cache}: {e}")
        
        if len(cached_datasets) == len(basins):
            LOGGER.info("Using GEFS data from existing caches (no NOAA fetch needed)")
            return xr.concat(cached_datasets, dim='basin')
        
        # Close any partially loaded caches
        for ds in cached_datasets:
            ds.close()
        
        # Fall back to fetching from NOAA with retry logic
        LOGGER.info("Fetching GEFS data from NOAA (with retry logic)...")
        
        # Load basin centroids
        basin_centroids_file = self.cfg.data_dir / "basin_centroids" / "basin_centroids.csv"
        if not basin_centroids_file.exists():
            LOGGER.warning(f"Basin centroids file not found: {basin_centroids_file}")
            return None
            
        centroids = load_basin_centroids(basin_centroids_file)
        centroids = centroids[centroids['basin_name'].isin(basins)]
        
        max_retries = 5
        retry_delay = 10  # seconds
        
        for attempt in range(1, max_retries + 1):
            try:
                LOGGER.info(f"Connecting to NOAA GEFS dataset (attempt {attempt}/{max_retries})...")
                ds = xr.open_zarr(
                    "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr?email=optional@email.com", 
                    decode_timedelta=True
                )
                
                # Identify GEFS variables (those containing _gefs)
                base_vars_needed = set()
                for var in self.cfg.forecast_inputs:
                    if '_gefs' in var:
                        base_var = var.replace('_gefs', '').replace('_q25', '').replace('_q50', '').replace('_q75', '')
                        base_vars_needed.add(base_var)
                
                available_base_vars = [v for v in base_vars_needed if v in ds.data_vars]
                if not available_base_vars:
                    LOGGER.warning("No GEFS base variables found in NOAA dataset.")
                    return None
                    
                ds = ds[available_base_vars]
                
                # Fetch
                basin_forecasts = fetch_forecasts_for_basins(ds, centroids)
                
                # Quartiles
                basin_forecasts_quartiles = self._compute_forecast_quartiles_as_variables(basin_forecasts, suffix_base="_gefs")
                
                # Interpolate
                max_hours = max(self._forecast_seq_len)
                basin_forecasts_hourly = interpolate_to_hourly(basin_forecasts_quartiles, max_hours=max_hours)
                
                # IMPORTANT: Load into memory to finalize remote fetch
                LOGGER.info("Loading GEFS data into memory...")
                basin_forecasts_hourly = basin_forecasts_hourly.compute()
                
                # Normalize dimension name: GEFS uses 'init_time', we standardize to 'issue_time'
                if 'init_time' in basin_forecasts_hourly.dims and 'issue_time' not in basin_forecasts_hourly.dims:
                    basin_forecasts_hourly = basin_forecasts_hourly.rename({'init_time': 'issue_time'})
                
                # Keep only configured GEFS inputs
                gefs_result = [v for v in self.cfg.forecast_inputs if v in basin_forecasts_hourly.data_vars]
                LOGGER.info(f"Successfully loaded GEFS data with {len(gefs_result)} variables")
                return basin_forecasts_hourly[gefs_result]
                
            except Exception as e:
                LOGGER.warning(f"Attempt {attempt}/{max_retries} to fetch GEFS failed: {e}")
                if attempt < max_retries:
                    LOGGER.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    LOGGER.error(f"Failed to fetch GEFS data after {max_retries} attempts")
                    raise

    def _merge_forecast_sources(self, icond2_ds: Optional[xr.Dataset], gefs_ds: Optional[xr.Dataset]) -> xr.Dataset:
        """Merge ICON-D2 and GEFS with zero-padding for short horizons."""
        if icond2_ds is None and gefs_ds is None:
            raise ValueError("No forecast data loaded from either source.")
        
        if icond2_ds is None:
            LOGGER.warning("Using only GEFS data.")
            return gefs_ds
        if gefs_ds is None:
            LOGGER.warning("Using only ICON-D2 data.")
            return icond2_ds

        # Ensure both datasets have integer lead_time (hours)
        if pd.api.types.is_timedelta64_dtype(icond2_ds.lead_time):
            LOGGER.info("Converting ICON-D2 lead_time from timedelta to integer hours")
            icond2_ds = icond2_ds.assign_coords(lead_time=(icond2_ds.lead_time / pd.Timedelta('1h')).astype(int))
        if pd.api.types.is_timedelta64_dtype(gefs_ds.lead_time):
            LOGGER.info("Converting GEFS lead_time from timedelta to integer hours")
            gefs_ds = gefs_ds.assign_coords(lead_time=(gefs_ds.lead_time / pd.Timedelta('1h')).astype(int))

        # Determine overlapping time range to avoid warnings about non-overlapping periods
        icond2_start = pd.to_datetime(icond2_ds.issue_time.values.min())
        icond2_end = pd.to_datetime(icond2_ds.issue_time.values.max())
        gefs_start = pd.to_datetime(gefs_ds.issue_time.values.min())
        gefs_end = pd.to_datetime(gefs_ds.issue_time.values.max())
        
        overlap_start = max(icond2_start, gefs_start)
        overlap_end = min(icond2_end, gefs_end)
        
        LOGGER.info(f"ICON-D2 coverage: {icond2_start} to {icond2_end}")
        LOGGER.info(f"GEFS coverage: {gefs_start} to {gefs_end}")
        LOGGER.info(f"Using overlapping period: {overlap_start} to {overlap_end}")
        
        if overlap_start > overlap_end:
            raise ValueError(f"No overlapping period between ICON-D2 ({icond2_start} to {icond2_end}) "
                           f"and GEFS ({gefs_start} to {gefs_end}).")
        
        # Pre-filter both datasets to the overlapping period
        icond2_ds = icond2_ds.sel(issue_time=slice(overlap_start, overlap_end))
        gefs_ds = gefs_ds.sel(issue_time=slice(overlap_start, overlap_end))

        # Align time range (intersection of issue times within the overlap period)
        common_times = np.intersect1d(icond2_ds.issue_time.values, gefs_ds.issue_time.values)
                           
        if len(common_times) == 0:
            raise ValueError("No overlapping issue times between ICON-D2 and GEFS data.")
            
        icond2_aligned = icond2_ds.sel(issue_time=common_times)
        gefs_aligned = gefs_ds.sel(issue_time=common_times)
        
        LOGGER.info(f"Merging forecasts on {len(common_times)} common issue times.")

        # Store valid ICON-D2 lead times BEFORE re-indexing (already converted to int in _load_icond2_forecasts)
        valid_icond2_leads = set(icond2_aligned.lead_time.values.tolist())
        
        # Pad ICON-D2 to full GEFS length
        # Re-index ICON-D2 to match GEFS lead times (1..240), filling missing with 0
        icond2_expanded = icond2_aligned.reindex(lead_time=gefs_aligned.lead_time, fill_value=0.0)
        
        # Create availability mask: 1 where ICON-D2 data existed, 0 where padded
        # We need dimensions (basin, issue_time, lead_time) to match other variables
        template_var = list(gefs_aligned.data_vars)[0]
        mask_da = xr.full_like(gefs_aligned[template_var], 0.0)  # Start with all zeros
        
        # Apply the lead-time mask across all basins and issue times
        # valid_icond2_leads are integer hours (1..48)
        gefs_lead_values = gefs_aligned.lead_time.values
        for lead in gefs_lead_values:
            if lead in valid_icond2_leads:
                mask_da.loc[{'lead_time': lead}] = 1.0
        mask_da.name = 'icond2_available'
        
        # Merge all
        merged = xr.merge([gefs_aligned, icond2_expanded, mask_da])
        return merged

    def _load_historical_xarray_data(self, basins: List[str] = None) -> xr.Dataset:
        """Load historical data directly as xarray dataset."""
        target_basins = basins if basins is not None else self.basins
        LOGGER.info(f"Loading historical data for basins: {target_basins}")
        
        basin_datasets = []
        for basin in target_basins:
            csv_file = self.cfg.data_dir / "timeseries" / f"hydromet_timeseries_{basin}.csv"
            if not csv_file.exists():
                LOGGER.warning(f"File not found for basin {basin}: {csv_file}")
                continue
                
            LOGGER.info(f"Loading data for basin {basin} from {csv_file}")
            df = pd.read_csv(csv_file)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            # Deduplicate wanted_cols while preserving order
            wanted_cols = list(dict.fromkeys(self.cfg.hindcast_inputs + self.cfg.target_variables))
            available_cols = df.columns.tolist()
            keep_cols = [col for col in wanted_cols if col in available_cols]
            
            if keep_cols:
                df = df[keep_cols]
                ds = xr.Dataset.from_dataframe(df)
                if 'date' in ds.dims:
                    ds = ds.rename({'date': 'time'})
                ds = ds.expand_dims(basin=[basin])
                basin_datasets.append(ds)
            else:
                LOGGER.warning(f"No requested variables found for basin {basin}")
        
        if not basin_datasets:
            LOGGER.warning("No historical data loaded for any basin")
            return None
        
        LOGGER.info("Merging basin datasets...")
        historical_ds = xr.concat(basin_datasets, dim='basin')
        return historical_ds
            
    def _compute_forecast_quartiles_as_variables(self,
                                                 forecast_ds: xr.Dataset,
                                                 quartiles: Tuple[float, ...] = (0.25, 0.5, 0.75),
                                                 suffix_base: str = "") -> xr.Dataset:
        """Compute requested ensemble quartiles as standalone forecast variables."""
        LOGGER.info(f"Computing quartiles {quartiles} for {list(forecast_ds.data_vars)}...")
        
        quartile_suffixes = {
            0.25: '_q25',
            0.5: '_q50', 
            0.75: '_q75'
        }
        
        new_data_vars = {}
        for var_name in forecast_ds.data_vars:
            var_data = forecast_ds[var_name]
            
            # If ensemble_member dimension exists, compute quartiles
            if 'ensemble_member' in var_data.dims:
                var_quartiles = var_data.quantile(quartiles, dim='ensemble_member')
                for i, q in enumerate(quartiles):
                    suffix = quartile_suffixes.get(q, f'_q{int(q*100)}')
                    new_var_name = f"{var_name}{suffix_base}{suffix}"
                    quartile_data = var_quartiles.isel(quantile=i).drop('quantile')
                    new_data_vars[new_var_name] = quartile_data
            else:
                # Deterministic variable, just rename with suffix if needed
                new_var_name = f"{var_name}{suffix_base}"
                new_data_vars[new_var_name] = var_data
        
        coords_to_keep = {k: v for k, v in forecast_ds.coords.items() if 'ensemble_member' not in v.dims}
        
        quartile_ds = xr.Dataset(
            data_vars=new_data_vars,
            coords=coords_to_keep,
            attrs=forecast_ds.attrs.copy()
        )
        return quartile_ds
        
    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        basin, indices = self.lookup_table[item]

        sample = {}
        for freq, seq_len, forecast_seq_len, pointer in zip(self.frequencies, self.seq_len, self._forecast_seq_len, indices):
            freq_suffix = '' if len(self.frequencies) == 1 else f'_{freq}'
            hindcast_idx = pointer['hindcast_idx']
            forecast_idx = pointer['forecast_idx']
            issue_time = pointer.get('issue_time')

            hindcast_history = seq_len - forecast_seq_len
            hindcast_start_idx = hindcast_idx + self._forecast_offset - hindcast_history + 1
            hindcast_end_idx = hindcast_idx + self._forecast_offset + 1
            global_end_idx = hindcast_idx + self._forecast_offset + forecast_seq_len + 1

            x_h = self._x_h[basin][freq][hindcast_start_idx:hindcast_end_idx]
            sample[f'x_h{freq_suffix}'] = torch.from_numpy(x_h)
            
            x_f = self._x_f[basin][freq][forecast_idx]
            sample[f'x_f{freq_suffix}'] = torch.from_numpy(x_f)
            
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
        
        available_basins = set(xr_hcst['basin'].values.tolist())
        requested_basins = set(self.basins)
        basin_coordinates = list(available_basins.intersection(requested_basins))
        
        if not basin_coordinates:
             raise ValueError(f"None of the requested basins {self.basins} found in the dataset.")

        for basin in tqdm(basin_coordinates, file=sys.stdout, disable=self._disable_pbar):
            basin_hcst = xr_hcst.sel(basin=basin, drop=True)
            
            start_dates = self.start_and_end_dates.get(basin, {}).get('start_dates', [])
            end_dates = self.start_and_end_dates.get(basin, {}).get('end_dates', [])
            
            basin_fcst = xr_fcst.sel(basin=basin, drop=True)
            
            if start_dates and end_dates:
                min_time = pd.to_datetime(min(start_dates))
                max_time = pd.to_datetime(max(end_dates)) + pd.Timedelta(days=1)
                basin_fcst = basin_fcst.sel({issue_dim: slice(min_time, max_time)})
            
            basin_fcst = basin_fcst.load()

            if time_dim not in basin_hcst.dims:
                LOGGER.warning("Time dimension not found for basin %s - skipping.", basin)
                basins_without_samples.append(basin)
                continue

            if issue_dim not in basin_fcst.dims:
                LOGGER.warning("Issue-time dimension not found for basin %s - skipping.", basin)
                basins_without_samples.append(basin)
                continue

            # Drop 'basin' coordinate if it exists as a scalar after selection
            if 'basin' in basin_hcst.coords and basin_hcst['basin'].dims == ():
                basin_hcst = basin_hcst.drop_vars('basin')
            
            hindcast_df = basin_hcst.to_dataframe().reset_index().set_index(time_dim).sort_index()
            # Remove any duplicate columns that may arise from coordinate expansion
            hindcast_df = hindcast_df.loc[:, ~hindcast_df.columns.duplicated()]
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
            
            # Drop 'basin' coordinate if it exists as a scalar after selection
            if 'basin' in basin_fcst.coords and basin_fcst['basin'].dims == ():
                basin_fcst = basin_fcst.drop_vars('basin')

            forecast_df = basin_fcst.to_dataframe().reset_index().set_index([issue_dim, 'lead_time']).sort_index()
            # Remove any duplicate columns that may arise from coordinate expansion
            forecast_df = forecast_df.loc[:, ~forecast_df.columns.duplicated()]
            if forecast_df.empty:
                LOGGER.warning("Forecast data empty for basin %s - skipping.", basin)
                basins_without_samples.append(basin)
                continue

            issue_times = forecast_df.index.get_level_values(issue_dim).unique()
            if issue_times.empty:
                LOGGER.warning("No forecast issue times available for basin %s - skipping.", basin)
                basins_without_samples.append(basin)
                continue

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
            if 'lead_time' in fc_inputs.dims:
                fc_inputs = fc_inputs.bfill(dim='lead_time')
            
            fc_tensor_list = []
            for var in available_forecast:
                da = fc_inputs[var]
                if 'lead_time' in da.dims:
                    da = da.transpose(issue_dim, 'lead_time', ...)
                    arr = da.values
                    # Squeeze out any extra dimensions (e.g., gauge_id) to get 2D array
                    while arr.ndim > 2:
                        arr = arr.squeeze(axis=-1)
                    fc_tensor_list.append(arr)
                else:
                    da = da.transpose(issue_dim, ...)
                    arr = da.values
                    # Squeeze out any extra dimensions to get 1D array, then add lead_time dim
                    while arr.ndim > 1:
                        arr = arr.squeeze(axis=-1)
                    fc_tensor_list.append(arr[:, np.newaxis])
            
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
            valid_indices = np.where(combined_validity)[0]

            if valid_indices.size == 0:
                basins_without_samples.append(basin)
                continue

            for freq in self.frequencies:
                cache_dir = self.cfg.train_dir / "binary_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                h_file = cache_dir / f"{basin}_{freq}_{self.period}_x_h.npy"
                f_file = cache_dir / f"{basin}_{freq}_{self.period}_x_f.npy"
                y_file = cache_dir / f"{basin}_{freq}_{self.period}_y.npy"
                
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
        """Validate that configured time periods are within available data ranges."""
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
        
        periods_to_check = []
        if hasattr(cfg, 'train_start_date') and hasattr(cfg, 'train_end_date'):
            if cfg.train_start_date and cfg.train_end_date:
                train_start = pd.to_datetime(cfg.train_start_date, format='%d/%m/%Y')
                train_end = pd.to_datetime(cfg.train_end_date, format='%d/%m/%Y')
                periods_to_check.append(('train', train_start, train_end))
        
        if hasattr(cfg, 'validation_start_date') and hasattr(cfg, 'validation_end_date'):
            if cfg.validation_start_date and cfg.validation_end_date:
                val_start = pd.to_datetime(cfg.validation_start_date, format='%d/%m/%Y')
                val_end = pd.to_datetime(cfg.validation_end_date, format='%d/%m/%Y')
                periods_to_check.append(('validation', val_start, val_end))
        
        if hasattr(cfg, 'test_start_date') and hasattr(cfg, 'test_end_date'):
            if cfg.test_start_date and cfg.test_end_date:
                test_start = pd.to_datetime(cfg.test_start_date, format='%d/%m/%Y')
                test_end = pd.to_datetime(cfg.test_end_date, format='%d/%m/%Y')
                periods_to_check.append(('test', test_start, test_end))
        
        errors = []
        warnings = []
        
        for period_name, period_start, period_end in periods_to_check:
            LOGGER.info(f"  Checking {period_name} period: {period_start.date()} to {period_end.date()}")
            needs_forecast_data = forecast_start is not None and period_end >= forecast_start
            needs_historical_data = historical_end is not None and period_start <= historical_end
            
            if needs_forecast_data and period_start < forecast_start:
                gap_days = (forecast_start - period_start).days
                errors.append(f"{period_name.capitalize()} period starts {gap_days} days before forecast data available.")
            
            if needs_historical_data and period_end > historical_end:
                gap_days = (period_end - historical_end).days
                errors.append(f"{period_name.capitalize()} period extends {gap_days} days beyond historical data.")
            
            if (forecast_start is not None and period_end < forecast_start) and \
               (historical_end is not None and period_start > historical_end):
                errors.append(f"{period_name.capitalize()} period falls entirely outside available data ranges.")
            
            if forecast_start is not None and period_end < forecast_start and needs_historical_data:
                gap_days = (forecast_start - period_end).days
                errors.append(f"{period_name.capitalize()} period ends {gap_days} days before forecast data available.")
            
            if historical_end is not None and period_start > historical_end:
                gap_days = (period_start - historical_end).days
                errors.append(f"{period_name.capitalize()} period starts {gap_days} days after historical data ends.")
        
        for warning in warnings:
            LOGGER.warning(f"  {warning}")
        
        if errors:
            error_msg = "Data availability validation failed:\n" + "\n".join([f"{i}. {e}" for i, e in enumerate(errors, 1)])
            raise ValueError(error_msg)
        
        LOGGER.info("Data availability validation passed")
