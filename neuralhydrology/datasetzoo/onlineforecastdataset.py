from pathlib import Path
import logging
import pickle
import shutil
import sys
import time
import warnings
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
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import NoEvaluationDataError, NoTrainDataError


LOGGER = logging.getLogger(__name__)


@dataclass
class DataAvailability:
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
        self._availability = DataAvailability()

        scaler = self._ensure_scaler(period=period, scaler=scaler, cfg=cfg)

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
            self._update_data_availability(merged_ds=cached_dataset)
            self._validate_data_availability(self.cfg)
            return cached_dataset

        if train_data_file is None or not self.is_train:
            dataset = self._build_dataset_from_sources()
            if self.is_train and self.cfg.save_train_data:
                self._save_dataset_cache(dataset)
            self._validate_data_availability(self.cfg)
            return dataset

        dataset = self._load_pickled_dataset(Path(train_data_file))
        if not self.frequencies:
            native_frequency = utils.infer_frequency(dataset["time"].values)
            self.frequencies = [native_frequency]
        self._update_data_availability(merged_ds=dataset)
        self._validate_data_availability(self.cfg)
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

        if 'init_time' in forecast_ds.dims and 'issue_time' not in forecast_ds.dims:
            forecast_ds = forecast_ds.rename({'init_time': 'issue_time'})
        if 'time' in forecast_ds.dims and 'issue_time' not in forecast_ds.dims:
            forecast_ds = forecast_ds.rename({'time': 'issue_time'})

        if 'issue_time' not in forecast_ds.dims:
            raise ValueError('Forecast dataset must expose an issue_time dimension.')
        if 'time' not in historical_ds.dims:
            raise ValueError('Historical dataset must expose a time dimension.')

        forecast_ds = forecast_ds.sortby('issue_time')
        historical_ds = historical_ds.sortby('time')

        issue_times = forecast_ds['issue_time'].values
        if issue_times.size == 0:
            raise ValueError('Forecast dataset does not contain any issue times.')
        hist_times = historical_ds['time'].values
        if hist_times.size == 0:
            raise ValueError('Historical dataset does not contain any timestamps.')

        self._update_data_availability(historical_ds=historical_ds, forecast_ds=forecast_ds)
        availability = self._availability
        forecast_start = availability.forecast_start
        forecast_end = availability.forecast_end
        historical_end = availability.historical_end

        if not self.frequencies:
            inferred_freq = utils.infer_frequency(historical_ds['time'].values)
            if inferred_freq is None:
                raise ValueError('Could not infer native frequency from historical dataset.')
            self.frequencies = [inferred_freq]

        reference_ts = pd.Timestamp('2000-01-01')  # Arbitrary anchor to convert DateOffsets to timedeltas
        warmup_offsets = []
        for i, freq in enumerate(self.frequencies):
            forecast_horizon = max(self._predict_last_n[i], self._forecast_seq_len[i])
            offset = (self.seq_len[i] - forecast_horizon) * to_offset(freq)
            warmup_offsets.append(reference_ts + offset - reference_ts)
        max_warmup = max(warmup_offsets) if warmup_offsets else pd.Timedelta(0)

        hist_start_limit = None
        if forecast_start is not None:
            hist_start_limit = forecast_start - max_warmup
        hist_end_limit = historical_end
        issue_start_limit = forecast_start
        issue_end_limit = forecast_end

        retry_attempts, retry_delay = self._get_retry_config()
        basin_datasets: List[xr.Dataset] = []
        basins_without_data: List[str] = []
        global_hist_start: Optional[pd.Timestamp] = None
        global_hist_end: Optional[pd.Timestamp] = None
        global_issue_start: Optional[pd.Timestamp] = None
        global_issue_end: Optional[pd.Timestamp] = None

        for basin in self.basins:
            if 'basin' not in historical_ds.coords or basin not in historical_ds['basin'].values:
                LOGGER.warning("Historical data not available for basin %s - skipping.", basin)
                basins_without_data.append(basin)
                continue
            if 'basin' not in forecast_ds.coords or basin not in forecast_ds['basin'].values:
                LOGGER.warning("Forecast data not available for basin %s - skipping.", basin)
                basins_without_data.append(basin)
                continue

            hist_basin = historical_ds.sel(basin=basin, drop=False).sortby('time')
            if hist_start_limit is not None or hist_end_limit is not None:
                hist_basin = hist_basin.sel(time=slice(hist_start_limit, hist_end_limit))
            if hist_basin['time'].size == 0:
                LOGGER.warning("Historical slice empty for basin %s - skipping.", basin)
                basins_without_data.append(basin)
                continue
            hist_basin = hist_basin.load()

            fcst_basin = self._materialize_forecast_slice(
                fcst_basin=forecast_ds.sel(basin=basin, drop=False),
                basin=basin,
                start_date=issue_start_limit,
                end_date=issue_end_limit,
                attempts=retry_attempts,
                delay=retry_delay,
            )
            if fcst_basin['issue_time'].size == 0:
                LOGGER.warning("Forecast slice empty for basin %s - skipping.", basin)
                basins_without_data.append(basin)
                continue

            hist_basin = hist_basin.sortby('time')
            fcst_basin = fcst_basin.sortby('issue_time')

            hist_index = pd.Index(hist_basin['time'].values)
            if hist_index.duplicated().any():
                hist_basin = hist_basin.isel(time=~hist_index.duplicated(keep='last'))

            fcst_index = pd.Index(fcst_basin['issue_time'].values)
            if fcst_index.duplicated().any():
                fcst_basin = fcst_basin.isel(issue_time=~fcst_index.duplicated(keep='last'))

            basin_datasets.append(xr.merge([hist_basin, fcst_basin], compat='override'))

            basin_hist_start = pd.to_datetime(hist_basin['time'].values[0])
            basin_hist_end = pd.to_datetime(hist_basin['time'].values[-1])
            basin_issue_start = pd.to_datetime(fcst_basin['issue_time'].values[0])
            basin_issue_end = pd.to_datetime(fcst_basin['issue_time'].values[-1])

            if global_hist_start is None or basin_hist_start < global_hist_start:
                global_hist_start = basin_hist_start
            if global_hist_end is None or basin_hist_end > global_hist_end:
                global_hist_end = basin_hist_end
            if global_issue_start is None or basin_issue_start < global_issue_start:
                global_issue_start = basin_issue_start
            if global_issue_end is None or basin_issue_end > global_issue_end:
                global_issue_end = basin_issue_end

        if basins_without_data:
            LOGGER.info("Skipped basins without complete data: %s", sorted(set(basins_without_data)))

        if not basin_datasets:
            if self.is_train:
                raise NoTrainDataError
            raise NoEvaluationDataError

        merged = xr.concat(basin_datasets, dim='basin')
        merged.attrs['onlineforecast_cache_version'] = self.CACHE_VERSION
        availability = self._availability
        merged.attrs.update(availability.to_attrs())
        if global_hist_start is not None:
            merged.attrs['cache_hist_start'] = str(global_hist_start)
            merged.attrs['cache_hist_end'] = str(global_hist_end)
        if global_issue_start is not None:
            merged.attrs['cache_issue_start'] = str(global_issue_start)
            merged.attrs['cache_issue_end'] = str(global_issue_end)

        return merged

    def _load_cached_dataset(self, train_data_file: Optional[Path]) -> Optional[xr.Dataset]:
        for cache_path in self._cache_candidates(train_data_file):
            if not cache_path.exists():
                continue

            LOGGER.info("Loading cached dataset from %s", cache_path)
            dataset = xr.open_zarr(store=cache_path, decode_timedelta=True)
            cache_version = dataset.attrs.get('onlineforecast_cache_version')
            if cache_version != self.CACHE_VERSION:
                LOGGER.info("Cached dataset version %s does not match expected %s. Rebuilding cache.",
                            cache_version, self.CACHE_VERSION)
                try:
                    dataset.close()
                except AttributeError:
                    pass
                if cache_path.is_dir():
                    shutil.rmtree(cache_path)
                else:
                    cache_path.unlink()
                continue
            if not self.frequencies:
                native_frequency = utils.infer_frequency(dataset["time"].values)
                self.frequencies = [native_frequency]
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
        if dataset.attrs.get('onlineforecast_cache_version') != self.CACHE_VERSION:
            dataset.attrs['onlineforecast_cache_version'] = self.CACHE_VERSION
            dataset.attrs.update(self._availability.to_attrs())
        dataset.chunk({'basin': max(1, len(self.basins))}).to_zarr(store=cache_path, mode='w')

    def _cache_candidates(self, train_data_file: Optional[Path]) -> List[Path]:
        train_dir = getattr(self.cfg, 'train_dir', None)
        if train_dir is None:
            return []

        train_dir_path = Path(train_dir)
        return [train_dir_path / 'train_data.zarr']

    def _load_pickled_dataset(self, train_data_file: Path) -> xr.Dataset:
        with train_data_file.open("rb") as fp:
            dataset_dict = pickle.load(fp)
        return xr.Dataset.from_dict(dataset_dict)

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

    def _get_retry_config(self) -> Tuple[int, float]:
        attempts = getattr(self.cfg, 'forecast_load_retries', 5) or 1
        delay = getattr(self.cfg, 'forecast_load_retry_delay', 5) or 0
        return max(1, int(attempts)), max(0.0, float(delay))

    def _materialize_forecast_slice(self,
                                    fcst_basin: xr.Dataset,
                                    basin: str,
                                    start_date: Optional[pd.Timestamp],
                                    end_date: Optional[pd.Timestamp],
                                    attempts: int,
                                    delay: float) -> xr.Dataset:
        last_exception: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                issue_dim = 'issue_time' if 'issue_time' in fcst_basin.dims else (
                    'time' if 'time' in fcst_basin.dims else None)
                if issue_dim is None:
                    raise ValueError('Forecast dataset must include an issue_time dimension.')
                slice_kwargs = {issue_dim: slice(start_date, end_date)}
                fcst_slice = fcst_basin.sel(slice_kwargs)
                if issue_dim != 'issue_time':
                    fcst_slice = fcst_slice.rename({issue_dim: 'issue_time'})
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
        for freq, seq_len, forecast_seq_len, pointer in zip(self.frequencies, self.seq_len, self._forecast_seq_len, indices):
            # if there's just one frequency, don't use suffixes.
            freq_suffix = '' if len(self.frequencies) == 1 else f'_{freq}'
            hindcast_idx = pointer['hindcast_idx']
            forecast_idx = pointer['forecast_idx']
            issue_time = pointer.get('issue_time')

            hindcast_history = seq_len - forecast_seq_len
            hindcast_start_idx = hindcast_idx + self._forecast_offset - hindcast_history
            hindcast_end_idx = hindcast_idx + self._forecast_offset
            global_end_idx = hindcast_idx + self._forecast_offset + forecast_seq_len

            sample[f'x_h{freq_suffix}'] = self._x_h[basin][freq][hindcast_start_idx:hindcast_end_idx]
            sample[f'x_f{freq_suffix}'] = self._x_f[basin][freq][forecast_idx]
            sample[f'y{freq_suffix}'] = self._y[basin][freq][hindcast_start_idx:global_end_idx]
            sample[f'date{freq_suffix}'] = self._dates[basin][freq][hindcast_start_idx:global_end_idx]
            if issue_time is not None:
                sample[f'issue_time{freq_suffix}'] = issue_time

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
        basin_coordinates = xr_hcst['basin'].values.tolist()

        for basin in tqdm(basin_coordinates, file=sys.stdout, disable=self._disable_pbar):
            basin_hcst = xr_hcst.sel(basin=basin, drop=True)
            basin_fcst = xr_fcst.sel(basin=basin, drop=True)

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

            # Explicitly load into memory to avoid dask/to_dataframe issues and get better error messages
            basin_fcst = basin_fcst.load()

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

            available_hindcast = [col for col in self.cfg.hindcast_inputs if col in hindcast_df.columns]
            if not available_hindcast:
                raise ValueError(f'No hindcast inputs available for basin {basin}. Check cfg.hindcast_inputs.')

            available_targets = [col for col in self.cfg.target_variables if col in hindcast_df.columns]
            if not available_targets:
                raise ValueError(f'No target variables available for basin {basin}. Check cfg.target_variables.')

            available_forecast = [col for col in self.cfg.forecast_inputs if col in forecast_df.columns]
            if not available_forecast:
                raise ValueError(f'No forecast inputs available for basin {basin}. Check cfg.forecast_inputs.')

            hindcast_matrix = hindcast_df[available_hindcast].to_numpy(dtype=np.float32)
            target_matrix = hindcast_df[available_targets].to_numpy(dtype=np.float32)
            date_values = hindcast_df.index.to_numpy()

            fc_inputs = basin_fcst[available_forecast]
            # NOAA GEFS precipitation rates are averaged since the previous step, so the first
            # lead-time in each issue can be NaN after interpolation; fill forward to retain samples.
            if 'lead_time' in fc_inputs.dims:
                fc_inputs = fc_inputs.bfill(dim='lead_time')
            fc_tensor_da = fc_inputs.to_array('variable')
            if 'lead_time' in fc_tensor_da.dims:
                fc_tensor = fc_tensor_da.transpose(issue_dim, 'lead_time', 'variable').values.astype(np.float32)
            else:
                fc_tensor = fc_tensor_da.transpose(issue_dim, 'variable').values.astype(np.float32)[:, np.newaxis, :]

            if fc_tensor.shape[1] < max(self._forecast_seq_len):
                LOGGER.warning("Forecast tensor shorter than forecast_seq_length for basin %s - skipping.", basin)
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

                    hindcast_start = anchor_idx + self._forecast_offset - hindcast_history
                    hindcast_end = anchor_idx + self._forecast_offset
                    forecast_end = anchor_idx + self._forecast_offset + self._forecast_seq_len[freq_idx]

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
                self._x_h[basin][freq] = torch.from_numpy(hindcast_matrix)
                self._x_f[basin][freq] = torch.from_numpy(fc_tensor)
                self._y[basin][freq] = torch.from_numpy(target_matrix)
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
    
    @staticmethod
    def collate_fn(
            samples: List[Dict[str, Union[torch.Tensor, np.ndarray, Dict[str, torch.Tensor]]]]
    ) -> Dict[str, Union[torch.Tensor, np.ndarray, Dict[str, torch.Tensor]]]:
        batch = {}
        if not samples:
            return batch
        features = list(samples[0].keys())
        for feature in features:
            if feature.startswith('date') or feature.startswith('issue_time'):
                # Dates and issue times are stored as a numpy array of datetime64, which we maintain as numpy array.
                batch[feature] = np.stack([sample[feature] for sample in samples], axis=0)
            elif feature.startswith('x_d'):
                # Dynamics are stored as dictionaries with feature names as keys.
                batch[feature] = {k: torch.stack([sample[feature][k] for sample in samples], dim=0)
                                  for k in samples[0][feature]}
            else:
                # Everything else is a torch.Tensor.
                batch[feature] = torch.stack([sample[feature] for sample in samples], dim=0)
        return batch
