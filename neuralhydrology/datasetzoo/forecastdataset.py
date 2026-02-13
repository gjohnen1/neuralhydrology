"""Unified forecast dataset with pluggable forecast loaders.

This module provides a unified ForecastDataset class that replaces:
- OnlineForecastDataset (GEFS forecasts)
- CombinedForecastDataset (GEFS + ICON-D2)
- PerfectForecastDataset (perfect prognosis)

The new architecture uses pluggable forecast loaders that can be configured
via config files, making it easy to add new forecast sources without code changes.
"""

from functools import reduce
import importlib.util
from pathlib import Path
import hashlib
import logging
import pickle
import shutil
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import xarray as xr
from ruamel.yaml import YAML

from neuralhydrology.datasetzoo.genericdataset import GenericDataset
from neuralhydrology.datasetzoo.forecast_loaders import (
    ForecastLoader,
    ForecastLoaderConfig,
    ForecastLoaderRegistry,
)
from neuralhydrology.datasetzoo.forecast_utils import DataAvailability, QuartileComputer
from neuralhydrology.datautils import utils
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import NoEvaluationDataError, NoTrainDataError


LOGGER = logging.getLogger(__name__)


class ForecastDataset(GenericDataset):
    """Unified forecast dataset with pluggable loaders.

    This dataset class handles operational forecast data with flexible loader architecture.
    It supports:
    - Multiple forecast sources (GEFS, ICON-D2, perfect forecasts, etc.)
    - Configurable via YAML (new style) or auto-detection from legacy configs
    - Memory-efficient caching with zarr
    - Multi-source merging with padding and availability masks

    The dataset handles both:
    - Hindcast variables: indexed by (basin, time) - historical observations
    - Forecast variables: indexed by (basin, issue_time, lead_time) - forecast data

    Parameters
    ----------
    cfg : Config
        Run configuration.
    is_train : bool
        Whether this is a training dataset.
    period : str
        One of 'train', 'validation', or 'test'.
    basin : str, optional
        Single basin to load (if None, loads from basin file).
    additional_features : List[Dict[str, pd.DataFrame]], optional
        Additional feature data per basin.
    id_to_int : Dict[str, int], optional
        Basin ID to integer mapping for one-hot encoding.
    scaler : Dict[str, Union[pd.Series, xr.DataArray]], optional
        Feature scaling parameters (required for validation/test).

    Attributes
    ----------
    _loaders : List[ForecastLoader]
        Initialized forecast loaders for this dataset.
    _availability : DataAvailability
        Tracks temporal extent of historical and forecast data.
    CACHE_VERSION : str
        Cache version string for invalidation.
    """

    CACHE_VERSION = "unified-v2"

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xr.DataArray]] = {}):

        # Initialize internal state
        self._x_h: Dict[str, Dict[str, torch.Tensor]] = {}
        self._x_f: Dict[str, Dict[str, torch.Tensor]] = {}
        self._x_s: Dict[str, Dict[str, torch.Tensor]] = {}
        self._y: Dict[str, Dict[str, torch.Tensor]] = {}
        self._dates: Dict[str, Dict[str, np.ndarray]] = {}
        self._issue_times: Dict[str, Dict[str, np.ndarray]] = {}
        self._availability = DataAvailability()
        self.period_starts: Dict[str, pd.Timestamp] = {}
        self._zarr_available = importlib.util.find_spec("zarr") is not None
        if not self._zarr_available:
            LOGGER.warning(
                "Package 'zarr' is not available. Unified/legacy zarr cache read/write is disabled for this run."
            )

        # Forecast-specific parameters (must be set before super().__init__ triggers _load_data)
        forecast_seq = cfg.forecast_seq_length
        self._forecast_seq_len = [forecast_seq] if isinstance(forecast_seq, int) else list(forecast_seq)
        self._forecast_offset = getattr(cfg, 'forecast_offset', 0)
        self._disable_pbar = getattr(cfg, 'verbose', 1) == 0

        # Initialize forecast loaders
        self._loaders = self._initialize_loaders(cfg)

        # Ensure scaler is available
        scaler = self._ensure_scaler(period=period, scaler=scaler, cfg=cfg)

        # Call parent constructor
        super().__init__(cfg=cfg,
                        is_train=is_train,
                        period=period,
                        basin=basin,
                        additional_features=additional_features,
                        id_to_int=id_to_int,
                        scaler=scaler)

    def _initialize_loaders(self, cfg: Config) -> List[ForecastLoader]:
        """Initialize forecast loaders from configuration.

        Supports both new-style configuration (forecast_sources) and legacy
        auto-detection from dataset type.

        Parameters
        ----------
        cfg : Config
            Run configuration.

        Returns
        -------
        List[ForecastLoader]
            Initialized loader instances.

        Raises
        ------
        ValueError
            If no loaders can be configured.
        """
        # Check for new-style configuration
        if hasattr(cfg, 'forecast_sources') and cfg.forecast_sources:
            loader_configs = self._parse_forecast_sources(cfg)
        else:
            # Fall back to legacy auto-detection
            LOGGER.info("No 'forecast_sources' in config - attempting legacy config detection")
            loader_configs = self._detect_legacy_config(cfg)

        if not loader_configs:
            raise ValueError(
                "No forecast loaders configured. Either specify 'forecast_sources' in config "
                "or use a legacy dataset name ('online_forecast', 'combined_forecast', 'perfect_forecast')."
            )

        # Instantiate loaders
        loaders = []
        for config in loader_configs:
            if not config.enabled:
                LOGGER.info(f"Skipping disabled loader: {config.name}")
                continue

            try:
                loader_class = ForecastLoaderRegistry.get_loader(config.type)
                loader = loader_class(config, cfg)
                loaders.append(loader)
                LOGGER.info(
                    f"Initialized {config.type} loader: '{config.name}' "
                    f"(suffix: '{config.suffix}', variables: {len(config.variables)})"
                )
            except Exception as e:
                LOGGER.error(f"Failed to initialize loader '{config.name}': {e}")
                raise

        if not loaders:
            raise ValueError("No enabled forecast loaders after initialization")

        return loaders

    def _detect_legacy_config(self, cfg: Config) -> List[ForecastLoaderConfig]:
        """Auto-detect forecast sources from legacy dataset configuration.

        This provides backward compatibility with existing configs that use:
        - dataset: online_forecast (GEFS only)
        - dataset: combined_forecast (GEFS + ICON-D2)
        - dataset: perfect_forecast (perfect prognosis)

        Parameters
        ----------
        cfg : Config
            Run configuration.

        Returns
        -------
        List[ForecastLoaderConfig]
            Detected loader configurations.
        """
        LOGGER.info(f"Detecting legacy configuration from dataset type: {cfg.dataset}")

        if cfg.dataset == 'online_forecast':
            # GEFS only, no suffix (backward compatible)
            variables = self._extract_base_variables(cfg.forecast_inputs, suffix='')
            return [ForecastLoaderConfig(
                name='gefs',
                type='gefs',
                suffix='',  # No suffix for backward compatibility
                variables=variables,
                quartiles=[0.25, 0.5, 0.75],
                enabled=True,
                loader_kwargs={'max_hours': max(cfg.forecast_seq_length) if isinstance(cfg.forecast_seq_length, list) else cfg.forecast_seq_length}
            )]

        elif cfg.dataset == 'combined_forecast':
            # GEFS + ICON-D2 with suffixes
            loaders = []

            # Detect GEFS variables
            gefs_vars = [v for v in cfg.forecast_inputs if '_gefs' in v]
            if gefs_vars:
                variables = self._extract_base_variables(gefs_vars, suffix='_gefs')
                loaders.append(ForecastLoaderConfig(
                    name='gefs',
                    type='gefs',
                    suffix='_gefs',
                    variables=variables,
                    quartiles=[0.25, 0.5, 0.75],
                    enabled=True,
                    loader_kwargs={'max_hours': max(cfg.forecast_seq_length) if isinstance(cfg.forecast_seq_length, list) else cfg.forecast_seq_length}
                ))

            # Detect ICON-D2 variables
            icond2_vars = [v for v in cfg.forecast_inputs if '_icond2' in v]
            if icond2_vars:
                variables = self._extract_base_variables(icond2_vars, suffix='_icond2')
                loaders.append(ForecastLoaderConfig(
                    name='icond2',
                    type='icond2',
                    suffix='_icond2',
                    variables=variables,
                    quartiles=[0.25, 0.5, 0.75],
                    enabled=True,
                    loader_kwargs={'data_dir': 'icond2', 'horizon_hours': 48}
                ))

            return loaders

        elif cfg.dataset == 'perfect_forecast':
            # Perfect forecast (median only)
            variables = self._extract_base_variables(cfg.forecast_inputs, suffix='_perfect')
            # If no _perfect suffix found, try without suffix
            if not variables:
                variables = self._extract_base_variables(cfg.forecast_inputs, suffix='')

            return [ForecastLoaderConfig(
                name='perfect',
                type='perfect_forecast',
                suffix='_perfect' if any('_perfect' in v for v in cfg.forecast_inputs) else '',
                variables=variables,
                quartiles=[],  # No quartiles for perfect forecasts (deterministic)
                enabled=True,
                loader_kwargs={'max_horizon': max(cfg.forecast_seq_length) if isinstance(cfg.forecast_seq_length, list) else cfg.forecast_seq_length}
            )]

        else:
            LOGGER.warning(f"Unknown dataset type for legacy detection: {cfg.dataset}")
            return []

    def _extract_base_variables(self, forecast_inputs: List[str], suffix: str) -> List[str]:
        """Extract base variable names from forecast inputs.

        Parameters
        ----------
        forecast_inputs : List[str]
            Forecast input variable names from config.
        suffix : str
            Suffix to remove (e.g., '_gefs', '_icond2').

        Returns
        -------
        List[str]
            Base variable names without quartile or source suffixes.
        """
        base_vars = set()
        for var in forecast_inputs:
            # Remove suffix if present
            if suffix and var.endswith(suffix) or suffix in var:
                var = var.replace(suffix, '')

            # Remove quartile suffixes
            base_var = var.replace('_q25', '').replace('_q50', '').replace('_q75', '')

            if base_var:
                base_vars.add(base_var)

        return sorted(base_vars)

    def _parse_forecast_sources(self, cfg: Config) -> List[ForecastLoaderConfig]:
        """Parse new-style forecast_sources configuration.

        Parameters
        ----------
        cfg : Config
            Run configuration with forecast_sources section.

        Returns
        -------
        List[ForecastLoaderConfig]
            Parsed loader configurations.
        """
        configs = []

        for source in cfg.forecast_sources:
            config = ForecastLoaderConfig(
                name=source.get('name', 'unnamed'),
                type=source['type'],
                suffix=source.get('suffix', ''),
                variables=source.get('variables', []),
                quartiles=source.get('quartiles', [0.25, 0.5, 0.75]),
                enabled=source.get('enabled', True),
                loader_kwargs=source.get('loader_kwargs', {})
            )
            configs.append(config)

        return configs

    def _load_attributes(self) -> pd.DataFrame:
        """Load catchment attributes from *_attributes.csv files in data_dir."""
        attributes_path = self.cfg.data_dir

        if not attributes_path.exists():
            raise FileNotFoundError(f"Attribute folder not found at {attributes_path}")

        txt_files = list(attributes_path.glob('*_attributes.csv'))

        if not txt_files:
            return pd.DataFrame()

        # Read attributes into one dataframe
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
        """Initialize frequency configuration for temporal resolution."""
        self.seq_len = self.cfg.seq_length
        self._forecast_seq_len = self.cfg.forecast_seq_length
        self._predict_last_n = self.cfg.predict_last_n
        self._forecast_offset = self.cfg.forecast_offset

        # This dataset does not support multiple frequencies
        if self.cfg.use_frequencies:
            LOGGER.warning(
                'Multiple timestep frequencies are not supported by ForecastDataset: '
                'defaulting to native frequency of input data'
            )
        self.frequencies = []

        if not self.frequencies:
            if not isinstance(self.seq_len, int) or not isinstance(self._predict_last_n, int):
                raise ValueError('seq_length and predict_last_n must be integers')
            self.seq_len = [self.seq_len]
            self._forecast_seq_len = [self._forecast_seq_len]
            self._predict_last_n = [self._predict_last_n]

    def _get_legacy_cache_path(self, basin: str) -> Optional[Path]:
        """Check for existing legacy zarr caches.

        Parameters
        ----------
        basin : str
            Basin ID.

        Returns
        -------
        Optional[Path]
            Path to legacy cache if found, None otherwise.
        """
        if self.cfg.dataset == 'online_forecast':
            path = self.cfg.data_dir / "zarr_cache" / f"{basin}.zarr"
        elif self.cfg.dataset == 'combined_forecast':
            path = self.cfg.data_dir / "zarr_cache" / f"{basin}_combined.zarr"
        elif self.cfg.dataset == 'perfect_forecast':
            path = self.cfg.data_dir / "zarr_cache_perfect" / f"{basin}.zarr"
        elif self.cfg.dataset == 'forecast':
            # Unified forecast config can still reuse legacy caches from prior dataset classes.
            loader_types = {loader.config.type for loader in self._loaders}
            candidates = []

            # Perfect prognosis legacy cache
            if loader_types == {'perfect_forecast'}:
                candidates.append(self.cfg.data_dir / "zarr_cache_perfect" / f"{basin}.zarr")

            # Combined GEFS + ICON-D2 legacy cache
            if 'icond2' in loader_types:
                candidates.append(self.cfg.data_dir / "zarr_cache" / f"{basin}_combined.zarr")

            # GEFS-only legacy cache
            if 'gefs' in loader_types:
                candidates.append(self.cfg.data_dir / "zarr_cache" / f"{basin}.zarr")

            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return None
        else:
            return None
        return path if path.exists() else None

    def _validate_legacy_cache(self, ds: xr.Dataset) -> bool:
        """Validate a legacy zarr cache (skip version check, only check variables).

        Parameters
        ----------
        ds : xr.Dataset
            Cached dataset.

        Returns
        -------
        bool
            True if cache contains all required variables.
        """
        required_vars = set(
            self.cfg.hindcast_inputs +
            self.cfg.forecast_inputs +
            self.cfg.target_variables
        )
        available_vars = set(ds.data_vars)

        if not required_vars.issubset(available_vars):
            missing = required_vars - available_vars
            LOGGER.warning(f"Legacy cache missing variables: {missing}")
            return False

        return True

    @staticmethod
    def _clean_legacy_dims(ds: xr.Dataset) -> xr.Dataset:
        """Remove extra dimensions from legacy zarr caches.

        Uses squeeze instead of drop_dims to preserve variables
        that may use the extra dimension.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset that may have extra dims.

        Returns
        -------
        xr.Dataset
            Dataset with extra dims squeezed out.
        """
        if 'gauge_id' in ds.dims:
            ds = ds.squeeze('gauge_id', drop=True)
        return ds

    def _load_or_create_xarray_dataset(self) -> xr.Dataset:
        """Load or create xarray dataset with unified caching.

        Checks for unified caches first, then falls back to legacy zarr caches.

        Returns
        -------
        xr.Dataset
            Merged dataset for all basins with hindcast and forecast data.

        Raises
        ------
        NoTrainDataError, NoEvaluationDataError
            If no data could be loaded for any basin.
        """
        basin_datasets = []
        use_zarr_cache = self._zarr_available

        if use_zarr_cache:
            # Unified cache directory
            cache_dir = self.cfg.data_dir / "zarr_cache_unified"
            cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            cache_dir = None

        cache_key = self._create_cache_key()

        for basin in self.basins:
            cache_path = cache_dir / f"{basin}_{cache_key}.zarr" if use_zarr_cache else None

            if use_zarr_cache:
                # Try loading from unified cache
                if cache_path.exists():
                    LOGGER.info(f"Loading cached dataset for basin {basin} from {cache_path}")
                    try:
                        ds = xr.open_zarr(store=cache_path, decode_timedelta=True)
                        if self._validate_cache(ds):
                            basin_datasets.append(ds)
                            continue
                        else:
                            LOGGER.info(f"Cache validation failed for basin {basin}. Rebuilding.")
                            ds.close()
                            shutil.rmtree(cache_path)
                    except Exception as e:
                        LOGGER.warning(f"Failed to load cache for basin {basin}: {e}. Rebuilding.")
                        if cache_path.exists():
                            shutil.rmtree(cache_path)

                # Try loading from legacy cache (existing zarr from old dataset classes)
                legacy_path = self._get_legacy_cache_path(basin)
                if legacy_path is not None:
                    LOGGER.info(f"Loading legacy cache for basin {basin} from {legacy_path}")
                    try:
                        ds = xr.open_zarr(store=legacy_path, decode_timedelta=True)
                        ds = self._clean_legacy_dims(ds)
                        if self._validate_legacy_cache(ds):
                            basin_datasets.append(ds)
                            continue
                        else:
                            LOGGER.warning(f"Legacy cache validation failed for basin {basin}.")
                            ds.close()
                    except Exception as e:
                        LOGGER.warning(f"Failed to load legacy cache for basin {basin}: {e}")

            # Build and cache if neither unified nor legacy cache available
            ds = self._build_and_cache_basin_dataset(basin, cache_path)
            basin_datasets.append(ds)

        if not basin_datasets:
            if self.is_train:
                raise NoTrainDataError
            raise NoEvaluationDataError

        # Merge all basin datasets
        merged = xr.concat(basin_datasets, dim='basin')

        # Infer frequency if needed
        if not self.frequencies:
            inferred_freq = utils.infer_frequency(merged['time'].values)
            self.frequencies = [inferred_freq]
            LOGGER.info(f"Inferred frequency from dataset: {inferred_freq}")

        # Update and validate availability
        self._update_data_availability(merged_ds=merged)
        self._validate_data_availability(self.cfg)

        return merged

    def _build_and_cache_basin_dataset(self, basin: str, cache_path: Optional[Path]) -> xr.Dataset:
        """Build dataset for one basin from all loaders.

        Parameters
        ----------
        basin : str
            Basin ID.
        cache_path : Optional[Path]
            Path to save cached dataset. If None, no zarr cache will be written.

        Returns
        -------
        xr.Dataset
            Cached dataset for this basin.

        Raises
        ------
        ValueError
            If data loading fails.
        """
        LOGGER.info(f"Building dataset for basin {basin}...")

        # Load historical data (shared across all forecast types)
        historical_ds = self._load_historical_xarray_data(basins=[basin])
        if historical_ds is None:
            raise ValueError(f"Failed to load historical data for basin {basin}")

        # Load forecasts from all loaders
        forecast_datasets = []
        for loader in self._loaders:
            try:
                LOGGER.info(f"Loading {loader.config.name} forecasts for basin {basin}...")
                loader_ds = loader.load(basins=[basin])
                if loader_ds is not None:
                    forecast_datasets.append(loader_ds)
                    LOGGER.info(
                        f"Loaded {loader.config.name}: {len(loader_ds.data_vars)} variables, "
                        f"{loader.get_horizon_hours()}h horizon"
                    )
                else:
                    LOGGER.warning(f"Loader {loader.config.name} returned no data for basin {basin}")
            except Exception as e:
                LOGGER.error(f"Loader {loader.config.name} failed for basin {basin}: {e}")
                if self.is_train:
                    raise  # Fail fast during training

        if not forecast_datasets:
            raise ValueError(f"No forecast data loaded for basin {basin}")

        # Merge forecast sources
        forecast_ds = self._merge_forecast_sources(forecast_datasets)

        # Standardize dimensions
        if 'init_time' in forecast_ds.dims and 'issue_time' not in forecast_ds.dims:
            forecast_ds = forecast_ds.rename({'init_time': 'issue_time'})

        # Sort
        forecast_ds = forecast_ds.sortby('issue_time')
        historical_ds = historical_ds.sortby('time')

        # Determine time slice range
        hist_start = pd.to_datetime(historical_ds['time'].values[0])
        hist_end = pd.to_datetime(historical_ds['time'].values[-1])
        fcst_start = pd.to_datetime(forecast_ds['issue_time'].values[0])
        fcst_end = pd.to_datetime(forecast_ds['issue_time'].values[-1])

        slice_start = fcst_start
        slice_end = hist_end

        # Slice forecast dataset
        forecast_ds = forecast_ds.sel(issue_time=slice(slice_start, slice_end))

        # Calculate warmup period needed for historical data
        max_warmup = self._calculate_warmup_period(historical_ds)
        hist_start_needed = slice_start - max_warmup

        # Slice historical dataset
        historical_ds = historical_ds.sel(time=slice(hist_start_needed, slice_end))

        # Merge historical and forecast
        merged = xr.merge([historical_ds, forecast_ds], compat='override')

        # Add metadata
        merged.attrs['forecast_cache_version'] = self.CACHE_VERSION
        merged.attrs['basin'] = basin
        merged.attrs['loaders'] = [l.config.name for l in self._loaders]
        merged.attrs.update(self._availability.to_attrs())

        # Compute and save
        LOGGER.info(f"Computing dataset for {basin}...")
        merged = merged.compute()

        if (cache_path is None) or (not self._zarr_available):
            return merged

        LOGGER.info(f"Saving cache: {cache_path}")
        # Ensure basin is string type for zarr compatibility
        if 'basin' in merged.coords:
            merged['basin'] = merged['basin'].astype(str)

        merged.to_zarr(store=cache_path, mode='w')
        merged.close()

        # Reload from cache
        ds_cached = xr.open_zarr(store=cache_path, decode_timedelta=True)

        return ds_cached

    def _merge_forecast_sources(self, datasets: List[xr.Dataset]) -> xr.Dataset:
        """Merge multiple forecast sources with alignment and padding.

        Parameters
        ----------
        datasets : List[xr.Dataset]
            List of forecast datasets from different loaders.

        Returns
        -------
        xr.Dataset
            Merged dataset with aligned time ranges and padded lead times.

        Raises
        ------
        ValueError
            If datasets have no overlapping time period.
        """
        required_lead_time = max(self._forecast_seq_len) if self._forecast_seq_len else 0

        if len(datasets) == 1:
            ds = datasets[0]
            loader = self._loaders[0]

            if 'lead_time' not in ds.dims:
                return ds

            max_lead_time = int(ds.lead_time.values.max())
            target_lead_time = max(max_lead_time, required_lead_time)
            if target_lead_time <= max_lead_time and not self._should_add_availability_mask(loader.config.name):
                return ds

            LOGGER.info(
                "Single forecast source '%s': padding lead_time from %dh to %dh",
                loader.config.name,
                max_lead_time,
                target_lead_time,
            )

            lead_time_range = np.arange(1, target_lead_time + 1)
            valid_leads = set(int(lt) for lt in ds.lead_time.values)
            ds_padded = ds.reindex(lead_time=lead_time_range, fill_value=np.nan)

            if self._should_add_availability_mask(loader.config.name):
                mask_name = f"{loader.config.name}_available"
                template_var = list(ds_padded.data_vars)[0]
                mask = xr.zeros_like(ds_padded[template_var])
                for lead in lead_time_range:
                    if lead in valid_leads:
                        mask.loc[{'lead_time': lead}] = 1.0
                mask.name = mask_name
                ds_padded = xr.merge([ds_padded, mask])
                LOGGER.info("Added availability mask: %s", mask_name)

            return ds_padded

        LOGGER.info(f"Merging {len(datasets)} forecast sources...")

        # Find overlapping issue_time range
        time_ranges = []
        for ds in datasets:
            start = pd.to_datetime(ds.issue_time.values.min())
            end = pd.to_datetime(ds.issue_time.values.max())
            time_ranges.append((start, end))

        overlap_start = max(r[0] for r in time_ranges)
        overlap_end = min(r[1] for r in time_ranges)

        if overlap_start > overlap_end:
            raise ValueError("No overlapping period between forecast sources")

        LOGGER.info(f"Aligning forecast sources to {overlap_start} - {overlap_end}")

        # Filter to overlap
        aligned = [ds.sel(issue_time=slice(overlap_start, overlap_end)) for ds in datasets]

        # Find common issue times
        common_times = reduce(
            lambda a, b: np.intersect1d(a, b),
            [ds.issue_time.values for ds in aligned]
        )

        aligned = [ds.sel(issue_time=common_times) for ds in aligned]
        LOGGER.info(f"Found {len(common_times)} common issue times")

        # Determine maximum lead_time
        max_lead_time = max(int(ds.lead_time.values.max()) for ds in aligned)
        target_lead_time = max(max_lead_time, required_lead_time)
        lead_time_range = np.arange(1, target_lead_time + 1)

        LOGGER.info(
            "Padding forecast sources to lead_time: %dh (data max=%dh, required=%dh)",
            target_lead_time,
            max_lead_time,
            required_lead_time,
        )

        # Pad shorter forecasts and add availability masks
        padded = []
        for i, ds in enumerate(aligned):
            loader = self._loaders[i]

            # Store valid lead times before padding
            valid_leads = set(int(lt) for lt in ds.lead_time.values)

            # Reindex to full range (NaN-padding for shorter-horizon sources)
            ds_padded = ds.reindex(lead_time=lead_time_range, fill_value=np.nan)

            # Add availability mask if configured
            if self._should_add_availability_mask(loader.config.name):
                mask_name = f"{loader.config.name}_available"
                # Create mask: 1 where data is valid, 0 where padded
                template_var = list(ds_padded.data_vars)[0]
                mask = xr.zeros_like(ds_padded[template_var])

                for lead in lead_time_range:
                    if lead in valid_leads:
                        mask.loc[{'lead_time': lead}] = 1.0

                mask.name = mask_name
                ds_padded = xr.merge([ds_padded, mask])
                LOGGER.info(f"Added availability mask: {mask_name}")

            padded.append(ds_padded)

        # Merge all sources
        merged = xr.merge(padded, compat='override')

        LOGGER.info(
            f"Successfully merged {len(padded)} forecast sources: "
            f"{len(merged.data_vars)} variables, {len(common_times)} issue times, "
            f"{max_lead_time}h horizon"
        )

        return merged

    def _should_add_availability_mask(self, loader_name: str) -> bool:
        """Check if availability mask should be added for this loader.

        Parameters
        ----------
        loader_name : str
            Name of the loader.

        Returns
        -------
        bool
            True if mask should be added.
        """
        if not hasattr(self.cfg, 'forecast_availability_masks'):
            return False

        masks = self.cfg.forecast_availability_masks
        if isinstance(masks, dict):
            return masks.get(loader_name, False)

        return False

    def _calculate_warmup_period(self, historical_ds: xr.Dataset) -> pd.Timedelta:
        """Calculate warmup period needed for historical data.

        Parameters
        ----------
        historical_ds : xr.Dataset
            Historical dataset.

        Returns
        -------
        pd.Timedelta
            Warmup period.
        """
        # Get frequency
        freq = utils.infer_frequency(historical_ds['time'].values)
        freq_offset = pd.tseries.frequencies.to_offset(freq)

        # Warmup needed = seq_length * frequency
        warmup_steps = max(self.seq_len)
        warmup = warmup_steps * freq_offset

        return warmup

    def _create_cache_key(self) -> str:
        """Create unique cache key from all loader configurations.

        Returns
        -------
        str
            12-character hash representing the complete loader configuration.
        """
        key_parts = [self.CACHE_VERSION]
        for loader in self._loaders:
            key_parts.append(loader.cache_key)

        combined = "_".join(key_parts)
        return hashlib.md5(combined.encode()).hexdigest()[:12]

    def _validate_cache(self, ds: xr.Dataset) -> bool:
        """Validate cached dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Cached dataset.

        Returns
        -------
        bool
            True if cache is valid.
        """
        # Check version
        if ds.attrs.get('forecast_cache_version') != self.CACHE_VERSION:
            LOGGER.info("Cache version mismatch")
            return False

        # Check all required variables present
        required_vars = set(
            self.cfg.hindcast_inputs +
            self.cfg.forecast_inputs +
            self.cfg.target_variables
        )
        available_vars = set(ds.data_vars)

        if not required_vars.issubset(available_vars):
            missing = required_vars - available_vars
            LOGGER.warning(f"Cache missing variables: {missing}")
            return False

        return True

    def _load_historical_xarray_data(self, basins: List[str] = None) -> Optional[xr.Dataset]:
        """Load historical observations from CSV files.

        Parameters
        ----------
        basins : List[str], optional
            Basins to load. If None, loads all basins.

        Returns
        -------
        Optional[xr.Dataset]
            Historical dataset or None if loading fails.
        """
        target_basins = basins if basins is not None else self.basins

        LOGGER.info(f"Loading historical data for {len(target_basins)} basins...")

        basin_datasets = []

        for basin in target_basins:
            csv_file = self.cfg.data_dir / "timeseries" / f"hydromet_timeseries_{basin}.csv"

            if not csv_file.exists():
                LOGGER.warning(f"File not found for basin {basin}: {csv_file}")
                continue

            try:
                df = pd.read_csv(csv_file)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()

                # Filter to hindcast_inputs and target_variables
                wanted_cols = self.cfg.hindcast_inputs + self.cfg.target_variables
                keep_cols = [col for col in wanted_cols if col in df.columns]

                if keep_cols:
                    df = df[keep_cols]
                    ds = xr.Dataset.from_dataframe(df)

                    if 'date' in ds.dims:
                        ds = ds.rename({'date': 'time'})

                    ds = ds.expand_dims(basin=[basin])
                    basin_datasets.append(ds)

                    LOGGER.info(
                        f"Loaded {len(df)} records for basin {basin} "
                        f"covering {df.index.min().date()} to {df.index.max().date()}"
                    )
                else:
                    LOGGER.warning(f"No requested variables found for basin {basin}")

            except Exception as e:
                LOGGER.error(f"Error loading historical data for basin {basin}: {e}")

        if not basin_datasets:
            LOGGER.warning("No historical data loaded for any basin")
            return None

        # Concatenate along basin dimension
        historical_ds = xr.concat(basin_datasets, dim='basin')

        return historical_ds

    def _ensure_scaler(self,
                      period: str,
                      scaler: Dict[str, Union[pd.Series, xr.DataArray]],
                      cfg: Config) -> Dict[str, Union[pd.Series, xr.DataArray]]:
        """Ensure scaler is available for non-training periods.

        Parameters
        ----------
        period : str
            Period name.
        scaler : dict
            Provided scaler.
        cfg : Config
            Run configuration.

        Returns
        -------
        dict
            Scaler dictionary.

        Raises
        ------
        ValueError
            If scaler cannot be loaded for non-training period.
        """
        if period == 'train':
            return scaler

        if scaler:
            return scaler

        # Try to load from train_dir
        if not hasattr(cfg, 'train_dir') or cfg.train_dir is None:
            raise ValueError(
                f"For period '{period}', either provide 'scaler' parameter or "
                f"set 'train_dir' in config to load scaler from training run"
            )

        try:
            scaler = utils.load_scaler(cfg.train_dir)
            LOGGER.info(f"Loaded scaler from {cfg.train_dir}")
            return scaler
        except Exception as e:
            LOGGER.error(f"Failed to load scaler from {cfg.train_dir}: {e}")
            raise ValueError(f"Could not load scaler for period '{period}'") from e

    def _update_data_availability(self, merged_ds: xr.Dataset):
        """Update availability from merged dataset.

        Parameters
        ----------
        merged_ds : xr.Dataset
            Merged dataset with historical and forecast data.
        """
        self._availability.update_from_dataset(merged_ds, 'time', 'historical')
        self._availability.update_from_dataset(merged_ds, 'issue_time', 'forecast')
        self._availability.update_from_attrs(merged_ds.attrs)

    def _validate_data_availability(self, cfg: Config):
        """Validate that configured periods fall within available data.

        Parameters
        ----------
        cfg : Config
            Run configuration.

        Raises
        ------
        ValueError
            If configured periods are outside available data ranges.
        """
        if not self._availability.is_complete():
            LOGGER.warning("Data availability incomplete - skipping validation")
            return

        # Check period dates
        period_attrs = {
            'train': (cfg.train_start_date, cfg.train_end_date),
            'validation': (cfg.validation_start_date, cfg.validation_end_date),
            'test': (cfg.test_start_date, cfg.test_end_date),
        }

        for period_name, (start_date, end_date) in period_attrs.items():
            if start_date is None or end_date is None:
                continue

            if start_date < self._availability.forecast_start:
                raise ValueError(
                    f"{period_name} start ({start_date}) is before forecast data start "
                    f"({self._availability.forecast_start})"
                )

            if end_date > self._availability.historical_end:
                raise ValueError(
                    f"{period_name} end ({end_date}) is after historical data end "
                    f"({self._availability.historical_end})"
                )

    def _filter_issue_times_for_period(self, basin: str, issue_times: np.ndarray) -> np.ndarray:
        """Filter forecast issue times to configured period.

        Parameters
        ----------
        basin : str
            Basin ID.
        issue_times : np.ndarray
            Array of forecast issue times.

        Returns
        -------
        np.ndarray
            Filtered issue times within the configured period.
        """
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

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        """Get a single sample for PyTorch DataLoader.

        Parameters
        ----------
        item : int
            Sample index.

        Returns
        -------
        Dict[str, torch.Tensor]
            Sample dictionary with hindcast, forecast, and target data.
        """
        basin, indices = self.lookup_table[item]

        sample = {}
        for freq, seq_len, forecast_seq_len, pointer in zip(
            self.frequencies, self.seq_len, self._forecast_seq_len, indices
        ):
            # If there's just one frequency, don't use suffixes
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
                sample[f'x_h{freq_suffix}'] = torch.concatenate(
                    [sample[f'x_h{freq_suffix}'], self.hindcast_counter], dim=-1
                )
                sample[f'x_f{freq_suffix}'] = torch.concatenate(
                    [sample[f'x_f{freq_suffix}'], self.forecast_counter], dim=-1
                )

        if self._per_basin_target_stds:
            sample['per_basin_target_stds'] = self._per_basin_target_stds[basin]
        if self.id_to_int:
            sample['x_one_hot'] = torch.nn.functional.one_hot(
                torch.tensor(self.id_to_int[basin]),
                num_classes=len(self.id_to_int)
            ).to(torch.float32)

        return sample

    def _create_lookup_table(self, xr_dataset: xr.Dataset):
        """Create lookup table for efficient sample retrieval.

        This method creates an index mapping from sample number to basin and data pointers.
        It also performs memory-mapping of numpy arrays for efficient data loading.

        Parameters
        ----------
        xr_dataset : xr.Dataset
            Merged xarray dataset with hindcast and forecast data.

        Raises
        ------
        NoTrainDataError, NoEvaluationDataError
            If no valid samples found.
        ValueError
            If required variables are missing.
        """
        import sys
        from tqdm import tqdm
        from typing import Tuple

        lookup: List[Tuple[str, List[Dict[str, Union[int, np.datetime64]]]]] = []
        if not self._disable_pbar:
            LOGGER.info("Create lookup table and convert to pytorch tensor")

        forecast_vars = [var for var in self.cfg.forecast_inputs if var in xr_dataset.data_vars]
        if not forecast_vars:
            raise ValueError('Configured cfg.forecast_inputs are missing from the merged dataset.')

        xr_fcst = xr_dataset[forecast_vars]

        hindcast_vars = [var for var in xr_dataset.data_vars if var not in forecast_vars]
        if not hindcast_vars:
            raise ValueError(
                'Dataset does not contain any hindcast or target variables after removing cfg.forecast_inputs.'
            )

        xr_hcst = xr_dataset[hindcast_vars]
        if 'lead_time' in xr_hcst.dims:
            xr_hcst = xr_hcst.drop_dims('lead_time')

        time_dim = 'time' if 'time' in xr_hcst.dims else 'date'
        issue_dim = 'issue_time' if 'issue_time' in xr_fcst.dims else (
            'time' if 'time' in xr_fcst.dims else 'date'
        )

        basins_without_samples: List[str] = []

        # Filter basin_coordinates to only include basins requested in self.basins
        available_basins = set(xr_hcst['basin'].values.tolist())
        requested_basins = set(self.basins)
        basin_coordinates = list(available_basins.intersection(requested_basins))

        if not basin_coordinates:
            raise ValueError(f"None of the requested basins {self.basins} found in the dataset.")

        for basin in tqdm(basin_coordinates, file=sys.stdout, disable=self._disable_pbar):
            basin_hcst = xr_hcst.sel(basin=basin, drop=True)

            # Optimization: Determine time range for this basin and period to slice BEFORE loading
            start_dates = self.start_and_end_dates.get(basin, {}).get('start_dates', [])
            end_dates = self.start_and_end_dates.get(basin, {}).get('end_dates', [])

            basin_fcst = xr_fcst.sel(basin=basin, drop=True)

            if start_dates and end_dates:
                # Convert to timestamps and find global min/max for this basin
                min_time = pd.to_datetime(min(start_dates))
                max_time = pd.to_datetime(max(end_dates)) + pd.Timedelta(days=1)

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
                LOGGER.warning(
                    "No forecast issue times within configured period for basin %s - skipping.", basin
                )
                basins_without_samples.append(basin)
                continue

            basin_fcst = basin_fcst.sel({issue_dim: filtered_issue_times})

            forecast_df = basin_fcst.to_dataframe().reset_index().set_index(
                [issue_dim, 'lead_time']
            ).sort_index()
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

            # Manual stacking to avoid xarray.to_array() reshaping errors
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

            # Replace NaN (from lead-time padding of shorter-horizon forecast sources) with 0.0.
            # After z-score normalization, 0.0 represents the feature mean -- a neutral value
            # that won't bias the LSTM when a forecast source is unavailable.
            fc_tensor = np.nan_to_num(fc_tensor, nan=0.0)

            # Apply explicit input gating: multiply features from shorter-horizon sources
            # by their availability mask, so padded lead times are cleanly zeroed out.
            if self.cfg.forecast_input_gating:
                for mask_name, gated_features in self.cfg.forecast_input_gating.items():
                    if mask_name in available_forecast:
                        mask_idx = available_forecast.index(mask_name)
                        mask_col = fc_tensor[:, :, mask_idx]
                        for feat_name in gated_features:
                            if feat_name in available_forecast:
                                feat_idx = available_forecast.index(feat_name)
                                fc_tensor[:, :, feat_idx] *= mask_col

            required_len = max(self._forecast_seq_len)
            if fc_tensor.shape[1] < required_len:
                LOGGER.warning(
                    f"Forecast tensor length ({fc_tensor.shape[1]}) is shorter than required "
                    f"forecast_seq_length ({required_len}) for basin {basin} - skipping."
                )
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
                    raise ValueError(
                        'seq_length must exceed forecast_seq_length to provide hindcast context.'
                    )

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

            # Compute per-basin target std for NSE loss compatibility
            obs = target_matrix  # shape (T, n_targets)
            stds = np.nanstd(obs, axis=0)
            if np.all(np.isfinite(stds)) and np.all(stds > 0):
                self._per_basin_target_stds[basin] = torch.tensor(stds.reshape(1, -1), dtype=torch.float32)

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
            LOGGER.info(
                "These basins do not have a single valid sample in the %s period: %s",
                self.period, basins_without_samples
            )

        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}
        self.num_samples = len(self.lookup_table)

        if self.num_samples == 0:
            if self.is_train:
                raise NoTrainDataError
            raise NoEvaluationDataError
