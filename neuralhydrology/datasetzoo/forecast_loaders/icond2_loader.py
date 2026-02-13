"""ICON-D2 forecast loader for local high-resolution forecasts."""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import xarray as xr

from neuralhydrology.datasetzoo.forecast_loaders.base import ForecastLoader
from neuralhydrology.datasetzoo.forecast_loaders.registry import ForecastLoaderRegistry
from neuralhydrology.datasetzoo.forecast_utils.quartile_computer import QuartileComputer


@ForecastLoaderRegistry.register('icond2')
class ICOND2Loader(ForecastLoader):
    """Loader for ICON-D2 (ICON-Germany) high-resolution local forecasts.

    This loader reads ICON-D2 forecast data from local NetCDF files. ICON-D2 provides:
    - 48-hour forecast horizon
    - Hourly temporal resolution
    - High spatial resolution (~2.2 km)
    - Both deterministic and ensemble forecasts

    The loader expects data organized as:
    - Deterministic: {data_dir}/forecasts/icond2_deterministic/CAMELS_DE_1h_deterministic_met_forecast_gregor_{catchment}.nc
    - Ensemble: {data_dir}/forecasts/icond2_ensemble/CAMELS_DE_1h_ensemble_met_forecast_gregor_{catchment}.nc

    Configuration options (in loader_kwargs):
    - data_dir: Directory containing ICON-D2 data (default: 'icond2', relative to cfg.data_dir)
    - horizon_hours: Maximum forecast horizon (default: 48)
    - basin_mapping: Custom basin-to-catchment mapping (optional, uses default if not provided)

    Example configuration:
        forecast_sources:
          - name: icond2
            type: icond2
            suffix: _icond2
            variables:
              - temperature_mean
              - precipitation_mean
            quartiles: [0.25, 0.5, 0.75]
            loader_kwargs:
              data_dir: icond2
              horizon_hours: 48
    """

    # Default basin-to-catchment mapping for CAMELS-DE
    DEFAULT_BASIN_MAPPING = {
        'DE1': 'innerste',
        'DE2': 'oker',
        'DE3': 'ecker',
        'DE4': 'soese',
        'DE5': 'grane',
    }

    HORIZON_HOURS = 48

    def _drop_gauge_id(self, ds: xr.Dataset, source_name: str) -> xr.Dataset:
        """Drop/squeeze gauge dimension to keep tensors 2D over issue_time x lead_time.

        Some ICON-D2 files carry a singleton ``gauge_id`` dimension even for per-catchment files.
        ForecastDataset expects forecast variables with dimensions ``(issue_time, lead_time)``
        (plus basin at merge stage), so we collapse ``gauge_id`` to avoid transpose failures.
        """
        if 'gauge_id' in ds.dims:
            gauge_count = int(ds.sizes.get('gauge_id', 0))
            if gauge_count > 1:
                self.logger.warning(
                    "ICON-D2 %s has %d gauge_id entries. Selecting the first entry for catchment-aligned loading.",
                    source_name,
                    gauge_count,
                )
            ds = ds.isel(gauge_id=0, drop=True)
        elif 'gauge_id' in ds.coords:
            ds = ds.drop_vars('gauge_id')
        return ds

    def __init__(self, config, cfg):
        """Initialize ICON-D2 loader.

        Parameters
        ----------
        config : ForecastLoaderConfig
            Loader configuration.
        cfg : Config
            Global run configuration.
        """
        super().__init__(config, cfg)

        # Parse loader-specific configuration
        data_dir_name = config.loader_kwargs.get('data_dir', 'icond2')
        self.data_dir = cfg.data_dir / data_dir_name
        self.horizon_hours = config.loader_kwargs.get('horizon_hours', self.HORIZON_HOURS)

        # Basin-to-catchment mapping (use custom or default)
        self.basin_mapping = config.loader_kwargs.get('basin_mapping', self.DEFAULT_BASIN_MAPPING)

    def load(self, basins: List[str]) -> Optional[xr.Dataset]:
        """Load ICON-D2 forecasts for specified basins.

        Parameters
        ----------
        basins : List[str]
            Basin IDs to load forecasts for.

        Returns
        -------
        Optional[xr.Dataset]
            Dataset with dimensions (basin, issue_time, lead_time) and forecast variables
            named with configured suffix and quartile suffixes (e.g., 'temperature_mean_icond2_q50').
            Returns None if no data could be loaded.
        """
        self.logger.info(f"Loading ICON-D2 forecasts for {len(basins)} basins...")

        if not self.data_dir.exists():
            self.logger.error(f"ICON-D2 data directory not found: {self.data_dir}")
            return None

        # Separate variables into deterministic and ensemble
        det_vars, ens_vars = self._categorize_variables()

        datasets = []
        for basin in basins:
            catchment = self.basin_mapping.get(basin)
            if not catchment:
                self.logger.warning(f"No catchment mapping for basin {basin} - skipping")
                continue

            self.logger.info(f"Loading ICON-D2 for basin {basin} (catchment: {catchment})...")

            basin_parts = []

            # Load deterministic forecasts
            if det_vars:
                det_ds = self._load_deterministic(catchment, det_vars)
                if det_ds is not None:
                    basin_parts.append(det_ds)

            # Load ensemble forecasts
            if ens_vars:
                ens_ds = self._load_ensemble(catchment, ens_vars)
                if ens_ds is not None:
                    basin_parts.append(ens_ds)

            if basin_parts:
                # Merge deterministic and ensemble
                merged = xr.merge(basin_parts)
                merged = merged.assign_coords(basin=basin)

                # Ensure lead_time is integer hours
                if pd.api.types.is_timedelta64_dtype(merged.lead_time):
                    merged = merged.assign_coords(
                        lead_time=(merged.lead_time / pd.Timedelta('1h')).astype(int)
                    )

                datasets.append(merged)
            else:
                self.logger.warning(f"No ICON-D2 data loaded for basin {basin}")

        if not datasets:
            self.logger.warning("No ICON-D2 data loaded for any basin")
            return None

        # Concatenate all basins
        combined = xr.concat(datasets, dim='basin')

        self.logger.info(
            f"Successfully loaded ICON-D2: {len(combined.data_vars)} variables, "
            f"{len(datasets)} basins, {self.horizon_hours}h horizon"
        )

        return combined

    def _categorize_variables(self) -> tuple:
        """Separate variables into deterministic and ensemble categories.

        Returns
        -------
        tuple
            (deterministic_vars, ensemble_vars) - lists of variable names.

        Notes
        -----
        Variables like precipitation appear in both deterministic and ensemble files.
        The deterministic value produces a suffixed variable (e.g., precipitation_mean_icond2),
        while the ensemble produces quartile-suffixed variables (e.g., precipitation_mean_icond2_q25).
        This matches the legacy CombinedForecastDataset behavior.
        """
        det_vars = []
        ens_vars = []

        for var in self.config.variables:
            if 'precipitation' in var.lower():
                # Precipitation exists in both deterministic and ensemble files
                det_vars.append(var)
                ens_vars.append(var)
            else:
                det_vars.append(var)

        return det_vars, ens_vars

    def _load_deterministic(self, catchment: str, variables: List[str]) -> Optional[xr.Dataset]:
        """Load deterministic ICON-D2 forecasts.

        Parameters
        ----------
        catchment : str
            Catchment name for file lookup.
        variables : List[str]
            Variables to load.

        Returns
        -------
        Optional[xr.Dataset]
            Dataset with deterministic forecasts, or None if file not found.
        """
        det_path = (self.data_dir / "forecasts" / "icond2_deterministic" /
                   f"CAMELS_DE_1h_deterministic_met_forecast_gregor_{catchment}.nc")

        if not det_path.exists():
            self.logger.warning(f"Deterministic file not found: {det_path}")
            return None

        try:
            ds = xr.open_dataset(det_path, decode_timedelta=True)

            # Filter to 00Z initialization only
            ds = ds.sel(init_time=ds.init_time.dt.hour == 0)

            # Standardize dimension names
            if 'init_time' in ds.dims:
                ds = ds.rename({'init_time': 'issue_time'})

            # Drop gauge_id if present (not needed)
            if 'gauge_id' in ds.coords:
                ds = ds.drop_vars('gauge_id')

            # Filter to requested variables
            available = [v for v in variables if v in ds.data_vars]
            if not available:
                self.logger.warning(
                    f"None of the requested variables {variables} found in {det_path.name}"
                )
                return None

            ds = ds[available]
            ds = self._drop_gauge_id(ds, source_name=det_path.name)

            # Add suffix to variable names
            rename_map = {v: self.add_suffix(v) for v in available}
            ds = ds.rename(rename_map)

            self.logger.info(f"Loaded {len(available)} deterministic variables from {det_path.name}")
            return ds

        except Exception as e:
            self.logger.error(f"Error loading deterministic file {det_path}: {e}")
            return None

    def _load_ensemble(self, catchment: str, variables: List[str]) -> Optional[xr.Dataset]:
        """Load ensemble ICON-D2 forecasts and compute quartiles.

        Parameters
        ----------
        catchment : str
            Catchment name for file lookup.
        variables : List[str]
            Variables to load (ensemble variables).

        Returns
        -------
        Optional[xr.Dataset]
            Dataset with quartile variables, or None if file not found.
        """
        ens_path = (self.data_dir / "forecasts" / "icond2_ensemble" /
                   f"CAMELS_DE_1h_ensemble_met_forecast_gregor_{catchment}.nc")

        if not ens_path.exists():
            self.logger.warning(f"Ensemble file not found: {ens_path}")
            return None

        try:
            ds = xr.open_dataset(ens_path, decode_timedelta=True)

            # Filter to 00Z initialization only
            ds = ds.sel(init_time=ds.init_time.dt.hour == 0)

            # Filter to requested variables
            available = [v for v in variables if v in ds.data_vars]
            if not available:
                self.logger.warning(
                    f"None of the requested variables {variables} found in {ens_path.name}"
                )
                return None

            ds = ds[available]
            ds = self._drop_gauge_id(ds, source_name=ens_path.name)

            # Compute quartiles
            ds_quartiles = QuartileComputer.compute_as_variables(
                ds,
                quartiles=tuple(self.config.quartiles),
                suffix_base=self.config.suffix
            )

            # Standardize dimension names
            if 'init_time' in ds_quartiles.dims:
                ds_quartiles = ds_quartiles.rename({'init_time': 'issue_time'})

            ds_quartiles = self._drop_gauge_id(ds_quartiles, source_name=ens_path.name)

            # Drop gauge_id if present
            if 'gauge_id' in ds_quartiles.coords:
                ds_quartiles = ds_quartiles.drop_vars('gauge_id')

            self.logger.info(f"Loaded {len(available)} ensemble variables from {ens_path.name}")
            return ds_quartiles

        except Exception as e:
            self.logger.error(f"Error loading ensemble file {ens_path}: {e}")
            return None

    def get_horizon_hours(self) -> int:
        """Return maximum forecast horizon in hours.

        Returns
        -------
        int
            Maximum forecast lead time in hours (configured horizon_hours).
        """
        return self.horizon_hours
