"""GEFS forecast loader for NOAA Global Ensemble Forecast System data."""

import time
from typing import List, Optional

import xarray as xr

from neuralhydrology.datasetzoo.forecast_loaders.base import ForecastLoader
from neuralhydrology.datasetzoo.forecast_loaders.registry import ForecastLoaderRegistry
from neuralhydrology.datasetzoo.forecast_utils.quartile_computer import QuartileComputer
from neuralhydrology.datautils.fetch_basin_forecasts import (
    load_basin_centroids,
    fetch_forecasts_for_basins,
    interpolate_to_hourly,
)


@ForecastLoaderRegistry.register('gefs')
class GEFSLoader(ForecastLoader):
    """Loader for NOAA GEFS (Global Ensemble Forecast System) 35-day forecasts.

    This loader connects to the NOAA GEFS remote zarr dataset, extracts forecasts
    at basin centroids, computes ensemble quartiles, and interpolates to hourly resolution.

    The GEFS dataset provides:
    - 35-day forecast horizon (updated daily)
    - 30-member ensemble
    - 3-hourly temporal resolution (interpolated to hourly)
    - Global coverage at ~0.5 degree resolution

    Configuration options (in loader_kwargs):
    - max_hours: Maximum forecast horizon in hours (default: 240 = 10 days)
    - retry_attempts: Number of connection retry attempts (default: 5)
    - exponential_backoff: Use exponential backoff for retries (default: True)

    Example configuration:
        forecast_sources:
          - name: gefs
            type: gefs
            suffix: _gefs
            variables:
              - temperature_2m
              - precipitation_surface
            quartiles: [0.25, 0.5, 0.75]
            loader_kwargs:
              max_hours: 240
              retry_attempts: 5
    """

    NOAA_URL = "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr?email=optional@email.com"
    DEFAULT_MAX_HOURS = 240
    DEFAULT_RETRY_ATTEMPTS = 5

    def __init__(self, config, cfg):
        """Initialize GEFS loader.

        Parameters
        ----------
        config : ForecastLoaderConfig
            Loader configuration.
        cfg : Config
            Global run configuration.
        """
        super().__init__(config, cfg)

        # Parse loader-specific configuration
        self.max_hours = config.loader_kwargs.get('max_hours', self.DEFAULT_MAX_HOURS)
        self.retry_attempts = config.loader_kwargs.get('retry_attempts', self.DEFAULT_RETRY_ATTEMPTS)
        self.exponential_backoff = config.loader_kwargs.get('exponential_backoff', True)

    def load(self, basins: List[str]) -> Optional[xr.Dataset]:
        """Load GEFS forecasts for specified basins.

        Parameters
        ----------
        basins : List[str]
            Basin IDs to load forecasts for.

        Returns
        -------
        Optional[xr.Dataset]
            Dataset with dimensions (basin, issue_time, lead_time) and forecast variables
            named with configured suffix and quartile suffixes (e.g., 'temperature_2m_gefs_q50').
            Returns None if loading fails.

        Raises
        ------
        ValueError
            If basin centroids file not found or no variables available.
        ConnectionError
            If all retry attempts to connect to NOAA fail.
        """
        self.logger.info(f"Loading GEFS forecasts for {len(basins)} basins...")

        # Load basin centroids
        basin_centroids_file = self.cfg.data_dir / "basin_centroids" / "basin_centroids.csv"
        if not basin_centroids_file.exists():
            raise ValueError(f"Basin centroids file not found: {basin_centroids_file}")

        centroids = load_basin_centroids(basin_centroids_file)
        centroids = centroids[centroids['basin_name'].isin(basins)]

        if len(centroids) == 0:
            self.logger.error(f"No basin centroids found for basins: {basins}")
            return None

        self.logger.info(f"Found centroids for {len(centroids)} basins")

        # Connect to NOAA GEFS with retry logic
        ds = self._connect_to_noaa_with_retry()

        # Filter to requested variables
        available_vars = [v for v in self.config.variables if v in ds.data_vars]
        if not available_vars:
            raise ValueError(
                f"None of the requested variables {self.config.variables} found in NOAA GEFS dataset. "
                f"Available variables (first 20): {list(ds.data_vars)[:20]}"
            )

        self.logger.info(f"Filtering NOAA dataset to {len(available_vars)} variables: {available_vars}")
        ds = ds[available_vars]

        # Extract forecasts for basin centroids
        self.logger.info("Extracting forecasts at basin centroids...")
        basin_forecasts = fetch_forecasts_for_basins(ds, centroids)

        # Compute quartiles
        self.logger.info(f"Computing quartiles: {self.config.quartiles}")
        basin_forecasts_quartiles = QuartileComputer.compute_as_variables(
            basin_forecasts,
            quartiles=tuple(self.config.quartiles),
            suffix_base=self.config.suffix
        )

        # Interpolate to hourly resolution
        self.logger.info(f"Interpolating to hourly resolution (max {self.max_hours}h)...")
        basin_forecasts_hourly = interpolate_to_hourly(
            basin_forecasts_quartiles,
            max_hours=self.max_hours
        )

        # Load into memory (finalize remote fetch)
        self.logger.info("Loading GEFS data into memory...")
        basin_forecasts_hourly = basin_forecasts_hourly.compute()

        # Standardize dimension names
        if 'init_time' in basin_forecasts_hourly.dims:
            basin_forecasts_hourly = basin_forecasts_hourly.rename({'init_time': 'issue_time'})

        self.logger.info(
            f"Successfully loaded GEFS: {len(basin_forecasts_hourly.data_vars)} variables, "
            f"{len(basins)} basins, {self.max_hours}h horizon"
        )

        return basin_forecasts_hourly

    def _connect_to_noaa_with_retry(self) -> xr.Dataset:
        """Connect to NOAA GEFS dataset with exponential backoff retry logic.

        Returns
        -------
        xr.Dataset
            Remote GEFS dataset (not yet computed/loaded into memory).

        Raises
        ------
        ConnectionError
            If all retry attempts fail.
        """
        last_error = None

        for attempt in range(1, self.retry_attempts + 1):
            try:
                self.logger.info(f"Connecting to NOAA GEFS (attempt {attempt}/{self.retry_attempts})...")
                ds = xr.open_zarr(self.NOAA_URL, decode_timedelta=True)
                self.logger.info("Successfully connected to NOAA GEFS")
                return ds

            except Exception as e:
                last_error = e
                self.logger.warning(f"GEFS connection attempt {attempt} failed: {e}")

                if attempt < self.retry_attempts:
                    if self.exponential_backoff:
                        wait_time = 10 * attempt  # 10s, 20s, 30s, 40s, 50s
                    else:
                        wait_time = 10

                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

        # All attempts failed
        raise ConnectionError(
            f"Failed to connect to NOAA GEFS after {self.retry_attempts} attempts. "
            f"Last error: {last_error}"
        )

    def get_horizon_hours(self) -> int:
        """Return maximum forecast horizon in hours.

        Returns
        -------
        int
            Maximum forecast lead time in hours (configured max_hours).
        """
        return self.max_hours
