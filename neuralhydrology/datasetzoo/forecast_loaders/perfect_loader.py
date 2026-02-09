"""Perfect forecast loader for pseudo-perfect prognosis (hindcast as forecast)."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from neuralhydrology.datasetzoo.forecast_loaders.base import ForecastLoader
from neuralhydrology.datasetzoo.forecast_loaders.registry import ForecastLoaderRegistry


@ForecastLoaderRegistry.register('perfect_forecast')
class PerfectForecastLoader(ForecastLoader):
    """Loader for perfect prognosis forecasts (historical observations as forecasts).

    This loader generates synthetic "perfect" forecasts by using historical observations
    as if they were forecasts. This is useful for:
    - Establishing an upper bound on forecast-based model performance
    - Debugging model architecture without forecast uncertainty
    - Comparing real vs perfect forecast scenarios

    The loader:
    1. Loads historical observations from CSV files
    2. Creates daily forecast issue times spanning the historical period
    3. Generates forecasts by shifting observations forward in time
       (forecast at issue_time T with lead L = observation at T+L)

    Configuration options (in loader_kwargs):
    - max_horizon: Maximum forecast horizon in hours (default: from cfg.forecast_seq_length)

    Example configuration:
        forecast_sources:
          - name: perfect
            type: perfect_forecast
            suffix: _perfect
            variables:
              - temperature_2m
              - precipitation
            quartiles: [0.5]  # Only median needed for deterministic perfect forecast
            loader_kwargs:
              max_horizon: 240
    """

    def __init__(self, config, cfg):
        """Initialize perfect forecast loader.

        Parameters
        ----------
        config : ForecastLoaderConfig
            Loader configuration.
        cfg : Config
            Global run configuration.
        """
        super().__init__(config, cfg)

        # Determine max horizon
        if 'max_horizon' in config.loader_kwargs:
            self.max_horizon = config.loader_kwargs['max_horizon']
        elif hasattr(cfg, 'forecast_seq_length'):
            if isinstance(cfg.forecast_seq_length, list):
                self.max_horizon = max(cfg.forecast_seq_length)
            else:
                self.max_horizon = cfg.forecast_seq_length
        else:
            self.max_horizon = 240  # Default to 10 days

        self.logger.info(f"Perfect forecast loader configured with max_horizon={self.max_horizon}h")

    def load(self, basins: List[str]) -> Optional[xr.Dataset]:
        """Load perfect forecasts for specified basins.

        Generates synthetic forecasts from historical observations by treating
        future observations as "perfect" forecasts.

        Parameters
        ----------
        basins : List[str]
            Basin IDs to load forecasts for.

        Returns
        -------
        Optional[xr.Dataset]
            Dataset with dimensions (basin, issue_time, lead_time) and forecast variables
            named with configured suffix and quartile suffixes (e.g., 'temperature_2m_perfect_q50').
            Returns None if no data could be loaded.
        """
        self.logger.info(f"Generating perfect forecasts for {len(basins)} basins...")

        # Load historical observations
        historical_ds = self._load_historical_data(basins)
        if historical_ds is None:
            self.logger.error("Failed to load historical data")
            return None

        # Determine issue times (daily at 00:00)
        hist_times = pd.to_datetime(historical_ds['time'].values)
        start_time = hist_times.min().normalize()
        end_time = hist_times.max().normalize()

        issue_times = pd.date_range(start=start_time, end=end_time, freq='D')
        self.logger.info(
            f"Generating forecasts for {len(issue_times)} daily issue times "
            f"from {start_time.date()} to {end_time.date()}"
        )

        # Map forecast variables to historical variables
        var_mapping = self._create_variable_mapping(historical_ds)
        if not var_mapping:
            self.logger.error("No variable mapping found between forecast inputs and historical data")
            return None

        self.logger.info(f"Variable mapping: {var_mapping}")

        # Generate forecast structure
        lead_times = np.arange(1, self.max_horizon + 1)
        leads_data = []

        self.logger.info(f"Generating forecasts for {len(lead_times)} lead times...")

        for lead in lead_times:
            # Target times = issue_times + lead hours
            # The forecast at issue_time T with lead L is the observation at T+L
            target_times = issue_times + pd.Timedelta(hours=int(lead))

            # Reindex historical data to target times (handles missing values with NaN)
            subset = historical_ds.reindex(time=target_times)

            # Reassign time coordinate to issue_times
            subset['time'] = issue_times
            subset = subset.rename({'time': 'issue_time'})

            # Extract and rename variables according to mapping
            ds_lead = xr.Dataset()
            for forecast_var, historical_var in var_mapping.items():
                ds_lead[forecast_var] = subset[historical_var]

            # Add lead_time coordinate
            ds_lead = ds_lead.assign_coords(lead_time=lead)

            leads_data.append(ds_lead)

        # Concatenate along lead_time dimension
        self.logger.info(f"Concatenating {len(lead_times)} lead times...")
        forecast_ds = xr.concat(leads_data, dim='lead_time')

        # Ensure lead_time is a coordinate
        if 'lead_time' not in forecast_ds.coords:
            forecast_ds = forecast_ds.assign_coords(lead_time=lead_times)

        self.logger.info(
            f"Successfully generated perfect forecasts: {len(forecast_ds.data_vars)} variables, "
            f"{len(basins)} basins, {self.max_horizon}h horizon"
        )

        return forecast_ds

    def _load_historical_data(self, basins: List[str]) -> Optional[xr.Dataset]:
        """Load historical observations from CSV files.

        Parameters
        ----------
        basins : List[str]
            Basin IDs to load data for.

        Returns
        -------
        Optional[xr.Dataset]
            Dataset with historical observations, or None if loading fails.
        """
        self.logger.info(f"Loading historical data for {len(basins)} basins...")

        basin_datasets = []

        for basin in basins:
            csv_file = self.cfg.data_dir / "timeseries" / f"hydromet_timeseries_{basin}.csv"

            if not csv_file.exists():
                self.logger.warning(f"Historical file not found for basin {basin}: {csv_file}")
                continue

            try:
                df = pd.read_csv(csv_file)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()

                # Keep only variables we need for forecast generation
                available_cols = df.columns.tolist()

                # We need the base variables (strip suffixes and quartiles)
                base_vars_needed = set()
                for var in self.config.variables:
                    # Strip quartile suffixes
                    base_var = (var.replace('_q25', '').replace('_q50', '')
                               .replace('_q75', '').replace(self.config.suffix, ''))
                    base_vars_needed.add(base_var)

                keep_cols = [col for col in base_vars_needed if col in available_cols]

                if keep_cols:
                    df = df[keep_cols]

                    # Convert to xarray
                    ds = xr.Dataset.from_dataframe(df)

                    # Rename date to time
                    if 'date' in ds.dims:
                        ds = ds.rename({'date': 'time'})

                    # Add basin dimension
                    ds = ds.expand_dims(basin=[basin])

                    basin_datasets.append(ds)
                    self.logger.info(
                        f"Loaded {len(df)} records for basin {basin} "
                        f"covering {df.index.min().date()} to {df.index.max().date()}"
                    )
                else:
                    self.logger.warning(f"No requested variables found for basin {basin}")

            except Exception as e:
                self.logger.error(f"Error loading historical data for basin {basin}: {e}")

        if not basin_datasets:
            self.logger.warning("No historical data loaded for any basin")
            return None

        # Concatenate along basin dimension
        historical_ds = xr.concat(basin_datasets, dim='basin')

        return historical_ds

    def _create_variable_mapping(self, historical_ds: xr.Dataset) -> Dict[str, str]:
        """Create mapping from forecast variable names to historical variable names.

        Parameters
        ----------
        historical_ds : xr.Dataset
            Historical dataset with available variables.

        Returns
        -------
        Dict[str, str]
            Mapping from forecast variable name (output) to historical variable name (input).

        Notes
        -----
        This method handles:
        - Stripping quartile suffixes (_q25, _q50, _q75)
        - Stripping configured suffix
        - Creating output names with suffix and quartile
        """
        mapping = {}
        available_vars = set(historical_ds.data_vars)

        for var in self.config.variables:
            # Strip quartile and suffix to find base variable
            base_var = (var.replace('_q25', '').replace('_q50', '')
                       .replace('_q75', '').replace(self.config.suffix, ''))

            if base_var not in available_vars:
                self.logger.warning(
                    f"Variable '{base_var}' (from config '{var}') not found in historical data. "
                    f"Available: {available_vars}"
                )
                continue

            # Create output variable names with suffix and quartile
            if self.config.quartiles:
                for q in self.config.quartiles:
                    # Build output name: base + suffix + quartile
                    q_suffix = f"_q{int(q*100)}"
                    output_var = f"{base_var}{self.config.suffix}{q_suffix}"
                    mapping[output_var] = base_var
            else:
                # No quartiles, just add suffix
                output_var = f"{base_var}{self.config.suffix}"
                mapping[output_var] = base_var

        return mapping

    def get_horizon_hours(self) -> int:
        """Return maximum forecast horizon in hours.

        Returns
        -------
        int
            Maximum forecast lead time in hours (configured max_horizon).
        """
        return self.max_horizon
