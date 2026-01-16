import logging
import shutil
import time
from pathlib import Path
from typing import List, Dict, Union, Tuple

import pandas as pd
import xarray as xr
import numpy as np
from neuralhydrology.datasetzoo.onlineforecastdataset import OnlineForecastDataset
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import NoEvaluationDataError, NoTrainDataError
from neuralhydrology.datautils import utils

LOGGER = logging.getLogger(__name__)

class PerfectForecastDataset(OnlineForecastDataset):
    """Dataset for Perfect Prognosis (hindcast as forecast).
    
    This dataset treats historical observations as "perfect" forecasts.
    It generates forecast structures (issue_time, lead_time) from the observed time series.
    """
    
    CACHE_VERSION = "perfect-forecast-v1"

    def _load_or_create_xarray_dataset(self) -> xr.Dataset:
        basin_datasets = []
        
        # Ensure cache directory exists - USING SEPARATE CACHE FOR PERFECT FORECAST
        cache_dir = self.cfg.data_dir / "zarr_cache_perfect"
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

    def _load_forecast_xarray_data(self, basins: List[str] = None) -> xr.Dataset:
        target_basins = basins if basins is not None else self.basins
        LOGGER.info(f"Generating Perfect Forecast (Hindcast) data for basins: {target_basins}")

        # Load historical data
        historical_ds = self._load_historical_xarray_data(basins=target_basins)
        if historical_ds is None:
            return None

        # Determine issue times: Daily at 00:00
        # We need to cover the entire range of historical data
        hist_times = pd.to_datetime(historical_ds['time'].values)
        if len(hist_times) == 0:
            return None
            
        start_time = hist_times.min().normalize()
        end_time = hist_times.max().normalize()
        
        # Enforce daily issue times
        issue_times = pd.date_range(start=start_time, end=end_time, freq='D')
        LOGGER.info(f"Generating forecasts for {len(issue_times)} daily issue times from {start_time} to {end_time}")
        
        # Mapping config variables
        # We need to map requested forecast inputs (e.g. 'temp_q50') to historical vars (e.g. 'temp')
        var_mapping = {}
        for var in self.cfg.forecast_inputs:
            # Strip quantile suffixes to find base variable
            base_var = var.replace('_q25', '').replace('_q50', '').replace('_q75', '')
            
            if base_var in historical_ds.data_vars:
                var_mapping[var] = base_var
            elif var in historical_ds.data_vars:
                 var_mapping[var] = var
            else:
                LOGGER.warning(f"Forecast input '{var}' (base '{base_var}') not found in historical data variables: {list(historical_ds.data_vars)}")

        if not var_mapping:
            LOGGER.error("No matching historical variables found for forecast inputs.")
            return None
            
        # Determine max horizon from config
        if isinstance(self._forecast_seq_len, list):
             max_horizon_hours = max(self._forecast_seq_len)
        else:
             max_horizon_hours = self._forecast_seq_len
             
        # Assume hourly lead times match the data frequency, or just hourly if standard
        lead_times = np.arange(1, max_horizon_hours + 1)
        
        leads_data = []
        
        # We iterate over lead times. 
        # For lead time L, the forecast value at issue_time T is the observation at T+L.
        for lead in lead_times:
            # Shift historical data backwards by L steps (assuming 1H frequency)
            # historical_ds['time'] is 'time'.
            # Robust way: Target times = issue_times + pd.Timedelta(hours=lead)
            target_times = issue_times + pd.Timedelta(hours=int(lead))
            
            # Select from historical_ds at target_times
            # We use reindex to handle missing times (will be NaN)
            subset = historical_ds.reindex(time=target_times)
            
            # Now we have data at target_times. We want to assign this to issue_times.
            subset['time'] = issue_times
            subset = subset.rename({'time': 'issue_time'})
            
            # Extract variables
            ds_lead = xr.Dataset()
            for tgt_var, src_var in var_mapping.items():
                ds_lead[tgt_var] = subset[src_var]

            # Explicitly add lead_time coord
            ds_lead = ds_lead.assign_coords(lead_time=lead)
            
            leads_data.append(ds_lead)
            
        # Concatenate along lead_time dimension
        LOGGER.info(f"Concatenating {len(lead_times)} lead times...")
        forecast_ds = xr.concat(leads_data, dim='lead_time')
        
        # Explicitly ensure lead_time is a coordinate variable
        if 'lead_time' not in forecast_ds.coords:
            forecast_ds = forecast_ds.assign_coords(lead_time=lead_times)

        return forecast_ds
