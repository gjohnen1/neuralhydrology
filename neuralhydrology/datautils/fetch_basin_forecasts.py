"""Fetch basin forecasts from NOAA GEFS 35-day forecast dataset."""
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings('ignore')


def load_basin_centroids(file_path: str) -> Optional[pd.DataFrame]:
    """Load basin centroid coordinates from CSV.
    
    Parameters
    ----------
    file_path : str
        Path to the basin centroids CSV file
        
    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame containing basin names and coordinates, or None on error
    """
    try:
        centroids = pd.read_csv(file_path)
        return centroids
    except Exception as e:
        raise FileNotFoundError(f"Error loading basin centroids: {e}")


def extract_forecast_for_basin(ds: xr.Dataset, basin_name: str, latitude: float, 
                              longitude: float, init_time: Optional[str] = None) -> xr.Dataset:
    """Extract forecast data for a specific basin point.
    
    Parameters
    ----------
    ds : xr.Dataset
        NOAA GEFS forecast dataset
    basin_name : str
        Name of the basin
    latitude : float
        Latitude of the basin centroid
    longitude : float
        Longitude of the basin centroid
    init_time : Optional[str]
        Initial forecast time (e.g., 'YYYY-MM-DDTHH'). If None, use all available times
        
    Returns
    -------
    xr.Dataset
        Forecast data for the basin
    """
    if init_time is not None:
        basin_ds = ds.sel(init_time=init_time)
    else:
        basin_ds = ds
    
    basin_ds = basin_ds.sel(latitude=latitude, longitude=longitude, method="nearest")
    basin_ds = basin_ds.assign_coords(basin=basin_name)
    
    return basin_ds


def fetch_forecasts_for_basins(ds: xr.Dataset, centroids: pd.DataFrame, 
                              init_time: Optional[str] = None) -> xr.Dataset:
    """Extract forecasts for all basin centroids while preserving original dimensions.
    
    Parameters
    ----------
    ds : xr.Dataset
        NOAA GEFS forecast dataset
    centroids : pd.DataFrame
        DataFrame with basin centroid coordinates
    init_time : Optional[str]
        Initial forecast time (e.g., 'YYYY-MM-DDTHH'). If None, use all available times
        
    Returns
    -------
    xr.Dataset
        Combined dataset with basin dimension replacing lat/lon
    """
    if init_time is not None:
        ds = ds.sel(init_time=init_time)
    
    basin_data_list = []
    
    for idx, row in centroids.iterrows():
        basin_name = row['basin_name']
        lat = row['latitude']
        lon = row['longitude']
        
        basin_ds = ds.sel(latitude=lat, longitude=lon, method="nearest")
        basin_ds = basin_ds.assign_coords(basin=basin_name)
        basin_data_list.append(basin_ds)
    
    combined_ds = xr.concat(basin_data_list, dim="basin")
    
    return combined_ds


def interpolate_to_hourly(ds: xr.Dataset, max_hours: int = 240) -> xr.Dataset:
    """Interpolate forecast data to hourly resolution for the first 10 days.
    
    Parameters
    ----------
    ds : xr.Dataset
        The forecast dataset containing the lead_time dimension
    max_hours : int, optional
        Maximum forecast horizon in hours. Default is 240 (10 days)
        
    Returns
    -------
    xr.Dataset
        Dataset with hourly resolution and truncated forecast horizon,
        with lead_time dimension as int64 ranging from 1 to max_hours
    """
    if 'lead_time' not in ds.dims:
        raise ValueError("Dataset must contain a 'lead_time' dimension")
    
    if isinstance(ds.lead_time.values[0], np.timedelta64):
        lead_hours = ds.lead_time.dt.total_seconds().values / 3600
    else:
        lead_hours = ds.lead_time.values
    
    ds_shortened = ds.sel(lead_time=ds.lead_time[lead_hours <= max_hours])
    
    if isinstance(ds.lead_time.values[0], np.timedelta64):
        hourly_lead_times = np.array([np.timedelta64(int(hour), 'h') 
                                      for hour in range(int(max_hours) + 1)])
    else:
        hourly_lead_times = np.arange(0, max_hours + 1, 1)
    
    ds_hourly = ds_shortened.interp(lead_time=hourly_lead_times, method='linear')
    ds_hourly = ds_hourly.isel(lead_time=slice(1, None))
    
    new_lead_time = np.arange(1, max_hours + 1)
    ds_hourly = ds_hourly.assign_coords(lead_time=new_lead_time)
    
    ds_hourly.lead_time.attrs['units'] = 'hours'
    ds_hourly.lead_time.attrs['long_name'] = 'Lead time in hours'
    
    ds_hourly.attrs['interpolation'] = 'Linear interpolation to hourly resolution'
    ds_hourly.attrs['original_resolution'] = '3-hourly for first 10 days'
    ds_hourly.attrs['max_forecast_hours'] = max_hours
    ds_hourly.attrs['lead_time_format'] = 'Integer hours from 1 to 240'
    
    return ds_hourly


def main() -> None:
    """Main function to run the basin forecast extraction process."""
    basin_centroids_file = "../data/basin_centroids.csv"
    output_dir = "../data/basin_forecasts"
    
    os.makedirs(output_dir, exist_ok=True)
    
    centroids = load_basin_centroids(basin_centroids_file)
    
    if centroids is not None:
        zarr_url = "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr?email=optional@email.com"
        try:
            ds = xr.open_zarr(zarr_url, decode_timedelta=True)
            init_time = None
            combined_ds = fetch_forecasts_for_basins(ds, centroids, init_time)
            
            if combined_ds is not None:                
                return combined_ds
        except Exception as e:
            raise RuntimeError(f"Error processing NOAA GEFS dataset from {zarr_url}: {e}")
    else:
        raise FileNotFoundError("Failed to load basin centroids.")


if __name__ == "__main__":
    main()