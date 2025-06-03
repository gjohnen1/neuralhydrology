"""Fetch historical weather data for basin centroids using the Open-Meteo API."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
import xarray as xr
from retry_requests import retry

def fetch_historical_for_basins(
    centroids: pd.DataFrame, 
    start_date: str, 
    end_date: str, 
    hourly_variables: Optional[List[str]] = None
) -> xr.Dataset:
    """Fetch historical hourly weather data for multiple basin centroids.

    Parameters
    ----------
    centroids : pd.DataFrame
        DataFrame with 'basin_name', 'latitude', 'longitude' columns.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    hourly_variables : List[str], optional
        List of hourly variables to fetch. Defaults to standard weather variables.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the historical data indexed by 'time' and 'basin'.
        
    Raises
    ------
    RuntimeError
        If no historical data could be fetched for any basin.
    """
    if hourly_variables is None:
        hourly_variables = [
            "temperature_2m", "relative_humidity_2m", "precipitation", "rain", "snowfall", 
            "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm", 
            "soil_moisture_100_to_255cm", "et0_fao_evapotranspiration", "surface_pressure", 
            "snow_depth_water_equivalent"
        ]
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    all_basin_data = []

    for index, row in centroids.iterrows():
        basin_name = row['basin_name']
        latitude = row['latitude']
        longitude = row['longitude']

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": hourly_variables,
            "models": "best_match",
            "timezone": "UTC"
        }
        
        try:
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]

            hourly = response.Hourly()
            
            time_range = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )
            
            hourly_data_dict = {"time": time_range}
            
            for i, var_name in enumerate(hourly_variables):
                hourly_data_dict[var_name] = hourly.Variables(i).ValuesAsNumpy()

            hourly_df = pd.DataFrame(data=hourly_data_dict)
            hourly_df = hourly_df.set_index('time')
            
            data_vars = {}
            for col in hourly_df.columns:
                data_vars[col] = (('time'), hourly_df[col].astype(np.float32).values) 
            basin_ds = xr.Dataset(
                data_vars,
                coords={
                    'time': hourly_df.index.values,
                    'basin': basin_name 
                }
            )
            basin_ds = basin_ds.expand_dims('basin')
            
            basin_ds.attrs['latitude'] = response.Latitude()
            basin_ds.attrs['longitude'] = response.Longitude()
            basin_ds.attrs['elevation'] = response.Elevation()
            basin_ds.attrs['timezone'] = response.Timezone()
            basin_ds.attrs['timezone_abbreviation'] = response.TimezoneAbbreviation()
            basin_ds.attrs['utc_offset_seconds'] = response.UtcOffsetSeconds()
            basin_ds.attrs['api_source'] = url
            basin_ds.attrs['api_model'] = params['models']

            all_basin_data.append(basin_ds)

        except Exception as e:
            raise RuntimeError(f"Error fetching data for {basin_name}: {e}")

    if not all_basin_data:
        raise RuntimeError("No historical data could be fetched for any basin.")

    try:
        return xr.concat(all_basin_data, dim="basin")
    except Exception as e:
        raise RuntimeError(f"Error combining historical datasets: {e}")


