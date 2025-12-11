# Data Sources and Preprocessing Pipeline

This document summarizes the data sources, formats, and the preprocessing pipeline implemented in `OnlineForecastDataset` for the operational forecasting workflow.

## Overview

The `OnlineForecastDataset` is designed for operational forecasting with mixed temporal indexing. It integrates:
1.  **Historical Data (Hindcast):** Observed meteorological and hydrological data from local files.
2.  **Forecast Data:** Ensemble weather forecasts fetched dynamically from the NOAA GEFS online Zarr store.
3.  **Static Attributes:** Catchment characteristics from local files.

## Data Sources

### 1. Historical Data (Hindcast)
*   **Source:** Local CSV files.
*   **Location:** `data/timeseries/hydromet_timeseries_{basin}.csv` (relative to the configured `data_dir`).
*   **Content:** Time series of observed variables (e.g., temperature, precipitation, discharge).
*   **Format:** CSV file where columns match the variable names specified in `hindcast_inputs` and `target_variables` in the configuration file.
*   **Indexing:** Indexed by `date` (renamed to `time` internally).

### 2. Forecast Data (NOAA GEFS)
*   **Source:** NOAA Global Ensemble Forecast System (GEFS) 35-day forecast.
*   **Location:** Online Zarr store at `https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr`.
*   **Content:** Ensemble weather forecasts (31 members).
*   **Variables:** Variables specified in `forecast_inputs` (e.g., temperature, precipitation).
*   **Indexing:** Indexed by `issue_time` (forecast initialization time), `lead_time` (hours ahead), and `ensemble_member`.

### 3. Static Attributes
*   **Source:** Local CSV files.
*   **Location:** `data/*_attributes.csv`.
*   **Content:** Static catchment attributes (e.g., area, elevation, soil properties).
*   **Format:** CSV files with a `gauge_id` column used as the index.

### 4. Basin Centroids
*   **Source:** Local CSV file.
*   **Location:** `data/basin_centroids/basin_centroids.csv`.
*   **Content:** Geographic coordinates (latitude, longitude) of basin centroids used to extract point forecasts.
*   **Format:** CSV with columns `basin_name`, `lat`, `lon`.

## Preprocessing Pipeline

The `OnlineForecastDataset` implements the following pipeline to prepare data for training and inference:

### Step 1: Initialization & Configuration
*   The dataset initializes by reading the run configuration (`Config`).
*   It determines the set of basins to process.
*   It loads or initializes a feature scaler (standardization) based on the run mode (train/test).

### Step 2: Data Loading (or Cache Retrieval)
*   **Caching:** The system checks for a pre-computed Zarr cache for each basin (`data/zarr_cache/{basin}.zarr`). If found and valid, it loads data directly from the cache to speed up initialization.
*   **Building from Sources:** If no cache exists for a basin, it triggers the build process:
    1.  **Historical Data:** Iterates through each basin, loads the corresponding CSV, filters for requested variables, and converts to an xarray Dataset.
    2.  **Forecast Data:**
        *   Loads basin centroids.
        *   Connects to the NOAA GEFS Zarr store.
        *   Filters the remote dataset for base variables (stripping `_q25`, `_q50`, etc.).
        *   **Extraction:** Fetches forecast time series for the specific lat/lon of each basin centroid (`fetch_forecasts_for_basins`).
        *   **Quartile Computation:** Computes the 25th, 50th, and 75th percentiles across the ensemble members. These are stored as separate variables (e.g., `temp_q25`, `temp_q50`, `temp_q75`).
        *   **Interpolation:** Interpolates the forecast data to hourly resolution (`interpolate_to_hourly`) up to the configured `forecast_seq_length`.
    3.  **Merging:** The historical and forecast datasets are merged into a single xarray Dataset per basin, aligned by basin ID.

### Step 3: Data Alignment & Lookup Table
*   The dataset creates a lookup table to map integer indices to specific (basin, time) samples.
*   It aligns the "hindcast" (historical) period with the "forecast" period based on the `issue_time`.
*   **Optimization:** For validation/testing, it lazily slices the Zarr array to load only the required time range into memory.

### Step 4: Sampling (`__getitem__`)
When a sample is requested:
1.  **Hindcast Sequence (`x_h`):** Extracts a window of historical data ending at the forecast issue time.
2.  **Forecast Sequence (`x_f`):** Extracts the forecast sequence starting from the issue time.
3.  **Static Attributes (`x_s`):** Appends static attributes to the inputs.
4.  **Normalization:** Applies scaling (centering and scaling) to all inputs.
5.  **Formatting:** Returns a dictionary of PyTorch tensors ready for the model (e.g., `x_d_hindcast`, `x_d_forecast`, `y`).

## Data Formats & Structures

### Internal Xarray Structure
The dataset (cached per basin as `data/zarr_cache/{basin}.zarr`) has the following structure:
*   **Dimensions:** `basin`, `time` (historical), `issue_time` (forecast), `lead_time` (forecast).
*   **Coordinates:** `basin` (string), `time` (datetime), `issue_time` (datetime).
*   **Data Variables:**
    *   Historical variables (e.g., `discharge_vol`, `precipitation`).
    *   Forecast variables (e.g., `temperature_2m_q50`, `precipitation_surface_q75`).

### Input File Requirements

**1. Hydromet Timeseries (`hydromet_timeseries_{basin}.csv`)**
```csv
date,precipitation,temperature,discharge_vol,...
2020-01-01 00:00:00,0.5,12.3,1.2,...
2020-01-01 01:00:00,0.0,12.1,1.1,...
...
```

**2. Basin Centroids (`basin_centroids.csv`)**
```csv
basin_name,lat,lon
DE1,51.5,10.5
DE2,51.6,10.6
...
```

**3. Attributes (`*_attributes.csv`)**
```csv
gauge_id,area,elevation,slope,...
DE1,100.5,450,0.05,...
DE2,200.1,500,0.06,...
...
```

## Key Configuration Parameters

The behavior of the pipeline is controlled by the `config.yml` file:
*   `data_dir`: Root directory for local data files.
*   `hindcast_inputs`: List of variables to use from historical data.
*   `forecast_inputs`: List of variables to use from forecast data (including quartile suffixes).
*   `target_variables`: Variables to predict (e.g., `discharge_vol`).
*   `seq_length`: Length of the input historical sequence.
*   `forecast_seq_length`: Length of the forecast sequence to use.
*   `predict_last_n`: Number of time steps to predict.
