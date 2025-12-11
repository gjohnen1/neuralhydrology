# Operational Harz Forecast Example

This directory contains example code and configuration to run a sequential forecast LSTM model on the Harz dataset, using the [neuralhydrology](https://github.com/neuralhydrology/neuralhydrology) library.

## Prerequisites

To run this example, you need to set up the `neuralhydrology` environment. Please follow the installation instructions in the [neuralhydrology documentation](https://neuralhydrology.readthedocs.io/en/latest/usage/quickstart.html#prerequisites).

## Data Setup

The actual hydrometeorological data and catchment attributes required to run this example are **not included** in this repository due to licensing/privacy restrictions.

**Example data is available upon request.** Please contact the author (see below) to obtain the necessary dataset.

Once you have obtained the data, please organize it as follows in the `data/harz` directory (relative to the project root):

```text
neuralhydrology/
    data/
        harz/
            timeseries/
                hydromet_timeseries_DE1.csv
                hydromet_timeseries_DE2.csv
                ...
            basin_centroids/
                basin_centroids.csv
            harz_climatic_attributes.csv
            harz_humaninfluence_attributes.csv
            ...
```

### Zarr Cache
The first time you run the code, it will automatically:
1. Load the historical data from the `timeseries/` CSV files.
2. Fetch the latest forecast data from the online NOAA GEFS archive.
3. Merge and cache the data into a Zarr store located at `data/harz/zarr_cache/`.

**Note:** If you have been provided with a pre-generated Zarr cache, you can place it directly in `data/harz/zarr_cache/`. In this case, the raw CSV files and online connection are not strictly required, as the model will load directly from the cache. However, for full reproducibility and to update forecasts, the raw data setup is recommended.

## Running the Example

1.  **Inspect Data:** Open and run `load_online_forecast_dataset.ipynb` to verify that the data loads correctly and to visualize the historical and forecast time series.
2.  **Train & Evaluate:** Open and run `train_eval_sequential_forecast_lstm.ipynb`. This notebook will:
    *   Train the Sequential Forecast LSTM model.
    *   Evaluate the model on the test set.
    *   Generate interactive plots for forecast analysis.

## Contact

For data access and questions, please contact:
gregor.johnen@uni-due.de
