# Operational Harz Forecast Example

This directory contains a configuration file to test my [neuralhydrology](https://github.com/gjohnen1/neuralhydrology.git) fork, specifically additions made in branch [feature/online-forecast-dataset](https://github.com/gjohnen1/neuralhydrology/tree/feature/online-forecast-dataset). 

## Installation

To run this example, you first need to set up the `neuralhydrology` environment (including PyTorch). Please follow the installation instructions in the [neuralhydrology documentation](https://neuralhydrology.readthedocs.io/en/latest/usage/quickstart.html#prerequisites).

Then, clone this repository and switch to the feature branch:

```bash
git clone https://github.com/gjohnen1/neuralhydrology.git
cd neuralhydrology
git checkout feature/online-forecast-dataset
pip install -e .
```

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

**Note:** If you have been provided with a pre-generated Zarr cache, you can place it directly in `data/harz/zarr_cache/`. This allows you to skip the raw `timeseries/` CSVs and the online connection. **However, the attribute CSV files (e.g., `harz_climatic_attributes.csv`) are still required** because static attributes are loaded separately from the time-series cache.

## Running the Example

1.  **Inspect Data:** Open and run `load_online_forecast_dataset.ipynb` to verify that the data loads correctly and to visualize the historical and forecast time series.
2.  **Train & Evaluate:** Open and run `train_eval_sequential_forecast_lstm.ipynb`. This notebook will:
    *   Train the Sequential Forecast LSTM model.
    *   Evaluate the model on the test set.
    *   Generate interactive plots for forecast analysis.

## Contact

For data access and questions, please contact:
gregor.johnen@uni-due.de
