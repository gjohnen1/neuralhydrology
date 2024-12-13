import xarray as xr
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ForecastDataset(Dataset):
    """Dataset for forecast rollout, aligning hindcast and forecast features."""

    def __init__(self, hindcast_data, forecast_data, basins):
        """
        Initialize the ForecastDataset.

        Parameters
        ----------
        hindcast_data : xr.Dataset
            Dataset containing hindcast features, indexed by basin and time.
        forecast_data : xr.Dataset
            Dataset containing forecast features, indexed by basin, time, and lead_time.
        basins : list of str
            List of basin identifiers.
        """
        self.hindcast_data = hindcast_data
        self.forecast_data = forecast_data
        self.basins = basins
        self.time_index = hindcast_data.time

    def __len__(self):
        return len(self.basins) * len(self.time_index)

    def __getitem__(self, idx):
        """
        Retrieve a sample by index.

        Parameters
        ----------
        idx : int
            Index for sampling.

        Returns
        -------
        dict
            Dictionary containing hindcast and forecast data for a basin and time.
        """
        basin_idx = idx // len(self.time_index)
        time_idx = idx % len(self.time_index)

        basin_id = self.basins[basin_idx]
        issue_time = self.time_index[time_idx].values

        hindcast_features = self.hindcast_data.sel(basin=basin_id, time=slice(issue_time - pd.Timedelta("30D"), issue_time))
        forecast_features = self.forecast_data.sel(basin=basin_id, time=issue_time)

        return {
            "hindcast_features": hindcast_features,
            "forecast_features": forecast_features
        }

    @staticmethod
    def create_dataset(hindcast_df, forecast_df):
        """
        Convert pandas DataFrames to xarray Datasets with the required indexing.

        Parameters
        ----------
        hindcast_df : pd.DataFrame
            Hindcast data as a DataFrame with columns ['basin', 'time', 'feature_1', ...].
        forecast_df : pd.DataFrame
            Forecast data as a DataFrame with columns ['basin', 'time', 'lead_time', 'feature_1', ...].

        Returns
        -------
        hindcast_data : xr.Dataset
            Hindcast dataset.
        forecast_data : xr.Dataset
            Forecast dataset.
        """
        hindcast_data = hindcast_df.set_index(['basin', 'time']).to_xarray()
        forecast_data = forecast_df.set_index(['basin', 'time', 'lead_time']).to_xarray()
        return hindcast_data, forecast_data