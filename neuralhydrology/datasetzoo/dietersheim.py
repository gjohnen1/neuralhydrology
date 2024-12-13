import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from forecastdataset import ForecastDataset

class Dietersheim(ForecastDataset):
    """
    Dataset class for a specific forecast dataset, inheriting from ForecastBaseDataset.

    This class implements the required methods `_load_basin_data` and `_load_attributes`.
    """
    def __init__(self,
             features,
             target,
             window_size,
             horizon):
        
        # Initialize `BaseDataset` class
        super(Dietersheim, self).__init__(features=features,
                                        target=target,
                                        window_size=window_size,
                                        horizon=horizon)

    def _load_basin_data(self, basin: str) -> xr.Dataset:
        """Load input and output data from preprocessed .p files."""
        preprocessed_dir = data_dir / "timeseries"

        # Load hindcast features
        input_files = sorted(file_path.glob("**/Input*_ptq*"))
        hindcast_dfs = [load_hindcast_features(file_path=file, filter_cols=selected_cols) for file in input_files]
        hindcast_df = pd.concat([hindcast_dfs[0]] + [df[-1:] for df in hindcast_dfs[1:]])
        hindcast_df.index.name = "date"

        # Load forecast features
        forecast_dfs = [load_forecast_features(file_path=file, filter_cols=selected_cols) for file in input_files]
        forecast_df = pd.concat(forecast_dfs)

        # Load targets
        target_files = sorted(file_path.glob(f"**/Output_fcst*"))
        target_dfs = [load_targets(file_path=file) for file in target_files]
        target_df = pd.concat([target_dfs[0]] + [df[-1:] for df in target_dfs[1:]])
        target_df.index.name = "date"

        # Convert to xarray datasets
        hindcast_xr = xr.Dataset.from_dataframe(hindcast_df.astype(np.float32))
        forecast_xr = xr.Dataset.from_dataframe(forecast_df.astype(np.float32))
        target_xr = xr.Dataset.from_dataframe(target_df.astype(np.float32))

        # Combine all data into one dataset
        combined_data = xr.Dataset(
            {**hindcast_xr.data_vars, **forecast_xr.data_vars, **target_xr.data_vars}
        ).assign_coords({"basin": basin})

        # Return the dataset
        return combined_data

    def _load_attributes(self) -> pd.DataFrame:
        """Load catchment attributes"""
        return load_camels_de_attributes(self.cfg.data_dir, basins=self.basins)
    

# Implement data loading from files in separate functions for reusability
def load_hindcast_features(file_path, horizon, filter_cols=None):
    with open(file_path, "r") as fp:
        df = pd.read_csv(filepath_or_buffer=fp, delimiter="\t", index_col="UTC", parse_dates=True)  
        if filter_cols:
            df = df[filter_cols]
                      
        return df[:-horizon]
    
def load_forecast_features(file_path, horizon, filter_cols=None):
    with open(file_path, "r") as fp:
        df = pd.read_csv(filepath_or_buffer=fp, delimiter="\t", index_col="UTC", parse_dates=True)[-horizon:]
        if filter_cols:
            df = df[filter_cols]

        df.index.name = "date"
        df = df.rename(columns=lambda x: f'fcst_{x}')
        issue_date = df.index[0] - pd.Timedelta(days=1)

        # Create the MultiIndex
        lead_times = np.arange(1, horizon+1)
        date_array = [issue_date] * len(lead_times)
        multi_index = pd.MultiIndex.from_arrays([date_array, lead_times], names=('date', 'lead_time'))

        forecast_df = pd.DataFrame(index=multi_index, columns=df.columns)

        # Populate the DataFrame
        for lead_time in lead_times:
            forecast_df.loc[(issue_date, lead_time)] = df.iloc[lead_time-1] 
              
        return forecast_df
    
def load_targets(file_path):
    with open(file_path, "r") as fp:
        df = pd.read_csv(filepath_or_buffer=fp, delimiter="\t", index_col="UTC", parse_dates=True)
        # Goal is to predict the next 10days or next 240 hours
        return df

def load_static_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """
    Load attributes for all basins.

    Parameters
    ----------
    data_dir : Path
        Path to the dataset directory.
    basins : List[str], optional
        If passed, return only attributes for the specified basins. Otherwise, return all attributes.

    Returns
    -------
    pd.DataFrame
        Basin-indexed DataFrame with attributes as columns.
    """
    attributes_file = data_dir / "basin_attributes.csv"
    df = pd.read_csv(attributes_file, index_col="basin")

    if basins:
        df = df.loc[basins]

    return df