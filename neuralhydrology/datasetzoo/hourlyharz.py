import sys
from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import xarray
from tqdm import tqdm

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class HourlyHarz(BaseDataset):
    """Data set class for the Harz data.

    For more efficient data loading during model training/evaluating, this dataset class expects the dataset
    in a processed format. Specifically, this dataset class works with per-basin csv files that contain all 
    timeseries data combined.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool 
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding 
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used. 
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset, and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or 
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).

    References
    ----------
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        
        # Initialize `BaseDataset` class
        super(HourlyHarz, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       basin=basin,
                                       additional_features=additional_features,
                                       id_to_int=id_to_int,
                                       scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load timeseries data of one specific basin"""
        return load_hourly_harz_timeseries(data_dir=self.cfg.data_dir, basin=basin)

    def _load_attributes(self) -> pd.DataFrame:
        """Load catchment attributes"""
        return load_hourly_harz_attributes(self.cfg.data_dir, basins=self.basins)


def load_hourly_harz_timeseries(data_dir: Path, basin: str) -> pd.DataFrame:
    """Load the time series data for one basin of the Harz.

    Parameters
    ----------
    data_dir : Path
        Path to the Harz directory. This folder must contain a folder called 'timeseries' containing the 
        per-basin csv files.
    basin : str
        Basin identifier number as string.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the time series data (forcings + discharge) data.
        
    Raises
    ------
    FileNotFoundError
        If no sub-folder called 'timeseries' exists within the root directory of the CAMELS DE dataset.
    """
    preprocessed_dir = data_dir / "timeseries"
    
    # make sure the CAMELS-CL data was already preprocessed and per-basin files exist.
    if not preprocessed_dir.is_dir():
        msg = [
            f"No preprocessed data directory found at {preprocessed_dir}."
        ]
        raise FileNotFoundError("".join(msg))
        
    # load the data for the specific basin into a time-indexed dataframe
    basin_file = preprocessed_dir / f"hydromet_timeseries_{basin}.csv"
    df = pd.read_csv(basin_file, index_col='date', parse_dates=['date'])
    return df


def load_hourly_harz_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """Load Harz attributes

    Parameters
    ----------
    data_dir : Path
        Path to the Harz directory. Assumes that a multiple files ending with *_attributes.csv exist.
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of all basins
        are returned.

    Returns
    -------
    pd.DataFrame
        Basin-indexed DataFrame, containing the attributes as columns.
    """
    attributes_path = data_dir

    if not attributes_path.exists():
        raise FileNotFoundError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('*_attributes.csv')

    # Read-in attributes into one big dataframe
    dfs = []
    for txt_file in txt_files:
        df_temp = pd.read_csv(txt_file, sep=',', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')

        dfs.append(df_temp)

    df = pd.concat(dfs, axis=1)


    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError('Some basins are missing static attributes.')
        df = df.loc[basins]

    return df