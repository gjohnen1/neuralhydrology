"""Data availability tracking for forecast datasets."""

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import xarray as xr


@dataclass
class DataAvailability:
    """Track data availability ranges for historical and forecast data.

    This class tracks the temporal extent of historical observations and forecast
    data across all basins in a dataset. It is used to validate that configured
    train/validation/test periods fall within available data ranges.

    Previously this was the private class _DataAvailability in OnlineForecastDataset
    and CombinedForecastDataset. It has been extracted and made public as part of
    the shared forecast utilities.

    Attributes
    ----------
    historical_start : Optional[pd.Timestamp]
        Earliest timestamp in historical observation data.
    historical_end : Optional[pd.Timestamp]
        Latest timestamp in historical observation data.
    forecast_start : Optional[pd.Timestamp]
        Earliest forecast issue time across all forecast sources.
    forecast_end : Optional[pd.Timestamp]
        Latest forecast issue time across all forecast sources.
    """

    historical_start: Optional[pd.Timestamp] = None
    historical_end: Optional[pd.Timestamp] = None
    forecast_start: Optional[pd.Timestamp] = None
    forecast_end: Optional[pd.Timestamp] = None

    def update_from_dataset(self, dataset: Optional[xr.Dataset], dim: str, kind: str) -> None:
        """Update availability from an xarray dataset coordinate.

        Parameters
        ----------
        dataset : Optional[xr.Dataset]
            Dataset to extract availability information from.
        dim : str
            Dimension name to check (e.g., 'time' for historical, 'issue_time' for forecast).
        kind : {'historical', 'forecast'}
            Type of data to update ('historical' for observations, 'forecast' for forecast data).

        Notes
        -----
        This method is typically called multiple times as data is loaded for different basins
        or forecast sources. It maintains the minimum start time and maximum end time across
        all calls.
        """
        if dataset is None or dim not in dataset.coords:
            return

        coord_values = dataset[dim].values
        if coord_values.size == 0:
            return

        start = pd.to_datetime(coord_values[0])
        end = pd.to_datetime(coord_values[-1])

        if kind == 'historical':
            if self.historical_start is None or start < self.historical_start:
                self.historical_start = start
            if self.historical_end is None or end > self.historical_end:
                self.historical_end = end
        elif kind == 'forecast':
            if self.forecast_start is None or start < self.forecast_start:
                self.forecast_start = start
            if self.forecast_end is None or end > self.forecast_end:
                self.forecast_end = end

    def update_from_attrs(self, attrs: Dict[str, str]) -> None:
        """Update availability from dataset attributes.

        This method is used when loading cached data to restore availability information
        that was previously computed and stored in dataset attributes.

        Parameters
        ----------
        attrs : Dict[str, str]
            Dataset attributes dictionary containing availability metadata.

        Notes
        -----
        Recognized attribute keys:
        - 'cache_hist_start', 'historical_data_end': Historical data range
        - 'cache_issue_start', 'cache_issue_end', 'forecast_data_start': Forecast data range
        """
        mapping = {
            'cache_hist_start': 'historical_start',
            'cache_hist_end': 'historical_end',
            'historical_data_end': 'historical_end',
            'cache_issue_start': 'forecast_start',
            'cache_issue_end': 'forecast_end',
            'forecast_data_start': 'forecast_start',
        }

        for attr_key, field in mapping.items():
            value = attrs.get(attr_key)
            if value is None:
                continue

            timestamp = pd.to_datetime(value)
            current = getattr(self, field)

            # Update start times with minimum, end times with maximum
            if field.endswith('start'):
                if current is None or timestamp < current:
                    setattr(self, field, timestamp)
            else:  # end times
                if current is None or timestamp > current:
                    setattr(self, field, timestamp)

    def to_attrs(self) -> Dict[str, str]:
        """Convert availability information to dataset attributes.

        Returns
        -------
        Dict[str, str]
            Dictionary of attributes suitable for storing in xarray dataset attrs.
            Keys follow the convention used in cached datasets for compatibility.

        Notes
        -----
        This method is used when caching datasets to store availability metadata
        for later retrieval.
        """
        attrs: Dict[str, str] = {}

        if self.historical_start is not None:
            attrs['cache_hist_start'] = str(self.historical_start)
        if self.historical_end is not None:
            attrs['cache_hist_end'] = str(self.historical_end)
            attrs['historical_data_end'] = str(self.historical_end)
        if self.forecast_start is not None:
            attrs['cache_issue_start'] = str(self.forecast_start)
            attrs['forecast_data_start'] = str(self.forecast_start)
        if self.forecast_end is not None:
            attrs['cache_issue_end'] = str(self.forecast_end)

        return attrs

    def is_complete(self) -> bool:
        """Check if all availability fields have been populated.

        Returns
        -------
        bool
            True if all four fields (historical_start/end, forecast_start/end) are set.
        """
        return all([
            self.historical_start is not None,
            self.historical_end is not None,
            self.forecast_start is not None,
            self.forecast_end is not None
        ])

    def __repr__(self) -> str:
        """String representation of availability ranges."""
        return (
            f"DataAvailability(\n"
            f"  historical: {self.historical_start} to {self.historical_end}\n"
            f"  forecast:   {self.forecast_start} to {self.forecast_end}\n"
            f")"
        )
