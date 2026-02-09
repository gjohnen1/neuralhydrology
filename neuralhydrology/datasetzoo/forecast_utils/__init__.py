"""Shared utilities for forecast datasets.

This package contains utilities shared across forecast dataset implementations and loaders:
- QuartileComputer: Compute ensemble quartiles as separate variables
- DataAvailability: Track temporal extent of historical and forecast data
"""

from neuralhydrology.datasetzoo.forecast_utils.data_availability import DataAvailability
from neuralhydrology.datasetzoo.forecast_utils.quartile_computer import QuartileComputer

__all__ = ['DataAvailability', 'QuartileComputer']
