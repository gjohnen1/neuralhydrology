"""Forecast loader infrastructure for pluggable forecast sources.

This package provides an abstract interface for loading forecast data from various sources
(NOAA GEFS, local NetCDF files, synthetic forecasts, etc.) and a registry for managing
loader implementations.

The main components are:
- ForecastLoader: Abstract base class that all loaders must implement
- ForecastLoaderConfig: Configuration dataclass for loader instances
- ForecastLoaderRegistry: Registry for discovering and instantiating loaders

Example usage:
    # Define a new loader
    @ForecastLoaderRegistry.register('my_source')
    class MyForecastLoader(ForecastLoader):
        def load(self, basins):
            # Load forecast data
            return dataset

        def get_horizon_hours(self):
            return 240

    # Use in configuration
    config = ForecastLoaderConfig(
        name='my_forecast',
        type='my_source',
        suffix='_my',
        variables=['temp', 'precip']
    )

    loader = ForecastLoaderRegistry.get_loader('my_source')(config, cfg)
    data = loader.load(['basin1', 'basin2'])
"""

from neuralhydrology.datasetzoo.forecast_loaders.base import ForecastLoader, ForecastLoaderConfig
from neuralhydrology.datasetzoo.forecast_loaders.registry import ForecastLoaderRegistry

# Import concrete loaders to trigger registration
from neuralhydrology.datasetzoo.forecast_loaders.gefs_loader import GEFSLoader
from neuralhydrology.datasetzoo.forecast_loaders.icond2_loader import ICOND2Loader
from neuralhydrology.datasetzoo.forecast_loaders.perfect_loader import PerfectForecastLoader

__all__ = [
    'ForecastLoader',
    'ForecastLoaderConfig',
    'ForecastLoaderRegistry',
    'GEFSLoader',
    'ICOND2Loader',
    'PerfectForecastLoader',
]
