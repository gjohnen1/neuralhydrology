"""Abstract base class for forecast data loaders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

import xarray as xr

from neuralhydrology.utils.config import Config


@dataclass
class ForecastLoaderConfig:
    """Configuration for a forecast loader.

    Parameters
    ----------
    name : str
        Unique identifier for this loader instance (e.g., 'gefs', 'icond2').
    type : str
        Loader type that maps to a registered loader class (e.g., 'gefs', 'icond2', 'perfect_forecast').
    suffix : str
        Variable name suffix to distinguish this forecast source (e.g., '_gefs', '_icond2').
        Empty string means no suffix (for backward compatibility).
    variables : List[str]
        Base variable names to load from this forecast source.
    quartiles : List[float], optional
        Quantile values to compute from ensemble forecasts. Default: [0.25, 0.5, 0.75].
    enabled : bool, optional
        Whether this loader is enabled. Default: True.
    loader_kwargs : dict, optional
        Additional loader-specific configuration passed to loader constructor.
    """
    name: str
    type: str
    suffix: str
    variables: List[str]
    quartiles: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    enabled: bool = True
    loader_kwargs: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Loader name cannot be empty")
        if not self.type:
            raise ValueError("Loader type cannot be empty")
        if not self.variables:
            raise ValueError("Loader must specify at least one variable")


class ForecastLoader(ABC):
    """Abstract base class for forecast data loaders.

    Each loader is responsible for:
    1. Loading raw forecast data from its specific source
    2. Computing ensemble quartiles (if applicable)
    3. Returning standardized xarray dataset with dimensions: (basin, issue_time, lead_time)

    Loaders are registered with the ForecastLoaderRegistry and instantiated by ForecastDataset
    based on configuration.

    Parameters
    ----------
    config : ForecastLoaderConfig
        Loader-specific configuration.
    cfg : Config
        Global run configuration.

    Attributes
    ----------
    config : ForecastLoaderConfig
        Configuration for this loader instance.
    cfg : Config
        Global run configuration.
    logger : logging.Logger
        Logger for this loader.
    """

    def __init__(self, config: ForecastLoaderConfig, cfg: Config):
        """Initialize loader.

        Parameters
        ----------
        config : ForecastLoaderConfig
            Loader-specific configuration.
        cfg : Config
            Global run configuration.
        """
        self.config = config
        self.cfg = cfg
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def load(self, basins: List[str]) -> Optional[xr.Dataset]:
        """Load forecast data for specified basins.

        This method must be implemented by subclasses to load forecast data from their
        specific source (remote API, local files, synthetic generation, etc.).

        The returned dataset must have:
        - Dimensions: (basin, issue_time, lead_time)
        - Coordinates:
          - basin: Basin IDs (string)
          - issue_time: Forecast issue/initialization times (datetime64)
          - lead_time: Forecast lead times in hours (int)
        - Variables: Forecast variables named with the configured suffix
          (e.g., 'temperature_2m_gefs_q50')

        Parameters
        ----------
        basins : List[str]
            Basin IDs to load data for.

        Returns
        -------
        Optional[xr.Dataset]
            Dataset with dimensions (basin, issue_time, lead_time) and variables named
            with the configured suffix. Returns None if data cannot be loaded.
        """
        pass

    @abstractmethod
    def get_horizon_hours(self) -> int:
        """Return maximum forecast horizon in hours.

        Returns
        -------
        int
            Maximum forecast lead time in hours.
        """
        pass

    def add_suffix(self, var_name: str) -> str:
        """Add configured suffix to variable name.

        Parameters
        ----------
        var_name : str
            Base variable name.

        Returns
        -------
        str
            Variable name with suffix added (if suffix is not empty and not already present).
        """
        if self.config.suffix and not var_name.endswith(self.config.suffix):
            return f"{var_name}{self.config.suffix}"
        return var_name

    @property
    def cache_key(self) -> str:
        """Generate unique cache key for this loader configuration.

        This key is used to determine if cached data is still valid. Any change in
        configuration should result in a different cache key.

        Returns
        -------
        str
            8-character hash representing this loader's configuration.
        """
        import hashlib
        config_str = (f"{self.config.name}_{self.config.type}_{self.config.suffix}_"
                     f"{','.join(sorted(self.config.variables))}_"
                     f"{','.join(str(q) for q in sorted(self.config.quartiles))}")
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
