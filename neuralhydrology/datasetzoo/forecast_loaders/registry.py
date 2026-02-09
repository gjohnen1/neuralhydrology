"""Registry for forecast loader classes."""

from typing import Dict, Type

from neuralhydrology.datasetzoo.forecast_loaders.base import ForecastLoader


class ForecastLoaderRegistry:
    """Registry for forecast loader classes.

    This registry uses the decorator pattern to register loader classes and provides
    a factory method to instantiate loaders by type string.

    Example usage:
        @ForecastLoaderRegistry.register('gefs')
        class GEFSLoader(ForecastLoader):
            ...

        # Later, instantiate by type:
        loader_class = ForecastLoaderRegistry.get_loader('gefs')
        loader = loader_class(config, cfg)
    """

    _loaders: Dict[str, Type[ForecastLoader]] = {}

    @classmethod
    def register(cls, loader_type: str):
        """Decorator to register a loader class.

        Parameters
        ----------
        loader_type : str
            Type identifier for this loader (e.g., 'gefs', 'icond2', 'perfect_forecast').

        Returns
        -------
        Callable
            Decorator function that registers the class.

        Example
        -------
        >>> @ForecastLoaderRegistry.register('my_loader')
        >>> class MyLoader(ForecastLoader):
        >>>     pass
        """
        def decorator(loader_class: Type[ForecastLoader]):
            if not issubclass(loader_class, ForecastLoader):
                raise TypeError(f"{loader_class.__name__} must be a subclass of ForecastLoader")

            if loader_type in cls._loaders:
                raise ValueError(f"Loader type '{loader_type}' is already registered")

            cls._loaders[loader_type] = loader_class
            return loader_class

        return decorator

    @classmethod
    def get_loader(cls, loader_type: str) -> Type[ForecastLoader]:
        """Get loader class by type string.

        Parameters
        ----------
        loader_type : str
            Type identifier for the loader.

        Returns
        -------
        Type[ForecastLoader]
            Loader class for the specified type.

        Raises
        ------
        ValueError
            If loader_type is not registered.
        """
        if loader_type not in cls._loaders:
            available = ', '.join(f"'{k}'" for k in sorted(cls._loaders.keys()))
            raise ValueError(
                f"Unknown loader type: '{loader_type}'. "
                f"Available loaders: {available}"
            )
        return cls._loaders[loader_type]

    @classmethod
    def list_loaders(cls) -> list[str]:
        """Get list of registered loader types.

        Returns
        -------
        list[str]
            Sorted list of registered loader type identifiers.
        """
        return sorted(cls._loaders.keys())
