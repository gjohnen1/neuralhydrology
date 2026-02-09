"""Utility for computing ensemble quartiles as separate variables."""

from typing import Dict, Tuple
import logging

import xarray as xr


LOGGER = logging.getLogger(__name__)


class QuartileComputer:
    """Utility class for computing ensemble quartiles as separate forecast variables.

    This class extracts the duplicated quartile computation logic previously found in
    OnlineForecastDataset and CombinedForecastDataset into a shared utility.

    The computation takes an xarray dataset with an 'ensemble_member' dimension and
    produces a new dataset where each ensemble variable is replaced by separate
    quartile variables (e.g., 'temperature_q25', 'temperature_q50', 'temperature_q75').
    """

    QUARTILE_SUFFIXES = {
        0.25: '_q25',
        0.5: '_q50',
        0.75: '_q75'
    }

    @staticmethod
    def compute_as_variables(forecast_ds: xr.Dataset,
                            quartiles: Tuple[float, ...] = (0.25, 0.5, 0.75),
                            suffix_base: str = "") -> xr.Dataset:
        """Compute ensemble quartiles as standalone forecast variables.

        Takes a dataset with ensemble_member dimension and computes specified quartiles,
        returning a new dataset where each variable is split into separate quartile variables.

        Parameters
        ----------
        forecast_ds : xr.Dataset
            Input dataset with 'ensemble_member' dimension for ensemble variables.
        quartiles : Tuple[float, ...], optional
            Quantile values to compute (between 0 and 1). Default: (0.25, 0.5, 0.75).
        suffix_base : str, optional
            Base suffix to add before quartile suffix (e.g., '_gefs'). This allows
            distinguishing variables from different forecast sources. Default: empty string.

        Returns
        -------
        xr.Dataset
            Dataset with ensemble_member dimension removed and each ensemble variable
            replaced by quartile variables. For example:
            - Input: temperature[ensemble_member=30, ...]
            - Output: temperature_gefs_q25[...], temperature_gefs_q50[...], temperature_gefs_q75[...]

        Notes
        -----
        - Variables without ensemble_member dimension are passed through unchanged
          (with suffix_base added if provided).
        - Quartiles are computed using xarray's quantile method along the ensemble_member dimension.
        - Original dataset attributes are copied to the output.
        """
        LOGGER.info(f"Computing quartiles {quartiles} from ensemble forecasts as separate variables...")

        # Create new dataset with quartile variables
        new_data_vars: Dict[str, xr.DataArray] = {}

        # Process each data variable
        for var_name in forecast_ds.data_vars:
            var_data = forecast_ds[var_name]

            if 'ensemble_member' in var_data.dims:
                # Compute quartiles for ensemble variable
                var_quartiles = var_data.quantile(quartiles, dim='ensemble_member')

                # Create separate variables for each quartile
                for i, q in enumerate(quartiles):
                    # Get suffix for this quartile (e.g., '_q25', '_q50')
                    q_suffix = QuartileComputer.QUARTILE_SUFFIXES.get(q, f'_q{int(q*100)}')

                    # Build full variable name: base_name + suffix_base + quartile_suffix
                    new_var_name = f"{var_name}{suffix_base}{q_suffix}"

                    # Extract the quartile data (remove the quantile dimension)
                    quartile_data = var_quartiles.isel(quantile=i).drop_vars('quantile')

                    # Add to new data variables
                    new_data_vars[new_var_name] = quartile_data
            else:
                # Deterministic variable (no ensemble dimension)
                # Just add suffix_base if provided
                new_var_name = f"{var_name}{suffix_base}" if suffix_base else var_name
                new_data_vars[new_var_name] = var_data

        # Create new dataset with the same coordinates (excluding ensemble_member)
        coords_to_keep = {k: v for k, v in forecast_ds.coords.items()
                         if 'ensemble_member' not in v.dims}

        # Create the new dataset
        quartile_ds = xr.Dataset(
            data_vars=new_data_vars,
            coords=coords_to_keep,
            attrs=forecast_ds.attrs.copy()
        )

        # Update attributes
        quartile_ds.attrs['quartile_processing'] = f'Computed quartiles {quartiles} as separate variables'
        if 'ensemble_member' in forecast_ds.dims:
            quartile_ds.attrs['original_ensemble_members'] = len(forecast_ds.ensemble_member)

        LOGGER.info(f"Created {len(new_data_vars)} quartile variables from {len(forecast_ds.data_vars)} input variables")

        return quartile_ds
