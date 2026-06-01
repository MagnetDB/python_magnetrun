"""
Legacy distance helpers for processing pipelines.

The authoritative implementations live in
:mod:`python_magnetrun.utils.scalar_metrics`.  This module re-exports them
and keeps the old names for backward compatibility.

.. deprecated::
    ``calc_mape`` here was misnamed — it computed MAE, not MAPE.
    Use :func:`~python_magnetrun.utils.scalar_metrics.calc_mae` (for MAE)
    or :func:`~python_magnetrun.utils.scalar_metrics.calc_mape` (for true
    MAPE) from :mod:`python_magnetrun.utils.scalar_metrics` instead.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..utils.scalar_metrics import calc_correlation, calc_euclidean, calc_mae


def calc_mape(actual: np.ndarray, predic: np.ndarray) -> float:
    """Mean absolute error (misnamed legacy function).

    .. deprecated::
        This function computes **MAE**, not MAPE, despite its name.
        Use :func:`~python_magnetrun.utils.scalar_metrics.calc_mae` for
        MAE or :func:`~python_magnetrun.utils.scalar_metrics.calc_mape`
        for true MAPE.

    Parameters
    ----------
    actual : numpy.ndarray
        Reference values.
    predic : numpy.ndarray
        Comparison values.

    Returns
    -------
    float
        ``mean(|actual - predic|)`` (MAE, not MAPE).
    """
    warnings.warn(
        "processing.distance.calc_mape computes MAE, not MAPE, and is deprecated. "
        "Use utils.scalar_metrics.calc_mae for MAE "
        "or utils.scalar_metrics.calc_mape for true MAPE.",
        DeprecationWarning,
        stacklevel=2,
    )
    return calc_mae(actual, predic)


__all__ = ["calc_euclidean", "calc_mae", "calc_mape", "calc_correlation"]
