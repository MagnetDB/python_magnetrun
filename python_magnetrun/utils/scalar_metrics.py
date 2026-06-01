"""
Primitive scalar distance and error metrics for magnetrun.

These functions are the single implementation used by both
:mod:`python_magnetrun.analysis.metrics` (cross-signal comparison) and
:mod:`python_magnetrun.utils.downsampling_metrics` (reconstruction error).

All functions accept 1-D :class:`~numpy.ndarray` inputs and return a
single ``float``.
"""

from __future__ import annotations

import numpy as np


def calc_euclidean(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Euclidean distance (L2 norm of the difference vector).

    Parameters
    ----------
    actual : numpy.ndarray
        Reference values.
    predicted : numpy.ndarray
        Comparison values.

    Returns
    -------
    float
        ``sqrt(sum((actual - predicted)²))`` [same units as input].

    Notes
    -----
    Unlike :func:`calc_rmse`, this is not normalised by the number of
    samples, so larger arrays always produce a larger value.

    Examples
    --------
    >>> calc_euclidean(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
    5.0
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.sum((actual - predicted) ** 2)))


def calc_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute root-mean-square error.

    Parameters
    ----------
    actual : numpy.ndarray
        Reference values.
    predicted : numpy.ndarray
        Comparison values.

    Returns
    -------
    float
        ``sqrt(mean((actual - predicted)²))`` [same units as input].

    Examples
    --------
    >>> calc_rmse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    0.0
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def calc_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute mean absolute error.

    Parameters
    ----------
    actual : numpy.ndarray
        Reference values.
    predicted : numpy.ndarray
        Comparison values.

    Returns
    -------
    float
        ``mean(|actual - predicted|)`` [same units as input].

    Examples
    --------
    >>> calc_mae(np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.2, 2.9]))
    0.133...
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def calc_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute mean absolute percentage error.

    Parameters
    ----------
    actual : numpy.ndarray
        Reference values (division by zero produces NaN/inf when zero-valued).
    predicted : numpy.ndarray
        Comparison values.

    Returns
    -------
    float
        ``mean(|actual - predicted| / |actual|)``.
        NaN when *actual* contains zeros.

    Examples
    --------
    >>> calc_mape(np.array([100.0, 200.0]), np.array([110.0, 190.0]))
    0.075
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.mean(np.abs((actual - predicted) / actual)))


def calc_max_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute worst-case absolute error (Chebyshev / ∞-norm).

    Parameters
    ----------
    actual : numpy.ndarray
        Reference values.
    predicted : numpy.ndarray
        Comparison values.

    Returns
    -------
    float
        ``max(|actual - predicted|)`` [same units as input].
        Returns 0.0 for empty arrays.

    Examples
    --------
    >>> calc_max_error(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.0, 2.8]))
    0.5
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    diff = np.abs(actual - predicted)
    return float(np.max(diff)) if diff.size else 0.0


def calc_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Pearson correlation coefficient between two arrays.

    Parameters
    ----------
    actual : numpy.ndarray
        Reference values.
    predicted : numpy.ndarray
        Comparison values.

    Returns
    -------
    float
        Pearson correlation coefficient in [-1, 1].
        Returns 0.0 when either array has zero variance.

    Examples
    --------
    >>> a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> calc_correlation(a, a)
    1.0
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    a_diff = actual - np.mean(actual)
    p_diff = predicted - np.mean(predicted)
    numerator = float(np.sum(a_diff * p_diff))
    denominator = float(np.sqrt(np.sum(a_diff**2)) * np.sqrt(np.sum(p_diff**2)))

    if denominator == 0:
        return 0.0
    return numerator / denominator


def calc_mahalanobis(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute the Mahalanobis distance between two 1-D signal distributions.

    Measures how many pooled standard deviations apart the means of the two
    signals are, accounting for their joint spread.  Equivalent to Cohen's d
    using the pooled within-group standard deviation (without the √2 factor).

    For two signals whose values are drawn from the same distribution this
    distance is close to 0.  A value of 1 means the means are one pooled
    standard deviation apart.

    Parameters
    ----------
    actual : numpy.ndarray
        Reference 1-D signal.
    predicted : numpy.ndarray
        Comparison 1-D signal.

    Returns
    -------
    float
        Mahalanobis distance ≥ 0.  Returns 0.0 when both signals are
        identical constants; returns ``inf`` when the pooled variance is
        zero but the means differ.

    Notes
    -----
    The pooled standard deviation is:

    .. math::

        s_p = \\sqrt{\\frac{s_1^2 + s_2^2}{2}}

    and the distance is :math:`d = |\\bar{x}_1 - \\bar{x}_2| / s_p`.

    For multivariate comparison (treating time-series samples as a
    joint distribution) use :func:`calc_mahalanobis_multivariate`.

    Examples
    --------
    >>> a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> calc_mahalanobis(a, b)
    0.0
    >>> b_shifted = b + 2.0
    >>> calc_mahalanobis(a, b_shifted)
    1.414...
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mean_diff = float(np.mean(actual) - np.mean(predicted))
    pooled_var = 0.5 * (float(np.var(actual, ddof=1)) + float(np.var(predicted, ddof=1)))

    if pooled_var < 1e-12:
        return 0.0 if abs(mean_diff) < 1e-12 else float("inf")

    return abs(mean_diff) / float(np.sqrt(pooled_var))


def calc_mahalanobis_multivariate(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Compute the point-wise Mahalanobis distance between paired samples.

    Stacks *actual* and *predicted* as a 2-column matrix, estimates the joint
    covariance, then returns the mean Mahalanobis distance of each row from
    the joint centroid.  Useful when both signals are recorded synchronously
    and you want to capture correlated deviations.

    Uses :func:`numpy.linalg.pinv` for numerical stability when the covariance
    is near-singular.

    Parameters
    ----------
    actual : numpy.ndarray
        Reference 1-D signal of length n.
    predicted : numpy.ndarray
        Comparison 1-D signal of length n.

    Returns
    -------
    float
        Mean point-wise Mahalanobis distance ≥ 0.

    Examples
    --------
    >>> a = np.linspace(0, 1, 100)
    >>> calc_mahalanobis_multivariate(a, a)
    0.0
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    X = np.column_stack([actual, predicted])  # (n, 2)
    mu = X.mean(axis=0)
    S_inv = np.linalg.pinv(np.cov(X.T))
    diff = X - mu
    distances = np.sqrt(np.einsum("ni,ij,nj->n", diff, S_inv, diff))
    return float(np.mean(distances))


__all__ = [
    "calc_euclidean",
    "calc_rmse",
    "calc_mae",
    "calc_mape",
    "calc_max_error",
    "calc_correlation",
    "calc_mahalanobis",
    "calc_mahalanobis_multivariate",
]
