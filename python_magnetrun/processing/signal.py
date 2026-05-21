"""
Signal processing utilities.

Provides general-purpose functions for normalizing and binarizing 1-D signals,
useful for kHz and other time-series data.

Public API
----------
normalize_signal  : scale a signal by its maximum absolute value
binarize_signal   : threshold a signal into a 0/1 binary array
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def normalize_signal(data: np.ndarray) -> np.ndarray:
    """
    Normalize a signal by its maximum absolute value.

    Parameters
    ----------
    data : np.ndarray
        Input signal data.

    Returns
    -------
    np.ndarray
        Signal divided by its maximum absolute value, or unchanged if max is 0.
    """
    max_abs = np.max(np.abs(data))
    if max_abs == 0:
        logger.warning("normalize_signal: all-zero input, returning unchanged")
        return data
    return data / max_abs


def _otsu_threshold(abs_data: np.ndarray, n_bins: int = 256) -> float:
    """Compute Otsu's optimal threshold on absolute-value data.

    Uses log-scale binning so the narrow near-zero off-state population
    gets adequate resolution when the signal dynamic range is large.
    """
    eps = 1e-9
    positive = abs_data[abs_data > eps]
    if positive.size == 0:
        return 0.0

    log_data = np.log10(positive)
    counts, bin_edges = np.histogram(log_data, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = counts.sum()
    if total == 0:
        return 0.0

    weights = counts / total
    cum_w = np.cumsum(weights)
    cum_mean = np.cumsum(weights * bin_centers)
    global_mean = cum_mean[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        between_var = np.where(
            (cum_w > 0) & (cum_w < 1),
            (global_mean * cum_w - cum_mean) ** 2 / (cum_w * (1.0 - cum_w)),
            0.0,
        )

    log_threshold = float(bin_centers[np.argmax(between_var)])
    return float(10**log_threshold)


def binarize_signal(
    data: np.ndarray,
    tolerance: float = 0.005,
    method: str = "otsu",
    n_bins: int = 256,
    normalize: bool = True,
    noise_percentile: float = 40.0,
) -> np.ndarray:
    """
    Return a binary array: 0 (signal off) or 1 (signal on).

    Parameters
    ----------
    data : array-like
        Input signal data (any scale).
    tolerance : float, optional
        Threshold value used only when method='fixed' (default: 0.005).
    method : str, optional
        'fixed'  — use the tolerance value directly.
        'otsu'   — compute the threshold automatically via Otsu's method
                   on the absolute values using log-scale bins (default).
        'noise'  — estimate the noise floor from the bottom
                   `noise_percentile` % of absolute values and use
                   3*sigma of that population as threshold.
    n_bins : int, optional
        Number of histogram bins for Otsu's method (default: 256).
    normalize : bool, optional
        If True (default), normalize the signal by its maximum absolute value
        before applying the threshold, so tolerance is scale-independent.
    noise_percentile : float, optional
        Percentile (0–100) defining the noise population when
        method='noise' (default: 40.0).

    Returns
    -------
    np.ndarray
        Integer array of 0s and 1s.
    """
    data = np.asarray(data, dtype=float)
    if normalize:
        data = normalize_signal(data)
        logger.debug(
            f"data stats: min={data.min():.4g}, max={data.max():.4g}, mean={data.mean():.4g}, std={data.std():.4g}"
        )
    abs_data = np.abs(data)

    if method == "fixed":
        threshold = tolerance
    elif method == "otsu":
        threshold = _otsu_threshold(abs_data, n_bins)
    elif method == "noise":
        noise_ceil = float(np.percentile(abs_data, noise_percentile))
        quiet = abs_data[abs_data <= noise_ceil]
        sigma = float(quiet.std())
        logger.debug(
            f"Noise floor (p{noise_percentile}={noise_ceil:.4g}), 3*sigma={3.0 * sigma:.4g} as threshold."
        )
        threshold = 3.0 * sigma
    else:
        raise ValueError(
            f"Unknown method {method!r}. Choose 'fixed', 'otsu', or 'noise'."
        )
    logger.debug(f"Using threshold: {threshold:.4g} (method={method})")
    return np.where(abs_data <= threshold, 0, 1)
