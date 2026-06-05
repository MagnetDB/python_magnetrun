"""
Shared downsampling utilities for magnetrun data.

Provides ``DownsampleConfig``, ``downsample_arrays``, and
``downsample_dataframe`` as a single, reusable module that can be imported
by ``HybridRun``, ``PandasMagnetData``, ``TdmsMagnetData``, and the
``analysis/`` pipeline.

Optional dependency: ``tsdownsample`` (declared in the ``hybrid`` extras group
in ``pyproject.toml``).  The package remains fully importable without it; only
the ``minmax_lttb``, ``lttb``, ``m4``, and ``nan_m4`` methods require it at
runtime.

Optional dependency: ``simplification`` (declared in the ``rdp`` extras group).
The ``rdp`` and ``vw`` methods require it at runtime.
Install with: ``pip install python_magnetrun[rdp]``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from tsdownsample import LTTBDownsampler, M4Downsampler, MinMaxLTTBDownsampler, NaNM4Downsampler

    HAS_TSDOWNSAMPLE = True
except ImportError:
    HAS_TSDOWNSAMPLE = False
    logger.debug("tsdownsample not available — downsampling will use simple stride")

try:
    from simplification.cutil import simplify_coords_idx, simplify_coords_vw_idx

    HAS_SIMPLIFICATION = True
except ImportError:
    HAS_SIMPLIFICATION = False
    logger.debug("simplification not available — rdp/vw methods will use simple stride")


@dataclass(frozen=True)
class DownsampleConfig:
    """Configuration for a single downsampling operation.

    Parameters
    ----------
    n_out : int
        Target number of output points.
    method : str
        Algorithm: ``'minmax_lttb'`` | ``'lttb'`` | ``'minmax'`` | ``'m4'``
        | ``'nan_m4'`` | ``'rdp'`` | ``'vw'`` | ``'stride'``.
        Falls back to ``'stride'`` when the required optional dependency is not
        installed.
    bucket_size : int, optional
        Used by the ``'minmax'`` method only; auto-computed from *n_out* when
        ``None``.
    epsilon : float, optional
        Geometry tolerance for ``'rdp'`` and ``'vw'`` methods.  Larger values
        produce more aggressive simplification.  Required when method is
        ``'rdp'`` or ``'vw'`` unless using
        :meth:`DownsampleConfig.from_n_out_rdp`.
    """

    n_out: int
    method: str = "stride"
    bucket_size: int | None = None
    epsilon: float | None = None

    @classmethod
    def from_percent(
        cls,
        data_len: int,
        percent: float,
        method: str = "stride",
    ) -> DownsampleConfig:
        """Build a config from a percentage of dataset length.

        Bridges the ``analysis/`` percentage model to the config-based model.

        Parameters
        ----------
        data_len:
            Length of the full dataset.
        percent:
            Target size as a percentage of *data_len* (0–100).
        method:
            Downsampling algorithm.
        """
        n_out = max(1, int(data_len * percent / 100.0))
        return cls(n_out=n_out, method=method)

    @classmethod
    def from_n_out_rdp(
        cls,
        data: np.ndarray,
        time: np.ndarray,
        n_out: int,
        method: str = "rdp",
        tol: float = 0.1,
        max_iter: int = 30,
    ) -> DownsampleConfig:
        """Find epsilon such that RDP/VW returns approximately *n_out* points.

        Binary-searches epsilon in ``[eps_lo, eps_hi]`` until the simplified
        output is within ``tol * n_out`` of the target.

        Parameters
        ----------
        data : numpy.ndarray
            1-D data array.
        time : numpy.ndarray
            1-D time array of the same length.
        n_out : int
            Desired output size.
        method : str
            ``'rdp'`` or ``'vw'``.
        tol : float
            Convergence tolerance as a fraction of *n_out* (default 0.10 = 10 %).
        max_iter : int
            Maximum binary-search iterations.

        Returns
        -------
        DownsampleConfig
            Config with the best-effort epsilon found.

        Raises
        ------
        ImportError
            If ``simplification`` is not installed.
        """
        if not HAS_SIMPLIFICATION:
            raise ImportError(
                "simplification package is required for from_n_out_rdp(); "
                "install with: pip install python_magnetrun[rdp]"
            )
        fn = simplify_coords_idx if method == "rdp" else simplify_coords_vw_idx
        coords = np.column_stack([time.astype(float), data.astype(float)])
        eps_lo, eps_hi = 1e-9, float(np.ptp(data))
        for _ in range(max_iter):
            eps = (eps_lo + eps_hi) / 2
            n = len(fn(coords, eps))
            if abs(n - n_out) <= tol * n_out:
                return cls(n_out=n_out, method=method, epsilon=eps)
            if n > n_out:
                eps_lo = eps
            else:
                eps_hi = eps
        return cls(n_out=n_out, method=method, epsilon=(eps_lo + eps_hi) / 2)


def _downsample_indices(
    data: np.ndarray,
    time: np.ndarray,
    config: DownsampleConfig,
) -> np.ndarray:
    """Return an index array selecting *config.n_out* points from *data*/*time*.

    Assumes NaN entries have already been stripped by the caller.
    """
    n = len(data)
    # For rdp/vw, n_out is only a post-filter cap — epsilon drives reduction,
    # so skip the early-exit even when n <= n_out.
    if n <= config.n_out and config.method not in ("rdp", "vw"):
        return np.arange(n)

    if config.method in ("minmax_lttb", "lttb"):
        if not HAS_TSDOWNSAMPLE:
            logger.warning(
                "method=%r requires tsdownsample (install with: pip install python_magnetrun[hybrid]); "
                "falling back to 'stride'",
                config.method,
            )
        elif config.method == "minmax_lttb":
            return MinMaxLTTBDownsampler().downsample(time, data, n_out=config.n_out)
        else:
            return LTTBDownsampler().downsample(time, data, n_out=config.n_out)

    if config.method == "m4":
        if not HAS_TSDOWNSAMPLE:
            logger.warning(
                "method='m4' requires tsdownsample (install with: pip install python_magnetrun[hybrid]); "
                "falling back to 'stride'",
            )
        else:
            return M4Downsampler().downsample(time, data, n_out=config.n_out)

    if config.method in ("rdp", "vw"):
        if not HAS_SIMPLIFICATION:
            logger.warning(
                "method=%r requires simplification (pip install python_magnetrun[rdp]); "
                "falling back to 'stride'",
                config.method,
            )
        else:
            if config.epsilon is None:
                raise ValueError(
                    f"DownsampleConfig.epsilon must be set when method='{config.method}'. "
                    "Use DownsampleConfig(n_out=..., method='rdp', epsilon=0.01) or "
                    "call DownsampleConfig.from_n_out_rdp() to auto-search epsilon."
                )
            coords = np.column_stack([time.astype(float), data.astype(float)])
            fn = simplify_coords_idx if config.method == "rdp" else simplify_coords_vw_idx
            indices = fn(coords, config.epsilon).astype(np.intp)
            if len(indices) > config.n_out:
                indices = indices[: config.n_out]
            return indices

    if config.method == "minmax":
        bucket_size = config.bucket_size or max(1, n // (config.n_out // 2))
        n_buckets = n // bucket_size
        indices: list[int] = []
        for i in range(n_buckets):
            start = i * bucket_size
            end = start + bucket_size
            bucket = data[start:end]
            min_i = int(start + np.argmin(bucket))
            max_i = int(start + np.argmax(bucket))
            if min_i < max_i:
                indices.extend([min_i, max_i])
            else:
                indices.extend([max_i, min_i])
        return np.array(indices[: config.n_out])

    # Default: stride
    stride = max(1, n // config.n_out)
    return np.arange(0, n, stride)[: config.n_out]


def downsample_arrays(
    data: np.ndarray,
    time: np.ndarray,
    config: DownsampleConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a (data, time) pair according to *config*.

    NaN entries are stripped before any algorithm sees them.  Gaps remain
    visible as jumps on the time axis.

    Parameters
    ----------
    data:
        1-D data array.
    time:
        1-D time array of the same length.
    config:
        Downsampling configuration.

    Returns
    -------
    tuple
        ``(downsampled_data, downsampled_time)``
    """
    # NaNM4 handles NaN natively — skip strip so gaps are preserved in output
    if config.method == "nan_m4":
        if not HAS_TSDOWNSAMPLE:
            logger.warning("method='nan_m4' requires tsdownsample; falling back to 'stride'")
            config = DownsampleConfig(n_out=config.n_out, method="stride")
        else:
            if len(data) <= config.n_out:
                return data, time
            indices = NaNM4Downsampler().downsample(time, data, n_out=config.n_out)
            return data[indices], time[indices]

    valid = ~np.isnan(time) & ~np.isnan(data)
    if not np.all(valid):
        n_nan = int(np.sum(~valid))
        logger.debug(f"downsample_arrays: stripping {n_nan} NaN entries")
        data = data[valid]
        time = time[valid]

    if len(data) == 0:
        return data, time
    # For rdp/vw, n_out is only a post-filter cap; allow them to run even
    # when the data length does not exceed n_out.
    if len(data) <= config.n_out and config.method not in ("rdp", "vw"):
        return data, time

    logger.info(f"Downsampling {len(data)} → {config.n_out} using {config.method}")
    indices = _downsample_indices(data, time, config)
    return data[indices], time[indices]


def downsample_dataframe(
    df: pd.DataFrame,
    time_col: str,
    value_cols: list[str],
    config: DownsampleConfig,
) -> pd.DataFrame:
    """Downsample a multi-column DataFrame and return the filtered rows.

    Uses the first entry in *value_cols* (or *time_col* if the list is empty)
    to determine the downsampling indices, then applies those same indices to
    the whole DataFrame so all columns share a consistent time axis.

    NaN rows (in *time_col* or the reference value column) are stripped first.

    Parameters
    ----------
    df:
        Input DataFrame.
    time_col:
        Name of the time column.
    value_cols:
        List of value columns; the first one is used as the reference signal
        for LTTB-based methods.
    config:
        Downsampling configuration.

    Returns
    -------
    pd.DataFrame
        Downsampled DataFrame with the same columns as *df*.
    """
    # Warn if this DataFrame was already downsampled.
    prior: DownsampleConfig | None = df.attrs.get("downsample_config")
    if prior is not None:
        logger.warning(
            "Downsampling already applied to this DataFrame "
            "(method=%r, n_out=%d). Requesting (method=%r, n_out=%d) "
            "on already-reduced data will further reduce fidelity. "
            "Pass the original full-resolution DataFrame instead.",
            prior.method, prior.n_out,
            config.method, config.n_out,
        )

    ref_col = value_cols[0] if value_cols else time_col

    # NaNM4 handles NaN natively — skip the NaN-strip path so gaps are preserved
    if config.method == "nan_m4":
        if not HAS_TSDOWNSAMPLE:
            logger.warning("method='nan_m4' requires tsdownsample; falling back to 'stride'")
            config = DownsampleConfig(n_out=config.n_out, method="stride")
        else:
            if len(df) <= config.n_out:
                result = df.copy()
                result.attrs["downsample_config"] = config
                return result
            time_arr = df[time_col].to_numpy(dtype=float)
            data_arr = df[ref_col].to_numpy(dtype=float)
            indices = NaNM4Downsampler().downsample(time_arr, data_arr, n_out=config.n_out)
            result = df.iloc[indices].reset_index(drop=True)
            result.attrs["downsample_config"] = config
            return result

    if len(df) <= config.n_out and config.method not in ("rdp", "vw"):
        result = df.copy()
        result.attrs["downsample_config"] = config
        return result

    # Strip NaN rows on time and reference column
    valid_mask = df[time_col].notna()
    if ref_col != time_col:
        valid_mask &= df[ref_col].notna()
    df_clean = df[valid_mask].reset_index(drop=True)

    if len(df_clean) <= config.n_out and config.method not in ("rdp", "vw"):
        df_clean = df_clean.copy()
        df_clean.attrs["downsample_config"] = config
        return df_clean

    time_arr = df_clean[time_col].to_numpy(dtype=float)
    data_arr = df_clean[ref_col].to_numpy(dtype=float)

    indices = _downsample_indices(data_arr, time_arr, config)
    result = df_clean.iloc[indices].reset_index(drop=True)
    result.attrs["downsample_config"] = config
    return result


__all__ = ["DownsampleConfig", "downsample_arrays", "downsample_dataframe"]
