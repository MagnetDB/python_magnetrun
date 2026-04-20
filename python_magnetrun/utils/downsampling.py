"""
Shared downsampling utilities for magnetrun data.

Provides ``DownsampleConfig``, ``downsample_arrays``, and
``downsample_dataframe`` as a single, reusable module that can be imported
by ``HybridRun``, ``PandasMagnetData``, ``TdmsMagnetData``, and the
``analysis/`` pipeline.

Optional dependency: ``tsdownsample`` (declared in the ``hybrid`` extras group
in ``pyproject.toml``).  The package remains fully importable without it; only
the ``minmax_lttb`` and ``lttb`` methods require it at runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from tsdownsample import MinMaxLTTBDownsampler

    HAS_TSDOWNSAMPLE = True
except ImportError:
    HAS_TSDOWNSAMPLE = False
    logger.debug("tsdownsample not available — downsampling will use simple stride")


@dataclass(frozen=True)
class DownsampleConfig:
    """Configuration for a single downsampling operation.

    Parameters
    ----------
    n_out:
        Target number of output points.
    method:
        Algorithm: ``'minmax_lttb'`` | ``'lttb'`` | ``'minmax'`` | ``'stride'``.
        Falls back to ``'stride'`` when ``tsdownsample`` is not installed.
    bucket_size:
        Used by the ``'minmax'`` method only; auto-computed from *n_out* when
        ``None``.
    """

    n_out: int
    method: str = "stride"
    bucket_size: int | None = None

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


def _downsample_indices(
    data: np.ndarray,
    time: np.ndarray,
    config: DownsampleConfig,
) -> np.ndarray:
    """Return an index array selecting *config.n_out* points from *data*/*time*.

    Assumes NaN entries have already been stripped by the caller.
    """
    n = len(data)
    if n <= config.n_out:
        return np.arange(n)

    if config.method in ("minmax_lttb", "lttb"):
        if not HAS_TSDOWNSAMPLE:
            logger.warning(
                "method=%r requires tsdownsample (install with: pip install python_magnetrun[hybrid]); "
                "falling back to 'stride'",
                config.method,
            )
        elif config.method == "minmax_lttb":
            downsampler = MinMaxLTTBDownsampler()
            return downsampler.downsample(time, data, n_out=config.n_out)
        else:
            from tsdownsample import LTTBDownsampler

            downsampler = LTTBDownsampler()
            return downsampler.downsample(time, data, n_out=config.n_out)

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
    valid = ~np.isnan(time) & ~np.isnan(data)
    if not np.all(valid):
        n_nan = int(np.sum(~valid))
        logger.debug("downsample_arrays: stripping %d NaN entries", n_nan)
        data = data[valid]
        time = time[valid]

    if len(data) == 0 or len(data) <= config.n_out:
        return data, time

    logger.info("Downsampling %d → %d using %s", len(data), config.n_out, config.method)
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

    if len(df) <= config.n_out:
        result = df.copy()
        result.attrs["downsample_config"] = config
        return result

    ref_col = value_cols[0] if value_cols else time_col

    # Strip NaN rows on time and reference column
    valid_mask = df[time_col].notna()
    if ref_col != time_col:
        valid_mask &= df[ref_col].notna()
    df_clean = df[valid_mask].reset_index(drop=True)

    if len(df_clean) <= config.n_out:
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
