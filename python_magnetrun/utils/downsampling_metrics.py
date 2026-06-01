"""
Downsampling quality metrics for magnetrun data.

Provides :class:`DownsampleMetrics`, :class:`SegmentMetrics`,
:func:`evaluate_downsampling`, :func:`evaluate_downsampling_segments`,
and :func:`benchmark_configs` for objective comparison of downsampling
algorithms.

The evaluation loop is::

    original (n points)
      → downsample_arrays(config)   → (data_ds, time_ds)
      → np.interp back to time grid → data_reconstructed
      → residual = original - data_reconstructed
      → compute quality metrics from residual

Memory measurement uses a 3-tier strategy:

* **Tier 1** (default) — ``tracemalloc``: zero extra dependencies, captures
  Python-heap allocations.  Misses native Rust/C heap (e.g. tsdownsample).
* **Tier 2** — subprocess isolation with ``resource.getrusage``: each call
  runs in a fresh process so the OS RSS counter starts cold.  ~100 ms overhead.
  Captures Python + native heap.  Unix only.
* **Tier 3** — ``memray`` with ``native_traces=True``: intercepts every
  malloc/free at C-library level.  Most accurate; 2–5× runtime overhead;
  Linux/macOS only.  Optional dependency.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time as _time
import tracemalloc
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .downsampling import DownsampleConfig, downsample_arrays
from .scalar_metrics import calc_mae, calc_mape, calc_max_error, calc_rmse

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DownsampleMetrics:
    """Quality metrics for one (data, config) pair.

    Parameters
    ----------
    method : str
        Name of the downsampling algorithm.
    n_original : int
        Number of points in the original signal.
    n_downsampled : int
        Number of points after downsampling.
    compression_ratio : float
        ``n_original / n_downsampled``.
    rmse : float
        Root-mean-square reconstruction error [signal units].
    mae : float
        Mean absolute reconstruction error [signal units].
    max_error : float
        Worst-case point deviation (Chebyshev / ∞-norm) [signal units].
    mape : float
        Mean absolute percentage error (NaN when signal contains zeros).
    hausdorff_distance : float
        Max of directed Hausdorff distances in normalised (t, y) space.
        NaN when scipy is not installed.
    peak_max_error : float
        ``|max(reconstructed) - max(original)| / range(original)``.
    peak_min_error : float
        ``|min(reconstructed) - min(original)| / range(original)``.
    energy_ratio : float
        ``‖reconstructed‖² / ‖original‖²``; ideal value is 1.0.
    elapsed_s : float
        Wall-clock time of the ``downsample_arrays`` call [s].
    peak_memory_bytes : int or None
        Peak Python-heap allocation during the call (tracemalloc, Tier 1).
        Misses native Rust/C heap unless Tier 2/3 is used.
        ``None`` when ``compute_memory=False`` (the default).
    input_memory_bytes : int or None
        Size of ``data + time`` input arrays [bytes].
        ``None`` when ``compute_memory=False``.
    output_memory_bytes : int or None
        Size of ``data_ds + time_ds`` output arrays [bytes].
        ``None`` when ``compute_memory=False``.
    memory_overhead_ratio : float or None
        ``peak_memory_bytes / input_memory_bytes``.
        ``None`` when ``compute_memory=False``.
    """

    method: str
    n_original: int
    n_downsampled: int
    compression_ratio: float
    rmse: float
    mae: float
    max_error: float
    mape: float
    hausdorff_distance: float
    peak_max_error: float
    peak_min_error: float
    energy_ratio: float
    elapsed_s: float
    # Memory fields — None when evaluate_downsampling is called with compute_memory=False
    peak_memory_bytes: int | None = field(default=None)
    input_memory_bytes: int | None = field(default=None)
    output_memory_bytes: int | None = field(default=None)
    memory_overhead_ratio: float | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        """Return all fields as a plain dict."""
        return asdict(self)

    def summary(self) -> str:
        """Return a one-line human-readable string."""
        return (
            f"{self.method}: "
            f"{self.n_original}→{self.n_downsampled} "
            f"(ratio={self.compression_ratio:.1f}x), "
            f"RMSE={self.rmse:.4g}, max_err={self.max_error:.4g}, "
            f"t={self.elapsed_s * 1000:.1f}ms"
        )


@dataclass
class SegmentMetrics:
    """Per-segment quality metrics (plateau vs transition regions).

    Parameters
    ----------
    plateau_rmse : float
        RMSE on plateau (steady-state) samples [signal units].
    plateau_mae : float
        MAE on plateau samples [signal units].
    plateau_fraction : float
        Fraction of samples classified as plateau (0–1).
    transition_rmse : float
        RMSE on transition (ramp) samples [signal units].
    transition_mae : float
        MAE on transition samples [signal units].
    transition_fraction : float
        Fraction of samples classified as transition (0–1).
    """

    plateau_rmse: float
    plateau_mae: float
    plateau_fraction: float
    transition_rmse: float
    transition_mae: float
    transition_fraction: float

    def to_dict(self) -> dict[str, Any]:
        """Return all fields as a plain dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Memory measurement helpers
# ---------------------------------------------------------------------------

_MEASURE_SCRIPT = textwrap.dedent(
    """
    import resource, sys, numpy as np
    from python_magnetrun.utils.downsampling import DownsampleConfig, downsample_arrays

    raw = sys.stdin.buffer.read()
    arr = np.frombuffer(raw, dtype=np.float64).reshape(2, -1)
    time_arr, data_arr = arr[0], arr[1]
    bucket_size = {bucket_size}
    config = DownsampleConfig(n_out={n_out}, method="{method}", bucket_size=bucket_size)

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    downsample_arrays(data_arr, time_arr, config)
    after  = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss unit: KB on Linux, bytes on macOS
    import platform
    scale = 1024 if platform.system() == "Linux" else 1
    print((after - before) * scale)
    """
)


def _measure_peak_rss_subprocess(
    data: np.ndarray,
    time: np.ndarray,
    config: DownsampleConfig,
) -> int:
    """Return peak RSS delta in bytes by running the call in an isolated subprocess.

    Parameters
    ----------
    data : numpy.ndarray
        1-D signal array.
    time : numpy.ndarray
        1-D time array.
    config : DownsampleConfig
        Downsampling configuration.

    Returns
    -------
    int
        Peak RSS delta in bytes.
    """
    script = _MEASURE_SCRIPT.format(
        n_out=config.n_out,
        method=config.method,
        bucket_size=repr(config.bucket_size),
    )
    payload = np.row_stack([time, data]).astype(np.float64).tobytes()
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=payload,
        capture_output=True,
        timeout=60,
    )
    return int(result.stdout.strip() or 0)


def _measure_peak_memray(
    data: np.ndarray,
    time: np.ndarray,
    config: DownsampleConfig,
) -> int:
    """Return peak allocated bytes using memray native tracing.

    Parameters
    ----------
    data : numpy.ndarray
        1-D signal array.
    time : numpy.ndarray
        1-D time array.
    config : DownsampleConfig
        Downsampling configuration.

    Returns
    -------
    int
        Peak allocated bytes.

    Raises
    ------
    ImportError
        When ``memray`` is not installed.
    """
    import memray

    dest_path = "/tmp/_memray_ds.bin"
    destination = memray.FileDestination(path=dest_path, overwrite=True)
    with memray.Tracker(destination, native_traces=True):
        downsample_arrays(data, time, config)
    reader = memray._memreader.FileReader(dest_path)
    return int(reader.metadata.peak_memory)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_downsampling(
    data: np.ndarray,
    time: np.ndarray,
    config: DownsampleConfig,
    *,
    compute_memory: bool = False,
    memory_tier: int = 1,
) -> DownsampleMetrics:
    """Evaluate the quality of one downsampling configuration.

    Parameters
    ----------
    data : numpy.ndarray
        Original 1-D signal (NaN entries are stripped before evaluation).
    time : numpy.ndarray
        Corresponding time axis, same length as *data*.
    config : DownsampleConfig
        Downsampling configuration to evaluate.
    compute_memory : bool, optional
        When ``True``, measure peak memory allocation during the downsample call
        and populate the memory fields of :class:`DownsampleMetrics`.
        Defaults to ``False`` — memory fields are ``None`` and no measurement
        overhead is incurred.
    memory_tier : int, optional
        Memory measurement strategy used when *compute_memory* is ``True``:

        * ``1`` (default) — ``tracemalloc``; zero extra deps; misses native heap.
        * ``2`` — subprocess isolation; ~100 ms overhead; captures native heap.
        * ``3`` — ``memray``; 2–5× overhead; Linux/macOS; optional dep.

    Returns
    -------
    DownsampleMetrics
        Full quality report; memory fields are ``None`` unless *compute_memory*
        is ``True``.
    """
    data = np.asarray(data, dtype=float)
    time = np.asarray(time, dtype=float)

    # Step 1: time the downsample call; optionally profile memory
    peak_mem: int | None = None
    input_memory_bytes: int | None = None
    output_memory_bytes: int | None = None
    memory_overhead_ratio: float | None = None

    if compute_memory:
        input_memory_bytes = data.nbytes + time.nbytes
        if memory_tier == 3:
            try:
                peak_mem = _measure_peak_memray(data, time, config)
            except ImportError:
                peak_mem = 0
            t0 = _time.monotonic()
            data_ds, time_ds = downsample_arrays(data, time, config)
            elapsed = _time.monotonic() - t0
        elif memory_tier == 2:
            peak_mem = _measure_peak_rss_subprocess(data, time, config)
            t0 = _time.monotonic()
            data_ds, time_ds = downsample_arrays(data, time, config)
            elapsed = _time.monotonic() - t0
        else:
            tracemalloc.start()
            t0 = _time.monotonic()
            data_ds, time_ds = downsample_arrays(data, time, config)
            elapsed = _time.monotonic() - t0
            _cur, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        output_memory_bytes = data_ds.nbytes + time_ds.nbytes
        memory_overhead_ratio = peak_mem / (input_memory_bytes or 1)
    else:
        t0 = _time.monotonic()
        data_ds, time_ds = downsample_arrays(data, time, config)
        elapsed = _time.monotonic() - t0

    # Step 2: interpolate back to original time grid (NaN-stripped reference)
    valid = ~np.isnan(time) & ~np.isnan(data)
    time_ref = time[valid]
    data_ref = data[valid]
    data_reconstructed = np.interp(time_ref, time_ds, data_ds)

    # Step 3: reconstruction error
    residual = data_ref - data_reconstructed
    rmse = calc_rmse(data_ref, data_reconstructed)
    mae = calc_mae(data_ref, data_reconstructed)
    max_error = calc_max_error(data_ref, data_reconstructed)
    mape = calc_mape(data_ref, data_reconstructed)

    # Step 4: Hausdorff distance (optional scipy)
    try:
        from scipy.spatial.distance import directed_hausdorff

        span = time_ref[-1] - time_ref[0] if len(time_ref) > 1 else 1.0
        t_norm = (time_ref - time_ref[0]) / (span + 1e-12)
        v_scale = float(np.ptp(data_ref)) or 1.0
        A = np.column_stack([t_norm, data_ref / v_scale])
        B = np.column_stack(
            [
                np.interp(time_ds, time_ref, t_norm),
                data_ds / v_scale,
            ]
        )
        hausdorff = float(
            max(directed_hausdorff(A, B)[0], directed_hausdorff(B, A)[0])
        )
    except ImportError:
        hausdorff = float("nan")

    # Step 5: feature preservation
    sig_range = float(np.ptp(data_ref)) or 1.0
    peak_max_error = abs(float(np.max(data_reconstructed)) - float(np.max(data_ref))) / sig_range
    peak_min_error = abs(float(np.min(data_reconstructed)) - float(np.min(data_ref))) / sig_range
    energy_ratio = float(np.sum(data_reconstructed**2) / (np.sum(data_ref**2) or 1.0))

    return DownsampleMetrics(
        method=config.method,
        n_original=len(data_ref),
        n_downsampled=len(data_ds),
        compression_ratio=len(data_ref) / (len(data_ds) or 1),
        rmse=rmse,
        mae=mae,
        max_error=max_error,
        mape=mape,
        hausdorff_distance=hausdorff,
        peak_max_error=peak_max_error,
        peak_min_error=peak_min_error,
        energy_ratio=energy_ratio,
        elapsed_s=elapsed,
        peak_memory_bytes=peak_mem,
        input_memory_bytes=input_memory_bytes,
        output_memory_bytes=output_memory_bytes,
        memory_overhead_ratio=memory_overhead_ratio,
    )


# ---------------------------------------------------------------------------
# Batch benchmark
# ---------------------------------------------------------------------------


def benchmark_configs(
    data: np.ndarray,
    time: np.ndarray,
    configs: list[DownsampleConfig],
    *,
    compute_memory: bool = False,
    memory_tier: int = 1,
) -> pd.DataFrame:
    """Run :func:`evaluate_downsampling` on each config and return a comparison table.

    Parameters
    ----------
    data : numpy.ndarray
        Original 1-D signal.
    time : numpy.ndarray
        Corresponding time axis.
    configs : list of DownsampleConfig
        Configurations to benchmark.
    compute_memory : bool, optional
        When ``True``, populate memory fields in each row.  Defaults to
        ``False``; memory columns will contain ``None``.
    memory_tier : int, optional
        Memory measurement tier forwarded to each :func:`evaluate_downsampling`
        call (1=tracemalloc, 2=subprocess, 3=memray).  Only used when
        *compute_memory* is ``True``.

    Returns
    -------
    :class:`~pandas.DataFrame`
        One row per config; columns match :class:`DownsampleMetrics` fields.
        Indexed by method name; duplicate method names receive a numeric suffix.
    """
    rows = [
        evaluate_downsampling(
            data, time, cfg,
            compute_memory=compute_memory,
            memory_tier=memory_tier,
        ).to_dict()
        for cfg in configs
    ]
    df = pd.DataFrame(rows)
    # Deduplicate index if the same method appears more than once
    methods = df["method"].tolist()
    seen: dict[str, int] = {}
    unique: list[str] = []
    for m in methods:
        if m in seen:
            seen[m] += 1
            unique.append(f"{m}_{seen[m]}")
        else:
            seen[m] = 0
            unique.append(m)
    df.index = pd.Index(unique, name="method")
    return df.drop(columns=["method"])


# ---------------------------------------------------------------------------
# Segment-aware metrics (Phase 3)
# ---------------------------------------------------------------------------


def evaluate_downsampling_segments(
    data: np.ndarray,
    time: np.ndarray,
    config: DownsampleConfig,
    *,
    threshold: float | None = None,
    window: int = 50,
    compute_memory: bool = False,
    memory_tier: int = 1,
) -> tuple[DownsampleMetrics, SegmentMetrics]:
    """Evaluate downsampling quality split by plateau vs transition regions.

    Parameters
    ----------
    data : numpy.ndarray
        Original 1-D signal.
    time : numpy.ndarray
        Corresponding time axis.
    config : DownsampleConfig
        Downsampling configuration to evaluate.
    threshold : float, optional
        Fixed binarisation threshold for separating plateau from transition.
        When ``None`` the threshold is determined automatically via Otsu's
        method.
    window : int, optional
        Smoothing window passed to ``binarize_signal`` (default: 50).
    memory_tier : int, optional
        Memory measurement tier (1/2/3).

    Returns
    -------
    tuple
        ``(DownsampleMetrics, SegmentMetrics)``
    """
    from python_magnetrun.processing.signal import binarize_signal

    base = evaluate_downsampling(
        data, time, config,
        compute_memory=compute_memory,
        memory_tier=memory_tier,
    )

    # Strip NaN to get the same reference used in base metrics
    valid = ~np.isnan(time) & ~np.isnan(data)
    data_ref = data[valid]
    time_ref = time[valid]
    data_ds, time_ds = downsample_arrays(data, time, config)
    data_reconstructed = np.interp(time_ref, time_ds, data_ds)
    residual = data_ref - data_reconstructed

    binarize_kwargs: dict = {"window": window} if window != 50 else {}
    if threshold is not None:
        binarize_kwargs["method"] = "fixed"
        binarize_kwargs["tolerance"] = threshold
    binary = binarize_signal(data_ref, **binarize_kwargs).astype(bool)

    plateau_mask = binary
    transition_mask = ~binary

    def _segment_metrics(mask: np.ndarray) -> tuple[float, float, float]:
        if not np.any(mask):
            return float("nan"), float("nan"), 0.0
        r = residual[mask]
        return (
            float(np.sqrt(np.mean(r**2))),
            float(np.mean(np.abs(r))),
            float(np.sum(mask) / len(mask)),
        )

    p_rmse, p_mae, p_frac = _segment_metrics(plateau_mask)
    t_rmse, t_mae, t_frac = _segment_metrics(transition_mask)

    seg = SegmentMetrics(
        plateau_rmse=p_rmse,
        plateau_mae=p_mae,
        plateau_fraction=p_frac,
        transition_rmse=t_rmse,
        transition_mae=t_mae,
        transition_fraction=t_frac,
    )
    return base, seg


__all__ = [
    "DownsampleMetrics",
    "SegmentMetrics",
    "evaluate_downsampling",
    "evaluate_downsampling_segments",
    "benchmark_configs",
]
