# Downsampling Quality Metrics Plan

Date: 2026-05-28

Effort key: **S** = ~1 h, **M** = half-day, **L** = 1–2 days.

---

## Motivation

After adding M4 and RDP/VW (see `m4-downsampling.plan.md` and `rdp-downsampling.plan.md`),
there is no systematic way to compare methods objectively.  The question "which algorithm is
best for this dataset at this compression level?" currently has no programmatic answer.

`analysis/metrics.py` has DTW, MAE, MAPE, Pearson for comparing two already-aligned series.
What is missing is the **evaluation loop** specific to downsampling:

```
original (n points)
  → downsample_arrays(config)   → (data_ds, time_ds)   [n_out << n points]
  → np.interp back to time grid → data_reconstructed   [n points again]
  → residual = original - data_reconstructed
  → compute quality metrics from residual
```

This pattern is not the same as comparing two independent signals; the reference is always the
original full-resolution array.  It warrants its own module.

---

## New file: `python_magnetrun/utils/downsampling_metrics.py`

Depends on `numpy`, `scipy` (optional, for Hausdorff), `tracemalloc` (stdlib), and the existing
`utils/downsampling.py`.  No circular imports.

---

## Memory measurement: why simple RSS delta is unreliable

Naïve RSS-before/after (`psutil`, `/proc/self/status`) only produces a non-zero value on the
**first call** to a method.  On subsequent calls the OS pages are already mapped and the kernel
reuses them without touching the RSS counter — the delta is zero.  Experiment on this codebase:

```
stride  RSS deltas over 5 calls: [392, 0, 0, 0, 0] KB
minmax  RSS deltas over 5 calls: [ 64, 0, 0, 0, 0] KB
```

`resource.getrusage(RUSAGE_SELF).ru_maxrss` gives the all-time process peak, not a per-call
figure.  Neither approach is usable for repeatable benchmarking.

The right solutions require either **subprocess isolation** (each run starts with a cold address
space) or **`malloc` interception** (capture every allocation regardless of whether the page
was already mapped).

---

## Memory measurement: 3-tier strategy

### Tier 1 — `tracemalloc` (always on, zero dependencies)

Tracks Python-heap allocations via CPython's memory allocator.  Accurate for pure-Python code
and numpy array creation.  **Misses** native Rust/C heap allocations inside `tsdownsample` and
`simplification`.

```python
import tracemalloc
tracemalloc.start()
data_ds, time_ds = downsample_arrays(data, time, config)
_current, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()
```

Use as the default `peak_memory_bytes` field in `DownsampleMetrics`.

---

### Tier 2 — Subprocess isolation with `resource.getrusage` (accurate, ~100 ms overhead)

Each method is measured in a **fresh subprocess** so the address space starts cold.  The
subprocess imports the minimum needed, runs the downsample call once, reads
`ru_maxrss - baseline`, and prints the result.  The parent collects the output.

```python
import subprocess, sys, textwrap

_MEASURE_SCRIPT = textwrap.dedent("""
    import resource, numpy as np
    from python_magnetrun.utils.downsampling import DownsampleConfig, downsample_arrays

    data = np.frombuffer(__import__('sys').stdin.buffer.read()).reshape(2, -1)
    time_arr, data_arr = data[0], data[1]
    config = DownsampleConfig(n_out={n_out}, method="{method}", epsilon={epsilon})

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # KB on Linux
    downsample_arrays(data_arr, time_arr, config)
    after  = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(after - before)
""")

def _measure_peak_rss_subprocess(
    data: np.ndarray,
    time: np.ndarray,
    config: "DownsampleConfig",
) -> int:
    """Return peak RSS delta in bytes by running the call in an isolated subprocess."""
    script = _MEASURE_SCRIPT.format(
        n_out=config.n_out,
        method=config.method,
        epsilon=config.epsilon or 0.0,
    )
    payload = np.row_stack([time, data]).astype(np.float64).tobytes()
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=payload, capture_output=True, timeout=60,
    )
    kb = int(result.stdout.strip() or 0)
    return kb * 1024   # convert KB → bytes
```

Activated by passing `memory_tier=2` to `evaluate_downsampling`.  Adds ~100 ms per
method — acceptable for a benchmark but not for production use.

**Captures**: Python heap + numpy allocations + native Rust/C heap (via OS page accounting).
**Limitation**: only measures the call's incremental RSS; shared library code pages (e.g.
the tsdownsample `.so`) are counted on first-ever load but not on repeated calls within the
subprocess (though the subprocess is always fresh, so this is a one-time cost per run).

---

### Tier 3 — `memray` with `native_traces=True` (most accurate, optional dependency)

`memray` intercepts every `malloc`/`calloc`/`free` at the C library level via `LD_PRELOAD`-style
hooking.  This captures Rust allocations, numpy buffer allocations, and Python-heap allocations in
a single unified trace.

```python
import memray, io

def _measure_peak_memray(
    data: np.ndarray,
    time: np.ndarray,
    config: "DownsampleConfig",
) -> int:
    """Return peak allocated bytes using memray native tracing."""
    destination = memray.FileDestination(path="/tmp/_memray_ds.bin", overwrite=True)
    with memray.Tracker(destination, native_traces=True):
        downsample_arrays(data, time, config)
    reader = memray._memreader.FileReader("/tmp/_memray_ds.bin")
    return reader.metadata.peak_memory   # bytes
```

`memray` is available on Linux and macOS (not Windows).  Declare as an optional dependency:

```toml
[project.optional-dependencies]
benchmark = ["memray>=1.0", "psutil>=5.9", "scipy>=1.9"]
```

Activated by passing `memory_tier=3` (or auto-detected when `memray` is importable).

**Captures**: all heap allocations regardless of language (Python, C, Rust, Fortran).
**Limitation**: ~2–5× runtime overhead during profiling; Linux/macOS only.

---

### Tier selection in `evaluate_downsampling`

```python
def evaluate_downsampling(
    data: np.ndarray,
    time: np.ndarray,
    config: DownsampleConfig,
    *,
    memory_tier: int = 1,   # 1=tracemalloc, 2=subprocess, 3=memray
) -> DownsampleMetrics:
```

| `memory_tier` | Dependency | Overhead | Captures Rust heap |
|:---:|---|---|:---:|
| 1 | stdlib only | ~0 ms | no |
| 2 | `resource` (stdlib, Unix) | ~100 ms | yes |
| 3 | `memray` (optional) | 2–5× slower | yes |

`benchmark_configs` accepts the same parameter and passes it through to each
`evaluate_downsampling` call.

---

## Result dataclass

```python
@dataclass
class DownsampleMetrics:
    """Quality metrics for one (data, config) pair."""

    method: str
    n_original: int
    n_downsampled: int

    # Compression
    compression_ratio: float        # n_original / n_downsampled

    # Reconstruction error  (interpolated back to original time grid)
    rmse: float                     # root-mean-square error
    mae: float                      # mean absolute error
    max_error: float                # worst-case point deviation (Chebyshev / ∞-norm)
    mape: float                     # mean absolute percentage error (NaN if signal has zeros)

    # Visual fidelity (geometry of the polyline in (t, y) space)
    hausdorff_distance: float       # max of directed Hausdorff distances; NaN if scipy absent

    # Feature preservation
    peak_max_error: float           # |max(ds_interp) - max(orig)| / range(orig)
    peak_min_error: float           # |min(ds_interp) - min(orig)| / range(orig)
    energy_ratio: float             # ||data_reconstructed||² / ||data_original||² (ideal: 1.0)

    # Timing
    elapsed_s: float                # wall-clock time of the downsample_arrays call

    # Memory
    peak_memory_bytes: int          # peak Python-heap allocation during the downsample call
                                    # (tracemalloc; misses native Rust/C heap — see note below)
    input_memory_bytes: int         # data + time input arrays in bytes (reference baseline)
    output_memory_bytes: int        # data_ds + time_ds output arrays in bytes
    memory_overhead_ratio: float    # peak_memory_bytes / input_memory_bytes
                                    # values < 1 mean the algorithm needs less memory than
                                    # the input; values > 1 mean it allocates temporary buffers

    def to_dict(self) -> dict[str, Any]: ...
    def summary(self) -> str: ...   # one-line human-readable string
```

---

## Phase 1 — Core: `evaluate_downsampling` (effort: S)

**`python_magnetrun/utils/downsampling_metrics.py`**

```python
def evaluate_downsampling(
    data: np.ndarray,
    time: np.ndarray,
    config: DownsampleConfig,
) -> DownsampleMetrics:
    """Evaluate the quality of one downsampling configuration.

    Parameters
    ----------
    data:
        Original 1-D signal (NaN-free or NaN-containing; stripping matches
        the behaviour of downsample_arrays for the given method).
    time:
        Corresponding time axis.
    config:
        Downsampling configuration to evaluate.

    Returns
    -------
    DownsampleMetrics
        Full quality report.
    """
```

Implementation steps inside the function:

1. **Time and memory-profile the downsample call** (dispatch by `memory_tier`):

   ```python
   import time as _time

   if memory_tier == 3:
       peak_mem = _measure_peak_memray(data, time, config)       # Tier 3
       t0 = _time.monotonic()
       data_ds, time_ds = downsample_arrays(data, time, config)  # re-run for timing
       elapsed = _time.monotonic() - t0
   elif memory_tier == 2:
       peak_mem = _measure_peak_rss_subprocess(data, time, config)  # Tier 2
       t0 = _time.monotonic()
       data_ds, time_ds = downsample_arrays(data, time, config)
       elapsed = _time.monotonic() - t0
   else:
       import tracemalloc
       tracemalloc.start()
       t0 = _time.monotonic()
       data_ds, time_ds = downsample_arrays(data, time, config)  # Tier 1
       elapsed = _time.monotonic() - t0
       _cur, peak_mem = tracemalloc.get_traced_memory()
       tracemalloc.stop()
   ```

   See the *3-tier strategy* section above for what each tier captures.

1a. **Compute static memory sizes** (deterministic, no measurement noise):
   ```python
   input_memory_bytes  = data.nbytes + time.nbytes
   output_memory_bytes = data_ds.nbytes + time_ds.nbytes
   memory_overhead_ratio = peak_mem / (input_memory_bytes or 1)
   ```

2. **Interpolate back to original time grid** (unchanged):
   ```python
   # Use only the non-NaN original points as the reference
   valid = ~np.isnan(time) & ~np.isnan(data)
   time_ref, data_ref = time[valid], data[valid]
   data_reconstructed = np.interp(time_ref, time_ds, data_ds)
   ```

3. **Reconstruction error** (all operate on `data_ref` vs `data_reconstructed`):
   ```python
   residual = data_ref - data_reconstructed
   rmse = float(np.sqrt(np.mean(residual ** 2)))
   mae  = float(np.mean(np.abs(residual)))
   max_error = float(np.max(np.abs(residual)))
   with np.errstate(divide="ignore", invalid="ignore"):
       mape = float(np.mean(np.abs(residual / data_ref)))
   ```

4. **Hausdorff distance** (optional scipy):
   ```python
   try:
       from scipy.spatial.distance import directed_hausdorff
       # Normalise axes so time and value are on comparable scales
       t_norm = (time_ref - time_ref[0]) / (time_ref[-1] - time_ref[0] + 1e-12)
       v_scale = np.ptp(data_ref) or 1.0
       A = np.column_stack([t_norm, data_ref / v_scale])
       B = np.column_stack([
           np.interp(time_ds, time_ref, t_norm),
           data_ds / v_scale,
       ])
       hausdorff = max(directed_hausdorff(A, B)[0], directed_hausdorff(B, A)[0])
   except ImportError:
       hausdorff = float("nan")
   ```

5. **Feature preservation**:
   ```python
   sig_range = np.ptp(data_ref) or 1.0
   peak_max_error = abs(np.max(data_reconstructed) - np.max(data_ref)) / sig_range
   peak_min_error = abs(np.min(data_reconstructed) - np.min(data_ref)) / sig_range
   energy_ratio   = float(np.sum(data_reconstructed**2) / (np.sum(data_ref**2) or 1.0))
   ```

6. **Assemble and return** `DownsampleMetrics(...)`, including:
   ```python
   peak_memory_bytes     = peak_mem,
   input_memory_bytes    = input_memory_bytes,
   output_memory_bytes   = output_memory_bytes,
   memory_overhead_ratio = memory_overhead_ratio,
   ```

---

## Phase 2 — Batch benchmark: `benchmark_configs` (effort: S)

```python
def benchmark_configs(
    data: np.ndarray,
    time: np.ndarray,
    configs: list[DownsampleConfig],
) -> pd.DataFrame:
    """Run evaluate_downsampling on each config and return a comparison table.

    Returns
    -------
    pd.DataFrame
        One row per config; columns match DownsampleMetrics fields.
        Indexed by method name; duplicate method names get a numeric suffix.
    """
    rows = [evaluate_downsampling(data, time, cfg).to_dict() for cfg in configs]
    return pd.DataFrame(rows).set_index("method")
```

Typical usage:

```python
configs = [
    DownsampleConfig(n_out=1000, method="stride"),
    DownsampleConfig(n_out=1000, method="m4"),
    DownsampleConfig(n_out=1000, method="minmax_lttb"),
    DownsampleConfig(n_out=1000, method="rdp", epsilon=0.01),
]
df = benchmark_configs(data, time, configs)
print(df[["compression_ratio", "rmse", "max_error", "hausdorff_distance", "elapsed_s"]])
```

---

## Phase 3 — Segment-aware metrics (effort: M)

Magnet run signals have distinct **plateau** (constant field) and **transition** (ramp) regions.
A method that performs well on average may still fail on ramps.  Segment-aware metrics split
the residual per region.

**Requires**: `binarize_signal` from `python_magnetrun/processing/signal.py` (already exists).

New function:

```python
@dataclass
class SegmentMetrics:
    plateau_rmse: float
    plateau_mae: float
    plateau_fraction: float     # fraction of samples classified as plateau
    transition_rmse: float
    transition_mae: float
    transition_fraction: float

def evaluate_downsampling_segments(
    data: np.ndarray,
    time: np.ndarray,
    config: DownsampleConfig,
    *,
    threshold: float | None = None,   # binarize_signal threshold; auto-detect if None
    window: int = 50,                  # smoothing window for binarize_signal
) -> tuple[DownsampleMetrics, SegmentMetrics]:
    """Evaluate quality split by plateau vs transition regions."""
```

Implementation: call `evaluate_downsampling` for the base metrics, then use
`binarize_signal(data, ...)` to get a binary mask, split `residual` by mask, and
compute per-segment RMSE/MAE.

---

## Phase 4 — CLI integration (effort: S)

Add `benchmark_downsample` as a function in `analysis/processing.py` (or a new
`commands/benchmark.py`) that:

1. Loads a single channel from a file using the existing loader pipeline.
2. Constructs a default set of `DownsampleConfig` objects covering all installed methods
   at the same `n_out` (derived from `--downsample-params`).
3. Calls `benchmark_configs`.
4. Prints the comparison table to stdout (or saves as CSV / JSON).
5. Optionally plots an overlay of the original signal and each reconstruction.

CLI flag added to `create_downsampling_parser`:

```
--benchmark-downsample
    Run all available downsampling methods on the first loaded channel
    and print a quality comparison table.
```

No new entry point needed — wires into the existing `analysis` CLI.

---

## Phase 5 — Tests (effort: S)

New file **`tests/test_downsampling_metrics.py`**:

| Test | What it checks |
|------|---------------|
| `test_no_downsample_zero_error` | `DownsampleConfig(n_out=len(data))` → RMSE ≈ 0 |
| `test_compression_ratio_correct` | `compression_ratio == n / n_out` |
| `test_elapsed_positive` | `elapsed_s > 0` |
| `test_m4_rmse_le_stride_same_n_out` | M4 RMSE ≤ stride RMSE for sin wave with n_out=200 |
| `test_peak_max_error_zero_exact_pass` | method that keeps exact max → `peak_max_error == 0` |
| `test_hausdorff_finite` | with scipy, Hausdorff is finite and > 0 for stride-reduced signal |
| `test_energy_ratio_near_one` | stride at 50 % → energy ratio in [0.8, 1.2] |
| `test_benchmark_configs_shape` | DataFrame has `len(configs)` rows and expected columns |
| `test_benchmark_configs_best_method` | M4 ranks first by RMSE among {stride, m4, minmax} |
| `test_segment_metrics_sum_to_one` | `plateau_fraction + transition_fraction ≈ 1.0` |
| `test_segment_transition_rmse_higher` | transition RMSE > plateau RMSE for a ramp signal |
| `test_output_memory_bytes_exact` | `output_memory_bytes == data_ds.nbytes + time_ds.nbytes` |
| `test_input_memory_bytes_exact` | `input_memory_bytes == data.nbytes + time.nbytes` |
| `test_memory_overhead_ratio_positive` | `memory_overhead_ratio > 0` for all methods |
| `test_output_bytes_lt_input_bytes` | output array is always smaller than input for `n_out < n` |
| `test_tier2_larger_than_tier1` | subprocess RSS peak ≥ tracemalloc peak for minmax_lttb (Rust allocs visible in Tier 2) |
| `test_tier3_memray_captures_native` | memray peak > tracemalloc peak when `tsdownsample` is used (skipped if memray absent) |

---

## File change summary

| File | Change | Effort |
|------|--------|--------|
| `python_magnetrun/utils/downsampling_metrics.py` | **New** — `DownsampleMetrics` (incl. memory fields), `SegmentMetrics`, `evaluate_downsampling`, `evaluate_downsampling_segments`, `benchmark_configs` | M |
| `python_magnetrun/utils/__init__.py` | Export `DownsampleMetrics`, `evaluate_downsampling`, `benchmark_configs` | S |
| `python_magnetrun/cli_args.py` | Add `--benchmark-downsample` and `--memory-tier {1,2,3}` flags | S |
| `python_magnetrun/analysis/processing.py` | Wire `benchmark_configs` call when `--benchmark-downsample` is set | S |
| `pyproject.toml` | Add `benchmark = ["memray>=1.0", "psutil>=5.9", "scipy>=1.9"]` extras group | S |
| `tests/test_downsampling_metrics.py` | New — 11 test cases | S |

**New extras group** in `pyproject.toml`:
```toml
[project.optional-dependencies]
benchmark = ["memray>=1.0", "psutil>=5.9", "scipy>=1.9"]
```
`tracemalloc` (Tier 1) and `resource` (Tier 2) are stdlib — no extras needed for them.
`memray` (Tier 3) is Linux/macOS only; the code guards with `try/except ImportError`.

**No changes required** in:
- `analysis/metrics.py` — already has cross-signal metrics; downsampling evaluation is a distinct
  concern and should stay in `utils/`
- `magnetdata_pandas.py`, `magnetdata_tdms.py` — metrics operate on numpy arrays, independent of
  the data abstraction layer

---

## Metric selection rationale

| Metric | Why included |
|--------|-------------|
| RMSE | Standard; sensitive to large deviations |
| MAE | Robust to outliers; same units as data |
| Max error | Worst case for plots — one bad spike matters |
| MAPE | Relative error; useful when signal magnitude varies across channels |
| Hausdorff distance | Geometry of the polyline; directly reflects visual deviation |
| `peak_max/min_error` | Magnet runs are characterised by their peak field; preserving it matters |
| Energy ratio | Checks global energy preservation; catches DC-shift artefacts |
| `compression_ratio` | Normalises error comparisons across different `n_out` values |
| `elapsed_s` | RDP binary-search is slower; timing lets users make speed/quality trade-offs |
| `peak_memory_bytes` | Python-heap peak during the call; exposes temporary-buffer cost of each algorithm |
| `input_memory_bytes` | Baseline: cost of holding the full-resolution arrays |
| `output_memory_bytes` | Deterministic: cost of storing the downsampled result |
| `memory_overhead_ratio` | `peak / input`; values > 1 flag algorithms that need more RAM than the data itself |
| Segment RMSE | Plateau errors and transition errors have different physical significance |

---

## Interaction with the overall plan

### Directly enables

- Objective comparison of `m4` vs `nan_m4` vs `rdp` vs `lttb` on real magnet run data.
- Automated regression guard: add a `test_m4_better_than_stride` style assertion to CI once
  the M4 plan is implemented, so that future refactoring cannot silently degrade downsampling
  quality.

### Depends on

- **`m4-downsampling.plan.md` Phase 1** (core M4 branch) — needed to make the M4 benchmark
  row meaningful.
- **`rdp-downsampling.plan.md` Phase 1** — needed for the RDP row; `evaluate_downsampling`
  works with any method without code changes (it calls `downsample_arrays` opaquely).

### Does not conflict with

- Pipeline redesign (polars/narwhals) — metrics operate on numpy arrays extracted from the
  DataFrame, before any narwhals layer is involved.
- Plotting refactoring — benchmark output is a DataFrame; the CLI phase can optionally call
  `downsample_dataframe` + the existing plotting backends for the overlay plot.

---

## Recommended execution order

1. **Phase 1** — `evaluate_downsampling` + `DownsampleMetrics` (~1 h).  Can be written
   and tested before M4/RDP are implemented (it simply exercises `stride` and `minmax`).
2. **Phase 5 (partial)** — tests 1–8 validate Phase 1.
3. **Phase 2** — `benchmark_configs` wrapper (~30 min); tests 9–10.
4. **Phase 3** — segment-aware metrics (after M4 plan complete, to have a meaningful
   reference for "transition RMSE").
5. **Phase 4** — CLI integration (after Phase 2 is stable).
