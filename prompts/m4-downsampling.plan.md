# M4 Aggregation Downsampling Plan

Date: 2026-05-28

Effort key: **S** = ~1 h, **M** = half-day.

---

## Motivation

The current `DownsampleConfig` supports four methods: `stride`, `minmax`, `minmax_lttb`, `lttb`.
The `minmax` variant keeps only 2 aggregates per bucket (min + max), which can misrepresent data
by losing the temporal order of extrema.  The **M4 algorithm** (Jugel et al., 2014) retains
**4 aggregates per bucket — first, last, min, max** — guaranteeing pixel-perfect visual
fidelity for line charts at any target resolution.

`tsdownsample` already ships `M4Downsampler` and `NaNM4Downsampler`; both return an index array
just like the existing LTTB/MinMaxLTTB downsamplers, so they slot directly into the current
`_downsample_indices` dispatch without any structural change.

---

## Algorithm details

| Method | Points per bucket | NaN handling | Visual fidelity |
|--------|------------------|--------------|-----------------|
| `stride` | 1 | strip first | poor for spiky signals |
| `minmax` | 2 | strip first | preserves envelope only |
| `lttb` | 1 | strip first | good perceptual fidelity |
| `minmax_lttb` | 1 | strip first | best perceptual fidelity |
| **`m4`** (new) | **4** | strip first | **pixel-perfect line chart** |
| **`nan_m4`** (new) | **4** | native — no strip | **pixel-perfect + gap-aware** |

`M4Downsampler().downsample(x, y, n_out=N)` divides the time range into `N/4` equal-width
buckets and returns exactly `N` sorted indices (first, last, argmin, argmax per bucket).

`NaNM4Downsampler` is identical but handles NaN natively — it skips NaN values during bucket
aggregation instead of requiring a clean input array, so time-series gaps remain visible in the
output as actual NaN values rather than being silently elided.

---

## Target design

### Phase 1 — `m4` method (effort: S)

**`python_magnetrun/utils/downsampling.py`**

1. Extend the `tsdownsample` import block to also pull in `M4Downsampler`:

   ```python
   try:
       from tsdownsample import M4Downsampler, MinMaxLTTBDownsampler
       HAS_TSDOWNSAMPLE = True
   except ImportError:
       HAS_TSDOWNSAMPLE = False
   ```

2. Add the `m4` branch to `_downsample_indices` (before the default stride fallback):

   ```python
   if config.method == "m4":
       if not HAS_TSDOWNSAMPLE:
           logger.warning(
               "method='m4' requires tsdownsample; falling back to 'stride'",
           )
       else:
           return M4Downsampler().downsample(time, data, n_out=config.n_out)

   # Default: stride
   ```

3. Update `DownsampleConfig` docstring to list `'m4'` as a valid method:

   ```
   method:
       Algorithm: 'minmax_lttb' | 'lttb' | 'minmax' | 'm4' | 'stride'.
   ```

NaN stripping already happens in `downsample_arrays` before `_downsample_indices` is called, so
`M4Downsampler` always receives clean data — no further changes needed.

---

### Phase 2 — `nan_m4` NaN-aware variant (effort: S)

`NaNM4Downsampler` handles NaN natively; the NaN stripping in `downsample_arrays` must be
bypassed for this method so that gap positions are preserved in the output.

**`python_magnetrun/utils/downsampling.py`**

1. Add `NaNM4Downsampler` to the import:

   ```python
   from tsdownsample import M4Downsampler, MinMaxLTTBDownsampler, NaNM4Downsampler
   ```

2. Add an early-exit path in `downsample_arrays` **before** the NaN-strip block:

   ```python
   def downsample_arrays(data, time, config):
       # NaNM4 handles NaN natively — skip strip so gaps are preserved
       if config.method == "nan_m4":
           if not HAS_TSDOWNSAMPLE:
               logger.warning("method='nan_m4' requires tsdownsample; falling back to 'stride'")
               config = DownsampleConfig(n_out=config.n_out, method="stride")
           else:
               if len(data) <= config.n_out:
                   return data, time
               indices = NaNM4Downsampler().downsample(time, data, n_out=config.n_out)
               return data[indices], time[indices]

       # existing NaN-strip + dispatch path below ...
   ```

3. Similarly short-circuit in `downsample_dataframe` when `config.method == "nan_m4"`:
   skip the `valid_mask` filter; call `NaNM4Downsampler` on the raw reference column;
   apply indices to the unfiltered DataFrame.

---

### Phase 3 — CLI surface (effort: S)

**`python_magnetrun/cli_args.py`**

1. Add `'m4'` and `'nan_m4'` to `DOWNSAMPLE_METHODS`:

   ```python
   DOWNSAMPLE_METHODS = ("none", "stride", "minmax", "minmax_lttb", "lttb", "m4", "nan_m4")
   ```

2. Update the `--downsample-method` help string and `create_downsampling_parser` docstring
   example:

   ```
   --downsample-method m4 --downsample-params '{"n_out": 4000}'
       M4 aggregation: pixel-perfect line chart, 4× more points per bucket than minmax.

   --downsample-method nan_m4
       As above, but preserves NaN gaps instead of stripping them.
   ```

No changes needed in `analysis/args.py` or `analysis/processing.py` — both already accept any
method string via `DownsampleConfig.from_percent(method=...)` and `create_downsampling_parser()`.

---

### Phase 4 — Tests (effort: S)

New file **`tests/test_downsampling.py`** (or extend if it already exists):

| Test | What it checks |
|------|---------------|
| `test_m4_returns_n_out_points` | `downsample_arrays` with method `'m4'` returns exactly `n_out` points |
| `test_m4_indices_are_sorted` | output time array is non-decreasing |
| `test_m4_four_aggregates_per_bucket` | first/last/min/max of a synthetic bucket appear in output |
| `test_m4_no_op_when_data_le_n_out` | when `len(data) <= n_out`, returns original unchanged |
| `test_m4_fallback_stride_no_tsdownsample` | with `HAS_TSDOWNSAMPLE=False` patched, method falls back to stride |
| `test_nan_m4_preserves_nan_in_output` | output of `nan_m4` on NaN-containing input still contains NaN at expected positions |
| `test_nan_m4_dataframe` | `downsample_dataframe` with `nan_m4` returns same number of rows as `n_out` and NaN preserved |
| `test_m4_downsample_dataframe` | DataFrame path produces correct row count with `'m4'` |

---

## File change summary

| File | Change | Effort |
|------|--------|--------|
| `python_magnetrun/utils/downsampling.py` | Import `M4Downsampler` / `NaNM4Downsampler`; add `'m4'` branch in `_downsample_indices`; add `nan_m4` early-exit in `downsample_arrays` and `downsample_dataframe`; update docstrings | S |
| `python_magnetrun/cli_args.py` | Add `'m4'`, `'nan_m4'` to `DOWNSAMPLE_METHODS`; update help text | S |
| `tests/test_downsampling.py` | New — 8 test cases | S |

**No changes required** in:
- `analysis/processing.py` — `DownsampleConfig.from_percent(method=...)` is already method-agnostic
- `analysis/args.py` — inherits updated `create_downsampling_parser()` automatically
- `magnetdata_pandas.py`, `magnetdata_tdms.py` — accept `DownsampleConfig` opaquely
- `hybrid/hybrid_run.py`, `hybrid/data_protocol.py` — same

---

## Interaction with the overall REVIEW.md plan

### Directly extends

- **Item 8 (downsampling refactoring — done)** — adds `m4` and `nan_m4` to the completed
  `utils/downsampling.py` module; no structural changes, only new method branches.

### Compatible with

- **Item 9 (plotting refactoring — done)** — `PlotlyResamplerBackend` uses `DownsampleConfig`
  opaquely; picking `m4` in `ProcessingConfig` automatically flows through to the backend.
- **Item 14 Phase E (`ComparisonSession`)** — `ComparisonSession` will accept one
  `DownsampleConfig` and pass it to all `getData()` calls; `m4`/`nan_m4` work transparently.
- **HoloViews migration** (optional) — datashader handles its own downsampling, but `m4` remains
  the recommended pre-downsampling step for non-datashader backends.

### Does not conflict with

- Pipeline redesign (polars/narwhals) — `_downsample_indices` operates on numpy arrays,
  independent of the upstream DataFrame backend.
- CLI consolidation — the `DOWNSAMPLE_METHODS` constant and `create_downsampling_parser()`
  are shared; adding new choices here is automatically visible to all CLI entry points.

---

## Recommended execution order

1. **Phase 1** — `'m4'` in `_downsample_indices` (~30 min).  Zero risk: new branch, no existing
   code changed.
2. **Phase 4 (partial)** — `test_m4_*` tests to validate Phase 1 immediately.
3. **Phase 2** — `'nan_m4'` early-exit in `downsample_arrays` and `downsample_dataframe`.
4. **Phase 4 (remainder)** — `test_nan_m4_*` tests.
5. **Phase 3** — CLI surface update (one-liner constant + help text).
