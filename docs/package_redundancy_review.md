# Package-wide Redundancy Review

Initial cross-package scan for duplicated, near-duplicate, and confusingly-named
functions/methods.  For the deeper dive into the `hybrid/` subpackage see
[hybrid_refactoring_notes.md](hybrid_refactoring_notes.md).

---

## Confirmed exact duplicates

### `_resolve_backend`

**Files:**
- [`hybrid/plotting.py:50`](../python_magnetrun/hybrid/plotting.py#L50)
- [`plotting/timeseries.py:34`](../python_magnetrun/plotting/timeseries.py#L34)

Identical 3-line body:

```python
def _resolve_backend(backend: str | PlottingBackend) -> PlottingBackend:
    if isinstance(backend, str):
        return get_backend(backend)
    return backend
```

**Fix:** Define once (e.g. in `plotting/_utils.py` or keep in
`plotting/timeseries.py`) and import in `hybrid/plotting.py`.

---

## Near-duplicates — same logic, two modules

### `compute_lag`

**Files:**
- [`processing/correlations.py:131`](../python_magnetrun/processing/correlations.py#L131)
- [`analysis/synchronization.py:258`](../python_magnetrun/analysis/synchronization.py#L258)

Same signature `(tkey, df1_data, df2_data, show, save, debug) -> Timedelta`.
The `analysis` version is the evolved one: full NumPy docstring, `logger.debug`
instead of `logger.info`, cleaner exception path.  The `processing` version has
no docstring.

**Fix:** Deprecate `processing.correlations.compute_lag` with a shim that
delegates to `analysis.synchronization.compute_lag`.

### `lag_correlation`

**Files:**
- [`processing/correlations.py:12`](../python_magnetrun/processing/correlations.py#L12)
- [`analysis/synchronization.py:375`](../python_magnetrun/analysis/synchronization.py#L375)

Same signature `(data1, data2, show, save, debug) -> timedelta`.  Same
cross-correlation algorithm (`scipy.signal.correlate`).

**Critical difference — incompatible dict schema for the `range` key:**

| Module | Access pattern | Expected shape |
|--------|---------------|----------------|
| `processing.correlations` | `data["range"][0]` / `[1]` | tuple `(start, end)` |
| `analysis.synchronization` | `data["range"]["start"]` / `["end"]` | dict |

A caller mixing data prepared for one version with a call to the other will get
a `TypeError` or silently wrong results.  Note: `analysis/processing.py:1046`
builds dicts with `"range": (0, None)` (tuple) — verify which `compute_lag` it
actually calls.

**Fix:** Standardise on the dict schema across both modules; deprecate the
`processing` copy with a shim; update all call sites.

---

## Similar but incompatible signatures

### `log_exception`

**Files:**
- [`hybrid/utils.py:32`](../python_magnetrun/hybrid/utils.py#L32) — `(message, exception, logger_instance=None, use_print=False, include_traceback=True)` — uses module-level `logger` as fallback
- [`log_utils.py:305`](../python_magnetrun/log_utils.py#L305) — `(logger, message, exception, logger_instance=None, use_print=False, include_traceback=True)` — caller supplies the fallback logger

`hybrid/cli.py` imports from `hybrid/utils` and calls the no-logger form.
`analysis/cli.py` imports from `log_utils` and calls the logger-first form.
Cannot be merged without updating callers.

**Fix:** Standardise on the `log_utils` signature (explicit `logger` arg), update
the six call sites in `hybrid/cli.py`, delete `hybrid/utils.log_exception`.

### `format_exception_location`

**Files:**
- [`hybrid/utils.py:97`](../python_magnetrun/hybrid/utils.py#L97)
- [`log_utils.py:361`](../python_magnetrun/log_utils.py#L361)

Near-identical 15-line body.  No callers outside `hybrid/` use the `hybrid/utils`
copy.  Delete it and update `hybrid/cli.py` to import from `log_utils` (same
change as `log_exception`).

---

## Same name, intentionally different interfaces

### `remove_outliers`

**Files:**
- [`outliers.py:518`](../python_magnetrun/outliers.py#L518) — `(data, time, method, threshold, window_size, strategy) -> (data, time, n_outliers)` — backward-compatible numpy array API wrapping `OutlierDetector`
- [`processing/hysteresis.py:15`](../python_magnetrun/processing/hysteresis.py#L15) — `(df, x_col, y_col, method, threshold) -> DataFrame` — convenience wrapper for x/y DataFrame data

Different inputs, different outputs, different use cases.  **Not redundant.**
Each docstring should cross-reference the other to avoid confusion.

### `apply_calibration`

**Files:**
- [`hybrid/kHz/fepc_reader.py:821`](../python_magnetrun/hybrid/kHz/fepc_reader.py#L821) — `(raw_data, calib_info, cnv_dict)` — CNV pre-loaded into dict
- [`hybrid/trigger/trigger_reader.py:705`](../python_magnetrun/hybrid/trigger/trigger_reader.py#L705) — `(data, calib, cnv_dir)` — loads CNV at call time

Both implement linear `a*x + b` with optional piecewise `np.interp` via a CNV
file.  Signatures differ; the shared CNV interpolation step could become a
helper `_apply_cnv_calibration(data, cnv_path) -> np.ndarray`.

---

## Name collisions — different responsibilities

### `_handle_output`

**Files:**
- [`hybrid/plotting.py:94`](../python_magnetrun/hybrid/plotting.py#L94) — `(b, fig, show, save)` — thin backend finalise/save/show wrapper
- [`commands/plot.py:271`](../python_magnetrun/commands/plot.py#L271) — `(fig, args, backend, input_files, fields, backend_name, dpi=300)` — full CLI output handler with filename generation

Not a duplication.  **Fix:** Rename the `commands/plot.py` version to
`_save_or_show_figure` (or similar) to make the different scope obvious.

---

## Legitimate patterns — no action needed

### `stats()`

Appears in `processing/stats.py`, `flow_params.py`, and as abstract + concrete
methods in `magnetdata_base.py`, `magnetdata_pandas.py`, `magnetdata_tdms.py`.
Standard polymorphism via a base class.

### `register()`

Appears in `field_defs.py`, `housing_config.py`, and `plotting/cli.py`.
Standard CLI subparser registration pattern for a modular CLI design.

### `create_argument_parser()` / `create_base_parser()`

Separate CLIs for different domains (`analysis/`, `hybrid/`, core).  Legitimate
separation; no consolidation needed.

---

## Summary table

| Finding | Files | Action |
|---------|-------|--------|
| `_resolve_backend` exact dupe | `hybrid/plotting.py`, `plotting/timeseries.py` | Move to one place, import in the other |
| `compute_lag` near-dupe | `processing/correlations.py`, `analysis/synchronization.py` | Deprecate processing version |
| `lag_correlation` near-dupe + incompatible `range` schema | same two files | Unify schema, deprecate processing version |
| `log_exception` incompatible signatures | `hybrid/utils.py`, `log_utils.py` | Unify on `log_utils` signature, update 6 callers |
| `format_exception_location` near-dupe | `hybrid/utils.py`, `log_utils.py` | Delete `hybrid/utils` copy, update `hybrid/cli.py` |
| `remove_outliers` same name, different interface | `outliers.py`, `processing/hysteresis.py` | Add cross-reference docstrings only |
| `apply_calibration` similar logic, different loading | `hybrid/kHz/fepc_reader.py`, `hybrid/trigger/trigger_reader.py` | Extract shared `_apply_cnv_calibration` helper |
| `_handle_output` name collision | `hybrid/plotting.py`, `commands/plot.py` | Rename CLI version |
