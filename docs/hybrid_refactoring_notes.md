# Hybrid Subpackage & Package — Redundancy & Refactoring Notes

Findings from a cross-module review of `python_magnetrun/hybrid/` and the
broader package.  Ordered by severity (highest first).

**Sections:**
- [Hybrid subpackage findings](#hybrid-subpackage-findings) — items 1–10
- [Broader package findings](#broader-package-findings) — items 11–16

---

## Hybrid subpackage findings

---

## 1. `RMSFileReader` and `VProcessFileReader` are near-identical classes

**Files:**
- [`hybrid/rms/rms_reader.py`](../python_magnetrun/hybrid/rms/rms_reader.py)
- [`hybrid/vprocess/vprocess_reader.py`](../python_magnetrun/hybrid/vprocess/vprocess_reader.py)

Both classes share the same attributes, the same method names, and the same
logic throughout:

| Member | `RMSFileReader` | `VProcessFileReader` |
|--------|-----------------|----------------------|
| `__init__` attributes | `filepath`, `header_lines`, `variables`, `metadata`, `data`, `endian` | identical |
| `parse_header()` | reads `#`-prefixed lines, dispatches to `_parse_*` | identical pattern |
| `_parse_variables()` | regex `NAME [key:val\|…]`, builds variable list | identical logic |
| `_parse_windows()` | regex `[UTC] start -> end`, `%d/%m/%Y-%H:%M:%S.%f` | identical |
| `_parse_frequency()` | regex `frequency = N Hz` | identical |
| `_parse_data_helper()` | regex for `offset:0x…`, `time:N(B)`, `width:N(B)` | identical |
| `_parse_format()` | regex `format = …` | identical |
| `read_binary_data()` | seek to offset, loop over samples, unpack 8-byte timestamp then float32/uint8 per variable | identical algorithm |
| `read()` / `get_variable_info()` / `get_metadata()` / `print_summary()` | same signatures and structure | same |
| Module helpers | `read_rms_file()` / `get_rms_info()` | `read_vprocess_file()` / `get_vprocess_info()` |
| Variable dataclass | `RMSVariable` (name, var_type, unit, min_val, max_val, display_format, is_analog) | `VProcessVariable` — same fields |

**Minor differences to preserve:**

- `RMSFileReader.parse_header()` — encoding `US-ASCII`; VProcess uses `utf-8`.
- `RMSFileReader.read_binary_data()` — converts timestamps via
  `pd.to_datetime(…, unit='s', utc=True)`; VProcess via
  `datetime.fromtimestamp(ts, tz=UTC)`.
- RMS has an extra `_parse_processed_info()` method; VProcess has `_parse_format()`
  dispatching on `"# vprocess data file"` instead of `"# format"`.
- VProcess `parse_header()` resets `self.variables = []` and `self.metadata = {}`
  at the top; RMS relies on constructor state.

**Recommended fix:** Extract an abstract base class
`_BinaryFileReaderBase(filepath, endian)` with the shared `parse_header`,
`_parse_variables`, `_parse_windows`, `_parse_frequency`, `_parse_data_helper`,
`read_binary_data`, `read`, `get_variable_info`, `get_metadata`, `print_summary`
logic.  Subclass it for RMS and VProcess, overriding only the encoding,
timestamp conversion, and format tag.  The variable dataclasses
(`RMSVariable`, `VProcessVariable`) can become a single `ChannelVariable`
dataclass in the shared module.

---

## 2. UTC→local hour conversion — four independent implementations

The same timezone arithmetic (`UTC hour → Europe/Paris local hour`) is
reimplemented independently in four places:

| Location | Form | Scope |
|---|---|---|
| [`hybrid/hybrid_data.py:489–512`](../python_magnetrun/hybrid/hybrid_data.py#L489) | 21-line inline block inside `read_khz_variable` | filter kHz bin files |
| [`hybrid/hybrid_data.py:747–770`](../python_magnetrun/hybrid/hybrid_data.py#L747) | same block inside `read_rms_variable`, uses `_parse_rms_filename_hour` helper | filter RMS files |
| [`analysis/processing.py:582–585`](../python_magnetrun/analysis/processing.py#L582) | nested `_utc_hour_to_local(utc_h)` inside `load_hybrid_data` | convert hours before calling `HybridRun.getData` |
| [`analysis/loaders.py:678–690`](../python_magnetrun/analysis/loaders.py#L678) | three separate lambdas `_khz_hour`, `_rms_hour`, `_trigger_hour` inside `_discover_hybrid_data` | filter files for file-set discovery |

All four implement the same core expression:

```python
_dt.datetime(year, month, day, utc_h, 0, 0, tzinfo=ZoneInfo("UTC"))
    .astimezone(ZoneInfo("Europe/Paris")).hour
```

**Recommended fix:** Add a single module-level utility in `hybrid/utils.py`:

```python
def utc_hour_to_local(utc_h: int, date_str: str, tz: str = "Europe/Paris") -> int:
    """Convert a UTC hour to a local hour for the given date."""
    import datetime as _dt
    from zoneinfo import ZoneInfo
    d = _dt.date.fromisoformat(date_str)
    return _dt.datetime(d.year, d.month, d.day, utc_h, 0, 0,
                        tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo(tz)).hour
```

Then replace all four sites.  Rename `_parse_rms_filename_hour` to a generic
`_parse_filename_hour` (or add a kHz equivalent) and consolidate into a shared
`_filter_files_by_local_hours(files, date_str, extract_utc_hour)` method on
`HybridData`.

---

## 3. `plot_khz_variable` / `plot_rms_variable` — ~80 % identical body

**File:** [`hybrid/plotting.py`](../python_magnetrun/hybrid/plotting.py)
- `plot_khz_variable` — lines 444–561
- `plot_rms_variable` — lines 564–675

Both follow the same eight-step pipeline:

```
read_data → get_unit → build_ylabel → stash_originals →
apply_outlier_strategy → downsample →
[ax-injection path | resolve_backend → plot_overlay → scatter_outliers → handle_output]
→ return (fig, ax)
```

**Differences:**

| | kHz | RMS |
|---|---|---|
| Read call | `read_khz_variable(…, apply_calib, cnv_dir)` | `read_rms_variable(…, file_idx)` |
| Unit getter | `_get_khz_unit(…)` | `_get_rms_unit(…, info_idx)` |
| Config guard | `load_khz_config()` validated after read | absent |
| Title suffix | none | `"\n[strategy: N outliers]"` appended |
| `downsample` default | `50000` | `None` |
| Extra params | `apply_calib`, `cnv_dir` | `file_idx` |

**Bug in `plot_rms_variable`:** In highlight mode, `orig_data, orig_time` are
stashed on line 629 but then **ignored** — the function re-calls
`hybrid_data.read_rms_variable()` twice more (lines 657–661 and 669–673).
The kHz version correctly uses the stash.  The double re-read wastes I/O and
can produce inconsistent results if the file changes between calls.

**Recommended fix:** Fix the bug first (use `orig_data`/`orig_time` in both
highlight branches).  Then extract the shared pipeline into a private helper:

```python
def _plot_variable_impl(
    hybrid_data, system, variable,
    read_fn,       # callable(system, variable, **kw) -> (data, time)
    get_unit_fn,   # callable(hybrid_data, system, variable) -> str
    title_prefix,
    ax, show, save,
    outlier_result, outlier_strategy,
    downsample, downsample_method, backend,
    **plot_kwargs,
) -> tuple: ...
```

---

## 4. `plot_khz_variables` / `plot_rms_variables` — ~75 % identical body

**File:** [`hybrid/plotting.py`](../python_magnetrun/hybrid/plotting.py)
- `plot_khz_variables` — lines 152–311
- `plot_rms_variables` — lines 314–441

Same guard (n==0 error, n==1 delegate), same per-variable loop, same DataFrame
merge, same `_plot_fn` dispatch, same highlight loop, same axes return.

**Inconsistency:** `plot_rms_variables` accepts no `downsample` parameter and
never applies y-axis unit labels to subplot axes.  `plot_khz_variables` does
both.  RMS callers silently get no downsampling and unlabelled axes.

**Recommended fix:** same `_plot_variables_impl` extraction, parameterised by
`read_fn` and `get_unit_fn`, with `downsample` added to the RMS signature.

---

## 5. `safe_float` defined twice in the same file

**File:** [`hybrid/kHz/fepc_reader.py`](../python_magnetrun/hybrid/kHz/fepc_reader.py)
- Line 298 — nested inside `parse_cfg_file()`
- Line 435 — nested inside a second function

Identical 5-line body (handle French decimal comma → period).

**Fix:** Hoist to module level once, remove both nested definitions.

---

## 6. `_resolve_backend` exact duplicate

**Files:**
- [`hybrid/plotting.py:50`](../python_magnetrun/hybrid/plotting.py#L50)
- [`plotting/timeseries.py:34`](../python_magnetrun/plotting/timeseries.py#L34)

Three identical lines.  Move to `hybrid/plotting.py` or to a shared
`plotting/_utils.py` and import in the other.

---

## 7. Trigger calibration mirrors kHz calibration

**Files:**
- [`hybrid/trigger/trigger_reader.py:705`](../python_magnetrun/hybrid/trigger/trigger_reader.py#L705) — `apply_calibration(data, calib, cnv_dir)`
- [`hybrid/kHz/fepc_reader.py:821`](../python_magnetrun/hybrid/kHz/fepc_reader.py#L821) — `apply_calibration(raw_data, calib_info, cnv_dict)`

Both support the same two-path logic: piecewise via CNV file (`np.interp`) or
linear `a*x + b`.  The trigger version loads the CNV file at call time; the kHz
version pre-loads it into a dict.  Signatures differ enough to prevent a
drop-in merge, but the interpolation logic could be shared in a helper
`_apply_cnv_calibration(data, cnv_path) -> np.ndarray`.

---

## 8. `compute_lag` / `lag_correlation` — duplicated between processing and analysis

**Files:**
- [`processing/correlations.py:12`](../python_magnetrun/processing/correlations.py#L12) and [`:131`](../python_magnetrun/processing/correlations.py#L131)
- [`analysis/synchronization.py:258`](../python_magnetrun/analysis/synchronization.py#L258) and [`:375`](../python_magnetrun/analysis/synchronization.py#L375)

`compute_lag` has the same signature `(tkey, df1_data, df2_data, show, save, debug)`
in both modules; the analysis version is the evolved one (full NumPy docstring,
`logger.debug` instead of `logger.info`, cleaner error path).

`lag_correlation` has an **incompatible dict schema for the `range` key**:
- `processing` expects a tuple `(start, end)` — indexed as `data["range"][0]`
- `analysis` expects a dict `{"start": …, "end": …}` — accessed as `data["range"]["start"]`

Callers must know which module they are using.

**Recommended fix:** Deprecate both functions in `processing/correlations.py`
with shims that call the `analysis.synchronization` equivalents.  Unify the
`range` schema (pick dict) and update callers.

---

## 9. `log_exception` — similar but incompatible signatures

**Files:**
- [`hybrid/utils.py:32`](../python_magnetrun/hybrid/utils.py#L32) — `(message, exception, logger_instance=None, …)` — uses module-level `logger` as fallback
- [`log_utils.py:305`](../python_magnetrun/log_utils.py#L305) — `(logger, message, exception, logger_instance=None, …)` — caller passes the fallback logger

`hybrid/cli.py` callers use the `hybrid/utils.py` form.
`analysis/cli.py` callers use the `log_utils.py` form.
These cannot be silently merged without updating all callers.

**Recommended fix:** Unify on the `log_utils.py` signature (explicit `logger`
arg), update the six call sites in `hybrid/cli.py`, then delete the copy in
`hybrid/utils.py`.

---

## 10. Trigger and VProcess not integrated into `HybridData`

**File:** [`hybrid/hybrid_data.py`](../python_magnetrun/hybrid/hybrid_data.py)

`HybridData` exposes `read_khz_variable` / `read_rms_variable` and matching
`plot_*` methods, but has **no** `read_trigger_variable`, `read_vprocess_variable`,
`plot_trigger_variable`, or `plot_vprocess_variable`.  Trigger and VProcess
readers exist and work independently but are not reachable through the unified
`HybridData` interface.

---

---

## Broader package findings

---

## 11. `format_exception_location` duplicated in `hybrid/utils.py`

**Files:**
- [`hybrid/utils.py:97`](../python_magnetrun/hybrid/utils.py#L97)
- [`log_utils.py:361`](../python_magnetrun/log_utils.py#L361)

Near-identical 15-line body.  `hybrid/cli.py` imports it from `hybrid/utils`;
`cli.py` and `analysis/cli.py` import it from `log_utils`.  No callers outside
`hybrid/` import the `hybrid/utils` copy, so the copy can be deleted and
`hybrid/cli.py` can import from `log_utils` instead (same change as item 9).

---

## 12. `remove_outliers` — same name, incompatible interfaces

**Files:**
- [`outliers.py:518`](../python_magnetrun/outliers.py#L518) — `(data, time, method, threshold, window_size, strategy) -> (data, time, n_outliers)` — backward-compatible numpy array API
- [`processing/hysteresis.py:15`](../python_magnetrun/processing/hysteresis.py#L15) — `(df, x_col, y_col, method, threshold) -> DataFrame` — convenience wrapper for x/y DataFrame data

These are **intentionally different** (different inputs, different outputs).  No
consolidation needed, but each docstring should cross-reference the other to
avoid caller confusion.

---

## 13. `_handle_output` — name collision, different responsibilities

**Files:**
- [`hybrid/plotting.py:94`](../python_magnetrun/hybrid/plotting.py#L94) — `(b, fig, show, save)` — thin backend finalise/save/show wrapper
- [`commands/plot.py:271`](../python_magnetrun/commands/plot.py#L271) — `(fig, args, backend, input_files, fields, backend_name, dpi=300)` — heavy CLI output handler with filename generation

Not a duplication — completely different responsibilities.  The collision is
a readability hazard.  Rename the `commands/plot.py` version to
`_save_or_show_figure` or similar.

---

## 14. `compute_lag` caller uses tuple `range` schema — hidden dependency

**File:** [`analysis/processing.py:1046–1055`](../python_magnetrun/analysis/processing.py#L1046)

`_compute_lag_correlation` builds data dicts with `"range": (0, None)` (tuple)
and calls `compute_lag` from `analysis.synchronization`.  But
`analysis.synchronization.compute_lag` accesses `df1_data["range"]` as a tuple
`(istart, iend)` — consistent.  However, `analysis.synchronization.lag_correlation`
accesses `data["range"]["start"]` (dict).  The two functions in the same module
expect **different schemas for the same key name** — `compute_lag` uses tuple,
`lag_correlation` uses dict.  A caller mixing the two will get a `TypeError` or
wrong results with no clear error message.

**Recommended fix:** Standardise on one schema (dict is more explicit) across
both functions and update all call sites.

---

## 15. `_resolve_backend` exact duplicate (package-level)

Already noted as item 6 within the hybrid subpackage.  Listed here for
completeness: the fix also affects the broader `plotting/` package.

---

## 16. `apply_calibration` — three independent implementations

**Files:**
- [`hybrid/kHz/fepc_reader.py:821`](../python_magnetrun/hybrid/kHz/fepc_reader.py#L821) — `(raw_data, calib_info, cnv_dict)` — pre-loaded CNV dict
- [`hybrid/trigger/trigger_reader.py:705`](../python_magnetrun/hybrid/trigger/trigger_reader.py#L705) — `(data, calib, cnv_dir)` — loads CNV at call time
- `processing/distance.py:24` — deprecated shim forwarding to `utils.scalar_metrics` (different domain, not calibration)

The kHz and trigger versions both implement: try piecewise via CNV → fall back to
linear `a*x + b`.  The CNV loading and dict pre-processing differ.  A shared
`_apply_cnv_calibration(data: np.ndarray, cnv_path: Path) -> np.ndarray` helper
(wrapping `np.interp`) would eliminate the duplicated `np.loadtxt` + `np.interp`
call.

---

## Summary — Recommended action order

| Priority | Item | Effort | Risk |
|---|---|---|---|
| 1 | Fix `plot_rms_variable` highlight-mode double-read bug (item 3) | S | Low |
| 2 | Hoist `safe_float` to module level in `fepc_reader.py` (item 5) | S | Low |
| 3 | Consolidate `_resolve_backend` (items 6, 15) | S | Low |
| 4 | Rename `commands/plot.py:_handle_output` (item 13) | S | Low |
| 5 | Unify UTC→local hour conversion into `utc_hour_to_local` utility (item 2) | M | Low |
| 6 | Unify `log_exception` + `format_exception_location` signatures (items 9, 11) | M | Medium |
| 7 | Standardise `range` schema in `analysis.synchronization` (item 14) | M | Medium |
| 8 | Deprecate `processing.correlations` lag functions (item 8) | M | Low |
| 9 | Share CNV calibration helper across trigger and kHz (items 7, 16) | M | Low |
| 10 | Add cross-references between `remove_outliers` variants (item 12) | S | Low |
| 11 | Extract `_plot_variable_impl` / `_plot_variables_impl` (items 3, 4) | L | Medium |
| 12 | Extract `_BinaryFileReaderBase` for RMS and VProcess (item 1) | L | Medium |
| 13 | Integrate trigger / vprocess into `HybridData` (item 10) | XL | Medium |
