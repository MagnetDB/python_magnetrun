# Hybrid Subpackage Refactoring Plan

## Context

The `hybrid` subpackage is generally well-integrated with the rest of the
package (reuses `utils/timestamps.py`, `utils/downsampling.py`, etc.) but
has internal duplication between `utils.py` and `outliers.py`, mixes concerns
inside `utils.py`, and contains scattered debug print statements.

---

## Phase 1 — Quick Wins (< 2 hours, zero functional risk)

### 1.1 Remove commented debug code

| File | Line | Item |
|------|------|------|
| `hybrid/hybrid_data.py` | ~557 | `# print("parts:", parts)` |

Remove entirely.

### 1.2 Replace `print()` with `logger`

Replace all bare `print()` calls so that `--quiet` / `--debug` flags work
correctly end-to-end.

**`hybrid/hybrid_data.py`**
- Lines ~306–339: `print_summary()` method — convert to `logger.info()` (keep
  as user-facing output only if called explicitly; add `--summary` CLI flag if
  needed)
- Line ~515: `print(f"  file_t0: ...")` — convert to `logger.debug()`

**`hybrid/utils.py`**
- Lines ~415, 430, 440: diagnostic prints inside `binarize_signal()` — convert
  to `logger.debug()`

Pattern:
```python
# before
print(f"  file_t0: {file_t0}")

# after
logger.debug("file_t0: %s", file_t0)
```

---

## Phase 2 — Remove Internal Outlier Duplication (2–3 hours)

`hybrid/utils.py` contains a near-complete reimplementation of outlier logic
that already exists — in a superior form — in `hybrid/outliers.py`.

| Function in utils.py | Lines | Counterpart in outliers.py | Overlap |
|----------------------|-------|---------------------------|---------|
| `remove_outliers()` | ~167–229 | `remove_outliers()` ~472–508 | ~90% |
| `_global_outlier_mask()` | ~232–274 | logic at ~249–318 | ~85% |
| `_rolling_outlier_mask()` | ~277–317 | logic at ~320–377 | ~85% |

`outliers.py` is the canonical version: it has `OutlierDetector`, `OutlierResult`
dataclass, and all algorithm variants (IQR, Z-score, MAD, Grubbs, percentile,
modified Z-score).

### Steps

1. Delete lines ~167–317 from `hybrid/utils.py` (the three duplicated
   functions).
2. Add at the top of `utils.py`:
   ```python
   from .outliers import remove_outliers, OutlierDetector, detect_outliers
   ```
3. Verify no call site relied on the old signature (the `utils.py` version
   only supported IQR with a fixed threshold; update any call site that passed
   `threshold=1.5` explicitly to use `OutlierDetector` or keyword args).
4. Run `pytest tests/` — fix any regressions.

---

## Phase 3 — Centralise Outlier Defaults (1 hour)

The value `1.5` (IQR threshold) is hardcoded in at least:
- `hybrid/utils.py` (being deleted in Phase 2)
- `hybrid/hybrid_data.py:plot_khz_variable()` default arg
- `hybrid/hybrid_data.py:plot_rms_variable()` default arg

### Steps

1. Add a `OUTLIER_DEFAULTS` dict to `hybrid/outliers.py`:
   ```python
   OUTLIER_DEFAULTS: dict[str, float] = {
       "iqr": 1.5,
       "zscore": 3.0,
       "mad": 3.5,
       "modified_zscore": 3.5,
       "percentile": 1.0,
       "grubbs": 0.05,
   }
   ```
2. Update `OutlierDetector.__init__()` to use `OUTLIER_DEFAULTS` when no
   threshold is supplied.
3. Update default arg in `hybrid_data.py` plot methods:
   ```python
   # before
   def plot_khz_variable(..., outlier_threshold=1.5, ...):

   # after
   from .outliers import OUTLIER_DEFAULTS
   def plot_khz_variable(..., outlier_threshold=OUTLIER_DEFAULTS["iqr"], ...):
   ```

---

## Phase 4 — Create `OutlierConfig` Dataclass (3–4 hours)

`outlier_method`, `outlier_threshold`, and `window_size` are passed as three
separate parameters through multiple call chains in `hybrid_data.py`. This
mirrors the problem that `DownsampleConfig` already solved for downsampling.

### Steps

1. Add `OutlierConfig` to `hybrid/outliers.py`:
   ```python
   @dataclass(frozen=True)
   class OutlierConfig:
       method: str = "iqr"
       threshold: float | None = None   # None → use OUTLIER_DEFAULTS[method]
       window_size: int | None = None   # None → global (no rolling window)
       strategy: str = "interpolate"   # remove | nan | interpolate | clip | median
   ```
2. Update `OutlierDetector.__init__()` to accept `OutlierConfig` as an
   alternative to individual kwargs.
3. Update `hybrid_data.py` plot methods to accept `OutlierConfig | None`
   instead of three separate parameters.
4. Update `hybrid/args.py` to build `OutlierConfig` from CLI arguments (similar
   to how `args_to_processing_config()` works in `analysis/args.py`).
5. Export `OutlierConfig` from `hybrid/__init__.py`.
6. Run tests.

---

## Phase 5 — Extract Signal Processing Module (1–2 hours)

`hybrid/utils.py` currently mixes three unrelated concerns:

| Lines | Concern |
|-------|---------|
| ~1–120 | Error/exception formatting, file/date listing |
| ~167–317 | Outlier detection (deleted in Phase 2) |
| ~320–444 | Signal processing (`normalize_signal`, `binarize_signal`, `_otsu_threshold`) |

### Steps

1. Create `hybrid/signal_processing.py`.
2. Move the following from `utils.py` to the new module:
   - `normalize_signal()`
   - `_otsu_threshold()`
   - `binarize_signal()`
3. Add backward-compatible re-exports in `utils.py`:
   ```python
   from .signal_processing import normalize_signal, binarize_signal  # noqa: F401
   ```
   Remove these shims once all call sites are updated.
4. Export the new module's public names from `hybrid/__init__.py`.

---

## Phase 6 — Low Priority Polish (< 2 hours total)

### 6.1 Document cache eviction in `hybrid_run.py`

Add a docstring to `_evict_oldest_cache_entry()` explaining the LRU strategy,
size limits, and when eviction is triggered.

### 6.2 Edge-case guards

- `hybrid_data.py:read_khz_variable()`: add a check after calibration — if the
  returned array is all-NaN, log a warning rather than returning silently.
- RMS/trigger filename parsing: validate the file exists after regex parsing;
  log a warning if the resolved path does not exist.

### 6.3 Standardise error handling

Current mix:
- Some methods raise `ValueError` on missing variables.
- Other methods log a warning and return `None`.

Decision: choose one per layer:
- **Public API methods** (`getData`, `read_khz_variable`, etc.) → raise typed
  exceptions.
- **Internal helpers** → log and return `None`.

---

## Validation Checklist (after each phase)

- [ ] `pytest tests/` passes with no regressions
- [ ] `python -m python_magnetrun.hybrid --help` works
- [ ] A representative kHz plot run produces correct output
- [ ] `--quiet` suppresses all output (no stray prints)
- [ ] `--debug` shows structured log lines

---

## Estimated Effort

| Phase | Description | Effort | Risk |
|-------|-------------|--------|------|
| 1 | Quick wins (prints, dead code) | < 2 h | Very low |
| 2 | Remove outlier duplication | 2–3 h | Low |
| 3 | Centralise outlier defaults | 1 h | Low |
| 4 | OutlierConfig dataclass | 3–4 h | Medium |
| 5 | Extract signal_processing.py | 1–2 h | Low |
| 6 | Polish (docs, guards, error handling) | < 2 h | Very low |
| **Total** | | **~10–14 h** | |

Phases 1–3 can be done and committed independently.
Phases 4–5 should be on a feature branch.
Phase 6 can be interleaved or deferred.
