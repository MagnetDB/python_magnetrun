# Hybrid Subpackage Refactoring Plan

## Context

The `hybrid` subpackage is generally well-integrated with the rest of the
package (reuses `utils/timestamps.py`, `utils/downsampling.py`, etc.) but
has internal duplication between `utils.py` and `outliers.py`, mixes concerns
inside `utils.py`, and contains scattered debug print statements.

---

## Phase 1 — Quick Wins ✅ DONE

### 1.1 Remove commented debug code ✅

- `hybrid/hybrid_data.py` ~557: `# print("parts:", parts)` removed.

### 1.2 Replace `print()` with `logger` ✅

- `hybrid/hybrid_data.py` line ~515: `print(f"  file_t0: ...")` → `logger.debug()`
- `hybrid/utils.py` diagnostic prints inside `binarize_signal()` → `logger.debug()`
- `hybrid/hybrid_data.py:print_summary()` — **kept as-is** (user-facing console output, intentional).

---

## Phase 2 — Remove Internal Outlier Duplication ✅ DONE

Deleted the three duplicated functions from `hybrid/utils.py`:
- `remove_outliers()` (~167–229)
- `_global_outlier_mask()` (~232–274)
- `_rolling_outlier_mask()` (~277–317)

`hybrid/utils.py` now re-exports from the canonical location:
```python
from ..outliers import OutlierDetector, detect_outliers, remove_outliers  # noqa: F401
```

---

## Phase 3 — Centralise Outlier Defaults ✅ DONE

`OUTLIER_DEFAULTS` added to `python_magnetrun/outliers.py` (see Phase 4 note):
```python
OUTLIER_DEFAULTS: dict[str, float] = {
    "iqr": 1.5, "zscore": 3.0, "mad": 3.5, "modified_zscore": 3.5,
    "percentile": 1.0, "grubbs": 0.05, "isolation_forest": 0.1,
}
```

`OutlierDetector.__init__()` uses `OUTLIER_DEFAULTS` when no threshold is
supplied. Fixed a latent bug: `threshold or ...` replaced with
`threshold if threshold is not None else ...` so `threshold=0.0` is honoured.

---

## Phase 4 — Create `OutlierConfig` Dataclass ✅ DONE

**Deviation from plan**: `outliers.py` was moved to the top-level package
(`python_magnetrun/outliers.py`) for clarity. `hybrid/outliers.py` is now a
backward-compat shim that re-exports everything.

`OutlierConfig` in `python_magnetrun/outliers.py`:
```python
@dataclass(frozen=True)
class OutlierConfig:
    method: str = "iqr"
    threshold: float | None = None
    window_size: int | None = None
    strategy: str = "interpolate"

    def resolved_threshold(self) -> float:
        if self.threshold is not None:
            return self.threshold
        return OUTLIER_DEFAULTS.get(self.method, 1.5)
```

`hybrid_data.py` plot methods now accept `outlier_config: OutlierConfig | None`
instead of three separate parameters.

`create_outlier_parser()` and `args_to_outlier_config()` placed in
`python_magnetrun/cli_args.py` (canonical home of all `create_*_parser()`
functions), re-exported through `python_magnetrun/args.py` and
`hybrid/args.py`.

---

## Phase 5 — Extract Signal Processing Module ✅ DONE

**Deviation from plan**: instead of `hybrid/signal_processing.py`, functions
were moved to `python_magnetrun/processing/signal.py` (more appropriate,
general-purpose location).

- `normalize_signal`, `_otsu_threshold`, `binarize_signal` → `python_magnetrun/processing/signal.py`
- `hybrid/utils.py` re-exports via `from ..processing.signal import ...  # noqa: F401`
- `hybrid/hybrid_run.py` lazy import updated to `from ..processing.signal import binarize_signal`
- `python_magnetrun/processing/__init__.py` exports `binarize_signal`, `normalize_signal`

**Bonus fix**: `test_loaders.py` was failing due to a missing `merge_data()`
function. Added to `python_magnetrun/analysis/loaders.py`. Test count: 833 → 866.

---

## Phase 6 — Low Priority Polish (TODO)

### 6.1 Document cache eviction in `hybrid_run.py` ✅ DONE

Extracted the inline eviction loop from `_add_to_cache()` into a dedicated
`_evict_oldest_cache_entry()` method with a docstring explaining the LRU
strategy (oldest-by-timestamp), the 1 GB default size limit, and when eviction
is triggered. `_add_to_cache()` now calls it in a loop.

### 6.2 Edge-case guards ✅ DONE

- `read_khz_variable()`: after calibration, `np.all(np.isnan(data))` check logs
  a `WARNING` naming the system, variable, and calibration directory.
- `read_rms_variable()`: each file is checked with `rms_file.exists()` before
  opening; missing files are logged and skipped. If all files are skipped,
  raises `FileNotFoundError` instead of crashing inside `np.concatenate`.

### 6.3 Standardise error handling ✅ DONE

Decision applied:
- **Public API methods** → raise typed exceptions.
- **Internal helpers** → log and return `None` / empty list.

Changes made:
- `load_khz_config()`: now raises `FileNotFoundError` instead of returning `None`; callers no longer need a None-check
- `get_khz_variables()` / `read_khz_variable()`: removed now-dead `if config is None: raise ValueError(...)` guards
- `_build_groups()`: wraps `_build_group_keys()` calls in try/except, logs warning and falls back to empty keys list; fixes root-logger calls (`logging.info` → `logger.debug`)
- `saveData()` in `hybrid_run.py`: guards against a group-key resolving to a dict instead of `(data, time)`, raises `ValueError` with a clear message

---

## Validation Checklist

- [x] `pytest tests/` passes with no regressions (866 passed, 6 skipped)
- [x] `python -m python_magnetrun.hybrid --help` works
- [ ] A representative kHz plot run produces correct output
- [ ] `--quiet` suppresses all output (no stray prints)
- [ ] `--debug` shows structured log lines

---

## Estimated Effort

| Phase | Description | Effort | Risk | Status |
|-------|-------------|--------|------|--------|
| 1 | Quick wins (prints, dead code) | < 2 h | Very low | ✅ Done |
| 2 | Remove outlier duplication | 2–3 h | Low | ✅ Done |
| 3 | Centralise outlier defaults | 1 h | Low | ✅ Done |
| 4 | OutlierConfig dataclass | 3–4 h | Medium | ✅ Done |
| 5 | Extract signal_processing.py | 1–2 h | Low | ✅ Done |
| 6 | Polish (docs, guards, error handling) | < 2 h | Very low | ⬜ TODO |
| **Total** | | **~10–14 h** | | |
