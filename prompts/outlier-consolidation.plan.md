# Plan: Consolidate outlier implementations

## Context

Outlier logic currently lives in at least four places:
- `python_magnetrun/hybrid/outliers.py` — canonical module with `OutlierDetector`, `OutlierResult`, `detect_outliers`, etc.
- `python_magnetrun/processing/hysteresis.py::remove_outliers` — reimplements IQR/zscore/MAD from scratch for 2-D scatter/hysteresis DataFrames
- `examples/outliers.py` — 213-line script that hand-rolls rolling-MAD inline using pandas, duplicating what the canonical module already exposes
- `tests/test-anomalies.py` + `tests/test-anomalies-optimized.py` — two near-identical CLI scripts (687 / 661 lines) that are not proper pytest modules

The goal is one canonical location, thin delegators everywhere else, and real tests.

---

## Step 1 — Delete `examples/outliers.py`

The file re-implements rolling-MAD detection inline (no import from `hybrid/outliers.py`).  
Nothing in the project imports it.  
**Action:** `git rm examples/outliers.py`

---

## Step 2 — Thin-delegate `processing/hysteresis.py::remove_outliers`

### Current state (lines 11–128)
```
remove_outliers(df, x_col, y_col, method='iqr', threshold=1.5) -> pd.DataFrame
```
Implements IQR, zscore, MAD, isolation_forest inline (~120 lines).

### Target state (~25 lines)
Replace the body with calls to `detect_outliers()` / `OutlierMethod` from the canonical module.

```python
from python_magnetrun.hybrid.outliers import OutlierMethod, detect_outliers

def remove_outliers(df, x_col, y_col, method="iqr", threshold=1.5):
    """Remove outliers from a hysteresis DataFrame using the canonical detector."""
    _METHOD_MAP = {
        "iqr": OutlierMethod.IQR,
        "zscore": OutlierMethod.ZSCORE,
        "mad": OutlierMethod.MAD,
    }
    if method not in _METHOD_MAP:
        raise ValueError(f"method must be one of {list(_METHOD_MAP)}; got {method!r}")

    mask_x = detect_outliers(df[x_col].values, method=_METHOD_MAP[method], threshold=threshold)
    mask_y = detect_outliers(df[y_col].values, method=_METHOD_MAP[method], threshold=threshold)
    return df[~(mask_x | mask_y)].reset_index(drop=True)
```

**Notes:**
- `isolation_forest` is removed from the supported methods: it was never exposed in the canonical module and the only callers in this codebase are the two test scripts being deleted. Add it to `OutlierMethod` later if needed.
- `remove_outliers_by_x_range`, `remove_low_x_outliers`, `remove_x_region_outliers` are left untouched; they are hysteresis-domain helpers and the canonical module has no equivalent.
- `tests/test_processing.py` imports `remove_outliers` from `hysteresis` — the signature is unchanged, so it will keep working. The only behaviour change: `isolation_forest` method will now raise `ValueError` instead of silently running. Adjust that test if it exercises isolation_forest.

---

## Step 3 — Replace the two script-style anomaly tests with one proper pytest module

### Delete
- `tests/test-anomalies.py`
- `tests/test-anomalies-optimized.py`

Both are CLI tools with argparse, not pytest; they require real TDMS files on disk and are not run by the test suite.

### Create `tests/test_outliers.py`

Proper pytest module using synthetic numpy data (no file I/O). Structure:

```
tests/test_outliers.py
  fixtures:
    clean_series()          — 200-point sine + noise, no outliers
    series_with_outliers()  — same, with 5 injected spikes

  TestOutlierDetector
    test_iqr_removes_spikes
    test_zscore_removes_spikes
    test_mad_removes_spikes
    test_percentile_removes_spikes
    test_rolling_iqr
    test_rolling_zscore
    test_rolling_mad
    test_rolling_percentile
    test_no_false_positives_on_clean_data   (all methods parameterized)

  TestOutlierResult
    test_outlier_ratio
    test_get_clean_mask
    test_apply_to_data_remove
    test_apply_to_data_nan
    test_apply_to_data_interpolate
    test_apply_to_data_clip
    test_summary_returns_string

  TestHelpers
    test_detect_outliers_functional_api
    test_find_outlier_segments
    test_get_outlier_summary
    test_analyze_outliers
```

All imports from `python_magnetrun.hybrid.outliers`.

---

## Files touched

| Action | File |
|--------|------|
| delete | `examples/outliers.py` |
| edit   | `python_magnetrun/processing/hysteresis.py` (lines 1–128 only) |
| delete | `tests/test-anomalies.py` |
| delete | `tests/test-anomalies-optimized.py` |
| create | `tests/test_outliers.py` |

Files **not** touched: `hybrid/utils.py::remove_outliers` (not mentioned), `examples/timeseries-anomaly-detection.py` (separate class, not a duplicate), `python_magnetcooling/` (separate package).

---

## Verification

```bash
source magnetrun-env/bin/activate

# existing tests still green
pytest tests/test_processing.py -v

# new tests green
pytest tests/test_outliers.py -v

# hysteresis delegation smoke-check
python -c "
import pandas as pd, numpy as np
from python_magnetrun.processing.hysteresis import remove_outliers
df = pd.DataFrame({'x': np.linspace(-1,1,100), 'y': np.sin(np.linspace(-1,1,100))})
df.loc[10,'y'] = 99
out = remove_outliers(df, 'x', 'y', method='iqr')
assert len(out) < 100
print('delegation OK, rows:', len(out))
"
```
