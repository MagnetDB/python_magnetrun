# Update example scripts and panel scripts

## Context

`MagnetData` has been removed from the package. The canonical entry points are now:
- `load_magnetdata(filename)` — dispatch by extension (`.txt`, `.csv`, `.tdms`)
- `MagnetRun.fromtxt / fromtdms / fromcsv / fromStringIO` — unchanged
- Concrete classes: `PandasMagnetData`, `TdmsMagnetData`, `EnsightMagnetData`, etc.
- `MagnetDataBase` — correct type for any polymorphic annotation

The test suite passes. Only external-facing scripts remain to be updated.

---

## Files to update

### 1. `examples/proposal.py`

**Problems:**
- `from ..magnetdata import MagnetData` — `MagnetData` no longer exists
- `def load_record(file: str) -> MagnetData:` — wrong return type
- `isinstance(data, MagnetData)` — wrong isinstance check

**Fix:**
```python
# remove:
from ..magnetdata import MagnetData

# add:
from ..magnetdata_base import MagnetDataBase

# change return annotation:
def load_record(file: str) -> MagnetDataBase:

# change isinstance check:
if data is None:
    raise RuntimeError(f"{file}: cannot load data")
# (MagnetRun.MagnetData is typed as MagnetData | None;
#  after the fix it holds a MagnetDataBase; check for None instead)
```

### 2. `python_magnetrun/panels/panel-mrecord.py`

**Problems:**
- `mrun.MagnetData.getType()` — `self.MagnetData` is the attribute name, not the
  class — this already works, but the `MagnetRun.fromtxt` call signature is wrong
  (only 2 args; should be `housing, site, filename`):

  ```python
  mrun = MagnetRun.fromtxt("M9", "../python_magnetsetup/data/...")
  ```

  should be:

  ```python
  mrun = MagnetRun.fromtxt("M9", "unknown", "../python_magnetsetup/data/...")
  ```

  No import change needed — `mrun.MagnetData` is the instance attribute, not the
  deleted class.

### 3. `python_magnetrun/panels/panel-mrecord-vs-time.py`

Same `fromtxt` call-signature issue as above. Check and fix the argument count.

---

## Files that need NO changes

### `rustfs/`

`rustfs/` (14 Python files across `rustfs/`, `rustfs/magnetfs/`) does **not**
import from `python_magnetrun` at all. No changes required.

### `examples/` and `python_magnetrun/panels/`

The following examples use `MagnetRun.from*` (which still works) and never
import or reference the `MagnetData` class directly:

| File | Why no change needed |
|---|---|
| `examples/bilan.py` | Uses `MagnetRun.fromtxt / fromtdms` only |
| `examples/cmp_fields.py` | Uses `MagnetRun.fromtxt / fromtdms` only |
| `examples/corr_Ih_Ib.py` | Uses `MagnetRun.fromtxt / fromtdms` only |
| `examples/get-record.py` | Already imports `MagnetDataBase` directly |
| `examples/outliers.py` | Uses `MagnetRun.fromtdms` only |
| `examples/plot_hybrid_minimal.py` | Uses `MagnetRun.fromtxt / fromtdms` only |
| `examples/plot_hybrid_with_pupitre_tdms.py` | Uses `MagnetRun.fromtxt / fromtdms` only |
| `examples/pupitre.py` | Uses `MagnetRun.fromtxt` only |
| `examples/timeseries-anomaly-detection.py` | Uses `MagnetRun.fromtdms` only |
| `examples/flow_params_magnetrun_pipeline.py` | No magnetdata import |
| All hybrid/RMS/trigger/FEPC examples | Use hybrid-specific readers only |

---

## Verification

```bash
source magnetrun-env/bin/activate

# Confirm no remaining bare MagnetData class references in examples/panels
grep -rn "MagnetData[^B]" examples/ python_magnetrun/panels/ \
  | grep -v "PandasMagnetData\|TdmsMagnetData\|EnsightMagnetData\|BProfileMagnetData\|FeelppMagnetData\|MagnetDataBase" \
  | grep -v "\.MagnetData\b"
# Expected: zero matches

# Quick import check on the two changed files
python -c "import examples.proposal" 2>&1
```
