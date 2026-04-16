# Fix Plan: Remaining Issues (post-REVIEW.md)

Date: 2026-04-13

---

## Pre-requisite — Confirm Hybrid Format Validation

All six steps from `FILE_FORMAT_VALIDATION.md` are already merged:

| Step | Status |
|---|---|
| `utils/validation.py` with `FileFormatError` + all validators | Done |
| `FileFormatError` exported from `utils/__init__.py` | Done |
| `magnetdata.py` factory methods call validators | Done |
| `tests/test_file_validation.py` with full coverage | Done |
| `rms_reader.py` + `vprocess_reader.py` call validators in `parse_header` | Done |
| `fepc_reader.py` + `trigger_reader.py` raise `FileFormatError` | Done |

**Gate: run the two format-validation test suites and confirm zero failures before
starting any item below.**

```bash
source magnetrun-env/bin/activate
pytest tests/test_file_validation.py tests/test_hybrid_api.py -v
```

Expected: all tests pass.  If any fail, fix them first — do not proceed until green.

---

## Issue 1 — `Data` attribute type divergence  *(Critical — unblocks #2 and #3)*

**Problem:**  
`MagnetDataBase.Data` is typed `pd.DataFrame | dict`
([magnetdata_base.py:74](../python_magnetrun/magnetdata_base.py#L74)).
Callers branch on `isinstance` instead of using `getData()`, which already
returns `pd.DataFrame` uniformly.

**Fix:**

1. Rename `self.Data` → `self._data` in `MagnetDataBase`, `PandasMagnetData`, and
   `TdmsMagnetData`.  Update the type annotation to reflect each concrete type
   (`pd.DataFrame` for Pandas, `dict[str, pd.DataFrame]` for TDMS).
2. Add a `@property data` shim in `MagnetDataBase` that delegates to `getData()`
   so any remaining `obj.Data` call (outside subclasses) gets a deprecation
   warning rather than a silent break.
3. Remove all `isinstance(self.MagnetData.Data, pd.DataFrame)` branches outside
   the subclasses; replace with `getData()`.

**Files:**
- `python_magnetrun/magnetdata_base.py`
- `python_magnetrun/magnetdata_pandas.py`
- `python_magnetrun/magnetdata_tdms.py`
- `python_magnetrun/MagnetRun.py` (see also Issue 3)

**Validation:**

```bash
pytest tests/test_magnetdata.py tests/test_magnetdata_tdms.py -v
# also run the full suite to catch regressions
pytest --tb=short -q
```

---

## Issue 2 — `TdmsMagnetData.getUnitKey` bypasses `self.units`  *(Significant)*

**Problem:**  
`getUnitKey` always calls `self.PigBrotherUnits(group)` regardless of whether
`self.units` (populated by `Units()` from the defs file) already contains the
answer.  This violates the resolution order documented on `Units()` and is an
LSP violation.
([magnetdata_tdms.py:175-189](../python_magnetrun/magnetdata_tdms.py#L175-L189))

**Fix:**

```python
def getUnitKey(self, key: str) -> str:
    if key in self.units:
        return self.units[key]
    # fallback: derive from PigBrother group→unit map
    group = ...  # existing group-extraction logic
    return self.PigBrotherUnits(group)
```

**Files:**
- `python_magnetrun/magnetdata_tdms.py`

**Validation:**

```bash
# unit test: load a TdmsMagnetData with a defs file that maps a key,
# assert getUnitKey returns the defs-file value, not the PigBrother fallback
pytest tests/test_magnetdata_tdms.py -v -k "unit_key"
```

Add a focused test in `tests/test_magnetdata_tdms.py` if one does not exist.

---

## Issue 3 — `MagnetRun.saveData` breaks the abstraction  *(Significant, unblocked by #1)*

**Problem:**  
`MagnetRun.saveData` ([MagnetRun.py:202-209](../python_magnetrun/MagnetRun.py#L202-L209))
calls `isinstance(self.MagnetData.Data, pd.DataFrame)` directly instead of
delegating to `self.MagnetData.saveData(...)`.  TDMS data silently falls through.

**Fix:**

1. Ensure `MagnetDataBase` declares an abstract `saveData(output_path: str) -> None`.
2. Implement concrete `saveData` in `PandasMagnetData` (existing CSV logic) and
   `TdmsMagnetData` (iterate groups, write per-group CSV or raise `NotImplementedError`
   with a clear message if not supported yet).
3. Replace the inline `isinstance` check in `MagnetRun.saveData` with a single
   `self.MagnetData.saveData(output_path)` call.

**Files:**
- `python_magnetrun/magnetdata_base.py`
- `python_magnetrun/magnetdata_pandas.py`
- `python_magnetrun/magnetdata_tdms.py`
- `python_magnetrun/MagnetRun.py`

**Validation:**

```bash
pytest tests/test_python_magnetrun.py -v -k "save"
# also smoke-test the CLI path that calls saveData
```

---

## Issue 4 — Protocol duplication: `DataProvider` vs `DataLoader`  *(Significant)*

**Problem:**  
`DataProvider` (`hybrid/hybrid_run.py`) and `DataLoader` (`hybrid/data_protocol.py`)
describe the same concept with slightly different signatures.

**Fix:**  
This is tracked as **Phase A0** in
[`prompts/cross-domain-comparison.prompt.md`](cross-domain-comparison.prompt.md).
Steps:

1. Delete `DataProvider` from `hybrid/hybrid_run.py`.
2. Extend `DataLoader` in `hybrid/data_protocol.py` with any methods that only
   existed on `DataProvider`.
3. Annotate `HybridRun` to implement `DataLoader` (it already satisfies the
   protocol structurally; the annotation makes it explicit).
4. Confirm `MagnetRun` still satisfies `DataLoader` via `isinstance` check in a test.

**Files:**
- `python_magnetrun/hybrid/hybrid_run.py`
- `python_magnetrun/hybrid/data_protocol.py`
- `python_magnetrun/MagnetRun.py`

**Validation:**

```bash
pytest tests/test_hybrid_api.py tests/analysis/test_loaders.py -v
python -c "
from python_magnetrun.hybrid.data_protocol import DataLoader
from python_magnetrun.hybrid.hybrid_run import HybridRun
from python_magnetrun.MagnetRun import MagnetRun
assert issubclass(HybridRun, DataLoader)
assert issubclass(MagnetRun, DataLoader)
print('Protocol compliance OK')
"
```

---

## Issue 5 — Hardcoded developer path as CLI default  *(Minor)*

**Problem:**  
`cli_args.py` line ~249 sets:
```python
default="/home/LNCMI-G/christophe.trophime/LNCMIG-Data/srv-data-install"
```
This silently resolves to a non-existent path on any other machine.

**Fix:**

```python
import os
default=os.environ.get("MAGNETRUN_DATA_DIR")  # None if not set
```

If `None` is not acceptable as an argparse default (e.g. the argument is required),
use `required=True` and remove the default entirely.

**Files:**
- `python_magnetrun/cli_args.py`

**Validation:**

```bash
pytest tests/test_cli_entrypoints.py -v
# smoke-test: confirm no hardcoded path survives in the installed package
grep -r "christophe.trophime" python_magnetrun/ --include="*.py" && echo "FOUND" || echo "Clean"
```

---

## Issue 6 — `tsdownsample` is an undeclared dependency  *(Minor)*

**Problem:**  
`hybrid/hybrid_run.py` imports `tsdownsample` inside a `try/except ImportError`
but the package is absent from `pyproject.toml`.

**Fix:**  
Add to `pyproject.toml` under an optional `hybrid` extras group:

```toml
[project.optional-dependencies]
hybrid = [
    "natsort",
    "tsdownsample",
]
```

If `tsdownsample` has a heavy native dependency that makes it unsuitable as a
required extra, add a comment in `hybrid_run.py` explaining the soft requirement
and update the README.

**Files:**
- `pyproject.toml`

**Validation:**

```bash
# in a clean venv (or the existing magnetrun-env after uninstalling):
pip install -e ".[hybrid]"
python -c "import tsdownsample; print('ok')"
pytest tests/test_hybrid_api.py -v
```

---

## Issue 7 — Editor backup file in the package  *(Trivial)*

**Problem:**  
`python_magnetrun/pigbrother-defs.json~` can be accidentally included in
sdist/wheel builds.

**Fix:**

```bash
git rm python_magnetrun/pigbrother-defs.json~
echo "*.json~" >> .gitignore
git add .gitignore
```

**Validation:**

```bash
git status   # json~ file no longer tracked
python -m build --sdist --no-isolation 2>/dev/null | head -5
# verify the .json~ file is absent from the generated tar.gz:
tar -tzf dist/*.tar.gz | grep "json~" && echo "STILL PRESENT" || echo "Clean"
```

---

## Summary Table

| # | Issue | Severity | Blocks | Gate |
|---|---|---|---|---|
| Pre | Hybrid format validation passes | Gate | All | `pytest test_file_validation.py test_hybrid_api.py` |
| 1 | `Data` type divergence | Significant | #2, #3 | `pytest test_magnetdata*.py` |
| 2 | `TdmsMagnetData.getUnitKey` | Significant | — | `pytest -k unit_key` |
| 3 | `MagnetRun.saveData` | Significant | — | `pytest -k save` |
| 4 | Protocol duplication | Significant | Phase A (cross-domain) | `pytest test_hybrid_api.py` + `isinstance` smoke |
| 5 | Hardcoded default path | Minor | — | `grep` for path + `pytest test_cli_entrypoints.py` |
| 6 | `tsdownsample` dependency | Minor | — | clean-env install |
| 7 | Editor backup file | Trivial | — | sdist tar check |
