# Prompt: Phase 1 — Quick Wins and Structural Fixes

## Context

`python_magnetrun` is a scientific data-analysis package for high-field magnet facility
runs (sites M8, M9, M10). It is at v0.2.0 (pre-alpha). This prompt covers **Phase 1** of
the improvement plan: low-risk, high-impact fixes that unblock all later phases. No public
API breaks are allowed; all existing CLI entry points must continue to work.

Reference document: `IMPROVEMENT_PLAN.md` §Phase 1.

---

## Objective

Apply ten targeted fixes to improve code quality, eliminate hidden failures, and enable CI.
All changes are confined to existing files — no new modules are created in this phase.

---

## Prerequisites

- Read `IMPROVEMENT_PLAN.md` in full before starting.
- Run `pytest tests/` to establish a green baseline. Record any pre-existing failures.
- All changes must leave the test suite at least as green as the baseline.

---

## Task 1.1 — Replace integer `Type` code with `DataType` enum

**File:** `python_magnetrun/magnetdata.py`

**Problem:** `self.Type: int = 0` encodes data source as a magic integer. The meaning
(`0=Pandas, 1=Tdms, 2=Ensight`) lives only in a comment at line 40.

**Fix:**

1. Add near the top of `magnetdata.py`, after the existing imports:

```python
from enum import IntEnum

class DataType(IntEnum):
    """Data source type for MagnetData."""
    PANDAS  = 0
    TDMS    = 1
    ENSIGHT = 2
```

2. Change the constructor signature default:
   ```python
   # before
   Type: int = 0,
   # after
   Type: DataType = DataType.PANDAS,
   ```

3. Replace every `if self.Type == 0:` / `if self.Type == 1:` etc. with:
   ```python
   if self.Type == DataType.PANDAS:
   if self.Type == DataType.TDMS:
   if self.Type == DataType.ENSIGHT:
   ```

4. Grep for all usages before changing:
   ```
   grep -n "\.Type ==" python_magnetrun/magnetdata.py
   grep -rn "\.Type ==" python_magnetrun/
   ```
   Update all callers found.

5. Keep integer values (0, 1, 2) so any serialised data is unaffected — `IntEnum`
   compares equal to its int value.

---

## Task 1.2 — Replace `print()` with `logger.*` in core modules

**Problem:** Core modules use `print()` instead of the already-initialised `logger`.
This bypasses log-level control and breaks log aggregation.

**Files to scan:**

```bash
grep -n "print(" python_magnetrun/MagnetRun.py
grep -n "print(" python_magnetrun/magnetdata.py
grep -n "print(" python_magnetrun/MRecord.py
grep -n "print(" python_magnetrun/analysis/loaders.py
grep -n "print(" python_magnetrun/analysis/processing.py
```

**Rules:**
- Debug/trace output → `logger.debug(...)`
- Normal progress → `logger.info(...)`
- Warnings → `logger.warning(...)`
- Errors that do not raise → `logger.error(...)`

Do **not** touch files under `examples/` or standalone scripts — only library code under
`python_magnetrun/`.

---

## Task 1.3 — Remove module-level matplotlib side effects

**Problem:** `matplotlib.rcParams["text.usetex"] = True` at import time breaks any
environment without a LaTeX installation (CI, Windows, the API server).

**Files:**

```bash
grep -rn "rcParams\[.text.usetex.\]" python_magnetrun/
```

**Fix for each occurrence:** Move the assignment inside the function that actually plots,
guarded by a `use_latex: bool = False` parameter:

```python
def plotData(self, ..., use_latex: bool = False):
    if use_latex:
        import matplotlib
        matplotlib.rcParams["text.usetex"] = True
    # existing plot code ...
```

Ensure the change is applied to `python_magnetrun.py`, `outliers.py`, `pupitre.py`, and
any other file flagged by the grep.

---

## Task 1.4 — Always initialise `self.data` in `MagnetData.__init__`

**File:** `python_magnetrun/magnetdata.py`

**Problem:** The current pattern:
```python
if Data is not None:
    self.Data = Data
```
leaves `self.Data` as an `AttributeError` bomb when `Data=None`.

**Fix:**

```python
# In __init__, replace the conditional block with:
self.Data: pd.DataFrame | dict | None = Data
```

Then add a private guard method used by any method that requires loaded data:

```python
def _require_data(self) -> pd.DataFrame | dict:
    """Raise RuntimeError if data has not been loaded yet."""
    if self.Data is None:
        raise RuntimeError(
            f"MagnetData '{self.FileName}' has no loaded data. "
            "Call a factory method (fromtxt, fromcsv, fromtdms, ...) first."
        )
    return self.Data
```

Replace direct `self.Data` access in methods that would fail on `None` with
`self._require_data()`. Start with `getData()`, `getKeys()`, `stats()`, `saveData()`.

---

## Task 1.5 — Rename `test-*.py` files and fix pytest config

**Problem:** Hyphenated test filenames cannot be imported as Python modules, which breaks
pytest plugins, `coverage`, and `python -m pytest` on some platforms.

**Steps:**

1. Find all hyphenated test files:
   ```bash
   find tests/ -name "test-*.py"
   ```

2. Rename each:
   ```bash
   # Example — apply for each file found:
   git mv tests/test-anomalies.py tests/test_anomalies.py
   git mv tests/test-fft.py tests/test_fft.py
   # ... etc.
   ```

3. In `pyproject.toml`, under `[tool.pytest.ini_options]`, remove `"test-*.py"` from
   `python_files`:
   ```toml
   [tool.pytest.ini_options]
   python_files = ["test_*.py"]
   ```

4. Run `pytest tests/` to verify all renamed tests are discovered and pass.

---

## Task 1.6 — Remove commented-out dead code

**Target locations:**

| File | What to remove |
|------|----------------|
| `MRecord.py` lines ~116–127 | Commented-out `__le__` and `__ge__` methods |
| `MagnetRun.py` line ~63 | `# data.removeData(...)` one-liner |
| `magnetdata.py` lines ~65–78 | `# print(...)` debug blocks inside `fromtdms` |
| Any file | Large blocks of code commented out with no explanatory note |

**Rule:** A comment is kept only when it explains *why*, not *what*. A commented-out code
block with no explanation should be deleted. If the block might be needed in future, add a
GitHub issue reference instead: `# TODO(#42): re-enable when X is available`.

Do a global search before and after:
```bash
grep -rn "^#.*=\|^# [a-zA-Z].*(" python_magnetrun/ | grep -v "^Binary"
```

---

## Task 1.7 — Fix absolute developer paths in `analysis/config.py`

**File:** `python_magnetrun/analysis/config.py` lines ~117–134

**Problem:** Default paths contain `/home/LNCMI-G/christophe.trophime/...` — fails
silently for every other user who has not set the env vars.

**Fix:**

```python
from pathlib import Path

_DEFAULT_SHARE = Path.home() / ".local" / "share" / "magnetrun"

DEFAULT_DATA_DIR: str = os.environ.get(
    "MAGNETRUN_DATA_DIR",
    str(_DEFAULT_SHARE / "data"),
)

DEFAULT_PIGBROTHER_DATA_DIR: str = os.environ.get(
    "MAGNETRUN_PIGBROTHER_DATA_DIR",
    str(_DEFAULT_SHARE / "pigbrother"),
)
```

Add to `README.md` under a new **Configuration** section:

```markdown
## Configuration

| Environment variable | Purpose | Default |
|----------------------|---------|---------|
| `MAGNETRUN_DATA_DIR` | Directory for Pupitre / log data files | `~/.local/share/magnetrun/data` |
| `MAGNETRUN_PIGBROTHER_DATA_DIR` | Directory for PigBrother TDMS files | `~/.local/share/magnetrun/pigbrother` |
| `MAGNETAPI_URL` | Base URL for the python_magnetapi REST API | *(none)* |
| `MAGNETAPI_KEY` | API authentication token | *(none)* |
```

---

## Task 1.8 — Fix `getInsert()` path logic in `MagnetRun.py`

**File:** `python_magnetrun/MagnetRun.py`

**Problem:** The method strips extensions incorrectly when the filename contains
a directory path.

**Fix:**

```python
from pathlib import Path

def getInsert(self) -> str:
    """Return the stem (filename without extension) of the data file."""
    return Path(self.MagnetData.FileName).stem
```

---

## Task 1.9 — Add `__all__` to `python_magnetrun/__init__.py`

**File:** `python_magnetrun/__init__.py`

Read the file first, then declare the public API:

```python
__all__ = [
    "MagnetData",
    "MagnetRun",
    "MRecord",
    "DataType",        # new enum from Task 1.1
]
```

Also add a `__version__` attribute:

```python
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("python_magnetrun")
except PackageNotFoundError:
    __version__ = "unknown"
```

---

## Task 1.10 — Add CI workflow

**File to create:** `.github/workflows/test.yml`

```yaml
name: Tests

on:
  push:
    branches: ["master", "claude/*"]
  pull_request:
    branches: ["master"]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install package and dev dependencies
        run: |
          pip install -e ".[dev]"

      - name: Lint with ruff
        run: ruff check python_magnetrun/

      - name: Run tests
        run: pytest tests/ -v --tb=short
```

Ensure `ruff` and `pytest` are in the `[project.optional-dependencies] dev` group in
`pyproject.toml`.

---

## Verification Checklist

Run each check before opening a PR:

```bash
# 1. All tests pass
pytest tests/ -v

# 2. No print() left in library code
grep -rn "^\s*print(" python_magnetrun/ --include="*.py" | grep -v "test_\|#"

# 3. No commented-out code blocks > 3 lines
# (manual review of diff)

# 4. Absolute paths gone
grep -rn "LNCMI-G\|christophe.trophime" python_magnetrun/

# 5. DataType enum used everywhere
grep -rn "\.Type == [012]" python_magnetrun/

# 6. Import smoke test
python -c "from python_magnetrun import MagnetData, MagnetRun, MRecord, DataType, __version__; print(__version__)"

# 7. Ruff passes
ruff check python_magnetrun/
```

---

## Commit Strategy

Make one commit per task above (10 commits total). Use commit messages in the form:

```
refactor(magnetdata): replace Type int with DataType IntEnum
fix(magnetdata): always initialise self.Data in __init__
fix(config): replace hardcoded developer paths with env-var defaults
chore(tests): rename test-*.py to test_*.py
ci: add GitHub Actions test workflow
...
```
