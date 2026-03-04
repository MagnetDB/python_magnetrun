# Phase 2 – Restructuring & Splitting (Weeks 4–8)

## Goal

Improve navigability by breaking large monolithic files into focused modules,
renaming confusing module names, and introducing a formal data-loader registry.
All existing entry points and public imports must remain working via shim
re-exports throughout this phase.

**Prerequisite:** Phase 1 is complete (enum, logging, annotations, tests, ruff/mypy green).

---

## Scope

### 2.1 Rename `python_magnetrun/requests/` → `python_magnetrun/fetchers/`

**Problem:** The directory name `requests` shadows the popular PyPI package,
causing import confusion and IDE false-positives.

**Steps:**

1. Create `python_magnetrun/fetchers/` with the same three files:
   - `fetchers/__init__.py` (copy from `requests/__init__.py`)
   - `fetchers/cli.py` (copy from `requests/cli.py`)
   - `fetchers/connect.py` (copy from `requests/connect.py`)
   - `fetchers/webscrapping.py` (copy from `requests/webscrapping.py`)

2. Update internal imports in the new files (replace `from python_magnetrun.requests` with `from python_magnetrun.fetchers`).

3. Update all callers:

   ```bash
   grep -rn "from python_magnetrun.requests\|import python_magnetrun.requests" python_magnetrun/ --include="*.py"
   ```

   Key files expected: `python_magnetrun/MRecord.py`, `python_magnetrun/python_magnetrun.py`.

4. Add a backwards-compat shim in the old location so users who imported from
   `python_magnetrun.requests` do not get immediate breakage:

   ```python
   # python_magnetrun/requests/__init__.py  (shim — Phase 2 only)
   import warnings
   warnings.warn(
       "python_magnetrun.requests is deprecated; use python_magnetrun.fetchers",
       DeprecationWarning,
       stacklevel=2,
   )
   from python_magnetrun.fetchers import *  # noqa: F401, F403
   ```

   The shim is removed in Phase 4.

5. Update `pyproject.toml` console script entry point:

   ```toml
   # Before
   srvdata-to-magnetrun = "python_magnetrun.requests.cli:main"
   # After
   srvdata-to-magnetrun = "python_magnetrun.fetchers.cli:main"
   ```

**Acceptance criteria:**
- `python -c "from python_magnetrun.fetchers.connect import connect"` succeeds.
- `python -c "from python_magnetrun.requests import connect"` prints a `DeprecationWarning` but does **not** raise an error.
- `srvdata-to-magnetrun --help` works.

---

### 2.2 Rename `python_magnetrun/python_magnetrun.py` → `python_magnetrun/cli_main.py`

**Problem:** A module named identically to its package is confusing to navigate
and breaks some import scenarios.

**Steps:**

1. Copy `python_magnetrun/python_magnetrun.py` → `python_magnetrun/cli_main.py`.

2. Update `pyproject.toml`:

   ```toml
   # Before
   python-magnetrun = "python_magnetrun.python_magnetrun:main"
   # After
   python-magnetrun = "python_magnetrun.cli_main:main"
   ```

3. Add a shim in the original location:

   ```python
   # python_magnetrun/python_magnetrun.py  (shim — Phase 2 only)
   import warnings
   warnings.warn(
       "python_magnetrun.python_magnetrun is deprecated; use python_magnetrun.cli_main",
       DeprecationWarning,
       stacklevel=2,
   )
   from python_magnetrun.cli_main import *  # noqa: F401, F403
   from python_magnetrun.cli_main import main
   ```

**Acceptance criteria:**
- `python-magnetrun --help` works.
- `python -c "from python_magnetrun.python_magnetrun import main"` prints a `DeprecationWarning` but does not raise.

---

### 2.3 Split `magnetdata.py` (1337 lines) into focused submodules

**Target structure:**

```
python_magnetrun/magnetdata/
├── __init__.py      ← re-exports MagnetData (backwards compat)
├── core.py          ← MagnetData class definition + constructors
├── loaders.py       ← fromtxt(), fromcsv(), fromtdms(), fromensight(), fromStringIO()
├── transforms.py    ← addData(), renameData(), removeData(), cleanupData()
└── stats.py         ← stats(), getStats(), summary methods
```

**Steps:**

1. Create the `magnetdata/` directory. Do **not** delete `magnetdata.py` yet.

2. Populate each submodule by cutting the relevant methods from `magnetdata.py`.
   Keep imports tight — each submodule only imports what it needs.

3. `magnetdata/__init__.py` must re-export `MagnetData` so all existing code
   continues to work:

   ```python
   from python_magnetrun.magnetdata.core import MagnetData
   from python_magnetrun.magnetdata.core import DataType  # already defined in Phase 1

   __all__ = ["MagnetData", "DataType"]
   ```

4. Convert `magnetdata.py` into a shim:

   ```python
   # python_magnetrun/magnetdata.py  (shim — Phase 2 only)
   import warnings
   warnings.warn(
       "Import from python_magnetrun.magnetdata (the package) instead.",
       DeprecationWarning,
       stacklevel=2,
   )
   from python_magnetrun.magnetdata import *  # noqa: F401, F403
   from python_magnetrun.magnetdata import MagnetData, DataType
   ```

   Remove this shim file in Phase 4.

5. Update any direct references to `magnetdata.py` by file path (e.g. in
   `MagnetRun.py`, `cli_main.py`).

**Cutting guide — which methods go where:**

| Target module | Methods |
|--------------|---------|
| `core.py` | `__init__`, `__repr__`, `__str__`, `getData`, `getKeys`, `FileName` property, `Type` property |
| `loaders.py` | `fromtxt`, `fromcsv`, `fromtdms`, `fromensight`, `fromStringIO` |
| `transforms.py` | `addData`, `renameData`, `removeData`, `cleanupData`, `setData` |
| `stats.py` | `stats`, `getStats`, `getStat`, `summary` |

**Acceptance criteria:**
- `python -c "from python_magnetrun.magnetdata import MagnetData"` succeeds.
- `python -c "from python_magnetrun import MagnetData"` succeeds.
- `pytest tests/test_magnetdata.py -v` — all tests pass.

---

### 2.4 Split `python_magnetrun/processing/hysteresis.py` (1122 lines) into sub-files

**Target structure:**

```
python_magnetrun/processing/hysteresis/
├── __init__.py      ← re-exports public symbols
├── analysis.py      ← core loop detection algorithms
├── plotting.py      ← visualization functions
└── outliers.py      ← outlier removal specific to hysteresis
```

**Cutting guide:**

| Target | Content |
|--------|---------|
| `analysis.py` | Loop detection, cycle counting, area computation functions |
| `plotting.py` | Any function that creates matplotlib figures or axes |
| `outliers.py` | Hysteresis-specific outlier removal functions |

**Backwards compat:** convert `hysteresis.py` into a shim that imports from
`hysteresis/` package. Remove in Phase 4.

**Acceptance criteria:**
- `from python_magnetrun.processing.hysteresis import <existing_public_symbol>` works.
- `pytest tests/ -x` — all tests pass.

---

### 2.5 Remove `matplotlib.rcParams` side effects at import time

**Problem:** Several modules set `matplotlib.rcParams["text.usetex"] = True`
at module level, which breaks any environment without a LaTeX installation.

**Files to fix:**
- `python_magnetrun/python_magnetrun.py` (now `cli_main.py`)
- `python_magnetrun/outliers.py`
- `python_magnetrun/pupitre.py`

**Fix:** Move the rcParams line inside the function that actually creates the
plot, guarded by an optional parameter:

```python
def plot_something(ax, *, use_latex: bool = False):
    if use_latex:
        import matplotlib
        matplotlib.rcParams["text.usetex"] = True
    ...
```

**Acceptance criteria:**
- `import python_magnetrun` in a headless environment (no LaTeX) raises no errors.
- `import python_magnetrun.outliers` and `import python_magnetrun.pupitre` raise no errors.

---

### 2.6 Introduce a `DataLoader` Protocol and format registry

**New file:** `python_magnetrun/protocols.py`

```python
"""
Protocols for pluggable data loading.

Register new loaders via _LOADER_REGISTRY to support additional file formats
without modifying MagnetData.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from python_magnetrun.magnetdata import MagnetData


@runtime_checkable
class DataLoader(Protocol):
    """Protocol that every file-format loader must satisfy."""

    @classmethod
    def can_load(cls, path: str) -> bool:
        """Return True if this loader handles the given path."""
        ...

    @classmethod
    def load(cls, path: str, **kwargs: object) -> "MagnetData":
        """Load the file and return a MagnetData instance."""
        ...

    def get_format_name(self) -> str:
        """Short human-readable name for this format, e.g. 'pupitre-txt'."""
        ...
```

**New file:** `python_magnetrun/loaders/__init__.py` (or extend `magnetdata/loaders.py`)

Implement concrete loader classes that satisfy the protocol:

```python
class PupitreLoader:
    @classmethod
    def can_load(cls, path: str) -> bool:
        return Path(path).suffix.lower() == ".txt"

    @classmethod
    def load(cls, path: str, **kwargs) -> MagnetData:
        return MagnetData.fromtxt(path, **kwargs)

    def get_format_name(self) -> str:
        return "pupitre-txt"
```

Create `CsvLoader`, `TdmsLoader`, `EnsightLoader` analogously.

**Registry and auto-detect factory:**

```python
# python_magnetrun/magnetdata/loaders.py  (addition)
_LOADER_REGISTRY: dict[str, type[DataLoader]] = {
    ".txt": PupitreLoader,
    ".csv": CsvLoader,
    ".tdms": TdmsLoader,
}

def from_file(path: str, **kwargs) -> MagnetData:
    """Auto-detect format by extension and load."""
    suffix = Path(path).suffix.lower()
    loader_cls = _LOADER_REGISTRY.get(suffix)
    if loader_cls is None:
        raise ValueError(
            f"No loader registered for extension {suffix!r}. "
            f"Registered: {list(_LOADER_REGISTRY)}"
        )
    return loader_cls.load(path, **kwargs)
```

Expose on `MagnetData` as a class method:

```python
# in magnetdata/core.py
@classmethod
def from_file(cls, path: str, **kwargs) -> "MagnetData":
    from python_magnetrun.magnetdata.loaders import from_file as _from_file
    return _from_file(path, **kwargs)
```

**Acceptance criteria:**
- `MagnetData.from_file("sample.txt")` loads a txt file.
- `MagnetData.from_file("sample.csv")` loads a csv file.
- `MagnetData.from_file("sample.xyz")` raises `ValueError`.
- `isinstance(PupitreLoader(), DataLoader)` is `True` (runtime-checkable).

---

### 2.7 Fix `getInsert()` path stripping

**File:** `python_magnetrun/MagnetRun.py` (lines 190–193)

Replace the hand-written extension stripping with:

```python
from pathlib import Path

def getInsert(self) -> str:
    return Path(self.MagnetData.FileName).stem
```

**Acceptance criteria:**
- `MagnetRun.getInsert()` returns the filename without extension and without path components.

---

### 2.8 Add `__all__` to package `__init__.py`

**File:** `python_magnetrun/__init__.py`

```python
from python_magnetrun.magnetdata import MagnetData, DataType
from python_magnetrun.MagnetRun import MagnetRun
from python_magnetrun.MRecord import MRecord

__all__ = [
    "MagnetData",
    "DataType",
    "MagnetRun",
    "MRecord",
]
```

**Acceptance criteria:**
- `from python_magnetrun import *` imports exactly the symbols in `__all__`.
- `python -c "import python_magnetrun; print(python_magnetrun.__all__)"` works.

---

## Files Modified / Created This Phase

| File | Action |
|------|--------|
| `python_magnetrun/fetchers/__init__.py` | New |
| `python_magnetrun/fetchers/cli.py` | New (moved from `requests/`) |
| `python_magnetrun/fetchers/connect.py` | New (moved from `requests/`) |
| `python_magnetrun/fetchers/webscrapping.py` | New (moved from `requests/`) |
| `python_magnetrun/requests/__init__.py` | Shim (remove Phase 4) |
| `python_magnetrun/cli_main.py` | New (moved from `python_magnetrun.py`) |
| `python_magnetrun/python_magnetrun.py` | Shim (remove Phase 4) |
| `python_magnetrun/magnetdata/` | New package directory |
| `python_magnetrun/magnetdata/__init__.py` | New |
| `python_magnetrun/magnetdata/core.py` | New |
| `python_magnetrun/magnetdata/loaders.py` | New |
| `python_magnetrun/magnetdata/transforms.py` | New |
| `python_magnetrun/magnetdata/stats.py` | New |
| `python_magnetrun/magnetdata.py` | Shim (remove Phase 4) |
| `python_magnetrun/processing/hysteresis/` | New package directory |
| `python_magnetrun/processing/hysteresis/__init__.py` | New |
| `python_magnetrun/processing/hysteresis/analysis.py` | New |
| `python_magnetrun/processing/hysteresis/plotting.py` | New |
| `python_magnetrun/processing/hysteresis/outliers.py` | New |
| `python_magnetrun/processing/hysteresis.py` | Shim (remove Phase 4) |
| `python_magnetrun/protocols.py` | New |
| `python_magnetrun/__init__.py` | Add `__all__` |
| `python_magnetrun/MagnetRun.py` | Fix `getInsert()` |
| `python_magnetrun/outliers.py` | Move rcParams inside plot fn |
| `python_magnetrun/pupitre.py` | Move rcParams inside plot fn |
| `pyproject.toml` | Update entry point for `cli_main`, `fetchers` |

---

## Verification Checklist

- [ ] `python -c "from python_magnetrun import MagnetData, MagnetRun, MRecord"` succeeds.
- [ ] `python -c "from python_magnetrun.fetchers.connect import connect"` succeeds.
- [ ] `python -c "from python_magnetrun.requests import connect"` → DeprecationWarning only.
- [ ] `python-magnetrun --help` works.
- [ ] `srvdata-to-magnetrun --help` works.
- [ ] `MagnetData.from_file("sample.txt")` loads correctly.
- [ ] `pytest tests/ -x` — all tests pass.
- [ ] `ruff check python_magnetrun/` — no new errors.
- [ ] `mypy python_magnetrun/ --ignore-missing-imports` — no new errors.

---

## Dependencies

- **Requires Phase 1** complete: enum, logging, annotations, ruff/mypy baseline.
- Phase 3 and Phase 4 build on the `fetchers/` and `magnetdata/` packages created here.
