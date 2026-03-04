# Phase 1 – Readability & Tooling (Weeks 1–3)

## Software Stack Decisions (locked for all phases)

| Concern | Choice | Rationale |
|---------|--------|-----------|
| **Target platform** | Ubuntu 24.04 LTS (Noble) | Ships Python 3.12; this is the minimum supported version. |
| **Python version floor** | **≥ 3.12** | `tomllib` is stdlib, improved type generics (`list[str]` without `from __future__`), better error messages. |
| **Linting + formatting** | **`ruff`** | Single tool replacing `flake8` + `isort` + `pyupgrade`; fast, zero config to start. |
| **Static type checking** | **`mypy`** | Dev/CI only — never shipped. Catches type mismatches before runtime. |
| **Config file format** | **TOML via `tomllib`** (stdlib) | Avoids dependency on `pyyaml`; prevents conflict with `python_magnetgeo` which already owns a YAML setup. |
| **Data validation** | **`pydantic` v2** (deferred to Phase 3) | Already used transitively via `fastapi` in `python_magnetdb` — consistent choice across the stack. |
| **CLI framework** | **`typer`** (deferred to Phase 4) | Consistent with `fastapi` ecosystem; upgrade from `argparse` deferred until CLI is split. |
| **Dashboard framework** | Deferred to Phase 4 | Decided when dashboard work starts. |
| **HTTP client** | Deferred to Phase 3 | Decided when `api/` module is built. |

> **Note on `tomllib`:** Python 3.11+ includes `tomllib` in the standard library (read-only TOML parser).
> For writing TOML (if ever needed) use `tomli-w`. Site configs in Phase 3 will use `.toml` files
> instead of `.yaml` to stay consistent with `pyproject.toml` and avoid the `pyyaml` dependency.

---

## Goal

Lay the groundwork for all subsequent phases without moving or deleting any
files. Every change must leave the existing CLI entry points and public API
intact. At the end of this phase:

- Core modules use structured logging instead of `print()`.
- `MagnetData.Type` is an `Enum`, not a raw integer.
- Dead commented-out code is removed.
- Public APIs carry type annotations.
- `mypy` and `ruff` are wired into the dev workflow.
- Unit tests exist for `MagnetData`, `MagnetRun`, and `MRecord`.

---

## Scope

### 1.1 Replace integer type codes with `DataType` Enum

**File:** `python_magnetrun/magnetdata.py` (line 40)

Replace:

```python
Type: int = 0  # 0=Pandas, 1=Tdms, 2=Ensight
```

With:

```python
from enum import Enum

class DataType(Enum):
    PANDAS = 0
    TDMS = 1
    ENSIGHT = 2
```

Then replace every occurrence of `self.Type == 0`, `self.Type == 1`,
`self.Type == 2` with `self.Type == DataType.PANDAS`, `DataType.TDMS`,
`DataType.ENSIGHT` throughout `magnetdata.py`.

Search for other files that read or set `.Type` directly:

```bash
grep -rn "\.Type\b" python_magnetrun/ --include="*.py"
```

Update those call sites too.

**Acceptance criteria:**
- `python -c "from python_magnetrun.magnetdata import DataType; print(DataType.PANDAS)"` succeeds.
- No integer literals `0`, `1`, `2` remain where `.Type` is compared.

---

### 1.2 Replace `print()` with structured logging

**Files:**
- `python_magnetrun/MagnetRun.py` (lines 26–28, 110–113, 119–130)
- `python_magnetrun/magnetdata.py` (line 57 and any remaining `print()` calls)
- `python_magnetrun/MRecord.py`

Each module already (or should) have a module-level logger:

```python
import logging
logger = logging.getLogger(__name__)
```

Replace:

```python
print(f"MagnetRun: loading {filename}")
```

With an appropriate level:

```python
logger.debug("MagnetRun: loading %s", filename)   # internal detail
logger.info("MagnetRun: loading %s", filename)    # user-facing progress
logger.warning("unexpected value: %s", val)       # non-fatal problem
```

Do **not** replace `print()` inside CLI entry points (`python_magnetrun.py`,
`analysis/cli.py`, `requests/cli.py`, `hybrid/cli.py`, `processing/cli.py`)
— those are intentionally user-facing.

**Acceptance criteria:**
- `grep -rn "^\s*print(" python_magnetrun/magnetdata.py python_magnetrun/MagnetRun.py python_magnetrun/MRecord.py` returns zero hits.
- Running the CLI at `-v`/`--verbose` still shows diagnostic output.

---

### 1.3 Remove commented-out dead code

**Files and locations:**

| File | Lines | Description |
|------|-------|-------------|
| `python_magnetrun/MRecord.py` | 116–127 | Commented `__le__` and `__ge__` methods |
| `python_magnetrun/MagnetRun.py` | ~63 | `# data.removeData(...)` |
| `python_magnetrun/magnetdata.py` | ~71–74 | Several commented `# print(...)` blocks |

Rule: remove code that is commented out **and** has no accompanying explanation
comment (i.e. `# TODO`, `# FIXME`, `# NOTE` blocks should be kept).

**Acceptance criteria:**
- `grep -n "^[[:space:]]*#.*def \|^[[:space:]]*#.*return \|^[[:space:]]*#.*print(" python_magnetrun/MRecord.py python_magnetrun/MagnetRun.py python_magnetrun/magnetdata.py` returns zero hits for dead code.

---

### 1.4 Add type annotations to core public APIs

**Priority targets:**

`python_magnetrun/magnetdata.py`:

```python
def getData(self) -> pd.DataFrame: ...
def getKeys(self) -> list[str]: ...
def addData(self, key: str, data: pd.Series) -> None: ...
def renameData(self, old: str, new: str) -> None: ...
def removeData(self, key: str) -> None: ...
```

`python_magnetrun/MagnetRun.py`:

```python
@classmethod
def fromtxt(cls, filename: str, *, housing: str = "") -> "MagnetRun": ...

@classmethod
def fromcsv(cls, filename: str, *, housing: str = "") -> "MagnetRun": ...

def prepareData(self) -> None: ...
def getStats(self) -> dict[str, Any]: ...
def getMData(self) -> MagnetData: ...
def getInsert(self) -> str: ...
```

`python_magnetrun/MRecord.py`:

```python
def to_json(self) -> str: ...
def getDataFilename(self) -> str: ...
```

Do **not** annotate private helpers or parameters you cannot confidently type —
leave those as `Any` or unannotated rather than guessing.

**Acceptance criteria:**
- `mypy python_magnetrun/magnetdata.py python_magnetrun/MagnetRun.py python_magnetrun/MRecord.py --ignore-missing-imports` reports no new errors beyond those already present before this phase.

---

### 1.5 Fix `fromtdms` file-opening anti-pattern

**File:** `python_magnetrun/magnetdata.py` (~line 62)

Current code opens the file as text just to check its extension. Replace with:

```python
from pathlib import Path

suffix = Path(name).suffix.lower()
if suffix != ".tdms":
    raise ValueError(f"Expected .tdms file, got: {name!r}")
```

Move this check **before** any file I/O.

**Acceptance criteria:**
- Passing a non-tdms path raises `ValueError` without attempting to open the file.
- Passing a valid `.tdms` path still loads correctly.

---

### 1.6 Add `mypy` and `ruff` configuration

**File:** `pyproject.toml`

Add:

```toml
[tool.mypy]
python_version = "3.12"
strict = false
ignore_missing_imports = true
warn_unused_ignores = true
warn_return_any = false

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
ignore = ["E501"]   # handled by formatter

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["S101"]   # allow assert in tests
```

Add a `py.typed` marker:

```bash
touch python_magnetrun/py.typed
```

Update `pyproject.toml` to include it:

```toml
[tool.setuptools.package-data]
python_magnetrun = ["py.typed"]
```

Add `mypy` and `ruff` to dev dependencies:

```toml
[project.optional-dependencies]
dev = [
    "mypy>=1.9",
    "ruff>=0.4",
    "pytest>=8",
    "pytest-cov",
]
```

**Acceptance criteria:**
- `ruff check python_magnetrun/` exits 0 (or with only pre-existing issues documented in a `# noqa` baseline).
- `mypy python_magnetrun/ --ignore-missing-imports` exits 0 or with a known baseline.

---

### 1.7 Write unit tests for core modules

**New files:**

#### `tests/test_magnetdata.py`

Tests to write:

- `test_fromtxt_loads_dataframe` – load a sample `.txt` file from `data/`, assert `getData()` returns a non-empty DataFrame.
- `test_getkeys_returns_list` – assert `getKeys()` returns a `list[str]`.
- `test_adddata_creates_new_column` – add a synthetic column, verify it appears.
- `test_renamedata_renames_column` – rename a column, verify old name gone, new name present.
- `test_removedata_removes_column` – remove a column, verify it is gone.
- `test_datatype_enum` – assert `DataType.PANDAS.value == 0` etc.

#### `tests/test_magnetrun.py`

- `test_fromtxt_returns_magnetrun` – load a sample file, assert result is `MagnetRun`.
- `test_getinsert_returns_str` – assert `getInsert()` returns a non-empty string.
- `test_getmdata_returns_magnetdata` – assert `getMData()` is a `MagnetData`.
- `test_preparedata_runs_without_error` – call `prepareData()`, assert no exception.

#### `tests/test_mrecord.py`

- `test_to_json_is_valid_json` – serialize an `MRecord`, parse the JSON.
- `test_eq_with_equal_records` – two identical records should compare equal.
- `test_getdatafilename_format` – assert the returned string follows expected naming.

Use the sample files already present in `data/` as fixtures via
`pathlib.Path(__file__).parent.parent / "data"`.

**Acceptance criteria:**
- `pytest tests/test_magnetdata.py tests/test_magnetrun.py tests/test_mrecord.py -v` — all tests pass.
- Code coverage for the three modules ≥ 40 % (baseline; will grow in later phases).

---

## Files Modified This Phase

| File | Change |
|------|--------|
| `python_magnetrun/magnetdata.py` | `DataType` enum, `print`→`logger`, dead code removal, type annotations, `fromtdms` fix |
| `python_magnetrun/MagnetRun.py` | `print`→`logger`, dead code removal, type annotations |
| `python_magnetrun/MRecord.py` | dead code removal, type annotations |
| `pyproject.toml` | `[tool.mypy]`, `[tool.ruff]`, `dev` extras, `py.typed` entry |
| `python_magnetrun/py.typed` | New (empty marker file) |
| `tests/test_magnetdata.py` | New |
| `tests/test_magnetrun.py` | New |
| `tests/test_mrecord.py` | New |

## Files NOT Modified This Phase

- `python_magnetrun/requests/` — still `requests/`, renamed in Phase 2.
- `python_magnetrun/python_magnetrun.py` — renamed in Phase 2.
- `python_magnetrun/processing/hysteresis.py` — split in Phase 2.
- Any dashboard or API code — Phases 3–4.

---

## Verification Checklist

- [ ] `python -c "from python_magnetrun.magnetdata import DataType; print(list(DataType))"` works.
- [ ] `grep -rn "print(" python_magnetrun/magnetdata.py python_magnetrun/MagnetRun.py python_magnetrun/MRecord.py` → no library-level `print()` calls.
- [ ] `ruff check python_magnetrun/` exits 0 (or documented baseline).
- [ ] `mypy python_magnetrun/ --ignore-missing-imports` exits 0 or documented baseline.
- [ ] `pytest tests/ -x` — all tests (old + new) pass.
- [ ] Existing CLI smoke test: `python-magnetrun --help` still works.

---

## Dependencies on Other Phases

None — this phase is self-contained and sets the foundation for Phases 2–4.
