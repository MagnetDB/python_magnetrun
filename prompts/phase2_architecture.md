# Prompt: Phase 2 — Architecture and Maintainability

## Context

`python_magnetrun` is a scientific data-analysis package for high-field magnet facility
runs (sites M8, M9, M10). This prompt covers **Phase 2** of the improvement plan:
non-breaking structural changes that make the codebase navigable, testable, and ready for
the extensibility work in Phase 3.

**Prerequisite:** Phase 1 must be complete and all tests green before starting Phase 2.

Reference document: `IMPROVEMENT_PLAN.md` §Phase 2.

---

## Objective

Restructure modules, unify configuration, and enforce code-quality tooling — all without
changing any public API or CLI entry point visible to users.

---

## Task 2.1 — Rename `requests/` → `fetchers/`

**Problem:** `python_magnetrun/requests/` shadows the popular `requests` PyPI library.
IDEs, type checkers, and humans get confused.

**Steps:**

1. Rename the directory:
   ```bash
   git mv python_magnetrun/requests python_magnetrun/fetchers
   ```

2. Update all internal imports referencing `python_magnetrun.requests`:
   ```bash
   grep -rn "from.*requests\." python_magnetrun/ --include="*.py"
   grep -rn "import.*magnetrun.requests" python_magnetrun/ --include="*.py"
   ```
   Change each `from python_magnetrun.requests.X import Y`
   to `from python_magnetrun.fetchers.X import Y`.

3. Update the CLI entry point in `pyproject.toml`:
   ```toml
   # before
   srvdata-to-magnetrun = "python_magnetrun.requests.cli:main"
   # after
   srvdata-to-magnetrun = "python_magnetrun.fetchers.cli:main"
   ```

4. Add a **one-release deprecation shim** so existing user code does not break
   immediately:
   ```python
   # python_magnetrun/requests/__init__.py  (new file — a shim)
   """
   Deprecated: use python_magnetrun.fetchers instead.
   This module will be removed in v0.4.0.
   """
   import warnings
   warnings.warn(
       "python_magnetrun.requests is deprecated. "
       "Use python_magnetrun.fetchers instead.",
       DeprecationWarning,
       stacklevel=2,
   )
   from python_magnetrun.fetchers import *  # noqa: F401, F403
   ```

5. Verify:
   ```bash
   python -c "from python_magnetrun.fetchers.cli import main; print('OK')"
   python -c "import python_magnetrun.requests"  # should print DeprecationWarning
   ```

---

## Task 2.2 — Rename `python_magnetrun/python_magnetrun.py` → `cli_main.py`

**Problem:** The file named `python_magnetrun.py` inside the `python_magnetrun` package
shadows the package itself and confuses every static analysis tool.

**Steps:**

1. Rename:
   ```bash
   git mv python_magnetrun/python_magnetrun.py python_magnetrun/cli_main.py
   ```

2. Update `pyproject.toml`:
   ```toml
   # before
   python-magnetrun = "python_magnetrun.python_magnetrun:main"
   # after
   python-magnetrun = "python_magnetrun.cli_main:main"
   ```

3. Add a shim at the old path for backward compat (if any user imports it directly):
   ```python
   # python_magnetrun/python_magnetrun.py
   """Deprecated: import from python_magnetrun.cli_main instead."""
   import warnings
   warnings.warn(
       "python_magnetrun.python_magnetrun is deprecated; "
       "use python_magnetrun.cli_main",
       DeprecationWarning, stacklevel=2,
   )
   from python_magnetrun.cli_main import *  # noqa: F401, F403
   ```

4. Verify CLI still works:
   ```bash
   python-magnetrun --help
   ```

---

## Task 2.3 — Split `magnetdata.py` into a `magnetdata/` subpackage

**Problem:** `magnetdata.py` is 1 300+ lines covering four distinct concerns: class
definition, data loading, data transforms, and statistics.

**Target structure:**

```
python_magnetrun/magnetdata/
├── __init__.py      # re-exports MagnetData and DataType — zero breaking changes
├── _core.py         # MagnetData class definition, __init__, __repr__, _require_data
├── _loaders.py      # fromtxt(), fromcsv(), fromtdms(), fromStringIO()
├── _transforms.py   # addData(), renameData(), removeData(), cleanupData()
└── _stats.py        # stats(), getStats(), extractData(), saveData()
```

**Migration steps:**

1. Create the directory and `__init__.py`:
   ```python
   # python_magnetrun/magnetdata/__init__.py
   """MagnetData public API — all imports preserved for backward compatibility."""
   from python_magnetrun.magnetdata._core import MagnetData, DataType

   __all__ = ["MagnetData", "DataType"]
   ```

2. Move each group of methods to the appropriate private module. The class definition
   stays in `_core.py`; the methods are added via `_loaders.py`, `_transforms.py`,
   `_stats.py` each doing:
   ```python
   # _loaders.py
   from python_magnetrun.magnetdata._core import MagnetData

   def _fromtxt(cls, name: str) -> "MagnetData":
       ...

   MagnetData.fromtxt = classmethod(_fromtxt)
   ```
   Alternatively, keep the class whole in `_core.py` and simply move the method bodies
   there. Choose the approach that produces the smallest diff.

3. Delete the old `magnetdata.py` file only after all imports resolve.

4. Verify:
   ```bash
   python -c "from python_magnetrun import MagnetData; print(MagnetData)"
   python -c "from python_magnetrun.magnetdata import MagnetData; print('OK')"
   pytest tests/ -v
   ```

---

## Task 2.4 — Split `cli_main.py` (44 KB) into `commands/` submodules

**Problem:** The single-file CLI is hard to navigate and impossible to unit-test
individual subcommands.

**Target structure:**

```
python_magnetrun/commands/
├── __init__.py
├── info.py       # --info, --list subcommands
├── plot.py       # --plot, --vs-time, --key-vs-key
├── stats.py      # --stats, --plateau
├── export.py     # --save, --output
└── main.py       # top-level argparse router; entry point
```

**Steps:**

1. Create `commands/` directory.
2. For each group of subcommands, create a module with:
   - A `register(subparsers)` function that adds the subcommand to the argument parser.
   - A `run(args, mrun)` function that executes the subcommand.
3. In `commands/main.py`, import and wire all `register()` calls.
4. Update `pyproject.toml`:
   ```toml
   python-magnetrun = "python_magnetrun.commands.main:main"
   ```
5. Keep `cli_main.py` as a shim calling `commands.main:main` until v0.4.0.

**Note:** Do not start this task until 2.2 is complete (the rename must happen first).

---

## Task 2.5 — Centralize housing/site configuration

**Problem:** M8/M9/M10 channel-rename rules and current-aggregation logic are
hard-coded in `MagnetRun.prepareData()` *and* repeated in `analysis/config.py:SITE_CONFIGS`.
Two sources of truth diverge over time.

**Steps:**

1. Extend `SiteConfig` in `analysis/config.py` with the fields needed by `prepareData`:

   ```python
   @dataclass(frozen=True)
   class SiteConfig:
       # ... existing fields ...

       # Channel renames applied during prepareData()
       # key = source name, value = target name
       column_renames: dict[str, str] = field(default_factory=dict)

       # Formula for the aggregate current IH_ref (as a lambda or column list)
       ih_ref_columns: tuple[str, ...] = ()
       ib_ref_columns: tuple[str, ...] = ()
   ```

2. Populate for each site in `SITE_CONFIGS`:

   ```python
   "M9": SiteConfig(
       ...,
       column_renames={"Flow1": "FlowH", "Flow2": "FlowB"},
       ih_ref_columns=("Idcct1", "Idcct2"),
       ib_ref_columns=("Idcct3", "Idcct4"),
   ),
   ```

3. Rewrite `MagnetRun.prepareData()` to:
   - Accept an optional `config: SiteConfig | None = None` argument.
   - If `config` is None, look up by `self.Site` from `SITE_CONFIGS`.
   - Apply `config.column_renames`, build `IH_ref`, `IB_ref` from `config.ih_ref_columns`.
   - Remove the chain of `if self.Housing == "M9": ... elif self.Housing == "M8": ...`

4. Keep `prepareData_legacy()` as a thin wrapper calling `prepareData()` with the
   legacy-derived config; do not change its signature.

5. Verify that all example files and tests using `prepareData` still produce identical
   output:
   ```bash
   pytest tests/ -v
   ```

---

## Task 2.6 — Support YAML/TOML site configuration files

**Problem:** Adding a new site (M1, M5, M7, M11 — all noted as TODO) requires editing
library source.

**Steps:**

1. Create `python_magnetrun/data/sites.yaml` bundled with the package:
   ```yaml
   M9:
     reference_gr1_current: IH
     reference_gr2_current: IB
     reference_gr1_flow: FlowH
     reference_gr2_flow: FlowB
     reference_gr1_rpm: RpmH
     reference_gr2_rpm: RpmB
     reference_gr1_pin: HPH
     reference_gr2_pin: HPB
     voltage_channels_gr1: [UH]
     voltage_channels_gr2: [UB, Ucoil15, Ucoil16]
     column_renames: {Flow1: FlowH, Flow2: FlowB}
     ih_ref_columns: [Idcct1, Idcct2]
     ib_ref_columns: [Idcct3, Idcct4]
   M8:
     ...
   M10:
     ...
   ```

2. Include the file in the package (update `pyproject.toml`):
   ```toml
   [tool.setuptools.package-data]
   "python_magnetrun" = ["data/*.yaml"]
   ```

3. Update `get_site_config()` in `analysis/config.py` to load from:
   a. `~/.config/magnetrun/sites.yaml` (user override) if it exists
   b. The bundled `data/sites.yaml` (built-in defaults)
   c. `SITE_CONFIGS` dict (hard-coded fallback)

   Use `tomllib` (Python ≥ 3.11 stdlib) or `pyyaml`:
   ```python
   import importlib.resources
   import yaml   # or tomllib for TOML format

   def _load_site_configs() -> dict[str, SiteConfig]:
       user_cfg = Path.home() / ".config" / "magnetrun" / "sites.yaml"
       if user_cfg.exists():
           with open(user_cfg) as f:
               raw = yaml.safe_load(f)
       else:
           pkg_data = importlib.resources.files("python_magnetrun") / "data" / "sites.yaml"
           raw = yaml.safe_load(pkg_data.read_text())
       return {name: SiteConfig(name=name, **values) for name, values in raw.items()}
   ```

4. Add `pyyaml` to core dependencies (or use `tomllib` + TOML format to avoid extra dep).

---

## Task 2.7 — Replace bare `except` blocks

**File:** `python_magnetrun/MagnetRun.py` and anywhere else with `except:` or
`except Exception:` that silently continues.

**Steps:**

1. Find all bare except blocks:
   ```bash
   grep -n "except:" python_magnetrun/*.py python_magnetrun/**/*.py
   grep -n "except Exception:" python_magnetrun/*.py python_magnetrun/**/*.py
   ```

2. For `MagnetRun.fromStringIO()` (the main offender):
   ```python
   # before
   except Exception:
       with open("wrongdata.txt", "w") as fout:
           fout.write(content)
       return None

   # after
   except (ValueError, pd.errors.ParserError) as exc:
       logger.exception("Failed to parse StringIO content: %s", exc)
       raise
   ```

3. Do not swallow exceptions without at minimum a `logger.exception()` call.

---

## Task 2.8 — Adopt `pathlib.Path` consistently

**Files:** `magnetdata/_loaders.py` (formerly `magnetdata.py`), `MagnetRun.py`,
`MRecord.py`, `fetchers/cli.py`.

**Replacement patterns:**

| Old | New |
|-----|-----|
| `os.path.exists(name)` | `Path(name).exists()` |
| `os.path.splitext(name)[-1]` | `Path(name).suffix` |
| `os.path.basename(name)` | `Path(name).name` |
| `os.path.join(a, b)` | `Path(a) / b` |
| `os.path.dirname(name)` | `Path(name).parent` |

Add `from pathlib import Path` at the top of each file modified.
Remove `import os` if it is no longer used (check for `os.environ` — keep if present,
replace with `os.environ.get(...)` calls which can coexist with pathlib).

---

## Task 2.9 — Replace hand-written serialization in `deserialize.py` with Pydantic

**File:** `python_magnetrun/deserialize.py`, `MRecord.py`, `GObject.py`, `HMagnet.py`

**Steps:**

1. Add `pydantic>=2.0` to `[project.dependencies]` in `pyproject.toml`.

2. Convert `MRecord` to a Pydantic model:
   ```python
   from pydantic import BaseModel, field_validator
   from datetime import datetime

   class MRecord(BaseModel):
       timestamp: datetime
       housing: str
       site: str
       link: str | None = None

       @field_validator("housing")
       @classmethod
       def housing_must_be_known(cls, v: str) -> str:
           known = {"M8", "M9", "M10"}
           if v not in known:
               raise ValueError(f"Unknown housing: {v!r}. Expected one of {known}")
           return v

       def to_json(self) -> str:
           return self.model_dump_json()

       @classmethod
       def from_json(cls, json_str: str) -> "MRecord":
           return cls.model_validate_json(json_str)
   ```

3. Do the same for `GObject` and `HMagnet`.

4. Delete `deserialize.py` once no code references `serialize_instance()`.

5. Verify:
   ```bash
   python -c "from python_magnetrun.MRecord import MRecord; print('OK')"
   pytest tests/test_mrecord.py -v  # if this file exists; create it if not
   ```

---

## Task 2.10 — Split `processing/hysteresis.py` (37 KB)

**Target:**

```
python_magnetrun/processing/hysteresis/
├── __init__.py      # re-exports all public names for backward compat
├── _analysis.py     # Loop detection algorithms
├── _fitting.py      # Linear, quadratic, exponential, power-law model classes
├── _plotting.py     # Visualization functions
└── _outliers.py     # Hysteresis-specific outlier removal
```

**Steps:**

1. Identify public names currently exported:
   ```bash
   grep -n "^def \|^class " python_magnetrun/processing/hysteresis.py
   ```

2. Distribute by concern into the four private modules.

3. In `__init__.py`, re-export everything:
   ```python
   from python_magnetrun.processing.hysteresis._analysis import *
   from python_magnetrun.processing.hysteresis._fitting  import *
   from python_magnetrun.processing.hysteresis._plotting import *
   from python_magnetrun.processing.hysteresis._outliers import *
   ```

4. Delete `python_magnetrun/processing/hysteresis.py`.

5. Verify imports from callers still work:
   ```bash
   grep -rn "from.*hysteresis import\|import.*hysteresis" python_magnetrun/ --include="*.py"
   pytest tests/ -v
   ```

---

## Task 2.11 — Add type annotations and configure mypy

**Steps:**

1. Create `python_magnetrun/py.typed` (empty file — PEP 561 marker):
   ```bash
   touch python_magnetrun/py.typed
   ```

2. Add to `pyproject.toml`:
   ```toml
   [tool.mypy]
   python_version = "3.11"
   ignore_missing_imports = true
   strict = false
   files = ["python_magnetrun"]
   ```

3. Annotate return types on all public API methods in the files touched during this phase.
   Priority:
   - `MagnetData.getData() -> pd.DataFrame | dict`
   - `MagnetData.getKeys() -> list[str]`
   - `MagnetRun.fromtxt() -> "MagnetRun"`
   - `MagnetRun.getData() -> pd.DataFrame`

4. Run mypy and fix any errors:
   ```bash
   mypy python_magnetrun/magnetdata/ python_magnetrun/MagnetRun.py
   ```

---

## Task 2.12 — Enforce ruff

Add to `pyproject.toml` (or update existing `[tool.ruff]`):

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
ignore  = []

[tool.ruff.lint.isort]
known-first-party = ["python_magnetrun"]
```

Run `ruff check --fix python_magnetrun/` and commit the result as a single formatting
commit (separate from logic changes).

---

## Verification Checklist

```bash
# 1. All tests pass
pytest tests/ -v

# 2. CLI entry points work
python-magnetrun --help
srvdata-to-magnetrun --help
magnetrun-analysis --help

# 3. Import paths unbroken
python -c "from python_magnetrun import MagnetData, MagnetRun, MRecord"
python -c "from python_magnetrun.magnetdata import MagnetData"
python -c "from python_magnetrun.fetchers.cli import main"

# 4. No more housing if-chains in prepareData
grep -n "Housing == " python_magnetrun/MagnetRun.py  # should be empty

# 5. No bare except blocks
grep -rn "except:" python_magnetrun/ --include="*.py"  # should be empty

# 6. Ruff and mypy pass
ruff check python_magnetrun/
mypy python_magnetrun/

# 7. Deprecation shims work
python -W all -c "import python_magnetrun.requests" 2>&1 | grep DeprecationWarning
```

---

## Commit Strategy

Group commits logically:

```
refactor: rename requests/ to fetchers/, add deprecation shim
refactor: rename python_magnetrun.py to cli_main.py
refactor(magnetdata): split 1300-line file into magnetdata/ subpackage
refactor(cli): split cli_main.py into commands/ submodules
refactor(config): centralize housing config in SITE_CONFIGS
feat(config): support user YAML site config at ~/.config/magnetrun/sites.yaml
fix: replace bare except blocks with specific exception handling
refactor: adopt pathlib.Path consistently
refactor(serialization): replace deserialize.py with pydantic models
refactor(hysteresis): split 37KB file into processing/hysteresis/ subpackage
chore: add py.typed marker, configure mypy
chore(lint): enforce ruff E/F/I/UP/B rules
```
