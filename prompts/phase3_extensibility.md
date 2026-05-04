# Prompt: Phase 3 — Extensibility

## Context

`python_magnetrun` is a scientific data-analysis package for high-field magnet facility
runs. This prompt covers **Phase 3** of the improvement plan: making the package
extensible without touching library internals — new data formats, new statistics, and new
analysis algorithms become first-class citizens.

**Prerequisite:** Phase 1 and Phase 2 must be complete, all tests green.

Reference document: `IMPROVEMENT_PLAN.md` §Phase 3.

---

## Objective

Introduce pluggable design patterns so that:
1. New file formats can be added by writing a single loader class.
2. New statistics can be registered with a decorator.
3. New signal-processing algorithms integrate into the CLI automatically.
4. Large FEPC runs can be efficiently stored and shared with the database backend.

---

## Task 3.1 — Define a `DataLoader` protocol

**File to create:** `python_magnetrun/protocols.py`

This generalises the pattern already started in `hybrid/data_protocol.py`.

```python
"""
Protocols for python_magnetrun plugin interfaces.

Third-party packages can implement these protocols to extend
python_magnetrun without modifying library source code.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from python_magnetrun.magnetdata import MagnetData


@runtime_checkable
class DataLoader(Protocol):
    """
    Protocol for data source loaders.

    Implement this protocol to add support for a new file format.

    Examples
    --------
    Register your loader so that MagnetData.from_file() can use it::

        from python_magnetrun.magnetdata import register_loader
        from python_magnetrun.protocols import DataLoader

        class MyLoader:
            @classmethod
            def can_load(cls, path: str) -> bool:
                return path.endswith(".myformat")

            @classmethod
            def load(cls, path: str, **kwargs) -> MagnetData:
                ...

            def get_format_name(self) -> str:
                return "myformat"

        register_loader(".myformat", MyLoader)
    """

    @classmethod
    def can_load(cls, path: str) -> bool:
        """Return True if this loader can handle the given file path."""
        ...

    @classmethod
    def load(cls, path: str, **kwargs) -> "MagnetData":
        """Load data from path and return a MagnetData instance."""
        ...

    def get_format_name(self) -> str:
        """Human-readable format name (e.g. 'TDMS', 'Pupitre TXT')."""
        ...


@runtime_checkable
class StatPlugin(Protocol):
    """
    Protocol for statistics plugins.

    Implement this to add custom statistics to MagnetData.stats().
    """

    def __call__(self, df: "pd.DataFrame") -> dict[str, float | int | str]:
        """Compute statistics and return as a dict."""
        ...
```

---

## Task 3.2 — Format auto-detection factory method

**File:** `python_magnetrun/magnetdata/_loaders.py` (or `_core.py`, wherever the
class is defined after Phase 2 split)

**Steps:**

1. Add a loader registry dict and a `register_loader` function:

```python
from pathlib import Path
from python_magnetrun.protocols import DataLoader

_LOADER_REGISTRY: dict[str, type] = {}


def register_loader(extension: str, loader_cls: type) -> None:
    """
    Register a DataLoader for a file extension.

    Parameters
    ----------
    extension : str
        File extension including dot (e.g., ".txt", ".tdms").
    loader_cls : type
        Class implementing the DataLoader protocol.
    """
    if not extension.startswith("."):
        raise ValueError(f"Extension must start with '.', got: {extension!r}")
    _LOADER_REGISTRY[extension.lower()] = loader_cls
```

2. Wrap the existing factory methods as loader classes implementing `DataLoader`:

```python
class _TxtLoader:
    @classmethod
    def can_load(cls, path: str) -> bool:
        return Path(path).suffix.lower() == ".txt"

    @classmethod
    def load(cls, path: str, **kwargs) -> "MagnetData":
        return MagnetData.fromtxt(path, **kwargs)

    def get_format_name(self) -> str:
        return "Pupitre TXT"


class _TdmsLoader:
    @classmethod
    def can_load(cls, path: str) -> bool:
        return Path(path).suffix.lower() == ".tdms"

    @classmethod
    def load(cls, path: str, **kwargs) -> "MagnetData":
        return MagnetData.fromtdms(path, **kwargs)

    def get_format_name(self) -> str:
        return "PigBrother TDMS"


class _CsvLoader:
    @classmethod
    def can_load(cls, path: str) -> bool:
        return Path(path).suffix.lower() == ".csv"

    @classmethod
    def load(cls, path: str, **kwargs) -> "MagnetData":
        return MagnetData.fromcsv(path, **kwargs)

    def get_format_name(self) -> str:
        return "CSV"


# Register built-in loaders at module import time
register_loader(".txt",  _TxtLoader)
register_loader(".tdms", _TdmsLoader)
register_loader(".csv",  _CsvLoader)
```

3. Add `from_file()` classmethod to `MagnetData`:

```python
@classmethod
def from_file(cls, path: str, **kwargs) -> "MagnetData":
    """
    Load data from a file, auto-detecting format by extension.

    Parameters
    ----------
    path : str
        Path to the data file.
    **kwargs
        Additional arguments forwarded to the loader.

    Returns
    -------
    MagnetData

    Raises
    ------
    ValueError
        If the file extension is not registered.

    Examples
    --------
    >>> data = MagnetData.from_file("run_20240315.txt")
    >>> data = MagnetData.from_file("pigbrother.tdms")
    """
    from python_magnetrun.magnetdata._loaders import _LOADER_REGISTRY
    suffix = Path(path).suffix.lower()
    loader_cls = _LOADER_REGISTRY.get(suffix)
    if loader_cls is None:
        supported = sorted(_LOADER_REGISTRY.keys())
        raise ValueError(
            f"No loader registered for extension {suffix!r}. "
            f"Supported: {supported}"
        )
    return loader_cls.load(path, **kwargs)
```

4. Export `register_loader` from the package:
   ```python
   # python_magnetrun/__init__.py
   from python_magnetrun.magnetdata._loaders import register_loader
   ```

5. Test:
   ```python
   # tests/test_loader_registry.py
   from python_magnetrun.magnetdata._loaders import register_loader, _LOADER_REGISTRY

   def test_register_custom_loader(tmp_path):
       class FakeLoader:
           @classmethod
           def can_load(cls, path): return path.endswith(".fake")
           @classmethod
           def load(cls, path, **kw): ...
           def get_format_name(self): return "Fake"

       register_loader(".fake", FakeLoader)
       assert ".fake" in _LOADER_REGISTRY

   def test_from_file_unknown_extension():
       import pytest
       from python_magnetrun import MagnetData
       with pytest.raises(ValueError, match="No loader registered"):
           MagnetData.from_file("data.xyz")
   ```

---

## Task 3.3 — Statistics plugin registry

**File to create:** `python_magnetrun/processing/registry.py`

```python
"""
Plugin registries for statistics and signal-processing algorithms.

Usage::

    from python_magnetrun.processing.registry import register_stat

    @register_stat("entropy")
    def compute_entropy(df: pd.DataFrame) -> dict[str, float]:
        import scipy.stats
        return {"entropy": float(scipy.stats.entropy(df.values.flatten()))}
"""
from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Statistics registry
# ---------------------------------------------------------------------------
_STAT_PLUGINS: dict[str, Callable[[pd.DataFrame], dict]] = {}


def register_stat(name: str) -> Callable:
    """
    Decorator to register a statistics plugin.

    Parameters
    ----------
    name : str
        Unique plugin name used to reference it in getStats(extra=[...]).

    Returns
    -------
    Callable
        The decorated function, unchanged.

    Examples
    --------
    >>> @register_stat("rms")
    ... def compute_rms(df):
    ...     return {"rms": float(np.sqrt((df**2).mean().mean()))}
    """
    def decorator(fn: Callable) -> Callable:
        if name in _STAT_PLUGINS:
            logger.warning("Overriding existing stat plugin: %r", name)
        _STAT_PLUGINS[name] = fn
        return fn
    return decorator


def run_stat_plugins(
    df: pd.DataFrame,
    plugins: list[str] | None = None,
) -> dict[str, object]:
    """
    Run registered statistics plugins on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    plugins : list[str] or None
        Plugin names to run. If None, runs all registered plugins.

    Returns
    -------
    dict
        Merged results from all selected plugins.
    """
    selected = plugins if plugins is not None else list(_STAT_PLUGINS.keys())
    results: dict = {}
    for plugin_name in selected:
        fn = _STAT_PLUGINS.get(plugin_name)
        if fn is None:
            logger.warning("Unknown stat plugin: %r. Available: %s",
                           plugin_name, list(_STAT_PLUGINS.keys()))
            continue
        try:
            results.update(fn(df))
        except Exception:
            logger.exception("Stat plugin %r failed", plugin_name)
    return results


# ---------------------------------------------------------------------------
# Smoother registry
# ---------------------------------------------------------------------------
_SMOOTHERS: dict[str, Callable] = {}


def register_smoother(name: str) -> Callable:
    """Decorator to register a smoothing algorithm."""
    def decorator(fn: Callable) -> Callable:
        _SMOOTHERS[name] = fn
        return fn
    return decorator


def get_smoother(name: str) -> Callable:
    """
    Retrieve a registered smoother by name.

    Raises
    ------
    KeyError
        If the smoother is not registered.
    """
    if name not in _SMOOTHERS:
        raise KeyError(
            f"Unknown smoother: {name!r}. Available: {sorted(_SMOOTHERS.keys())}"
        )
    return _SMOOTHERS[name]


# ---------------------------------------------------------------------------
# Detector registry  (plateau, breakpoint, anomaly detectors)
# ---------------------------------------------------------------------------
_DETECTORS: dict[str, Callable] = {}


def register_detector(name: str) -> Callable:
    """Decorator to register a signal detector."""
    def decorator(fn: Callable) -> Callable:
        _DETECTORS[name] = fn
        return fn
    return decorator


def get_detector(name: str) -> Callable:
    """Retrieve a registered detector by name."""
    if name not in _DETECTORS:
        raise KeyError(
            f"Unknown detector: {name!r}. Available: {sorted(_DETECTORS.keys())}"
        )
    return _DETECTORS[name]
```

**Wire up built-in algorithms:**

In each existing algorithm module, add registration at module level:

```python
# processing/smoothers.py (after existing function definitions)
from python_magnetrun.processing.registry import register_smoother

register_smoother("savgol")(savgol_smooth)     # existing function
register_smoother("lowess")(lowess_smooth)
```

```python
# processing/plateaux.py
from python_magnetrun.processing.registry import register_detector
register_detector("plateau")(detect_plateau)
```

**Update `MagnetData.getStats()`** to accept `extra` parameter:

```python
def getStats(
    self,
    field: str,
    extra: list[str] | None = None,
) -> dict:
    """
    Compute statistics for a field.

    Parameters
    ----------
    field : str
        Column name to analyse.
    extra : list[str], optional
        Names of registered stat plugins to also run.
    """
    from python_magnetrun.processing.registry import run_stat_plugins
    base = self._compute_base_stats(field)   # existing logic
    if extra:
        df = self._require_data()
        if isinstance(df, pd.DataFrame):
            base.update(run_stat_plugins(df[[field]], plugins=extra))
    return base
```

---

## Task 3.4 — `addData()` callable support

**File:** `python_magnetrun/magnetdata/_transforms.py`

**Problem:** Formula strings (`"IH_ref = Idcct1 + Idcct2"`) are fragile and invisible
to type checkers.

**Fix:** Accept both string formulas (legacy, preserved) and callables:

```python
from typing import Callable

def addData(
    self,
    key: str,
    formula: str | Callable[[pd.DataFrame], pd.Series],
    unit: str = "",
) -> None:
    """
    Add a computed column to the dataset.

    Parameters
    ----------
    key : str
        Name of the new column.
    formula : str or Callable
        Either a formula string (legacy, e.g. "Idcct1 + Idcct2")
        or a callable that receives the DataFrame and returns a Series.
    unit : str, optional
        Physical unit of the new column.

    Examples
    --------
    # Callable form (preferred):
    data.addData("IH_ref", lambda df: df["Idcct1"] + df["Idcct2"], unit="A")

    # String form (legacy, still supported):
    data.addData("IH_ref", "Idcct1 + Idcct2")
    """
    df = self._require_data()
    if callable(formula):
        df[key] = formula(df)
    else:
        # existing string eval path — preserved unchanged
        self._addData_legacy(key, formula)
    if unit:
        self.units[key] = unit
```

Add a test:

```python
# tests/test_magnetdata.py
def test_addData_callable(sample_magnetdata):
    sample_magnetdata.addData("sum_IH", lambda df: df["IH"] + df["IB"])
    assert "sum_IH" in sample_magnetdata.getKeys()

def test_addData_string_legacy(sample_magnetdata):
    # Existing string formula must still work
    sample_magnetdata.addData("IH_ref", "IH + IB")
    assert "IH_ref" in sample_magnetdata.getKeys()
```

---

## Task 3.5 — Parquet and HDF5 export in `saveData()`

**File:** `python_magnetrun/magnetdata/_stats.py` (or wherever `saveData` lives
after the Phase 2 split)

```python
from pathlib import Path

def saveData(
    self,
    filename: str,
    fmt: str | None = None,
    **kwargs,
) -> None:
    """
    Save the dataset to a file.

    Parameters
    ----------
    filename : str
        Output file path.
    fmt : str, optional
        Format: "csv", "parquet", or "hdf5".
        If None, inferred from the file extension.
    **kwargs
        Additional arguments forwarded to the pandas writer.

    Examples
    --------
    >>> run.saveData("output.csv")
    >>> run.saveData("output.parquet")
    >>> run.saveData("archive.h5", fmt="hdf5", key="magnetrun")
    """
    df = self._require_data()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("saveData requires a DataFrame (Type=PANDAS)")

    path = Path(filename)
    if fmt is None:
        fmt = {".csv": "csv", ".parquet": "parquet",
               ".h5": "hdf5", ".hdf5": "hdf5"}.get(path.suffix.lower(), "csv")

    match fmt:
        case "csv":
            df.to_csv(filename, **kwargs)
        case "parquet":
            df.to_parquet(filename, **kwargs)
        case "hdf5":
            key = kwargs.pop("key", "magnetrun")
            df.to_hdf(filename, key=key, **kwargs)
        case _:
            raise ValueError(
                f"Unknown format: {fmt!r}. Supported: csv, parquet, hdf5"
            )

    logger.info("Saved %d rows to %s (format=%s)", len(df), filename, fmt)
```

Add `pyarrow` or `fastparquet` as an optional dependency:

```toml
[project.optional-dependencies]
parquet = ["pyarrow>=15.0"]
```

---

## Task 3.6 — Expand test coverage

Create the following test files, using fixtures that load sample data from `data/`:

### `tests/test_magnetdata.py`

```python
"""Tests for MagnetData core functionality."""
import pytest
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def sample_txt_file():
    """Path to a sample Pupitre TXT file."""
    files = list(DATA_DIR.glob("*.txt"))
    if not files:
        pytest.skip("No .txt sample files in data/")
    return str(files[0])


@pytest.fixture
def sample_magnetdata(sample_txt_file):
    from python_magnetrun import MagnetData
    return MagnetData.fromtxt(sample_txt_file)


class TestFromTxt:
    def test_loads_successfully(self, sample_magnetdata):
        assert sample_magnetdata is not None

    def test_has_keys(self, sample_magnetdata):
        assert len(sample_magnetdata.getKeys()) > 0

    def test_data_is_dataframe(self, sample_magnetdata):
        assert isinstance(sample_magnetdata.getData(), pd.DataFrame)


class TestAddData:
    def test_callable_formula(self, sample_magnetdata):
        keys_before = set(sample_magnetdata.getKeys())
        sample_magnetdata.addData("_test_col", lambda df: df.iloc[:, 0] * 0)
        assert "_test_col" in sample_magnetdata.getKeys()

    def test_requires_data(self):
        from python_magnetrun import MagnetData
        md = MagnetData.__new__(MagnetData)
        md.Data = None
        with pytest.raises(RuntimeError, match="no loaded data"):
            md._require_data()


class TestFromFile:
    def test_auto_detect_txt(self, sample_txt_file):
        from python_magnetrun import MagnetData
        md = MagnetData.from_file(sample_txt_file)
        assert md is not None

    def test_unknown_extension_raises(self, tmp_path):
        from python_magnetrun import MagnetData
        import pytest
        f = tmp_path / "data.xyz"
        f.write_text("dummy")
        with pytest.raises(ValueError, match="No loader registered"):
            MagnetData.from_file(str(f))
```

### `tests/test_magnetrun.py`

```python
"""Tests for MagnetRun."""
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def sample_magnetrun(tmp_path):
    from python_magnetrun import MagnetRun
    files = list(DATA_DIR.glob("*.txt"))
    if not files:
        pytest.skip("No .txt sample files in data/")
    return MagnetRun.fromtxt(str(files[0]))


def test_getkeys(sample_magnetrun):
    assert len(sample_magnetrun.getKeys()) > 0


def test_getdata_returns_dataframe(sample_magnetrun):
    import pandas as pd
    assert isinstance(sample_magnetrun.getData(), pd.DataFrame)


def test_getstats_returns_dict(sample_magnetrun):
    keys = sample_magnetrun.getKeys()
    if keys:
        stats = sample_magnetrun.getStats(keys[0])
        assert isinstance(stats, dict)
```

### `tests/test_registry.py`

```python
"""Tests for the plugin registry."""
import pandas as pd
import numpy as np
from python_magnetrun.processing.registry import (
    register_stat, run_stat_plugins, _STAT_PLUGINS,
    register_smoother, get_smoother,
)


def test_register_and_run_stat():
    @register_stat("_test_mean")
    def _test_mean(df):
        return {"_test_mean": float(df.mean().mean())}

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    results = run_stat_plugins(df, plugins=["_test_mean"])
    assert "_test_mean" in results
    assert abs(results["_test_mean"] - 2.0) < 1e-9


def test_unknown_stat_does_not_raise():
    df = pd.DataFrame({"x": [1.0]})
    results = run_stat_plugins(df, plugins=["nonexistent_plugin"])
    assert results == {}


def test_register_smoother():
    def my_smoother(signal, **kwargs):
        return signal

    register_smoother("_test_smoother")(my_smoother)
    fn = get_smoother("_test_smoother")
    assert fn is my_smoother


def test_get_unknown_smoother_raises():
    import pytest
    with pytest.raises(KeyError, match="Unknown smoother"):
        get_smoother("nonexistent")
```

---

## Verification Checklist

```bash
# 1. All tests pass
pytest tests/ -v

# 2. from_file auto-detection works for all built-in formats
python -c "
from python_magnetrun import MagnetData
import glob, os
for f in glob.glob('data/*.txt')[:1]:
    md = MagnetData.from_file(f)
    print('txt OK:', md.getKeys()[:3])
"

# 3. Callable addData works
python -c "
from python_magnetrun import MagnetData
import glob
for f in glob.glob('data/*.txt')[:1]:
    md = MagnetData.fromtxt(f)
    md.addData('_test', lambda df: df.iloc[:,0] * 0)
    print('addData callable OK')
"

# 4. Custom stat plugin works
python -c "
from python_magnetrun.processing.registry import register_stat, run_stat_plugins
import pandas as pd
@register_stat('_check')
def _(df): return {'_check': 42}
r = run_stat_plugins(pd.DataFrame({'x':[1]}), ['_check'])
assert r['_check'] == 42
print('plugin OK')
"

# 5. Parquet save/load roundtrip
python -c "
import pandas as pd, tempfile, os
from python_magnetrun import MagnetData
import glob
for f in glob.glob('data/*.txt')[:1]:
    md = MagnetData.fromtxt(f)
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        md.saveData(tmp.name)
        df = pd.read_parquet(tmp.name)
        print('parquet roundtrip OK: rows=', len(df))
        os.unlink(tmp.name)
"

# 6. Registry-backed CLI: smoother flag dispatches correctly
python-magnetrun --help | grep smoother  # should show option if wired
```

---

## Commit Strategy

```
feat(protocols): add DataLoader and StatPlugin protocols
feat(magnetdata): add from_file() auto-detection factory
feat(loaders): wrap built-in formats as DataLoader classes
feat(registry): add stat/smoother/detector plugin registries
feat(magnetdata): addData() accepts callables
feat(magnetdata): saveData() supports parquet and hdf5 formats
test: add test_magnetdata, test_magnetrun, test_registry
```
