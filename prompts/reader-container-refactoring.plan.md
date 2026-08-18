# Reader/Container Split Refactoring Plan

*Created: 2026-05-28*

## Goal

Format-parsing logic (separator, skip rows, encoding, channel naming) is currently fused
into the `PandasMagnetData` / `TdmsMagnetData` container classes as factory classmethods,
making those files ~1000+ lines each. Extracting dedicated reader objects:

- Makes adding new formats (HTS, new feelpp/paraview variants) a one-class change
- Isolates format spec so it is explicit and independently testable
- Enables `HybridData` to join the `MagnetDataBase` hierarchy (highest-value outcome)
- Allows HTS to switch from CSV to TDMS by swapping the reader only

Public API (`load_magnetdata()`, `MagnetDataBase` interface) is **unchanged**.

---

## Layering

```
DataLoader protocol / ComparisonSession         ← cross-domain plan (unaffected)
       ↑
MagnetDataBase / PandasMagnetData / TdmsMagnetData  ← container layer (shrinks)
       ↑
readers/ subpackage                             ← new: pure I/O, no data manipulation
       ↑
*-defs.json + load_units_from_json()            ← pattern feature (Phase H, separate plan)
```

---

## Target layout

```
python_magnetrun/
  readers/
    __init__.py        # exports Reader protocol + READERS registry
    base.py            # Reader Protocol / ABC
    csv_readers.py     # PupitreReader, BProfileReader, EnsightReader, FeelppReader, CsvReader
    tdms_reader.py     # TdmsReader
    hts_reader.py      # HtsReader  (new format — ';' sep, units-in-header)
    hybrid_reader.py   # HybridReader (composite over kHz/rms/trigger sub-readers)
    registry.py        # READERS dict + detect_type()
```

---

## Reader Protocol (readers/base.py)

```python
from typing import Protocol, runtime_checkable
from pathlib import Path
import pandas as pd

@runtime_checkable
class Reader(Protocol):
    def read(self, path: Path) -> pd.DataFrame | dict[str, pd.DataFrame]: ...
    def validate(self, path: Path) -> bool: ...
```

---

## Phase R1 — Extract CSV readers

**File:** `python_magnetrun/readers/csv_readers.py`

Each class holds only the `pd.read_csv` parameters currently buried in factory classmethods.
No logic is moved — only the *configuration* is extracted.

```python
class PupitreReader:
    sep = r"\s+"
    engine = "python"
    skip_rows = 1
    on_bad_lines = "warn"
    defs_file = "pupitre-defs.json"

    def read(self, path: Path) -> pd.DataFrame:
        with _open_text_with_fallback(path) as f:
            return pd.read_csv(f, sep=self.sep, engine=self.engine,
                               skiprows=self.skip_rows, on_bad_lines=self.on_bad_lines)

    def read_stub(self, path: Path) -> pd.DataFrame:
        """Read first row only — used for lazy loading construction."""
        with _open_text_with_fallback(path) as f:
            return pd.read_csv(f, sep=self.sep, engine=self.engine,
                               skiprows=self.skip_rows, on_bad_lines=self.on_bad_lines,
                               nrows=1)

    def read_kwargs(self) -> dict:
        """Return kwargs for pd.read_csv — stored on the container for lazy loading."""
        return {"sep": self.sep, "engine": self.engine,
                "skiprows": self.skip_rows, "on_bad_lines": self.on_bad_lines}

    def validate(self, path: Path) -> bool:
        from ..utils.validation import validate_txt_format
        validate_txt_format(str(path))
        return True


class BProfileReader:
    sep = r"\s+"       # note: comma was wrong in earlier versions
    engine = "python"
    skip_rows = 0
    expected_cols = ["Index", "Position", "Profile"]
    defs_file = None


class EnsightReader:
    sep = ","
    engine = "python"
    skip_rows = 2
    defs_file = None


class FeelppReader:
    sep = ","
    engine = "python"
    skip_rows: int = 0   # configurable per-file via constructor arg
    defs_file = "feelpp-defs.json"

    def __init__(self, skip_rows: int = 0):
        self.skip_rows = skip_rows


class CsvReader:
    sep = ","
    engine = "python"
    skip_rows = 0
    on_bad_lines = "warn"
    defs_file = None
```

**Wire into existing factory methods** — each becomes a one-liner that delegates:

```python
# magnetdata_pandas.py  (after)
@classmethod
def fromtxt(cls, name: str, defs_file="pupitre-defs.json"):
    from .readers.csv_readers import PupitreReader
    reader = PupitreReader()
    reader.validate(name)
    check_pupitre_truncation(name, ...)
    stub = reader.read_stub(name)
    Keys = _dataframe_keys(stub)
    return cls(name, {}, Keys, stub, defs_file=defs_file, _read_kwargs=reader.read_kwargs())
```

No change to `PandasMagnetData.__init__`, `getData()`, `Units()`, or any public method.

**Files modified:** `magnetdata_pandas.py` (factory methods only, ~40 lines changed)
**Files created:** `readers/__init__.py`, `readers/base.py`, `readers/csv_readers.py`

---

## Phase R2 — Extract TdmsReader

**File:** `python_magnetrun/readers/tdms_reader.py`

Move TDMS-specific config out of `TdmsMagnetData._fromtdms()` / `magnetdata.py`:

```python
class TdmsReader:
    required_group = "Courants_Alimentations"
    t_offsets = {"Overview": 0.5, "Archive": 1 / 240.0}
    defs_file = "pigbrother-defs.json"

    def read(self, path: Path) -> dict[str, pd.DataFrame]:
        """Returns dict[group_name → DataFrame]; lazy loading handled by container."""
        ...

    def validate(self, path: Path) -> bool:
        """Checks magic bytes b'TDSm' at offset 0."""
        from ..utils.validation import validate_tdms_format
        validate_tdms_format(str(path))
        return True
```

`TdmsMagnetData._fromtdms()` delegates to `TdmsReader().validate()`.
Lazy group loading (`_LazyGroupDict`) stays in the container — it is data management, not parsing.

**Files modified:** `magnetdata_tdms.py` (factory method only), `magnetdata.py`
**Files created:** `readers/tdms_reader.py`

---

## Phase R3 — Add HtsReader (new format)

**File:** `python_magnetrun/readers/hts_reader.py`

HTS files use `;` separator and embed units in column headers (`"Temps [s]"` → column `Temps`,
unit `s`). This logic does not fit cleanly into the existing CSV path:

```python
class HtsReader:
    sep = ";"
    skip_rows = 0
    header_units = True     # parse "Col [unit]" from header
    defs_file = "feelpp-defs.json"  # or "hts-defs.json" if created separately

    def read(self, path: Path) -> pd.DataFrame:
        # 1. Read header line → extract column names and units
        # 2. Read data with cleaned column names
        # 3. Return DataFrame with normalised column names
        ...

    def extracted_units(self, path: Path) -> dict[str, str]:
        """Return {column_name: unit_string} parsed from the header."""
        ...
```

**Add `DataType.HTS = 4`** to enum in `magnetdata_base.py`.

When HTS switches to TDMS (anticipated): add `HtsTdmsReader`, update registry entry — container unchanged.

**Files modified:** `magnetdata_base.py` (enum only)
**Files created:** `readers/hts_reader.py`

---

## Phase R4 — HybridReader + HybridData joins MagnetDataBase

This is the highest-value change. It removes all `isinstance(data, HybridData)` branches
in `processing.py` and the plotting code.

### HybridReader (readers/hybrid_reader.py)

```python
class HybridReader:
    """Composite reader — delegates to existing sub-readers in hybrid/."""

    sub_readers = {
        "kHz":      "hybrid.kHz.fepc_reader.FepcReader",
        "rms":      "hybrid.rms.rms_reader.RmsReader",
        "trigger":  "hybrid.trigger.trigger_reader.TriggerReader",
        "vprocess": "hybrid.vprocess.vprocess_reader.VProcessReader",
    }

    def read(self, base_dir: Path) -> dict[str, pd.DataFrame]:
        result = {}
        for name, reader_path in self.sub_readers.items():
            subdir = base_dir / name
            if subdir.exists():
                ReaderClass = _import_reader(reader_path)
                result[name] = ReaderClass().read(subdir)
        return result
```

### HybridData inherits MagnetDataBase

`HybridData.Data` is already `dict[str, DataFrame]` — structurally identical to `TdmsMagnetData.Data`.

Changes to `hybrid/hybrid_data.py`:
- Inherit from `MagnetDataBase` instead of being a standalone class
- Remove the 4 `NotImplementedError` stubs (`addData`, `saveData`, `computeData`, `extractData`)
- Keep all existing `read_khz_variable`, `load_rms_variable`, `getInfo` methods

```python
# before
class HybridData:
    ...

# after
class HybridData(MagnetDataBase):
    _TYPE: DataType = DataType.HYBRID
    ...
```

**Files modified:** `hybrid/hybrid_data.py`, any callers with `isinstance(data, HybridData)` branches
**Files created:** `readers/hybrid_reader.py`

---

## Phase R5 — Reader registry + load_magnetdata() cleanup

**File:** `python_magnetrun/readers/registry.py`

```python
from ..magnetdata_base import DataType

READERS: dict[DataType, type] = {
    DataType.PUPITRE:  PupitreReader,
    DataType.BPROFILE: BProfileReader,
    DataType.HTS:      HtsReader,
    DataType.TDMS:     TdmsReader,
    DataType.ENSIGHT:  EnsightReader,
    DataType.HYBRID:   HybridReader,
}

CONTAINERS: dict[DataType, type] = {
    DataType.PUPITRE:  PandasMagnetData,
    DataType.BPROFILE: PandasMagnetData,   # BProfileMagnetData can be dissolved
    DataType.HTS:      PandasMagnetData,
    DataType.TDMS:     TdmsMagnetData,
    DataType.ENSIGHT:  PandasMagnetData,   # EnsightMagnetData can be dissolved
    DataType.HYBRID:   HybridData,
}

def detect_type(path: Path, fmt: str | None = None) -> DataType:
    """Detect DataType from extension + optional explicit fmt override."""
    if fmt is not None:
        return DataType[fmt.upper()]
    suffix = path.suffix.lower()
    if suffix == ".tdms":
        return DataType.TDMS
    if suffix == ".txt":
        return DataType.PUPITRE
    if suffix == ".csv":
        return DataType.PUPITRE   # further disambiguation by content if needed
    raise ValueError(f"Cannot detect DataType for {path}")
```

`magnetdata.py:load_magnetdata()` simplifies to:

```python
def load_magnetdata(filename, fmt=None, defs_file=None):
    from .readers.registry import READERS, CONTAINERS, detect_type
    data_type = detect_type(Path(filename), fmt)
    reader = READERS[data_type](defs_file=defs_file)
    raw = reader.read(filename)
    container_cls = CONTAINERS[data_type]
    return container_cls.from_raw(raw, data_type, defs_file=defs_file)
```

**Thin subclasses that exist only for format params** (`BProfileMagnetData`, `EnsightMagnetData`,
`FeelppMagnetData`) can optionally be dissolved — they become `PandasMagnetData` constructed
with a different reader. This is a cleanup step and can be deferred.

**Files modified:** `magnetdata.py`
**Files created:** `readers/registry.py`

---

## Migration order

```
R1 (CSV readers)      safe, no behaviour change — test each reader in isolation
R2 (TdmsReader)       safe, validates via magic-bytes check
R3 (HtsReader)        additive only (new DataType, new file)
R4 (HybridData)       highest impact; unblocks cross-domain Phase E
R5 (registry)         cleanup after R1–R4 all pass tests
```

---

## Files summary

| Action | Files |
|--------|-------|
| Create | `readers/__init__.py`, `readers/base.py`, `readers/csv_readers.py` |
| Create | `readers/tdms_reader.py`, `readers/hts_reader.py`, `readers/hybrid_reader.py` |
| Create | `readers/registry.py` |
| Modify | `magnetdata_pandas.py` — factory methods delegate to readers (~40 lines) |
| Modify | `magnetdata_tdms.py` — `_fromtdms()` delegates to `TdmsReader` |
| Modify | `magnetdata.py` — `load_magnetdata()` uses registry |
| Modify | `magnetdata_base.py` — add `DataType.HTS = 4` |
| Modify | `hybrid/hybrid_data.py` — inherit `MagnetDataBase`, remove `NotImplementedError` stubs |
| Modify | `processing.py`, plotting code — remove `isinstance(data, HybridData)` branches |

---

## Verification

```bash
# All existing tests must stay green
pytest tests/ -x -q

# New reader unit tests (one per reader class)
pytest tests/readers/ -v

# HybridData hierarchy
python -c "
from python_magnetrun.hybrid.hybrid_data import HybridData
from python_magnetrun.magnetdata_base import MagnetDataBase
assert issubclass(HybridData, MagnetDataBase)
print('HybridData hierarchy: OK')
"
```
