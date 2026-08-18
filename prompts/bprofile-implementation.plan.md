# Plan: Implement B-Profile Data Loading

## Context

B-profile files (`M9_profiles.txt`, 3000 rows, comma-separated) store the axial magnetic
field profile `(Position mm, Profile %)` for a magnet at one or more operating conditions.
They are the data source behind `BFieldRun` (Phase C of `cross-domain-comparison.prompt.md`).

**Current state — partially broken:**

| Component | Status | Problem |
|---|---|---|
| `BProfileMagnetData.frombprofile()` | **Broken** | `sep=r"\s+"` fails on comma-separated files |
| `BFieldRun.from_bprofile()` | **Broken** | inherits the separator bug |
| `validate_bprofile_format()` | **Missing** | generic `validate_csv_format()` used |
| `bprofile-defs.json` | **Missing** | no field definitions file |
| `load_magnetdata(fmt="bprofile")` | **Missing** | no dispatch path |
| Test in `test_magnetdata.py` | **Workaround** | uses `fromcsv()` instead of `frombprofile()` |
| `tests/test_bprofile.py` | **Missing** | only protocol tests in `test_protocol.py` |

The realistic test at `tests/test_magnetdata.py:1188` silently works around the bug by
calling `PandasMagnetData.fromcsv()` instead of `BProfileMagnetData.frombprofile()`.

---

## File Format

```
Index,Position (mm),Profile at Tr (%),Profile at max (%)
0,299.7,-62.548..., -51.836...
...
2998,599.5,-62.598..., -51.878...
```

- Separator: `,` (comma)
- Header: single line with required columns `Index`, `Position (mm)`
- No `Date`/`Time` columns; purely spatial data
- File named `<site>_profiles.txt` (e.g. `M9_profiles.txt`)

---

## Files to Modify / Create

| File | Change |
|---|---|
| `python_magnetrun/utils/validation.py` | Add `validate_bprofile_format()` |
| `python_magnetrun/magnetdata_pandas.py` | Fix `BProfileMagnetData.frombprofile()`: sep + lazy loading |
| `python_magnetrun/magnetdata.py` | Add `fmt="bprofile"` dispatch in `load_magnetdata()` |
| `python_magnetrun/bfield_run.py` | No change needed (inherits fix from frombprofile) |
| `python_magnetrun/bprofile-defs.json` | **New** — 4-column definitions |
| `tests/data/sample_bprofile.txt` | **New** — 10-row fixture |
| `tests/test_bprofile.py` | **New** — full test suite |
| `tests/test_magnetdata.py` | Fix `TestRealisticM9Profiles` to use `frombprofile()` |

---

## Step-by-Step Implementation

### 1. `utils/validation.py` — add `validate_bprofile_format`

Following the `FILE_FORMAT_VALIDATION.md` pattern: check existence, extension is not
enforced (`.txt` and `.csv` both used in the wild), and that the required header
tokens are present.

```python
def validate_bprofile_format(path: str) -> None:
    """Validate a B-profile CSV file: exists, non-empty, Index + Position columns present.

    :raises FileNotFoundError: if *path* does not exist
    :raises FileFormatError: if the file fails structural checks
    """
    validate_file_exists(path)
    if os.path.getsize(path) == 0:
        raise FileFormatError(f"{path}: file is empty")
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            header = f.readline()
    except OSError as exc:
        raise FileFormatError(f"{path}: cannot read: {exc}") from exc
    if not header.strip():
        raise FileFormatError(f"{path}: header line is empty")
    for required in ("Index", "Position"):
        if required not in header:
            raise FileFormatError(
                f"{path}: missing required column '{required}' in header — not a B-profile file"
            )
```

Also export `validate_bprofile_format` from `utils/__init__.py`.

---

### 2. `magnetdata_pandas.py` — fix `BProfileMagnetData.frombprofile()`

Two fixes:
- Change `sep=r"\s+"` → `sep=","` (actual format is comma-separated)
- Add lazy loading via `_read_kwargs` (same pattern as `fromtxt()`)
- Replace generic `validate_csv_format` with `validate_bprofile_format`

```python
@classmethod
def frombprofile(
    cls, name: str, defs_file: str | None = "bprofile-defs.json"
) -> BProfileMagnetData:
    """Create from a B-profile CSV file (Index, Position, Profile columns).

    Lazy-loads: only the first row is read at construction time; the full
    file is loaded on first data access via _ensure_data_loaded().
    """
    from .utils.validation import validate_bprofile_format

    validate_bprofile_format(name)
    _csv_kwargs = {
        "sep": ",",
        "engine": "python",
        "skiprows": 0,
        "on_bad_lines": "warn",
    }
    with _open_text_with_fallback(name) as f:
        stub = pd.read_csv(f, **_csv_kwargs, nrows=1)
    if stub.empty:
        from .utils.validation import FileFormatError
        raise FileFormatError(f"{name}: no data rows found")
    Keys = _dataframe_keys(stub)
    return cls(name, {}, Keys, stub, defs_file=defs_file, _read_kwargs=_csv_kwargs)
```

---

### 3. `magnetdata.py` — extend `load_magnetdata()`

Follow the same `fmt=` kwarg pattern decided for HTS (see `hts-magnetdata.plan.md`):

```python
def load_magnetdata(
    filename: str,
    defs_file: str | None = None,
    fmt: str | None = None,
) -> MagnetDataBase:
    ...
    elif ext in (".txt", ".csv"):
        if fmt == "bprofile":
            return BProfileMagnetData.frombprofile(
                filename, defs_file=defs_file or "bprofile-defs.json"
            )
        if fmt == "hts":
            return HTSMagnetData.fromhts(filename, defs_file=defs_file or "hts-defs.json")
        if ext == ".txt":
            return PandasMagnetData.fromtxt(filename, defs_file=defs_file or "pupitre-defs.json")
        return PandasMagnetData.fromcsv(filename, defs_file=defs_file)
```

Also add `BProfileMagnetData` to `__all__` if not already present (it is — line 33 of `magnetdata.py`).

---

### 4. `python_magnetrun/bprofile-defs.json` — new file

Column names include special characters (`(mm)`, `(%)`) — these are the literal JSON
keys, which is valid. The `unit` values must be pint-parseable strings.

```json
{
  "_comment": "Field definitions for B-profile CSV files (M<site>_profiles.txt).",
  "Index": {
    "description": "Row index (0-based integer)",
    "symbol": "n",
    "unit": null
  },
  "Position (mm)": {
    "description": "Axial position along the magnet bore",
    "symbol": "z",
    "unit": "millimeter"
  },
  "Profile at Tr (%)": {
    "description": "Normalized field at transition current (% of central on-axis field)",
    "symbol": "B/B0",
    "unit": "percent"
  },
  "Profile at max (%)": {
    "description": "Normalized field at maximum current (% of central on-axis field)",
    "symbol": "B/B0",
    "unit": "percent"
  }
}
```

Note: `"percent"` must be registered in pint via `_make_ureg()` (it already is — defined
in `magnetdata_base.py:66` as `"percent = 1 / 100 = %"`).

---

### 5. `tests/data/sample_bprofile.txt` — new fixture

Create a 10-row trimmed file with the exact header from `M9_profiles.txt`:

```
Index,Position (mm),Profile at Tr (%),Profile at max (%)
0,299.7,-62.54857446812466,-51.83603303013679
1,299.8,-62.515206120643384,-51.80767721498118
...
9,300.6,-62.24655465166157,-51.57970625563604
```

---

### 6. `tests/test_bprofile.py` — new test file

```python
from pathlib import Path
import pandas as pd
import pytest
from python_magnetrun.magnetdata import load_magnetdata
from python_magnetrun.magnetdata_pandas import BProfileMagnetData
from python_magnetrun.bfield.bfield_run import BFieldRun
from python_magnetrun.utils.validation import FileFormatError

SAMPLE_BPROFILE = Path(__file__).parent / "data" / "sample_bprofile.txt"


class TestBProfileMagnetData:
    def test_frombprofile_returns_instance(self):
        md = BProfileMagnetData.frombprofile(str(SAMPLE_BPROFILE))
        assert isinstance(md, BProfileMagnetData)

    def test_type_is_zero(self):
        md = BProfileMagnetData.frombprofile(str(SAMPLE_BPROFILE))
        assert md.getType() == 0

    def test_keys_contain_required_columns(self):
        md = BProfileMagnetData.frombprofile(str(SAMPLE_BPROFILE))
        assert "Index" in md.getKeys()
        assert "Position (mm)" in md.getKeys()

    def test_data_is_dataframe(self):
        md = BProfileMagnetData.frombprofile(str(SAMPLE_BPROFILE))
        df = md.getData()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_magnetdata_fmt_bprofile(self):
        md = load_magnetdata(str(SAMPLE_BPROFILE), fmt="bprofile")
        assert isinstance(md, BProfileMagnetData)

    def test_invalid_file_raises(self, tmp_path):
        bad = tmp_path / "bad.txt"
        bad.write_text("col1,col2\n1,2\n")
        with pytest.raises(FileFormatError):
            BProfileMagnetData.frombprofile(str(bad))

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        with pytest.raises(FileFormatError):
            BProfileMagnetData.frombprofile(str(empty))


class TestBFieldRun:
    def test_from_bprofile_returns_instance(self):
        brun = BFieldRun.from_bprofile(str(SAMPLE_BPROFILE), housing="M9")
        assert isinstance(brun, BFieldRun)

    def test_getkeys_delegation(self):
        brun = BFieldRun.from_bprofile(str(SAMPLE_BPROFILE))
        assert "Position (mm)" in brun.getKeys()

    def test_getdata_delegation(self):
        brun = BFieldRun.from_bprofile(str(SAMPLE_BPROFILE))
        df = brun.getData()
        assert isinstance(df, pd.DataFrame)

    def test_getdomain(self):
        brun = BFieldRun.from_bprofile(str(SAMPLE_BPROFILE))
        assert brun.getDomain() == "bfield"
```

---

### 7. Fix `tests/test_magnetdata.py::TestRealisticM9Profiles`

Change the fixture to use `BProfileMagnetData.frombprofile()` instead of the
`fromcsv()` workaround:

```python
# Before (workaround for broken frombprofile):
return PandasMagnetData.fromcsv(str(M9_PROFILES))

# After (correct):
from python_magnetrun.magnetdata_pandas import BProfileMagnetData
return BProfileMagnetData.frombprofile(str(M9_PROFILES))
```

---

## Verification

```bash
source magnetrun-env/bin/activate

# Run new test suite
pytest tests/test_bprofile.py -v

# Confirm the fixed realistic test passes
pytest tests/test_magnetdata.py::TestRealisticM9Profiles -v

# Confirm protocol tests still pass (BFieldRun satisfies DataLoader)
pytest tests/test_protocol.py -v

# Full suite regression check
pytest tests/ -v

# Smoke-test with actual file
python -c "
from python_magnetrun.magnetdata import load_magnetdata
md = load_magnetdata('M9_profiles.txt', fmt='bprofile')
print('Type:', md.getType())
print('Keys:', md.getKeys())
md.Units()
df = md.getData()
print('Shape:', df.shape)
print(df.head(3))
"

# BFieldRun integration
python -c "
from python_magnetrun.bfield.bfield_run import BFieldRun
brun = BFieldRun.from_bprofile('M9_profiles.txt', housing='M9', site='grenoble')
print('Domain:', brun.getDomain())
print('Keys:', brun.getKeys())
df = brun.getData()
print('Shape:', df.shape)
"
```
