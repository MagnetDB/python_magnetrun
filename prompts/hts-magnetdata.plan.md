# Plan: Add HTSMagnetData for HTS Superconductor Files

## Context

HTS (High-Temperature Superconductor) magnet experiments produce `.txt` files with a
semicolon-separated format distinct from the existing pupitre `.txt` format. Example:
`20220420_test21_200-250-300A_05As_SF.txt` (56 channels, 143 K rows).

Differences from pupitre `.txt`:
- Separator is `;` (pupitre uses whitespace)
- Column names embed units: `Temps [s]`, `Ips [A]`, `Hall1_2519 [T]`, etc.
- Time column is numeric `Temps [s]` — no `Date`/`Time` string pair
- Filename prefix is `YYYYMMDD_` (not `YYYY.MM.DD - HH:MM:SS`)
- Trailing empty column from trailing `;`

Goal: make HTS files loadable as `HTSMagnetData` (a `PandasMagnetData` subclass),
plugged into `load_magnetdata(fmt="hts")`.

---

## Files to Modify / Create

| File | Change |
|---|---|
| `python_magnetrun/magnetdata_base.py` | Add `HTS = 4` to `DataType` enum |
| `python_magnetrun/utils/timestamps.py` | Add `parse_hts_filename()`, extend dispatcher |
| `python_magnetrun/utils/validation.py` | Add `validate_hts_format()` |
| `python_magnetrun/magnetdata_pandas.py` | Add `HTSMagnetData` class |
| `python_magnetrun/magnetdata.py` | Import, `__all__`, extend `load_magnetdata()` |
| `python_magnetrun/hts-defs.json` | **New** — 56-channel definitions |
| `tests/data/sample_hts.txt` | **New** — 10-row trimmed fixture |
| `tests/test_magnetdata_hts.py` | **New** — test suite |

---

## Step-by-Step Implementation

### 1. `magnetdata_base.py` — extend DataType

```python
class DataType(IntEnum):
    PUPITRE = 0
    TDMS = 1
    ENSIGHT = 2
    HYBRID = 3
    HTS = 4          # ← add this
```

---

### 2. `utils/timestamps.py` — add `parse_hts_filename`

The filename `20220420_test21_200-250-300A_05As_SF.txt` encodes only a date in its
first `_`-separated component (`20220420`). Parse it as `%Y%m%d` returning midnight.

```python
def parse_hts_filename(filename: str) -> datetime | None:
    """Parse date from HTS filename: YYYYMMDD_description.txt → datetime(Y,M,D,0,0,0)."""
    name, ext = os.path.splitext(os.path.basename(filename))
    if ext != ".txt":
        return None
    first = name.split("_")[0]
    try:
        return datetime.strptime(first, "%Y%m%d")
    except ValueError:
        return None
```

Also export it and update `parse_filename_timestamp()` — but only call it from
`HTSMagnetData.fromhts()` directly (not from the dispatcher, since the dispatcher
cannot distinguish HTS from pupitre `.txt` at the extension level).

---

### 3. `utils/validation.py` — add `validate_hts_format`

Check that the file is a `.txt` with `;` as separator and `Temps [s]` in the header.

```python
def validate_hts_format(path: str) -> None:
    """Validate an HTS .txt file: exists, .txt extension, semicolon-separated, Temps [s] header."""
    validate_file_exists(path)
    if os.path.splitext(path)[-1] != ".txt":
        raise FileFormatError(f"{path}: expected .txt extension")
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            header = f.readline()
    except OSError as exc:
        raise FileFormatError(f"{path}: cannot read: {exc}") from exc
    if ";" not in header:
        raise FileFormatError(f"{path}: not semicolon-separated — not an HTS file")
    if "Temps [s]" not in header:
        raise FileFormatError(f"{path}: missing 'Temps [s]' time column — not an HTS file")
```

---

### 4. `magnetdata_pandas.py` — add `HTSMagnetData`

Add a static helper and a thin subclass **after** the existing subclasses (line ~1090).

#### 4a. Column-strip helper

```python
_HTS_UNIT_MAP = {"s": "second", "A": "ampere", "T": "tesla", "V": "volt", "K": "kelvin"}

def _strip_hts_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename 'Name [unit]' columns → 'Name', alias 'Temps' → 't'.

    Returns (renamed_df, {clean_name: pint_unit_str}).
    """
    rename = {}
    auto_units: dict[str, str] = {}
    for col in df.columns:
        m = re.match(r"^(.*?)\s*\[([^\]]+)\]\s*$", str(col))
        if m:
            name, raw_unit = m.group(1).strip(), m.group(2).strip()
        else:
            name, raw_unit = str(col).strip(), None
        clean = "t" if name == "Temps" else name
        rename[col] = clean
        if raw_unit:
            auto_units[clean] = _HTS_UNIT_MAP.get(raw_unit, raw_unit)
    return df.rename(columns=rename), auto_units
```

Note: add `import re` at the top of the file if not already present.

#### 4b. `HTSMagnetData` class

```python
class HTSMagnetData(PandasMagnetData):
    """HTS superconductor .txt data (semicolon-separated, Type=4).

    Column names are normalised at load time: 'Name [unit]' → 'Name',
    'Temps [s]' → 't'. Units extracted from the header auto-populate
    self.units when no defs_file entry exists for a key.
    """

    _TYPE: DataType = DataType.HTS
    _hts_auto_units: dict[str, str]  # set by fromhts(), used by Units()

    @classmethod
    def fromhts(cls, name: str, defs_file: str | None = "hts-defs.json") -> HTSMagnetData:
        """Create from an HTS semicolon-separated .txt file (lazy-loading)."""
        from .utils.validation import validate_hts_format
        from .utils.timestamps import parse_hts_filename

        validate_hts_format(name)

        _csv_kwargs = {
            "sep": ";",
            "engine": "python",
            "skiprows": 0,
            "on_bad_lines": "warn",
        }
        with _open_text_with_fallback(name) as f:
            stub = pd.read_csv(f, **_csv_kwargs, nrows=1)

        # Drop the trailing empty column produced by the trailing semicolon.
        stub = stub.loc[:, stub.columns.str.strip() != ""]

        stub, auto_units = _strip_hts_columns(stub)
        if stub.empty:
            from .utils.validation import FileFormatError
            raise FileFormatError(f"{name}: no data rows found")

        Keys = _dataframe_keys(stub)

        # Store renaming so _ensure_data_loaded can reproduce it.
        instance = cls(name, {}, Keys, stub, defs_file=defs_file, _read_kwargs=_csv_kwargs)
        instance._hts_auto_units = auto_units

        # Override start_timestamp from HTS filename format (YYYYMMDD_...).
        dt = parse_hts_filename(name)
        if dt is not None:
            from .utils.timezone import local_to_utc_naive
            instance.start_timestamp = local_to_utc_naive(pd.Timestamp(dt), "Europe/Paris")

        return instance

    def _ensure_data_loaded(self) -> None:
        """Extend lazy loading to also strip column units after full read."""
        if self._data_loaded:
            return
        super()._ensure_data_loaded()
        # Post-process: drop trailing empty column and strip [unit] suffixes.
        df = self._data
        df = df.loc[:, df.columns.str.strip() != ""]
        df, auto_units = _strip_hts_columns(df)
        self._hts_auto_units = auto_units
        self._data = df
        self.Keys = _dataframe_keys(df)

    def Units(self, debug: bool = False, json_file: str | None = None) -> None:
        """Populate units: defs JSON first, then auto-extracted header units, then legacy patterns."""
        from .magnetdata_base import _make_ureg

        # 1. JSON defs (highest priority)
        resolved = json_file or self.defs_file
        if resolved is not None:
            self.load_units_from_json(resolved, debug=debug)

        ureg = _make_ureg()

        # 2. Auto-extracted from column headers
        auto = getattr(self, "_hts_auto_units", {})
        for key in self.Keys:
            if key in self.units:
                continue
            if key in auto:
                try:
                    self.units[key] = (key, ureg.parse_units(auto[key]))
                except Exception:
                    logger.warning(f"Units: cannot parse auto unit {auto[key]!r} for {key!r}")
            # 3. Legacy fallback (inherited behaviour)
            else:
                logger.warning(f"Units: no definition for {key!r}, skipping")
```

---

### 5. `magnetdata.py` — integrate

```python
# imports (add HTSMagnetData)
from .magnetdata_pandas import (
    BProfileMagnetData,
    EnsightMagnetData,
    FeelppMagnetData,
    HTSMagnetData,     # ← new
    PandasMagnetData,
)

# __all__ (add "HTSMagnetData")

# load_magnetdata signature
def load_magnetdata(
    filename: str,
    defs_file: str | None = None,
    fmt: str | None = None,           # ← new; "hts" to force HTS parsing
) -> MagnetDataBase:
    ...
    elif ext == ".txt":
        if fmt == "hts":
            return HTSMagnetData.fromhts(filename, defs_file=defs_file or "hts-defs.json")
        return PandasMagnetData.fromtxt(filename, defs_file=defs_file or "pupitre-defs.json")
```

---

### 6. `python_magnetrun/hts-defs.json` — new file

Create with entries for all 56 channels (stripped names). Example pattern:

```json
{
  "_comment": "Field definitions for HTS superconductor .txt files.",
  "t":               {"description": "Elapsed time",                 "symbol": "t",   "unit": "second"},
  "Ips":             {"description": "Power supply current",         "symbol": "I",   "unit": "ampere"},
  "Icoil":           {"description": "Coil current",                 "symbol": "I",   "unit": "ampere"},
  "Hall1_2519":      {"description": "Hall probe 1 (s/n 2519)",      "symbol": "B",   "unit": "tesla"},
  "Hall2_2514":      {"description": "Hall probe 2 (s/n 2514)",      "symbol": "B",   "unit": "tesla"},
  ...
  "Cernox_top_X89606_10": {"description": "Cernox top temperature",  "symbol": "T",   "unit": "kelvin"},
  ...
}
```

Full list derived from the 56 parsed column names. Units come from the `[unit]`
suffix in the original header — no guessing needed.

---

### 7. Test fixture — `tests/data/sample_hts.txt`

Create a trimmed version of the example file: header line + 10 data rows.
Use the exact header from `20220420_test21_200-250-300A_05As_SF.txt`.

---

### 8. `tests/test_magnetdata_hts.py` — new test file

Mirror the pattern of `tests/test_magnetdata.py`:

```python
SAMPLE_HTS = Path(__file__).parent / "data" / "sample_hts.txt"

class TestHTSMagnetData:
    def test_fromhts_returns_instance(self):
        md = HTSMagnetData.fromhts(str(SAMPLE_HTS))
        assert isinstance(md, HTSMagnetData)

    def test_type_is_hts(self):
        md = HTSMagnetData.fromhts(str(SAMPLE_HTS))
        assert md.getType() == 4  # DataType.HTS

    def test_time_column_renamed(self):
        md = HTSMagnetData.fromhts(str(SAMPLE_HTS))
        assert "t" in md.getKeys()
        assert "Temps [s]" not in md.getKeys()

    def test_columns_stripped(self):
        md = HTSMagnetData.fromhts(str(SAMPLE_HTS))
        assert "Ips" in md.getKeys()
        assert "Ips [A]" not in md.getKeys()

    def test_no_trailing_empty_column(self):
        md = HTSMagnetData.fromhts(str(SAMPLE_HTS))
        assert all(k.strip() != "" for k in md.getKeys())

    def test_load_magnetdata_fmt_hts(self):
        md = load_magnetdata(str(SAMPLE_HTS), fmt="hts")
        assert isinstance(md, HTSMagnetData)

    def test_data_is_dataframe(self):
        md = HTSMagnetData.fromhts(str(SAMPLE_HTS))
        df = md.getData()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_start_timestamp_parsed(self):
        # Filename 20220420_... → 2022-04-20 UTC
        md = HTSMagnetData.fromhts(str(SAMPLE_HTS))
        # sample_hts.txt may not have the date prefix; test with a real-name mock
        ...
```

---

## Verification

```bash
source magnetrun-env/bin/activate

# Run new tests
pytest tests/test_magnetdata_hts.py -v

# Run full suite to check no regressions
pytest tests/ -v

# Smoke-test with actual file
python -c "
from python_magnetrun.magnetdata import load_magnetdata
md = load_magnetdata('20220420_test21_200-250-300A_05As_SF.txt', fmt='hts')
print('Type:', md.getType())
print('Keys:', md.getKeys()[:5])
md.Units()
df = md.getData()
print('Shape:', df.shape)
print(df.head(2))
"
```
