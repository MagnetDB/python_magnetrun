# Plan: Add `to_tdms()` to PandasMagnetData

## Context

`PandasMagnetData` (pupitre .txt/.csv files) currently only exports data as CSV via `saveData()`.
The goal is to enable saving a `PandasMagnetData` object as a TDMS file in a structure compatible
with `TdmsMagnetData` (pigbrother format) — grouping flat DataFrame columns into named TDMS groups
and channels, with physical metadata (unit, start time, sample rate) stored as channel properties.

Pupitre data is nominally sampled at 1Hz but may contain:
- **Duplicate timestamps** (same second recorded more than once) — must be removed at load time.
- **Missing samples** (gaps in the 1Hz grid) — must be filled with NaN before writing TDMS, so
  that `wf_increment = 1.0 s` holds uniformly for all channels.

---

## Approach

### Group Definition: extend `pupitre-defs.json`

Add a top-level `"_tdms_groups"` key to `pupitre-defs.json` that explicitly maps TDMS group names
to lists of pupitre column names:

```json
"_tdms_groups": {
  "Courants_Alimentations": ["Idcct1", "Idcct2", "Idcct3", "Idcct4", "Field"],
  "Tensions_Aimant":        ["Ucoil1", "Ucoil2", ...],
  "Temperatures":           ["Tin1", "Tin2", "Tout", "teb", "tsb", "TAlimout"],
  "Pressions":              ["HP1", "HP2", "BP"],
  "Debits":                 ["Flow1", "Flow2", "debitbrut"],
  "Vitesses":               ["Rpm1", "Rpm2"]
}
```

- Keys prefixed with `_` are already skipped by `load_units_from_json()` and `load_defs()`, so
  no changes to existing loading code are required.
- TDMS **channel names** within each group are derived from:
  1. `aliases.pigbrother` in the field entry (e.g. `"Idcct1"` → `"Courants_Alimentations/Courant_A1"` → channel `"Courant_A1"`).
  2. Fallback: the column name itself (for columns without a pigbrother alias).

### Columns excluded from TDMS output

The following columns are time-indexing artifacts and must be skipped:
- `t` (elapsed seconds — reconstructable from `wf_increment` + `wf_start_time`)
- `timestamp` (UTC datetime — reconstructable from `wf_start_time`)
- `Date`, `Time` (raw string columns from the pupitre file)

### Deduplication (at load / addTime time)

After `addTime()` has created the `timestamp` column, drop rows with duplicate timestamps,
keeping the first occurrence and logging a warning with the count removed. This must happen
before any resampling so the 1Hz grid is built from clean data.

Location: end of `PandasMagnetData.addTime()`.

### 1Hz resampling (inside `to_tdms()`)

Before writing channels:
1. Set `timestamp` as the DataFrame index (`df.set_index("timestamp")`).
2. Call `df.resample("1s").mean()` — averages values within each 1-second bin; gaps become NaN.
3. Derive TDMS time properties from the resampled result:
   - `wf_start_time` = `resampled.index[0]` (first bin, UTC-aware)
   - `wf_increment` = `1.0` (always exactly 1 second)
   - `wf_samples` = `len(resampled)`

This guarantees a uniform time grid for all channels regardless of input irregularities.

---

## Files to modify

| File | Change |
|------|--------|
| `python_magnetrun/pupitre-defs.json` | Add top-level `"_tdms_groups"` key |
| `python_magnetrun/field_defs.py` | Add `load_tdms_groups()` and `get_pigbrother_channel_name()` helpers |
| `python_magnetrun/magnetdata_pandas.py` | (1) Deduplication at end of `addTime()`; (2) new `to_tdms()` method |

---

## Implementation Steps

### Step 1 — Extend `pupitre-defs.json`

Add the `"_tdms_groups"` key with the correct column groupings. Use the existing column names in
`pupitre-defs.json` as the values. Example structure (fill all relevant columns):

```json
{
  "_tdms_groups": {
    "Courants_Alimentations": ["Field", "Idcct1", "Idcct2", "Idcct3", "Idcct4", "IH", "IB"],
    "Tensions_Aimant": ["Ucoil1", "Ucoil2", "Ucoil3", "Ucoil4", "Ucoil5", "Ucoil6", "Ucoil7", "Ucoil15", "Ucoil16"],
    "Temperatures": ["Tin1", "Tin2", "Tout", "teb", "tsb", "TAlimout"],
    "Pressions": ["HP1", "HP2", "BP"],
    "Debits": ["Flow1", "Flow2", "debitbrut"],
    "Vitesses": ["Rpm1", "Rpm2"]
  },
  ... existing entries ...
}
```

### Step 2 — Add `load_tdms_groups()` to `field_defs.py`

```python
def load_tdms_groups(json_file: str | Path) -> dict[str, list[str]]:
    """Return the ``_tdms_groups`` mapping from a defs JSON file.

    Returns a dict mapping TDMS group name → list of pupitre column names.
    Returns an empty dict if no ``_tdms_groups`` key is present.
    """
    defs = load_defs(json_file)
    return defs.get("_tdms_groups", {})
```

Also add a helper `get_pigbrother_channel_name(key, defs)` that extracts the TDMS channel name
from `aliases.pigbrother` if available, falling back to the column name itself:

```python
def get_pigbrother_channel_name(key: str, defs: dict) -> str:
    alias = defs.get(key, {}).get("aliases", {}).get("pigbrother", "")
    if alias and "/" in alias:
        return alias.split("/", 1)[1]
    return key
```

### Step 3a — Deduplication in `addTime()` (`magnetdata_pandas.py`)

At the end of the existing `addTime()` method, after the `timestamp` column is created, add:

```python
# Drop duplicate timestamps (pupitre files occasionally record the same second twice)
n_before = len(self._data)
self._data = self._data.drop_duplicates(subset=["timestamp"], keep="first")
n_after = len(self._data)
if n_before != n_after:
    logger.warning(
        "addTime: dropped %d duplicate timestamp row(s) from %r",
        n_before - n_after, self.FileName,
    )
self.Keys = _dataframe_keys(self._data)
```

### Step 3b — Add `to_tdms()` to `PandasMagnetData`

Location: `magnetdata_pandas.py`, in the "persist / display" section after `saveData()`.

```python
def to_tdms(
    self,
    filename: str,
    defs_file: str | None = None,
    groups: dict[str, list[str]] | None = None,
) -> None:
    """Write this dataset to a TDMS file resampled to 1 Hz.

    Parameters
    ----------
    filename:
        Output path for the .tdms file.
    defs_file:
        Path to a *-defs.json file with a ``_tdms_groups`` key.
        Defaults to ``self.defs_file`` then ``"pupitre-defs.json"``.
    groups:
        Explicit ``{group_name: [col_name, ...]}`` mapping. Overrides defs file.
    """
    from nptdms import TdmsWriter, ChannelObject, GroupObject, RootObject
    import numpy as np
    import pytz
    from .field_defs import load_defs, load_tdms_groups, get_pigbrother_channel_name

    if "timestamp" not in self.Keys:
        raise RuntimeError("to_tdms: call addTime() before to_tdms()")

    _EXCLUDED = {"t", "timestamp", "Date", "Time"}

    # Resolve group mapping
    if groups is None:
        resolved_defs = defs_file or self.defs_file or "pupitre-defs.json"
        groups = load_tdms_groups(resolved_defs)
        raw_defs = load_defs(resolved_defs)
    else:
        raw_defs = {}

    # Collect columns not assigned to any group → "Pupitre" fallback group
    assigned = {col for cols in groups.values() for col in cols}
    unassigned = [k for k in self.Keys if k not in assigned and k not in _EXCLUDED]
    if unassigned:
        groups = dict(groups)
        groups.setdefault("Pupitre", []).extend(unassigned)

    # --- 1 Hz resampling ---
    # Use timestamp as index; resample('1s').mean() fills gaps with NaN.
    df = self.Data.copy()
    df = df.set_index("timestamp")
    data_cols = [c for c in df.columns if c not in _EXCLUDED]
    resampled = df[data_cols].resample("1s").mean()

    wf_increment = 1.0  # always exactly 1 second after resampling
    wf_start = resampled.index[0].tz_localize(pytz.UTC) if resampled.index[0].tzinfo is None else resampled.index[0]

    with TdmsWriter(filename) as writer:
        writer.write_segment([RootObject(properties={"source": self.FileName})])

        for group_name, col_names in groups.items():
            writer.write_segment([GroupObject(group_name)])
            for col in col_names:
                if col not in resampled.columns:
                    logger.warning("to_tdms: column %r not in data, skipping", col)
                    continue
                channel_name = get_pigbrother_channel_name(col, raw_defs)
                data = resampled[col].to_numpy(dtype=np.float64)

                props: dict = {
                    "wf_increment": wf_increment,
                    "wf_samples": len(data),
                    "wf_start_time": wf_start,
                }
                if col in self.units:
                    _, punit = self.units[col]
                    if punit is not None:
                        props["unit_string"] = format(punit, "~P")

                writer.write_segment([ChannelObject(group_name, channel_name, data, props)])
```

---

## Key reused infrastructure

| Existing component | File | How it's reused |
|--------------------|------|-----------------|
| `load_defs()` | `field_defs.py` | Load raw JSON including `_tdms_groups` and `aliases` |
| `TdmsWriter`, `ChannelObject`, `GroupObject`, `RootObject` | `nptdms` (already a dep) | TDMS file writing |
| `self.defs_file` | `PandasMagnetData` | Default defs file path |
| `self.units` | `MagnetDataBase` | Unit strings for channel properties |
| `self.start_timestamp` | `MagnetDataBase` | `wf_start_time` channel property |

---

## Verification

1. Load a pupitre .txt file: `obj = magnetdata.load("some_pupitre.txt")`
2. Call `obj.addTime()` — check warning count for duplicates (if any)
3. Call `obj.Units()`
4. Call `obj.to_tdms("output.tdms")`
5. Re-read with `nptdms`:
   ```python
   from nptdms import TdmsFile
   f = TdmsFile.open("output.tdms")
   for g in f.groups():
       for ch in g.channels():
           print(g.name, ch.name, ch.properties)
   ```
   Verify: groups/channels match `_tdms_groups`, `wf_increment == 1.0`, `wf_start_time` is correct, `unit_string` is set.
6. Confirm resampled data has no duplicate indices and NaN where gaps existed.
7. Optionally load with `magnetdata.load("output.tdms")` → `TdmsMagnetData` and confirm keys/data are accessible.
8. Run existing tests: `pytest tests/` — deduplication change to `addTime()` must not break existing test fixtures (verify no test data has duplicate timestamps).
