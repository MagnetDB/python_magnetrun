# Plan: Add `to_rms_tdms()` and `to_khz_tdms()` to HybridData

## Context

`HybridData` (kHz + RMS FEPC files) has no export capability — `saveData()` is not implemented.
The goal is to enable saving kHz and RMS hybrid data as TDMS files compatible with the pigbrother
format used by `TdmsMagnetData`, so hybrid recordings can be archived and re-read with the
existing TDMS infrastructure.

kHz (1000 Hz) and RMS (variable rate, typically 0.1–10 Hz) cannot share the same TDMS file
because `wf_increment` must be uniform per channel. The two data types produce **separate .tdms
files**: one for RMS, one for kHz.

TDMS group definitions are stored explicitly as `"_tdms_groups_rms"` and `"_tdms_groups_khz"`
top-level keys in `hybrid-defs.json` (same `_`-prefix convention used for pupitre).

---

## Data access recap

| Type | Access method | Return type | Time info |
|------|--------------|-------------|-----------|
| RMS | `load_rms_data(system, file_idx)` | `pd.DataFrame` with DatetimeIndex (UTC) | `reader.metadata["frequency"]` (Hz) |
| kHz | `read_khz_variable(system, variable, hours=)` | `(data: np.ndarray, time: np.ndarray)` elapsed-s from `global_t0` | `global_t0` = Unix UTC float from `compute_hour_t0()` |

---

## Approach

### Group definition: extend `hybrid-defs.json`

Add two top-level keys (both skipped by existing `load_defs()` / `load_units_from_json()` due to
`_` prefix):

```json
"_tdms_groups_rms": {
  "Courants_Alimentations": ["FEPC-AUX-LNCMI/ALIM1_J1", "FEPC-AUX-LNCMI/ALIM1_J2", ...],
  "Tensions_Aimant": ["FEPC-AUX-LNCMI/PH_V8", "FEPC-AUX-LNCMI/PH_V9", ...],
  "Supraconducteur": ["FEPC-LNCMI/I_BOB", "FEPC-LNCMI/I_DCCT", ...]
},
"_tdms_groups_khz": {
  "Courants_Alimentations": ["FEPC-AUX-LNCMI/ALIM1_J1", "FEPC-AUX-LNCMI/ALIM1_J2", ...],
  "Tensions_Aimant": ["FEPC-AUX-LNCMI/PH_V8", ...],
  "SC_coil_voltages": ["FEPC-LNCMI/DP1_V1", "FEPC-LNCMI/DP2_V2", ...]
}
```

- Values use the short-form `SYSTEM/VARIABLE` keys already in `hybrid-defs.json`.
- TDMS channel name for each entry = variable name part (after last `/`), with the pigbrother
  alias channel name used when an `aliases.pigbrother` entry exists (same rule as pupitre plan).
- Channels present in the data but not in any group → fallback TDMS group named after the
  FEPC system (e.g., `"FEPC-AUX-LNCMI"`).

---

## Files to modify

| File | Change |
|------|--------|
| `python_magnetrun/hybrid-defs.json` | Add `_tdms_groups_rms` and `_tdms_groups_khz` top-level keys |
| `python_magnetrun/field_defs.py` | Add `load_tdms_groups_rms()`, `load_tdms_groups_khz()`, `get_hybrid_channel_name()` helpers |
| `python_magnetrun/hybrid/hybrid_data.py` | Add `to_rms_tdms()` and `to_khz_tdms()` methods |

---

## Implementation Steps

### Step 1 — Extend `hybrid-defs.json`

Add the two group-mapping keys. Example structure (fill all relevant variables):

```json
{
  "_tdms_groups_rms": {
    "Courants_Alimentations": [
      "FEPC-AUX-LNCMI/ALIM1_J1", "FEPC-AUX-LNCMI/ALIM1_J2",
      "FEPC-AUX-LNCMI/ALIM2_J1", "FEPC-AUX-LNCMI/ALIM2_J2"
    ],
    "Tensions_Aimant": [
      "FEPC-AUX-LNCMI/PH_V8", "FEPC-AUX-LNCMI/PH_V9", "FEPC-AUX-LNCMI/PH_V10",
      "FEPC-AUX-LNCMI/PH_V11", "FEPC-AUX-LNCMI/PH_V12", "FEPC-AUX-LNCMI/PH_V13",
      "FEPC-AUX-LNCMI/PH_V14", "FEPC-AUX-LNCMI/BITTER_V1", "FEPC-AUX-LNCMI/BITTER_V2"
    ],
    "Supraconducteur": ["FEPC-LNCMI/I_BOB", "FEPC-LNCMI/I_DCCT"],
    "SC_coil_voltages": [
      "FEPC-LNCMI/DP1_V1", "FEPC-LNCMI/DP2_V2", ...all DP channels...
    ]
  },
  "_tdms_groups_khz": {
    ... same structure, potentially same or different grouping ...
  },
  ... existing entries unchanged ...
}
```

### Step 2 — Add helpers to `field_defs.py`

```python
def load_tdms_groups_rms(json_file: str | Path) -> dict[str, list[str]]:
    """Return ``_tdms_groups_rms`` mapping: TDMS group → list of 'SYSTEM/VAR' keys."""
    return load_defs(json_file).get("_tdms_groups_rms", {})


def load_tdms_groups_khz(json_file: str | Path) -> dict[str, list[str]]:
    """Return ``_tdms_groups_khz`` mapping: TDMS group → list of 'SYSTEM/VAR' keys."""
    return load_defs(json_file).get("_tdms_groups_khz", {})


def get_hybrid_channel_name(short_key: str, defs: dict) -> str:
    """Derive TDMS channel name from a 'SYSTEM/VAR' key.

    Uses ``aliases.pigbrother`` channel part when available; falls back to
    the variable name (part after the last '/').
    """
    alias = defs.get(short_key, {}).get("aliases", {}).get("pigbrother", "")
    if alias and "/" in alias:
        return alias.split("/", 1)[1]
    return short_key.split("/")[-1]
```

### Step 3 — Add `to_rms_tdms()` to `HybridData`

```python
def to_rms_tdms(
    self,
    filename: str,
    defs_file: str | None = None,
    groups: dict[str, list[str]] | None = None,
) -> None:
    """Export all RMS data for this day to a TDMS file.

    Parameters
    ----------
    filename : str
        Output path for the .tdms file.
    defs_file : str, optional
        Path to a *-defs.json with a ``_tdms_groups_rms`` key.
        Defaults to ``self.defs_file`` then ``"hybrid-defs.json"``.
    groups : dict, optional
        Explicit ``{tdms_group: ['SYSTEM/VAR', ...]}`` override.
    """
    from nptdms import TdmsWriter, ChannelObject, GroupObject, RootObject
    import numpy as np
    import pytz
    from ..field_defs import load_defs, load_tdms_groups_rms, get_hybrid_channel_name

    if not self._info.rms_available:
        raise RuntimeError("to_rms_tdms: no RMS data available")

    resolved_defs = defs_file or self.defs_file or "hybrid-defs.json"
    if groups is None:
        groups = load_tdms_groups_rms(resolved_defs)
    raw_defs = load_defs(resolved_defs)

    # --- Concatenate all RMS files for each system ---
    # Key: short_form "SYSTEM/VAR" → column in combined DataFrame per system
    combined: dict[str, pd.DataFrame] = {}
    wf_increment_by_system: dict[str, float] = {}

    for system, rms_files in self._info.rms_files.items():
        dfs = []
        freq = None
        for idx in range(len(rms_files)):
            df = self.load_rms_data(system, file_idx=idx)
            if freq is None:
                reader = self._get_rms_reader(system, idx)  # or re-parse header
                freq = reader.metadata.get("frequency", 1.0)
            dfs.append(df)
        if not dfs:
            continue
        full_df = pd.concat(dfs).sort_index()
        # Resample to uniform grid to fill any inter-file gaps with NaN
        period_s = 1.0 / freq
        full_df = full_df.resample(f"{period_s}s").mean()
        combined[system] = full_df
        wf_increment_by_system[system] = period_s

    # Build short_key → (system, col_name) lookup
    def _short_key_lookup(short_key: str):
        system, var = short_key.split("/", 1)
        return system, var

    with TdmsWriter(filename) as writer:
        writer.write_segment([RootObject(properties={"source": self.FileName})])

        assigned: set[str] = set()
        for group_name, short_keys in groups.items():
            writer.write_segment([GroupObject(group_name)])
            for short_key in short_keys:
                system, col = _short_key_lookup(short_key)
                df = combined.get(system)
                if df is None or col not in df.columns:
                    logger.warning("to_rms_tdms: %r not available, skipping", short_key)
                    continue
                assigned.add(short_key)
                channel_name = get_hybrid_channel_name(short_key, raw_defs)
                data = df[col].to_numpy(dtype=np.float64)
                wf_start = df.index[0]
                if wf_start.tzinfo is None:
                    wf_start = wf_start.tz_localize(pytz.UTC)
                props = {
                    "wf_increment": wf_increment_by_system[system],
                    "wf_samples": len(data),
                    "wf_start_time": wf_start,
                }
                full_key = f"rms/{short_key}"
                if full_key in self.units:
                    _, punit = self.units[full_key]
                    if punit is not None:
                        props["unit_string"] = format(punit, "~P")
                writer.write_segment([ChannelObject(group_name, channel_name, data, props)])

        # Fallback group for unassigned channels
        for system, df in combined.items():
            for col in df.columns:
                short_key = f"{system}/{col}"
                if short_key in assigned:
                    continue
                writer.write_segment([GroupObject(system)])
                data = df[col].to_numpy(dtype=np.float64)
                wf_start = df.index[0]
                if wf_start.tzinfo is None:
                    wf_start = wf_start.tz_localize(pytz.UTC)
                props = {
                    "wf_increment": wf_increment_by_system[system],
                    "wf_samples": len(data),
                    "wf_start_time": wf_start,
                }
                writer.write_segment([ChannelObject(system, col, data, props)])
```

**Note:** `_get_rms_reader(system, idx)` is a small private helper that constructs a `RMSFileReader`
and calls `parse_header()` to populate `metadata["frequency"]` without reading binary data:

```python
def _get_rms_reader(self, system: str, file_idx: int = 0):
    from .rms.rms_reader import RMSFileReader
    rms_file = self._info.rms_files[system][file_idx]
    reader = RMSFileReader(str(rms_file), endian=self.endian)
    reader.parse_header()
    return reader
```

### Step 4 — Add `to_khz_tdms()` to `HybridData`

```python
def to_khz_tdms(
    self,
    filename: str,
    defs_file: str | None = None,
    groups: dict[str, list[str]] | None = None,
    hours: range | list[int] | None = None,
) -> None:
    """Export kHz data to a TDMS file.

    Parameters
    ----------
    filename : str
        Output path for the .tdms file.
    defs_file : str, optional
        Path to a *-defs.json with a ``_tdms_groups_khz`` key.
    groups : dict, optional
        Explicit ``{tdms_group: ['SYSTEM/VAR', ...]}`` override.
    hours : range or list of int, optional
        Hours to export (default: all available). Strongly recommended to
        limit scope — a full day at 1 kHz = ~86 M samples per channel.
    """
    from nptdms import TdmsWriter, ChannelObject, GroupObject, RootObject
    import numpy as np
    import pytz
    from ..field_defs import load_defs, load_tdms_groups_khz, get_hybrid_channel_name
    from .kHz.fepc_reader import compute_hour_t0

    if not self._info.khz_available:
        raise RuntimeError("to_khz_tdms: no kHz data available")

    resolved_defs = defs_file or self.defs_file or "hybrid-defs.json"
    if groups is None:
        groups = load_tdms_groups_khz(resolved_defs)
    raw_defs = load_defs(resolved_defs)

    KHZ_DT = 0.001  # 1/1000 Hz — uniform for all kHz channels

    # Determine global_t0 from first available bin file (shared across all channels)
    # compute_hour_t0 returns a Unix UTC float timestamp.
    global_t0: float | None = None
    for system in self._info.fepc_systems:
        bin_files = self._info.khz_files.get(system, [])
        if bin_files:
            global_t0 = compute_hour_t0(str(bin_files[0]), self.date_str)
            break
    if global_t0 is None:
        raise RuntimeError("to_khz_tdms: no kHz bin files found to determine t0")
    wf_start = pd.Timestamp(global_t0, unit="s", tz=pytz.UTC)

    def _short_key_lookup(short_key: str):
        return short_key.split("/", 1)  # system, variable

    with TdmsWriter(filename) as writer:
        writer.write_segment([RootObject(properties={"source": self.FileName})])

        assigned: set[str] = set()
        for group_name, short_keys in groups.items():
            writer.write_segment([GroupObject(group_name)])
            for short_key in short_keys:
                system, variable = _short_key_lookup(short_key)
                try:
                    data, time = self.read_khz_variable(system, variable, hours=hours)
                except (ValueError, FileNotFoundError) as exc:
                    logger.warning("to_khz_tdms: %r skipped — %s", short_key, exc)
                    continue
                assigned.add(short_key)
                channel_name = get_hybrid_channel_name(short_key, raw_defs)
                props = {
                    "wf_increment": KHZ_DT,
                    "wf_samples": len(data),
                    "wf_start_time": wf_start,
                }
                full_key = f"kHz/{short_key}"
                if full_key in self.units:
                    _, punit = self.units[full_key]
                    if punit is not None:
                        props["unit_string"] = format(punit, "~P")
                writer.write_segment([ChannelObject(group_name, channel_name, data, props)])

        # Fallback: unassigned kHz variables → group named after FEPC system
        for system in self._info.fepc_systems:
            if system not in self._info.khz_files:
                continue
            try:
                all_vars = self.get_khz_variables(system)["analog"]
            except Exception:
                continue
            for variable in all_vars:
                short_key = f"{system}/{variable}"
                if short_key in assigned:
                    continue
                try:
                    data, _ = self.read_khz_variable(system, variable, hours=hours)
                except Exception as exc:
                    logger.warning("to_khz_tdms fallback: %r skipped — %s", short_key, exc)
                    continue
                props = {
                    "wf_increment": KHZ_DT,
                    "wf_samples": len(data),
                    "wf_start_time": wf_start,
                }
                writer.write_segment([ChannelObject(system, variable, data, props)])
```

---

## Key reused infrastructure

| Component | File | How it's reused |
|-----------|------|-----------------|
| `load_defs()` | `field_defs.py` | Load raw JSON + `_tdms_groups_*` and aliases |
| `read_khz_variable()` | `hybrid_data.py` | Returns `(data, time)` per variable, with `hours=` filtering |
| `load_rms_data()` | `hybrid_data.py` | Returns DataFrame with DatetimeIndex per file |
| `compute_hour_t0()` | `kHz/fepc_reader.py` | Returns Unix UTC float for `wf_start_time` |
| `RMSFileReader.metadata["frequency"]` | `rms/rms_reader.py` | Gives `wf_increment = 1/frequency` |
| `TdmsWriter`, `ChannelObject`, `GroupObject`, `RootObject` | `nptdms` (already a dep) | TDMS file writing |
| `self.units` | `HybridData` | `unit_string` channel properties |

---

## Notes on `field_meta` bug

The exploration found that `HybridData.__init__()` is **missing** `self.field_meta = {}`.
Lines 996 and 1021 of `hybrid_data.py` access `self.field_meta` and will raise `AttributeError`
after `load_units_from_json()`. Fix this as a prerequisite:

```python
# In HybridData.__init__(), after line 142 (self.units = {}):
self.field_meta: dict = {}
```

---

## Verification

1. Construct `HybridData(base_dir, date_str)` on a real data directory.
2. Call `hd.load_units_from_json("hybrid-defs.json")`.
3. **RMS test:**
   ```python
   hd.to_rms_tdms("output_rms.tdms")
   from nptdms import TdmsFile
   f = TdmsFile.open("output_rms.tdms")
   for g in f.groups():
       for ch in g.channels():
           print(g.name, ch.name, len(ch[:]), ch.properties.get("wf_increment"))
   ```
   Verify: groups/channels match `_tdms_groups_rms`, `wf_increment = 1/frequency`, data length correct.
4. **kHz test:**
   ```python
   hd.to_khz_tdms("output_khz.tdms", hours=range(10, 12))
   ```
   Verify: `wf_increment = 0.001`, `wf_start_time` matches first bin file's HH:00:00 UTC.
5. Run `pytest tests/` — no regressions expected (additive changes).
