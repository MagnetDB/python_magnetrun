# Migrate Callers to `prepareData` — Remove Inline TDMS ETL

Date: 2026-04-08

## Goal

Remove the inline `Référence_GR1`/`Référence_GR2` computation from
`MagnetData.fromtdms` (Step D) and retire `prepareData_legacy` by migrating
every caller to `prepareData`.  Also delete the now-superseded
`field_mappings.py` (Step G).

---

## Context

### What was done in the previous implementation

| File | Change |
|---|---|
| `magnetdata_tdms.py` | Added `addTime()` + `cleanupData()` overrides |
| `housing_config.py` | Added `pupitre_formula_map`, `pigbrother_formula_map`, `hybrid_formula_map`; added `get_pupitre_rename_map()`, `get_pupitre_voltage_formulas()`, `get_hybrid_voltage_formulas()` |
| `M9/M8/M10-housing-config.json` | Populated new ETL fields; removed `UH`/`UB` from `voltage_channels_gr*` |
| `runetl.py` | `prepareData()` auto-builds ETL maps from `HousingConfig` by `data.Type` |
| `runetl.py` | `prepareData_legacy()` now uses `cfg.pupitre_formula_map` + `cfg.get_pupitre_rename_map()` |

### What is still blocking Step D

`MagnetData.fromtdms` (`magnetdata.py:171–189`) contains this inline ETL:

```python
# Add reference for GR1, GR2
if "Référence_A1" in Data["Courants_Alimentations"]:
    Data["Courants_Alimentations"]["Référence_GR1"] = (
        Data["Courants_Alimentations"]["Référence_A1"]
        + Data["Courants_Alimentations"]["Référence_A2"]
    )
    Keys.append("Courants_Alimentations/Référence_GR1")
    ...
```

It runs unconditionally because `fromtdms` does **not** receive a `housing`
argument — it is a pure file-loading factory.  The housing is only known by
the callers higher up the call chain.

---

## Caller Inventory

### TDMS path

| Caller | File | Has housing? | Action |
|---|---|---|---|
| `MagnetRun.fromtdms(housing, site, filename)` | `MagnetRun.py:38` | ✅ | Add `prepareData(data, housing)` after `MagnetData.fromtdms` |
| `MagnetRun.fromtdms` called by `processing/cli.py:164` | downstream | via `MagnetRun` | No change — fixed at `MagnetRun` level |
| `MagnetRun.fromtdms` called by `hybrid/hybrid_run.py:25` | downstream | via `MagnetRun` | No change |
| `MagnetRun.fromtdms` called by `analysis/loaders.py:359,588` | downstream | via `MagnetRun` | No change |
| `MagnetRun.fromtdms` called by `utils/files.py:179,361` | downstream | via `MagnetRun` | No change |
| `MagnetRun.fromtdms` called by `cli.py:112` | downstream | via `MagnetRun` | No change |

### Pupitre path (switch `prepareData_legacy` → `prepareData`)

| Caller | File | Has housing? | Notes |
|---|---|---|---|
| `MagnetRun.fromtxt` | `MagnetRun.py:78` | ✅ | Has TODO comment to switch |
| `MagnetRun.fromStringIO` | `MagnetRun.py:118` | ✅ | |
| `concat_files` | `utils/txt2csv.py:77` | ✅ | |

---

## Gap: `cleanupData_legacy` Operations Not Yet Migrated

`prepareData_legacy` calls `data.cleanupData_legacy()` after the formula/rename
steps.  `cleanupData_legacy` in `magnetdata_pandas.py` does three things that
`prepareData` does **not** yet replicate:

1. **Remove all-zero columns** — drops zero columns, except `Flow*` and `Field`.
2. **Remove duplicate Icoil columns** — keeps only 2 non-duplicate `IcoilN` columns.
3. **Compute `UH`/`UB`** by dynamically detecting consecutive Ucoil groups.
   (This is now superseded by `get_pupitre_voltage_formulas()` + `HousingConfig`.)

Items 1 and 2 must be preserved; item 3 can be dropped once `get_pupitre_voltage_formulas`
is used instead.

### Recommendation

Add a `cleanupPupitreData(data, housing)` helper in `runetl.py` that:
- Removes all-zero columns (except `Flow*` / `Field`)
- Resolves duplicate Icoil columns (keep two non-duplicate)
- Renames `Icoil[first]→IH`, `Icoil[last]→IB`

Call it from `prepareData` when `data.Type == DataType.PUPITRE`, **after**
`cleanupData()`.  This replaces the final three steps of `prepareData_legacy`.

```python
# runetl.py — new helper (pupitre-only post-processing)
def _cleanup_pupitre_icoil(data: MagnetData) -> None:
    """Remove zero/duplicate Icoil columns and rename Icoil→IH/IB."""
    import re
    from natsort import natsorted

    # Drop all-zero columns (skip Flow* and Field)
    zero_cols = [
        col for col in data.Data.columns
        if (data.Data[col] == 0).all()
        and not col.startswith("Flow")
        and not col.startswith("Field")
    ]
    if zero_cols:
        data.Data.drop(columns=zero_cols, inplace=True)
        for col in zero_cols:
            if col in data.Keys:
                data.Keys.remove(col)

    # Resolve duplicate Icoil columns
    Ikeys = natsorted([k for k in data.getKeys() if re.match(r"Icoil\d+", k)])
    if len(Ikeys) > 2:
        ikeys_df = data.Data[Ikeys]
        remove = []
        for i in range(len(Ikeys)):
            for j in range(i + 1, len(Ikeys)):
                diff = ikeys_df[Ikeys[i]] - ikeys_df[Ikeys[j]]
                if abs(diff.mean()) <= 1e-2:
                    remove.append(Ikeys[j])
        if remove:
            data.Data.drop(columns=remove, inplace=True)
            for k in remove:
                if k in data.Keys:
                    data.Keys.remove(k)
        Ikeys = natsorted([k for k in data.getKeys() if re.match(r"Icoil\d+", k)])

    # Rename Icoil[0]→IH, Icoil[-1]→IB
    if Ikeys:
        data.renameData(columns={Ikeys[0]: "IH"})
        data.renameData(columns={Ikeys[-1]: "IB"})
```

---

## Fix `runetl.py` HYBRID Case

The current HYBRID branch in `prepareData`:

```python
elif data.Type == DataType.HYBRID:
    keys_to_add = cfg.get_hybrid_voltage_formulas(data.getKeys())
```

must also include `cfg.hybrid_formula_map` (M8 derived currents):

```python
elif data.Type == DataType.HYBRID:
    keys_to_add = {
        **cfg.hybrid_formula_map,
        **cfg.get_hybrid_voltage_formulas(data.getKeys()),
    }
```

---

## Changes Required

### 1. `runetl.py`

**a. Fix HYBRID `keys_to_add`** (see above).

**b. Add `_cleanup_pupitre_icoil` helper** (see above).

**c. Extend PUPITRE branch of `prepareData` to call the helper:**

```python
if data.Type == DataType.PUPITRE:
    ...
    # after cleanupData():
    _cleanup_pupitre_icoil(data)
```

### 2. `MagnetRun.py` — `fromtdms`

After `data = MagnetData.fromtdms(filename)`, add:

```python
from .runetl import prepareData
prepareData(data, housing)
```

Remove the `prepareData_legacy` import from the module top (once `fromtxt` and
`fromStringIO` are migrated too).

### 3. `MagnetRun.py` — `fromtxt`

Replace:
```python
prepareData_legacy(data, housing)
```
with:
```python
from .runetl import prepareData
prepareData(data, housing)
```
Remove the TODO comment.

### 4. `MagnetRun.py` — `fromStringIO`

Same substitution as `fromtxt`.

### 5. `utils/txt2csv.py` — `concat_files`

Replace:
```python
from ..runetl import prepareData_legacy
...
prepareData_legacy(data, housing)
```
with:
```python
from ..runetl import prepareData
...
prepareData(data, housing)
```

### 6. `magnetdata.py` (`fromtdms`) — Step D

Remove lines 171–189 (the inline `Référence_GR1`/`Référence_GR2` computation):

```python
# DELETE — now handled by prepareData(data, housing) in MagnetRun.fromtdms
if "Référence_A1" in Data["Courants_Alimentations"]:
    Data["Courants_Alimentations"]["Référence_GR1"] = (...)
    Keys.append("Courants_Alimentations/Référence_GR1")
    Groups["Courants_Alimentations"]["Référence_GR1"] = ...
if "Référence_A3" in Data["Courants_Alimentations"]:
    Data["Courants_Alimentations"]["Référence_GR2"] = (...)
    Keys.append("Courants_Alimentations/Référence_GR2")
    Groups["Courants_Alimentations"]["Référence_GR2"] = ...
```

`TdmsMagnetData.cleanupData` already skips keys that exist (idempotent), so
no data is lost — `pigbrother_formula_map` in `HousingConfig` replaces this.

### 7. `field_mappings.py` — Step G

`field_mappings.py` is no longer imported anywhere (verified by grep).
Delete the file entirely.

---

## Migration Path

| Step | File | Change |
|---|---|---|
| **1** | `runetl.py` | Fix HYBRID `keys_to_add` to include `hybrid_formula_map` |
| **2** | `runetl.py` | Add `_cleanup_pupitre_icoil` helper; call it in PUPITRE branch |
| **3** | `MagnetRun.py` | Add `prepareData(data, housing)` in `fromtdms` |
| **4** | `MagnetRun.py` | Switch `fromtxt` and `fromStringIO` to `prepareData` |
| **5** | `utils/txt2csv.py` | Switch `concat_files` to `prepareData` |
| **6** | `magnetdata.py` | Remove inline TDMS ETL from `fromtdms` (Step D) |
| **7** | `magnetdata_pandas.py` | Remove `cleanupData_legacy` (superseded; only called by `prepareData_legacy`) |
| **8** | `runetl.py` | Delete `prepareData_legacy` |
| **9** | `MagnetRun.py` | Remove `prepareData_legacy` import |
| **10** | `field_mappings.py` | Delete file (Step G) |

> **Do steps 1–5 first, run the full test suite, then do 6–10.**
> Steps 6–10 are destructive — confirm no callers remain before deleting.

---

## Test Coverage Plan

| Test | File | What it verifies |
|---|---|---|
| `test_magnetrun_fromtdms_calls_prepareData` | `tests/test_magnetrun.py` | `MagnetRun.fromtdms` result has `Référence_GR1` key |
| `test_prepareData_pupitre_icoil_rename` | `tests/test_runetl.py` | After `prepareData` on pupitre data, `IH` and `IB` present |
| `test_prepareData_hybrid_includes_formula_map` | `tests/test_runetl.py` | HYBRID branch merges `hybrid_formula_map` into `keys_to_add` |
| `test_fromtdms_no_inline_etl` | `tests/test_magnetdata.py` | After Step D: raw `fromtdms` result does NOT contain `Référence_GR1` |
