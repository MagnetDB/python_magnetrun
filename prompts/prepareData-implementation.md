# `prepareData` — Unified ETL for Pupitre and Pigbrother Data

Date: 2026-04-08

## Goal

Make a single `runetl.prepareData(data, housing, ...)` call work for both
`PandasMagnetData` (pupitre) and `TdmsMagnetData` (pigbrother) without any
type-switch in the caller.

---

## Current Gaps

| Operation called by `prepareData` | Pupitre | Pigbrother |
|---|---|---|
| `data.addTime()` | ✅ builds `timestamp`+`t` from Date/Time | ❌ no-op (base class) — must use `addTdmsTime()` |
| `data.getDuration()` | ✅ | ✅ |
| `data.cleanupData(keys_to_add, keys_to_rename, keys_to_remove)` | ✅ | ❌ no-op (base class) |

---

## Changes Required

### 1. `TdmsMagnetData.addTime()` — delegate to `addTdmsTime()`

Add to `magnetdata_tdms.py`, in the *time utilities* section near `addTdmsTime`:

```python
def addTime(self) -> int:  # noqa: N802
    """Implement the MagnetDataBase.addTime contract for TDMS data.

    Delegates to :meth:`addTdmsTime` (processes all non-Infos groups).
    """
    return self.addTdmsTime()
```

### 2. `TdmsMagnetData.cleanupData()` — handle `keys_to_add` and `keys_to_remove`

The base-class no-op must be overridden. `keys_to_rename` stays unsupported
(TDMS channel renaming is not meaningful) and is silently ignored.

Add to `magnetdata_tdms.py`, after `renameData`:

```python
def cleanupData(  # noqa: N802
    self,
    keys_to_remove: list[str] | None = None,
    keys_to_rename: dict[str, str] | None = None,
    keys_to_add: dict[str, str] | None = None,
    debug: bool = False,
) -> int:
    """Apply ETL operations to TDMS data.

    ``keys_to_add`` entries must use ``"Group/Channel"`` syntax consistent
    with :meth:`addData`.  ``keys_to_rename`` is ignored (TDMS does not
    support channel renaming).

    :param keys_to_remove: list of ``"Group/Channel"`` keys to drop.
    :param keys_to_rename: unused for TDMS; a warning is emitted if non-empty.
    :param keys_to_add: ``{"Group/Channel": "formula"}`` pairs; each formula
        is evaluated via :meth:`addData`.
    :param debug: passed through to :meth:`addData`.
    :return: 0 on success.
    """
    if keys_to_rename:
        logger.warning(
            "cleanupData: TDMS does not support renaming channels; "
            "keys_to_rename=%s ignored",
            list(keys_to_rename),
        )

    if keys_to_add:
        for key, formula in keys_to_add.items():
            self.addData(key, formula, debug=debug)

    if keys_to_remove:
        assert isinstance(self.Data, dict)
        for key in keys_to_remove:
            if "/" not in key:
                logger.warning("cleanupData: skip non-TDMS key %r (no '/' separator)", key)
                continue
            group, channel = key.split("/", 1)
            if group in self.Data and channel in self.Data[group].columns:
                self.Data[group].drop(columns=[channel], inplace=True)
                if key in self.Keys:
                    self.Keys.remove(key)
            else:
                logger.debug("cleanupData: key %r not found, skipping removal", key)

    return 0
```

---

## ETL Map: What to Pass for Each Format

### Pupitre (`PandasMagnetData`)

`keys_to_add` and `keys_to_rename` come from `HousingConfig` (to be added — see
the HousingConfig extension plan below).

Example for M9 (`pupitre_formula_map` + `get_pupitre_rename_map()`):

```python
# stored in HousingConfig.pupitre_formula_map
keys_to_add = {
    "IH_ref": "IH_ref = Idcct1 + Idcct2",
    "IB_ref": "IB_ref = Idcct3 + Idcct4",
}
# derived via cfg.get_pupitre_rename_map() (reference_gr1_flow="FlowH" → h_idx=1)
keys_to_rename = {
    "Flow1": "FlowH", "Flow2": "FlowB",
    "Rpm1":  "RpmH",  "Rpm2":  "RpmB",
    "Tin1":  "TinH",  "Tin2":  "TinB",
    "HP1":   "HPH",   "HP2":   "HPB",
}
```

Example for M8/M10 (`reference_gr1_flow="FlowB"` → `h_idx=2`):

```python
keys_to_add = {
    "IH_ref": "IH_ref = Idcct3 + Idcct4",
    "IB_ref": "IB_ref = Idcct1 + Idcct2",
}
# derived via cfg.get_pupitre_rename_map()
keys_to_rename = {
    "Flow1": "FlowB", "Flow2": "FlowH",
    "Rpm1":  "RpmB",  "Rpm2":  "RpmH",
    "Tin1":  "TinB",  "Tin2":  "TinH",
    "HP1":   "HPB",   "HP2":   "HPH",
}
```

### Pigbrother (`TdmsMagnetData`)

`Référence_GR1` and `Référence_GR2` are currently computed inline in `magnetdata.py/fromtdms`
as the sum of `Référence_A1+A2` and `Référence_A3+A4`.  That inline ETL should be moved to
`pigbrother_formula_map` in `HousingConfig` and applied via `prepareData`.

Example:

```python
keys_to_add = {
    "Courants_Alimentations/Référence_GR1":
        "Courants_Alimentations/Référence_GR1 = Référence_A1 + Référence_A2",
    "Courants_Alimentations/Référence_GR2":
        "Courants_Alimentations/Référence_GR2 = Référence_A3 + Référence_A4",
}
```

Note: `Référence_A1/A2` are within the same group (`Courants_Alimentations`), so
`TdmsMagnetData.addData` already handles intra-group formulas correctly.

---

## `HousingConfig` Extensions Required

The fields below need to be added to the `HousingConfig` dataclass and the three
`*-housing-config.json` files so that `prepareData` can be driven entirely by config.

### New dataclass fields

```python
# Pupitre ETL — formulas added after loading
pupitre_formula_map: dict = field(default_factory=dict)
# e.g. {"IH_ref": "IH_ref = Idcct1 + Idcct2", "IB_ref": "IB_ref = Idcct3 + Idcct4"}
# field(default_factory=dict) is the required dataclass idiom for a mutable default
# (= {} is forbidden in frozen dataclasses).

# Pigbrother ETL — formulas added after loading
pigbrother_formula_map: dict = field(default_factory=dict)
# e.g. {"Courants_Alimentations/Référence_GR1": "Référence_GR1 = Référence_A1 + Référence_A2"}

# Hybrid ETL — derived current formulas (M8-only; empty string = not applicable)
hybrid_gr1_current_formula: str = ""
# e.g. "FEPC-AUX-LNCMI/ALIM2_J1 + FEPC-AUX-LNCMI/ALIM2_J2"
hybrid_gr2_current_formula: str = ""
# e.g. "FEPC-AUX-LNCMI/ALIM1_J1 + FEPC-AUX-LNCMI/ALIM1_J2"
# The caller injects the acquisition-type prefix (kHz/, rms/) at call time.

# Voltage channel semantics (split UH/UB from source Ucoil list)
reference_gr1_voltage: str = "UH"   # derived aggregate field name
reference_gr2_voltage: str = "UB"
# voltage_channels_gr1/gr2 should contain ONLY the source Ucoil probes, not UH/UB
```

### `pupitre_rename_map` — computed, not stored

Do **not** add `pupitre_rename_map` as a stored field. The rename map
(`Flow1→FlowH`, `Flow2→FlowB`, ...) is fully derivable from the existing
`reference_gr1_flow` / `reference_gr2_flow` fields:

- M9 has `reference_gr1_flow = "FlowH"` → GR1 is wired to index 1 → rename
  `Flow1→FlowH`, `Flow2→FlowB`.
- M8/M10 has `reference_gr1_flow = "FlowB"` → rename `Flow1→FlowB`,
  `Flow2→FlowH`.

Add a computed method instead:

```python
_PUPITRE_FLOW_FIELDS = ("Flow", "Rpm", "Tin", "HP")

def get_pupitre_rename_map(self) -> dict[str, str]:
    """Derive the Flow/Rpm/Tin/HP index→role rename map from the GR1 flow role."""
    if not self.reference_gr1_flow:
        return {}
    # Determine which numeric index belongs to GR1 (H or B)
    # Convention: fields are named e.g. "FlowH" or "FlowB"; index suffix is "1" or "2"
    if self.reference_gr1_flow.endswith("H"):   # GR1 = H supply → index 1
        h_idx, b_idx = "1", "2"
    else:                                         # GR1 = B supply → index 1
        h_idx, b_idx = "2", "1"
    result = {}
    for base in _PUPITRE_FLOW_FIELDS:
        result[f"{base}{h_idx}"] = f"{base}H"
        result[f"{base}{b_idx}"] = f"{base}B"
    return result
```

This eliminates the stored field entirely — any housing added in the future
automatic gets the correct rename map as long as its `reference_gr1_flow` follows
the `"<base>H"` / `"<base>B"` suffix convention.

### Update `ROLE_TO_FIELD` and `_TUPLE_FIELDS`

```python
ROLE_TO_FIELD = {
    ...existing entries...,
    "pupitre_formula_map":         "pupitre_formula_map",
    "pigbrother_formula_map":      "pigbrother_formula_map",
    "hybrid_gr1_current_formula":  "hybrid_gr1_current_formula",
    "hybrid_gr2_current_formula":  "hybrid_gr2_current_formula",
    "reference_gr1_voltage":       "reference_gr1_voltage",
    "reference_gr2_voltage":       "reference_gr2_voltage",
}
# _TUPLE_FIELDS: no changes needed (new fields are dicts/strs, not tuples)
# pupitre_rename_map is NOT a stored field — see get_pupitre_rename_map() above
```

### Voltage formula methods

Two new methods derive the `UH`/`UB` summation formulas dynamically, filtering on
`available_keys` (today = DataFrame columns; future = MagnetDB site data):

```python
def get_pupitre_voltage_formulas(
    self, available_keys: list[str] | None = None
) -> dict[str, str]:
    """Return {reference_gr1_voltage: formula, reference_gr2_voltage: formula}.

    Only Ucoil channels present in available_keys are summed.  If
    available_keys is None all channels in voltage_channels_gr1/gr2 are used.
    """
    def _sum_formula(target: str, channels: tuple, keys: list[str] | None) -> str:
        srcs = [c for c in channels if keys is None or c in keys]
        return f"{target} = {' + '.join(srcs)}" if srcs else ""

    result = {}
    f1 = _sum_formula(self.reference_gr1_voltage, self.voltage_channels_gr1, available_keys)
    f2 = _sum_formula(self.reference_gr2_voltage, self.voltage_channels_gr2, available_keys)
    if f1:
        result[self.reference_gr1_voltage] = f1
    if f2:
        result[self.reference_gr2_voltage] = f2
    return result
```

Hybrid needs the same pattern but over `hybrid_voltage_channels_gr1/gr2`
(separate fields, M8-only):

```python
def get_hybrid_voltage_formulas(
    self, available_keys: list[str] | None = None
) -> dict[str, str]:
    """Return voltage sum formulas for hybrid (kHz) data (M8-only).

    Uses hybrid_voltage_channels_gr1/gr2 (separate from the pupitre Ucoil lists).
    Empty dict for housings without hybrid format.
    """
    if not self.supports_format("hybrid"):
        return {}
    # same logic as get_pupitre_voltage_formulas but over hybrid channel lists
    ...
```

Note: `hybrid_gr1_current_formula` / `hybrid_gr2_current_formula` handle
M8-specific derived currents (sum of ALIM power supply channels that do not
exist in pupitre/pigbrother data), while `get_hybrid_voltage_formulas` handles
the GR1/GR2 voltage sums.  Both are needed for M8 hybrid ETL.

---

## Updated `prepareData` Call — Pupitre Example

```python
# runetl.py
from .housing_config import get_housing_config

def prepareData(data, housing, keys_to_remove=None, keys_to_rename=None,
                keys_to_add=None, debug=False):
    cfg = get_housing_config(housing)

    # Build ETL maps from HousingConfig if caller did not supply them
    if keys_to_add is None and keys_to_rename is None:
        from .magnetdata_base import DataType
        if data.Type == DataType.PUPITRE:
            available = data.getKeys()
            keys_to_add = {
                **cfg.pupitre_formula_map,
                **cfg.get_pupitre_voltage_formulas(available),
            }
            keys_to_rename = cfg.get_pupitre_rename_map()
        elif data.Type == DataType.TDMS:
            keys_to_add = cfg.pigbrother_formula_map
        elif data.Type == DataType.HYBRID:
            keys_to_add = cfg.get_hybrid_voltage_formulas(data.getKeys())
            # hybrid_gr1/2_current_formula are injected separately with the
            # acquisition-type prefix (kHz/, rms/) by the hybrid ETL caller

    data.addTime()
    _duration = data.getDuration()
    data.cleanupData(
        keys_to_remove=keys_to_remove,
        keys_to_rename=keys_to_rename,
        keys_to_add=keys_to_add,
        debug=debug,
    )
```

---

## Migration Path

| Step | File(s) | Change |
|---|---|---|
| **A** | `magnetdata_tdms.py` | Add `addTime()` + `cleanupData()` overrides (the two targeted additions) |
| **B** | `housing_config.py` | Add new dataclass fields (`pupitre_formula_map`, `pigbrother_formula_map`, `hybrid_gr1/2_current_formula`, `reference_gr1/2_voltage`); add `get_pupitre_rename_map()` and `get_pupitre_voltage_formulas()`; update `ROLE_TO_FIELD` |
| **C** | `M9/M8/M10-housing-config.json` | Populate `pupitre_formula_map`, `pigbrother_formula_map`; M8 also gets `hybrid_gr1/2_current_formula`; remove `UH`/`UB` from `voltage_channels_gr*` |
| **D** | `magnetdata.py` (`fromtdms`) | Remove inline `Référence_GR1/GR2` computation (moved to `pigbrother_formula_map`) |
| **E** | `runetl.py` (`prepareData`) | Auto-build ETL maps from `HousingConfig` when caller passes `None` |
| **F** | `runetl.py` (`prepareData_legacy`) | Replace hardcoded `if housing == "M9"` blocks with `HousingConfig`-driven logic |
| **G** | `field_mappings.py` | Delete (superseded by `HousingConfig`) |

> **Start with Step A.** It is self-contained, adds tests, and immediately unblocks
> calling `prepareData` on pigbrother data without waiting for the `HousingConfig`
> extension (Steps B–C).

---

## `inline ETL in fromtdms` — Detail

The current inline code in `magnetdata.py` (lines ~173–189):

```python
if "Référence_A1" in Data["Courants_Alimentations"]:
    Data["Courants_Alimentations"]["Référence_GR1"] = (
        Data["Courants_Alimentations"]["Référence_A1"]
        + Data["Courants_Alimentations"]["Référence_A2"]
    )
    Keys.append("Courants_Alimentations/Référence_GR1")
    Groups["Courants_Alimentations"]["Référence_GR1"] = ...
```

After Step D this should become a call to `prepareData` (or direct `addData`) driven
by `cfg.pigbrother_formula_map`.  Note: `fromtdms` does not know the housing, so the
caller must pass the housing (or config) and call `prepareData` after `fromtdms`
returns.  Until Step D is complete the inline code stays in place to avoid breaking
existing callers.

---

## Test Coverage Plan

| Test | File | What it verifies |
|---|---|---|
| `test_tdms_addTime_delegates` | `tests/test_magnetdata_tdms.py` | `addTime()` calls `addTdmsTime()` and adds `t` column to all groups |
| `test_tdms_cleanupData_keys_to_add` | same | `cleanupData(keys_to_add={...})` calls `addData` for each entry |
| `test_tdms_cleanupData_keys_to_remove` | same | key dropped from Data and Keys |
| `test_tdms_cleanupData_rename_warns` | same | `keys_to_rename` emits `logger.warning` without raising |
| `test_prepareData_pupitre` | `tests/test_runetl.py` | end-to-end: pupitre fixture → IH_ref, FlowH present after call |
| `test_prepareData_pigbrother` | same | end-to-end: TDMS fixture → `Référence_GR1` present after call |
