# Refactor magnetdata.py — Replace `self.Type` branching with ABC hierarchy

*Created: 2026-03-30 — branch `separate-cooling`*

## Context

`magnetdata.py` (1,496 lines, single class) uses integer `self.Type` (0=Pandas, 1=TDMS, 2=Ensight) to branch inside 20 of 28 methods — 37 total `if self.Type` conditionals. Adding new data formats (bprofile, feelpp CSV) requires touching the already-large class.

This refactoring addresses two items from the IMPLEMENTATION_CHECKLIST:
- **Phase 3A** — "Break up `magnetdata.py`": replaces the tentative `io.py` / `transform.py` / `query.py` split with a superior decomposition by data type (ABC hierarchy)
- **Phase 2A** — "MagnetRun satisfies DataLoader protocol": after the refactor `MagnetDataBase` gains the methods needed to verify protocol compliance, and `get_time_range()` (Phase 2B prerequisite) is added at the right layer

Architecture layer recap:
- `DataLoader` protocol (`hybrid/data_protocol.py`) → implemented by `MagnetRun` and `HybridRun` (high-level, require `getSite`, `getHousing`, `getData`, `getKeys`, `getType`)
- `MagnetDataBase` ABC → low-level data container, used *by* `MagnetRun` internally
- These are two different layers; the ABC refactor does not directly make `MagnetData` satisfy `DataLoader`

---

## Class Hierarchy

```
MagnetDataBase (ABC)                   ← new: magnetdata_base.py
  ├── PandasMagnetData                 ← new: magnetdata_pandas.py
  │     ├── EnsightMagnetData          ← thin subclass, Type=2
  │     ├── BProfileMagnetData         ← new format (Index,Position,Profile CSV)
  │     └── FeelppMagnetData           ← new format (feelpp simulation CSV)
  └── TdmsMagnetData                   ← new: magnetdata_tdms.py

MagnetData(MagnetDataBase)             ← keep: magnetdata.py (factory + backward compat)
```

`HybridData` (Type=3) stays in `hybrid/`; registered with `MagnetDataBase.register(HybridData)`.

---

## File Organization

```
python_magnetrun/
  magnetdata_base.py      ← MagnetDataBase ABC (new)
  magnetdata_pandas.py    ← PandasMagnetData + subclasses (new)
  magnetdata_tdms.py      ← TdmsMagnetData (new)
  magnetdata.py           ← MagnetData factory (gut method bodies, keep classmethods + __init__)
```

This directly replaces the Phase 3A suggestion of splitting into `io.py` / `transform.py` / `query.py`, which would still leave the `if self.Type` problem untouched. The ABC split is strictly better: it eliminates branching AND separates concerns.

External imports (`from .magnetdata import MagnetData`) stay unchanged. `MagnetDataBase` is also re-exported from `magnetdata.py`.

---

## Abstract Interface — `MagnetDataBase` (magnetdata_base.py)

Abstract methods (must be implemented by every subclass):
- `Type: int` — property
- `getData(key)` — core data access
- `getKeys()` — list of available columns/channels
- `Units(debug)` — populate `self.units`
- `getUnitKey(key)` — look up unit for one key
- `extractData(keys)` — return a DataFrame subset

Default no-op or `raise NotImplementedError` implementations for everything else:
- `cleanupData_legacy`, `cleanupData`, `removeData`, `renameData` → no-op
- `addData`, `computeData`, `saveData`, `plotData`, `stats` → `raise NotImplementedError`
- `getStartDate`, `getDuration`, `addTime`, `shiftTime` → return `()` / `0.0` / `0`
- `info()` → `print(f"{self.__class__.__name__}: {self.FileName}")`
- `getType()` → `return self.Type`
- `get_time_range()` → `raise NotImplementedError` ← **Phase 2B hook**: `MagnetRun` will delegate to this

The `get_time_range()` stub at the `MagnetDataBase` level creates the hook needed to complete Phase 2B (time alignment layer) without waiting for this refactor to be done first.

---

## Subclass Method Distribution

| Method | PandasMagnetData | TdmsMagnetData |
|--------|-----------------|----------------|
| `getData` | `self.getPandasData(key)` | dict dispatch via `group/channel` keys |
| `Units` | pandas column loop | TDMS groups loop |
| `getUnitKey` | `self.units[key]` | call `PigBrotherUnits()` |
| `cleanupData_legacy` / `cleanupData` | full impl | no-op |
| `removeData` / `renameData` | pandas impl | no-op |
| `addData` | `DataFrame.eval(formula)` | TDMS group `.eval(nformula)` |
| `computeData` | pandas impl | `raise NotImplementedError` |
| `getStartDate` / `getDuration` | Date/Time columns | `wf_start_time` / `wf_increment` from Groups |
| `addTime` / `shiftTime` | full impl | no-op |
| `addTdmsTime` | not defined | full impl (guard removed) |
| `extractData`, `extractDataThreshold`, `extractTimeData` | pandas `.loc` impl | TDMS channel `.loc` impl |
| `saveData` | `DataFrame.to_csv` | concat groups + to_csv |
| `plotData` / `stats` / `info` | pandas impl | TDMS per-group impl |
| `get_time_range` | parse Date+Time cols → `(datetime, datetime)` | parse `wf_start_time` + duration |

`EnsightMagnetData` — zero overrides. Also fixes existing bug where `getData` raises `RuntimeError` for Type=2.

`BProfileMagnetData` / `FeelppMagnetData` — zero overrides. Only differ in how the factory loads the file.

---

## Phase 2A Integration

The `DataLoader` protocol (`hybrid/data_protocol.py`) requires: `getData`, `getKeys`, `getType`, `getSite`, `getHousing`.

- `getSite` / `getHousing` live on `MagnetRun` (not `MagnetDataBase`) — `MagnetRun` is the `DataLoader` implementor
- After this refactor, `MagnetRun` delegates to `MagnetDataBase` subclass methods cleanly
- **To close Phase 2A**, add a test: `assert isinstance(mrun, DataLoader)` — this passes since `MagnetRun` already has all 5 protocol methods (`getSite` at line 292, `getHousing` at line 296, plus `getData`, `getKeys`, `getType`)

---

## `MagnetData` Factory (magnetdata.py after refactor)

Keeps `__init__` (tests construct it directly) and all `from*` classmethods now returning subclasses:

```python
class MagnetData(MagnetDataBase):
    @classmethod
    def fromtxt(cls, name: str) -> PandasMagnetData: ...
    @classmethod
    def fromtdms(cls, name: str) -> TdmsMagnetData: ...
    @classmethod
    def fromensight(cls, name: str) -> EnsightMagnetData: ...
    @classmethod
    def fromcsv(cls, name: str) -> PandasMagnetData: ...
    @classmethod
    def fromStringIO(cls, name, sep, skiprows) -> PandasMagnetData: ...
    @classmethod
    def frombprofile(cls, name: str) -> BProfileMagnetData: ...   # NEW
    @classmethod
    def fromfeelpp(cls, name: str, skiprows: int = 0) -> FeelppMagnetData: ...  # NEW
```

---

## Backward Compatibility

| Concern | Impact | Resolution |
|---------|--------|------------|
| `isinstance(md, MagnetData)` in `tests/test_magnetdata.py:57` | Breaks | Update test to `isinstance(md, MagnetDataBase)` |
| `MagnetData("test.txt", {}, keys, 0, df)` in tests | Works — `__init__` kept | No change |
| `mdata.Type == 0` in `python_magnetrun.py` (9×), `processing/cli.py`, `plateaux.py` | Works — `Type` property preserved | Phase 4 cleanup |
| `isinstance(self.MagnetData.Data, pd.DataFrame)` in `MagnetRun.py:353` | Works | No change |
| `mdata.Type == 1` in `hybrid/data_protocol.py:211` | Works | Phase 4 cleanup |
| `from .magnetdata import MagnetData` (10 files) | Works unchanged | No change |

---

## Migration Phases

**Phase 1 — Create ABC + PandasMagnetData (zero risk)**
1. Write `magnetdata_base.py` with `MagnetDataBase` ABC (including `get_time_range` stub)
2. Write `magnetdata_pandas.py` with `PandasMagnetData` (copy Type-0 branches verbatim) + subclasses
3. Import in `magnetdata.py` — no behavior change, tests still pass

**Phase 2 — Create TdmsMagnetData + rewire factory**
4. Write `magnetdata_tdms.py` with `TdmsMagnetData` (copy Type-1 branches verbatim)
5. Rewire all `from*` classmethods to return subclass instances
6. Make `MagnetData` extend `MagnetDataBase`
7. Update `tests/test_magnetdata.py` line 57: `isinstance(md, MagnetData)` → `isinstance(md, MagnetDataBase)`

**Phase 3 — Remove dead code**
8. Delete all 37 `if self.Type` method bodies from `magnetdata.py`
9. Full test suite passes; `grep "if self.Type" python_magnetrun/magnetdata*.py` → zero results

**Phase 4 — Close Phase 2A and Phase 3A checklist items**
10. Add `assert isinstance(mrun, DataLoader)` test to close Phase 2A
11. Replace `mdata.Type == 0/1` with `isinstance` checks in `python_magnetrun.py`, `processing/cli.py`, `plateaux.py`, `hybrid/data_protocol.py`
12. Register `HybridData` with `MagnetDataBase.register(HybridData)`
13. Add `BPROFILE`, `FEELPP`, `ENSIGHT` to `DataSourceType` enum

---

## Adding New Formats in the Future

Zero existing files need to be touched — add a subclass in `magnetdata_pandas.py` and a classmethod in `magnetdata.py`.

---

## Critical Files

- [magnetdata.py](../python_magnetrun/magnetdata.py) — gut method bodies, keep `__init__` + classmethods
- [magnetdata_base.py](../python_magnetrun/magnetdata_base.py) — create new
- [magnetdata_pandas.py](../python_magnetrun/magnetdata_pandas.py) — create new
- [magnetdata_tdms.py](../python_magnetrun/magnetdata_tdms.py) — create new
- [tests/test_magnetdata.py](../tests/test_magnetdata.py) — update `isinstance` assertion (line 57)
- [hybrid/data_protocol.py](../python_magnetrun/hybrid/data_protocol.py) — Phase 4
- [python_magnetrun.py](../python_magnetrun/python_magnetrun.py) — Phase 4 (9 `.Type` checks)
- [MagnetRun.py](../python_magnetrun/MagnetRun.py) — Phase 4 (add DataLoader verification)

---

## Verification

1. `pytest tests/test_magnetdata.py` — all tests pass after Phase 2
2. `grep -r "if self.Type" python_magnetrun/magnetdata*.py` → zero results after Phase 3
3. `isinstance(MagnetData.fromtxt(...), MagnetDataBase)` → True
4. `isinstance(MagnetData.fromtdms(...), MagnetDataBase)` → True
5. `MagnetData("test.txt", {}, keys, 0, df)` — direct construction still works
6. `mdata.Type == 0` returns `0` for all pandas-backed subclasses
7. Manual smoke test with real pupitre `.txt` and `.tdms` files via `MagnetRun`

## IMPLEMENTATION_CHECKLIST Impact

After completing all phases, update:
- Phase 3A: `magnetdata.py` → ✅ Done (split into base/pandas/tdms + factory)
- Phase 2A: `MagnetRun satisfies DataLoader` → ✅ Done (verified by test in Phase 4)
