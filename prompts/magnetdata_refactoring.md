# Refactor magnetdata.py — Replace `self.Type` branching with ABC hierarchy

*Created: 2026-03-30 — branch `separate-cooling`*
*Updated: 2026-04-14 — branch `rework_analysis`*

## Context

`magnetdata.py` (1,496 lines, single class) used integer `self.Type` (0=Pandas, 1=TDMS, 2=Ensight) to branch inside 20 of 28 methods — 37 total `if self.Type` conditionals. Adding new data formats (bprofile, feelpp CSV) required touching the already-large class.

This refactoring addresses two items from the IMPLEMENTATION_CHECKLIST:
- **Phase 3A** — "Break up `magnetdata.py`": replaces the tentative `io.py` / `transform.py` / `query.py` split with a superior decomposition by data type (ABC hierarchy)
- **Phase 2A** — "MagnetRun satisfies DataLoader protocol": after the refactor `MagnetDataBase` gains the methods needed to verify protocol compliance, and `get_time_range()` (Phase 2B prerequisite) is added at the right layer

Architecture layer recap:
- `DataLoader` protocol (`hybrid/data_protocol.py`) → implemented by `MagnetRun` and `HybridRun` (high-level, require `getSite`, `getHousing`, `getData`, `getKeys`, `getType`)
- `MagnetDataBase` ABC → low-level data container, used *by* `MagnetRun` internally
- These are two different layers; the ABC refactor does not directly make `MagnetData` satisfy `DataLoader`

---

## Class Hierarchy (as implemented)

```
MagnetDataBase (ABC)                   ← magnetdata_base.py (268 lines)
  ├── PandasMagnetData                 ← magnetdata_pandas.py (765 lines)
  │     ├── EnsightMagnetData          ← thin subclass (line 708)
  │     ├── BProfileMagnetData         ← new format (line 730)
  │     └── FeelppMagnetData           ← new format (line 749)
  └── TdmsMagnetData                   ← magnetdata_tdms.py (790 lines)

magnetdata.py                          ← pure factory (144 lines): load_magnetdata() + _fromtdms()
```

`HybridData` (Type=3) stays in `hybrid/`; can be registered with `MagnetDataBase.register(HybridData)`.

---

## Implementation Status

### Phase 1 — ✅ Done
- `magnetdata_base.py` with `MagnetDataBase` ABC (including `get_time_range` stub)
- `magnetdata_pandas.py` with `PandasMagnetData` + `EnsightMagnetData`, `BProfileMagnetData`, `FeelppMagnetData`
- No behavior changes, backward compatible

### Phase 2 — ✅ Done
- `magnetdata_tdms.py` with `TdmsMagnetData`
- `magnetdata.py` rewired as pure factory (`load_magnetdata()` standalone function)
- `MagnetData` shim class removed (commit `274d6bd`)

### Phase 3 — ✅ Done
- Zero `if self.Type` conditionals remain in `magnetdata*.py`
- `get_time_range()` concrete implementations on `PandasMagnetData` (`:476`) and `TdmsMagnetData` (`:528`)

### Phase 4 — Partially done / In progress

**Done:**
- `get_time_range()` available on all `MagnetDataBase` subclasses

**Still remaining:**
- `mdata.Type` checks in external files (see below)
- `DataProvider` protocol in `hybrid_run.py:55` — duplicates `DataLoader`; needs removal (Phase A0 of cross-domain comparison plan)
- `getDomain()` not yet added to `DataLoader` protocol or any implementation
- `isinstance(mrun, DataLoader)` verification test not yet added

---

## Remaining `.Type` checks (Phase 4 cleanup)

| File | Lines | Action |
|------|-------|--------|
| `commands/stats.py` | 64, 72, 182, 186, 308 | Replace with `isinstance(mdata, TdmsMagnetData)` |
| `commands/select.py` | 148, 175, 179, 206, 218 | Replace with `isinstance` checks |
| `cli.py` | 140, 143 | Replace with `isinstance` checks |
| `hybrid/data_protocol.py` | 212 | Replace `mdata.Type == 1` with `isinstance` check |
| `processing/cli.py` | 193 | Replace with `isinstance` check |

---

## File Organization (current)

```
python_magnetrun/
  magnetdata_base.py      ← MagnetDataBase ABC (268 lines)
  magnetdata_pandas.py    ← PandasMagnetData + subclasses (765 lines)
  magnetdata_tdms.py      ← TdmsMagnetData (790 lines)
  magnetdata.py           ← pure factory: load_magnetdata() + _fromtdms() (144 lines)
```

External imports: `from .magnetdata import load_magnetdata` or direct subclass imports.

---

## Abstract Interface — `MagnetDataBase` (magnetdata_base.py)

Abstract methods (must be implemented by every subclass):
- `Type: int` — property
- `getData(key)` — core data access
- `getKeys()` — list of available columns/channels
- `Units(debug)` — populate `self.units`
- `getUnitKey(key)` — look up unit for one key
- `extractData(keys)` — return a DataFrame subset

`get_time_range()` — concrete stub at base level raises `NotImplementedError`; concrete impls on `PandasMagnetData` and `TdmsMagnetData`.

---

## Phase 2A Integration

The `DataLoader` protocol (`hybrid/data_protocol.py`) requires: `getData`, `getKeys`, `getType`, `getSite`, `getHousing`.

- `getSite` / `getHousing` live on `MagnetRun` (not `MagnetDataBase`) — `MagnetRun` is the `DataLoader` implementor
- After this refactor, `MagnetRun` delegates to `MagnetDataBase` subclass methods cleanly
- **To close Phase 2A**, add a test: `assert isinstance(mrun, DataLoader)` — this passes since `MagnetRun` already has all 5 protocol methods (`getSite` at line 143, `getHousing` at line 147, plus `getData`, `getKeys`, `getType`)

---

## Backward Compatibility

| Concern | Impact | Status |
|---------|--------|--------|
| `isinstance(md, MagnetData)` | Breaks — `MagnetData` class removed | Tests updated to use `isinstance(md, MagnetDataBase)` |
| `mdata.Type == 0` in `commands/`, `cli.py` | Works but stale | Phase 4 cleanup pending |
| `mdata.Type == 1` in `hybrid/data_protocol.py:212` | Works but stale | Phase 4 cleanup pending |
| `from .magnetdata import MagnetData` (10 files) | Breaks — use `load_magnetdata` | Migration complete |

---

## Next Steps

1. **Complete Phase 4 cleanup** — replace remaining `mdata.Type` integer checks with `isinstance` across `commands/stats.py`, `commands/select.py`, `cli.py`, `hybrid/data_protocol.py`, `processing/cli.py`
2. **Phase A0** (cross-domain plan) — remove `DataProvider` from `hybrid_run.py:55`; it duplicates `DataLoader` in `data_protocol.py`
3. **Phase A1** (cross-domain plan) — add `getDomain()` + `get_time_range()` to `DataLoader` protocol definition
4. **Phase A2** (cross-domain plan) — implement `getDomain()` on `MagnetRun` and `HybridRun`; add `get_time_range()` delegation wrapper on `MagnetRun`
5. **Close Phase 2A** — add `assert isinstance(mrun, DataLoader)` compliance test

See [cross-domain-comparison.prompt.md](cross-domain-comparison.prompt.md) for the full Phase A–G implementation plan.

---

## Verification

1. `pytest tests/test_magnetdata.py` — all tests pass after Phase 2 ✅
2. `grep -r "if self.Type" python_magnetrun/magnetdata*.py` → zero results ✅
3. `isinstance(load_magnetdata(...txt), MagnetDataBase)` → True ✅
4. `isinstance(load_magnetdata(...tdms), MagnetDataBase)` → True ✅
5. `mdata.Type == 0` returns `0` for all pandas-backed subclasses ✅
6. `mdata.Type == 1` returns `1` for `TdmsMagnetData` ✅
7. Manual smoke test with real pupitre `.txt` and `.tdms` files via `MagnetRun` ✅

## IMPLEMENTATION_CHECKLIST Impact

- Phase 3A: `magnetdata.py` → ✅ Done (split into base/pandas/tdms + pure factory)
- Phase 2A: `MagnetRun satisfies DataLoader` → Pending (test not yet written)
- Phase 4 `.Type` cleanup → In progress (5 files remaining)
