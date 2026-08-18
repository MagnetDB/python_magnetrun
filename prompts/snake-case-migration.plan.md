# Plan: Adopt snake_case across python_magnetrun

Date: 2026-04-13 (updated 2026-06-09)

---

## Current Status (2026-06-09)

**Prerequisites completed:**
- `fix-remaining-issues.plan.md` — done
- `cli-consolidation.plan.md` — fully implemented

**What exists now:**
- 47 methods have `# noqa: N802` but **zero `DeprecationWarning`** — they are still primary implementations, not shims
- `MagnetRun.py` has 14 more camelCase methods without even the noqa comment
- No snake_case counterparts exist yet (the only snake_case method added so far is `get_time_range`)
- Extensive internal callers: `commands/`, `processing/`, `analysis/`, `runetl.py`, `waterflow_pipeline.py`, tests

**Immediate next action:** Begin Pass 1 — add snake_case implementations and camelCase shims across the four core files.

---

## Open Questions (must resolve before Pass 1)

1. **`hybrid/hybrid_run.py`** — its camelCase methods (`getData`, `getKeys`, etc.) mirror `MagnetRun`'s. Should they follow the same shim pattern, or is `HybridRun` considered internal-only (no external callers)?
2. **`requests/` subpackage** — `GObject.py`, `HMagnet.py`, `MRecord.py`, `webscrapping.py` all have camelCase. Are these in scope for Pass 1, or treated separately (they are closer to external API)?
3. **`simulation/simulation_run.py` and `bfield/bfield_run.py`** — forward-delegate to wrapped objects using camelCase. Should their own camelCase facade methods be shimmed, or are they purely internal?
4. **`panels/*.py`** — scripts, not importable modules. Update in-place in Pass 2 or defer?

---

## Context

The package was originally written with JavaBean-style conventions (camelCase methods like `getData()`, `getKeys()`) and PascalCase module filenames (`MagnetRun.py`, `GObject.py`). These violate PEP 8 and make the API inconsistent with the rest of the Python ecosystem. This plan migrates everything to snake_case without breaking existing callers abruptly, using deprecation aliases during the transition.

> **Prerequisite:** Complete all issues in `fix-remaining-issues.plan.md` before starting Pass 1. See [Ordering advice](#ordering-advice-vs-fix-remaining-issuesplanmd) below.

---

## Scope of Changes

### 1. Module file renames (7 files)

| Old | New |
|-----|-----|
| `python_magnetrun/MagnetRun.py` | `python_magnetrun/magnetrun.py` |
| `python_magnetrun/BandH.py` | `python_magnetrun/band_h.py` |
| `python_magnetrun/requests/GObject.py` | `python_magnetrun/requests/gobject.py` |
| `python_magnetrun/requests/HMagnet.py` | `python_magnetrun/requests/hmagnet.py` |
| `python_magnetrun/requests/MRecord.py` | `python_magnetrun/requests/mrecord.py` |
| `python_magnetrun/panels/panel-mrecord.py` | `python_magnetrun/panels/panel_mrecord.py` |
| `python_magnetrun/panels/panel-mrecord-vs-time.py` | `python_magnetrun/panels/panel_mrecord_vs_time.py` |

Each old filename gets a one-line shim file for backwards compatibility (imports `*` from the new name), deleted in Pass 5.

### 2. Instance attributes in `MagnetDataBase` (`magnetdata_base.py`)

After Issue 1 from `fix-remaining-issues.plan.md` has renamed `self.Data → self._data`, the snake_case migration adds a public `data` property (replacing the temporary `Data` compat shim). Full set:

| PascalCase attr | snake_case attr | Notes |
|----------------|----------------|-------|
| `self.FileName` | `self.filename` | rename directly |
| `self.Groups` | `self.groups` | rename directly |
| `self.Keys` | `self.keys` | rename directly |
| `self.Data` / `self._data` | `self.data` (public property → `self._data`) | coordinated with Issue 1 fix |

Properties with the old PascalCase names emit `DeprecationWarning` for one release cycle.

### 3. Methods (~130 camelCase) — rename + deprecation shims

| camelCase | snake_case |
|-----------|-----------|
| `getData` | `get_data` |
| `getKeys` | `get_keys` |
| `getType` | `get_type` |
| `getInsert` | `get_insert` |
| `getSite` | `get_site` |
| `getHousing` | `get_housing` |
| `getStats` | `get_stats` |
| `getStartDate` | `get_start_date` |
| `getDuration` | `get_duration` |
| `cleanupData` | `cleanup_data` |
| `addData` | `add_data` |
| `removeData` | `remove_data` |
| `renameData` | `rename_data` |
| `computeData` | `compute_data` |
| `addTime` | `add_time` |
| `shiftTime` | `shift_time` |
| `extractData` | `extract_data` |
| `extractDataThreshold` | `extract_data_threshold` |
| `extractTimeData` | `extract_time_data` |
| `saveData` | `save_data` |
| `plotData` | `plot_data` |
| `fromStringIO` | `from_string_io` |
| `getUnitKey` | `get_unit_key` |
| `getTdmsData` | `get_tdms_data` |
| `addTdmsTime` | `add_tdms_time` |
| `addTdmsTimestamp` | `add_tdms_timestamp` |
| `getInfo` | `get_info` |
| `getUnit` | `get_unit` |
| `setCadref` / `getCadref` | `set_cadref` / `get_cadref` |
| `setStatus` / `getStatus` | `set_status` / `get_status` |
| `setCategory` / `getCategory` | `set_category` / `get_category` |
| `getMaterial` / `setMaterial` | `get_material` / `set_material` |
| `getMaterialProperty` / `setMaterialProperty` | `get_material_property` / `set_material_property` |
| `setParts` / `addPart` / `getParts` | `set_parts` / `add_part` / `get_parts` |
| `getTimestamp` / `setTimestamp` | `get_timestamp` / `set_timestamp` |
| `getLink` / `setLink` | `get_link` / `set_link` |
| `getDataFilename` | `get_data_filename` |
| `getTable` / `getMagnetPart` etc. | `get_table` / `get_magnet_part` etc. |
| `createSession` | `create_session` |
| `prepareData` | `prepare_data` |
| `setSite` / `setHousing` | `set_site` / `set_housing` |

Shim pattern:

```python
def getData(self, *args, **kwargs):
    import warnings
    warnings.warn(
        "getData() is deprecated and will be removed in v1.0. Use get_data() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.get_data(*args, **kwargs)
```

### 4. Local camelCase variables (9 instances)

| File | Old | New |
|------|-----|-----|
| `magnetdata.py` | `rawData` | `raw_data` |
| `flow_params.py` | `xHat`, `yHat`, `eqnHat` | `x_hat`, `y_hat`, `eqn_hat` |
| `processing/correlations.py` | `nFields` | `n_fields` |
| `processing/plateaux.py` | `xField`, `yField` | `x_field`, `y_field` |
| `commands/stats.py` | `xField`, `yField` | `x_field`, `y_field` |

---

## Migration Strategy

**Rename + deprecation alias**: keep old camelCase names as `DeprecationWarning` shims so external callers (`python_magnetcooling`, `examples/`, `tests/`) don't break immediately. Update internal callers in a second pass, then strip shims in a final pass.

---

## Versioning & Breaking Changes

| Pass | Version bump | What changes |
|------|-------------|--------------|
| Pass 1–2 (aliases added, internal callers updated) | **minor** (e.g. `0.x → 0.(x+1)`) | New snake_case API added; old camelCase still works with `DeprecationWarning` |
| Pass 3–4 (tests / examples / external subpackage updated) | patch within same minor | No API change, just housekeeping |
| Pass 5 (aliases removed) | **major** (e.g. `0.x → 1.0`) | camelCase names fully removed — breaking release |

### CHANGELOG entry for the major bump

```
## [1.0.0] — Breaking Changes

### Removed
- All camelCase method aliases (`getData`, `getKeys`, `plotData`, etc.) removed.
  Use snake_case equivalents (`get_data`, `get_keys`, `plot_data`, …) instead.
- PascalCase module files removed (`MagnetRun.py`, `GObject.py`, `HMagnet.py`,
  `MRecord.py`, `BandH.py`). Import from lowercase equivalents.
- PascalCase instance attributes (`FileName`, `Groups`, `Keys`, `Data`) removed
  from `MagnetDataBase`. Use `filename`, `groups`, `keys`, `data`.

### Migration guide
Run the following one-liner to find remaining camelCase call sites in your code:
    grep -rn '\.[a-z][a-zA-Z]*[A-Z][a-zA-Z]*(' your_project/ --include='*.py'
```

---

## Execution Order

### Pass 1 — Core class definitions (minor version bump; no external breakage)
Add snake_case names, keep camelCase as `DeprecationWarning` shims.

Files:
- `magnetdata_base.py`
- `magnetdata_pandas.py`
- `magnetdata_tdms.py`
- `MagnetRun.py` → `magnetrun.py` (rename file + add shim `MagnetRun.py`)
- `hybrid/hybrid_run.py`
- `hybrid/data_protocol.py`
- `hybrid/hybrid_data.py`
- `requests/GObject.py` → `gobject.py` (+ shim)
- `requests/HMagnet.py` → `hmagnet.py` (+ shim)
- `requests/MRecord.py` → `mrecord.py` (+ shim)
- `requests/webscrapping.py`
- `requests/connect.py`
- `runetl.py`

### Pass 2 — Internal callers
Update all call sites within `python_magnetrun/` to use snake_case (no callers should trigger DeprecationWarnings in the package's own code after this pass):
- `commands/` (stats.py, etc.)
- `processing/` (correlations.py, plateaux.py, filters.py)
- `analysis/`
- `panels/` (also rename hyphenated filenames)
- `waterflow_pipeline.py`
- Local variable renames (scope 4 above)

### Pass 3 — Tests and examples
- `tests/test_python_magnetrun.py`, `tests/test_waterflow_pipeline.py`
- `tests/analysis/` (test_metrics, test_synchronization, test_plotting, test_loaders)
- `examples/`

### Pass 4 — External subpackage (coordinate with `python_magnetcooling` maintainer)
`python_magnetcooling/` uses: `getKeys`, `getInsert`, `getStartDate`, `getDuration`, `getSite`, `getHousing`.

### Pass 5 — Cleanup (major version bump)
- Remove all deprecation shim methods
- Delete PascalCase shim module files
- Verify with grep (see Verification below)

---

## Critical Files

- [magnetdata_base.py](../python_magnetrun/magnetdata_base.py) — ABC with PascalCase attrs + abstract methods
- [magnetdata_pandas.py](../python_magnetrun/magnetdata_pandas.py) — 19 methods
- [magnetdata_tdms.py](../python_magnetrun/magnetdata_tdms.py) — 18 methods
- [MagnetRun.py](../python_magnetrun/MagnetRun.py) — 13 methods, to be renamed
- [hybrid/hybrid_run.py](../python_magnetrun/hybrid/hybrid_run.py) — 14 methods
- [requests/GObject.py](../python_magnetrun/requests/GObject.py) — 10 methods
- [requests/HMagnet.py](../python_magnetrun/requests/HMagnet.py) — 7 methods
- [requests/MRecord.py](../python_magnetrun/requests/MRecord.py) — 10 methods

---

## Verification

After each pass:

```bash
source magnetrun-env/bin/activate
python -m pytest tests/ -x -q
python -m pytest python_magnetcooling/tests/ -x -q   # after Pass 4
```

After Pass 2, confirm no internal DeprecationWarnings:

```bash
python -W error::DeprecationWarning -m pytest tests/ -x -q
```

After Pass 5, confirm no camelCase call sites remain:

```bash
grep -rn '\.[a-z][a-zA-Z]*[A-Z][a-zA-Z]*(' python_magnetrun/ --include='*.py'
# expected: zero matches
```

---

## Ordering advice vs `fix-remaining-issues.plan.md`

**Fix remaining issues first, then do the snake_case migration.**

Reasons:

1. **Separate concerns in git history.** Issues 1–4 are behavioral fixes (abstraction
   violations, LSP violation, protocol duplication). The snake_case migration is a pure
   rename. Mixing them makes PRs harder to review and bisect.

2. **Issue 1 and this plan overlap on `self.Data`.** `fix-remaining-issues.plan.md`
   renames `self.Data → self._data` (private storage). This plan then adds a public
   `self.data` property that wraps `self._data`. If done in the right order the two
   plans compose cleanly; if reversed there is a conflict.

3. **Issue 4 (protocol consolidation)** reduces the number of things to rename: once
   `DataProvider` is merged into `DataLoader`, Pass 1 of this plan only has one
   protocol to deal with.

4. **Snake_case touches every file.** A pure rename sweep is easiest to review when the
   code is already functionally correct — no logic changes hidden among hundreds of
   identifier renames.
