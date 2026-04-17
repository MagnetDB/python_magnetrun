# Package Review: `python_magnetrun`

Date: 2026-04-16 (updated)

---

## Package Structure

```
python_magnetrun/
├── magnetdata_base.py       # ABC
├── magnetdata_pandas.py     # Pandas impl
├── magnetdata_tdms.py       # TDMS impl
├── magnetdata.py            # Factory + backward-compat shim
├── MagnetRun.py             # Session container
├── runetl.py                # ETL helpers
├── field_defs.py / housing_config.py  # Config layer
├── cli.py / cli_args.py / args.py  # CLI entry points
├── commands/                # Modular CLI subcommands
├── analysis/                # Analysis pipeline
├── hybrid/                  # FEPC kHz/RMS/Trigger data
├── processing/              # Signal processing
├── utils/ / runlogs/ / requests/ / configAlims/
```

Overall the layering is sensible: ABC → implementations → session wrapper → CLI. But there are several coherence and implementation issues worth addressing.

---

## Class Hierarchy

```
MagnetDataBase (ABC)
├── PandasMagnetData
│   ├── EnsightMagnetData
│   ├── BProfileMagnetData
│   └── FeelppMagnetData
└── TdmsMagnetData

load_magnetdata(filename)   ← standalone factory (magnetdata.py)

MagnetRun                   ← owns a MagnetDataBase instance, uses load_magnetdata
HybridRun                   ← mirrors MagnetRun by convention, not contract
```

---

## Issues by Severity

### Critical

**1. Three parallel sources of truth for housing/sensor-role mapping** *(done)*

`housing_config.py` is now the single source of truth. `field_mappings.py` has been deleted and
`runetl.prepareData_legacy` has been removed. `runetl.prepareData` is fully driven by
`HousingConfig` (formula maps, rename map, voltage formulas). `MagnetRun.fromtxt` and
`MagnetRun.fromtdms` both call `prepareData`. See plan below for a summary of completed steps.

**2. `MagnetData` is a factory masquerading as a subclass** *(done)*

`magnetdata.py` is now a factory entry-point module, not a subclass. The old `MagnetData` class
has been replaced by `load_magnetdata(filename, defs_file)` which dispatches on file extension:
`.tdms` → `TdmsMagnetData` via the internal `_fromtdms()` helper; `.txt`/`.csv` →
`PandasMagnetData`. `MagnetRun.fromtxt`, `fromtdms`, and `fromcsv` all call `load_magnetdata`.
`isinstance` checks are now reliable since callers get the concrete subclass directly.

**3. `runetl.prepareData_legacy` hardcodes housing logic** *(done)*

`prepareData_legacy` has been removed. `prepareData` is the only ETL entry point and is driven
entirely by `HousingConfig`. `MagnetRun.fromtxt` calls it directly.

---

### Significant

**4. Dead/unreachable code in `PandasMagnetData.Units`** *(done)*

`magnetdata_pandas.py` `Units()` now uses a clean resolution order: JSON file → legacy pattern
matching fallback. The unconditional `raise RuntimeError` is gone.

**5. `MagnetRun.saveData` breaks the abstraction** *(done)*

`MagnetRun.saveData` now delegates to `self.MagnetData.saveData(self.MagnetData.getKeys(), filename)`
([MagnetRun.py:201-204](python_magnetrun/MagnetRun.py#L201-L204)). The inline `isinstance` check is gone.

**6. `TdmsMagnetData.getUnitKey` ignores `self.units`** *(done)*

`getUnitKey` now checks `self.units[key]` first and falls back to `PigBrotherUnits` only as a last
resort ([magnetdata_tdms.py:313-316](python_magnetrun/magnetdata_tdms.py#L313-L316)). The resolution
order is now consistent with `Units()`.

**7. Incompatible `Data` attribute type across subclasses** *(done)*

All external callers outside the two subclasses have been cleaned up — `MagnetRun.py` no longer
accesses `.Data` directly. The `pd.DataFrame | dict` union type annotation remains in
`magnetdata_base.py:103` as an internal documentation detail; all external access goes through
`getData()`. `Data` is effectively a private implementation detail of each subclass.

**8. Two conflicting Protocol definitions for the `MagnetRun`/`HybridRun` interface** *(done)*

`DataProvider` has been removed from `hybrid/hybrid_run.py`. `DataLoader` in
`hybrid/data_protocol.py` is now the single protocol definition.

---

### Minor

**8. Hardcoded developer path as CLI default** *(done)*

The hardcoded `default="/home/LNCMI-G/christophe.trophime/LNCMIG-Data/srv-data-install"` has been
removed from `cli_args.py`. The argument now defaults to `None` or an environment variable lookup.

**9. `analysis/__init__.py` exports 80+ names flat**

The `analysis/` subpackage feels monolithic. Config, loaders, synchronization, metrics, and
plotting are all dumped into one namespace. Splitting into explicit sub-namespaces (e.g.,
`analysis.metrics`, `analysis.plot`) would improve discoverability.

**10. `processing/cli.py` and `analysis/cli.py` are independent CLIs**

They do not participate in the `commands/` subpackage pattern used by `cli.py`. Two parallel
mini-CLI systems with different argument conventions exist side by side.

**11. Editor backup file in the package**

`python_magnetrun/pigbrother-defs.json~` could be accidentally included in sdist/wheel builds.
Add it to `.gitignore` and remove it from the repository.

**12. `tsdownsample` is an undeclared dependency**

Used in `hybrid/hybrid_run.py` behind a `try/except ImportError`, but absent from `pyproject.toml`.
Either add it to a `hybrid` extras group or document the soft requirement explicitly.

---

## Code Duplication Summary

| Duplicate area | Locations |
|---|---|
| Protocol for `MagnetRun`/`HybridRun` interface | `DataProvider` in `hybrid_run.py`, `DataLoader` in `data_protocol.py` |
| Plot logic | `commands/plot.py`, legacy `viewcsv.py` |
| Argument parsing for smoothing/logging | `cli_args.py` builders vs. `processing/cli.py` inline argparse |

---

## Overall Assessment

| Area | Status |
|---|---|
| ABC design (`MagnetDataBase`) | Good |
| `field_defs.py` + JSON defs | Good |
| `HousingConfig` dataclass + user override path | Good |
| `HybridRun` LRU cache / Protocol approach | Good |
| `MagnetData` shim architecture | Done |
| Housing config de-duplication | Done |
| `prepareData_legacy` hardcoding | Done |
| `PandasMagnetData.Units` dead code | Done |
| `Units`/`getUnitKey` consistency in TDMS | Done |
| `Data` attribute type divergence (`DataFrame` vs `dict`) | Done |
| CLI consolidation | Needs work |
| `saveData` abstraction in `MagnetRun` | Done |
| Hardcoded default path in `cli_args.py` | Done |
| Protocol duplication (`DataProvider` / `DataLoader`) | Done |
| Timestamp convention (Pandas + TDMS) | Done |
| `HybridData` timestamp support | Pending (out of current scope) |
| Cross-domain comparison (`DataLoader` extension, Phase A1–A3) | In progress — A0 done, A1 partial |
| Downsampling refactoring (`DownsampleConfig`, shared module) | Planned — see `downsampling-refactoring.plan.md` |
| Plotting refactoring (`plotting/` subpackage, backend protocol, JS path) | Planned — see `plotting-refactoring.plan.md` |

The core abstractions are well-conceived — the ABC, the defs system, and `HousingConfig` are solid
foundations. All major structural issues are now resolved: housing config consolidation, `MagnetData`
shim replacement, `getUnitKey` fix, `saveData` delegation, hardcoded-path removal, `Data` type
divergence (external callers cleaned up), Protocol unification (`DataLoader` only), and timestamp
convention (`PandasMagnetData` + `TdmsMagnetData` both store naive UTC). Remaining work is:
Phase A1–A3 of the cross-domain comparison plan (`getDomain()` in protocol + runners,
`MagnetRun.get_time_range()` delegation, protocol compliance tests), CLI fragmentation, minor
housekeeping (`tsdownsample` extras, editor backup file), and `HybridData` timestamp support
(out of current scope, tracked in [`prompts/hybriddata-timestamp-plan.md`](hybriddata-timestamp-plan.md)).

---

## Recommended Priority Order

### Issue 1 — Consolidate housing/sensor-role config *(done)*

**All steps completed:**
- `site_config.py` → `housing_config.py`; `SiteConfig` → `HousingConfig`; `SITE_CONFIGS` → `HOUSING_CONFIGS`
- `*-site-config.json` → `*-housing-config.json`
- `AnalysisConfig.for_site` → `for_housing`; `AnalysisConfig.site` field → `housing`
- `HousingConfig` extended with `pupitre_formula_map`, `pigbrother_formula_map`,
  `hybrid_formula_map`, `reference_gr1/2_voltage`, `get_pupitre_rename_map()`,
  `get_pupitre_voltage_formulas()`, `get_hybrid_voltage_formulas()`
- `runetl.prepareData` fully driven by `HousingConfig`; `prepareData_legacy` removed
- `field_mappings.py` deleted
- `MagnetRun.fromtxt` and `MagnetRun.fromtdms` both call `prepareData`

**Note — plan/code discrepancy resolved: `hybrid_formula_map: dict` is correct**

`prompts/prepareData-implementation.md` originally specified two plain string fields
(`hybrid_gr1_current_formula`, `hybrid_gr2_current_formula`). The code correctly uses
`hybrid_formula_map: dict` instead, keyed by channel name (e.g. `"FEPC-AUX-LNCMI/ALIM1"`,
`"FEPC-AUX-LNCMI/ALIM2"`). This is consistent with `pupitre_formula_map` and
`pigbrother_formula_map`, and is more general (not limited to exactly two GR currents).
The prompt doc is the stale artifact — it should be updated to reflect the dict approach,
but the code is complete as-is. `runetl.prepareData` already unpacks `cfg.hybrid_formula_map`
correctly (`runetl.py:99`).

---

### Issue 2 — Replace `MagnetData` shim with a standalone factory function *(done)*

`magnetdata.py` is now a pure factory module exposing `load_magnetdata(filename, defs_file)`.
`MagnetRun.fromtxt`, `fromtdms`, and `fromcsv` all call it. TDMS loading logic lives in the
private `_fromtdms()` helper. No `MagnetData` class remains; `isinstance` checks are reliable.

---

### Remaining issues (priority order)

Effort key: **S** = ~1 h, **M** = half-day, **L** = 1–2 days, **XL** = several days.

1. **Timestamp convention** *(done)* — `PandasMagnetData.addTime()` now converts local → naive UTC
   ([magnetdata_pandas.py:471-479](python_magnetrun/magnetdata_pandas.py#L471-L479)).
   `TdmsMagnetData.addTime()` stores naive UTC. Both subclasses are consistent.
   See [`prompts/timestamp-utc-refactoring.plan.md`](timestamp-utc-refactoring.plan.md).

   **`HybridData` timestamp support** *(pending, out of current scope)* — tracked in
   **[`prompts/hybriddata-timestamp-plan.md`](hybriddata-timestamp-plan.md)**:
   add `start_timestamp`, `end_timestamp`, `_infer_timestamps()`, `addTime()`,
   `getStartDate()`, `getDuration()` to `HybridData`.

2. **`Data` attribute type divergence** *(done)* — no external `.Data` access remains outside
   the two subclasses. `MagnetRun.py` and all callers route through `getData()`. The
   `pd.DataFrame | dict` annotation in `magnetdata_base.py:103` is an internal detail only.

3. **`TdmsMagnetData.getUnitKey`** *(done)* — now checks `self.units[key]` first, falling back
   to `PigBrotherUnits` only as a last resort
   ([magnetdata_tdms.py:313-316](python_magnetrun/magnetdata_tdms.py#L313-L316)).

4. **`MagnetRun.saveData`** *(done)* — now delegates to `self.MagnetData.saveData(...)`
   ([MagnetRun.py:201-204](python_magnetrun/MagnetRun.py#L201-L204)).

5. **Protocol duplication** *(done)* — `DataProvider` removed from `hybrid/hybrid_run.py`;
   `DataLoader` in `hybrid/data_protocol.py` is the single protocol.

6. **Hardcoded default path** *(done)* — removed from `cli_args.py`.

7. **`tsdownsample` + downsampling refactoring** *(effort: M)* — extract `downsample_data()` from
   `hybrid_run.py` into `python_magnetrun/utils/downsampling.py`; introduce `DownsampleConfig`
   dataclass; add downsampling support to `PandasMagnetData` and `TdmsMagnetData`; update
   `DownsamplingLoader` protocol; reconcile `analysis/processing.py` percentage model.
   Add `tsdownsample` to `pyproject.toml` as a `hybrid` extras dependency.
   Full plan: **[`prompts/downsampling-refactoring.plan.md`](downsampling-refactoring.plan.md)**.

8. **Plotting refactoring** *(effort: L)* — create `python_magnetrun/plotting/` subpackage with
   `PlottingBackend` protocol, `MatplotlibBackend`, `PlotlyBackend`, `plot_subplots()`,
   `plot_overlay()` (with normalization), and `AnnotationManager`.  Adds JS-frontend path via
   `to_json()` and native marimo / voilà support via the Plotly backend.
   Full plan: **[`prompts/plotting-refactoring.plan.md`](plotting-refactoring.plan.md)**.
   Depends on downsampling plan Steps 1–2 for full method selection.

9. **Editor backup file** *(effort: S)* — remove `pigbrother-defs.json~` and add `*.json~` to
   `.gitignore`.

---

### Cross-Domain Comparison — Phase A remaining work

Full plan in **[`prompts/cross-domain-comparison.prompt.md`](cross-domain-comparison.prompt.md)**.

| Phase | Task | Status |
|---|---|---|
| A0 | Delete `DataProvider` from `hybrid_run.py` | Done |
| A1 | Add `get_time_range()` to `DataLoader` protocol | Done (`data_protocol.py:177`) |
| A1 | Add `getDomain()` to `DataLoader` protocol | **Todo** |
| A2 | `HybridRun.get_time_range()` | Done (`hybrid_run.py:822`) |
| A2 | `HybridRun.getDomain() → "operational"` | **Todo** |
| A2 | `MagnetRun.get_time_range()` delegation | **Todo** |
| A2 | `MagnetRun.getDomain() → "operational"` | **Todo** |
| A3 | Protocol compliance tests (`tests/test_protocol.py`) | **Todo** |
| B | `SimulationRun` adapter | Not started |
| C | `BFieldRun` adapter | Not started |
| D | `CHANNEL_ALIASES` + `KeyMapping` in `analysis/config.py` | Not started |
| E | `ComparisonSession` | Not started |
| F | `magnetrun-compare` CLI | Not started |
| G | `tests/test_comparison.py` | Not started |

Immediate next step: complete Phase A1–A3 (add `getDomain()` to protocol + both runners,
add `MagnetRun.get_time_range()` delegation, write `tests/test_protocol.py`). ~2 h total.
