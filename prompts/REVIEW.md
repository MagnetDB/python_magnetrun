# Package Review: `python_magnetrun`

Date: 2026-06-05 (updated)

---

## Package Structure

```
python_magnetrun/
├── magnetdata_base.py       # ABC (DataType enum: PUPITRE/TDMS/ENSIGHT/HYBRID/HTS)
├── magnetdata_pandas.py     # Pandas impl — factory methods delegate to readers/
├── magnetdata_tdms.py       # TDMS impl
├── magnetdata.py            # Factory entry point (load_magnetdata, _fromtdms)
├── MagnetRun.py             # Session container
├── runetl.py                # ETL helpers
├── outliers.py              # Canonical outlier detection (OutlierConfig, OutlierDetector, …)
├── field_defs.py / housing_config.py  # Config layer
├── cli.py                   # CLI entry point
├── cli_args.py / args.py    # CLI argument parsing
├── commands/                # Modular CLI subcommands
├── readers/                 # Pure I/O readers — one class per format (Stream 3.6 ✅)
│   ├── base.py              #   Reader protocol (runtime-checkable)
│   ├── csv_readers.py       #   PupitreReader, BProfileReader, EnsightReader, FeelppReader, CsvReader
│   ├── tdms_reader.py       #   TdmsReader (validate + t-offset config)
│   ├── hts_reader.py        #   HtsReader (new: ; sep, units-in-header)
│   ├── hybrid_reader.py     #   HybridReader (composite discovery)
│   └── registry.py          #   READERS/CONTAINERS dicts + detect_type()
├── analysis/                # Analysis pipeline
├── hybrid/                  # FEPC kHz/RMS/Trigger data (outliers.py is a shim → python_magnetrun.outliers)
│   └── hybrid_data.py       #   HybridData now inherits MagnetDataBase (Stream 3.6 R4 ✅)
├── processing/              # Signal processing (signal.py: normalize_signal, binarize_signal)
├── plotting/                # Plotting backends & utilities
├── utils/ / runlogs/ / requests/ / configAlims/
```

Overall the layering is sensible: ABC → implementations → session wrapper → CLI. Major structural issues have been resolved. The package is production-ready for core use cases.

---

## Class Hierarchy

**Current (Stream 3.6 complete):**
```
MagnetDataBase (ABC)
├── PandasMagnetData
│   ├── EnsightMagnetData
│   ├── BProfileMagnetData
│   └── FeelppMagnetData
├── TdmsMagnetData
└── HybridData              ← joined hierarchy (Stream 3.6 R4 ✅); field_meta init bug fixed

readers/ subpackage         ← pure I/O, no data manipulation (Stream 3.6 ✅)
  PupitreReader, BProfileReader, EnsightReader, FeelppReader, CsvReader
  TdmsReader, HtsReader (DataType.HTS = 4), HybridReader (composite)
  registry.py: READERS/CONTAINERS dicts + detect_type()

load_magnetdata(filename, fmt=)  ← uses detect_type() from registry (magnetdata.py)

MagnetRun                   ← owns a MagnetDataBase instance
HybridRun                   ← satisfies DataLoader protocol

Thin subclasses (EnsightMagnetData, BProfileMagnetData, FeelppMagnetData) may optionally
be dissolved in a future cleanup — they differ only in their reader and _TYPE.
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

`prepareData_legacy` has been removed entirely. `prepareData` is the only ETL entry point and is driven
entirely by `HousingConfig`. `MagnetRun.fromtxt` and `MagnetRun.fromtdms` both call it directly.

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

**7. Incompatible `Data` attribute type across subclasses** *(done — upgraded to abstract property)*

`Data` is now an abstract `@property` (getter + setter) on `MagnetDataBase`
([magnetdata_base.py:145-152](python_magnetrun/magnetdata_base.py#L145-L152)).
`PandasMagnetData` stores data in a private `_data: pd.DataFrame` backing attribute and
exposes it via `Data`; the property getter calls `_ensure_data_loaded()` for lazy loading.
`TdmsMagnetData` uses a `_LazyGroupDict` backing attribute; the property returns the container
and per-group loading is deferred to `_LazyGroupDict.__getitem__`.  The old `__getattribute__`
override in `PandasMagnetData` is gone.  `close()` and context-manager support (`__enter__` /
`__exit__`) are now part of the base class interface.  The `_validate_start_timestamp` method
accesses `self._data` directly (bypassing the property) to avoid triggering a full load during
`__init__`.  See [`prompts/data-property-abc.plan.md`](data-property-abc.plan.md).

**8. Two conflicting Protocol definitions for the `MagnetRun`/`HybridRun` interface** *(done — Phase 2A complete)*

`DataProvider` has been removed from `hybrid/hybrid_run.py`. `DataLoader` in
`hybrid/data_protocol.py` is now the single protocol definition. Both `MagnetRun` and `HybridRun`
satisfy the `DataLoader` protocol with `get_time_range()` and `getDomain()` methods implemented.
Cross-domain comparison Phase A0–A3 is complete.

---

### Minor

**8. Hardcoded developer path as CLI default** *(done)*

The hardcoded `default="/home/LNCMI-G/christophe.trophime/LNCMIG-Data/srv-data-install"` has been
removed from `cli_args.py`. The argument now defaults to `None` or an environment variable lookup.

**9. `analysis/__init__.py` exports 80+ names flat** *(minor — open)*

The `analysis/` subpackage feels monolithic. Config, loaders, synchronization, metrics, and
plotting are all dumped into one namespace. Splitting into explicit sub-namespaces (e.g.,
`analysis.metrics`, `analysis.plot`) would improve discoverability.

**10. `processing/cli.py` and `analysis/cli.py` are independent CLIs** *(minor — planned)*

They do not participate in the `commands/` subpackage pattern used by `cli.py`. Two parallel
mini-CLI systems with different argument conventions exist side by side.
Full plan: **[`prompts/cli-consolidation.plan.md`](cli-consolidation.plan.md)**.

**11. Editor backup file in the package** *(trivial — done)*

`pigbrother-defs.json~` removed; `*.json~` added to `.gitignore`.

**12. `tsdownsample` is an undeclared dependency** *(done)*

Now added to `pyproject.toml` as a `hybrid` extras dependency.

**13. Truncated / malformed pupitre files not handled** *(done)*

`PandasMagnetData.fromtxt` now uses a two-attempt encoding helper (`UTF-8` → `Latin-1` fallback),
passes `on_bad_lines="warn"` to `pd.read_csv`, raises a clear `FileFormatError` for header-only
files, and calls `check_pupitre_truncation()` (new in `utils/validation.py`) before and after
loading.  `analysis/loaders.py` catch blocks now include `UnicodeDecodeError`.
See [`prompts/truncated-pupitre-files.plan.md`](truncated-pupitre-files.plan.md) and
`tests/test_truncated_pupitre.py` (153 lines, 6 test cases).

**14. `addData` / `computeData` lacked metadata** *(done)*

Both methods on `MagnetDataBase` / `PandasMagnetData` now accept `symbol`, `unit`, `label`,
and `description` parameters.  On success they store a `FieldMeta` entry in `self.field_meta`
so that plot labels and unit annotations are automatically propagated.  Housing-config formula
maps (`pupitre_formula_map`, `pigbrother_formula_map`, `hybrid_formula_map`) and JSON definition
files now carry the same four keys.  `commands/add.py` passes all four when calling
`addData` / `computeData`.

**15. `HybridRun.getData` failed on `hybrid_formula_map` keys** *(done)*

Keys in `hybrid_formula_map` (e.g. `"FEPC-AUX-LNCMI/ALIM1"`) were parsed as `type/system` by
`HybridRun.getData`, causing `ValueError: Unknown data type`.  A new
`_resolve_hybrid_formula()` helper is called before the parse block: it splits the formula
string into operands, maps each to a `kHz/…` channel, calls `getData` recursively (with
caching), and returns the element-wise sum.  Only `+` is supported; other operators raise
`NotImplementedError`.  See [`prompts/hybrid-formula-key-resolution.plan.md`](hybrid-formula-key-resolution.plan.md)
and `tests/test_hybrid_formula_resolution.py`.

---

## Code Duplication Summary

| Duplicate area | Locations | Status |
|---|---|---|
| Protocol for `MagnetRun`/`HybridRun` interface | `DataProvider` in `hybrid_run.py`, `DataLoader` in `data_protocol.py` | Done |
| `Référence_GR → Courant_GR` mapping | `config.py ChannelMapping` + `analysis/cli.py:162-165` | Done — cli.py now uses `channel_map.to_dict()` |
| Plot logic | `commands/plot.py`, legacy `viewcsv.py` | Refactored — `plotting/` subpackage created |
| Argument parsing for smoothing/logging | `cli_args.py` builders vs. `processing/cli.py` inline argparse | Planned — see `cli-consolidation.plan.md` |
| Outlier detection | `hybrid/outliers.py` (canonical) + `processing/hysteresis.py::remove_outliers` (inline IQR/zscore/MAD) + `examples/outliers.py` (rolling-MAD inline) + 2 CLI-style test scripts | Done — canonical moved to `python_magnetrun/outliers.py`; `hybrid/outliers.py` is a shim; `OutlierConfig` dataclass + `OUTLIER_DEFAULTS` added; `create_outlier_parser`/`args_to_outlier_config` in `cli_args.py`; `hybrid_data.py` plot methods accept `OutlierConfig`; signal functions in `processing/signal.py`; `test_outliers.py` (142 tests); `isolation_forest` in `OutlierMethod` |
| `RMSFileReader` / `VProcessFileReader` near-identical classes | `hybrid/rms/rms_reader.py`, `hybrid/vprocess/vprocess_reader.py` | Open — Stream 3.9 L2 (`_BinaryFileReaderBase` + single `ChannelVariable` dataclass) |
| UTC→local hour conversion — 4 independent implementations | `hybrid_data.py` (×2), `analysis/processing.py`, `analysis/loaders.py` | Partial — Phase 2B B0.5 adds `utc_hour_to_local()` to `hybrid/utils.py` and consolidates 3 sites; `analysis/loaders.py` already UTC |
| `plot_khz_variable` / `plot_rms_variable` ~80 % identical pipeline | `hybrid/plotting.py:444` and `:564` | Open — Stream 3.9 L1; highlight-mode double-read bug in RMS tracked as Phase 2B B2.5 |
| `plot_khz_variables` / `plot_rms_variables` ~75 % identical | `hybrid/plotting.py:152` and `:314` | Open — Stream 3.9 L1 (same `_plot_variables_impl` extraction); RMS missing `downsample` param |
| `_resolve_backend` exact duplicate | `hybrid/plotting.py:50`, `plotting/timeseries.py:34` | Open — Stream 3.9 S2 |
| `log_exception` / `format_exception_location` — incompatible or duplicated signatures | `hybrid/utils.py:32`/`:97`, `log_utils.py:305`/`:361` | Open — Stream 3.9 M1 |
| `apply_calibration` — 3 independent implementations | `hybrid/kHz/fepc_reader.py:821`, `hybrid/trigger/trigger_reader.py:705`, `processing/distance.py:24` (different domain) | Open — Stream 3.9 M4 (shared `_apply_cnv_calibration` helper) |
| `compute_lag` / `lag_correlation` — duplicated with incompatible `range` key schema | `processing/correlations.py`, `analysis/synchronization.py` | Open — Stream 3.9 M2/M3; `processing/correlations.py` to be deprecated via shims |

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
| `Data` attribute type divergence — now abstract property with lazy loading | Done |
| Truncated / malformed pupitre file handling | Done — see `truncated-pupitre-files.plan.md` |
| `addData`/`computeData` metadata (`symbol`, `unit`, `label`, `description`) | Done |
| `HybridRun.getData` formula-key resolution (`hybrid_formula_map`) | Done — see `hybrid-formula-key-resolution.plan.md` |
| `Ih`/`Ib` defined via `Idcct` in housing configs | Done |
| CLI consolidation (8 → 3 entry points, `magnetrun` dispatcher, `register()` pattern) | Done — see `cli-consolidation.plan.md` |
| Outlier deduplication (`hybrid/outliers.py` canonical, thin delegates, proper tests) | Done — `examples/outliers.py` deleted; `hysteresis.py` delegates to `detect_outliers()`; `tests/test_outliers.py` (142 tests); `ISOLATION_FOREST` added to `OutlierMethod` |
| `saveData` abstraction in `MagnetRun` | Done |
| Hardcoded default path in `cli_args.py` | Done |
| Protocol duplication (`DataProvider` / `DataLoader`) | Done |
| Timestamp convention (Pandas + TDMS) | Done |
| `HybridData` timestamp support | Pending (prerequisite `analysis/` Phase 6 is now done) |
| Cross-domain comparison (`DataLoader` extension, Phase A0–A3) | Done |
| Cross-domain comparison (Phases D–G: `ComparisonSession`, adapters, CLI) | Pending (Phases B–C done) |
| Reader/container split (`readers/` subpackage, Phases R1–R5) | Done — see `reader-container-refactoring.plan.md`; `HybridData` in hierarchy; `field_meta` bug fixed; 46 new tests |
| Pattern entries in `*-defs.json` + `feelpp-defs.json` (Phase H) | Pending — extends cross-domain plan |
| TDMS export (`PandasMagnetData.to_tdms()`) | Pending — see `pupitre_to_tdms_export.md` |
| TDMS export (`HybridData.to_rms_tdms()` + `to_khz_tdms()`) | Pending — see `hybrid_to_tdms_export.md`; `field_meta` fix now done (3.6 R4) |
| M4 / NaN-M4 downsampling methods | Done — see `m4-downsampling.plan.md` |
| RDP / Visvalingam-Whyatt downsampling methods | Done — see `rdp-downsampling.plan.md` |
| Downsampling quality metrics (`DownsampleMetrics`, `benchmark_configs`) | Done — see `downsampling-metrics.plan.md` |
| Downsampling refactoring (`DownsampleConfig`, shared module) | Done — see `downsampling-refactoring.plan.md` |
| Hybrid code quality — `_BinaryFileReaderBase`, `_plot_variable_impl`, UTC utility, `log_exception` unification (Stream 3.9, 16 findings) | Open — see `docs/hybrid_refactoring_notes.md` |
| Trigger & VProcess integration into `HybridData` (Stream 4.6) | Pending — depends on Stream 3.9 L2 (`_BinaryFileReaderBase`) |
| Plotting refactoring (`plotting/` subpackage, backend protocol) | Done — see `plotting-refactoring.plan.md` or `holoviews-migration.plan.md` for alternative |
| `analysis/` internal refactoring (data loading, channel mapping, decomposition) | Done — see `analysis-subpackage-refactoring.plan.md` |
| `hybrid/` internal refactoring (outlier dedup, `OutlierConfig`, signal processing) | Done — all 6 phases complete; see `hybrid-subpackage-refactoring.plan.md` |
| Pipeline redesign: polars npTDMS, narwhals, no-double-load pipeline | Planned — see `mrun-cache-implementation.plan.md` (3 phases; Phase 1 independent) |
| File validation infrastructure | Done — `utils/validation.py` committed and integrated |
| Logging infrastructure | Done — `log_utils.py` in place; `print()` migration ongoing |

The core abstractions are well-conceived — the ABC, the defs system, and `HousingConfig` are solid
foundations. All major structural issues are now resolved: housing config consolidation, `MagnetData`
shim replacement, `getUnitKey` fix, `saveData` delegation, hardcoded-path removal, `Data` promoted to
an abstract property with lazy loading (`PandasMagnetData._ensure_data_loaded` + `TdmsMagnetData._LazyGroupDict`
+ context-manager support), Protocol unification (`DataLoader` only, Phase 2A complete), timestamp
convention (`PandasMagnetData` + `TdmsMagnetData` both store naive UTC), downsampling refactoring
(`DownsampleConfig`, shared `utils/downsampling.py`, `tsdownsample` extras; M4/NaN-M4, RDP/VW, and
quality metrics all complete), plotting refactoring (subpackage, backends, label/legend uniformization),
file validation infrastructure (committed and integrated), resilient pupitre-file loading (encoding
fallback, `on_bad_lines="warn"`, empty-data guard, truncation check), `addData`/`computeData` metadata
parameters (`symbol`, `unit`, `label`, `description` → `FieldMeta`), `HybridRun.getData` formula-key
resolution (`_resolve_hybrid_formula`), CLI consolidation (13-subcommand `magnetrun` dispatcher,
`register()` pattern, `main.py`), and reader/container split (`readers/` subpackage, `HybridData`
joins `MagnetDataBase`, `DataType.HTS`, registry + `detect_type()`; `field_meta` init bug fixed;
971 tests pass).
A cross-module review (`docs/hybrid_refactoring_notes.md`) identified 16 code-quality findings in the
`hybrid/` subpackage and broader package; tracked as Stream 3.9 (S/M/L items — safe_float, _resolve_backend,
log_exception, range schema, CNV calibration, plot unification, _BinaryFileReaderBase) and Stream 4.6
(XL — trigger/VProcess integration into `HybridData`) in ROADMAP.
An optional HoloViews-based plotting system (~8 d) would replace the current three-backend
implementation with `hv.extension()` + Panel + datashader and subsume `analysis/` downsampling Phase 2.

**Package is production-ready for core use cases.** Remaining work in priority order:
(1) known regressions (multiple-file plotting); (2) CI already in place (`test.yml` + `docs.yml`; `ruff` via pre-commit); (3) logging migration completion;
(4) `hybrid/` internal refactoring — **all 6 phases complete** (`hybrid-subpackage-refactoring.plan.md`);
(5) CLI consolidation — ✅ done (`cli-consolidation.plan.md`);
(6) hybrid code quality — Stream 3.9 S/M/L items + Stream 4.6 trigger/VProcess integration (see `docs/hybrid_refactoring_notes.md`);
(7) `HybridData` timestamp support (prerequisite `analysis/` Phase 6 complete — unblocked);
(8) cross-domain Phases D–G (`ComparisonSession`, CLI);
(9) optional HoloViews migration; (10) pipeline redesign (polars/narwhals).

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
   `HybridData` timestamp support is deferred — see item 10.

2. **`Data` attribute type divergence** *(done — upgraded to abstract property)* — `Data` is
   now an abstract `@property` on `MagnetDataBase`, implemented via `_data` backing attributes
   in both subclasses.  Lazy loading is now part of the declared contract.  `close()` and
   context-manager support added.  See item 7 above and
   [`prompts/data-property-abc.plan.md`](data-property-abc.plan.md).

3. **`TdmsMagnetData.getUnitKey`** *(done)* — now checks `self.units[key]` first, falling back
   to `PigBrotherUnits` only as a last resort
   ([magnetdata_tdms.py:313-316](python_magnetrun/magnetdata_tdms.py#L313-L316)).

4. **`MagnetRun.saveData`** *(done)* — now delegates to `self.MagnetData.saveData(...)`
   ([MagnetRun.py:201-204](python_magnetrun/MagnetRun.py#L201-L204)).

5. **Protocol duplication** *(done)* — `DataProvider` removed from `hybrid/hybrid_run.py`;
   `DataLoader` in `hybrid/data_protocol.py` is the single protocol.

6. **Hardcoded default path** *(done)* — removed from `cli_args.py`.

7. **Cross-domain comparison — Phase A1–A3** *(effort: M, do next)* — complete protocol work
   before tackling downsampling/plotting. No dependencies on items 8–9.

   Full plan: **[`prompts/cross-domain-comparison.prompt.md`](cross-domain-comparison.prompt.md)**.

   | Phase | Task | Status |
   |---|---|---|
   | A0 | Delete `DataProvider` from `hybrid_run.py` | Done |
   | A1 | Add `get_time_range()` to `DataLoader` protocol | Done (`data_protocol.py`) |
   | A1 | Add `getDomain()` to `DataLoader` protocol | Done (`data_protocol.py`) |
   | A2 | `HybridRun.get_time_range()` | Done (`hybrid_run.py`) |
   | A2 | `HybridRun.getDomain() → "operational"` | Done (`hybrid_run.py`) |
   | A2 | `MagnetRun.get_time_range()` delegation | Done (`MagnetRun.py`) |
   | A2 | `MagnetRun.getDomain() → "operational"` | Done (`MagnetRun.py`) |
   | A3 | Protocol compliance tests (`tests/test_protocol.py`) | Done |

8. **`tsdownsample` + downsampling refactoring** *(done)* — `downsample_data()` extracted from
   `hybrid_run.py` into `python_magnetrun/utils/downsampling.py`; `DownsampleConfig` dataclass
   introduced; downsampling support added to `PandasMagnetData` and `TdmsMagnetData`; `DownsamplingLoader`
   protocol updated; `analysis/processing.py` percentage model reconciled.
   `tsdownsample` added to `pyproject.toml` as a `hybrid` extras dependency.
   Full plan: **[`prompts/downsampling-refactoring.plan.md`](downsampling-refactoring.plan.md)**.

9. **Plotting refactoring** *(done — or replace with HoloViews migration)* —
   `python_magnetrun/plotting/` subpackage created with `PlottingBackend` protocol,
   `MatplotlibBackend`, `PlotlyBackend`, `PlotlyResamplerBackend`, `plot_subplots()`,
   `plot_overlay()` (with normalization), and `AnnotationManager`.  JS-frontend path via
   `to_json()`.  `analysis/plotting.py` now imports `PlotStyle`/`PlotColors` from
   `plotting.style` and uses `AnnotationManager`.  `commands/plot.py` gains `--backend`
   and `--json` flags.  `pyproject.toml` gains `plotting` and `resampler` extras groups.
   Label/legend uniformization sub-plan also complete: canonical `"symbol [unit]"` format,
   shared `format_label()` utility, consistent `ax.set_*` API across all backends.
   Full plan: **[`prompts/plotting-refactoring.plan.md`](plotting-refactoring.plan.md)**.

   **Alternative path:** replace the `PlottingBackend`/three-backend system with HoloViews +
   Panel + datashader (~8 days effort). This supersedes the current implementation and
   simplifies `analysis-subpackage-refactoring` Phase 2 (downsampling). Both the current
   state and HoloViews migration preserve `PlotStyle`/`PlotColors`/`format_label()`.
   Full plan: **[`prompts/holoviews-migration.plan.md`](holoviews-migration.plan.md)**.

10. **`analysis/` internal refactoring** *(done — all 6 phases)* — tracked in
    **[`prompts/analysis-subpackage-refactoring.plan.md`](analysis-subpackage-refactoring.plan.md)**.
    Completed (branch `rework_analysis`):
    - Phase 1: dead code removed, logging migrated, directory constants centralised
    - Phase 2: downsampling unified (`DownsampleConfig` adopted in `analysis/plotting.py`)
    - Phase 3: `utils/files.py` canonical data-loading; `analysis/loaders.py` imports from it
    - Phase 4: 4 new `HousingConfig` methods (`get_pupitre_current_channel`, `get_pupitre_group_keys`, `get_pupitre_flow_keys`, `get_hybrid_group_keys`); 5 processing.py wrappers deleted
    - Phase 5: `discover()`, `process_overview_file()`, and `main()` decomposed into focused helpers
    - Phase 6: `add_time_columns(df, t0, sampling_rate)` added to `utils/timestamps.py`; all call sites unified; **HybridData timestamp support (item 11) is now unblocked**
    - Items still open not covered by the plan: `analysis/__init__.py` flat export (Minor 9) and CLI consolidation (Minor 10)

10b. **Pipeline redesign: polars/narwhals/no-double-load** *(effort: XL, multi-phase)* — tracked in
    **[`prompts/mrun-cache-implementation.plan.md`](mrun-cache-implementation.plan.md)**.
    Four phases, each independently testable. Package is **usable today** without any phase
    implemented — the double-load is a performance issue only.

    | Phase | Task | Dependency |
    |-------|------|------------|
    | 1+2b-tdms | Custom npTDMS (polars) + `TdmsMagnetData` internal migration | **Must land together** — polars npTDMS breaks pandas-specific internals |
    | 2 | `nw.from_native()` wrap in `PandasMagnetData.getData()` only | Phase 1+2b-tdms |
    | 3 | Pipeline restructure: `select_files()` returns `(path, MagnetRun)` pairs | Phase 2; merge with item 10 `analysis/` refactoring |
    | 2b-pandas | Full internal migration of `PandasMagnetData` internals; rename class | Long-term; after Phase 3; incremental method-by-method |

    Cross-cutting sequencing:
    ```
    Phase 1+2b-tdms  →  Phase 2  →  Phase 2 + item 10 (single pass)  →  Phase 3  →  item 14 Phases D–E  →  Phase 2b-pandas
    ```

11. **`HybridData` timestamp support** *(effort: M, now unblocked)* — tracked in
    **[`prompts/hybriddata-timestamp-plan.md`](hybriddata-timestamp-plan.md)**:
    add `start_timestamp`, `end_timestamp`, `_infer_timestamps()`, `addTime()`,
    `getStartDate()`, `getDuration()` to `HybridData`. Required before `HybridRun` can
    participate as a source in `ComparisonSession` (Phase E).
    **Prerequisite** `analysis/` Phase 6 (`add_time_columns` utility) is now complete.

12. **Outlier deduplication** *(done)* — tracked in
    **[`prompts/outlier-consolidation.plan.md`](outlier-consolidation.plan.md)**.
    Completed: (1) `examples/outliers.py` deleted; (2) `processing/hysteresis.py::remove_outliers`
    thin-delegates to `detect_outliers()` from `hybrid/outliers.py` (~120 lines → ~15 lines);
    (3) `tests/test-anomalies.py` and `tests/test-anomalies-optimized.py` deleted; replaced by
    `tests/test_outliers.py` (142 tests, synthetic data).  Also: `ISOLATION_FOREST` added to
    `OutlierMethod` enum in `hybrid/outliers.py` (sklearn backend, contamination threshold, rolling
    rejected with clear error).  `_VALID_METHODS` in `hysteresis.py` updated to include it.

12b. **`hybrid/` internal refactoring** *(done — all 6 phases)* — tracked in
    **[`prompts/hybrid-subpackage-refactoring.plan.md`](hybrid-subpackage-refactoring.plan.md)**.
    - Phases 1–3: print→logger, outlier dedup, `OUTLIER_DEFAULTS` centralised
    - Phase 4: `OutlierConfig` frozen dataclass (mirrors `DownsampleConfig`); `hybrid_data.py` plot methods use `outlier_config: OutlierConfig | None`; `create_outlier_parser`/`args_to_outlier_config` in `cli_args.py`; canonical moved to `python_magnetrun/outliers.py`
    - Phase 5: `normalize_signal`, `binarize_signal`, `_otsu_threshold` moved to `python_magnetrun/processing/signal.py`; `processing/__init__.py` exports public names; `hybrid/utils.py` and `hybrid/hybrid_run.py` updated
    - Phase 6: cache eviction extracted to `_evict_oldest_cache_entry()`; all-NaN guard in `read_khz_variable`; file-existence guard in `read_rms_variable`; `load_khz_config` raises `FileNotFoundError` instead of returning `None`; `_build_groups` wraps key-discovery in try/except; `saveData` guards against group-key; 866 tests pass

13. **CLI consolidation** *(done)* — tracked in
    **[`prompts/cli-consolidation.plan.md`](cli-consolidation.plan.md)**.
    Single `magnetrun` dispatcher in `python_magnetrun/main.py` with 13 subcommands (info, add,
    plot, select, stats, signature, analysis, processing, hybrid, logparser, fetch, config +
    compare placeholder). `register(subparsers)` pattern on all modules; `input_file` per
    subcommand (eliminating `_normalize_argv`); old entry points kept as deprecated aliases in
    `pyproject.toml` for one release cycle.
    `magnetrun compare` remains pending (blocked on `comparison/cli.py` — Phase F of cross-domain comparison).

14. **Cross-domain comparison — Phases B–G** *(effort: XL)* — depends on item 11.

    | Phase | Task |
    |---|---|
    | B | `SimulationRun` adapter (`python_magnetrun/simulation/`) | Done |
    | C | `BFieldRun` adapter (`python_magnetrun/bfield/`) | Done |
    | D | Extend `*-defs.json` with `simulation`/`bfield` aliases; `KeyMapping` in `comparison/key_mapping.py` (reuses `field_defs.build_crossref()`, no hardcoded dict) |
    | E | `ComparisonSession` (`python_magnetrun/comparison/session.py`) |
    | F | `magnetrun compare` subcommand via `comparison/cli.py::register()` wired into unified dispatcher — **no** standalone `magnetrun-compare` entry point (see `cli-consolidation.plan.md`) |
    | G | `tests/test_comparison.py` |

12. **Editor backup file** *(done)* — `pigbrother-defs.json~` removed; `*.json~` added to `.gitignore`.

15. **Reader/container split** *(done — Stream 3.6)* — `python_magnetrun/readers/` subpackage created:
    - R1: `PupitreReader`, `BProfileReader`, `EnsightReader`, `FeelppReader`, `CsvReader` — factory classmethods delegate to readers
    - R2: `TdmsReader` — `_fromtdms()` uses `TdmsReader.validate()` + `t_offset_for()`; `required_group` on reader
    - R3: `HtsReader` + `DataType.HTS = 4` (`;`-sep, `extracted_units()` parses `"Col [unit]"` headers)
    - R4: `HybridData(MagnetDataBase)` — `Data`/`Type` as abstract properties; `extractData`/`renameData` stubs; `getData` accepts `downsample` kwarg; `field_meta` init bug fixed; `HybridReader` composite
    - R5: `readers/registry.py` (`READERS`, `CONTAINERS`, `detect_type()`); `load_magnetdata()` accepts `fmt=`; 46 new tests in `tests/readers/`; 971 tests pass

    **Full plan:** [`reader-container-refactoring.plan.md`](reader-container-refactoring.plan.md)

16. **Pattern entries in `*-defs.json`** *(effort: S)* — feelpp/paraview data may have hundreds of
    similarly-named columns (`U_0`…`U_239`). A `"match"` regex key in a defs entry allows one
    definition to cover all matching column names via `fullmatch()` in `load_units_from_json()`.
    - H1: two-pass `load_units_from_json()` in `magnetdata_base.py`
    - H2: new `feelpp-defs.json` with `U_\d+`, `T_\d+` pattern entries
    - H3: `FeelppMagnetData.fromfeelpp()` and `SimulationRun.from_feelpp()` default to `feelpp-defs.json`
    - H4: optional `--match` flag on `field add` CLI subcommand

    **Full plan:** Phase H of [`cross-domain-comparison.prompt.md`](cross-domain-comparison.prompt.md)

17. **Hybrid subpackage code quality** *(effort: S→XL — see priority table in [`docs/hybrid_refactoring_notes.md`](../docs/hybrid_refactoring_notes.md))* — 16 findings from cross-module review. Tracked as **Stream 3.9** (S/M/L items) and **Stream 4.6** (XL) in ROADMAP. Items B0.5 and B2.5 are in Phase 2B; items 12 (cross-refs) and 13 (rename `_handle_output`) are in Quick Wins.

    | Priority | Item | Effort | Notes |
    |---|---|---|---|
    | S1 | Hoist `safe_float` to module level in `hybrid/kHz/fepc_reader.py` | S | Defined twice (lines 298, 435) |
    | S2 | Consolidate `_resolve_backend` into `plotting/_utils.py` | S | Exact duplicate in `hybrid/plotting.py` and `plotting/timeseries.py` |
    | M1 | Unify `log_exception` / `format_exception_location` signatures | M | 6 call sites in `hybrid/cli.py` to update; copy in `hybrid/utils.py` to delete |
    | M2 | Standardise `range` schema in `analysis.synchronization` (dict) | M | `compute_lag` uses tuple, `lag_correlation` uses dict — same module |
    | M3 | Deprecate `processing/correlations.py` lag functions via shims | M | Forward to `analysis.synchronization` equivalents |
    | M4 | Share `_apply_cnv_calibration` helper (kHz + trigger) | M | `np.loadtxt` + `np.interp` duplicated in `fepc_reader.py` and `trigger_reader.py` |
    | L1 | Extract `_plot_variable_impl` / `_plot_variables_impl` in `hybrid/plotting.py` | L | Unifies kHz/RMS pipelines; adds `downsample` param to RMS variants |
    | L2 | Extract `_BinaryFileReaderBase` for `RMSFileReader` / `VProcessFileReader` | L | Single `ChannelVariable` dataclass; subclass for encoding + timestamp |
    | XL | Integrate trigger & VProcess into `HybridData` (Stream 4.6) | XL | Depends on L2; `read_trigger_variable`, `plot_trigger_variable`, etc. |
