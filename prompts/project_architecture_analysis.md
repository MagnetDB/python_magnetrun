---
name: python_magnetrun architecture analysis
description: Current state of the package, what exists vs what's missing, and the architectural gap for cross-domain comparison goals
type: project
---

## Goals (stated by user, 2026-04-08)

1. Plot comparisons for operational data from different systems (pupitre, pigbrother, hybrid/FEPC)
2. Plot comparisons of operational data vs. simulation data (feelpp, ensight, magnettools)
3. Compute metrics for operational data comparisons
4. Same features for magnetic field measurements (bprofile data)

---

## Data sources

| Domain | Formats | Status |
|---|---|---|
| Operational | pupitre (.txt), pigbrother (.tdms), hybrid/FEPC (binary kHz/RMS/trigger) | Fully implemented |
| Simulation | ensight, feelpp (CSV-like), magnettools | ensight+feelpp classes exist but disconnected; magnettools missing |
| B-field | bprofile | Class exists but standalone, not integrated |

---

## What exists and works (as of 2026-04-17)

- `MagnetDataBase` ABC in `magnetdata_base.py` with `getData()`, `getKeys()`, `Units()`, `extractData()`, `get_time_range()` stub
- `PandasMagnetData` (pupitre/CSV), `TdmsMagnetData` (TDMS/pigbrother) — `get_time_range()` fully implemented on both
- `EnsightMagnetData`, `FeelppMagnetData`, `BProfileMagnetData` — thin subclasses in `magnetdata_pandas.py`
- `magnetdata.py` — pure factory: `load_magnetdata()` + `_fromtdms()`; `MagnetData` shim class removed
- `HybridData`/`HybridRun` (FEPC) — `get_time_range()` implemented (`hybrid_run.py:837`)
- `DataLoader` protocol (PEP 544, `hybrid/data_protocol.py`) — runtime-checkable; `get_time_range()` added to protocol (`data_protocol.py:177`); implemented by `MagnetRun` and `HybridRun`
- `runlogs/` submodule — `LogParser` (pigbrother/DAQmx), `CirrusRunlogLoader` + `discover_pupitre_runlogs` (pupitre Cirrus logs); replaces deleted `tdms/log_parser.py`
- `vprocess-defs.json` — initial virtual-process definitions
- `analysis/` module: time synchronization (DTW, cross-correlation), distance metrics (MAE, MAPE, Pearson, DTW), comparison plotting, downsampling
- `processing/` module: smoothing, peak detection, filtering, hysteresis, change-point detection, outlier detection
- 6 CLI entry points: `python-magnetrun`, `magnetrun-analysis`, `hybrid-magnetrun`, `srvdata-to-magnetrun`, `magnetrun-alimconfig`, `magnetrun-pigbrother-logparser`

---

## Completed refactors (as of 2026-04-17)

- Phase 1–3 ABC hierarchy refactor: `MagnetDataBase`, `PandasMagnetData`, `TdmsMagnetData`, `EnsightMagnetData`, `BProfileMagnetData`, `FeelppMagnetData` ✅
- Zero `if self.Type` conditionals in `magnetdata*.py` ✅
- `load_magnetdata()` standalone factory — pure factory module; `MagnetData` shim removed ✅
- `get_time_range()` — concrete on `PandasMagnetData`, `TdmsMagnetData`, `HybridRun` ✅
- `HousingConfig` consolidation — single source of truth; `prepareData_legacy` removed ✅
- `MagnetRun.saveData` delegates to `self.MagnetData.saveData()` ✅
- `DataProvider` removed from `hybrid_run.py`; `DataLoader` is the single protocol ✅
- Timestamp convention: `PandasMagnetData` + `TdmsMagnetData` both store naive UTC ✅
- `get_time_range()` added to `DataLoader` protocol (Phase A1 partial) ✅
- `tdms/log_parser.py` moved to `runlogs/pigbrother.py`; `utils/convert.py` deleted ✅
- New plan files: `downsampling-refactoring.plan.md`, `plotting-refactoring.plan.md`, `hybriddata-timestamp-plan.md` ✅

---

## Remaining work — Phase 4 `.Type` cleanup

`mdata.Type` integer checks (stale, still work but should use `isinstance`):

| File | Lines |
|------|-------|
| `commands/stats.py` | 65, 73, 183, 187, 309 |
| `commands/select.py` | 149, 176, 180, 207, 219 |
| `cli.py` | 140, 144 |
| `hybrid/data_protocol.py` | 214 |
| `processing/cli.py` | 193 |

---

## Architectural gap

`getDomain()` is not yet in the `DataLoader` protocol nor implemented on `MagnetRun`/`HybridRun`.
`MagnetRun.get_time_range()` delegation wrapper is missing (protocol says it exists, class doesn't implement it).
Simulation (`EnsightMagnetData`, `FeelppMagnetData`) and B-field (`BProfileMagnetData`) classes still don't implement `DataLoader`.
No channel-name normalization across domains (e.g. "IH" vs "I_H1" vs "Icoil_helix").
No unified `ComparisonSession` entry point.
`HybridData` timestamp support still pending (tracked in `prompts/hybriddata-timestamp-plan.md`).

---

## Suggested architecture evolution (additive, no rewrite)

```
DataLoader protocol (extend to ALL sources)
  getData(key) → pd.Series
  getKeys() → list[str]
  getDomain() → "operational" | "simulation" | "bfield"
  getTimeRange() → (t0, t1)
       ↑              ↑               ↑
  MagnetRun     SimulationRun     BFieldRun      ← new thin adapters
  HybridRun     (wraps Ensight,   (wraps
                FeelppMagnet,     BProfile)
                magnettools)
                      ↓
       ComparisonSession(sources, key_map)
          ├── plot_comparison()
          ├── compute_metrics()
          └── report()
```

**Key additions needed:**
1. `getDomain()` on `DataLoader` protocol + `MagnetRun` + `HybridRun` (Phase A1 remainder + A2)
2. `MagnetRun.get_time_range()` delegation wrapper (Phase A2)
3. Protocol compliance tests `tests/test_protocol.py` (Phase A3)
4. `SimulationRun` adapter — wraps `EnsightMagnetData`, `FeelppMagnetData`, and a new `MagnetToolsMagnetData`; implements `DataLoader` (Phase B)
5. `BFieldRun` adapter — wraps `BProfileMagnetData`; implements `DataLoader` (Phase C)
6. `KeyMapping` config — per-domain channel name normalization (Phase D)
7. `ComparisonSession` — accepts list of `DataLoader`, handles time resampling to common axis, calls existing `metrics.py` and `analysis/plotting.py` (Phase E)
8. `magnettools` loader — new reader for the missing simulation format
9. `magnetrun-compare` CLI (Phase F)

---

## Cross-domain comparison — prerequisite status

| Phase | Task | Status |
|---|---|---|
| A0 | Delete `DataProvider` from `hybrid_run.py` | Done ✅ |
| A1 | Add `get_time_range()` to `DataLoader` protocol | Done ✅ (`data_protocol.py:177`) |
| A1 | Add `getDomain()` to `DataLoader` protocol | **Todo** |
| A2 | `HybridRun.get_time_range()` | Done ✅ (`hybrid_run.py:837`) |
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

Immediate next step: complete Phase A1–A3 (~2 h total).

---

## Planned refactors (tracked in prompts/)

- **Downsampling refactoring** (`prompts/downsampling-refactoring.plan.md`, effort M) — extract `downsample_data()` into `utils/downsampling.py`, introduce `DownsampleConfig`, add to `PandasMagnetData`/`TdmsMagnetData`, reconcile `analysis/processing.py` percentage model, add `tsdownsample` to `hybrid` extras.
- **Plotting refactoring** (`prompts/plotting-refactoring.plan.md`, effort L) — `python_magnetrun/plotting/` subpackage, `PlottingBackend` protocol, `MatplotlibBackend`, `PlotlyBackend`, `AnnotationManager`, JS-frontend path. Depends on downsampling Steps 1–2.
- **HybridData timestamp support** (`prompts/hybriddata-timestamp-plan.md`) — add `start_timestamp`, `end_timestamp`, `_infer_timestamps()`, `addTime()`, `getStartDate()`, `getDuration()` to `HybridData`.

---

## Related prompts in repo

- `prompts/magnetdata_refactoring.md` — ABC hierarchy refactor: Phase 1–3 done, Phase 4 remaining
- `prompts/cross-domain-comparison.prompt.md` — full cross-domain implementation plan (Phase A–G)
- `prompts/REVIEW.md` — package-wide review, issues 1–8, all critical/significant resolved
- `prompts/BREAKING_CHANGES.md` — breaking changes log
