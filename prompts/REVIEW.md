# Package Review: `python_magnetrun`

Date: 2026-04-08 (updated)

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
├── utils/ / tdms/ / requests/ / configAlims/
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

**5. `MagnetRun.saveData` breaks the abstraction**

`MagnetRun.py` calls `isinstance(self.MagnetData.Data, pd.DataFrame)` directly instead of
delegating to `self.MagnetData.saveData(...)`. Any change to the underlying data type silently
fails for TDMS data.

**6. `TdmsMagnetData.getUnitKey` ignores `self.units`**

In `magnetdata_tdms.py`, `getUnitKey` always calls `self.PigBrotherUnits(group)` and ignores what
`Units()` loaded from the defs file. This violates the resolution order documented in `Units()` and
is an LSP violation.

**7. Incompatible `Data` attribute type across subclasses**

`MagnetDataBase.__init__` (`magnetdata_base.py:74`) declares `self.Data: pd.DataFrame | dict`
— a union type that acknowledges the divergence rather than enforcing a contract:

- `PandasMagnetData.Data` → `pd.DataFrame`
- `TdmsMagnetData.Data` → `dict[str, pd.DataFrame]` keyed by group name

Any caller that touches `.Data` directly must either branch on `isinstance` (as `MagnetRun.saveData`
does at `MagnetRun.py:205`) or `assert isinstance(...)` (as `TdmsMagnetData.get_time_range` does at
`magnetdata_tdms.py:242`). This defeats the purpose of the ABC. The fix is to remove direct `.Data`
access from all callers outside the subclasses and route everything through `getData()`, which already
returns `pd.DataFrame` uniformly — then `Data` can become a private implementation detail.

**Note:** `MagnetRun.saveData` (issue #5) is a direct symptom of this problem.

**8. Two conflicting Protocol definitions for the `MagnetRun`/`HybridRun` interface**

`DataProvider` (in `hybrid/hybrid_run.py`) and `DataLoader` (in `hybrid/data_protocol.py`) describe
the same concept with slightly different signatures. Pick one and have both `MagnetRun` and
`HybridRun` declare it.

---

### Minor

**8. Hardcoded developer path as CLI default**

`cli_args.py` line ~250 sets:
```python
default="/home/LNCMI-G/christophe.trophime/LNCMIG-Data/srv-data-install"
```
This silently uses a non-existent path on any other machine. Use `None` or an environment variable.

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
| `Units`/`getUnitKey` consistency in TDMS | Needs fix |
| `Data` attribute type divergence (`DataFrame` vs `dict`) | Needs fix |
| CLI consolidation | Needs work |
| `saveData` abstraction in `MagnetRun` | Needs fix (symptom of `Data` divergence) |
| Timestamp convention (`timestamp` column UTC vs local) | Needs fix — plan ready |

The core abstractions are well-conceived — the ABC, the defs system, and `HousingConfig` are solid
foundations. The housing config consolidation and the `MagnetData` shim replacement are now complete.
The remaining weaknesses are TDMS unit lookup inconsistency, Protocol duplication in the hybrid layer
(scheduled as Phase A0 of the cross-domain comparison plan), CLI fragmentation, and timestamp
convention inconsistency between `PandasMagnetData` (local) and `TdmsMagnetData` (UTC).

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

1. **Timestamp convention** *(effort: M)* — `PandasMagnetData.timestamp` stores naive **local** time;
   `TdmsMagnetData.timestamp` stores naive **UTC** by default. Both `start_timestamp` /
   `end_timestamp` are already naive UTC. Convention must be unified: all `timestamp` columns
   store naive UTC; local-time conversion happens only at display/filter boundaries
   (`plotData`, `extractTimeData`). `addTime()` must also become eager (computes `t` and
   `timestamp` for all TDMS groups at once, removing scattered lazy guards). See full plan in
   **[`prompts/timestamp-utc-refactoring.plan.md`](timestamp-utc-refactoring.plan.md)**.

   *Breakdown*: `magnetdata_pandas.py` + `magnetdata_tdms.py` + `magnetdata_base.py` signatures
   (~2 h); update `commands/select.py` caller (~30 min); rewrite `TestExtractTimeData` + add
   TDMS timestamp tests (~1 h); smoke validation (~30 min).

   **Must be done before item #2** (Data divergence) to minimise churn: the timestamp plan
   adds new internal `self.Data[group]` accesses inside TDMS subclass methods; doing it first
   means zero extra updates when `Data` is made private.

   **Caller-side change in `commands/select.py`**: `extractTimeData` timerange format changes
   from `"HH:MM:SS;HH:MM:SS"` to `"YYYY-MM-DD HH:MM:SS;YYYY-MM-DD HH:MM:SS"` (local
   datetime strings). `select.py` line 176 must be updated in the same commit.

   **`HybridData` not in scope** for this plan — needs its own follow-up (add
   `start_timestamp`, `end_timestamp`, `addTime()`).

2. **`Data` attribute type divergence** *(effort: L)* — `MagnetDataBase.Data` is typed
   `pd.DataFrame | dict` ([magnetdata_base.py:74](python_magnetrun/magnetdata_base.py#L74));
   callers branch on `isinstance` instead of using `getData()`. Fix: make `Data` private in
   subclasses; route all external access through `getData()`. Resolving this also unblocks
   item #3. *The grep for direct `.Data` access outside the two subclasses will determine the
   actual scope — likely a dozen call sites across `MagnetRun.py`, `commands/`, `analysis/`,
   and `examples/`.*

3. **`TdmsMagnetData.getUnitKey`** *(effort: S)* — fix to return `self.units[key]` when
   populated, falling back to `PigBrotherUnits` only as a last resort; currently bypasses
   `self.units` entirely
   ([magnetdata_tdms.py:175-189](python_magnetrun/magnetdata_tdms.py#L175-L189)).

4. **`MagnetRun.saveData`** *(effort: S, unblocked by #2)* — delegate to
   `self.MagnetData.saveData(...)` instead of the inline
   `isinstance(self.MagnetData.Data, pd.DataFrame)` check
   ([MagnetRun.py:202-209](python_magnetrun/MagnetRun.py#L202-L209)).

5. **Protocol duplication** *(effort: M)* — unify `DataProvider` (`hybrid/hybrid_run.py`) and
   `DataLoader` (`hybrid/data_protocol.py`); annotate `MagnetRun` and `HybridRun` to declare
   the chosen one. **Tracked as Phase A0 in
   `prompts/cross-domain-comparison.prompt.md`** — will be resolved as part of the
   `DataLoader` protocol extension work.

6. **Hardcoded default path** *(effort: S)* — replace `cli_args.py` line 249 with `None` or
   an env-var lookup.

7. **`tsdownsample`** *(effort: S)* — add to `pyproject.toml` as a `hybrid` extras dependency.

8. **Editor backup file** *(effort: S)* — remove `pigbrother-defs.json~` and add `*.json~` to
   `.gitignore`.
