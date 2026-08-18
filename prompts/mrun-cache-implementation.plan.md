# MagnetRun Pipeline — Redesign Plan

## Problem

`MagnetRun` objects are expensive to construct: each call to `MagnetRun.fromtxt()` or
`MagnetRun.fromtdms()` parses a file from disk. In the current flow the same file
is loaded **twice** for every path processed:

1. **`extract_data()`** — called by `select_files()` to obtain the time-range
   (`start_ftimestamp`, `end_ftimestamp`) so that files can be filtered.
2. **`load_df()`** — called by `load_data()` to actually read channel data into a
   `DataFrame`.

When `select_files()` is run before `load_data()` (the normal pipeline), every file
that passes the filter is loaded a second time from scratch.

Additional constraints:
- Files can be very large (especially hybrid TDMS data).
- Memory usage must be kept as low as possible.
- A full migration from pandas to narwhals internally is planned long-term.

---

## Context: Existing Caching in the Hybrid Subpackage

The hybrid subpackage already has a sophisticated, memory-aware caching strategy that
this plan must not duplicate or conflict with:

- **`HybridRun._cache`** — byte-limited LRU cache (`_cache_max_size_bytes = 1 GB`)
  with eviction by `loaded_at` timestamp.
- **`LazyKHzLoader`** — numpy `memmap`-backed lazy loader; large kHz arrays stay on
  disk until actually accessed, so loading a `MagnetRun` for timestamp extraction is
  cheap.
- **`HybridData._khz_configs` / `_rms_readers`** — dictionary caches for config files
  and discovery results to avoid re-reads.

The double-load concern is therefore most acute for `.txt` files, where the full file
must be parsed to extract timestamps.

---

## Solution: Five-Phase Redesign

### Phase 1+2b-tdms — Custom npTDMS (polars) + `TdmsMagnetData` internals migration *(must land together)*

These two tasks are coupled: custom npTDMS returns polars DataFrames, which
`TdmsMagnetData` stores in `self.Data: dict[str, pd.DataFrame]`. If only Phase 1
lands, all internal methods in `TdmsMagnetData` that use pandas-specific patterns
break immediately:

| Broken pattern | Fix |
|----------------|-----|
| `self.Data[group].eval(formula, inplace=True)` | Expression-based transform, reassign |
| `self.Data[gname]["t"] = values` | `self.Data[gname] = self.Data[gname].with_columns(...)` |
| `self.Data[group][channel].loc[...]` | polars slice / filter |
| `pd.Timestamp(...) + pd.to_timedelta(...)` | polars / narwhals equivalents |
| `self.Data[group].drop(columns=[...], inplace=True)` | `.drop([...])`, reassign |

**Substeps:**
1. Implement polars output in the custom npTDMS fork.
2. Validate custom npTDMS against existing TDMS test fixtures.
3. Change `TdmsMagnetData.Data` type to `dict[str, pl.DataFrame]`.
4. Rewrite all internal methods in `TdmsMagnetData` to use polars / narwhals API,
   method by method, with tests after each step.
5. Wrap `TdmsMagnetData.getData()` return value with `nw.from_native()`.

### Phase 2 — Narwhals boundary at `getData()` for `PandasMagnetData`

With `TdmsMagnetData.getData()` already returning narwhals frames (Phase 1+2b-tdms),
add the equivalent wrap to `PandasMagnetData.getData()`:

```python
# PandasMagnetData
def getData(self, ...) -> nw.DataFrame:
    df = self._getData_internal(...)   # still pd.DataFrame internally
    return nw.from_native(df)
```

Update `MagnetDataBase.getData()` return type annotation to `nw.DataFrame`.

- `self.Data` in `PandasMagnetData` stays `pd.DataFrame` — **zero internal rewrite**.
- All external callers (`analysis/`, `waterflow_pipeline.py`, plotting,
  `MagnetRun.getDataFrame()`) now receive narwhals frames from both subclasses.
- No `if backend == "pandas"` branches anywhere downstream.

### Phase 3 — Pipeline Restructure (eliminate double-load)

Restructure `select_files()` and `load_data()` so that objects loaded during
selection are passed directly to the load stage.

**Current flow:**
```
select_files()  →  [filepath, ...]
load_data()     →  loads each filepath again from scratch
```

**New flow:**
```
select_files()  →  [(filepath, MagnetRun), ...]   # rejected files discarded immediately
load_data()     →  receives pre-loaded MagnetRun, skips re-load
```

Key properties:
- **No persistent module-level cache** — no `lru_cache`, no unbounded memory growth.
- **Rejected files are freed immediately** — only files that pass the timestamp filter
  stay in memory.
- **Peak memory** = one `MagnetRun` at a time during selection, then only the
  filtered subset during loading.
- For TDMS files the `LazyKHzLoader` memmap keeps heavy channel data on disk; the
  `MagnetRun` held in memory is lightweight until channel data is explicitly accessed.

API change: `select_files()` return type widens from `list[str]` to
`list[tuple[str, MagnetRun]]`. `load_df()` gains an optional `mrun` parameter; when
provided it skips the file load.

### Phase 2b-pandas — Full internal migration of `PandasMagnetData` *(long-term)*

Replace `self.Data: pd.DataFrame` with narwhals throughout `PandasMagnetData`.
Same pattern rewrites as Phase 1+2b-tdms. Done incrementally, method by method.
When complete, pandas is no longer a direct dependency and `PandasMagnetData` can
be renamed to `MagnetData`.

---

## Implementation Order

| Step | Phase | Action |
|------|-------|--------|
| 1 | 1+2b-tdms | Implement polars output in custom npTDMS fork |
| 2 | 1+2b-tdms | Validate custom npTDMS against existing TDMS test fixtures |
| 3 | 1+2b-tdms | Add `narwhals` dependency to `pyproject.toml`; replace `nptdms` with custom fork |
| 4 | 1+2b-tdms | Change `TdmsMagnetData.Data` type to `dict[str, pl.DataFrame]` |
| 5 | 1+2b-tdms | Rewrite `TdmsMagnetData` internal methods to polars/narwhals, method by method |
| 6 | 1+2b-tdms | Wrap `TdmsMagnetData.getData()` return with `nw.from_native()` |
| 7 | 2 | Wrap `PandasMagnetData.getData()` return with `nw.from_native()` |
| 8 | 2 | Update `MagnetDataBase.getData()` return type annotation to `nw.DataFrame` |
| 9 | 2 | Update `MagnetRun.getDataFrame()` return type annotation to `nw.DataFrame` |
| 10 | 2 | Migrate downstream consumers to narwhals API (`analysis/`, `waterflow_pipeline.py`, plotting) |
| 11 | 3 | Change `select_files()` to return `list[tuple[str, MagnetRun]]` |
| 12 | 3 | Add optional `mrun` parameter to `load_df()` |
| 13 | 3 | Update `load_data()` to pass pre-loaded objects through |
| 14 | 3 | Update tests |
| 15 | 2b-pandas | Migrate `PandasMagnetData` internals to narwhals API, method by method |
| 16 | 2b-pandas | Drop pandas as a direct dependency; rename `PandasMagnetData` → `MagnetData` |

Each phase boundary is independently testable; the pipeline stays functional between phases.

---

## Affected Files

| File | Phase | Change |
|------|-------|--------|
| custom `npTDMS` fork | 1+2b-tdms | Polars DataFrame output |
| `python_magnetrun/magnetdata_tdms.py` | 1+2b-tdms | Internal migration to polars/narwhals; `getData()` wraps with `nw.from_native()` |
| `python_magnetrun/magnetdata_pandas.py` | 2 / 2b-pandas | Phase 2: `getData()` wraps with `nw.from_native()`; Phase 2b: full internal migration + rename |
| `python_magnetrun/magnetdata_base.py` | 2 | Update `getData()` return type annotation to `nw.DataFrame` |
| `python_magnetrun/MagnetRun.py` | 2 | `getDataFrame()` return type → `nw.DataFrame` |
| `python_magnetrun/analysis/loaders.py` | 2+3 | Narwhals API; pipeline restructure |
| `python_magnetrun/waterflow_pipeline.py` | 2 | Narwhals API |
| `tests/analysis/test_loaders.py` | 3 | Updated for new `select_files()` return type |
| `pyproject.toml` | 1+2b-tdms | Add `narwhals`; replace `nptdms` with custom fork |

---

## Interactions with Other Plans

### Phase 3 vs `analysis-subpackage-refactoring.plan.md`
Both touch `loaders.py`. Running them independently risks merge conflicts.
**Resolution**: Phase 3 should either land first (analysis refactoring absorbs the
new `select_files()` return type) or be merged into a single pass with the analysis
refactoring. Do not work on both in parallel.

### Phase 2 + Cross-domain comparison (REVIEW.md item 14 Phases D–E)
`ComparisonSession` and adapters will consume `MagnetRun` objects. The adapter
design (Phase D) should assume narwhals frames from the start to avoid retrofitting.
Low risk — Phases D–E are not yet started.

### Recommended sequencing across all plans
```
Phase 1+2b-tdms (custom npTDMS + TdmsMagnetData internals)  ← one coordinated pass
    ↓
Phase 2 (nw.from_native() in PandasMagnetData.getData())
    ↓
Phase 2 + analysis/ refactoring                              ← merge into one pass
    ↓
Phase 3 (pipeline restructure, inside analysis/ refactoring)
    ↓
Cross-domain comparison Phases D–E                           ← narwhals-aware adapters
    ↓
Phase 2b-pandas (PandasMagnetData internals, long-term)
```

---

## Limitations & Notes

- `MagnetRun` objects are assumed **immutable** after construction. If any caller
  mutates the narwhals frame the mutation will be visible to all pipeline stages.
  Callers that need mutation should work on a copy.
- Long-running notebooks that re-use the same filenames with different content must
  re-instantiate the pipeline (no persistent cache to clear).
- Until Phase 2b-pandas is complete, pandas remains a transitive dependency (used
  internally by `PandasMagnetData`). External callers are insulated from it via the
  narwhals API after Phase 2.
- The package is **usable today** without any of these phases implemented. The
  double-load is a performance issue only; no correctness is affected.
