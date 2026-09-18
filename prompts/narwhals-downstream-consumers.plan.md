# Narwhals Downstream-Consumer Migration Plan

Companion to [polars-magnetdata-pupitre.plan.md](polars-magnetdata-pupitre.plan.md)
(whose Rollout step 2 is blocked on this) and
[mrun-cache-implementation.plan.md](mrun-cache-implementation.plan.md) Phase 2
(Implementation Order step 10, "Migrate downstream consumers to narwhals
API" — originally scoped for TDMS only; this doc covers the same problem,
now confirmed to also block pupitre).

## Why this exists

`PolarsMagnetData.getData()` returns a narwhals-wrapped frame (per
`mrun-cache-implementation.plan.md`'s Phase 2 design). Code that was
written assuming `getData()` returns something pandas-shaped — `.iloc`,
`.loc`, `.mode()`, `.values`, pandas' `.plot()` — breaks the moment a
pupitre file is backed by `PolarsMagnetData` instead of `PandasMagnetData`,
regardless of what `PolarsMagnetData` itself implements. This is a
different problem from Phase A/B (which are about the container class) —
it's about everything downstream of `getData()`.

## Audit — first 3 breakages found (before the full Phase 0 pass below)

1. **`processing/stats.py::stats()`** — backs the default `magnetrun stats`
   CLI path (via `commands/stats.py:62`). Does
   `Data.getData([f])[f].mode().iloc[0]` — pandas-Series-only chaining
   directly on the `getData()` result.
2. **`MagnetRun.getDataFrame()`** — its docstring and type-hint promise
   `pd.DataFrame` for `DataType.PUPITRE`; a narwhals frame doesn't satisfy
   that contract, silently or otherwise, for any caller relying on the
   documented return type.
3. **`processing/plateaux.py`** — `_t = Data.getData(["t"])["t"]` then
   `.iloc[p[0]]`, explicitly gated on `if Data.Type == DataType.PUPITRE:`
   (feeds `commands/stats.py --plateau`). This one is pupitre-only code —
   guaranteed to break, not just at risk.

**One confirmed non-issue:** `utils/txt2csv.py` constructs
`PandasMagnetData` directly, bypassing `load_magnetdata()`/
`MagnetRun.fromtxt()` entirely — unaffected by the pupitre dispatch switch.

## Phase 0 audit — ✅ complete

All 10 originally-suspected files traced line-by-line. Result: **broader
than the placeholder table suggested** — 7 confirmed (not 3), plus one
newly-discovered call site that isn't just the already-known
`extractTimeData` usage.

### Confirmed risk (7)

| File | What breaks |
|---|---|
| `commands/plot.py` | `df.copy()` and `df["t"] = df["t"] + delta_t` (item assignment) in the multi-file overlay path behind `magnetrun plot`. Neither works on a narwhals frame. |
| `processing/cli.py` | `pd.DataFrame(mdata.getData(...))` (wrapping a narwhals frame in the pandas constructor), plus `.getData("Date").iloc[0]` / `.getData("Time").iloc[0]` — pupitre-specific columns, guaranteed reachable. |
| `commands/select.py` | A **second** issue beyond the known `extractTimeData` calls: `data["t"].isin(times)`, `.copy()`, `df["timestamp"] = ...` (item assignment) — explicitly gated `if mdata.Type == DataType.PUPITRE:`. Guaranteed break, same shape as `processing/plateaux.py`. |
| `analysis/loaders.py` | Same `pd.DataFrame(mdata.getData(desired))` wrapping pattern as `processing/cli.py`. |
| `waterflow_pipeline.py` | `getMData().getData()` fed into `extract_hydraulic_data()`; internals of that helper not traced fully, but same risk shape. |
| `processing/filters.py` | `mrun.getData()` (a real method — confirmed at `MagnetRun.py:477`, distinct from `getDataFrame()`) then `.mean()`, `.var()`, `.rolling().median().bfill().ffill()`, `.assign()`. |
| `panels/panel-mrecord.py`, `panels/panel-mrecord-vs-time.py` | `.set_index("t")` on the `getData()` result. Read like standalone demo/example scripts rather than a served dashboard, but still real code. |

### Not actually pupitre risks (2)

- **`analysis/processing.py`** — the flagged call is on `hrun` (`HybridRun`), unreachable via pupitre. Relevant to hybrid's own eventual narwhals migration, not this one.
- **`processing/correlations.py::pearson()`** — gated behind `isinstance(Data, pd.DataFrame)`, which is `False` for any real `MagnetDataBase` subclass under the documented calling convention. Looks like already-dead/unreachable code, independent of polars.

### Murky — needs a closer look before Phase 1 touches it (1)

- **`processing/correlations.py::tlcc()`/`crosscorr()`** — does call `.getData()` and chains `.shift()`/`.corr()`, but the pre-existing pandas-side semantics already look questionable (single-column DataFrames passed where Series behavior seems assumed). Likely low real-world usage; worth a deeper look during Phase 1, not blocking the rest.

**Combined with the 3 originally found, the full confirmed-risk list for
Phase 1 is 10 files**: `processing/stats.py`, `MagnetRun.getDataFrame()`,
`processing/plateaux.py`, `commands/plot.py`, `processing/cli.py`,
`commands/select.py`, `analysis/loaders.py`, `waterflow_pipeline.py`,
`processing/filters.py`, `panels/panel-mrecord*.py`.

## Strategy: two options

### Option A — Full narwhals-native rewrite

Rewrite every consumer to use only narwhals' common API
(`.filter()`, `.select()`, narwhals-supported aggregations, etc.) instead
of pandas-specific idioms. Gets cross-backend correctness everywhere —
the original ambition of Phase 2 in the master plan. Effort: large,
comparable to or exceeding Phase A itself, spread across ~10-15 files.

### Option B — Boundary conversion (recommended)

At each risky consumption point, call `.to_pandas()` on the narwhals
result before any further processing — confirmed to work cleanly:

```python
import narwhals as nw
df = mdata.getData(key)          # narwhals.DataFrame
df = df.to_pandas()              # plain pandas.DataFrame — existing code works unchanged
```

(`narwhals.Series.to_pandas()` also confirmed to work, for the `Data.getData([f])[f]`-style single-column access pattern.)

**Why recommended:** pupitre's actual performance win — the ~3× faster,
lower-memory *load* — is already captured entirely inside
`PolarsMagnetData`'s ETL chain (Phase A), before any of this consumer code
runs. Late-stage consumers (CLI stats display, plateau detection, plotting)
converting to pandas at the point of use gives up zero-copy for that one
DataFrame, which none of them need — they're not hot loops, they run once
per CLI invocation. This is surgical: existing pandas-idiom code is
untouched, only the boundary right after `getData()`/`getDataFrame()`
changes. Option A remains available later if cross-backend consumers
become genuinely worth it (e.g. once TDMS also emits narwhals frames and
consumers plausibly want to skip a conversion for both backends).

## Phased plan

1. **Complete the audit** — ✅ **Done**, see Phase 0 above (10 confirmed
   files, 2 false positives, 1 murky/deferred).
2. **Apply boundary conversions** (Option B) at each of the 10 confirmed
   call sites. `MagnetRun.getDataFrame()` first (converting before
   returning restores its documented `pd.DataFrame` contract for all
   types uniformly, and several other consumers may end up calling it
   rather than the container's `getData()` directly).
3. **Regression test per fixed call site** — ✅ **Done**, smoke-tested
   directly against a real `PolarsMagnetData` instance for every fixed
   site (not all added as permanent pytest cases — see Testing below).
4. **Re-run `polars-magnetdata-pupitre.plan.md` Rollout step 2** — flip
   `load_magnetdata()`'s `.txt` branch to `PolarsMagnetData`, now that its
   consumers are shielded. **Ready** — see next section.

*Deferred, not blocking:* a closer look at `processing/correlations.py::tlcc()`/
`crosscorr()` (murky, likely low usage) — fix if/when it turns out to matter.

## Phase 1 — ✅ complete

New shared helper: `utils/narwhals_compat.py::to_pandas(data)` — converts a
narwhals **or raw Polars** frame/series to pandas; passes through anything
else unchanged. The "raw Polars" half matters because `extractData()`/
`extractTimeData()` return unwrapped `pl.DataFrame` (they don't go through
the narwhals boundary the way `getData()` does) — a gap found only once
implementation started, not visible during the Phase 0 audit. Detection
uses `narwhals.dependencies.is_polars_dataframe()`/`is_polars_series()`,
so the module works whether or not Polars is installed. `narwhals` promoted
from the `polars` optional-dependency group to a base dependency in
`pyproject.toml`, since core CLI modules now import it unconditionally.

Applied at all 10 confirmed sites, each smoke-tested end-to-end against a
real `PolarsMagnetData` instance:

| File | Fix |
|---|---|
| `MagnetRun.getDataFrame()` | wrap both the pupitre and TDMS branches |
| `processing/stats.py::stats()` | wrap before `.mode().iloc[0]` |
| `processing/plateaux.py` | 3 sites: `nplateaus()`, `plateaus()`, the `_t.iloc[]` plateau loop |
| `commands/plot.py` | `_get_df_with_time()`'s pupitre branch, the pair-plot branch |
| `processing/cli.py` | the two `.getData("Date"/"Time").iloc[0]` sites (the TDMS-only `addtime()` helper was a false positive — confirmed not pupitre-reachable) |
| `commands/select.py` | 4 sites (`output_keys`, `extract_pairkeys`, `output_timerange`, `output_time`) — a **5th issue found here beyond the original audit**: `.to_csv()` calls on `extractData()`/`extractTimeData()` results, which return raw Polars (no `.to_csv()` method at all) |
| `analysis/loaders.py` | `load_files_data()`, replacing an unsafe `pd.DataFrame(getData(...))` wrap |
| `waterflow_pipeline.py` | `compute_waterflow_from_run()` |
| `panels/panel-mrecord.py`, `panels/panel-mrecord-vs-time.py` | both scripts' `getData()` call |

**Two sites reclassified as pre-existing bugs, unrelated to polars — left
untouched:**
- `processing/filters.py`: `mrun.getData()` with no arguments already
  raises `KeyError` today, for `PandasMagnetData` too (`MagnetRun.getData()`
  defaults `key=""`, and `""` is never a real column name). Confirmed by
  reproducing the error against a real `PandasMagnetData` fixture. Dead
  code, not a narwhals regression.
- `commands/select.py::convert_to_csv()`: calls `mdata.to_csv(...)`
  directly on the `MagnetDataBase` object — no such method exists on
  `PandasMagnetData` either. Same category.

**Also confirmed harmless, no fix needed:** `processing/cli.py`'s "smooth"
command's pupitre branch uses `extractData()` (raw Polars) with `.head()`,
`[key].mean()`, `[key].to_numpy()` — all of which already work identically
on Polars and pandas, so nothing broke there despite not going through
`to_pandas()`.

### Testing

New `tests/test_narwhals_compat.py` (6 tests) covers the helper itself —
narwhals DataFrame/Series, raw Polars DataFrame/Series, and pass-through
for plain pandas / non-frame values. The 10 call-site fixes were verified
by direct smoke tests against real `PolarsMagnetData` instances (not all
committed as permanent pytest cases) rather than added as new regression
tests in each of the 10 modules — most of those modules have no existing
test coverage at all (only `waterflow_pipeline.py` and `analysis/loaders.py`
had pre-existing suites, both re-run clean). Full suite: 1248 passed, 19
skipped, zero regressions.

## Open question

Should `MagnetRun.getDataFrame()`'s boundary conversion apply uniformly
(always return `pd.DataFrame`, converting away from narwhals even though
Phase 2 elsewhere wants narwhals as the boundary), or should it return the
narwhals frame and let *callers* convert? Recommendation: convert inside
`getDataFrame()` itself — its docstring already promises `pd.DataFrame`,
and changing that contract package-wide is exactly the "migrate downstream
consumers" scope Option A would represent, which this plan is deliberately
deferring.
