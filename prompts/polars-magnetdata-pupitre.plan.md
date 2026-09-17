# PolarsMagnetData — Pupitre Polars Container Plan

Companion to [tdms-pupitre-polars-findings.md](tdms-pupitre-polars-findings.md)
(benchmark results + cost estimate) and
[mrun-cache-implementation.plan.md](mrun-cache-implementation.plan.md) (the
overall ROADMAP 4.3 design, which this plan slots into as "Phase 1b-pupitre").

## Why a new class instead of rewriting `PandasMagnetData`

`PandasMagnetData` (1687 lines) is **shared**: it's the base class for
`EnsightMagnetData`, `BProfileMagnetData`, `FeelppMagnetData`, and is used
directly for HTS. Rewriting it in place to use polars — the
`tdms-pupitre-polars-findings.md` "full path" option — drags all four other
formats along whether or not that's wanted, and was already flagged there as
XL/long-term.

A sibling `PolarsMagnetData(MagnetDataBase)` avoids that entirely: it only
serves pupitre, `PandasMagnetData` and its subclasses are untouched, and it
composes cleanly with the narwhals boundary already planned for Phase 2 of
`mrun-cache-implementation.plan.md` — `PolarsMagnetData.getData()` wraps its
output with `nw.from_native()`, exactly like `TdmsMagnetData` will after
Phase 1+2b-tdms and `PandasMagnetData` will after Phase 2. Downstream code
stays backend-agnostic regardless of which of the three it's holding.

**Trade-off, stated plainly (per the earlier findings doc):** this is not
*less* implementation work than the shared-class rewrite — `PandasMagnetData`
has ~35 substantial methods, many pandas-idiom-heavy, and most would need a
polars-native equivalent either way. The saving is risk isolation (Ensight/
BProfile/Feelpp/HTS can't regress) and independent scheduling, not effort.

## Integration points (confirmed by reading the actual call paths)

1. **`python_magnetrun/magnetdata.py::load_magnetdata()`** (lines 80-86) —
   this is the *real*, sole dispatch point for `.txt`/`.csv` loading. It
   currently hardcodes `PandasMagnetData.fromtxt()` / `.fromcsv()` for
   `DataType.PUPITRE`. This is what needs to change (directly, or behind a
   flag during rollout).
2. **`python_magnetrun/readers/registry.py::CONTAINERS`** — maps
   `DataType.PUPITRE → PandasMagnetData` today, but this mapping is
   currently **unused for real dispatch** (only referenced in the module's
   own docstring example — `load_magnetdata()` does not consult it). Update
   it for consistency/future registry-driven dispatch, but note it is not
   itself the mechanism that needs fixing.
3. **`python_magnetrun/readers/csv_readers.py::PupitreReader`** — needs a
   polars-returning read path. Already prototyped in
   [examples/benchmark_pupitre_polars.py](../examples/benchmark_pupitre_polars.py),
   including the two format-robustness fixes found there (malformed doubled-tab
   header, header-only/zero-data-row files) — those fixes need to move from
   the benchmark script into the real reader, not be redone from scratch.

No other call site constructs pupitre data directly (`MagnetRun.fromtxt`
goes through `load_magnetdata()`).

## Phase A — load + ETL parity (required)

Scope determined by tracing the actual call chain `MagnetRun.fromtxt` →
`runetl.prepareData()`:

| Method | Notes |
|---|---|
| `__init__`, `Data` property/setter, `Type` | `self.Data: pl.DataFrame` |
| `getKeys()` | trivial — `df.columns` is already `list[str]` in polars |
| `Units()` | field-def / unit lookup — likely backend-agnostic already (metadata dict, not DataFrame ops) — verify during implementation |
| `getStartDate()`, `getDuration()`, `addTime()` | timestamp handling — port `pd.Timestamp`/`pd.to_timedelta` calls to polars/narwhals equivalents |
| `cleanupData()` | **the real work item** — see below |
| `addData()`, `computeData()` | formula evaluation backing `cleanupData`'s `keys_to_add`; currently pandas `.eval()`-based (per `mrun-cache-implementation.plan.md`'s "Broken pattern" table) — needs expression-based polars rewrite |
| `removeData()`, `renameData()` | straightforward — `.drop()` / `.rename()` exist in polars with similar semantics |
| `getData()` | wraps result with `nw.from_native()` — the narwhals boundary |

**`cleanupData()` specifics** (read directly from
[magnetdata_pandas.py:452-589](../python_magnetrun/magnetdata_pandas.py#L452-L589)):
drops all-zero columns via `self.Data.columns[(self.Data == 0).all()]` +
`.drop(cols, axis=1)`, and drops exact-duplicate columns via two pandas-only
helpers that need polars ports:
- `_dataframe_keys(df)` — trivial, `df.columns` already list-of-str in polars.
- `_get_duplicate_columns(df)` — O(n²) pairwise column comparison using
  `.iloc[:, x]` / `pd.Series.equals()`; polars equivalent uses `df[:, x]` /
  `pl.Series.equals()` or a `frame.equals()`-based approach. Small, contained
  rewrite (~10-15 lines).
- `find_duplicates(df, name, key)` in `utils/duplicates.py` — uses
  `df[key].value_counts()` (pandas Series) vs. polars' `value_counts()`
  (returns a 2-column DataFrame) — needs adjusting, not redesigning.

**Phase A effort estimate:** M, ~1-1.5 weeks — bigger than a single method
port because `cleanupData`/`addData`/`computeData` carry real logic, but
bounded: no more than ~15 methods, most of which map to existing polars
primitives directly.

## Phase B — analysis/plotting surface (incremental, add as exercised)

Everything else on `PandasMagnetData` (`plotData`, `stats`, `extractData`,
`extractDataThreshold`, `extractTimeData`, `saveData`, `shiftTime`,
`add_field`, `getStartDate`/`info`/`__repr__` variants not already covered)
is only needed once analysis/plotting code actually runs against a
`PolarsMagnetData` instance. Per "Simplicity first" — implement these
on demand, one at a time, as real call sites hit `AttributeError` /
`NotImplementedError`, rather than reimplementing all ~20 remaining methods
upfront on spec.

**Phase B effort:** not scoped as a lump sum — S per method, incremental,
driven by actual usage.

## Testing

- New `tests/test_magnetdata_polars.py`, mirroring the structure of
  `tests/test_magnetdata_tdms.py`.
- Two tests ported directly from the benchmark scripts' ad hoc checks, made
  into real regression tests rather than one-off validation:
  - malformed doubled-tab header row (synthetic fixture, small)
  - header-only / zero-data-row file
- Existing pupitre tests (`tests/test_truncated_pupitre.py`, etc.) should
  keep passing unchanged against `PandasMagnetData` until the
  `load_magnetdata()` switch-over happens; add parallel coverage for
  `PolarsMagnetData` rather than converting them in place, so both paths can
  be validated side by side during rollout.

## Rollout

1. Implement `PolarsMagnetData` (Phase A) behind a flag or explicit `fmt=`
   override in `load_magnetdata()` — do not flip the default until Phase A
   is validated against real fixtures (reuse the representative samples from
   `tdms-pupitre-polars-findings.md`: 14 real files across 7 housings).
2. Flip `load_magnetdata()`'s `DataType.PUPITRE` branch to `PolarsMagnetData`
   by default once validated.
3. Update `readers/registry.py::CONTAINERS[DataType.PUPITRE]` to match, for
   consistency (even though it isn't the active dispatch mechanism today).
4. Phase B methods added incrementally afterward, as needed.

**Total estimate: Phase A ~1-1.5 weeks, plus incremental Phase B cost spread
over subsequent work** — smaller and more boundable than the shared-class
rewrite alternative, at the cost of a new class to maintain alongside
`PandasMagnetData` going forward.
