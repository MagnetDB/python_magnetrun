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
   this is the *real*, sole dispatch point for `.txt`/`.csv` loading:

   ```python
   if ext == ".txt":
       return PandasMagnetData.fromtxt(filename, defs_file=...)
   return PandasMagnetData.fromcsv(filename, defs_file=defs_file)
   ```

   Only the `.txt`/`fromtxt` branch is in scope. `.fromcsv()` uses
   `CsvReader` — a different reader entirely, not pupitre's tab format —
   even though both branches share the same `DataType.PUPITRE` enum value
   today (a slightly misleading name: it really means "delimited-text
   formats routed through `PandasMagnetData`", not strictly pupitre).
   `.fromcsv()`/`CsvReader` stay on `PandasMagnetData`, untouched.

   The existing `fmt: str | None` parameter can't be reused as a
   `PandasMagnetData`-vs-`PolarsMagnetData` toggle — it only overrides
   `detect_type()`'s *type* detection (tdms/pupitre/hybrid/hts), not which
   container class handles a given type. Simplest path: don't touch
   `load_magnetdata()` at all during Phase A — validate against
   `PolarsMagnetData.fromtxt()` directly (as the reader-level tests already
   do), and make the eventual switch a single one-line change
   (`PandasMagnetData.fromtxt` → `PolarsMagnetData.fromtxt`) only once
   Phase A is fully validated. No flag-plumbing needed.

2. **`python_magnetrun/readers/registry.py::CONTAINERS`** — maps
   `DataType.PUPITRE → PandasMagnetData`, but confirmed via grep that
   `CONTAINERS[dtype]` is referenced **exactly once in the whole codebase**,
   inside `registry.py`'s own module docstring — an illustrative snippet,
   never executed. `detect_type()` itself is called in only two places
   total: inside `load_magnetdata()` (feeding hardcoded if/elif branches,
   not `CONTAINERS`) and that same docstring. Nothing anywhere actually
   consults this registry for dispatch today.

   **Do not update this entry as part of the pupitre migration.** Doing so
   would actually be wrong, not merely redundant: `DataType.PUPITRE` covers
   both `.txt` (→ `PolarsMagnetData`, once switched) and `.csv` (→ stays on
   `PandasMagnetData` via `CsvReader`/`fromcsv`, see point 1) — a single
   `CONTAINERS[DataType.PUPITRE]` entry can't represent both. Leave it
   pointing at `PandasMagnetData` (stale/inert either way) until the type
   system distinguishes pupitre-`.txt` from generic-`.csv`, or until
   something actually builds the registry-driven dispatch path the
   docstring describes.
3. **`python_magnetrun/readers/csv_readers.py::PupitreReader`** — needs a
   polars-returning read path. ✅ **Done**: `read_polars()`, `read_stub_polars()`,
   and shared `_parse_header()` / `_read_polars_impl()` helpers added, porting
   the logic from
   [examples/benchmark_pupitre_polars.py](../examples/benchmark_pupitre_polars.py)
   including both known format-robustness fixes (malformed doubled-tab header,
   header-only/zero-data-row files). `polars` imported lazily inside the new
   methods only; added as an optional `polars` dependency group in
   `pyproject.toml`. **A third real quirk surfaced during implementation**:
   the existing `tests/data/sample_pupitre.txt` fixture has no trailing tab
   at all (unlike every file benchmarked earlier), so the trailing-artifact
   check now accepts `n_extra ∈ {0, 1}` instead of requiring exactly 1.
   New fixtures `tests/data/pupitre_malformed_header.txt` and
   `tests/data/pupitre_header_only.txt`, plus a `TestPupitreReaderPolars`
   test class (gated on `pytest.importorskip("polars")`) in
   `tests/readers/test_csv_readers.py` — 87/87 tests pass in
   `tests/readers/` + `tests/test_truncated_pupitre.py`.
   **Not yet done:** nothing wired into `load_magnetdata()` or
   `PandasMagnetData` — today's default pupitre-loading behavior is
   unchanged; these new reader methods aren't called from anywhere yet.

No other call site constructs pupitre data directly (`MagnetRun.fromtxt`
goes through `load_magnetdata()`).

## Phase A — load + ETL parity (required)

**Status: ✅ Done.** `python_magnetrun/magnetdata_polars.py` implements
`PolarsMagnetData(MagnetDataBase)` with every method in the table below.
Validated directly via `PolarsMagnetData.fromtxt()` (no `load_magnetdata()`
changes, per integration point 1) against real fixtures, including a
full ETL-chain parity check against `PandasMagnetData` (`Units()` →
`addTime()` → `cleanupData()` with real housing-config formulas) on both
`M10_2020.10.23---20_10_41.txt` (clean) and `M9_2019.02.14---23_00_38.txt`
(malformed header) — identical keys, `np.allclose` on every numeric column.
30 new tests in `tests/test_magnetdata_polars.py`; full suite (1233 tests)
passes with zero regressions.

Notable implementation decisions:
- **`addData`/`computeData` formula grammar**: checked all 5 housing-config
  JSON files — every real formula is a pure column sum (`"IH = Idcct1 +
  Idcct2"`, generated voltage sums, etc.), no functions/division/etc. So
  `addData` uses a small `ast`-based evaluator supporting only `+ - * /`,
  unary `+ -`, parentheses, columns, and numeric literals — not the full
  `pandas_builtins` function set (`sqrt`, `sin`, …), which is unused in
  practice. Add functions there if a real formula ever needs one.
- **`addTime`'s DST-ambiguity handling**: reused
  `utils.timezone.series_local_to_utc_naive` (already-vetted pandas logic
  for DST fall-back edge cases) via a small one-time round-trip through a
  pandas Series, rather than reimplementing that logic in Polars. Runs
  once per file, not a hot loop, so the conversion cost is negligible.
- **`cleanupData`'s all-zero-column check**: restricted to numeric dtypes
  explicitly (`dtype.is_numeric()`) — comparing a non-numeric column to `0`
  raises in Polars, unlike pandas which silently evaluates to all-False.
- **`getData()`** wraps its result with `nw.from_native()` as planned;
  `narwhals>=1.0` added to the `polars` optional-dependency group.

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

**Status: 🔶 4 methods done, driven by real call-site usage** —
`plotData()`, `stats()`, `extractTimeData()`, `saveData()` implemented,
found by grepping actual call sites of `PolarsMagnetData`-bound methods
across the package (`viewcsv.py:45` → `plotData`, `commands/select.py:178,183`
→ `extractTimeData`, `MagnetRun.py:568` → `stats`, `MagnetRun.py:618`/
`viewcsv.py` → `saveData`). Notable implementation notes:
- `plotData()` cannot reuse pandas' `DataFrame.plot()` (Polars has no
  equivalent) — draws directly via `ax.plot()` on numpy arrays instead.
  `x="timestamp"` local-time display reuses the same pandas-round-trip
  pattern as `addTime()`.
- `stats()` uses Polars' own `.describe()` (returns a tidy `statistic`-row
  DataFrame) rather than porting pandas' `.describe()` shape exactly.

Remaining (`extractDataThreshold`, `add_field`, `getStartDate`/`info`/
`__repr__` variants not already covered): still genuinely on-demand,
no real call site found yet exercising them for pupitre specifically.

**⚠️ Implementing these 4 methods is necessary but NOT sufficient to
safely execute Rollout step 2 (see below) — a deeper, separate issue was
found while tracing call sites:** `python_magnetrun/processing/stats.py::stats()`
(the function backing the `magnetrun stats` CLI's default path, via
`commands/stats.py:62`) calls `Data.getData([f])[f].mode().iloc[0]` —
pandas-Series-only chaining (`.mode()`, `.iloc`) applied directly to
`getData()`'s result. Since `PolarsMagnetData.getData()` returns a
narwhals-wrapped frame (not pandas), this breaks regardless of what the
container class itself implements. This is the exact "migrate downstream
consumers to narwhals API" work `mrun-cache-implementation.plan.md`
already tracks as a separate, larger item for TDMS (Phase 2, Implementation
Order step 10) — it turns out to apply to pupitre too, once the switch is
flipped. `MagnetRun.getDataFrame()` has the same problem in miniature: its
docstring/type-hint promises `pd.DataFrame` for `DataType.PUPITRE`, which a
narwhals frame doesn't satisfy.

**Phase B effort:** not scoped as a lump sum — S per method, incremental,
driven by actual usage. The downstream-consumer audit above is a distinct,
unscoped piece of work, not a Phase B method.

## Testing

- New `tests/test_magnetdata_polars.py`, mirroring the structure of
  `tests/test_magnetdata_tdms.py` — **still pending**, since the class
  itself doesn't exist yet.
- Two tests ported directly from the benchmark scripts' ad hoc checks, made
  into real regression tests rather than one-off validation: ✅ **Done at
  the reader level** — `TestPupitreReaderPolars` in
  `tests/readers/test_csv_readers.py` covers the malformed doubled-tab
  header and the header-only/zero-data-row file, plus the no-trailing-tab
  case found along the way. Will need equivalent coverage at the
  `PolarsMagnetData` level once that class exists.
- Existing pupitre tests (`tests/test_truncated_pupitre.py`, etc.) should
  keep passing unchanged against `PandasMagnetData` until the
  `load_magnetdata()` switch-over happens; add parallel coverage for
  `PolarsMagnetData` rather than converting them in place, so both paths can
  be validated side by side during rollout.

## Rollout

1. Implement `PolarsMagnetData` (Phase A), validating directly against
   `PolarsMagnetData.fromtxt()` — no changes to `load_magnetdata()` needed
   yet. Validate against real fixtures (reuse the representative samples
   from `tdms-pupitre-polars-findings.md`: 14 real files across 7 housings).
2. ✅ **Done.** Flipped `load_magnetdata()`'s `.txt` branch (only) from
   `PandasMagnetData.fromtxt()` to `PolarsMagnetData.fromtxt()`
   ([magnetdata.py:82-85](../python_magnetrun/magnetdata.py#L82-L85)) — a
   single one-line change; the `.csv`/`fromcsv()` branch is untouched.

   Running the full suite against the flipped default surfaced **one more
   real consumer the audit had missed**: `runetl.py::_cleanup_pupitre_icoil()`
   (called by every `MagnetRun.fromtxt()` via `prepareData()`) does
   `(df[col] == 0).all()` across *every* column, including `timestamp` —
   Polars raises `NotImplementedError: Series of type Datetime(...) does
   not have eq operator` there, where pandas silently returns all-`False`.
   This is a genuinely different failure *mode* than anything Phase 0's
   idiom-based grep could catch (`.mean()`/`.all()` exist on both
   backends — the break is Polars' stricter type-comparison semantics, not
   a missing method), which is why only a real end-to-end test run caught
   it. Fixed the same way as everything else: `to_pandas()` at both
   `getData()` call sites in that function.

   Also surfaced a real, pre-existing test
   (`tests/test_magnetdata.py::TestRealisticM9Txt::test_extract_threshold_field_20`)
   exercising `extractDataThreshold()` on pupitre data — genuinely not
   implemented yet on `PolarsMagnetData` (a Phase B method with no
   previously-known caller). Implemented it (mirrors `PandasMagnetData`'s
   `.loc[df[key] >= threshold]` as `self.Data.filter(pl.col(key) >= threshold)`,
   returning raw Polars like `extractData`/`extractTimeData`).

   Full suite: 1248 passed, 19 skipped, zero failures. Also smoke-tested
   the real `MagnetRun.fromtxt()` → `getDataFrame()`/`stats()`/`plotData()`
   chain directly (not just pytest) — confirmed working end-to-end.
3. **Do not** update `readers/registry.py::CONTAINERS[DataType.PUPITRE]` —
   see integration point 2 above; it can't correctly represent the
   `.txt`-vs-`.csv` split and nothing reads it today regardless.
4. Phase B methods added incrementally afterward, as needed
   (`extractDataThreshold` now done; `add_field`, `info`/`__repr__`
   variants still genuinely unexercised).

**Total estimate: Phase A ~1-1.5 weeks, plus incremental Phase B cost spread
over subsequent work** — smaller and more boundable than the shared-class
rewrite alternative, at the cost of a new class to maintain alongside
`PandasMagnetData` going forward.
