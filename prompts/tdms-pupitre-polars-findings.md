# TDMS / Pupitre Polars Benchmark — Findings & Cost Estimate

Companion to [tdms-polars-benchmark.plan.md](tdms-polars-benchmark.plan.md)
(the spike plan) and [mrun-cache-implementation.plan.md](mrun-cache-implementation.plan.md)
(ROADMAP item 4.3's full design). This doc records what the benchmarks
actually showed, and what a real replacement would cost, so the 4.3
go/no-go decision doesn't have to be re-derived from scratch later.

Benchmark scripts: [examples/benchmark_tdms_polars.py](../examples/benchmark_tdms_polars.py),
[examples/benchmark_pupitre_polars.py](../examples/benchmark_pupitre_polars.py).

---

## Scope of the evidence

This covers **two formats only**: TDMS (pigbrother) and pupitre `.txt`. It
says nothing about `BProfileReader`, `EnsightReader`, `FeelppReader`,
`HtsReader`, or the hybrid binary readers (kHz FEPC, RMS, VProcess) — none
of those were benchmarked.

---

## TDMS (pigbrother) results

Uses the `Trophime/npTDMS` fork (`849d3889` "add support for polars",
`cc79e1be` "add support for cuDF and cudf-polars"), which switched
`TdmsFile`/`TdmsGroup`/`TdmsChannel.as_dataframe()` to return polars
unconditionally (a breaking change — see Cost Estimate below).

**Sample 1** (repo fixtures, 2 files): `M9_Default_200921-123303_Courants50Hz.tdms`
(251 MB) and `M9_Archive_251202-1430.tdms` (32 MB). Result: polars 2-3×
faster and ~2.7× lower peak memory on the large file; CPU time roughly
equal on the smaller, many-small-segments file (later shown to be an
outlier, not a category effect).

**Sample 2** (representative, 8 files from `~/LNCMIG-Data/records/pbsurv`,
5 housings — M1, M5, M8, M9, M19 — all 3 categories — Archive/Default/Overview
— sizes 512 B to 105 MB): across 40 group comparisons, **polars averaged
~2.1× faster (min time) and ~2.6× lower peak memory**, holding consistently
across every housing/category/size, including a near-empty 0-group file
(handled cleanly, no crash). This resolved the Sample 1 anomaly: other
Archive-category files did *not* show the flat CPU-time pattern, so it was
specific to that one file's segment layout, not a general Archive trait.

Every group/channel sanity-checked via `np.allclose` against the pandas
export path (`nptdms.export.pandas_export`) — no data corruption found.

## Pupitre `.txt` results

Production path: `PupitreReader.read()` →
`pd.read_csv(f, sep=r"\s+", skiprows=1, on_bad_lines="warn")`.

**Sample 1** (2 repo fixtures): `M10_2020.10.23---20_10_41.txt` (4.8 MB),
`M9_2019.02.14---23_00_38.txt` (12 MB, malformed header — see below).
1.1-2.3× faster with polars.

**Sample 2** (2 more repo fixtures): `apps/dashboards/magnetdb/tests/2025.12.02 - 14:30:46.txt`
(clean format, different channel layout) and
`python_magnetrun/tests/data/sample_pupitre.txt` (tiny, 6 lines).

**Sample 3** (representative, 10 files from
`~/LNCMIG-Data/records/srv-data-install/{housing}`, 5 new housings — M1,
M3, M5, M7, M8 — sizes 0 B to 19.2 MB): for the 5 files large enough to
give a meaningful signal (5.9-18.3 MB), **polars averaged ~3.1× faster
(min time) and ~1.2× lower memory** (pandas figure is a true
`tracemalloc` peak; polars figure is `estimated_size()` of the final
result only — not directly comparable, see Memory Metric Caveat below).
Smaller speed-to-memory ratio than TDMS, consistent with this being plain
float CSV data (less pandas object/representation overhead than TDMS's
per-group export path).

**Two real format-robustness gaps found and fixed in the benchmark
script** (not yet fixed in the production `PupitreReader`):

1. **Malformed header row** (`M9_2019.02.14---23_00_38.txt`): doubled tabs
   after `Date` and `Time` in the header line only, inconsistent with the
   single-tab data rows below it. Pandas' `sep=r"\s+"` regex collapses
   this; polars' literal-separator `read_csv` does not. Fix used: parse
   the header line separately with the same `\s+` regex, read the data
   body headerless, verify the column-count mismatch is exactly the known
   trailing-tab artifact, then apply the correctly-parsed names.
2. **Header-only files** (metadata line + header line, zero data rows —
   found in production data, e.g. 164 B / 220 B files under
   `srv-data-install`): pandas returns a correctly-shaped 0-row
   DataFrame; `pl.read_csv` raises `NoDataError`. Fix used: catch
   `NoDataError` and construct an empty `pl.DataFrame` from the
   parsed header names.

**Memory metric caveat:** `tracemalloc` only sees allocations passing
through CPython's allocator — accurate for pandas/numpy buffers, blind to
polars' native Rust allocations (verified: a `pl.read_csv` result showing
~0.02 MiB via `tracemalloc` had an actual `estimated_size()` of ~6.9 MiB).
All polars memory figures in both scripts use `estimated_size()` (final
result size) instead, and are documented as not directly comparable to
pandas' peak-during-parse figure.

---

## Cost Estimate

### TDMS — builds on `mrun-cache-implementation.plan.md` Phase "1+2b-tdms"

`TdmsMagnetData` (1702 lines) is TDMS-only — not shared with any other
format's container.

| Step | Status / Effort |
|---|---|
| Polars output in the npTDMS fork | **Done** (upstream, already merged into `Trophime/npTDMS@master`) |
| Validation against real fixtures | **Done** (this session — 8-file representative sample) |
| Add `narwhals` dep; swap `nptdms` → fork in `pyproject.toml` | S, ~0.5 day |
| Rewrite `TdmsMagnetData`'s pandas-specific call sites (15 found: `.rename(inplace=True)`, `.memory_usage()`, etc.) + fix the one broken call site (`_ensure_group_loaded`) + update `tests/test_magnetdata_tdms.py`, `tests/readers/test_tdms_reader.py` | M, ~3-5 days |
| Phase 2 narwhals boundary at `getData()`, TDMS-scoped | S, ~1 day |
| **Subtotal** | **~1.5-2.5 weeks** |

Well under the ROADMAP's original "XL, 4-6 weeks" for all of 4.3 — that
figure bundled the fork implementation (now done, was the biggest
unknown) and phases out of scope here (pipeline restructure, full
pandas-side rewrite).

### Pupitre — no existing scoped plan; two different-sized paths

`PandasMagnetData` (1687 lines, 30 pandas-specific call sites) is
**shared** — it's the base class for `EnsightMagnetData`,
`BProfileMagnetData`, `FeelppMagnetData`, and is also used directly for
HTS. A pupitre-only rewrite of this class is not actually pupitre-only.

- **Narrow path** (read-step only): swap `PupitreReader.read()`/`read_stub()`
  to polars, convert to pandas before handing off to `PandasMagnetData`.
  Isolated to `readers/csv_readers.py`. S-M, ~1-3 days. Captures the
  parse-speed win only (~3×) — `PandasMagnetData.Data` stays pandas, so no
  downstream memory benefit.
- **Full path — shared-class rewrite** (real downstream win, but drags in
  Ensight/BProfile/Feelpp/HTS): the existing plan already calls this
  "Phase 2b-pandas" and flags it as long-term — 30 call sites in a class
  serving 4-5 formats, plus auditing ~26 other files across the package
  calling `.getData()`/`.Data[`. Genuinely XL, and not a pupitre-specific
  cost since the other formats ride along regardless of whether that's
  wanted.
- **Full path — `PolarsMagnetData` sibling class** (real downstream win,
  isolated to pupitre): a new class serving pupitre only, avoiding the
  shared-class risk above entirely. Not smaller in raw effort than the
  shared-class rewrite (`PandasMagnetData` has ~35 substantial methods to
  port either way), but independently schedulable and risk-isolated. See
  [polars-magnetdata-pupitre.plan.md](polars-magnetdata-pupitre.plan.md)
  for the full design: Phase A (load/ETL parity, ~1-1.5 weeks, bounded to
  ~15 methods traced from the actual `MagnetRun.fromtxt` → `prepareData`
  call chain) and Phase B (analysis/plotting methods, incremental, added
  only as exercised).
- **Not yet estimated at all, any path:** moving the two
  format-robustness fixes (malformed header, header-only file) from the
  benchmark script into the real `PupitreReader` — needed either way;
  `polars-magnetdata-pupitre.plan.md` covers this for the `PolarsMagnetData`
  path specifically.

---

## Bottom line

Performance evidence is solid for both formats — real, diverse, repeated
samples, not a hunch. TDMS has a clear, bounded, ~1.5-2.5 week path
forward because its container is isolated and the hardest part (fork
implementation) is already done. Pupitre's performance case is actually
*stronger* (~3× vs ~2×); its cost was initially uncertain because its
container is shared with four other formats, but the `PolarsMagnetData`
sibling-class option (see `polars-magnetdata-pupitre.plan.md`) now gives it
a bounded, isolated path too — Phase A at ~1-1.5 weeks, comparable in shape
to TDMS's path, with Phase B cost spread incrementally over later work.
