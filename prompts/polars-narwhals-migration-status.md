# Polars/Narwhals Migration — Status Snapshot

Rollup across all three tracks of ROADMAP item 4.3 ("Pipeline Redesign —
polars/narwhals"). This is a status summary, not a plan — each track's
detailed plan doc stays the source of truth; update those first, then
reflect the change here.

**Detailed plans:** [polars-magnetdata-pupitre.plan.md](polars-magnetdata-pupitre.plan.md) ·
[narwhals-downstream-consumers.plan.md](narwhals-downstream-consumers.plan.md) ·
[mrun-cache-implementation.plan.md](mrun-cache-implementation.plan.md) (TDMS + master phase list) ·
[tdms-pupitre-polars-findings.md](tdms-pupitre-polars-findings.md) (original benchmark evidence + cost estimate)

---

## Pupitre — ✅ live

`load_magnetdata()`'s `.txt` branch dispatches to `PolarsMagnetData` by
default. Every pupitre `.txt` file in the package loads via Polars.
`.csv` is untouched (`PandasMagnetData`/`CsvReader`).

- ✅ Reader layer (`PupitreReader.read_polars()` + 3 format-robustness fixes)
- ✅ Phase A — full ETL-chain parity vs. `PandasMagnetData` on real fixtures
- ✅ Phase B — 5 methods implemented, each backed by a real call site found
  during the work: `plotData`, `stats`, `extractTimeData`, `saveData`,
  `extractDataThreshold`
- ✅ Rollout — the one-line `load_magnetdata()` switch, executed
- ⬜ Phase B remainder (`add_field`, `info`/`__repr__` variants) — no known
  caller yet; stays on-demand per the plan's own philosophy

Full suite: 1248 passed, 19 skipped, zero failures. Benchmarked: `to_pandas()`
boundary-conversion cost confirmed negligible (<2% of the Polars load time
it rides on top of) — see `examples/benchmark_to_pandas.py`.

## Downstream consumers — ✅ done

Needed because `PolarsMagnetData.getData()` returns a narwhals-wrapped
frame, and `extractData()`/`extractTimeData()`/`extractDataThreshold()`
return raw Polars — either way, code written assuming pandas breaks
without a boundary conversion.

- ✅ Phase 0 audit — 76 `getData()` call sites surveyed package-wide
- ✅ Phase 1 fixes — 11 sites fixed via `utils/narwhals_compat.py::to_pandas()`
  (10 from the audit + `runetl.py::_cleanup_pupitre_icoil()`, found only
  once the full test suite ran against the flipped default — a different
  failure *mode*, Polars' stricter type-comparison semantics rather than a
  missing pandas method)
- ⬜ Deferred, not blocking: `processing/correlations.py::tlcc()`/`crosscorr()`
  — murky pre-existing semantics, likely low usage

Two unrelated pre-existing bugs surfaced along the way are tracked
separately in `REVIEW.md` (items 16-17) — not part of this migration, not
blocking anything.

**Open design note** (not yet acted on): `PolarsMagnetData`'s `getData()`
returns narwhals but its three `extract*` methods return raw Polars — an
asymmetry worth cleaning up on that one class for internal consistency,
since `to_pandas()` already handles both cases transparently either way.
Full cross-backend uniformity (every method, every container class,
always narwhals) would need Phase 2 below plus a TDMS equivalent — out of
scope for now, deliberately: that's the "Option A" full rewrite this
migration chose *not* to do.

## TDMS / pigbrother — ⬜ not started

`mrun-cache-implementation.plan.md` Phase 1+2b-tdms.

- ✅ Polars output in the npTDMS fork (already existed upstream)
- ✅ Validated against real fixtures (8 files, 5 housings — ~2.1× speed, ~2.6× memory)
- ⬜ Swap `nptdms` → the fork in `pyproject.toml`
- ⬜ Rewrite `TdmsMagnetData`'s ~15 pandas-specific call sites (`.rename(inplace=True)`, `.memory_usage()`, etc.)
- ⬜ Wrap `TdmsMagnetData.getData()` with `nw.from_native()`
- ⬜ Its own downstream-consumer audit — pupitre proved this is a real,
  necessary step (11 sites, one found only via full-suite testing), not
  optional; budget for it here too, not just the container-class work

**Remaining: ~1.5-2.5 weeks**, independently schedulable from pupitre.

## Not started / not currently planned

- **Phase 2** (master plan) — narwhals boundary for `PandasMagnetData.getData()`.
  Needed for `EnsightMagnetData`, `BProfileMagnetData`, `FeelppMagnetData`,
  and HTS, which still share `PandasMagnetData` untouched.
- **Phase 2b-pandas** — full `PandasMagnetData` internal rewrite to
  polars/narwhals. XL, long-term. Pupitre deliberately avoided needing
  this via the `PolarsMagnetData` sibling-class approach; only relevant if
  Ensight/BProfile/Feelpp/HTS should also move to polars later.
- **Phase 3** — pipeline restructure (eliminate the `select_files`/
  `load_data` double-load). Independent of the backend question.
- **Never benchmarked or scoped**: `BProfileReader`, `EnsightReader`,
  `FeelppReader`, `HtsReader`, the hybrid binary readers (kHz FEPC, RMS,
  VProcess). No polars work exists or is planned for these.

## Bottom line

Pupitre is the one format that's finished and live end-to-end, including
its downstream-consumer shielding. TDMS is next in line, independently
schedulable, and should budget for its own downstream-consumer audit up
front rather than discovering it late the way pupitre did. Everything else
(other readers, full pandas-side rewrite, pipeline restructure) is
unstarted and not currently on a schedule.
