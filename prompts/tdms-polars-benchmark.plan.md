# TDMS Polars Export Benchmark — Spike Plan

## Context

[ROADMAP.md](ROADMAP.md) §4.3 ("Pipeline Redesign (polars/narwhals)") is rated
XL (~4-6 weeks) and listed under "Out of Scope (Deferred)" — the package works
fine without it; this is a performance-only track. The detailed design lives
in [mrun-cache-implementation.plan.md](mrun-cache-implementation.plan.md),
whose **Phase "1+2b-tdms"** covers exactly this: swap `nptdms` for a custom
fork with polars output, then migrate `TdmsMagnetData` internals to
polars/narwhals. Substeps 1-2 of that phase are "implement polars output in
the custom npTDMS fork" and "validate against existing TDMS test fixtures."

The fork now has that polars output: two commits landed on
[`Trophime/npTDMS@master`](https://github.com/Trophime/npTDMS) —
`849d3889` ("add support for polars") and `cc79e1be` ("add support for cuDF
and cudf-polars").

**Key finding — this is a breaking change, not an additive one.**
`TdmsFile.as_dataframe()`, `TdmsGroup.as_dataframe()`, and
`TdmsChannel.as_dataframe()` now unconditionally call the new
`nptdms.export.polars_export` module instead of `pandas_export`. There is no
`df_lib=` toggle. The old pandas path still exists as
`nptdms.export.pandas_export` (gated behind a `pandas` extra) but is only
reachable by importing and calling it directly — not through `.as_dataframe()`
anymore.

This confirms the coupling `mrun-cache-implementation.plan.md` already
predicted: [magnetdata_tdms.py:144-153](../python_magnetrun/magnetdata_tdms.py#L144-L153)
(`TdmsMagnetData._ensure_group_loaded`) calls `group.as_dataframe(...)` and
immediately does `df.memory_usage(deep=True).sum()` and
`df.rename(columns=..., inplace=True)` — both pandas-only. Swapping the
dependency as-is would break this method immediately.

The fork also ships its own `pytest-benchmark` suite
(`test_benchmarks_dataframe.py`, `plot_dataframe_benchmarks.py`) with
published numbers on **synthetic** data: ~8-42× CPU speedup, lower memory,
scaling up with channel count. That doesn't tell us how it behaves on our
actual large TDMS files, which is the gap this spike fills.

## Goal

Measure real-world TDMS group-export speed/memory (polars vs. pandas) on our
actual fixture files, to get a go/no-go signal on Phase 1+2b-tdms — without
committing to any of the internal `TdmsMagnetData` rewrite yet.

## Scope

**In scope:**
- A new branch in the `python_magnetrun` submodule.
- An isolated env with the fork + `polars` + `pandas` installed.
- `python_magnetrun/examples/benchmark_tdms_polars.py` — benchmark script.
- A correctness sanity check between the two export paths.

**Out of scope (deferred pending this spike's results):**
- narwhals wrapping (`mrun-cache-implementation.plan.md` Phase 2) — kept out
  so the numbers measure raw export speed only, not narwhals overhead on top.
- Rewriting `TdmsMagnetData._ensure_group_loaded`/other internals to be
  polars-native (Phase 1+2b-tdms substeps 3-5) — separate, larger step;
  needs its own plan if we proceed.
- cuDF / cudf-polars GPU export (`cc79e1be`) — not requested; noted only
  because this machine has a GPU (`Quadro RTX 4000`) available if it becomes
  relevant later.

## Fixtures used

| File | Size |
|---|---|
| `python_magnetrun/data/M9_Default_200921-123303_Courants50Hz.tdms` | 251 MB |
| `apps/dashboards/magnetdb/tests/M9_Archive_251202-1430.tdms` | 32 MB |

## Steps

1. **Branch** — `git checkout -b spike/tdms-polars-benchmark` in
   `python_magnetrun/` (off current `HEAD`; the submodule currently has
   uncommitted changes in `field_defs.py`, `magnetdata.py`,
   `magnetdata_base.py`, `magnetdata_tdms.py` from other in-progress work —
   these carry over untouched into the new branch's working tree and are not
   staged/committed as part of this spike).
   **Verify:** `git branch --show-current` reports the new branch; `git
   status` still shows the same pre-existing modified files.

2. **Isolated env** — throwaway venv; `pip install
   "git+https://github.com/Trophime/npTDMS.git@master" polars pandas`.
   **Verify:** `import polars, nptdms` resolves `nptdms.__file__` to the
   fork's install location, not PyPI 1.7.0.

3. **Benchmark script** — `python_magnetrun/examples/benchmark_tdms_polars.py`,
   following the existing `examples/benchmark_loading.py` convention
   (argparse CLI, dataclass result record, `time.perf_counter`, NumPy-style
   docstrings with units). For each fixture, per group: time + peak-memory
   (`tracemalloc`) `group.as_dataframe(time_index=False, absolute_time=False,
   scaled_data=True)` (polars, new default) vs.
   `nptdms.export.pandas_export.from_group(group, time_index=False,
   absolute_time=False, scaled_data=True)` (old path, called directly) — the
   same arguments `_ensure_group_loaded` uses today. Print a comparison
   table.
   **Verify:** script runs end-to-end on both fixtures without error.

4. **Correctness sanity check** — `np.allclose` between the pandas- and
   polars-derived arrays for one sampled channel.
   **Verify:** check passes (no data corruption from the export-path swap).

5. **Stage only the new file** — `git add
   python_magnetrun/examples/benchmark_tdms_polars.py`; leave the
   pre-existing modified files untouched/uncommitted; no commit unless
   separately requested.
   **Verify:** `git status` shows only the new file staged.

## Decision point

Once the benchmark numbers are in: decide whether to start the real
Phase 1+2b-tdms (pyproject dependency swap, `TdmsMagnetData` internal
rewrite to polars, then narwhals boundary at `getData()`) as its own planned
piece of work, or to leave 4.3 deferred as-is.
