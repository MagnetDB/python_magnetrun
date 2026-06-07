# Test Coverage — Road to 80%

*Created: 2026-06-05 — branch `rework_analysis`*

## Context

Current coverage is **39%** (6,090 / 15,689 statements). Target is **80%** by end of Q3 2026
(see `CHECK_IMPLEMENTATION.md` Success Metrics). The gap is ~6,460 additional statements.

---

## Step 0 — Add omit patterns to `pyproject.toml` (5 min, biggest leverage)

Exclude modules that are genuinely untestable as unit tests (plotting side effects, live network,
standalone validation CLI scripts):

```toml
[tool.coverage.run]
source = ["python_magnetrun"]
omit = [
    "tests/*",
    # plotting-heavy — require display; mocking gives false assurance
    "python_magnetrun/commands/plot.py",
    "python_magnetrun/hybrid/plotting.py",
    "python_magnetrun/utils/plots.py",
    # network/server-dependent — require live LNCMI infrastructure
    "python_magnetrun/requests/webscrapping.py",
    "python_magnetrun/requests/cli.py",
    # standalone validation scripts — not importable library code
    "python_magnetrun/hybrid/rms/validate_rms.py",
    "python_magnetrun/hybrid/trigger/validate_trigger_reader.py",
    # interactive viewer
    "python_magnetrun/viewcsv.py",
]
```

**Effect:** removes ~2,160 uncovered statements from the denominator.
New effective total: ~13,530 statements. New 80% target: ~10,824 covered.
New gap from current 6,090: ~4,734 statements.

→ verify: `python -m pytest tests/ -q` → TOTAL coverage increases without adding a single test.

---

## Tier 1 — Pure-logic tests (~574 statements, ~2–3 days)

No external dependencies beyond `numpy`/`pandas`/`pytest`. One new test file per module.
Use `matplotlib.use("Agg")` at the top of any file that imports matplotlib transitively.

### `tests/test_processing_correlations.py` (new)

Target: `processing/correlations.py` — 191 uncovered, 9% → 70%.

Key functions to cover:
- `crosscorr(s1, s2, lag)` — already partially tested; add edge cases (zero-lag, anticorrelated)
- `compute_lag(data1, data2)` — pure math; build minimal dict fixtures
- `lag_correlation(data1, data2, show=False, save=False)` — needs `matplotlib.use("Agg")`

```python
# fixture pattern
data1 = {"df": pd.Series(...), "field": "B", "range": {"start": 0, "end": None}}
```

### Extend `tests/test_processing.py`

Target: `processing/plateaux.py` — 185 uncovered, 8% → 55%. (+95 stmts)

- `nplateaus` — mock `load_magnetdata`; pass a pre-built `PandasMagnetData` instance
- `detect_plateaux` variants — feed a synthetic step-function `pd.Series`

Target: `processing/peaks.py` — 52 uncovered, 0% → 70%. (+36 stmts)

- `detect_peaks(ts, ...)` — `debug=False` skips `plt.gca()` branch entirely
- Feed a `pd.Series` with a known spike; assert returned `pd.Series` dtype

Target: `processing/trends.py` — 80 uncovered, 25% → 75%. (+53 stmts)

- `fit_linear_approximation`, `fit_linear_data` — linear synthetic data

Target: `processing/filters.py` — 30 uncovered, 19% → 85%. (+24 stmts)

- Filter functions are pure signal operations; feed `np.sin` arrays

### `tests/test_utils_timestamps.py` (new or extend `test_utils.py`)

Target: `utils/timestamps.py` — 67 uncovered, 16% → 80%. (+51 stmts)

- `add_time_columns(df, t0, sampling_rate)` — main function; test with zero/nonzero offset
- `add_time_column_with_offset`, `add_time_column` — verify they delegate correctly
- UTC-naive datetime assertions

### `tests/test_utils_timezone.py` (new)

Target: `utils/timezone.py` — 22 uncovered, 31% → 90%. (+17 stmts)

- Round-trip: UTC → local → UTC
- DST boundary behaviour

### Extend `tests/test_utils.py`

Target: `utils/sequence.py` — 29 uncovered, 12% → 90%. (+26 stmts)

- `list_duplicates_of`, `list_sequence` — feed lists with known duplicates

### `tests/test_txt2csv.py` (new)

Target: `utils/txt2csv.py` — 101 uncovered, 0% → 70%. (+71 stmts)

- Use `tmp_path` + `io.StringIO`; no real file dependency
- Test header parsing, comment stripping, column alignment

### `tests/test_signature.py` (new)

Target: `signature.py` — 111 uncovered, 0% → 55%. (+61 stmts)

- Understand public API first (`grep "^def \|^class "`)
- Feed synthetic current/field arrays

---

## Tier 2 — Extend existing well-started test files (~600 statements, ~3–4 days)

### Extend `tests/analysis/test_processing.py`

Target: `analysis/processing.py` — 293 uncovered, 29% → 70%. (+170 stmts)

Uncovered paths (from `--cov-report=term-missing`):
- `_collect_input_files` with mocked directory containing `.txt` files
- `_run_combined_analysis` with a minimal mocked `HousingConfig`
- `_emit_metrics` — pass a synthetic `pd.DataFrame`
- `--benchmark-downsample` flag path in `process_overview_file`

### Extend `tests/analysis/test_synchronization.py`

Target: `analysis/synchronization.py` — 95 uncovered, 54% → 80%. (+37 stmts)

- `synchronize_data`, `apply_lag_correction` with pre-built DataFrames
- Test `compute_lag` with known-offset sinusoids

### Extend `tests/analysis/test_metrics.py`

Target: `analysis/metrics.py` — 93 uncovered, 56% → 80%. (+37 stmts)

- Exercise remaining metric types; test `None`/empty-DataFrame edge cases

### Extend `tests/analysis/test_loaders.py`

Target: `analysis/loaders.py` — 103 uncovered, 61% → 80%. (+50 stmts)

- `merge_data` with mismatched-column DataFrames
- `load_data` error paths: missing file, bad encoding

### Extend `tests/analysis/test_plotting.py`

Target: `analysis/plotting.py` — 98 uncovered, 60% → 80%. (+49 stmts)

- Set `matplotlib.use("Agg")` before import
- Call `plot_data(...)` with `show=False, save=False` using sample pupitre data

### Extend `tests/test_utils.py` (files section)

Target: `utils/files.py` — 138 uncovered, ~52% → 75%. (+67 stmts)

- `find_files(path, pattern)` — use `tmp_path` with synthetic `.txt` files
- `select_files` — feed a list of paths; verify filter logic
- `load_df` — pass `tmp_path / "sample.txt"` (copy `tests/data/sample_pupitre.txt`)

### Extend `tests/test_file_validation.py`

Target: `utils/validation.py` — 98 uncovered, 12% → 75%. (+70 stmts)

- Run the existing `--cov-report=html` to identify which validators aren't hit
- Missing validators likely: `check_pupitre_truncation`, `FileFormatError` subclasses

### Extend `tests/test_magnetdata.py`

Target: `magnetdata_pandas.py` — 146 uncovered, 73% → 88%. (+55 stmts)

- `addData` / `computeData` with `symbol`/`unit`/`label`/`description` kwargs
- Context manager (`__enter__`/`__exit__`)
- `to_csv` round-trip (write to `tmp_path`, re-read)

Target: `magnetdata_tdms.py` — 285 uncovered, 52% → 70%. (+105 stmts)

- Lazy loading: access `Data` on a freshly-loaded `TdmsMagnetData`
- `getData` branches for different channel names
- Error paths: non-existent group key

---

## Tier 3 — CLI subcommand smoke tests (~500 statements, ~1 day)

Pattern for all CLI modules: use `click.testing.CliRunner` or call `argparse` directly.
`--help` alone covers ~30% of each file (argument registration code). One real invocation
with `tests/data/sample_pupitre.txt` covers the happy path.

```python
from click.testing import CliRunner
from python_magnetrun.commands.stats import register  # or main entry
```

| Module | File to create/extend | Key invocations |
|--------|-----------------------|----------------|
| `commands/stats.py` (205 uncovered) | `tests/test_cli_commands.py` | `--help`, `stats sample.txt --key B` |
| `commands/select.py` (152 uncovered) | same | `--help`, `select sample.txt --key B --tstart 0` |
| `commands/add.py` (95 uncovered) | same | `--help`, addData path with mock formula |
| `analysis/cli.py` (197 uncovered) | `tests/analysis/test_cli.py` | `--help`, `--input sample.txt` |
| `processing/cli.py` (229 uncovered) | `tests/test_processing_cli.py` | `--help`, one processing subcommand |
| `cli.py` (147 uncovered) | extend `tests/test_cli_entrypoints.py` | `magnetrun <subcommand> --help` for all 13 |

→ verify: each `--help` exits with code 0; no `SystemExit` propagates.

---

## Tier 4 — Data-driven tests for hybrid and runlogs (~700 statements, ~1 week)

These need either binary fixtures or `unittest.mock.patch`.

### `tests/test_hybrid_data_extended.py` (new)

Target: `hybrid/hybrid_data.py` — 374 uncovered, 27% → 65%. (+192 stmts)

Strategy: patch `RMSFileReader.__init__` and `FEPCFileReader.__init__` to return mocked objects.
Cover:
- `getData` with downsample config
- `plot_khz_variable(show=False, save=False)` after setting `matplotlib.use("Agg")`
- `_build_groups` error path (directory not found)
- `saveData` guard for missing group key

### Extend `tests/test_hybrid_formula_resolution.py`

Target: `hybrid/hybrid_run.py` — 216 uncovered, 43% → 70%. (+103 stmts)

- Exercise more `getData` branches: missing key, formula with 3 operands
- `get_time_range()` — mock `compute_hour_t0`

### `tests/test_khz_cfg_analyzer.py` (new)

Target: `hybrid/kHz/cfg_analyzer.py` — 168 uncovered, 0% → 60%. (+101 stmts)

- Write a minimal synthetic `.CFG` file to `tmp_path`
- Parse with `load_khz_config` / `CfgAnalyzer`

### Extend `tests/test_hybrid_api.py`

Target: `hybrid/rms/rms_reader.py` — 153 uncovered, 30% → 65%. (+77 stmts)

- Use existing binary RMS fixtures if present; otherwise mock `open` + `struct.unpack`

### `tests/test_pigbrother.py` (new)

Target: `runlogs/pigbrother.py` — 500 uncovered, 19% → 50%. (+192 stmts)

- Write a minimal synthetic pigbrother log file to `tmp_path` (ASCII, known structure)
- Exercise `parse_line`, `read_log`, `to_dataframe`
- Mock `DIR_DEFAULT` / `DIR_SPIKE` constants

### `tests/test_flow_params.py` (new)

Target: `flow_params.py` — 184 uncovered, 0% → 50%. (+92 stmts)

- `setup()` — pure dict; no mocking needed
- `stats(df, ...)` — build a synthetic DataFrame; set `matplotlib.use("Agg")`

---

## Projected coverage after each tier

| Milestone | Covered stmts | Effective total | Coverage |
|-----------|--------------|-----------------|---------|
| Now | 6,090 | 15,689 | 39% |
| + Step 0 (omit exclusions) | 6,090 | ~13,530 | ~45% |
| + Tier 1 | ~6,664 | ~13,530 | ~49% |
| + Tier 2 | ~7,264 | ~13,530 | ~54% |
| + Tier 3 | ~7,764 | ~13,530 | ~57% |
| + Tier 4 | ~8,464 | ~13,530 | ~63% |
| + push 60–75% modules to 90% | ~10,800 | ~13,530 | **~80%** |

The final jump from 63% → 80% comes from pushing the many already-partially-covered modules
(60–75%) up to 85–90%. Run `pytest --cov-report=html` after Tier 4, open `htmlcov/index.html`,
and work through uncovered lines module-by-module — these are the easiest gains at that stage.

---

## Verification command

```bash
source magnetrun-env/bin/activate
python -m pytest tests/ -q  # coverage printed automatically via addopts
```

Success criterion: `TOTAL … XX%` ≥ 80%.

---

## Related

- [CHECK_IMPLEMENTATION.md](CHECK_IMPLEMENTATION.md) — Success Metrics (Q3 2026 target)
- [ROADMAP.md](ROADMAP.md) — strategic direction
