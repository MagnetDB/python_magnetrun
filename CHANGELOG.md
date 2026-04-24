# Changelog

## Breaking Changes

### v0.3.0 — Timestamp convention unified to naive UTC

`timestamp` columns in all `MagnetData` classes are now stored as **naive UTC**.
Previously `PandasMagnetData` stored local time, which silently broke comparisons with
TDMS/hybrid timestamps.

| Class | Before (≤ 0.2.x) | After (≥ 0.3.0) |
|---|---|---|
| `PandasMagnetData.Data["timestamp"]` | naive local time | naive UTC |
| `TdmsMagnetData.Data[group]["timestamp"]` | naive UTC | naive UTC (unchanged) |
| `start_timestamp` / `end_timestamp` | naive UTC (unchanged) | naive UTC (unchanged) |

#### Affected methods

##### `addTime(time_zone="Europe/Paris")`

Added `time_zone` parameter to all three classes (`MagnetDataBase`, `PandasMagnetData`,
`TdmsMagnetData`).

**`PandasMagnetData.addTime`** now stores `timestamp` as naive UTC instead of naive local time.
The `time_zone` argument tells it which timezone the source `Date`/`Time` columns are in
(default: `"Europe/Paris"`).

**`TdmsMagnetData.addTime`** is now *eager*: it calls both `addTdmsTime()` and
`addTdmsTimestamp()` for all non-`Infos` groups in one call. Previously only `addTdmsTime()`
was called and callers had to invoke `addTdmsTimestamp()` separately.

##### `extractTimeData(timerange, group=None, time_zone="Europe/Paris")`

**Timerange format changed** — was `"HH:MM:SS;HH:MM:SS"`, now
`"YYYY-MM-DD HH:MM:SS;YYYY-MM-DD HH:MM:SS"`. The strings are interpreted as **local time**
in the `time_zone` timezone and converted to UTC internally before filtering.

Both boundaries remain inclusive.

`PandasMagnetData.extractTimeData` now raises `RuntimeError` if `addTime()` has not been
called yet (previously it would silently read the raw `Time` string column, often returning
wrong results).

`TdmsMagnetData.extractTimeData` now requires `group` to be non-`None` and raises
`RuntimeError` if `addTime()` has not been called yet. Previously it called
`addTdmsTimestamp()` lazily and returned a boolean Series instead of a DataFrame.

##### `plotData(..., time_zone="Europe/Paris")`

Added `time_zone` parameter. When `x == "timestamp"` the UTC-stored column is converted to
naive local time before being passed to matplotlib, so axis labels display local time.

#### Migration guide

```python
# Old (≤ 0.2.x)
md.addTime()
df = md.extractTimeData("09:53:00;10:00:00")

# New (≥ 0.3.0)
md.addTime()                          # time_zone="Europe/Paris" is the default
df = md.extractTimeData("2025-11-05 09:53:00;2025-11-05 10:00:00")
```

If you read `timestamp` values directly and compare them to other timestamps, ensure both
sides are naive UTC. Previously `PandasMagnetData` timestamps were naive local — code that
compared them to TDMS timestamps or `start_timestamp` would have been silently wrong.

#### What is NOT changed

- `start_timestamp` / `end_timestamp` attributes — already naive UTC, no change.
- `t` column — elapsed seconds, no change.
- `TdmsMagnetData.addTdmsTime()` / `addTdmsTimestamp()` — still available as lower-level
  helpers; their signatures and behaviour are unchanged.
- `analysis/loaders.py`, `analysis/processing.py` — use `timestamp` as UTC for
  synchronization; already correct, no changes needed.
- `HybridData` — out of scope for this release; needs its own follow-up.

---

## To-do

**Fix**
- [X] Fix time data in Hybrid kHz files
- [X] Fix python_magnetrun/cli.py input_args
- [X] Fix python_magnetrun/cli.py plot with multiple files
- [X] Plot_xy: raise an exception when trying to plot pairs that have different units
- [X] Fix python_magnetrun/analysis/plotting.py remove downsample_percent handling
- [X] Fix analysis -- no pupitre nor Reference displayed -- check also with plotly backend
- [ ] Add demo use of downsampling in the analysis CLI

**Finish**
- [ ] hybrid data loading and unified interface
- [ ] hybrid data validation and comparison with Pupitre/PigBrother
- [X] `HybridData` timestamp unification (follow-up to v0.3.0)

**Refactor:**
- [X] Split argparse options into separate Python files
- [X] Add an example / a test for each subcommand in `python_magnetrun`
- [X] Rework `MagnetData` into base + pandas + tdms classes
- [ ] Store stats (plateaus, duration) in a DataFrame, CSV, or database
- [X] Refactor plot functions to use a common interface and support multiple backends (Matplotlib, Plotly, Seaborn)
- [ ] Refactor `analysis` module to separate synchronization, metrics, and visualization into distinct classes/functions
- [ ] Add hybrid data support to `analysis` module (e.g. synchronization with Pupitre/PigBrother, distance metrics)

**Docs:**
- [X] Docs for aggregate
- [X] Add a note to mount PigBrother data
- [X] Add note to mount Pupitre data if applicable
- [X] Document ETL, waterflow, and thermal pipeline modules

**CI/CD:**
- [X] Add code coverage with `pytest-cov` (generates `coverage.xml`)
- [X] Upload coverage reports to [Codecov](https://codecov.io/gh/MagnetDB/python_magnetrun) via `codecov/codecov-action@v5`
- [ ] Authorize the `MagnetDB/python_magnetrun` repository on [codecov.io](https://codecov.io) (sign in with GitHub) to activate the badge

**Units:**
- [ ] Use python_magnetunits for unit conversions and dimensional analysis in formulas (e.g. power, busbar losses)

**Dashboard:**
- [ ] Streamlit and Panel dashboards via `rustfs/` object storage integration
- [ ] Add Marimo notebooks equivalents
- [ ] Add standalone voila dashboard for non-technical users — along with a Dockerfile for easy deployment

**Features:**
- [X] ETL functions to clean and normalise Pupitre data (`runetl`)
- [X] Waterflow and thermal pipeline modules
- [X] Rewrite `txt2csv` to use methods in `utils` and `plots`
- [ ] Check `addData` complex formulas (involving `freesteam` / `iapws`) — with `pyparsing`?
- [ ] Export data to `great_tables`, `tabular`, `rich` or `csv2md`
- [ ] Add support for Origin files (`liborigin` / Python bindings)
- [ ] For `select`, support multiple field criteria
- [ ] Cross-lag correlations
- [ ] Forecast Teb from historical data
- [ ] Check independent variables (Ih, Teb, Qbrut) on plateau experiments
- [ ] Link with magnet user DB (`xdds.csv`)
- [ ] Classification of field profiles
- [ ] Link with `magnettools`/`hifimagnet` for R(i) and L(i)
