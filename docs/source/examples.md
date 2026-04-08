# Examples

All example scripts live in the [`examples/`](https://github.com/MagnetDB/python_magnetrun/tree/main/examples) directory.

## Data Collection & Record Browsing

### `collect-data.py`

Collects pbsurv and srv-data-install files for a given housing between two dates.
For M8 it also gathers CEA/kHz, CEA/rms, CEA/vprocess, and CEA/trigger directories.
Results can optionally be archived to a `.tar.gz` file.

```console
python examples/collect-data.py --housing M9 --start 2024-01-01 --end 2024-03-31 \
    --output /tmp/collected --archive
```

Key options: `--housing`, `--start`, `--end`, `--output`, `--copy-to`, `--archive`, `--data-type`

---

### `get-record.py`

Console utility for filtering, analysing, and plotting magnet run records from `.txt` files.
Provides four subcommands:

| Subcommand  | Purpose                                      |
|-------------|----------------------------------------------|
| `select`    | Filter runs by date range or field criteria  |
| `stats`     | Compute summary statistics per run           |
| `plot`      | Plot individual or aggregated time series    |
| `aggregate` | Build aggregated statistics across runs      |

```console
python examples/get-record.py stats run.txt
python examples/get-record.py plot run.txt --key IH
```

---

### `file_stats.py`

Reports file count, mean, min, max, median, and quartiles for files with a given extension
in a directory.

```console
python examples/file_stats.py /data/runs txt
```

---

### `userdb.py`

Queries the proposals-for-ct API endpoint. Fetches and exports proposal data in JSON or CSV.

```console
python examples/userdb.py --server https://api.example.org --token $TOKEN \
    --command proposals --output csv
```

Key options: `--server`, `--token`, `--command`, `--output`, `--limit`, `--debug`

---

### `proposal.py`

Loads a proposal CSV from the User DataBase, attaches magnet records by timestamp,
computes plateau statistics with `nplateaus()`, and generates plateau analysis plots.

```console
python examples/proposal.py proposals.csv --mdatadir /data/runs --show
```

---

## Signal Analysis

### `cmp_fields.py`

Compares two time series (e.g., two current or field channels) from magnet records.
Computes Pearson correlation, detects lag, calculates DTW distance, and optionally
fits a piecewise linear approximation.

```console
python examples/cmp_fields.py run.tdms --xkey IH --ykey IB --dtw --save
```

Key options: `--xkey`, `--ykey`, `--lagcorrelation`, `--range`, `--to`, `--dtw`, `--save`, `--outputdir`

---

### `corr_Ih_Ib.py`

Analyses correlation between helix (Ih) and bottom (Ib) currents using multiple breakpoint
detection algorithms: piecewise aggregation, `pwlf`, piecewise regression, and `ruptures`.

```console
python examples/corr_Ih_Ib.py run.tdms --xkey IH --ykey IB --algo pwlf --breakpoints 3
```

Key options: `--xkey`, `--ykey`, `--breakpoints`, `--algo`, `--normalize`, `--save`, `--find`

---

### `outliers.py`

Detects and visualises outliers in TDMS data using Median Absolute Deviation (MAD),
comparing reference vs actual current measurements with rolling window statistics.

```console
python examples/outliers.py run.tdms --site mysite --insert M9 --plot --threshold 3.5
```

Key options: `--site`, `--insert`, `--threshold`, `--window`, `--normalize`, `--plot`, `--save`

---

### `timeseries-anomaly-detection.py`

Applies multiple anomaly detection methods to magnet run time series:
Z-score, IQR, moving average, moving median, moving Z-score, and Isolation Forest.

```console
python examples/timeseries-anomaly-detection.py run.tdms --site mysite --insert M9 \
    --plot --threshold 3.0 --window 50
```

Key options: `--site`, `--insert`, `--threshold`, `--window`, `--normalize`, `--plot`, `--save`

---

### `pupitre.py`

Analyses trends in pupitre data (cooling system parameters) using rolling window statistics,
signal detection, and lag correlation analysis between `Pmagnet` and `TAlimout`.

```console
python examples/pupitre.py pupitre_run.txt
```

---

## Hybrid / kHz Data

### `plot_hybrid_minimal.py`

Minimal example: loads kHz hybrid data, auto-discovers corresponding pupitre (`.txt`) and
TDMS files by date, maps field names, and overlays all three sources on one graph.

```console
python examples/plot_hybrid_minimal.py
```

Demonstrates core `HybridRun` + `MagnetRun` loading patterns.

---

### `plot_hybrid_with_pupitre_tdms.py`

Full-featured version of the hybrid comparison example. Supports flexible field-name mapping,
auto-discovery of pupitre/TDMS files, normalization, and time-range filtering.

```console
python examples/plot_hybrid_with_pupitre_tdms.py \
    --date 2024-02-15 --fepc-system FEPC-LNCMI \
    --hybrid-dir /data/kHz --pupitre-dir /data/pupitre \
    --key IH --normalize --save
```

Key options: `-d/--date`, `-s/--fepc-system`, `-k/--key`, `--site`, `--hours`,
`--hybrid-dir`, `--pupitre-dir`, `--pigbrother-dir`, `--normalize`, `--show`, `--save`,
`--log-level`, `--log-file`

---

### `example_fepc_usage.py`

Six self-contained examples of the FEPC reader API:

1. Parse a `.cfg` file to get slot configuration
2. Read a single analog block
3. Read a full hour file
4. Extract variables by name
5. Plot FEPC data with matplotlib
6. Apply calibration parameters

```console
python examples/example_fepc_usage.py
```

---

### `rms_examples.py`

Ten examples demonstrating the RMS file reader for FEPC-AUX-LNCMI files:

1. Quick read
2. Detailed file info
3. Selective channel reading
4. Time-range filtering
5. Digital signal analysis
6. Analog signal analysis
7. Plotting
8. Export to CSV / JSON
9. Temperature channel analysis
10. Batch processing

```console
python examples/rms_examples.py 1   # run example 1
python examples/rms_examples.py 7   # run the plotting example
```

---

### `plot_rms.py`

Quick CLI plotter for RMS file variables.

```console
python examples/plot_rms.py data.rms CH1 CH2 --same-plot -o plot.png
```

Key options: `file`, `variables`, `-o/--output`, `--same-plot`

---

### `plot_fepc_data.py`

Comprehensive FEPC binary data plotter. Supports variable search, automatic file discovery,
calibration application, and outlier removal (IQR / zscore / MAD / percentile).

```console
python examples/plot_fepc_data.py -c config.cfg -v IH -d 2024-02-15 \
    --remove-outliers --outlier-threshold 3.0
```

Key options: `-c/--cfg`, `-v/--variable`, `-s/--slot`, `-o/--output`, `-d/--date`,
`--remove-outliers`, `--outlier-threshold`, `--cnv-dir`, `--endian`

---

### `plot_trigger_data.py`

Plots trigger data from FEPC binary trigger files. Supports single or multiple trigger
comparison, calibration application, and custom PRE/POST time windows.

```console
python examples/plot_trigger_data.py --base-dir /data/triggers \
    --date 2024-02-15 --system FEPC-LNCMI --variable IH --save
```

Key options: `--base-dir`, `--trigger-dir`, `--date`, `--system`, `--variable`,
`--all`, `--endian`, `--save`, `--no-show`, `--no-calib`, `--cnv-dir`

---

### `example_trigger_usage.py`

Nine step-by-step examples of the trigger reader API:

1. List available triggers
2. Read trigger metadata
3. Enumerate trigger files
4. Load trigger configuration
5. Read variable data
6. Read slot data
7. PRE/POST window analysis
8. Plot trigger events
9. Custom time window analysis

```console
python examples/example_trigger_usage.py
```

---

### `plot_vprocess.py`

VProcess data plotter. Supports four plot types: variables plot, overview,
variable comparison (2×2 layout), and correlation heatmap.

```console
python examples/plot_vprocess.py data.vprocess --overview --save
python examples/plot_vprocess.py data.vprocess --vars V1 V2 --compare
python examples/plot_vprocess.py data.vprocess --heatmap
```

Key options: `filepath`, `--vars`, `--overview`, `--max-vars`, `--compare`,
`--heatmap`, `--layout`, `--save`, `--no-show`

---

## Energy Balance & Water Flow

### `bilan.py`

Energy balance analysis: loads PigBrother (TDMS) and pupitre (`.txt`) data, computes HT
phase and amplifier power, performs Busbar power balance, and calculates secondary loop
properties using water properties from `python_magnetcooling`.

```console
python examples/bilan.py run.tdms --pigbrother_datadir /data/pb \
    --pupitre_datadir /data/pupitre --show
```

Key options: `input_file`, `--pigbrother_datadir`, `--pupitre_datadir`, `--show`, `--debug`

---

### `flow_params_pipeline.py`

Standalone flow parameter extraction pipeline using `scipy.optimize.curve_fit` to fit
pump speed, flow rate, and pressure curves, then builds a `WaterFlow` object and runs
hydraulic calculations.

```console
python examples/flow_params_pipeline.py --show-plots
```

---

### `flow_params_magnetrun_pipeline.py`

Same pipeline using `python_magnetrun` methods: automatic Imax detection via `pwlf`
breakpoint fitting followed by `python_magnetrun.processing.fit` for flow/pressure curves.

```console
python examples/flow_params_magnetrun_pipeline.py --show-plots --debug
```

---

### `waterflow_debitbrut_example.py`

Four examples demonstrating the `debitbrut()` method with hysteresis model for secondary
cooling loop flow rates:

1. Basic `debitbrut` usage
2. JSON loading with hysteresis parameters
3. Array-based power cycle analysis with hysteresis plot
4. Error handling

```console
python examples/waterflow_debitbrut_example.py
```

---

## Diagnostics

### `test_error_logging.py`

Demonstrates the enhanced error logging utilities from `python_magnetrun.hybrid.utils`:
`log_exception()` for full traceback logging and `format_exception_location()` for concise
error location reporting, including nested exception handling.

```console
python examples/test_error_logging.py
```
