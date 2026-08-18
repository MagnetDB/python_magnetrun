# Examples

This directory contains example scripts demonstrating various features of the `python_magnetrun` package.

## Data Collection

### `collect-data.py`

Collects `pbsurv` (TDMS) and `srv-data-install` (pupitre `.txt`) files for a given housing (magnet site) between two dates. For M8, also collects CEA/kHz, CEA/rms, CEA/vprocess and CEA/trigger directories.

**Usage:**
```bash
# List all files for M9 in 2024
python collect-data.py --housing M9 --start 2024-01-01 --end 2024-12-31

# List only pigbrother (pbsurv TDMS) files
python collect-data.py --housing M9 --start 2024-01-01 --end 2024-12-31 --data-type pigbrother

# List only pupitre (srv-data-install .txt) files
python collect-data.py --housing M9 --start 2024-01-01 --end 2024-12-31 --data-type pupitre

# List only CEA/hybrid data (M8 only)
python collect-data.py --housing M8 --start 2024-01-01 --end 2024-12-31 --data-type hybrid

# Save file list to a text file
python collect-data.py --housing M8 --start 2023-06-01 --end 2023-12-31 --output results.txt

# Copy all found files to a destination directory
python collect-data.py --housing M10 --start 2024-01-01 --end 2024-12-31 --copy-to /path/to/dest

# Archive all data into three separate archives
python collect-data.py --housing M8 --start 2024-01-01 --end 2024-12-31 --archive M8_2024
# Produces:
#   M8_2024-pupitre.tar.gz   — srv-data-install .txt files (gzip compressed)
#   M8_2024-pbsurv.tar       — pbsurv .tdms files (uncompressed)
#   M8_2024-cea.tar          — CEA kHz/rms/vprocess/trigger dirs (uncompressed)

# Archive only pigbrother data (produces M9_pb_2024-pbsurv.tar only)
python collect-data.py --housing M9 --start 2024-01-01 --end 2024-12-31 --data-type pigbrother --archive M9_pb_2024
```

**Arguments:**
| Argument | Description |
|---|---|
| `--housing` | Housing name (M1–M10), required |
| `--start` | Start date inclusive (YYYY-MM-DD), required |
| `--end` | End date inclusive (YYYY-MM-DD), required |
| `--output` | Write file list to this path instead of stdout |
| `--copy-to` | Copy all found files/dirs into this destination directory |
| `--archive` | Base name for up to three archives (see below); any `.tar`/`.tar.gz` suffix is stripped automatically |
| `--data-type` | Filter data source: `pigbrother` (pbsurv TDMS only), `pupitre` (srv-data-install `.txt` only), `hybrid` (CEA data only, M8 only); omit to collect all sources |

**Archive format:**

When `--archive BASE` is given, up to three archives are created depending on which data sources are collected:

| File | Contents | Compression |
|---|---|---|
| `BASE-pupitre.tar.gz` | `srv-data-install` `.txt` files | gzip (text compresses well) |
| `BASE-pbsurv.tar` | `pbsurv` `.tdms` binary files | none (binary, compression gives no benefit) |
| `BASE-cea.tar` | CEA kHz/rms/vprocess/trigger directories | none (large binary data) |

Paths inside every archive are relative to `LNCMIG-Data`. Only archives that have matching items are created.

**Data directories searched:**

Data directories are resolved from `python_magnetrun.data_dirs` using the same priority chain as the rest of the package: environment variables (`MAGNETRUN_PUPITRE_DATA_DIR`, `MAGNETRUN_PIGBROTHER_DATA_DIR`, `MAGNETRUN_HYBRID_DATA_DIR`) → `~/.config/python_magnetrun/data_dirs.json` → built-in defaults.

| Source | Directory |
|---|---|
| pupitre `.txt` | `BASE_DIR/srv-data-install/{housing}/` |
| pigbrother `.tdms` | `BASE_DIR/pbsurv/{housing}/` and `BASE_DIR/pbsurv/Fichiers_Data_ACQ_ENET_{YYYY}/{housing}/` |
| CEA hybrid | `BASE_DIR/CEA/{kHz,rms,vprocess,trigger}/` (M8 only) |

**File naming patterns matched:**

Each data source uses a different naming convention; the script filters by date using these patterns:

| Source | Pattern | Example |
|---|---|---|
| pupitre `.txt` | `YYYY.MM.DD - HH:MM:SS.txt` | `2024.03.15 - 09:42:01.txt` |
| pbsurv `.tdms` | `*_YYMMDD-HHMM[SS][_label].tdms` | `M9_Archive_240315-1042.tdms` |
| CEA dated dirs | `YYYY-MM-DD` (flat) or `YYYY/YYYY-MM-DD` (pre-2025) | `2024-03-15` |
| CEA trigger dirs | `TRIGGER__YYYY-MM-DD__HH-MM` | `TRIGGER__2024-03-15__10-42` |

**pbsurv directory tree scanned:**

```
pbsurv/
└── {housing}/
    ├── Fichiers_Archive/      ← TDMS files
    ├── Fichiers_Default/
    ├── Fichiers_Manuel_Trig/
    ├── Fichiers_Spike/
    ├── Overview/
    ├── Model/
    └── Models/
pbsurv/
└── Fichiers_Data_ACQ_ENET_{YYYY}/   ← legacy yearly layout
    └── {housing}/
        └── Fichiers_*/
```

**Sample output** (list mode, no `--archive`/`--copy-to`):

```
Housing : M9
Period  : 2024-01-01 → 2024-12-31

[pbsurv]  (42 item(s))
  /…/pbsurv/M9/Fichiers_Archive/M9_Archive_240315-1042.tdms
  …

[srv-data-install]  (87 item(s))
  /…/srv-data-install/M9/2024.01.03 - 08:11:22.txt
  …
```

---

### `pupitre_to_duckdb.py`

Reads one or more pupitre `.txt` files, extracts a user-chosen set of fields
(always including `timestamp`), and upserts them into a dedicated DuckDB file.
Each housing (M9, M10, …) is stored in its own table; rows with duplicate
timestamps are silently skipped.

**Usage:**
```bash
# Extract Field, Icoil1, Ucoil1 and Pmagnet from all M9 files into pupitre.duckdb
python pupitre_to_duckdb.py \
    --fields Field Icoil1 Ucoil1 Pmagnet \
    --output pupitre.duckdb \
    "srv-data-install/M9/*.txt"

# Multiple housings in one pass (housing auto-detected from parent directory)
python pupitre_to_duckdb.py \
    --fields Field IH IB Pmagnet \
    --output pupitre.duckdb \
    "srv-data-install/M9/*.txt" "srv-data-install/M10/*.txt"

# Override housing when the parent directory name is not the magnet name
python pupitre_to_duckdb.py \
    --fields Field Pmagnet \
    --housing M9 \
    --output pupitre.duckdb \
    /tmp/flat_dir/*.txt
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `PATTERN` | (required) | File paths or glob patterns (e.g. `"M9/*.txt"`); expanded inside the script so quoted patterns always work |
| `--fields` | (required) | Data columns to extract; `timestamp` is always included |
| `--output` | `pupitre.duckdb` | DuckDB output file (dedicated to pupitre data) |
| `--housing` | auto | Housing name override (e.g. `M9`, `M10`); default: detected from the parent directory of each file |

**Notes:**
- Timestamps are stored in naive UTC (converted from local Paris time by `MagnetRun.fromtxt`).
- Fields requested but absent in a given file are inserted as `NULL` for that file's rows and a warning is logged.
- Re-running the script on the same files is safe: existing rows are skipped (`INSERT OR IGNORE` on the `timestamp` primary key).
- The DuckDB schema is: `timestamp TIMESTAMP NOT NULL PRIMARY KEY, <field1> DOUBLE, <field2> DOUBLE, …`

---

## Record Selection, Plotting and Statistics

### `get-record.py`

Multi-purpose script to select, plot, and compute statistics over a set of pupitre `.txt` records. Records are sorted by timestamp before processing.

**Subcommands:**

#### `select` — Filter records by duration and field threshold

```bash
python -m python_magnetrun.examples.get-record srvdata/M8*.txt select --duration 60 --field 18.
```

| Option | Default | Description |
|---|---|---|
| `--duration` | 60 s | Minimum record duration |
| `--field` | 18 T | Minimum magnetic field threshold |

#### `plot` — Plot field(s) vs time (or another x-axis) over multiple records

```bash
# Plot teb vs timestamp for all M8 records
python -m python_magnetrun.examples.get-record srvdata/M8*.txt plot --xfield timestamp --fields teb --show
```

| Option | Default | Description |
|---|---|---|
| `--xfield` | `timestamp` | X-axis field |
| `--fields` | — | Y-axis field(s) to plot |
| `--show` | — | Display plot interactively |
| `--save` | — | Save plot as PNG |

#### `aggregate` — Concatenate a field across records and plot monthly trends

```bash
python -m python_magnetrun.examples.get-record srvdata/M*---*.txt aggregate --fields teb --show
```

Saves a CSV named `aggregate-<fields>.csv` and a seaborn per-month/per-year line plot.

| Option | Description |
|---|---|
| `--fields` | Fields to aggregate |
| `--name` | Base name for output files |
| `--show` / `--save` | Display / save plots |

#### `stats` — Compute statistics and correlations

```bash
python -m python_magnetrun.examples.get-record srvdata/M8*.txt stats --fields teb --pearson --show
```

| Option | Description |
|---|---|
| `--fields` | Fields to analyse |
| `--pearson` | Compute Pearson correlation matrix |
| `--pairplot` | Generate seaborn pair-plot |
| `--tlcc` | Time-lagged cross-correlation |
| `--dtw` | Dynamic time-warping correlation |
| `--show` / `--save` | Display / save plots |

---

## Data Inspection and Downsampling Quality

### `field_meta_example.py`

Loads a single pupitre (`.txt`) or pigbrother (`.tdms`) file via `load_mrun`, then:

1. Prints symbol, unit, label, and description for every field read from the file.
2. Lists the fields belonging to each group.
3. Shows the first rows of one chosen data column.
4. Applies a downsampling algorithm to that column and reports reconstruction-quality metrics (RMSE, MAE, max error, MAPE, Hausdorff distance, energy ratio, elapsed time).

**Usage:**
```bash
# Pupitre — default key, default stride downsampling at 10 % of signal length
python field_meta_example.py data/2025.11.05\ -\ 09:53:00.txt --housing M8

# Pigbrother TDMS — explicit key, LTTB method, 500 output points
python field_meta_example.py data/M8_Overview_251105-0949.tdms --housing M8 \
    --key Courants_Alimentations/Champ_magn --method lttb --n-out 500

# minmax with custom bucket size
python field_meta_example.py data/2025.11.05\ -\ 09:53:00.txt --housing M8 \
    --method minmax --n-out 200 --bucket-size 10

# rdp with explicit epsilon
python field_meta_example.py data/2025.11.05\ -\ 09:53:00.txt --housing M8 \
    --method rdp --n-out 300 --epsilon 0.01

# rdp with auto-searched epsilon (from_n_out_rdp)
python field_meta_example.py data/2025.11.05\ -\ 09:53:00.txt --housing M8 \
    --method rdp --n-out 300
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `file` | — | Path to a pupitre `.txt` or pigbrother `.tdms` file |
| `--housing` | `unknown` | Housing name, e.g. `M8`, `M9` |
| `--key` | first meaningful field | Field key to preview and downsample |
| `--method` | `stride` | Downsampling algorithm: `stride`, `lttb`, `minmax_lttb`, `m4`, `nan_m4`, `minmax`, `rdp`, `vw` |
| `--n-out` | 10 % of signal | Target number of output points |
| `--epsilon` | auto-searched | Geometry tolerance for `rdp`/`vw`; when omitted, auto-searched via `from_n_out_rdp` |
| `--bucket-size` | auto-computed | Bucket size for the `minmax` method |
| `--debug` | off | Enable debug logging |

**Optional dependencies:**

| Method | Extra required |
|---|---|
| `lttb`, `minmax_lttb`, `m4`, `nan_m4` | `pip install python_magnetrun[hybrid]` (`tsdownsample`) |
| `rdp`, `vw` | `pip install python_magnetrun[rdp]` (`simplification`) |

---

### `field_correspondence_example.py`

Prints the pupitre <-> pigbrother field correspondence for a given housing, combining
housing-independent aliases (`*-defs.json`) with housing-dependent GR role assignments
(`housing_config.py`). No data files needed — the correspondence is a property of the
configuration, not of any particular run.

**Usage:**
```bash
python field_correspondence_example.py --housing M9
python field_correspondence_example.py --housing M8
python field_correspondence_example.py --housing M10
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--housing` | `M9` | Housing name, e.g. `M8`, `M9`, `M10` |

**Output:**

1. Fixed field aliases (housing-independent) — e.g. `Idcct1` <-> `Courants_Alimentations/Courant_A1`.
2. Housing-dependent role assignments — current, voltage, flow, rpm, and inlet-pressure roles
   for GR1/GR2, showing e.g. that GR1 current is `IH` on M9 but `IB` on M8/M10. Flow/rpm/pressure
   have no pigbrother counterpart.
3. Per-probe voltage channels by GR — `Ucoil1`..`Ucoil16` grouped by GR1/GR2 for the housing,
   cross-referenced with their fixed `Tensions_Aimant/Interne*`/`Externe*` pigbrother alias.

---

### `field_comparison_demo.py`

Demonstrates `python_magnetrun.analysis.field_comparison`: discovers pupitre <-> pigbrother
aliased fields, computes a single reference lag per source (`Idcct1`/`Courant_A1`, falling
back to `Idcct3`/`Courant_A3`) and compares two fields against it, then benchmarks the two
lag algorithms in `analysis.synchronization` — `compute_lag` (fixed 1 s resample) vs.
`compute_lag_interpolated` (common fine grid) — for timing and accuracy against a known
injected lag. Uses synthetic data throughout, so no real Overview/Archive/pupitre files
are needed.

**Usage:**
```bash
# Discovery + comparison + benchmark table (no plots, fast)
python field_comparison_demo.py

# More timing repetitions / a different RNG seed for the irregular sampling
python field_comparison_demo.py --repeat 10 --seed 7

# With plots (off by default so the script stays fast/non-interactive)
python field_comparison_demo.py --show
python field_comparison_demo.py --output-dir ./plots
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--repeat` | `5` | Timing repetitions per lag method |
| `--seed` | `0` | RNG seed for the irregular pupitre-like sampling |
| `--show` | off | Display plots interactively (requires a display) |
| `--output-dir` | — | Directory to save plots as PNG (created if missing) |

**Output:**

- Console: discovered field count, reference lag, per-field comparison summary
  (correlation, MAPE, Euclidean), and a benchmark table (time, recovered lag,
  absolute error) for both an Overview-like (1 Hz) and an Archive-like (120 Hz)
  scenario.
- With `--show`/`--output-dir`: two field-comparison overlay plots plus one
  benchmark chart (`lag_benchmark.png`) with time and log-scaled accuracy panels.

---

## User Database Integration

### `userdb.py`

Queries the `proposals-for-ct` REST API of the MagnetDB user database to retrieve and export experimental proposals.

**Environment variables:**
| Variable | Description |
|---|---|
| `USERDB_SERVER` | API server hostname/IP (default: `147.173.81.141`) |
| `USERDB_API_KEY` | Bearer token for authentication |

**Usage:**
```bash
# Fetch the first 20 proposals (JSON output)
python userdb.py --limit 20

# Export all proposals to CSV
python userdb.py --command export --output proposals.csv

# Use a specific server and token
python userdb.py --server my.server.example --token <token> --limit 5
```

**Arguments:**
| Argument | Default | Description |
|---|---|---|
| `--server` | `$USERDB_SERVER` | API server address |
| `--token` | `$USERDB_API_KEY` | Bearer token |
| `--command` | `get-proposals` | `get-proposals` or `export` |
| `--output` | `proposals.csv` | Output file for export |
| `--limit` | 10 | Number of proposals to fetch |

---

### `proposal.py`

Demonstrator that links experimental proposals from the user database to local pupitre records, then performs per-project statistics and plateau detection.

**Usage:**
```bash
python proposal.py proposals.csv --mdatadir srvdata --show
```

**Arguments:**
| Argument | Default | Description |
|---|---|---|
| `csvfile` | — | Path to proposals CSV file (e.g. exported by `userdb.py`) |
| `--mdatadir` | `srvdata` | Directory containing pupitre `.txt` record files |
| `--show` | — | Display plots (requires X11) |
| `--save` | — | Save plots as PNG |
| `--thresold` | `1e-3` | Field threshold for plateau detection |
| `--bthresold` | `1e-3` | Secondary threshold for plateau detection |
| `--dthresold` | `10` | Minimum plateau duration |
| `--window` | `10` | Window size for plateau detection |

**What it does:**
1. Reads and anonymises the proposals CSV (removes user/affiliation columns).
2. Selects experiments done in Grenoble with status `Done` or `InProgress`.
3. Matches each proposal's time window to pupitre record files.
4. Computes per-record statistics and detects current plateaux.
5. Exports `anonymized_proposals.csv` and `project_records.csv`.

---

### `mysql_connect.py`

Connects DuckDB to a remote MySQL server to inspect, export, or live-plot measurement data.
Four modes are available: `live` (schema inspection), `export` (table copy), `view` (rich terminal table), `plot` (one-shot chart), and `poll` (live chart).

Connection parameters can be supplied as CLI flags or via environment variables (`MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DB`).

**Usage:**
```bash
# Inspect schema — print all tables and column types
python mysql_connect.py --mode live \
    --host myhost --user myuser --password mypw --database mydb

# Export all tables to CSV
python mysql_connect.py --mode export --format csv --output-dir ./out \
    --host myhost --user myuser --password mypw --database mydb

# Export only selected tables to a DuckDB file
python mysql_connect.py --mode export --format duckdb \
    --output magnetdb_mysql.duckdb --tables sites magnets \
    --host myhost --user myuser --password mypw --database mydb

# View the most recent 50 rows of a table in the terminal
python mysql_connect.py --mode view --table measurements --limit 50 \
    --host myhost --user myuser --password mypw --database mydb

# One-shot static chart (matplotlib)
python mysql_connect.py --mode plot --table measurements \
    --fields timestamp Icoil Ucoil --x-field timestamp \
    --host myhost --user myuser --password mypw --database mydb

# Live-updating chart, polling every 10 s (matplotlib)
python mysql_connect.py --mode poll --table measurements \
    --fields timestamp Icoil Ucoil --x-field timestamp \
    --interval 10 --limit 200 --plot matplotlib \
    --host myhost --user myuser --password mypw --database mydb

# Two-source poll: measurements (Icoil, Ucoil) and temperatures (tsb, teb)
# in two subplots sharing the same x-axis — matplotlib backend
python mysql_connect.py --mode poll \
    --table measurements --fields timestamp Icoil Ucoil --x-field timestamp \
    --table2 temperatures --fields2 tsb teb \
    --interval 10 --plot matplotlib \
    --host myhost --user myuser --password mypw --database mydb

# Two-source poll using raw SQL queries — Dash interactive web app
python mysql_connect.py --mode poll \
    --query "SELECT t, Icoil, Ucoil FROM mysqldb.measurements ORDER BY t LIMIT 500" \
    --fields Icoil Ucoil --x-field t \
    --query2 "SELECT t, tsb, teb FROM mysqldb.temperatures ORDER BY t LIMIT 500" \
    --fields2 tsb teb \
    --interval 10 --plot dash \
    --host myhost --user myuser --password mypw --database mydb
```

> **Two-source mode** (`--table2` / `--query2`): adds a second independent data source
> displayed in a separate subplot sharing the same x-axis as the first source.
> Supported backends: `matplotlib`, `plotly`, `dash` (not `table` or `textual`).
> `--table2` and `--query2` are mutually exclusive.
> Use `--fields2` to select which columns to plot from the second source, and
> `--where2` to filter it (applies to `--table2` only).

> **Note — selecting tables for export:**  
> Use `--tables` to restrict the export to a subset of tables.
> Multiple table names can be given as a space-separated list:
> ```bash
> python mysql_connect.py --mode export --format duckdb \
>     --output magnetdb_mysql.duckdb --tables sites magnets records \
>     --host myhost --user myuser --password mypw --database mydb
> ```
> When `--tables` is omitted all tables in the database are exported.

**Export arguments:**

| Argument | Default | Description |
|---|---|---|
| `--format` | `csv` | Output format: `csv`, `parquet`, `duckdb`, or `excel` (requires `pandas openpyxl`) |
| `--output` | `magnetdb_mysql.duckdb` | Output file for `--format duckdb` |
| `--output-dir` | `.` | Output directory for CSV/Parquet files |
| `--tables` | all | Subset of table names to export |
| `--export-fields` | all | Columns to include (single-table export only) |
| `--time-field` | auto | TIMESTAMP column for `--start`/`--end` filtering |
| `--start` | — | Start of time range, ISO 8601 |
| `--end` | — | End of time range, ISO 8601 |

**Poll/plot arguments:**

| Argument | Default | Description |
|---|---|---|
| `--table` | — | MySQL table to query |
| `--query` | — | Raw SELECT query (mutually exclusive with `--table`) |
| `--fields` | all numeric | Columns to plot on the y-axis |
| `--x-field` | auto | Column to use as x-axis (auto-selects first TIMESTAMP column) |
| `--where` | — | SQL WHERE filter |
| `--limit` | 200 (poll) / 0 (plot) | Max rows fetched; `0` = no cap |
| `--interval` | 5 s | Seconds between polls |
| `--plot` | `matplotlib` | Backend: `matplotlib`, `plotly`, `dash`, `textual`, `table` |
| `--plot-options` | — | JSON object with style options (`type`, `layout`, `colors`, `font`, …) |
| `--table2` | — | Second MySQL table for two-source mode (mutually exclusive with `--query2`); supported by `matplotlib`, `plotly`, `dash` |
| `--query2` | — | Raw SELECT for the second subplot (mutually exclusive with `--table2`) |
| `--fields2` | all numeric | Columns to plot from the second source |
| `--where2` | — | SQL WHERE filter for `--table2` only |

### `mysql_csv_to_magnetdata.py`

Demonstrates the round-trip from a MySQL CSV export to a
:class:`~python_magnetrun.magnetdata_pandas.PandasMagnetData` object.

Two subcommands:

| Subcommand | Description |
|---|---|
| `generate` | Create a synthetic DuckDB-style CSV (no MySQL needed) then load and inspect it |
| `load` | Load any CSV produced by `mysql_connect.py --mode export` |

```bash
# Generate a 300-row synthetic CSV, load it, and plot
python mysql_csv_to_magnetdata.py generate --output sample.csv --plot

# Load a real export and inspect all columns
python mysql_csv_to_magnetdata.py load measurements.csv

# Load, parse a specific timestamp column, plot two fields
python mysql_csv_to_magnetdata.py load measurements.csv \
    --timestamp-col timestamp --fields Icoil Ucoil --plot
```

> **Note — TIMESTAMP columns:** DuckDB serialises MySQL `TIMESTAMP`/`DATETIME`
> columns as ISO 8601 strings (`"2024-03-15 08:00:00"`).  `CsvReader` loads
> them as `object` dtype.  Pass `--timestamp-col <name>` (or the column name
> is auto-detected) to have the script call `pd.to_datetime()` before plotting.

---

## External Weather Data

### `openmeteo_temperature.py`

Fetches current or historical hourly temperature at the machine's location using the
[Open-Meteo](https://open-meteo.com/) API (no API key required). Location is determined
automatically from the machine's public IP via `geocoder`.

**Usage:**
```bash
# Current temperature
python openmeteo_temperature.py

# Historical — plain table
python openmeteo_temperature.py --start 2026-06-27 --end 2026-06-28

# Historical — styled Rich table
python openmeteo_temperature.py --start 2026-06-27 --end 2026-06-28 --table

# Export to CSV (default format)
python openmeteo_temperature.py --start 2026-06-27 --end 2026-06-28 --output temps.csv

# Export to Excel
python openmeteo_temperature.py --start 2026-06-27 --end 2026-06-28 --output temps.xlsx --format xlsx

# Export to DuckDB
python openmeteo_temperature.py --start 2026-06-27 --end 2026-06-28 --output temps.duckdb --format duckdb

# Overwrite an existing output file
python openmeteo_temperature.py --start 2026-06-27 --end 2026-06-28 --output temps.csv --force

# Plot temperature time series
python openmeteo_temperature.py --start 2026-06-27 --end 2026-06-28 --plot

# Combine: Rich table + export + plot
python openmeteo_temperature.py --start 2026-06-27 --end 2026-06-28 \
    --table --output temps.xlsx --format xlsx --plot
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--start` | — | Start date for historical range (`YYYY-MM-DD`); requires `--end` |
| `--end` | — | End date for historical range (`YYYY-MM-DD`); requires `--start` |
| `--table` | off | Display as a colour-coded Rich table (requires `--start`/`--end`) |
| `--output` | — | Destination file path for export (requires `--start`/`--end`) |
| `--format` | `csv` | Output format: `csv`, `xlsx`, or `duckdb` |
| `--force` | off | Overwrite output file if it already exists (requires `--output`) |
| `--plot` | off | Plot hourly temperature as a time series (requires `--start`/`--end`) |

**Dependencies:** `geocoder`, `requests`, `rich`, `matplotlib`, `pandas`,
`openpyxl` (xlsx export), `duckdb` (duckdb export).

---

## Run Analysis Examples

### `bilan.py`

Computes an energy balance for a magnet run from a pigbrother (TDMS) file. Combines electrical power from current/voltage measurements with cooling water thermal power (using `python_magnetcooling` fluid properties).

**Usage:**
```bash
python bilan.py <input_tdms_file> [--show] [--pigbrother_datadir DIR] [--pupitre_datadir DIR] [--debug]
```

**Dependencies:** `python_magnetcooling`

---

### `outliers.py`

Detects and visualises outliers in time-series data across multiple pupitre records using MAD (Median Absolute Deviation) and mean-MAD statistics.

**Usage:**
```bash
python outliers.py <file1.txt> [file2.txt ...] --site M9 [--insert NAME] [--plot] [--save] [--debug]
```

---

### `corr_Ih_Ib.py`

Plots and fits the relationship between two current channels (by default `IH` vs `IB`) across one or more pupitre records using piecewise-linear regression (`pwlf`).

**Usage:**
```bash
python corr_Ih_Ib.py <file1.txt> [file2.txt ...] [--xkey IH] [--ykey IB] [--breakpoints 1] [--show] [--save] [--debug]
```

**Dependencies:** `pwlf`

---

### `cmp_fields.py`

Compares two time-series from pupitre records: computes Euclidean distance, MAPE and Pearson correlation, and optionally overlays a LOWESS smoothed fit.

**Usage:**
```bash
python cmp_fields.py <file1.txt> [file2.txt ...] --key1 Field --key2 IH [--show] [--save] [--debug]
```

---

### `pupitre.py`

Demonstrates lag-correlation and trend analysis on pupitre records. Computes rolling statistics and trend lines for a set of physical quantities.

**Usage:**
```bash
python pupitre.py <file1.txt> [file2.txt ...] [--window 10] [--show] [--save] [--debug]
```

---

### `timeseries-anomaly-detection.py`

Demonstrates multiple anomaly-detection algorithms (Z-score, IQR, rolling statistics, Isolation Forest) on a time series extracted from a pupitre record.

**Usage:**
```bash
python timeseries-anomaly-detection.py <file.txt> --key Field [--show] [--debug]
```

**Dependencies:** `scikit-learn`

---

## Hybrid Sub-system Plot Scripts

### `plot_fepc_data.py`

Reads FEPC binary (`.bin`) kHz data files and plots a specific variable for one or more time slots. Supports calibration files (`.cnv`) and outlier removal.

**Usage:**
```bash
python plot_fepc_data.py -c HOST_2_DATA.CFG -v ALIM1_J1 -s 4 [--date-range START END] [--no-calib] [--save FILE]
```

---

### `plot_rms.py`

Quickly plots one or more variables from an RMS data file.

**Usage:**
```bash
python plot_rms.py <file.rms> VAR1 [VAR2 ...] [--output FILE] [--same-plot]
```

---

### `plot_trigger_data.py`

Plots trigger data from FEPC trigger binary files. Supports single-trigger and all-triggers-for-a-date modes, optional calibration, and figure export.

**Usage:**
```bash
# Plot a single trigger directory
python plot_trigger_data.py --trigger-dir /data/trigger/2025-01-06T10:00:00 --variable I_H1

# Plot all triggers for a date
python plot_trigger_data.py --base-dir /data --date 2025-01-06 --system FEPC-LNCMI --variable I_H1 --all
```

---

### `plot_vprocess.py`

Plots variables from VProcess (`.vprocess`) slow-data files. Supports individual variable plots, overview dashboards, and side-by-side comparisons.

**Usage:**
```bash
python plot_vprocess.py data.vprocess --vars TT115A TT508A
python plot_vprocess.py data.vprocess --overview
python plot_vprocess.py data.vprocess --compare TT115A TT508A
```

---

## Cooling / Hydraulic Examples

### `flow_params_pipeline.py`

Standalone demonstration of the complete flow parameter extraction pipeline using synthetic data:

1. Generates synthetic pump speed, flow rate, and pressure curves.
2. Fits the curves with polynomial regression.
3. Builds a `flow_params` dictionary.
4. Creates a `WaterFlow` object via `waterflow_factory`.
5. Performs hydraulic calculations.

**Usage:**
```bash
python flow_params_pipeline.py
```

Requires `python_magnetcooling` to be installed or on `PYTHONPATH`.

---

### `flow_params_magnetrun_pipeline.py`

Same pipeline as above but uses `python_magnetrun` fitting utilities and piecewise-linear fitting (`pwlf`) for the pump speed curve with automatic `Imax` detection.

**Usage:**
```bash
python flow_params_magnetrun_pipeline.py
```

**Additional dependencies:** `pwlf`, `sympy`, `tabulate`.

---

### `waterflow_debitbrut_example.py`

Demonstrates the `debitbrut()` method of `WaterFlow` for computing secondary cooling loop flow rates from power, including hysteresis modelling.

**Usage:**
```bash
python waterflow_debitbrut_example.py
```

Requires `python_magnetcooling` to be installed or on `PYTHONPATH`.

---

## Hybrid Data Plotting Examples

### 1. Minimal Example: `plot_hybrid_minimal.py`

A simplified example showing the core concepts of loading and plotting hybrid kHz data alongside pupitre and TDMS data.

**Features:**
- Simple, easy-to-understand code
- Hardcoded configuration (edit the script to customize)
- Demonstrates field name mapping using a dictionary
- Shows date-based file discovery

**Usage:**
```bash
# Edit the script to set your data directories and preferences
python plot_hybrid_minimal.py
```

**Good for:** Learning the basics, quick prototyping

### 2. Full-Featured Example: `plot_hybrid_with_pupitre_tdms.py`

A complete command-line tool with extensive options and error handling.

**Features:**
- Full command-line argument parsing
- Configurable data directories
- Hour-based filtering (comma list or colon range)
- Signal normalization
- Downsampling (stride, minmax_lttb, and others)
- Save or show plots (mutually exclusive)
- Comprehensive error handling and structured logging
- Automatic file discovery

**Usage:**
```bash
# Basic usage
python plot_hybrid_with_pupitre_tdms.py \
    -d 2025-01-27 \
    -s FEPC-AUX-LNCMI \
    -k ALIM1_J1 \
    --housing M8 \
    --show

# With custom directories
python plot_hybrid_with_pupitre_tdms.py \
    -d 2025-01-27 \
    -s FEPC-AUX-LNCMI \
    -k ALIM1_J1 \
    --housing M8 \
    --hybrid-dir /path/to/hybrid/data \
    --pupitre_datadir /path/to/pupitre \
    --pigbrother_datadir /path/to/pigbrother \
    --hours 10,11,12 \
    --save output.png

# Filter hours with colon range notation (10:13 means hours 10, 11, 12)
python plot_hybrid_with_pupitre_tdms.py \
    -d 2025-01-27 \
    -s FEPC-AUX-LNCMI \
    -k ALIM1_J1 \
    --housing M8 \
    --hours 10:13

# Normalize signals for shape comparison
python plot_hybrid_with_pupitre_tdms.py \
    -d 2025-01-27 \
    -s FEPC-AUX-LNCMI \
    -k ALIM1_J1 \
    --housing M8 \
    --normalize --show

# Downsample to 5000 points using stride (fast, no extra dependency)
python plot_hybrid_with_pupitre_tdms.py \
    -d 2025-01-27 \
    -s FEPC-AUX-LNCMI \
    -k ALIM1_J1 \
    --housing M8 \
    --downsample-method stride --downsample-params '{"n_out": 5000}' --show

# Downsample using minmax_lttb (requires tsdownsample)
python plot_hybrid_with_pupitre_tdms.py \
    -d 2025-01-27 \
    -s FEPC-AUX-LNCMI \
    -k ALIM1_J1 \
    --housing M8 \
    --downsample-method minmax_lttb --downsample-params '{"n_out": 10000}' --show
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `-d`/`--date` | — | Date in `YYYY-MM-DD` format (required) |
| `-s`/`--fepc-system` | — | FEPC system: `FEPC-LNCMI` or `FEPC-AUX-LNCMI` (required) |
| `-k`/`--key` | — | Variable name to plot, e.g. `ALIM1_J1`, `I_H1` (required) |
| `--housing` | `M8` | Housing name (M8, M9, M10, …) |
| `--insert` | `notdefined` | Insert name (e.g. `M25032101`) |
| `--hybrid-dir` | `$MAGNETRUN_HYBRID_DATA_DIR` | Base directory for hybrid kHz/rms/trigger data |
| `--pupitre_datadir` | configured default | Directory containing pupitre `.txt` files |
| `--pigbrother_datadir` | configured default | Directory containing TDMS pigbrother files |
| `--hours` | all hours | Hours to select: comma list `10,11,12` or colon range `10:13` (stop excluded) |
| `--normalize` | off | Normalize each signal by its maximum absolute value before plotting |
| `--downsample-method` | `none` | Downsampling method: `stride`, `minmax_lttb`, `lttb`, `none` |
| `--downsample-params` | `{}` | JSON params for the chosen method, e.g. `'{"n_out": 5000}'` |
| `--save FILE` | — | Save plot to FILE; mutually exclusive with `--show` |
| `--show` | — | Display plot interactively; mutually exclusive with `--save` |
| `--log-level` | `WARNING` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `--log-file` | console | Path to log file |

**Good for:** Production use, automation, detailed analysis

See [README_hybrid_plotting.md](README_hybrid_plotting.md) for detailed documentation.

## Interactive Visualization

### `pywry_demo.py`

Opens a MagnetRun data file in a native desktop window powered by
[pywry](https://github.com/OpenBB-finance/pywry) and Plotly.  No browser tab
is required — the chart renders inside a lightweight OS webview.  All standard
Plotly interactions are available: zoom, pan, hover tooltip, and legend-click
to show/hide individual traces.

If pywry is not installed the figure is saved to a temporary HTML file and
opened in the default browser instead (useful for a quick preview).

**Requirements:**
```bash
pip install python_magnetrun[gui]
# Linux also needs the WebKit2GTK system package:
apt install python3-gi gir1.2-webkit2-4.1
```

**Usage:**
```bash
# Bundled M9 sample (no arguments needed)
python pywry_demo.py

# Pupitre .txt file
python pywry_demo.py data/M9_2019.02.14---23_00_38.txt --housing M9

# Pigbrother TDMS file
python pywry_demo.py overview.tdms --housing M10

# Restrict to a subset of channels
python pywry_demo.py run.txt --housing M9 --keys Field Courant_A1 Debit

# Use wall-clock timestamps on the x-axis instead of elapsed seconds
python pywry_demo.py run.txt --housing M9 --time-col timestamp

# Adjust the window size
python pywry_demo.py run.txt --housing M9 --width 1600 --height 900
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `file` | bundled M9 sample | Path to a `.txt`, `.tdms`, or `.csv` data file |
| `--housing` | `M9` | Housing name, e.g. `M9`, `M10` |
| `--keys` | all channels | One or more channel names to display |
| `--time-col` | auto-detect | X-axis column: `t` (elapsed seconds) or `timestamp` (wall clock). Auto-detect picks `timestamp` if present, otherwise `t`. |
| `--width` | `1400` | Window width in pixels |
| `--height` | `780` | Window height in pixels |

**Public API** (importable from other scripts):

```python
from pywry_demo import build_figure, show

# Build a go.Figure and do something with it
fig = build_figure(mrun, keys=["Field", "Courant_A1"], time_col="t")
fig.write_html("output.html")

# Or open the native window directly
show("my_run.txt", housing="M9", keys=["Field"], time_col="timestamp")
```

---

## Performance Benchmarking

### `benchmark_loading.py`

Measures wall-clock load time for magnet data files across all file types.
For each TDMS overview file supplied, [`FileDiscovery`](../python_magnetrun/analysis/loaders.py)
discovers the related archive, pupitre, and incident files automatically; standalone `.txt` and
`.csv` inputs are benchmarked directly.  Results are printed as a per-category statistics table
and optionally saved as a figure (box plot / scatter / bar).

**Usage:**
```bash
# Benchmark all files discovered from one overview (uses default data dirs)
python benchmark_loading.py M8_Overview_251105-0949.tdms --housing M8 --show

# Benchmark two overviews and write figure + CSV
python benchmark_loading.py M8_Overview_25*.tdms --housing M8 \
    --save --output-dir results

# Mix of TDMS overviews and standalone pupitre files
python benchmark_loading.py M8_Overview_25*.tdms 2025*.txt \
    --housing M8 --pigbrother_datadir /data/pbsurv --pupitre_datadir /data/pupitre --show

# Skip file discovery — benchmark only the supplied files
python benchmark_loading.py M8_Overview_251105-0949.tdms --housing M8 --no-discovery

# Repeat each load 3 times for more stable timing
python benchmark_loading.py M8_Overview_25*.tdms --housing M8 --repeat 3 --show
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `input_file` | — | One or more `.tdms`, `.txt`, or `.csv` files (globs expanded by the shell) |
| `--housing` | `notdefined` | Housing name (e.g. `M8`, `M9`, `M10`); forwarded to the loader and discovery |
| `--pigbrother_datadir` | configured default | Root directory for TDMS files |
| `--pupitre_datadir` | configured default | Root directory for pupitre `.txt` files |
| `--no-discovery` | off | Benchmark only the supplied input files; skip `FileDiscovery` |
| `--repeat N` | `1` | Number of load repetitions per file for stable timing |
| `--output-dir DIR` | `.` | Directory for saved figure (`benchmark_loading.png`) and CSV (`benchmark_loading.csv`) |
| `--show` | — | Display figure interactively (requires X11); mutually exclusive with `--save` |
| `--save` | — | Save figure and raw CSV to `--output-dir`; mutually exclusive with `--show` |
| `--log-level` | `WARNING` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

**Output:**

- Console: per-category table with count, mean/max file size [MB], min/mean/max load time [s], median row count.
- Figure (3 panels): load-time box plot per category · load time vs file size scatter · median row count bar.
- `benchmark_loading.csv`: raw per-file results (with `--save`).

---

## Key Concepts Demonstrated

### 1. Field Name Mapping

Different data sources use different field names for the same physical quantity. Use dictionaries to map between them:

```python
FIELD_MAPPING = {
    # hybrid_key -> (pupitre_field, tdms_field)
    "kHz/FEPC-AUX-LNCMI/ALIM1_J1": ("IH", "Référence_GR1"),
    "kHz/FEPC-LNCMI/I_H1": ("IH", "Référence_GR1"),
}
```

### 2. Date-Based File Discovery

Find corresponding files across different data sources using the date:

```python
# Pupitre files: M10/2025.01.27---*.txt
pupitre_pattern = f"{pupitre_dir}/{site}/{year}.{month:02d}.{day:02d}*.txt"

# TDMS files: M10/Overview/M10_Overview_250127-*.tdms
tdms_pattern = f"{pigbrother_dir}/{site}/Overview/{site}_Overview_{yy:02d}{mm:02d}{dd:02d}-*.tdms"
```

### 3. Loading Different Data Sources

```python
from python_magnetrun.hybrid.hybrid_run import HybridRun
from python_magnetrun.MagnetRun import load_mrun

# Hybrid data
hrun = HybridRun.fromdir(base_dir, date_str, fepc_system=fepc_system, housing=housing)

# Pupitre (.txt) and TDMS (.tdms) data — load_mrun dispatches by extension
pupitre = load_mrun(pupitre_file, housing=housing, assembly=insert)
tdms    = load_mrun(tdms_file,    housing=housing, assembly=insert)
```

### 4. Data Access and Plotting

```python
from python_magnetrun.utils.downsampling import DownsampleConfig

# Hybrid: returns (data_array, time_array); pass a DownsampleConfig or int for n_out
downsample = DownsampleConfig(method="stride", n_out=5000)
data, time = hrun.getData(hybrid_key, hours=[10, 11, 12], downsample=downsample)

# Pupitre and TDMS: getData returns a pandas DataFrame
mdata = pupitre.getMData()
df = mdata.getData(["t", pupitre_field], downsample=downsample)
pupitre_time   = df["t"].to_numpy()
pupitre_values = df[pupitre_field].to_numpy()
```

## Directory Structure Expected

```
workspace/
├── hybrid_data/                    # Hybrid kHz/RMS/Trigger data
│   ├── kHz/
│   │   └── YYYY-MM-DD/
│   │       ├── FEPC-LNCMI/
│   │       └── FEPC-AUX-LNCMI/
│   ├── rms/
│   └── trigger/
│
├── pupitre_datadir/                # Pupitre text files
│   ├── M9/
│   │   └── YYYY.MM.DD---HH:MM:SS.txt
│   ├── M10/
│   └── M8/
│
└── pigbrother_datadir/             # TDMS files
    └── Fichiers_Data/
        ├── M9/
        │   └── Overview/
        │       └── M9_Overview_YYMMDD-HHMM.tdms
        ├── M10/
        └── M8/
```

## Configuration

Default data directories are defined in `python_magnetrun/analysis/config.py`:

```python
DEFAULT_DATA_DIR = "/home/LNCMI-G/christophe.trophime/LNCMIG-Data/srv-data-install"
DEFAULT_PIGBROTHER_DATA_DIR = "/home/.../pigbrotherdata/Fichiers_Data"
```

You can override these using command-line arguments or by editing the scripts.

## Site-Specific Configurations

Different sites (M8, M9, M10) have different channel naming conventions:

| Site | GR1 Current | GR2 Current | Note            |
| ---- | ----------- | ----------- | --------------- |
| M9   | IH          | IB          | H=High, B=Bas   |
| M10  | IB          | IH          | Swapped from M9 |
| M8   | IB          | IH          | Same as M10     |

See `python_magnetrun/analysis/config.py` for complete site configurations.

## Troubleshooting

### Import Errors

Make sure the hybrid module is in your Python path:

```bash
export PYTHONPATH=/path/to/python_magnetrun:$PYTHONPATH
```

Or run from the repository root:

```bash
cd /path/to/python_magnetrun
python examples/plot_hybrid_minimal.py
```

### No Files Found

- Check your date format
- Verify data directories exist and contain files for the specified date
- Check site name matches directory structure (M9, M10, M8)

### Field Not Found

- Use `print(data.getMData().getKeys())` to see available fields
- Update the mapping dictionaries to match your actual field names
- Check site-specific configurations

## Further Reading

- [Hybrid Module Documentation](../python_magnetrun/hybrid/README.md)
- [Analysis Configuration](../python_magnetrun/analysis/config.py)
- [File Discovery](../python_magnetrun/analysis/loaders.py)
- [Main README](../README.md)

## Contributing

To add new examples:

1. Create a descriptive filename (e.g., `plot_xyz.py`)
2. Include docstrings and comments
3. Add usage examples to this README
4. Consider creating both minimal and full-featured versions

## License

Same as parent project.
