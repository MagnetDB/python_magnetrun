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

**Data directories searched** (hardcoded in the script, edit `BASE_DIR` as needed):
- `LNCMIG-Data/pbsurv/{housing}/` — TDMS files
- `LNCMIG-Data/srv-data-install/{housing}/` — pupitre `.txt` files
- `LNCMIG-Data/CEA/` — kHz, rms, vprocess, trigger (M8 only)

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
pupitre = load_mrun(pupitre_file, housing=housing, site=insert)
tdms    = load_mrun(tdms_file,    housing=housing, site=insert)
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
