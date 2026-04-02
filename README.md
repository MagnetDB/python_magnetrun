# Python `MagnetRun`

Python `MagnetRun` contains utilities to view and analyze Magnet runs from LNCMI control/monitoring systems (`Pupitre`, `PigBrother`, and hybrid FEPC acquisition systems).

- Free software: MIT license
- Documentation: <https://python-magnetrun.readthedocs.io>

---

## Table of Contents

- [Installation](#installation)
- [Data Sources](#data-sources)
- [Mounting Data Directories](#mounting-data-directories)
- [Features](#features)
- [Basic Usage](#basic-usage)
- [Analysis](#analysis)
- [Running Tests](#running-tests)
- [To-do](#to-do)
- [Credits](#credits)

---

## Installation

### Using a Python virtual environment

**Linux / macOS:**

```bash
python3 -m venv [--system-site-packages] magnetrun-env
source ./magnetrun-env/bin/activate
cd python_magnetcooling
python3 -m pip install -e ".[dev]"
cd ..
python3 -m pip install -e ".[dev]"
```

**Windows:**

```bat
C:\Python35\python -m venv C:\path\to\magnetrun-env
C:\path\to\magnetrun-env\Scripts\activate.bat
C:\Python35\python -m pip install -r requirements.txt
```

To leave the virtual environment, run `deactivate`.

### Devcontainer

A `.devcontainer` configuration is provided for VS Code / GitHub Codespaces.

---

## Data Sources

### `Pupitre`

`Pupitre` is the LNCMI magnet control/monitoring system. It records time-series data for each magnet run (currents, voltages, temperatures, water-flow rates, …) and saves them as plain-text (`.txt`) files named after the magnet site and the start timestamp, e.g.:

```
M9_2024.05.09---16_34_03.txt
```

Each file contains one row per sample and one column per measured quantity (field, IH, IB, UH, UB, TinH, TinB, FlowH, FlowB, …).

See [Mounting Data Directories](#mounting-data-directories) for how to access `Pupitre` data.

### `PigBrother`

`PigBrother` is a secondary surveillance/monitoring system that records data independently of `Pupitre`, typically at a higher sampling rate and with a wider set of voltage and current channels. Data are stored as National Instruments TDMS (`.tdms`) files.

Files are organised by magnet site and acquisition type under a base directory:

```
<pigbrother_datadir>/
  Fichiers_Data/
    <housing>/                  e.g. M9/, M10/
      Overview/              downsampled overview files (one per run)
        <housing>_Overview_YYMMDD-HHMM.tdms
      Fichiers_Archive/      full-rate archive files
        <housing>_Archive_YYMMDD-HHMM.tdms
      Fichiers_Incidents/    incident (fault) recordings
        <housing>_Incidents_YYMMDD-HHMM.tdms
      Fichiers_Spike/        spike / transient recordings
        <housing>_Spikes_YYMMDD-HHMM.tdms
```

See [Mounting Data Directories](#mounting-data-directories) for how to access `PigBrother` data.

### Hybrid data (kHz / RMS / Trigger)

Hybrid data from FEPC acquisition systems captures high-frequency (kHz) current waveforms, RMS summaries, and trigger events. It is stored as binary files under:

```
<hybrid_datadir>/
  kHz/
    FEPC-LNCMI/    or    FEPC-AUX-LNCMI/
      YYYY-MM-DD/
        ...
  rms/
    ...
  trigger/
    ...
```

Pass `--hybrid_datadir` and `--hybrid_date` to any `plot` subcommand to overlay hybrid data alongside Pupitre/PigBrother sources.

---

## Mounting Data Directories

### `Pupitre` — SSHFS

Mount the `Pupitre` file server as a local directory:

```bash
sshfs -o uid=$(id -u),gid=$(id -g) -o IdentityFile=~/.ssh/id_ecdsa \
    $SRVDATA_SERVER:$SRVDATA_DIR ~/LNCMIG-Data/
```

To unmount:

```bash
fusermount -u ~/LNCMIG-Data/
```

> [!NOTE]
> SSHFS can be unstable. If it stops, just relaunch the command above.

Alternatively, retrieve files programmatically without mounting:

```bash
python3 -m python_magnetrun.requests.cli --user email --datadir datadir [--save]
```

### `PigBrother` — CIFS

Create the target directories first, then mount:

```bash
mkdir -p pigbrotherdata pigbrothercolddata

# Main data share
sudo mount -v -t cifs //pigbrother_server_ip/d ./pigbrotherdata \
    -o user=pbsurv,password=passwd

# Cold data share
sudo mount -v -t cifs //pigbrother_server_ip/df ./pigbrothercolddata \
    -o user=pbsurv,password=passwd
```

To unmount:

```bash
sudo umount ./pigbrotherdata
sudo umount ./pigbrothercolddata
```

> [!NOTE]
> Replace `pigbrother_server_ip`, `pbsurv`, and `passwd` with the actual server address and credentials.

### Hybrid data — local path

Hybrid data does not require network mounting; simply point `--hybrid_datadir` to the local base directory that follows the layout described in [Data Sources → Hybrid data](#hybrid-data-khz--rms--trigger).

---

## Features

- Load `txt`, `csv`, and `tdms` files from control/monitoring systems
- Extract and list recorded fields
- Plot field(s) vs time or field vs field
- Cross-source comparison (Pupitre vs PigBrother vs hybrid kHz)
- Statistics and plateau detection
- Compute derived quantities via formulas (power, busbar losses, etc.)
- Breakpoint detection and run signature (UPD)
- Anomaly detection (Z-score, IQR, rolling mean/std, DBSCAN, MAD)
- Piecewise linear regression (`piecewise_regression`, `pwlf`)
- Field factor identification via OLS regression
- Time-series synchronization between data sources
- Distance and similarity metrics (Euclidean, MAE, MAPE, DTW, TLCC)
- Extract data from `srv-data-lncmi `
- Prepare data for injection into `magnetdb`

---

## Basic Usage

### List available fields

```bash
python3 -m python_magnetrun.cli \
    srvdata/M9_2019.02.14---23:00:38.txt info --list
```

### Select records by criteria

List records lasting at least 60 s with a magnetic field above 18 T:

```bash
python3 -m python_magnetrun.examples.get-record \
    srvdata/M8*.txt select --duration 60 --field 18.
```

### Plotting

Plot the magnetic field over time:

```bash
python3 -m python_magnetrun.cli \
    srvdata/M9_2019.02.14---23:00:38.txt \
    plot --vs_time "Field"
```

Compare current from PigBrother and Pupitre side by side:

```bash
python3 -m python_magnetrun.cli \
    srvdata/M10_2025.01.27---*.txt \
    pigbrotherdata/Fichiers_Data/M10/Overview/M10_Overview_250127-1605.tdms \
    plot \
    --key_vs_key timestamp-IH \
    --key_vs_key timestamp-Courants_Alimentations/Référence_GR1
```

Overlay PigBrother, Pupitre, and hybrid (kHz) currents for comparison:

```bash
python3 -m python_magnetrun.cli \
    ~/M9_Overview_240509-1634.tdms \
    ~/M9_2024.05.09---16_34_03.txt \
    --hybrid_datadir /path/to/hybrid \
    --hybrid_date 2024-05-09 \
    --fepc_system FEPC-LNCMI \
    plot \
    --vs_time Courants_Alimentations/Courant_GR1 \
    --vs_time IH \
    --vs_time_hybrid "kHz/FEPC-LNCMI/I_H1"
```

> [!NOTE]
> One `--vs_time` argument is required per input file extension type (`.tdms`, `.txt`).
> Hybrid keys are passed separately via `--vs_time_hybrid`.
> Use `--hybrid_downsample N` to control how many kHz points are rendered (default: 50000).
> Use `--fepc_system FEPC-AUX-LNCMI` for the auxiliary FEPC system.
> Field name correspondence between sources is defined in [python_magnetrun/field_mappings.py](python_magnetrun/field_mappings.py).

### Statistics and plateau detection

Compute statistics for all M8 records:

```bash
python3 -m python_magnetrun.cli srvdata/M8*.txt stats
```

Detect plateaux:

```bash
python3 -m python_magnetrun.cli srvdata/M8*.txt stats --plateau
```

Aggregate a specific field across records:

```bash
python3 -m python_magnetrun.examples.get-record \
    srvdata/M*---*.txt aggregate --fields teb --show
```

### Derived quantities

Compute and plot power dissipated in Helices from a Pupitre file:

```bash
python3 -m python_magnetrun.cli \
    srvdata/M10_2020.10.03---09:56:20.txt \
    add --formula "PowerH = IH * UH / 1.e+6" --plot
```

Compute power from a PigBrother TDMS file:

```bash
python3 -m python_magnetrun.cli \
    pigbrotherdata/Fichiers_Data/M10/Overview/M10_Overview_201003-0956.tdms \
    add --formula "Tensions_Aimant/Power_internes = Tensions_Aimant/ALL_internes * Courants_Alimentations/Courant_GR2 / 1.e+6" \
    --plot
```

---

## Analysis

The `python_magnetrun.analysis` module provides higher-level tools for cross-source analysis, synchronization, and metrics computation.

### CLI

Process one or more TDMS overview files:

```bash
python3 -m python_magnetrun.analysis.cli M9_Overview_*.tdms --show
```

With optional synchronization, lag computation, and distance metrics:

```bash
python3 -m python_magnetrun.analysis.cli input.tdms \
    --synchronize \
    --lag \
    --distance \
    --downsample 10 \
    --show --save \
    --debug --log-file analysis.log
```

Write structured JSON logs for pipeline integration:

```bash
python3 -m python_magnetrun.analysis.cli input.tdms \
    --json-log analysis.json --quiet
```

---

## Python API

### Configuration

```python
from python_magnetrun.analysis import AnalysisConfig

# Load pre-defined site configuration (M8, M9, M10)
config = AnalysisConfig.for_site("M9")

# Access site-specific channel mappings
print(config.site.reference_gr1_current)   # "IH"
print(config.site.reference_gr2_current)   # "IB"

# Get threshold for a channel
threshold = config.thresholds.get("Courant_GR1")
```

### Loading data

**Single Pupitre file (`.txt`):**

```python
from python_magnetrun.magnetdata import MagnetData

data = MagnetData.fromtxt("srvdata/M9_2024.05.09---16_34_03.txt")
print(data.Keys)          # list of available field names
df = data.Data            # pandas DataFrame
print(df[["t", "Field", "IH", "IB"]].head())
```

**Single PigBrother file (`.tdms`):**

```python
from python_magnetrun.magnetdata import MagnetData

data = MagnetData.fromtdms("pigbrotherdata/Fichiers_Data/M9/Overview/M9_Overview_240509-1634.tdms")
print(data.Keys)          # list of "Group/Channel" keys
# Access a specific group as a DataFrame
df = data.Data["Courants_Alimentations"]
print(df[["Référence_GR1", "Courant_GR1"]].head())
```

**Multi-source discovery (overview + archive + pupitre):**

```python
from python_magnetrun.analysis import load_data, FileDiscovery

# Discover files associated with an overview TDMS
discovery = FileDiscovery("M9_Overview_250303-1234.tdms", pupitre_datadir="srvdata/")
file_set = discovery.discover()

# Load DataFrames from a file set
data = load_data(file_set)
df_overview = data["overview"]
df_pupitre  = data["pupitre"]
```

**Single Hybrid RMS file:**

```python
from python_magnetrun.hybrid.rms.rms_reader import read_rms_file

# Returns a pandas DataFrame indexed by timestamp
df = read_rms_file("hybrid/rms/2024-05-09/FEPC-LNCMI/data.rms")
print(df.columns.tolist())          # list of variable names (analog + digital)
print(df[["PT205", "TT200A"]].head())
```

**Single Hybrid kHz file (binary):**

```python
from python_magnetrun.hybrid.kHz.fepc_reader import parse_cfg_file, read_hour_file

# 1. Parse the configuration file to get card/channel layout
cfg = parse_cfg_file("hybrid/kHz/2024-05-09/FEPC-LNCMI/HOST_2_DATA.CFG")
print(f"Analog slots: {cfg.get_analog_slots()}")

# 2. Read one hour of data for a given slot (returns a NumPy array)
slot = cfg.get_analog_slots()[0]
card = cfg.get_card_by_slot(slot)
data = read_hour_file(
    f"hybrid/kHz/2024-05-09/FEPC-LNCMI/00HOST_2_LIST_{slot}.bin",
    card.card_type,
)
print(f"Shape: {data.shape}")       # (n_samples, n_channels)

# 3. Extract a named channel by index
ch = card.variable_names.index("I_H1")
import numpy as np
time = np.arange(len(data)) / 1000.0   # 1 kHz → seconds
current = data[:, ch]
```

**Unified interface (kHz + RMS + Trigger for a full day):**

```python
from python_magnetrun.hybrid.hybrid_data import HybridData

hd = HybridData(
    base_dir="/path/to/hybrid",
    date_str="2024-05-09",
    fepc_system="FEPC-LNCMI",
)

# Inspect what is available
print(hd.Keys)                      # all discoverable keys

# Load RMS data as a DataFrame
df_rms = hd.load_rms_data("FEPC-LNCMI")
print(df_rms.head())

# Load kHz configuration and list variables
cfg = hd.load_khz_config("FEPC-LNCMI")
print(hd.get_khz_variables("FEPC-LNCMI"))
```

---

## Advanced Usage

### Time synchronization

```python
from python_magnetrun.analysis import synchronize_data, compute_lag

# Compute lag between overview and pupitre time series
lag_result = compute_lag(df_overview["Référence_GR1"], df_pupitre["IH"])
print(f"Lag: {lag_result.lag_seconds:.2f} s  (reliable: {lag_result.is_reliable})")

# Apply correction and return aligned DataFrames
sync_result = synchronize_data(df_overview, df_pupitre, key="Référence_GR1")
```

### Distance and similarity metrics

```python
from python_magnetrun.analysis import (
    calc_euclidean, calc_mae, calc_mape,
    calc_correlation,
    compute_dtw_distance,
    compute_tlcc,
)

series1 = df_overview["Référence_GR1"].values
series2 = df_pupitre["IH"].values

# Standard distance metrics
print(calc_euclidean(series1, series2).value)
print(calc_mae(series1, series2).value)
print(calc_mape(series1, series2).value)    # percentage
print(calc_correlation(series1, series2).value)

# Dynamic Time Warping (for shorter series, ≤ 5000 pts)
dtw = compute_dtw_distance(series1, series2)
print(f"DTW similarity score: {dtw.similarity_score:.4f}")

# Time-Lagged Cross-Correlation
tlcc = compute_tlcc(series1, series2, max_lag=50)
```

### Breakpoint detection and run signature

Each run is assigned a signature based on detected breakpoints:
- **U** — ramp up
- **P** — plateau
- **D** — ramp down

```bash
python3 -m tests.test-signature \
    srvdata/M10_2025.01.27---15:39:29.txt \
    --window=10 --threshold 1.e-2
```

Detect breakpoints in PigBrother overview files and synchronize:

```bash
python3 -m python_magnetrun.analysis \
    pigbrotherdata/Fichiers_Data/M10/Overview/M10_Overview_250211-*.tdms \
    --key Référence_GR1 --show --synchronize
```

### Anomaly detection

Detect anomalies with multiple methods (interactive CLI):

```bash
python3 tests/test-anomalies.py \
    pigbrotherdata/Fichiers_Data/M9/Fichiers_Spike/M9_Spikes_251207-115319.tdms \
    --group Courants_Alimentations \
    --methods dbscan mad \
    --method-params dbscan.eps=0.3 mad.threshold=4.0
```

With a JSON config for scripting:

```bash
python3 tests/test-anomalies.py data.tdms \
    --methods dbscan mad \
    --method-params-json '{"dbscan": {"eps": 0.3}, "mad": {"threshold": 4.0}}'
```

With a config file (YAML or JSON), optionally overriding specific params:

```bash
python3 tests/test-anomalies.py data.tdms \
    --config params.yaml \
    --method-params dbscan.eps=0.1
```

### Piecewise linear regression

Fit a piecewise-linear model for the Ih/Ib relationship:

```bash
python3 -m python_magnetrun.corr_Ih_Ib \
    srvdata/M9_2024.11.06---16:43:44.txt \
    --xkey IH --ykey IB \
    --algo piecewise_regression --breakpoints 2
```

Fit Field(t) with multiple breakpoints:

```bash
python3 -m python_magnetrun.corr_Ih_Ib \
    srvdata/M9_2024.11.06---16:43:44.txt \
    --xkey t --ykey Field \
    --algo pwlf --breakpoints 11
```

### Field factor identification

Estimate the field factors (fH, fB) for a given magnet site via OLS regression:

```bash
python3 -m python_magnetrun.test-fieldfactor \
    ~/M9_2024.05.13---16_30_51.txt
```

The regression output reports `fh` and `fB` coefficients. Cross-reference with field maps in [MagnetInfo](https://labs.core-cloud.net/ou/UPR3228/MagnetInfo/SitePages/Field-maps.aspx) for validation.

> [!NOTE]
> Ih and Ib are piecewise collinear. Use `--algo piecewise_regression` or `--algo pwlf` once the number of breakpoints is known for better results.

---

## Running Tests

Run the standard test suite:

```bash
pytest
```

Some tests require actual data files and are excluded by default. To include them:

```bash
pytest --on-demand
```

Run a specific on-demand test:

```bash
pytest --on-demand tests/test-paramident.py
pytest --on-demand tests/test-signature.py
pytest --on-demand tests/test-fft.py
pytest --on-demand tests/test-breakpoint-analysis.py
pytest --on-demand tests/test-fieldfactor.py
pytest --on-demand tests/test-intercept.py
pytest --on-demand tests/test-simu.py
pytest --on-demand tests/test-tin.py
```

---

## To-do

**Refactor:**
- [X] Split argparse options into separate Python files
- [ ] Add an example / a test for each subcommand in `python_magnetrun`
- [ ] Store stats (plateaus, duration) in a DataFrame, CSV, or database

**Docs:**
- [X] Docs for aggregate
- [X] Add a note to mount PigBrother data
- [ ] Add note to mount Pupitre data if applicable

**Features:**
- [ ] Store processed Pupitre data in a specific file format (ETL output)
- [ ] Rewrite `txt2csv` to use methods in `utils` and `plots`
- [ ] Check `addData` complex formulas (involving `freesteam` / `iapws`) with `pyparsing`
- [ ] Add columns from `freesteam` or `iapws` (e.g., rho, cp)
- [ ] Export data to `prettytables`, `tabular`, or `csv2md`
- [ ] Add support for Origin files (`liborigin` / Python bindings)
- [ ] Data from M1, M3, M5, M7 for complete stats
- [ ] Add missing fields (U1, Pe1, Tout1, U2, …) — link with `magnetdb`
- [ ] For `select`, support multiple field criteria
- [ ] Cross-lag correlations
- [ ] Forecast Teb from historical data
- [ ] Check independent variables (Ih, Teb, Qbrut) on plateau experiments
- [ ] Link with magnet user DB (`xdds.csv`)
- [ ] Classification of field profiles
- [ ] Link with `magnettools`/`hifimagnet` for R(i) and L(i)
- [ ] Estimate heat exchanger parameters (NTU model)
- [ ] Calorimetric balance for AC/DC converter dissipated power (Talim)

---

## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
