# Python `MagnetRun`

[![Run Tests](https://github.com/MagnetDB/python_magnetrun/actions/workflows/test.yml/badge.svg)](https://github.com/MagnetDB/python_magnetrun/actions/workflows/test.yml)
[![Documentation](https://github.com/MagnetDB/python_magnetrun/actions/workflows/docs.yml/badge.svg)](https://magnetdb.github.io/python_magnetrun/)
[![codecov](https://codecov.io/gh/MagnetDB/python_magnetrun/branch/main/graph/badge.svg)](https://codecov.io/gh/MagnetDB/python_magnetrun)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Python `MagnetRun` contains utilities to view and analyze Magnet runs from LNCMI control/monitoring systems (`Pupitre`, `PigBrother`, and hybrid FEPC acquisition systems).

- Free software: MIT license
- Documentation: <https://magnetdb.github.io/python_magnetrun/>

---

## Table of Contents

- [Installation](#installation)
- [Data Sources](#data-sources)
- [Mounting Data Directories](#mounting-data-directories)
- [Features](#features)
- [Basic Usage](#basic-usage)
  - [List available fields](#list-available-fields)
  - [Select records by criteria](#select-records-by-criteria)
  - [Plotting vs time](#plotting-vs-time)
  - [Plotting key vs key](#plotting-key-vs-key)
  - [Customising plot style per field](#customising-plot-style-per-field)
  - [Statistics and plateau detection](#statistics-and-plateau-detection)
  - [Derived quantities](#derived-quantities)
- [Analysis](#analysis)
- [Downsampling utilities](#downsampling-utilities)
- [ETL and Pipelines](#etl-and-pipelines)
- [Object Storage (RustFS)](#object-storage-rustfs)
- [Running Tests](#running-tests)
- [Changelog](CHANGELOG.md)
- [Credits](#credits)

---

## Installation

### Getting the source code

Clone the repository from GitHub, including the `python_magnetcooling` submodule:

```bash
git clone --recurse-submodules https://github.com/MagnetDB/python_magnetrun.git
cd python_magnetrun
```

If you already cloned without `--recurse-submodules`, initialise the submodule afterwards:

```bash
git submodule update --init --recursive
```

To get a specific release, check out a tag after cloning:

```bash
git checkout v1.2.3   # replace with the desired tag
git submodule update --recursive
```

> [!NOTE] To get a list of existing branches and tags, run `git tag` and `git branch -a`.

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

All acquisition data (Pupitre, PigBrother, Hybrid) is stored on a NAS server under a common root.  The NAS sub-directory layout is:

| Acquisition system | NAS sub-path | Environment variable |
|---|---|---|
| Pupitre (`.txt`) | `records/srv-data-install` | `MAGNETRUN_PUPITRE_DATA_DIR` |
| PigBrother (`.tdms`) | `records/pbsurv` | `MAGNETRUN_PIGBROTHER_DATA_DIR` |
| Hybrid (kHz / RMS) | `records/CEA` | `MAGNETRUN_HYBRID_DATA_DIR` |

Two mounting strategies are supported depending on your setup:

| Strategy | Mount root | Typical use |
|---|---|---|
| **rclone** | `~/LNCMIG-Data/records` | Laptop / remote workstation |
| **autofs** | `/mnt/LNCMIG-Data/records` | LNCMI on-premise workstation |

> [!TIP]
> The interactive Marimo notebook `marimo/00_nas_setup.py` walks you through the full setup, checks prerequisites, and writes `~/.config/python_magnetrun/data_dirs.json` automatically.

---

### Option A — rclone (laptop / remote workstation)

rclone mounts the NAS over SFTP and exposes it as a local FUSE filesystem.

#### 1. Install rclone and fuse3

```bash
# Debian / Ubuntu
sudo apt install rclone fuse3
```

See <https://rclone.org/downloads/> for other platforms.

#### 2. Create an SSH key (if you don't have one)

```bash
ssh-keygen -t ed25519 -C "$(hostname)-rclone"
# Accept the default path (~/.ssh/id_ed25519).
# Leave the passphrase blank for unattended daemon mounts.
```

Install the public key on the NAS and verify access:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub <user>@<nas-host>
ssh -i ~/.ssh/id_ed25519 <user>@<nas-host>   # should log in without password
```

#### 3. Configure the rclone remote

```bash
rclone config
# n  → new remote
# name:  LNCMIG
# type:  sftp
# host:  <NAS hostname or IP>
# user:  <your login>
# key_file: ~/.ssh/id_ed25519
```

#### 4. Mount the NAS

```bash
mkdir -p ~/LNCMIG-Data/records
rclone mount LNCMIG:/records ~/LNCMIG-Data/records \
    --vfs-cache-mode writes --daemon
```

To unmount:

```bash
fusermount3 -u ~/LNCMIG-Data/records
```

> [!NOTE]
> After mounting, the data directories become:
> - Pupitre: `~/LNCMIG-Data/records/srv-data-install`
> - PigBrother: `~/LNCMIG-Data/records/pbsurv`
> - Hybrid: `~/LNCMIG-Data/records/CEA`

#### 5. Auto-mount at login with systemd (Linux)

Create a systemd **user** service so the NAS is mounted automatically whenever you log in:

```bash
mkdir -p ~/.config/systemd/user
```

Write `~/.config/systemd/user/rclone-lncmig.service`:

```ini
[Unit]
Description=rclone mount — LNCMIG NAS
After=network-online.target
Wants=network-online.target

[Service]
Type=notify
ExecStartPre=/bin/mkdir -p %h/LNCMIG-Data/records
ExecStart=rclone mount LNCMIG:/records %h/LNCMIG-Data/records \
    --vfs-cache-mode writes \
    --log-level INFO
ExecStop=fusermount3 -u %h/LNCMIG-Data/records
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

Enable and start it:

```bash
systemctl --user daemon-reload
systemctl --user enable --now rclone-lncmig.service
```

Check status:

```bash
systemctl --user status rclone-lncmig.service
journalctl --user -u rclone-lncmig.service -f   # live logs
```

> [!NOTE]
> `%h` expands to your home directory inside the unit file.
> The service uses `Type=notify` which requires rclone ≥ 1.57 (the `--rc` / sd-notify support).
> If you are on an older version, change `Type=notify` to `Type=forking` and add `--daemon` to `ExecStart`.

#### Auto-mount at login with Task Scheduler (Windows)

On Windows, rclone mount requires **WinFsp** (the Windows FUSE driver).  Install it first:

1. Download and install WinFsp from <https://winfsp.dev/rel/>.
2. Install rclone for Windows from <https://rclone.org/downloads/> and add it to `%PATH%`.
3. Configure the remote the same way as on Linux (`rclone config`).

The simplest way to auto-mount at logon is a **Task Scheduler** task.  Open PowerShell as a normal user and run:

```powershell
$action  = New-ScheduledTaskAction -Execute "rclone" `
               -Argument 'mount LNCMIG:/records Z: --vfs-cache-mode writes'
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit 0 `
               -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "rclone-lncmig" `
    -Action $action -Trigger $trigger -Settings $settings `
    -RunLevel Limited -Force
```

This mounts the NAS as drive **`Z:`**.  Adjust the drive letter to taste.

Start it immediately without logging out:

```powershell
Start-ScheduledTask -TaskName "rclone-lncmig"
```

To remove the task:

```powershell
Unregister-ScheduledTask -TaskName "rclone-lncmig" -Confirm:$false
```

> [!NOTE]
> Update `MAGNETRUN_*_DATA_DIR` paths to use the Windows drive letter, e.g.
> `Z:\srv-data-install`, `Z:\pbsurv`, `Z:\CEA`.
> For a proper Windows service (survives without a logged-in session) use
> [NSSM](https://nssm.cc/) or [WinSW](https://github.com/winsw/winsw) to wrap the same `rclone mount` command.

---

### Option B — autofs (on-premise workstation)

autofs mounts the NAS automatically on first access and unmounts it after a period of inactivity.  Ask your sysadmin to configure `/etc/auto.master` and the relevant map file.  Once set up, the NAS appears at `/mnt/LNCMIG-Data/records` without any manual mount step.

---

### Configuring data directory paths

After mounting (either way), tell `python_magnetrun` where to find each data source.

**Persistent config file** — the recommended approach; written once and read on every import:

```bash
mkdir -p ~/.config/python_magnetrun
cat > ~/.config/python_magnetrun/data_dirs.json << 'EOF'
{
  "MAGNETRUN_PUPITRE_DATA_DIR":    "~/LNCMIG-Data/records/srv-data-install",
  "MAGNETRUN_PIGBROTHER_DATA_DIR": "~/LNCMIG-Data/records/pbsurv",
  "MAGNETRUN_HYBRID_DATA_DIR":     "~/LNCMIG-Data/records/CEA"
}
EOF
```

For autofs workstations, replace `~/LNCMIG-Data/records` with `/mnt/LNCMIG-Data/records`.

**Shell environment** — override or supplement the config file:

```bash
# add to ~/.bashrc or ~/.envrc (direnv)
export MAGNETRUN_PUPITRE_DATA_DIR="$HOME/LNCMIG-Data/records/srv-data-install"
export MAGNETRUN_PIGBROTHER_DATA_DIR="$HOME/LNCMIG-Data/records/pbsurv"
export MAGNETRUN_HYBRID_DATA_DIR="$HOME/LNCMIG-Data/records/CEA"
```

Once set, `load_mrun()` and all CLI commands resolve bare filenames against these directories automatically (see [Python API → Loading data](#loading-data)).

---

## Features

- Load `txt`, `csv`, and `tdms` files from control/monitoring systems
- Extract and list recorded fields
- Plot field(s) vs time or field vs field
- Cross-source comparison (Pupitre vs PigBrother vs hybrid kHz)
- Statistics and plateau detection
- Compute derived quantities via formulas (power, busbar losses, etc.)
- Breakpoint detection and run signature (UPD)
- Anomaly detection (Z-score, IQR, rolling mean/std, DBSCAN, MAD, isolation forest) via `OutlierConfig` / `OutlierDetector`
- Signal processing utilities (`normalize_signal`, `binarize_signal`) in `python_magnetrun.processing`
- Piecewise linear regression (`piecewise_regression`, `pwlf`)
- Field factor identification via OLS regression
- Time-series synchronization between data sources
- Scalar distance metrics (Euclidean, RMSE, MAE, MAPE, max error, Pearson, Mahalanobis) shared across analysis and downsampling in `python_magnetrun.utils.scalar_metrics`
- Distance and similarity metrics for cross-source comparison (Euclidean, MAE, MAPE, DTW, TLCC, Mahalanobis) in `python_magnetrun.analysis.metrics`
- Downsampling utilities (`DownsampleConfig`, `downsample_arrays`, `downsample_dataframe`) with pluggable algorithms (`stride`, `lttb`, `minmax_lttb`, `minmax`)
- Downsampling quality evaluation (`evaluate_downsampling`, `benchmark_configs`, `evaluate_downsampling_segments`) with per-segment (plateau vs transition) metrics and optional memory profiling
- Extract data from `srv-data-lncmi`
- Prepare data for injection into `magnetdb`
- Field-definition management (`*-defs.json`) with cross-format aliases
- Per-housing sensor role configuration (`<Housing>-housing-config.json`)
- ETL pipeline: clean Pupitre data, rename channels, compute thermal/hydraulic quantities
- Waterflow and thermal pipeline modules (integrates with `python_magnetcooling`)
- Object storage integration via `rustfs/` (S3-compatible, RustFS/MinIO)

---

## Field definitions and housing configuration

### Field definitions (`*-defs.json`)

Every acquisition format has a companion JSON file that maps channel names to
physical metadata (symbol, unit, description) and cross-format aliases:

| File | Format | Format doc |
|---|---|---|
| [`python_magnetrun/pupitre-defs.json`](python_magnetrun/pupitre-defs.json) | Pupitre `.txt` columns | [docs/pupitre.md](docs/pupitre.md) |
| [`python_magnetrun/pigbrother-defs.json`](python_magnetrun/pigbrother-defs.json) | PigBrother `Group/Channel` keys | [docs/pigbrother.md](docs/pigbrother.md) |
| [`python_magnetrun/hybrid-defs.json`](python_magnetrun/hybrid-defs.json) | Hybrid `FEPC_system/variable` keys | [docs/Hybride.md](docs/Hybride.md) |

**`"aliases"`** entries express housing-independent name correspondences between
formats (e.g. `Idcct1` in pupitre = `Courants_Alimentations/Courant_A1` in
pigbrother = `FEPC-AUX-LNCMI/ALIM1_J1` in hybrid). Housing-dependent mappings
(e.g. `IH` is GR1 in M9 but GR2 in M8) belong in the housing-config files.

#### File resolution

When a bare filename (e.g. `"pupitre-defs.json"`) is passed to any API function
or CLI command, it is resolved in this order:

1. Absolute path — used directly.
2. Relative path that exists in the current directory — used as-is.
3. `~/.config/magnetrun/<filename>` — user override (takes precedence over the bundle).
4. File bundled with the installed package — the shipped default.

To permanently override a bundled file, drop your edited copy into
`~/.config/magnetrun/`:

```bash
cp /path/to/my-pupitre-defs.json ~/.config/magnetrun/pupitre-defs.json
# All subsequent calls to load_defs("pupitre-defs.json") will use your copy.
```

Manage defs files with the `magnetrun-config field` CLI (bare names work after
installation — no need to specify the full path to the bundled file):

```bash
# List all field definitions (including aliases)
magnetrun-config field pupitre-defs.json list

# Add a new field
magnetrun-config field pupitre-defs.json add NewSensor I ampere \
    --description "New coil current"

# Update an existing field
magnetrun-config field pupitre-defs.json update Field \
    --symbol Bz --description "Axial field"

# Add a cross-format alias
magnetrun-config field pupitre-defs.json alias-add Idcct1 hybrid \
    "FEPC-AUX-LNCMI/ALIM1_J1"

# Show aliases for one field
magnetrun-config field pupitre-defs.json alias-show Idcct1

# Build a cross-reference index across all three formats
magnetrun-config field pupitre-defs.json crossref \
    --format pupitre=pupitre-defs.json \
    --format pigbrother=pigbrother-defs.json \
    --format hybrid=hybrid-defs.json
```

Python API:

```python
from python_magnetrun.field_defs import (
    load_defs, add_field_def, update_field_def,
    add_alias, get_aliases, build_crossref,
    resolve_defs_file,
)

# Bare names resolve automatically to the bundled default (or user override)
aliases = get_aliases("pupitre-defs.json", "Idcct1")
# {"pigbrother": "Courants_Alimentations/Courant_A1",
#  "hybrid":     "FEPC-AUX-LNCMI/ALIM1_J1"}

# Inspect where a file resolves to
p = resolve_defs_file("pupitre-defs.json")
print(p)   # e.g. /usr/lib/python3.11/site-packages/python_magnetrun/pupitre-defs.json

# Build a unified index across all formats
index = build_crossref({
    "pupitre":    "pupitre-defs.json",
    "pigbrother": "pigbrother-defs.json",
    "hybrid":     "hybrid-defs.json",
})
index["pupitre"]["Ucoil1"]
# {"pigbrother": "Tensions_Aimant/Interne1",
#  "hybrid":     "FEPC-AUX-LNCMI/PH_V8"}
```

### Housing configuration (`<Housing>-housing-config.json`)

Each housing maps the same physical sensor set to GR roles differently.
Bundled JSON templates ship with the package and are also hardcoded in the
`HOUSING_CONFIGS` dict as an always-available fallback:

| Bundled file | Housing | Format docs |
|---|---|---|
| [`python_magnetrun/M9-housing-config.json`](python_magnetrun/M9-housing-config.json) | M9 (resistive only) | [pupitre](docs/pupitre.md), [pigbrother](docs/pigbrother.md) |
| [`python_magnetrun/M8-housing-config.json`](python_magnetrun/M8-housing-config.json) | M8 (resistive + SC insert, hybrid) | [pupitre](docs/pupitre.md), [pigbrother](docs/pigbrother.md), [hybrid](docs/Hybride.md) |
| [`python_magnetrun/M10-housing-config.json`](python_magnetrun/M10-housing-config.json) | M10 (resistive only) | [pupitre](docs/pupitre.md), [pigbrother](docs/pigbrother.md) |

#### File resolution

`get_housing_config(housing)` resolves configurations in this order:

1. Explicit `json_file` argument — used directly.
2. `~/.config/magnetrun/<Housing>-housing-config.json` — persistent user override.
3. Hardcoded built-in default from `HOUSING_CONFIGS`.

To persist a customized config for a housing, copy it to the user config dir:

```bash
# Start from the bundled template
magnetrun-config housing M9-housing-config.json create M9 --from-builtin M9

# Copy to user config dir so get_housing_config("M9") picks it up automatically
cp M9-housing-config.json ~/.config/magnetrun/M9-housing-config.json

# Edit in place
magnetrun-config housing ~/.config/magnetrun/M9-housing-config.json update \
    --gr1-current IB --gr2-current IH
```

Manage with the `magnetrun-config housing` CLI:

```bash
# Show a housing config
magnetrun-config housing M9-housing-config.json show

# Create a new housing config initialised from M9 defaults
magnetrun-config housing M11-housing-config.json create M11 --from-builtin M9

# Update role fields in place
magnetrun-config housing M9-housing-config.json update \
    --gr1-current IB --gr2-current IH
```

Python API:

```python
from python_magnetrun.housing_config import (
    get_housing_config, load_housing_config, save_housing_config, update_housing_config,
    get_bundled_housing_config_path, get_user_housing_config_path,
)

# Built-in default (also checks ~/.config/magnetrun/ first)
cfg = get_housing_config("M9")
cfg.reference_gr1_current      # "IH"
cfg.supports_format("hybrid")  # False

# Find where the bundled template lives (read-only, inside the package)
get_bundled_housing_config_path("M9")
# PosixPath('.../site-packages/python_magnetrun/M9-housing-config.json')

# Find (and create) the user-writable config path
get_user_housing_config_path("M9")
# PosixPath('/home/you/.config/magnetrun/M9-housing-config.json')

# From an explicit file
cfg = get_housing_config("M9", json_file="M9-housing-config.json")

# Runtime override (e.g. GR1/GR2 swapped for an atypical run)
cfg = get_housing_config("M9", overrides={"gr1_current": "IB", "gr2_current": "IH"})

# Update a file in place and get the new config back
cfg = update_housing_config("M9-housing-config.json", {"gr1_current": "IB"})
```

---

## Basic Usage

### List available fields

```bash
python3 -m python_magnetrun.cli \
    data/M9_2019.02.14-23_00_38.txt info --list
```

or use data from predefined directories:

```bash
python3 -m python_magnetrun.cli \
    --housing M9 "2019.02.14 - 23:00:38.txt" info --list
```

### Select records by criteria

List records lasting at least 60 s with a magnetic field above 18 T:

```bash
python3 examples/get-record.py \
    srvdata/M8*.txt select --duration 60 --field 18.
```

or use data from predefined directories:

```bash
python3 examples/get-record.py \
    --housing M8 "2025.*.txt" select --duration 60 --field 18.
```

### Plotting vs time

#### Single field

```bash
python3 -m python_magnetrun.cli \
    srvdata/M9_2019.02.14---23:00:38.txt \
    plot --vs_time Field
```

#### Multiple fields from a PigBrother TDMS overview

Two fields on one axes — each drawn in a **distinct colour** from the default palette:

```bash
python3 -m python_magnetrun.cli \
    M9_Overview_260331-1316.tdms \
    --housing M9 \
    plot --vs_time Courants_Alimentations/Référence_A1 \
                  Courants_Alimentations/Référence_A2
```

#### Separate subplots

```bash
python3 -m python_magnetrun.cli \
    M9_Overview_260331-1316.tdms \
    --housing M9 \
    plot --vs_time Courants_Alimentations/Référence_A1 \
                  Courants_Alimentations/Référence_A2 \
    --subplots
```

#### Normalise before overlaying

```bash
python3 -m python_magnetrun.cli \
    M9_Overview_260331-1316.tdms \
    --housing M9 \
    plot --vs_time Courants_Alimentations/Référence_A1 \
                  Courants_Alimentations/Référence_A2 \
    --normalize
```

#### Override display units

```bash
python3 -m python_magnetrun.cli \
    M9_Overview_260331-1316.tdms \
    --housing M9 \
    plot --vs_time Courants_Alimentations/Courant_A1 \
    --unit Courant_A1=kiloampere
```

#### Compare PigBrother, Pupitre, and hybrid (kHz) currents

```bash
python3 -m python_magnetrun.cli \
    M9_Overview_240509-1634.tdms \
    M9_2024.05.09---16_34_03.txt \
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

### Plotting key vs key

Plot `IH` vs `UH` from a Pupitre file:

```bash
python3 -m python_magnetrun.cli \
    "2026.03.31 - 13:22:40.txt" \
    --housing M9 \
    plot --key_vs_key "IH-UH"
```

Plot with **markers only** (no connecting lines), using the default `o` marker:

```bash
python3 -m python_magnetrun.cli \
    "2026.03.31 - 13:22:40.txt" \
    --housing M9 \
    plot --key_vs_key "IH-UH" --no-lines
```

Plot with a custom marker **and** connecting lines:

```bash
python3 -m python_magnetrun.cli \
    "2026.03.31 - 13:22:40.txt" \
    --housing M9 \
    plot --key_vs_key "IH-UH" --marker "+"
```

Plot with markers only using a specific symbol:

```bash
python3 -m python_magnetrun.cli \
    "2026.03.31 - 13:22:40.txt" \
    --housing M9 \
    plot --key_vs_key "IH-UH" --marker "s" --no-lines
```

Compare PigBrother and Pupitre current vs voltage side by side:

```bash
python3 -m python_magnetrun.cli \
    srvdata/M10_2025.01.27---*.txt \
    pigbrotherdata/Fichiers_Data/M10/Overview/M10_Overview_250127-1605.tdms \
    plot \
    --key_vs_key IH-UH \
    --key_vs_key Courants_Alimentations/Référence_GR1-Tensions_Aimant/Interne1
```

Use `--field_style` to control marker and line style per series.  The `FIELD`
key is the **y-column** of the pair (the part after `-` in `X-Y`):

```bash
# IH-UH plotted as circle markers only (no connecting line)
python3 -m python_magnetrun.cli \
    "2026.03.31 - 13:22:40.txt" \
    --housing M9 \
    plot \
    --key_vs_key "IH-UH" \
    --field_style "UH=o"

# Two pairs from different files – first pair square markers every 20 points,
# second pair dashed line
python3 -m python_magnetrun.cli \
    srvdata/M10_2025.01.27---*.txt \
    pigbrotherdata/Fichiers_Data/M10/Overview/M10_Overview_250127-1605.tdms \
    plot \
    --key_vs_key IH-UH \
    --key_vs_key Courants_Alimentations/Référence_A2-Tensions_Aimant/Interne1 \
    --field_style "UH=-s:20" \
    --field_style "Interne1=--"
```

### Customising plot style per field

Use `--field_style FIELD=STYLESPEC` (repeatable) to assign **different linestyles,
markers, markevery, and opacity** to individual fields in a `--vs_time` overlay.

**`STYLESPEC` syntax:** `[LINESTYLE][MARKER][:N][@ALPHA]`

| Part | Values | Meaning |
|---|---|---|
| `LINESTYLE` | `-`, `--`, `-.` | Line style; omit to suppress lines when a marker is given |
| `MARKER` | `o`, `+`, `x`, `s`, `D`, … | Any [matplotlib marker](https://matplotlib.org/stable/gallery/lines_bars_and_markers/marker_reference.html) |
| `:N` | integer | Draw marker every *N* data points |
| `@ALPHA` | float in `[0, 1]` | Opacity (1 = fully opaque, 0 = invisible) |

Examples:

```bash
# A1 with solid line only; A2 with circle markers every 10 points (no line)
python3 -m python_magnetrun.cli \
    M9_Overview_260331-1316.tdms --housing M9 \
    plot --vs_time Courants_Alimentations/Référence_A1 \
                  Courants_Alimentations/Référence_A2 \
    --field_style "Référence_A1=-" \
    --field_style "Référence_A2=o:10"

# A1 dashed; A2 solid line with square markers every 5 points
python3 -m python_magnetrun.cli \
    M9_Overview_260331-1316.tdms --housing M9 \
    plot --vs_time Courants_Alimentations/Référence_A1 \
                  Courants_Alimentations/Référence_A2 \
    --field_style "Référence_A1=--" \
    --field_style "Référence_A2=-s:5"

# A1 solid line at 50% opacity; A2 dashed at 80% opacity
python3 -m python_magnetrun.cli \
    M9_Overview_260331-1316.tdms --housing M9 \
    plot --vs_time Courants_Alimentations/Référence_A1 \
                  Courants_Alimentations/Référence_A2 \
    --field_style "Référence_A1=-@0.5" \
    --field_style "Référence_A2=--@0.8"

# Combined: A2 solid line + circle markers every 5 points at 60% opacity
python3 -m python_magnetrun.cli \
    M9_Overview_260331-1316.tdms --housing M9 \
    plot --vs_time Courants_Alimentations/Référence_A1 \
                  Courants_Alimentations/Référence_A2 \
    --field_style "Référence_A2=-o:5@0.6"
```

The `FIELD` key in `--field_style` can be either the short channel name
(`Référence_A2`) or the full `Group/Channel` form
(`Courants_Alimentations/Référence_A2`).

The same `--field_style` options work with the **Plotly backend** (and
`--subplots` / `--normalize`).  Matplotlib marker names are translated
automatically: `o` → `circle`, `s` → `square`, `D` → `diamond`, `^` →
`triangle-up`, etc.  Linestyles map to Plotly dashes: `-` → `solid`, `--` →
`dash`, `-.` → `dashdot`, `:` → `dot`.  When `markevery` is used with Plotly
the data is sub-sampled before rendering since Plotly has no native equivalent.
`@ALPHA` maps to Plotly `opacity`.

```bash
# Plotly overlay – A1 dashed at 70% opacity, A2 solid line + circle markers every 20 points
python3 -m python_magnetrun.cli \
    M9_Overview_260331-1316.tdms --housing M9 \
    plot --vs_time Courants_Alimentations/Référence_A1 \
                  Courants_Alimentations/Référence_A2 \
    --backend plotly \
    --field_style "Référence_A1=--@0.7" \
    --field_style "Référence_A2=-o:20"

# Plotly stacked subplots – markers only (no line) on one field
python3 -m python_magnetrun.cli \
    M9_Overview_260331-1316.tdms --housing M9 \
    plot --vs_time Courants_Alimentations/Référence_A1 \
                  Courants_Alimentations/Référence_A2 \
    --backend plotly --subplots \
    --field_style "Référence_A2=o:50"

# Plotly with normalisation – combined with per-field style and alpha
python3 -m python_magnetrun.cli \
    M9_Overview_260331-1316.tdms --housing M9 \
    plot --vs_time Courants_Alimentations/Référence_A1 \
                  Courants_Alimentations/Référence_A2 \
    --backend plotly --normalize \
    --field_style "Référence_A1=-." \
    --field_style "Référence_A2=-s:10@0.5"
```

### Plot colour palette

By default each field gets a **distinct colour** from the Matplotlib `tab10` palette,
cycling when there are more fields than palette entries.

Use `--same-color-per-type` to assign one fixed colour per file type instead
(`.txt` → green, `.tdms` → red):

```bash
python3 -m python_magnetrun.cli \
    M9_Overview_260331-1316.tdms \
    "2026.03.31 - 13:22:40.txt" \
    --housing M9 \
    plot --vs_time Courants_Alimentations/Courant_GR1 \
         --vs_time IH \
    --same-color-per-type
```

Customise the palette via a plot-config JSON (see `magnetrun-config plot init`):

```bash
magnetrun-config plot init           # writes plot_config.json in the current directory
# edit palette, colors, style …
python3 -m python_magnetrun.cli … plot … --plot-config plot_config.json
```

### Statistics and plateau detection

Compute statistics for all M8 records:

```bash
python3 -m python_magnetrun.cli srvdata/M8*.txt stats
```

Detect plateaux:

```bash
python3 -m python_magnetrun.cli srvdata/M8*.txt stats --plateau --keys Field
```

#### Plateau detection parameters

The algorithm groups consecutive points whose step-to-step absolute variation stays below `--threshold`, then discards groups shorter than `--dthreshold` seconds.

| Argument | Default | Meaning |
|---|---|---|
| `--keys KEY [KEY …]` | — | Channel(s) to analyse (required with `--plateau`) |
| `--threshold FLOAT` | `1e-3` | Max point-to-point absolute variation to still be considered flat (in signal units, e.g. T for `Field`) |
| `--dthreshold FLOAT` | `10` | Minimum plateau duration in seconds (`num_points = dthreshold / sampling_period`) |

**Tuning guide:**

| Symptom | Remedy |
|---|---|
| Too many short / fragmented plateaus | Increase `--threshold` |
| Real plateaus not detected or split | Decrease `--threshold` or `--dthreshold` |
| Short transients wrongly included | Increase `--dthreshold` |

Run with `--log-level DEBUG` to see per-group statistics and the distribution of step differences, which helps calibrate `--threshold` for a given signal.

Aggregate a specific field across records:

```bash
python3 examples/get-record.py \
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

### Loading data

The `load_mrun()` function provides a convenient unified interface for loading any supported file type with automatic file resolution.

**Simple filename (auto-resolves to standard data directories):**

```python
from python_magnetrun.MagnetRun import load_mrun

# Just provide the filename - it searches automatically!
mrun = load_mrun('2026.03.31 - 13:22:40.txt', housing='M9')
print(f"Loaded: {mrun.MagnetData.FileName}")
print(f"StartTime: {mrun.StartTime}")

mdata = mrun.getMData()
df = mdata.Data  # pandas DataFrame with all fields
print(df.head())

```

**How auto-resolution works:**

When `auto_resolve=True` (the default), `load_mrun()` searches for files in priority order:

1. Current working directory
2. Extension-specific data directory (e.g., `/mnt/LNCMIG-Data/records/srv-data-install/` for `.txt`)
3. Extension-specific data directory + housing subdirectory (e.g., `.../srv-data-install/M9/`)

This uses the same resolution logic as the CLI, so scripts and CLI commands find files consistently.

**Full path (backwards compatible):**

```python
# Still works with absolute paths
mrun = load_mrun('/mnt/LNCMIG-Data/records/srv-data-install/M9/2026.03.31 - 13:22:40.txt')
```

**Disable auto-resolution:**

```python
# Only check current directory (raises FileNotFoundError if not found)
mrun = load_mrun('file.txt', auto_resolve=False)
```

**Supports all file types:**

```python
# TDMS files
mrun = load_mrun('M9_Overview_260331-1316.tdms', housing='M9')

# CSV files
mrun = load_mrun('data.csv', housing='M9')
```

**Environment variable overrides:**

Customize data directories via environment variables (see [Mounting Data Directories](#mounting-data-directories)):

```bash
export MAGNETRUN_PUPITRE_DATA_DIR="/custom/path/to/pupitre"
export MAGNETRUN_PIGBROTHER_DATA_DIR="/custom/path/to/pigbrother"
```

```python
# Now load_mrun() searches your custom directories automatically
mrun = load_mrun('2026.03.31 - 13:22:40.txt', housing='M9')
```

### Configuration

```python
from python_magnetrun.analysis import AnalysisConfig

# Load pre-defined housing configuration (M8, M9, M10)
config = AnalysisConfig.for_housing("M9")

# Access housing-specific channel mappings
print(config.housing.reference_gr1_current)   # "IH"
print(config.housing.reference_gr2_current)   # "IB"

# Get threshold for a channel
threshold = config.thresholds.get("Courant_GR1")
```

### Lower-level data loading APIs

For more control over the loading process, use the underlying `load_magnetdata()` and format-specific methods:

**Single Pupitre file (`.txt`):**

```python
from python_magnetrun.magnetdata import load_magnetdata
from python_magnetrun.runetl import prepareData

data = load_magnetdata("data/M9_2019.02.14-23_00_38.txt")
prepareData(data, housing="M9")  # applies housing-specific renaming and timestamp parsing
data.Units()

print(data.Keys)          # list of available field names
df = data.Data            # pandas DataFrame
print(df[["t", "Field", "IH", "IB"]].head())
```

**Single PigBrother file (`.tdms`):**

```python
from python_magnetrun.magnetdata import load_magnetdata
from python_magnetrun.runetl import prepareData

data = load_magnetdata("pigbrotherdata/Fichiers_Data/M9/Overview/M9_Overview_240509-1634.tdms")
prepareData(data, housing="M9")  # applies housing-specific renaming and timestamp parsing
data.Units()

print(data.Keys)          # list of "Group/Channel" keys
# Access a specific group as a DataFrame
df = data.Data["Courants_Alimentations"]
print(df[["Référence_GR1", "Courant_GR1"]].head())
```

**Multi-source discovery (overview + archive + pupitre):**

```python
from python_magnetrun.analysis import load_data, FileDiscovery

# Discover files associated with an overview TDMS
discovery = FileDiscovery("M9_Overview_250303-1234.tdms", pupitre_datadir="data/")
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
from python_magnetrun.outliers import OutlierConfig

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
vars = hd.get_khz_variables("FEPC-LNCMI")
print(vars["analog"])

# Read a variable with outlier removal
cfg = OutlierConfig(method="iqr")   # threshold resolved from OUTLIER_DEFAULTS
hd.plot_khz_variable("FEPC-LNCMI", "I_H1", outlier_config=cfg, show=False)
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

The scalar primitives live in `python_magnetrun.utils.scalar_metrics` and are
re-exported by `python_magnetrun.analysis.metrics` for backwards compatibility.
Import from whichever location makes sense for your context — both paths are stable.

```python
# Canonical low-level path (no analysis-layer dependency)
from python_magnetrun.utils.scalar_metrics import (
    calc_euclidean, calc_rmse, calc_mae, calc_mape, calc_max_error,
    calc_correlation,
    calc_mahalanobis,            # 1-D mean-difference / pooled-std distance
    calc_mahalanobis_multivariate,  # point-wise Mahalanobis on paired samples
)

# High-level analysis path (adds DTW, TLCC, DistanceResult, …)
from python_magnetrun.analysis.metrics import (
    calc_euclidean, calc_mae, calc_mape, calc_correlation,
    compute_all_distances,   # → DistanceResult
    compute_dtw_distance,
    compute_tlcc,
)

series1 = df_overview["Référence_GR1"].values
series2 = df_pupitre["IH"].values

# Compute all basic distances at once
result = compute_all_distances(series1, series2)
print(result.euclidean, result.rmse, result.mae, result.mape)
print(result.mahalanobis)

# Dynamic Time Warping (for shorter series, ≤ 5000 pts)
dtw = compute_dtw_distance(series1, series2)
print(f"DTW similarity score: {dtw.similarity_score:.4f}")

# Time-Lagged Cross-Correlation
tlcc = compute_tlcc(series1, series2, seconds=5, fps=30)
print(f"Optimal lag: {tlcc.optimal_lag}")
```

### Breakpoint detection and run signature

Each run is assigned a signature based on detected breakpoints:
- **U** — ramp up
- **P** — plateau
- **D** — ramp down

```bash
python3 tests/test-signature.py \
    data/M10_2025.01.27---15:39:29.txt \
    --window=10 --threshold 1.e-2
```

Detect breakpoints in PigBrother overview files and synchronize:

```bash
python3 -m python_magnetrun.analysis.cli \
    pigbrotherdata/Fichiers_Data/M10/Overview/M10_Overview_250211-*.tdms \
    --show [--synchronize]
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
python3 examples/corr_Ih_Ib.py \
    data/M9_2024.11.06---16:43:44.txt \
    --xkey IH --ykey IB \
    --algo piecewise_regression --breakpoints 2
```

Fit Field(t) with multiple breakpoints:

```bash
python3 examples/corr_Ih_Ib.py \
    data/M9_2024.11.06---16:43:44.txt \
    --xkey t --ykey Field \
    --algo pwlf --breakpoints 11
```

### Field factor identification

Estimate the field factors (fH, fB) for a given magnet site via OLS regression:

```bash
python3 tests/test-fieldfactor.py \
    ~/M9_2024.05.13---16_30_51.txt
```

The regression output reports `fh` and `fB` coefficients. Cross-reference with field maps in [MagnetInfo](https://labs.core-cloud.net/ou/UPR3228/MagnetInfo/SitePages/Field-maps.aspx) for validation.

> [!NOTE]
> Ih and Ib are piecewise collinear. Use `--algo piecewise_regression` or `--algo pwlf` once the number of breakpoints is known for better results.

---

## Downsampling utilities

`python_magnetrun.utils` provides a unified downsampling layer used by every
data source (Pupitre, PigBrother, Hybrid kHz) and the analysis pipeline.

### Downsampling data

```python
from python_magnetrun.MagnetRun import load_mrun
from python_magnetrun.utils import DownsampleConfig, downsample_arrays, downsample_dataframe

# Load a PigBrother overview file
mrun = load_mrun("M9_Overview_240511-1150.tdms", housing="M9")
df = mrun.getMData().Data["Courants_Alimentations"]  # DataFrame for one TDMS group

time = df.index.to_numpy(dtype=float)               # [s] elapsed time
data = df["Courant_GR1"].to_numpy(dtype=float)      # [A]

# Reduce to 5 000 points using MinMax-LTTB (requires tsdownsample)
config = DownsampleConfig(n_out=5_000, method="minmax_lttb")
data_ds, time_ds = downsample_arrays(data, time, config)

# Build from a percentage of the dataset length
config_pct = DownsampleConfig.from_percent(len(data), percent=5.0, method="lttb")

# Downsample an entire group DataFrame at once
df_ds = downsample_dataframe(df, time_col="t", config=config)
```

Available methods:

| Method | Description | Extra dependency |
|---|---|---|
| `stride` | Uniform stride (every k-th point) | — |
| `minmax` | Min/max per bucket — preserves peaks | `tsdownsample` |
| `lttb` | Largest-Triangle-Three-Buckets | `tsdownsample` |
| `minmax_lttb` | MinMax pre-filter + LTTB selection | `tsdownsample` |

Install optional backend:

```bash
pip install tsdownsample
```

### Evaluating downsampling quality

`evaluate_downsampling` reconstructs the signal from the downsampled points
(linear interpolation back onto the original time grid) and returns a full
quality report as a `DownsampleMetrics` dataclass.

```python
from python_magnetrun.MagnetRun import load_mrun
from python_magnetrun.utils import (
    DownsampleConfig,
    evaluate_downsampling,
    evaluate_downsampling_segments,
    benchmark_configs,
)

mrun = load_mrun("M9_Overview_240511-1150.tdms", housing="M9")
df = mrun.getMData().Data["Courants_Alimentations"]
time = df.index.to_numpy(dtype=float)
data = df["Courant_GR1"].to_numpy(dtype=float)

config = DownsampleConfig(n_out=5_000, method="minmax_lttb")
metrics = evaluate_downsampling(data, time, config)

print(metrics.summary())
# minmax_lttb: 36000→5000 (ratio=7.2x), RMSE=0.0031, max_err=0.012, t=4.1ms

print(metrics.compression_ratio)   # 7.2
print(metrics.rmse)                # reconstruction RMSE [A]
print(metrics.hausdorff_distance)  # max directed Hausdorff in normalised (t,y) space
print(metrics.energy_ratio)        # ‖reconstructed‖²/‖original‖², ideal = 1.0
print(metrics.peak_max_error)      # |max(reconstructed) - max(original)| / range
```

With optional memory profiling:

```python
# Tier 1 (default): tracemalloc — Python heap only, zero overhead
metrics = evaluate_downsampling(data, time, config, compute_memory=True)
print(metrics.peak_memory_bytes, metrics.memory_overhead_ratio)

# Tier 2: subprocess isolation — captures Python + native (Rust/C) heap; ~100 ms
metrics = evaluate_downsampling(data, time, config, compute_memory=True, memory_tier=2)

# Tier 3: memray native tracing — most accurate; requires `pip install memray`
metrics = evaluate_downsampling(data, time, config, compute_memory=True, memory_tier=3)
```

### Per-segment quality (plateau vs transition)

`evaluate_downsampling_segments` splits the reconstruction error into
plateau (steady-state) and transition (ramp) regions using `binarize_signal`,
which is especially meaningful for magnet run data that alternates between ramp
and flat-top phases.

```python
base_metrics, seg_metrics = evaluate_downsampling_segments(
    data, time, config,
    threshold=None,   # None → automatic Otsu threshold
    window=50,        # smoothing window for binarisation
)

print(f"Plateau RMSE:    {seg_metrics.plateau_rmse:.4g}  (fraction={seg_metrics.plateau_fraction:.0%})")
print(f"Transition RMSE: {seg_metrics.transition_rmse:.4g}  (fraction={seg_metrics.transition_fraction:.0%})")
```

### Benchmarking multiple configurations

`benchmark_configs` runs `evaluate_downsampling` for each config and returns
a tidy `DataFrame` for easy comparison or export:

```python
configs = [
    DownsampleConfig(n_out=1_000, method="stride"),
    DownsampleConfig(n_out=1_000, method="minmax_lttb"),
    DownsampleConfig(n_out=5_000, method="minmax_lttb"),
    DownsampleConfig(n_out=5_000, method="lttb"),
]

df_bench = benchmark_configs(data, time, configs)
print(df_bench[["compression_ratio", "rmse", "max_error", "elapsed_s"]])
```

---

## ETL and Pipelines

### ETL (`runetl`)

`python_magnetrun.runetl` provides functions to clean and normalise raw Pupitre data
before further processing or ingestion into a database:

- drop all-zero non-essential columns
- rename `Icoil` → `IH` / `IB` based on housing config
- attach computed `timestamp` (naive UTC)

```python
from python_magnetrun.runetl import prepare_pupitre

data = prepare_pupitre("data/M9_2024.05.09---16_34_03.txt", housing="M9")
```

### Waterflow pipeline (`waterflow_pipeline`)

Extracts hydraulic operating points (flow rate, pressure drop, pump speed) from a
prepared Pupitre DataFrame and fits pump curves using `python_magnetcooling`:

```python
from python_magnetrun.waterflow_pipeline import extract_hydraulic_data, fit_pump_curve

hyd = extract_hydraulic_data(df, housing_cfg=cfg)
result = fit_pump_curve(hyd)
print(result.coefficients)
```

### Thermal pipeline (`thermal_pipeline`)

Computes per-circuit and global thermal quantities (inlet/outlet temperature, heat
load, efficiency) from prepared DataFrames using an iterative calorimetric scheme:

```python
from python_magnetrun.thermal_pipeline import compute_thermal

thermal = compute_thermal(df, housing_cfg=cfg)
print(thermal.heat_load_gr1)
```

---

## Object Storage (RustFS)

The `rustfs/` subdirectory provides a self-contained `magnetfs` Python package and
Docker Compose setup for local S3-compatible object storage (RustFS / MinIO-compatible).

**Workflow:**

1. Start RustFS locally:

```bash
cd rustfs
docker compose up -d
```

2. Convert and upload Pupitre `.txt` data to Parquet:

```bash
magnetfs upload data/M9_2024.05.09---16_34_03.txt
```

3. Read and plot data directly from the bucket:

```bash
magnetfs plot M9_2024.05.09---16_34_03 --key Field
```

A Streamlit dashboard (`app_streamlit.py`) and a Panel dashboard (`app_panel.py`) are
also included for interactive exploration.

See [rustfs/README.md](rustfs/README.md) for full setup instructions.

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

See [CHANGELOG.md](CHANGELOG.md) for breaking changes and the to-do list.

---

## Todos

- [x] rewrite notes on mounting NAS data
- [x] add notes on rclone setup
- [ ] add installation notes for Windows 11 (without wsl)
- [ ] create a test for autofs/rclone solution
- [ ] add Hybrid data plot in analysis
- [ ] add notebooks demo (Jupyter, Marimo)
- [ ] starting from analysis, use Overview to cleanup Pupitre data
- [ ] are there "standalone" pupitre/Overview files
- [ ] add support for custom npTDMS version with polar support
- [ ] use narwhals for dataframe

## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
