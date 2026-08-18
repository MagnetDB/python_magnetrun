# Hybrid Data Plotting with Pupitre and TDMS Comparison

This directory contains example scripts for plotting hybrid magnet data alongside corresponding pupitre and TDMS (pigbrother) data.

## Overview

The `plot_hybrid_with_pupitre_tdms.py` script demonstrates how to:

1. **Load hybrid kHz data** from FEPC-AUX-LNCMI or FEPC-LNCMI systems
2. **Find corresponding files** using date-based search in pupitre and pigbrother directories
3. **Map field names** between different data sources using dictionaries
4. **Plot all data sources** on the same graph for comparison

## Field Name Mapping

The script uses dictionaries to map field names between data sources:

### Hybrid to Pupitre Mapping

Setup for M8 housing:

```python
HYBRID_TO_PUPITRE_MAP = {
    "kHz/FEPC-AUX-LNCMI/ALIM1_J1": "Idcct1",
    "kHz/FEPC-AUX-LNCMI/ALIM1_J2": "Idcct2",
    "kHz/FEPC-AUX-LNCMI/ALIM2_J1": "Idcct3",
    "kHz/FEPC-AUX-LNCMI/ALIM2_J2": "Idcct4",
}
```

### Hybrid to TDMS Mapping

```python
HYBRID_TO_TDMS_MAP = {
    "kHz/FEPC-AUX-LNCMI/ALIM1_J1": "Courant_A1",
    "kHz/FEPC-AUX-LNCMI/ALIM1_J2": "Courant_A2",
    "kHz/FEPC-AUX-LNCMI/ALIM2_J1": "Courant_A3",
    "kHz/FEPC-AUX-LNCMI/ALIM2_J2": "Courant_A4",
}
```

**Note:** You should customize these mappings based on your specific setup and channel configurations.

## Data Directory Structure

The script expects the following directory structure (as defined in `python_magnetrun.analysis.config`):

### Hybrid Data
```
hybrid_data/
├── kHz/
│   └── YYYY-MM-DD/
│       ├── FEPC-LNCMI/
│       │   ├── 00h/
│       │   ├── 01h/
│       │   └── ...
│       └── FEPC-AUX-LNCMI/
│           ├── 00h/
│           └── ...
├── rms/
│   └── YYYY-MM-DD/
│       └── ...
└── trigger/
    └── TRIGGER__YYYY-MM-DD__HH-MM/
        └── ...
```

### Pupitre Data
```
pupitre_datadir/
├── M9/
│   ├── 2025.01.27---15:39:29.txt
│   ├── 2025.01.27---16:20:15.txt
│   └── ...
├── M10/
│   └── ...
└── M8/
    └── ...
```

### Pigbrother (TDMS) Data
```
pigbrother_datadir/
├── M9/
│   ├── Overview/
│   │   ├── M9_Overview_250127-1605.tdms
│   │   ├── M9_Overview_250127-1620.tdms
│   │   └── ...
│   ├── Fichiers_Archive/
│   └── ...
├── M10/
│   └── Overview/
│       └── ...
└── M8/
    └── Overview/
        └── ...
```

## Usage Examples

### Basic Usage

Plot FEPC-AUX-LNCMI data for a specific date:

```bash
python plot_hybrid_with_pupitre_tdms.py \
    -d 2025-11-02 \
    -s FEPC-AUX-LNCMI \
    -k ALIM1_J1 \
    --site M8 \
    --show
```

### Custom Data Directories

Specify custom paths for data directories:

```bash
python plot_hybrid_with_pupitre_tdms.py \
    -d 2025-01-27 \
    -s FEPC-LNCMI \
    -k I_H1 \
    --site M9 \
    --hybrid-dir /path/to/hybrid/data \
    --pupitre-dir /home/LNCMI-G/christophe.trophime/LNCMIG-Data/srv-data-install \
    --pigbrother-dir /path/to/pigbrother/Fichiers_Data \
    --show
```

### Plot Specific Hours

Plot only data from specific hours (useful for reducing data volume):

```bash
python plot_hybrid_with_pupitre_tdms.py \
    -d 2025-01-27 \
    -s FEPC-AUX-LNCMI \
    -k ALIM1_J1 \
    --site M10 \
    --hours 10,11,12 \
    --show
```

### Save Plot to File

Save the plot instead of showing it interactively:

```bash
python plot_hybrid_with_pupitre_tdms.py \
    -d 2025-01-27 \
    -s FEPC-AUX-LNCMI \
    -k ALIM1_J1 \
    --site M10 \
    --save comparison_plot.png
```

### Both Save and Show

```bash
python plot_hybrid_with_pupitre_tdms.py \
    -d 2025-01-27 \
    -s FEPC-AUX-LNCMI \
    -k ALIM1_J1 \
    --site M10 \
    --save comparison_plot.png \
    --show
```

## Command-Line Arguments

| Argument            | Required | Description                                                |
| ------------------- | -------- | ---------------------------------------------------------- |
| `-d, --date`        | Yes      | Date in YYYY-MM-DD format                                  |
| `-s, --fepc-system` | Yes      | FEPC system: `FEPC-LNCMI` or `FEPC-AUX-LNCMI`              |
| `-k, --key`         | Yes      | Variable name to plot (e.g., `ALIM1_J1`, `I_H1`)           |
| `--site`            | Yes      | Measurement site: `M8`, `M9`, or `M10`                     |
| `--hybrid-dir`      | No       | Base directory for hybrid data (default: `../hybrid_data`) |
| `--pupitre-dir`     | No       | Base directory for pupitre data (from config)              |
| `--pigbrother-dir`  | No       | Base directory for pigbrother data (from config)           |
| `--hours`           | No       | Comma-separated list of hours to plot (e.g., `0,1,2`)      |
| `--insert`          | No       | Insert name (default: `Unknown`)                           |
| `--show`            | No       | Show plot interactively                                    |
| `--save`            | No       | Save plot to file (provide filename)                       |

## How It Works

### 1. Date-Based File Discovery

The script uses the date to find corresponding files:

- **Pupitre files**: `M10/2025.01.27---*.txt`
- **TDMS files**: `M10/Overview/M10_Overview_250127-*.tdms`

### 2. Field Name Mapping

The script uses dictionaries to translate field names:

```python
# User specifies: -k ALIM1_J1
# Script constructs: "kHz/FEPC-AUX-LNCMI/ALIM1_J1"
# Maps to pupitre: "IH"
# Maps to TDMS: "Référence_GR1"
```

### 3. Data Loading

- **Hybrid data**: Uses `HybridRun.fromdir()` with optional downsampling
- **Pupitre data**: Uses `MagnetRun.fromtxt()`
- **TDMS data**: Uses `MagnetRun.fromtdms()`

### 4. Time Alignment

- Hybrid kHz data: Converted to relative seconds from start
- Pupitre data: Uses 't' field (already in seconds)
- TDMS data: Uses 't' field (already in seconds)

## Customization

### Adding New Field Mappings

Edit the dictionaries at the top of the script:

```python
HYBRID_TO_PUPITRE_MAP = {
    "kHz/FEPC-AUX-LNCMI/YOUR_CHANNEL": "PUPITRE_FIELD",
    # Add more mappings...
}

HYBRID_TO_TDMS_MAP = {
    "kHz/FEPC-AUX-LNCMI/YOUR_CHANNEL": "TDMS_FIELD",
    # Add more mappings...
}
```

### Site-Specific Mappings

For site-specific field mappings, you can check the site configuration:

```python
from python_magnetrun.analysis.config import SITE_CONFIGS

# Get M9 configuration
m9_config = SITE_CONFIGS["M9"]
print(f"M9 GR1 current channel: {m9_config.reference_gr1_current}")  # "IH"
print(f"M9 GR2 current channel: {m9_config.reference_gr2_current}")  # "IB"

# Get M10 configuration (different from M9!)
m10_config = SITE_CONFIGS["M10"]
print(f"M10 GR1 current channel: {m10_config.reference_gr1_current}")  # "IB"
print(f"M10 GR2 current channel: {m10_config.reference_gr2_current}")  # "IH"
```

## Troubleshooting

### No pupitre/TDMS files found

- Check that the date format matches your file naming convention
- Verify the data directories are correctly set
- Check that files exist for the specified date and site

### Field not found in data

- Check the available keys using: `print(data.getMData().getKeys())`
- Update the mapping dictionaries to match your actual field names
- Ensure the field exists in the source data

### Import errors

Make sure the hybrid module is in your Python path:

```bash
export PYTHONPATH=/path/to/python_magnetrun:$PYTHONPATH
```

Or run from the repository root:

```bash
cd /path/to/python_magnetrun
python examples/plot_hybrid_with_pupitre_tdms.py ...
```

---

## Additional Plotting Scripts for Hybrid data

### RMS Data Plotting (`plot_rms.py`)

Plot variables from RMS (root-mean-square) files.

#### Plot one or more variables (separate subplots)

```bash
python plot_rms.py path/to/file.rms I_H1 U_H1
```

#### Overlay all variables on the same axes (multiple y-axes)

```bash
python plot_rms.py path/to/file.rms I_H1 U_H1 --same-plot
```

### Save plot to file

```bash
python plot_rms.py path/to/file.rms I_H1 -o rms_plot.png
```

#### Command-Line Arguments

| Argument        | Required | Description                                               |
| --------------- | -------- | --------------------------------------------------------- |
| `file`          | Yes      | Path to the RMS file                                      |
| `variables`     | Yes      | One or more variable names to plot                        |
| `-o, --output`  | No       | Save plot to file (e.g., `output.png`)                    |
| `--same-plot`   | No       | Overlay all variables with independent y-axes             |

---

### kHz FEPC Data Plotting (`plot_fepc_data.py`)

Plot a variable from FEPC binary files using a CFG configuration file.

#### Basic usage

```bash
python plot_fepc_data.py -c /path/to/HOST_1_DATA.CFG -v ALIM1_J1
```

### Specify slot explicitly

```bash
python plot_fepc_data.py -c HOST_1_DATA.CFG -v I_H1 -s 4
```

#### Filter by date range

```bash
python plot_fepc_data.py -c HOST_1_DATA.CFG -v I_H1 -d 2025-11-05 2025-11-06
```

#### Remove outliers and compare before/after

```bash
python plot_fepc_data.py -c HOST_1_DATA.CFG -v I_H1 --remove-outliers iqr
```

#### Save plot to file

```bash
python plot_fepc_data.py -c HOST_1_DATA.CFG -v I_H1 -o fepc_plot.png
```

### Command-Line Arguments

| Argument               | Required | Description                                                      |
| ---------------------- | -------- | ---------------------------------------------------------------- |
| `-c, --cfg`            | Yes      | Path to `HOST_X_DATA.CFG` configuration file                     |
| `-v, --variable`       | Yes      | Variable name to plot (e.g., `ALIM1_J1`, `I_H1`)                |
| `-s, --slot`           | No       | Card slot number (searches all slots if omitted)                 |
| `-o, --output`         | No       | Save plot to file (PNG, PDF, …)                                  |
| `-d, --date`           | No       | Date range for files: `YYYY-MM-DD start end`                     |
| `-e, --endian`         | No       | `big` (default) or `little`                                      |
| `--cnv-dir`            | No       | Directory containing CNV calibration files (default: CFG dir)   |
| `--remove-outliers`    | No       | Outlier method: `iqr`, `zscore`, `mad`, or `percentile`         |
| `--outlier-threshold`  | No       | Threshold for outlier detection (default: `1.5`)                 |
| `--outlier-window`     | No       | Rolling window size for outlier detection                        |
| `--debug`              | No       | Plot data incrementally as each file is loaded                   |

---

### VProcess Data Plotting (`plot_vprocess.py`)

Plot variables from `.vprocess` slow-data files.

#### Plot specific variables (separate subplots)

```bash
python plot_vprocess.py data.vprocess --vars TT115A TT508A
```

#### Overlay variables on the same axes

```bash
python plot_vprocess.py data.vprocess --vars TT115A TT508A --layout overlay
```

#### Overview of first N analog variables

```bash
python plot_vprocess.py data.vprocess --overview --max-vars 8
```

#### Compare two variables (time series + scatter + histograms)

```bash
python plot_vprocess.py data.vprocess --compare TT115A TT508A
```

#### Correlation heatmap

```bash
python plot_vprocess.py data.vprocess --heatmap
python plot_vprocess.py data.vprocess --heatmap --vars TT115A TT508A TT600A
```

#### Resolve from hybrid data tree

```bash
python plot_vprocess.py --hybrid_datadir /mnt/LNCMIG-Data/records/CEA \
    --hybrid_date 2025-11-05 --vars TT115A TT508A
```

#### Save without displaying

```bash
python plot_vprocess.py data.vprocess --vars TT115A --save vprocess_plot.png --no-show
```

#### Command-Line Arguments

| Argument            | Required | Description                                                         |
| ------------------- | -------- | ------------------------------------------------------------------- |
| `input_file`        | No*      | Path to `.vprocess` file (required unless `--hybrid_date`)          |
| `--vars`            | No**     | Variable name(s) to plot                                            |
| `--overview`        | No**     | Plot overview of first N analog variables                           |
| `--compare VAR1 VAR2` | No**  | Compare two variables (time series, scatter, histograms)            |
| `--heatmap`         | No**     | Correlation heatmap for selected or first N variables               |
| `--max-vars`        | No       | Maximum variables for `--overview`/`--heatmap` (default: `10`)      |
| `--layout`          | No       | `subplots` (default) or `overlay` for `--vars`                      |
| `--hybrid_datadir`  | No       | Base data directory (used to resolve bare filenames)                |
| `--hybrid_date`     | No       | Date `YYYY-MM-DD` used when no `input_file` is given                |
| `-s, --save`        | No       | Save figure to file                                                 |
| `--no-show`         | No       | Do not open an interactive window                                   |

*Provide either `input_file` or `--hybrid_date`+`--hybrid_datadir`.
**At least one of `--vars`, `--overview`, `--compare`, or `--heatmap` is required.

---

## Trigger Data Plotting (`plot_trigger_data.py`)

Plot waveforms from FEPC trigger binary files stored under `trigger/TRIGGER__YYYY-MM-DD__HH-MM/`.

#### List available variables

```bash
python plot_trigger_data.py TRIGGER__2025-11-05__08-16 --list-variables
```

#### Plot a single variable

```bash
python plot_trigger_data.py TRIGGER__2025-11-05__08-16 --variable I_H1
```

#### Plot several variables (one figure per variable)

```bash
python plot_trigger_data.py TRIGGER__2025-11-05__08-16 --variable I_H1 I_BOB
```

#### Plot all triggers for a date

```bash
python plot_trigger_data.py TRIGGER__2025-11-05__08-16 --variable I_H1 --all
```

#### Save plots (per-variable suffix added automatically)

```bash
python plot_trigger_data.py TRIGGER__2025-11-05__08-16 \
    --variable I_H1 I_BOB \
    --save trigger_plot.png \
    --no-show
# Produces: trigger_plot_I_H1.png, trigger_plot_I_BOB.png
```

#### Command-Line Arguments

| Argument              | Required | Description                                                         |
| --------------------- | -------- | ------------------------------------------------------------------- |
| `input_dir`           | No       | Trigger directory name or path (required unless `--hybrid_date`)    |
| `--variable`          | No*      | Variable name(s) to plot; accepts multiple values                   |
| `--list-variables`    | No       | Print all variables available in the config and exit                |
| `--all`               | No       | Plot all trigger directories for the inferred date                  |
| `--fepc_system`       | No       | `FEPC-LNCMI` (default) or `FEPC-AUX-LNCMI`                         |
| `--hybrid_datadir`    | No       | Base data directory (used to resolve bare trigger directory names)  |
| `--hybrid_date`       | No       | Date `YYYY-MM-DD` used when no `input_dir` is given                 |
| `--endian`            | No       | `big` (default) or `little`                                         |
| `--no-calib`          | No       | Skip calibration (plot raw ADC values)                              |
| `--cnv-dir`           | No       | Directory containing CNV piecewise-calibration files                |
| `--save`              | No       | Save plot to file (per-variable suffix added when plotting several) |
| `--no-show`           | No       | Do not open an interactive window                                   |
| `--fig-size`          | No       | Figure size as `width,height` (default: `12,6`)                     |

*`--variable` is required unless `--list-variables` is specified.

## Related Documentation

- [Hybrid Module README](../python_magnetrun/hybrid/README.md)
- [Analysis Configuration](../python_magnetrun/analysis/config.py)
- [File Discovery](../python_magnetrun/analysis/loaders.py)

## License

Same as parent project.
