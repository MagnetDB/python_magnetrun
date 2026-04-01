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

## Related Documentation

- [Hybrid Module README](../python_magnetrun/hybrid/README.md)
- [Analysis Configuration](../python_magnetrun/analysis/config.py)
- [File Discovery](../python_magnetrun/analysis/loaders.py)

## License

Same as parent project.
