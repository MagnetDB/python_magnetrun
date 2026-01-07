# Hybrid Data Module

Unified interface for hybrid magnet data from FEPC acquisition systems.

## Overview

This module provides a unified interface for reading and accessing three types of hybrid magnet data:
- **kHz**: High-frequency (1 kHz) data from FEPC analog and digital cards
- **RMS**: Root Mean Square data at lower frequency (typically 10 Hz)
- **Trigger**: Event-triggered data

## Features

- **Unified API**: Single `HybridData` class to access all data types
- **MagnetData-compatible**: Interface similar to the existing `MagnetData` class
- **Built-in plotting**: Plot kHz, RMS, or combined data with a single method call
- **Outlier removal**: Multiple methods for detecting and removing outliers (IQR, Z-score, MAD, percentile)
- **Calibration support**: Automatic calibration using CNV files
- **CLI interface**: Command-line tools for quick data exploration

## Directory Structure

The data is expected to be organized as follows:

```
base_dir/
├── kHz/
│   └── YYYY-MM-DD/
│       ├── FEPC-LNCMI/
│       │   ├── HOST_1_DATA.CFG
│       │   ├── 00HOST_1_LIST_0.bin
│       │   ├── 00HOST_1_LIST_1.bin
│       │   └── ...
│       └── FEPC-AUX-LNCMI/
├           |── HOST_2_DATA.CFG
│           ├── 00HOST_2_LIST_0.bin
│           ├── 00HOST_2_LIST_1.bin
│           └── ...
├── rms/
│   └── YYYY-MM-DD/
│       ├── FEPC-LNCMI/
│       |   └── FEPC-LNCMI_YYYY-MM-DD_0000—YYYY-MM-DD_0100.rms
|       |   └── ...
|       |   └── FEPC-LNCMI_YYYY-MM-DD_0000—YYYY-MM-DD+1_0000.rms
│       └── FEPC-AUX-LNCMI/
│           └── FEPC-AUX-LNCMI_YYYY-MM-DD_0000—YYYY-MM-DD_0100.rms
|           └── ...
|           └── FEPC-AUX-LNCMI_YYYY-MM-DD_0000—YYYY-MM-DD+1_0000.rms
└── trigger/
    └── TRIGGER__YYYY-MM-DD__HH-MM/
        ├── FEPC-LNCMI/
        └── FEPC-AUX-LNCMI/
```

## FEPC Systems

Two FEPC systems are supported:
- **FEPC-LNCMI**: Main FEPC system (6 analog + 2 digital cards)
- **FEPC-AUX-LNCMI**: Auxiliary FEPC system (3 analog + 2 digital cards)

## Quick Start

### Basic Usage

```python
from hybrid_data import HybridData

# Create instance for a specific date
data = HybridData(
    base_dir="/path/to/data",
    date_str="2025-01-06"
)

# Print summary
data.print_summary()

# Get available data keys
print(data.getKeys())
```

### Reading kHz Data

```python
# Get available kHz variables
vars = data.get_khz_variables("FEPC-LNCMI")
print("Analog:", vars['analog'])
print("Digital:", vars['digital'])

# Read a specific variable (with calibration)
values, time = data.read_khz_variable(
    system="FEPC-LNCMI",
    variable="ALIM1_J1",
    hours=[0, 1, 2],  # Optional: specific hours
    apply_calib=True
)

# Plot
import matplotlib.pyplot as plt
plt.plot(time, values)
plt.xlabel("Time (s)")
plt.ylabel("ALIM1_J1")
plt.show()
```

### Reading RMS Data

```python
# Get RMS variable information
vars_df = data.get_rms_variables("FEPC-LNCMI")
print(vars_df)

# Load RMS data as DataFrame
rms_df = data.load_rms_data("FEPC-LNCMI")
print(rms_df.head())
```

### MagnetData-Compatible Interface

The `HybridData` class provides an interface similar to `MagnetData`:

```python
# Get data using key
rms_data = data.getData("rms/FEPC-LNCMI")

# Get data type
print(data.getType())  # 3 for HybridData

# Get available keys
print(data.getKeys())
```

## Command Line Interface

### Basic Commands

```bash
# List available dates
python -m hybrid.hybrid_data --base-dir /data/hybrid --list-dates

# Show summary for a specific date
python -m hybrid.hybrid_data --base-dir /data/hybrid --date 2025-01-06

# Show kHz variables for a system
python -m hybrid.hybrid_data -d 2025-01-06 --khz-vars FEPC-LNCMI

# Show RMS variables for a system
python -m hybrid.hybrid_data -d 2025-01-06 --rms-vars FEPC-AUX-LNCMI
```

### Plotting Commands

```bash
# Plot a kHz variable
python -m hybrid.hybrid_data -d 2025-01-06 -s FEPC-LNCMI --plot-khz ALIM1_J1

# Plot specific hours without calibration
python -m hybrid.hybrid_data -d 2025-01-06 -s FEPC-LNCMI --plot-khz ALIM1_J1 --hours 0,1,2 --no-calib

# Plot an RMS variable
python -m hybrid.hybrid_data -d 2025-01-06 -s FEPC-LNCMI --plot-rms ALIM1_J1

# Plot kHz and RMS together
python -m hybrid.hybrid_data -d 2025-01-06 -s FEPC-LNCMI --plot-both ALIM1_J1

# Save plot to file
python -m hybrid.hybrid_data -d 2025-01-06 -s FEPC-LNCMI --plot-khz ALIM1_J1 --save output.png
```

### Outlier Removal

The module supports multiple outlier detection methods:

| Method | Description | Default Threshold |
|--------|-------------|-------------------|
| `iqr` | Interquartile Range | 1.5 (use 3.0 for extreme outliers) |
| `zscore` | Z-score based | 3.0 |
| `mad` | Median Absolute Deviation | 3.5 |
| `percentile` | Percentile-based clipping | 1.0 (clips 1% from each end) |

```bash
# Plot with IQR outlier removal (default threshold 1.5)
python -m hybrid.hybrid_data -d 2025-01-06 -s FEPC-LNCMI --plot-khz ALIM1_J1 --remove-outliers iqr

# Plot with Z-score method and custom threshold
python -m hybrid.hybrid_data -d 2025-01-06 -s FEPC-LNCMI --plot-khz ALIM1_J1 --remove-outliers zscore --outlier-threshold 3.0

# Plot with rolling window outlier removal (local detection)
python -m hybrid.hybrid_data -d 2025-01-06 -s FEPC-LNCMI --plot-khz ALIM1_J1 --remove-outliers mad --outlier-window 1000
```

When outlier removal is enabled, a side-by-side comparison plot is generated showing:
- Original data (left)
- Cleaned data with outliers removed (right)

## API Reference

### HybridData Class

```python
class HybridData:
    def __init__(
        self,
        base_dir: str,
        date_str: str,
        fepc_system: str = None,  # Optional: 'FEPC-LNCMI' or 'FEPC-AUX-LNCMI'
        endian: str = "big"
    )
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `FileName` | str | Identifier string |
| `Groups` | dict | Data groups by type |
| `Keys` | list | Available data keys |
| `Type` | int | Data type (3 for HybridData) |
| `Data` | dict | Loaded data |

#### Methods

| Method | Description |
|--------|-------------|
| `getType()` | Return data type identifier |
| `getKeys()` | Return list of available keys |
| `getInfo()` | Return HybridDataInfo object |
| `print_summary()` | Print data summary |
| `getData(key)` | Get data (MagnetData-compatible) |

#### kHz Methods

| Method | Description |
|--------|-------------|
| `load_khz_config(system)` | Load kHz configuration |
| `get_khz_variables(system)` | Get available variables |
| `read_khz_variable(system, variable, ...)` | Read variable data |
| `plot_khz_variable(system, variable, ...)` | Plot kHz data with optional outlier removal |

#### RMS Methods

| Method | Description |
|--------|-------------|
| `load_rms_data(system, file_idx)` | Load RMS data as DataFrame |
| `get_rms_variables(system, file_idx)` | Get variable information |
| `plot_rms_variable(system, variable, ...)` | Plot RMS data |

#### Combined Plotting

| Method | Description |
|--------|-------------|
| `plot_khz_with_rms(system, khz_variable, ...)` | Plot kHz and RMS data together |

#### Trigger Methods

| Method | Description |
|--------|-------------|
| `list_trigger_files(system)` | List available trigger files |

### Plotting Methods

```python
# Plot kHz variable
fig, ax = data.plot_khz_variable(
    system="FEPC-LNCMI",
    variable="ALIM1_J1",
    hours=[0, 1, 2],           # Optional: specific hours
    apply_calib=True,          # Apply calibration
    save="output.png",         # Save to file
    show=True                  # Display plot
)

# Plot with outlier removal
fig, ax = data.plot_khz_variable(
    system="FEPC-LNCMI",
    variable="ALIM1_J1",
    remove_outliers_method="iqr",  # 'iqr', 'zscore', 'mad', 'percentile'
    outlier_threshold=1.5,         # Threshold for detection
    outlier_window=None            # Optional: rolling window size
)

# Plot RMS variable
fig, ax = data.plot_rms_variable(
    system="FEPC-LNCMI",
    variable="ALIM1_J1",
    save="rms_output.png"
)

# Plot kHz and RMS together
fig, axes = data.plot_khz_with_rms(
    system="FEPC-LNCMI",
    khz_variable="ALIM1_J1",
    rms_variable="ALIM1_J1",  # Defaults to khz_variable
    hours=[0, 1, 2],
    save="combined.png"
)
```

### Outlier Removal Function

```python
from hybrid.hybrid_data import remove_outliers

# Remove outliers from data
clean_data, clean_time, n_outliers = remove_outliers(
    data=values,
    time=time_array,
    method="iqr",        # 'iqr', 'zscore', 'mad', 'percentile'
    threshold=1.5,       # Method-specific threshold
    window_size=None     # Optional: rolling window for local detection
)

print(f"Removed {n_outliers} outliers")
```

### Utility Functions

```python
# List available dates
from hybrid_data import list_available_dates

dates = list_available_dates("/data/hybrid", "kHz")
print(dates)  # ['2025-01-05', '2025-01-06', ...]
```

## Integration with MagnetRun

The `HybridData` class can be used alongside `MagnetData`:

```python
from python_magnetrun.magnetdata import MagnetData
from hybrid.hybrid_data import HybridData

# Load pupitre/pigbrother data
magnet_data = MagnetData.fromtdms("M9_Overview.tdms")

# Load hybrid data for the same day
hybrid_data = HybridData("/data/hybrid", "2025-01-06")

# Compare data from both sources
# ...
```

## Dependencies

- numpy
- pandas
- matplotlib (for plotting)
- fepc_reader (from hybrid/kHz)
- rms_reader (from hybrid/rms)

## CLI Options Reference

| Option | Short | Description |
|--------|-------|-------------|
| `--base-dir` | `-b` | Base directory with kHz, rms, trigger subdirectories |
| `--date` | `-d` | Date in YYYY-MM-DD format |
| `--fepc-system` | `-s` | FEPC system (FEPC-LNCMI or FEPC-AUX-LNCMI) |
| `--endian` | `-e` | Endianness of binary data (big/little) |
| `--list-dates` | | List available dates |
| `--khz-vars` | | Show kHz variables for a system |
| `--rms-vars` | | Show RMS variables for a system |
| `--plot-khz` | | Plot a kHz variable |
| `--plot-rms` | | Plot an RMS variable |
| `--plot-both` | | Plot kHz and RMS together |
| `--rms-var` | | RMS variable name for --plot-both |
| `--hours` | | Hours to plot (comma-separated) |
| `--no-calib` | | Skip calibration |
| `--save` | | Save plot to file |
| `--remove-outliers` | | Outlier removal method (iqr, zscore, mad, percentile) |
| `--outlier-threshold` | | Threshold for outlier detection |
| `--outlier-window` | | Rolling window size for local outlier detection |

## See Also

- [kHz README](kHz/README.md) - Detailed kHz data format documentation
- [RMS README](rms/README.md) - Detailed RMS data format documentation
- [MagnetData](../python_magnetrun/magnetdata.py) - Main MagnetData class

---
*Last updated: January 6, 2026*
