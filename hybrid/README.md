# Hybrid Data Module

Unified interface for hybrid magnet data from FEPC acquisition systems.

## Overview

This module provides a unified interface for reading and accessing three types of hybrid magnet data:
- **kHz**: High-frequency (1 kHz) data from FEPC analog and digital cards
- **RMS**: Root Mean Square data at lower frequency (typically 10 Hz)
- **Trigger**: Event-triggered data

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
│       │   └── *.rms
│       └── FEPC-AUX-LNCMI/
│           └── *.rms
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

```bash
# List available dates
python hybrid_data.py --base-dir /data/hybrid --list-dates

# Show summary for a specific date
python hybrid_data.py --base-dir /data/hybrid --date 2025-01-06

# Show kHz variables for a system
python hybrid_data.py --base-dir /data/hybrid --date 2025-01-06 --khz-vars FEPC-LNCMI

# Show RMS variables for a system
python hybrid_data.py --base-dir /data/hybrid --date 2025-01-06 --rms-vars FEPC-AUX-LNCMI
```

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

#### RMS Methods

| Method | Description |
|--------|-------------|
| `load_rms_data(system, file_idx)` | Load RMS data as DataFrame |
| `get_rms_variables(system, file_idx)` | Get variable information |

#### Trigger Methods

| Method | Description |
|--------|-------------|
| `list_trigger_files(system)` | List available trigger files |

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
- matplotlib (for plotting examples)
- fepc_reader (from hybrid/kHz)
- rms_reader (from hybrid/rms)

## See Also

- [kHz README](kHz/README.md) - Detailed kHz data format documentation
- [RMS README](rms/README.md) - Detailed RMS data format documentation
- [MagnetData](../python_magnetrun/magnetdata.py) - Main MagnetData class

---
*Last updated: January 2026*
