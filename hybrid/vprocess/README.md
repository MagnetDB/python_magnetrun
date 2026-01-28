# VProcess File Reader for LNCMI

A Python library for reading and analyzing VProcess data files from the LNCMI (Laboratoire National des Champs Magnétiques Intenses) control system.

## Overview

VProcess files are binary data files with ASCII headers that contain process monitoring data from LNCMI experiments. Unlike kHz and RMS data which are FEPC-specific, VProcess files contain general process variables from the control system.

### File Structure

- **Header**: 8 lines of ASCII text starting with `#` containing metadata
- **Binary Data**: Time-series data with timestamps and variable values
- **Sampling**: Typically 1 Hz (1 sample per second)
- **Duration**: Usually 1 hour per file (3600 samples)

**Data Format:**
- Timestamp: 8 bytes (float64, Unix epoch)
- Analog variables: 4 bytes each (float32)
- Digital variables: 1 byte each (uint8)
- Sample width = 8 + (N_analog × 4) + (N_digital × 1)

## Features

- ✅ Parse ASCII headers with metadata extraction
- ✅ Read binary data with proper type handling (float32 for analog, uint8 for digital)
- ✅ Support for both analog and digital variables
- ✅ Automatic timestamp conversion
- ✅ Export to pandas DataFrame
- ✅ Variable information extraction
- ✅ Time-based filtering
- ✅ Compatible with HybridRun interface

## Installation

### Requirements

```bash
pip install numpy pandas
```

## Quick Start

### Basic Usage

```python
from vprocess.vprocess_reader import read_vprocess_file

# Read VProcess file into a pandas DataFrame
df = read_vprocess_file('path/to/your/file.vprocess')

print(df.head())
print(df.info())
```

### Detailed Information

```python
from vprocess.vprocess_reader import VProcessFileReader

# Create reader instance
reader = VProcessFileReader('path/to/your/file.vprocess')

# Parse header only
reader.parse_header()

# Print file summary
reader.print_summary()

# Get variable information
var_info = reader.get_variable_info()
print(var_info)

# Get metadata
metadata = reader.get_metadata()
print(metadata)

# Read all data
df = reader.read()
```

## API Reference

### Main Classes

#### `VProcessFileReader`

Main class for reading VProcess files.

**Methods:**

- `parse_header()`: Parse the ASCII header and extract metadata
- `read_binary_data()`: Read and parse the binary data portion
- `read()`: Complete read operation (header + data)
- `get_variable_info()`: Return DataFrame with variable specifications
- `get_metadata()`: Return dictionary with file metadata
- `print_summary()`: Print formatted summary of file contents

**Properties:**

- `variables`: List of `VProcessVariable` objects
- `metadata`: Dictionary containing file metadata
- `data`: pandas DataFrame with the data (after reading)

#### `VProcessVariable`

Represents a single variable in the VProcess file.

**Attributes:**

- `name`: Variable name (str)
- `var_type`: Type ('float32' or 'dig')
- `unit`: Physical unit (str, optional)
- `min_val`: Minimum value (float, optional)
- `max_val`: Maximum value (float, optional)
- `display_format`: Display format string (str, optional)
- `is_analog`: Boolean indicating if analog variable
- `byte_size`: Size in bytes (4 for analog, 1 for digital)

### Convenience Functions

#### `read_vprocess_file(filepath, endian='little')`

Quick read function that returns a DataFrame directly.

**Parameters:**
- `filepath`: Path to VProcess file
- `endian`: 'big' or 'little' (default: 'little')

**Returns:** pandas DataFrame

#### `get_vprocess_info(filepath, endian='little')`

Get metadata and variable information without reading all data.

**Returns:** Tuple of (metadata_dict, variables_dataframe)

## Header Format

The VProcess header consists of 8 lines:

```
# vprocess data file - v3.0 (Version info)
# processed on ... (Processing info)
# header [encoding:UTF-8 - line-ending:unix]
# variables = VAR1 [type:float32|unit:K|min:0.00|max:325.00|df:%.2f]; VAR2 [...]
# windows = [UTC] DD/MM/YYYY-HH:MM:SS.mmm -> DD/MM/YYYY-HH:MM:SS.mmm
# frequency = 1.000 Hz
# timestamp = absolute
# data-helper [offset:0x2c8a - time:8(B) - width:764(B)]
```

**Key fields:**
- `variables`: Alphabetically sorted list of all variables with properties
- `windows`: Time range covered by the file
- `frequency`: Sampling frequency (typically 1 Hz)
- `data-helper`: Binary data parameters (offset, timestamp size, sample width)

## Binary Data Format

Each sample in the binary section contains:

```
┌──────────────┬─────────────┬─────────────┬────┬──────────────┐
│  Timestamp   │   Var_1     │   Var_2     │... │   Var_N      │
│   (8 bytes)  │  (4 bytes)  │  (4 bytes)  │    │  (1/4 bytes) │
└──────────────┴─────────────┴─────────────┴────┴──────────────┘
     float64      float32       float32           uint8/float32
```

## Integration with HybridRun

The VProcess reader can be integrated into the HybridRun interface:

```python
from hybrid.hybrid_run import HybridRun

# Load hybrid data including vprocess
hrun = HybridRun.fromdir("/data/hybrid", "2025-01-06")

# Access vprocess data
data = hrun.getData("vprocess/SYSTEM/VARIABLE")
```

## Command-Line Tools

### File Validation

```bash
# Basic validation
python validate.py data.vprocess

# Full validation with data checking
python validate.py data.vprocess --check-data

# Quiet mode (only errors)
python validate.py data.vprocess --quiet
```

### Batch Processing

```bash
# Merge all files in directory to CSV
python batch.py --dir ./data --output merged.csv --merge

# Export specific variables to HDF5
python batch.py --dir ./data --vars TT115A TT508A --format hdf5 --merge

# List common variables across all files
python batch.py --dir ./data --list-common-vars

# Analyze files and create summary
python batch.py --dir ./data --analyze --output summary.csv
```

### Data Visualization

```bash
# Plot specific variables
python plot_vprocess.py data.vprocess --vars TT115A TT508A

# Plot overview of first 10 variables
python plot_vprocess.py data.vprocess --overview

# Compare two variables
python plot_vprocess.py data.vprocess --compare TT115A TT508A

# Create correlation heatmap
python plot_vprocess.py data.vprocess --heatmap

# Save plot without displaying
python plot_vprocess.py data.vprocess --vars TT115A --save plot.png --no-show
```

### Unified CLI

```bash
# All operations through single interface
python cli.py info data.vprocess
python cli.py validate data.vprocess --check-data
python cli.py plot data.vprocess --vars TT115A TT508A
python cli.py batch --dir ./data --merge --output merged.csv
python cli.py test
```

### Testing

```bash
# Run all tests with mock data
python test.py

# Create a custom mock file
python test.py --create-mock --output test.vprocess --samples 1000

# Test with specific file
python test.py --test-file your_data.vprocess
```

## File Naming Convention

VProcess files follow a specific naming pattern with start and end timestamps:

**Format**: `YYYYMMDD_HHMMSS__YYYYMMDD_HHMMSS.vprocess`

**Examples**:
- `20251105_000000__20251105_005959.vprocess` - Nov 5, 2025, 00:00:00 to 00:59:59
- `20251105_060000__20251105_065959.vprocess` - Nov 5, 2025, 06:00:00 to 06:59:59
- `20251105_230000__20251105_235959.vprocess` - Nov 5, 2025, 23:00:00 to 23:59:59

A full day typically consists of 24 files (one per hour).

### Parsing Filenames

```python
from vprocess import parse_vprocess_filename

# Parse filename
filename = "20251105_000000__20251105_005959.vprocess"
start_time, end_time = parse_vprocess_filename(filename)

print(f"Start: {start_time}")  # 2025-11-05 00:00:00
print(f"End: {end_time}")      # 2025-11-05 00:59:59
```

### Finding Files by Date

```python
from datetime import datetime
from vprocess import find_vprocess_files_for_date

# Find all files for a specific date
date = datetime(2025, 11, 5)
files = find_vprocess_files_for_date('./data', date)

print(f"Found {len(files)} files for {date.date()}")
for file in files:
    print(f"  - {file.name}")
```

### Directory Organization

Files are typically organized by date:
```
vprocess/
├── 2025-11-05/
│   ├── 20251105_000000__20251105_005959.vprocess
│   ├── 20251105_010000__20251105_015959.vprocess
│   ├── ...
│   └── 20251105_230000__20251105_235959.vprocess
├── 2025-11-06/
│   └── ...
└── 2025-11-07/
    └── ...
```

## Comparison with Other LNCMI Formats

| Feature | VProcess | RMS | kHz |
|---------|----------|-----|-----|
| Frequency | ~1 Hz | ~10 Hz | kHz range |
| Source | Process system | FEPC RMS | FEPC kHz |
| Data type | Mixed | Mixed | Raw + Calibrated |
| File size | ~2.7 MB/hour | ~10 MB/hour | GB/hour |
| Calibration | Pre-converted | Pre-converted | Requires CNV |
| Use case | Process monitoring | Medium-freq monitoring | Fast transients |

## Examples

### Read and Plot

```python
import matplotlib.pyplot as plt
from vprocess.vprocess_reader import read_vprocess_file

# Read data
df = read_vprocess_file('data.vprocess')

# Plot a variable
plt.figure(figsize=(12, 4))
plt.plot(df.index, df['TT115A'])
plt.xlabel('Time')
plt.ylabel('TT115A')
plt.title('Temperature Sensor TT115A')
plt.grid(True)
plt.show()
```

### Time-based Analysis

```python
from vprocess.vprocess_reader import read_vprocess_file

df = read_vprocess_file('data.vprocess')

# Get statistics
print(df.describe())

# Filter by time
mask = (df.index >= '2025-01-06 10:00:00') & (df.index <= '2025-01-06 11:00:00')
df_filtered = df[mask]

# Resample to lower frequency
df_10s = df.resample('10S').mean()
```

### Batch Processing

```python
from pathlib import Path
from vprocess.vprocess_reader import read_vprocess_file
import pandas as pd

# Read multiple files
vprocess_dir = Path('vprocess/2025-01-06')
dfs = []

for file in sorted(vprocess_dir.glob('*.vprocess')):
    df = read_vprocess_file(file)
    dfs.append(df)

# Concatenate
df_day = pd.concat(dfs)
print(f"Total samples: {len(df_day)}")
print(f"Time range: {df_day.index[0]} to {df_day.index[-1]}")
```

## Troubleshooting

### Wrong Endianness

If data looks corrupted, try changing endianness:

```python
# Try little-endian (default)
df = read_vprocess_file('file.vprocess', endian='little')

# If that doesn't work, try big-endian
df = read_vprocess_file('file.vprocess', endian='big')
```

### Variable Not Found

Variables are case-sensitive and must match exactly:

```python
# Check available variables
from vprocess.vprocess_reader import VProcessFileReader

reader = VProcessFileReader('file.vprocess')
reader.parse_header()
var_info = reader.get_variable_info()
print(var_info['name'].tolist())
```

### Missing Data

Check the time window in metadata:

```python
reader = VProcessFileReader('file.vprocess')
reader.parse_header()
metadata = reader.get_metadata()
print(f"Start: {metadata['start_time']}")
print(f"End: {metadata['end_time']}")
```

## See Also

- [RMS README](../rms/README.md) - Similar format for FEPC RMS data
- [kHz README](../kHz/README.md) - High-frequency FEPC data
- [HybridRun](../hybrid_run.py) - Unified interface for all data types

---
*Last updated: January 14, 2026*
