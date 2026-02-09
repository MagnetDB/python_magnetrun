# FEPC Trigger Data Reader

Python tools for reading and analyzing FEPC trigger data files from LNCMI experiments.

## Overview

This toolset reads trigger event data from FEPC (Fast Event Protection Controller) acquisition systems. Trigger files capture:
- **PRE window**: 20 seconds of data before the trigger
- **POST window**: 50 seconds of data after the trigger
- **Total duration**: 70 seconds at 10 kHz = 700,000 samples
- **Data types**: Analog (MIVA, 16-bit) and Digital (MAD, 32-bit)

## File Structure

### Directory Organization

```
trigger/
└── TRIGGER_YYYY-MM-DD_HH-MM/
    ├── FEPC-LNCMI/
    │   ├── HOST_1_DATA.CFG          # Configuration file
    │   ├── EventInfo.properties     # Trigger metadata
    │   ├── host_1_trig_0.bin        # Analog data (slot 0)
    │   ├── host_1_trig_1.bin        # Analog data (slot 1)
    │   ├── ...
    │   ├── DUP9_V2.CNV              # Calibration files
    │   └── ...
    └── FEPC-AUX-LNCMI/
        ├── HOST_2_DATA.CFG
        ├── EventInfo.properties
        ├── host_2_trig_0.bin
        └── ...
```

### EventInfo.properties

Contains trigger event metadata:
```properties
# ASNet trigger properties
storage.version=3
trig.sample.idx=199999                # Sample index in kHz data
trig.timestamp.approx=06/06/2025-05:44:16.921 UTC[+0000]
trig.timestamp.approx.s=1749188656
trig.timestamp.approx.ms=921
trig.rtblock.id=1664719               # Correspondence with kHz data
trig.rtblock.phase=17
```

### Binary File Format

**Header (8 bytes):**
- Trigger timestamp (ms since 1970-01-01, big endian uint64)

**Analog Files (MIVA):**
- File size: 22,400,008 bytes (700,000 × 16 × 2 + 8)
- Format: 16 channels, 16-bit unsigned integers
- Encoding: IEEE big endian

**Digital Files (MAD):**
- File size: 2,800,008 bytes (700,000 × 4 + 8)
- Format: 32 channels, packed in 32-bit unsigned integers
- Encoding: IEEE big endian

## Installation

```bash
# Required packages
pip install numpy matplotlib pandas
```

## Quick Start

### 1. List Available Triggers

```python
from trigger_reader import find_trigger_directories

# Find all triggers
triggers = find_trigger_directories(Path("/data/hybrid"))
print(f"Found {len(triggers)} triggers")

# Find triggers for specific date
triggers = find_trigger_directories(Path("/data/hybrid"), date="2025-11-05")
```

### 2. Read Trigger Information

```python
from trigger_reader import parse_trigger_directory

trigger_dir = Path("/data/hybrid/trigger/TRIGGER_2025-11-05_08-16")
trigger_info = parse_trigger_directory(trigger_dir)

print(f"Timestamp: {trigger_info.timestamp}")
print(f"Sample index: {trigger_info.sample_idx}")
print(f"RT Block ID: {trigger_info.rtblock_id}")
```

### 3. Read Trigger Data

```python
from trigger_reader import read_trigger_data

# Read specific variable
data, timestamp, config = read_trigger_data(
    trigger_dir,
    system="FEPC-LNCMI",
    variable_name="I_H1"
)

print(f"Data shape: {data.shape}")
print(f"Trigger timestamp: {timestamp}")

# Read entire slot
data, timestamp, config = read_trigger_data(
    trigger_dir,
    system="FEPC-LNCMI",
    slot=0
)
# Returns: (700000, 16) for analog card
```

### 4. Create Time Array

```python
from trigger_reader import create_time_array

# Create time array (700,000 samples at 10 kHz)
time = create_time_array(700000)

# Time relative to trigger (trigger at 20s)
time_rel = time - 20.0
```

### 5. Plot Trigger Data

```python
from plot_trigger_data import plot_trigger_variable

plot_trigger_variable(
    trigger_dir,
    system="FEPC-LNCMI",
    variable="I_H1",
    show_plot=True,
    apply_calib=True
)
```

## Command-Line Usage

### trigger_reader.py

**List triggers:**
```bash
# All triggers
python trigger_reader.py --base-dir /data/hybrid --list-triggers

# Triggers for specific date
python trigger_reader.py --base-dir /data/hybrid --list-triggers --date 2025-11-05
```

**Show trigger info:**
```bash
python trigger_reader.py --trigger-dir /data/hybrid/trigger/TRIGGER_2025-11-05_08-16 --info
```

**Read data:**
```bash
# Read specific variable
python trigger_reader.py --trigger-dir /data/hybrid/trigger/TRIGGER_2025-11-05_08-16 \
                         --system FEPC-LNCMI --variable I_H1

# Read specific slot
python trigger_reader.py --trigger-dir /data/hybrid/trigger/TRIGGER_2025-11-05_08-16 \
                         --system FEPC-LNCMI --slot 0
```

### plot_trigger_data.py

**Plot single trigger:**
```bash
python plot_trigger_data.py --trigger-dir /data/hybrid/trigger/TRIGGER_2025-11-05_08-16 \
                             --system FEPC-LNCMI --variable I_H1
```

**Plot all triggers for a date:**
```bash
python plot_trigger_data.py --base-dir /data/hybrid --date 2025-11-05 \
                             --system FEPC-LNCMI --variable I_H1 --all
```

**Save plot:**
```bash
python plot_trigger_data.py --trigger-dir /data/hybrid/trigger/TRIGGER_2025-11-05_08-16 \
                             --system FEPC-LNCMI --variable I_H1 \
                             --save trigger_I_H1.png
```

**Skip calibration:**
```bash
python plot_trigger_data.py --trigger-dir /data/hybrid/trigger/TRIGGER_2025-11-05_08-16 \
                             --system FEPC-LNCMI --variable I_H1 --no-calib
```

## Data Characteristics

### Sampling
- **Frequency**: 10 kHz (0.1 ms per sample)
- **PRE window**: 200,000 samples (20 seconds)
- **POST window**: 500,000 samples (50 seconds)
- **Total**: 700,000 samples (70 seconds)

### Time Reference
The trigger point occurs at `PRE` seconds (20s) into the data array:
- `time[0]`: 20 seconds before trigger
- `time[200000]`: Trigger point (t=0)
- `time[699999]`: 50 seconds after trigger

### Data Quality
- **Resolution**: 16-bit for analog (0-65535)
- **Calibration**: Linear or piecewise (CNV files)
- **Units**: Physical units after calibration (A, V, bar, etc.)

## Calibration

### Linear Calibration
Applied using coefficients from CFG file:
```python
physical_value = a * raw_value + b
```

### Piecewise Calibration (CNV Files)
For non-linear sensors, CNV files provide lookup tables:
```python
# CNV file format: raw_value, physical_value
0,     0.0
1000,  1.234
2000,  2.567
...
```

### Manual Calibration
```python
from plot_trigger_data import apply_calibration
from fepc_reader import CalibrationInfo

# Linear calibration
calib = CalibrationInfo(a=0.001, b=0.0, unit='A')
calibrated_data = apply_calibration(raw_data, calib)

# Piecewise calibration
calib = CalibrationInfo(cnv_file='SENSOR.CNV', unit='bar')
calibrated_data = apply_calibration(raw_data, calib, cnv_dir=Path('/path/to/cnv'))
```

## Integration with kHz Data

Trigger events can be correlated with continuous kHz data using the `rtblock.id` and `rtblock.phase` fields from EventInfo.properties:

```python
from trigger_reader import parse_trigger_directory

# Get trigger metadata
trigger_info = parse_trigger_directory(trigger_dir)

# Find corresponding kHz block
rtblock_id = trigger_info.rtblock_id      # e.g., 1664719
rtblock_phase = trigger_info.rtblock_phase  # e.g., 17

# Calculate kHz timestamp
# Each block = 50 samples at 10 kHz = 5 ms
# rtblock.id is the block number in kHz data
kHz_time_offset = rtblock_id * 0.005  # seconds from start of day
```

## Advanced Usage

### Read Multiple Triggers

```python
from trigger_reader import find_trigger_directories, read_trigger_data
import pandas as pd

# Find all triggers for a date
triggers = find_trigger_directories(Path("/data/hybrid"), "2025-11-05")

# Read same variable from all triggers
all_data = []
for trigger_dir in triggers:
    data, timestamp, config = read_trigger_data(
        trigger_dir,
        system="FEPC-LNCMI",
        variable_name="I_H1"
    )
    all_data.append({
        'timestamp': timestamp,
        'data': data,
        'trigger_dir': trigger_dir.name
    })
```

### Compare PRE and POST Windows

```python
# Read trigger data
data, timestamp, config = read_trigger_data(trigger_dir, "FEPC-LNCMI", "I_H1")

# Split into PRE and POST
pre_data = data[:200000]   # 20s before trigger
post_data = data[200000:]  # 50s after trigger

# Analyze
print(f"PRE mean: {pre_data.mean():.3f}")
print(f"POST mean: {post_data.mean():.3f}")
print(f"PRE std: {pre_data.std():.3f}")
print(f"POST std: {post_data.std():.3f}")
```

### Custom Time Windows

```python
# Read data
data, timestamp, config = read_trigger_data(trigger_dir, "FEPC-LNCMI", "I_H1")
time = create_time_array(len(data))

# Select specific time window (e.g., -5s to +10s around trigger)
start_time = 15.0  # PRE time - 5s
end_time = 30.0    # PRE time + 10s
mask = (time >= start_time) & (time <= end_time)

windowed_data = data[mask]
windowed_time = time[mask]
```

## Module Reference

### trigger_reader.py

**Key Functions:**
- `parse_trigger_directory(trigger_dir)`: Parse trigger metadata
- `read_trigger_file(filepath, card_type, endian)`: Read binary file
- `read_trigger_data(trigger_dir, system, variable_name, slot)`: Read and extract data
- `load_trigger_config(trigger_dir, system)`: Load FEPC configuration
- `list_trigger_files(trigger_dir, system)`: List available binary files
- `create_time_array(num_samples, sampling_freq)`: Create time array
- `find_trigger_directories(base_dir, date)`: Find trigger directories

**Data Classes:**
- `TriggerInfo`: Trigger event metadata
- `TriggerFileInfo`: Binary file information

### plot_trigger_data.py

**Key Functions:**
- `plot_trigger_variable(trigger_dir, system, variable, ...)`: Plot single trigger
- `plot_multiple_triggers(base_dir, date, system, variable, ...)`: Plot all triggers for date
- `apply_calibration(data, calib, cnv_dir)`: Apply calibration to raw data

## Troubleshooting

### File Not Found
```
Error: System directory not found: /data/hybrid/trigger/TRIGGER_2025-11-05_08-16/FEPC-LNCMI
```
**Solution**: Check that the trigger directory contains the correct FEPC system subdirectory.

### Variable Not Found
```
Error: Variable 'I_H1' not found in config
```
**Solution**: List available variables using `--info` flag or check CFG file.

### File Size Mismatch
```
Warning: File size mismatch for host_1_trig_0.bin: expected 22400008, got 22400000
```
**Solution**: Some files may be truncated. Check file integrity.

### Calibration Error
```
Warning: Failed to apply CNV calibration: [Errno 2] No such file or directory
```
**Solution**: Ensure CNV files are in the correct directory, or use `--cnv-dir` to specify location.

## See Also

- [kHz README](../kHz/README.md) - Continuous kHz data format
- [RMS README](../rms/README.md) - RMS data format
- [Hybrid README](../README.md) - Main hybrid module documentation

---
*Last updated: January 14, 2026*
