# FEPC Calibration Quick Reference

## Overview

Analog FEPC channels convert raw 16-bit ADC values (N = 0-65535) to physical units using calibration. Two methods are supported:

## 1. Linear Calibration

**Formula:** `Signal = A × (COEF_A × N + COEF_B) + B`

**Parameters** (from CFG file):
- `COEF_A`, `COEF_B`: Linear coefficients
- `A`, `B`: Scale and offset

**Example from documentation:**
```
COEF_A = 1.0
COEF_B = 0.0
A = 3.0518044E-4
B = -10.0

Results:
N = 0     → Signal = -10.0 V
N = 32768 → Signal = 0.0 V
N = 65535 → Signal = +10.0 V
```

**Usage:**
```python
from fepc_reader import parse_cfg_file, apply_calibration

config = parse_cfg_file("HOST_1_DATA.CFG")
card = config.get_card_by_slot(0)
calib = card.calibrations[0]  # First channel

# Apply calibration
calibrated = apply_calibration(raw_data, calib_info=calib)
```

## 2. Piecewise Linear Calibration

Uses lookup table from .CNV files with linear interpolation.

**CNV File Format:**
```
N1;Physical_Value1
N2;Physical_Value2
N3;Physical_Value3
...
```

**Example (CLN_1.CNV):**
```
24578;250,0     # At N=24578, physical value is 250.0 mV
32123;20,0      # At N=32123, physical value is 20.0 mV  
33435;-20,0     # At N=33435, physical value is -20.0 mV
40981;-250,0    # At N=40981, physical value is -250.0 mV
```

**Usage:**
```python
from fepc_reader import load_calibration, apply_calibration

# Load CNV file
cnv_dict = load_calibration("CLN_1.CNV")

# Apply calibration
calibrated = apply_calibration(raw_data, cnv_dict=cnv_dict)
```

## 3. Automatic Calibration (Recommended)

The `calibrate_channel()` function automatically selects the correct method:

```python
from fepc_reader import parse_cfg_file, read_hour_file, calibrate_channel

# Load configuration
config = parse_cfg_file("HOST_1_DATA.CFG")

# Read data
data = read_hour_file("00HOST_1_LIST_0.bin", "ANA", num_blocks=100)

# Get card info
card = config.get_card_by_slot(0)

# Calibrate channel 0 (automatically uses CNV file if available)
calibrated = calibrate_channel(
    raw_data=data[:, 0],
    card=card,
    channel_idx=0,
    cnv_directory="."  # Directory with .CNV files
)
```

## Calibration Information Storage

### In CFG File

Each variable line may contain:
```
NAME=DUP9_V1;COEFA=1.0;COEFB=-0.0;FILE_PROC=D:\WWW\HTML\FRONTAUX\DUP9_V1.CNV;...
```

This information is automatically parsed into the `CalibrationInfo` object.

### In Code

```python
# Access calibration info
card = config.get_card_by_slot(0)

for i, (var_name, calib) in enumerate(zip(card.variable_names, card.calibrations)):
    print(f"Channel {i}: {var_name}")
    
    if calib.cnv_file:
        print(f"  Type: Piecewise (CNV file: {calib.cnv_file})")
    else:
        print(f"  Type: Linear")
        print(f"  A = {calib.a}")
        print(f"  B = {calib.b}")
        print(f"  COEF_A = {calib.coef_a}")
        print(f"  COEF_B = {calib.coef_b}")
```

## Complete Example Workflow

```python
from fepc_reader import parse_cfg_file, read_hour_file, calibrate_channel
import numpy as np

# 1. Load configuration
config = parse_cfg_file("HOST_1_DATA.CFG")

# 2. Find variable of interest
target_var = "DUP1_V1"
for card in config.cards:
    if target_var in card.variable_names:
        slot = card.slot
        channel = card.variable_names.index(target_var)
        break

# 3. Read raw data
filepath = f"00HOST_1_LIST_{slot}.bin"
raw_data = read_hour_file(filepath, "ANA", num_blocks=1000)
raw_channel = raw_data[:, channel]

# 4. Apply calibration
card = config.get_card_by_slot(slot)
calibrated = calibrate_channel(raw_channel, card, channel, cnv_directory=".")

# 5. Analyze
print(f"Variable: {target_var}")
print(f"Raw range: [{raw_channel.min()}, {raw_channel.max()}]")
print(f"Calibrated range: [{calibrated.min():.3f}, {calibrated.max():.3f}]")
print(f"Mean value: {calibrated.mean():.3f}")
```

## Tools

### View Calibration Parameters

```bash
# Display all calibration info from CFG file
python cfg_analyzer.py HOST_1_DATA.CFG
```

### Calibration Demo

```bash
# Comprehensive calibration demonstration
python calibration_demo.py
```

This will:
- Show all calibration parameters
- Demonstrate linear calibration
- Load and display CNV files
- Apply calibration to real data
- Generate comparison plots
- Export calibration info to CSV

## Key Points

✓ **Analog channels only** - Digital channels have no calibration  
✓ **Automatic selection** - Use `calibrate_channel()` for convenience  
✓ **CNV files** - Place .CNV files in same directory as your script  
✓ **Linear fallback** - If CNV file not found, uses linear calibration  
✓ **Parameters in CFG** - All calibration info is in HOST_X_DATA.CFG  

## Typical Use Cases

### Case 1: Simple Voltage Reading
```python
# For a simple voltage signal with linear calibration
calibrated_voltage = calibrate_channel(raw_data, card, channel, ".")
```

### Case 2: Multiple Channels
```python
# Calibrate all channels in a slot
calibrated_data = np.zeros_like(raw_data, dtype=float)
for ch in range(card.num_channels):
    calibrated_data[:, ch] = calibrate_channel(raw_data[:, ch], card, ch, ".")
```

### Case 3: Time Series Analysis
```python
# Read, calibrate, and analyze time series
time = np.arange(len(calibrated)) / 10000.0  # 10 kHz → seconds
plt.plot(time, calibrated)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
```

---
*Updated: December 2025*
