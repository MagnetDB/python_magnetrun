# FEPC kHz Data Reader

Python tools for reading and analyzing FEPC (Fast Event Protection Controller) kHz data files.

## Overview

This toolset reads high-frequency data from FEPC acquisition systems used at LNCMI. The system consists of:
- **Analog cards (MIVA)**: 16 channels, 16-bit resolution
- **Digital cards (MAD)**: 32 channels, 1-bit resolution
- **Sampling rate**: 10 kHz (configurable)
- **Data organization**: Binary files per hour per card

## File Structure

### Configuration File: `HOST_X_DATA.CFG`
Contains metadata about the FEPC setup:
- Line 1: Header with FEPC name and card configuration
- Lines 2-N: Variable names for each card (analog cards first, then digital cards)

### Data Files: `XXHOST_Y_LIST_Z.bin`
Binary data files organized as:
- `XX`: Hour (00-23)
- `Y`: FEPC number (1, 2, ...)
- `Z`: Slot number (0-7 for FEPC-LNCMI, 0-4 for FEPC-AUX-LNCMI)

Example:
- `00HOST_1_LIST_0.bin`: Hour 00, FEPC 1, Slot 0 (first analog card)
- `23HOST_1_LIST_7.bin`: Hour 23, FEPC 1, Slot 7 (last digital card)

### Data Structure

**Analog files (MIVA cards):**
- 72,000 blocks per hour (one block = 50 ms = 50 samples)
- Each block: 1,614 bytes
  - Header: 14 bytes (7 × 16-bit values)
  - Data: 1,600 bytes (50 samples × 16 channels × 16 bits)
- File size: ~113 MB per hour per card

**Digital files (MAD cards):**
- 72,000 blocks per hour
- Each block: 212 bytes
  - Header: 12 bytes (6 × 16-bit values)
  - Data: 200 bytes (50 samples × 32 bits)
- File size: ~14 MB per hour per card

## Installation

```bash
# Required packages
pip install numpy matplotlib
```

## Quick Start

### 1. Read Configuration

```python
from fepc_reader import parse_cfg_file

# Parse the CFG file
config = parse_cfg_file("HOST_1_DATA.CFG")

# Display basic info
print(f"FEPC: {config.fepc_name}")
print(f"Cards: {config.num_cards}")
print(f"Analog slots: {config.get_analog_slots()}")
print(f"Digital slots: {config.get_digital_slots()}")

# Get info for a specific slot
card = config.get_card_by_slot(0)
print(f"Slot 0 variables: {card.variable_names}")

# Check calibration info
for i, (var, calib) in enumerate(zip(card.variable_names, card.calibrations)):
    if calib.cnv_file:
        print(f"  {var}: Piecewise calibration ({calib.cnv_file})")
    else:
        print(f"  {var}: Linear calibration (A={calib.a:.3e}, B={calib.b:.3f})")
```

### 2. Read Data from a File

```python
from fepc_reader import read_hour_file

# Read complete hour of data (all 72,000 blocks)
data = read_hour_file("00HOST_1_LIST_0.bin", "ANA")
# Returns: numpy array, shape (3,600,000, 16) for analog

# For faster testing, read only first few blocks
data = read_hour_file("00HOST_1_LIST_0.bin", "ANA", num_blocks=100)
# Returns: numpy array, shape (5,000, 16) - 5 seconds of data
```

### 3. Apply Calibration (Convert to Physical Units)

```python
from fepc_reader import calibrate_channel

# Read raw data
raw_data = read_hour_file("00HOST_1_LIST_0.bin", "ANA", num_blocks=100)

# Get card info
card = config.get_card_by_slot(0)

# Calibrate a specific channel (e.g., channel 0)
channel_idx = 0
raw_channel = raw_data[:, channel_idx]

# Apply calibration (automatically uses correct method)
calibrated = calibrate_channel(raw_channel, card, channel_idx, cnv_directory=".")

print(f"Raw range: [{raw_channel.min()}, {raw_channel.max()}]")
print(f"Calibrated range: [{calibrated.min():.3f}, {calibrated.max():.3f}]")
```

### 4. Extract Specific Variable

```python
# Find which slot contains your variable
target_var = "DUP1_V1"

for card in config.cards:
    if target_var in card.variable_names:
        slot = card.slot
        channel = card.variable_names.index(target_var)
        print(f"Found in slot {slot}, channel {channel}")
        
        # Read the data
        filepath = f"00HOST_1_LIST_{slot}.bin"
        data = read_hour_file(filepath, card.card_type)
        
        # Extract the specific channel
        variable_data = data[:, channel]
        break
```

## Calibration

Analog channels use calibration to convert raw ADC values (16-bit integers) to physical units. Two calibration methods are supported:

### Linear Calibration

Formula: **Signal = A × (COEF_A × N + COEF_B) + B**

Where:
- N = raw ADC value (0-65535)
- COEF_A, COEF_B, A, B = calibration parameters from CFG file

**Example from documentation:**
```python
# Parameters: COEF_A=1, COEF_B=0, A=3.0518044E-4, B=-10
# N=0 → Signal = -10V
# N=65535 → Signal = +10V
```

### Piecewise Linear Calibration

Uses lookup table from .CNV files with linear interpolation between points.

**CNV file format:**
```
24578;250,0     # N=24578 → 250.0 mV
32123;20,0      # N=32123 → 20.0 mV
33435;-20,0     # N=33435 → -20.0 mV
40981;-250,0    # N=40981 → -250.0 mV
```

### Using Calibration

**Automatic calibration (recommended):**
```python
from fepc_reader import calibrate_channel

# This automatically selects the right calibration method
calibrated = calibrate_channel(
    raw_data=raw_channel,
    card=card,
    channel_idx=0,
    cnv_directory="."  # Directory containing .CNV files
)
```

**Manual calibration:**
```python
from fepc_reader import apply_calibration, load_calibration

# Linear calibration
calib_info = card.calibrations[channel_idx]
calibrated = apply_calibration(raw_data, calib_info=calib_info)

# Piecewise calibration
cnv_dict = load_calibration("variable_name.CNV")
calibrated = apply_calibration(raw_data, cnv_dict=cnv_dict)
```

### Calibration Information

Calibration parameters are stored in the `CardInfo.calibrations` list:

```python
card = config.get_card_by_slot(0)

for i, (var_name, calib) in enumerate(zip(card.variable_names, card.calibrations)):
    print(f"Channel {i}: {var_name}")
    
    if calib.cnv_file:
        print(f"  Type: Piecewise")
        print(f"  CNV file: {calib.cnv_file}")
    else:
        print(f"  Type: Linear")
        print(f"  Parameters: A={calib.a}, B={calib.b}, Ca={calib.coef_a}, Cb={calib.coef_b}")
```

### 4. Analyze CFG File

Use the dedicated analyzer tool:

```bash
python cfg_analyzer.py HOST_1_DATA.CFG
```

This will:
- Display FEPC structure
- Show all slots and variables
- Create a slot map
- Export variables to CSV

## Example Scripts

### `cfg_analyzer.py`
Dedicated tool to analyze CFG files and display slot configuration.

```bash
# Run with default file
python cfg_analyzer.py

# Run with specific file
python cfg_analyzer.py /path/to/HOST_1_DATA.CFG
```

**Output:**
- Complete slot information
- Variable lists for each slot
- Calibration parameters for analog channels
- File naming patterns
- Exported CSV file with all variables

### `calibration_demo.py`
Comprehensive calibration demonstration and analysis tool.

```bash
python calibration_demo.py
```

**Features:**
- Display all calibration parameters from CFG file
- Demonstrate linear calibration with examples
- Load and analyze CNV files (piecewise calibration)
- Apply calibration to real data
- Plot raw vs calibrated data comparison
- Export calibration info to CSV

**Output:**
- Calibration parameter tables
- Example calculations
- Visualization plots (if data files present)
- `calibration_info.csv` with all parameters

### `example_fepc_usage.py`
Comprehensive examples demonstrating all features:
- Reading configuration
- Reading single blocks
- Reading complete hour files
- Extracting specific variables
- Plotting data
- Applying calibration

```bash
python example_fepc_usage.py
```

## Module Reference

### `fepc_reader.py`

Main module with all core functionality.

#### Classes

**`CalibrationInfo`**
```python
@dataclass
class CalibrationInfo:
    coef_a: float = 1.0      # Linear coefficient A
    coef_b: float = 0.0      # Linear coefficient B  
    a: float = 1.0           # Scale factor
    b: float = 0.0           # Offset
    cnv_file: str = None     # CNV filename for piecewise calibration
```

**`CardInfo`**
```python
@dataclass
class CardInfo:
    slot: int              # Slot number (0-7)
    card_type: str         # 'ANA' or 'DIG'
    sampling_freq: int     # Sampling frequency (Hz)
    buffer_pre: int        # Pre-quench buffer (s)
    buffer_post: int       # Post-quench buffer (s)
    num_channels: int      # Number of channels (16 or 32)
    variable_names: List[str]      # Variable names
    calibrations: List[CalibrationInfo]  # Calibration for each channel (analog only)
```

**`FEPCConfig`**
```python
@dataclass
class FEPCConfig:
    fepc_name: str         # FEPC identifier
    num_cards: int         # Total number of cards
    cards: List[CardInfo]  # List of card configurations
    
    # Methods
    get_card_by_slot(slot)    # Get card info by slot number
    get_analog_slots()        # Get list of analog card slots
    get_digital_slots()       # Get list of digital card slots
```

#### Functions

**`parse_cfg_file(cfg_path)`**
- Parse HOST_X_DATA.CFG file
- Returns: `FEPCConfig` object

**`read_analog_block(file, block_idx)`**
- Read one analog block (50 samples, 16 channels)
- Returns: `(data, header)` where data is shape (50, 16)

**`read_digital_block(file, block_idx)`**
- Read one digital block (50 samples, 32 channels)
- Returns: `(data, header)` where data is shape (50, 32)

**`read_hour_file(filepath, card_type, num_blocks=72000)`**
- Read complete hour file (or specified number of blocks)
- Returns: numpy array
  - Analog: shape (N×50, 16), dtype=int16
  - Digital: shape (N×50, 32), dtype=bool

**Calibration Functions:**

**`load_calibration(cnv_path)`**
- Load piecewise calibration from .CNV file
- Returns: dict with 'n_values' and 'physical_values' arrays

**`apply_calibration(raw_data, calib_info=None, cnv_dict=None)`**
- Apply calibration to raw analog data
- Supports both linear and piecewise methods
- Returns: calibrated data as float array

**`calibrate_channel(raw_data, card, channel_idx, cnv_directory=".")`**
- Convenience function to calibrate a single channel
- Automatically selects appropriate calibration method
- Returns: calibrated data as float array

## Data Organization Example

**FEPC-LNCMI Configuration:**
- 6 analog cards (MIVA): Slots 0-5
- 2 digital cards (MAD): Slots 6-7

**Files for one hour (e.g., hour 14):**
```
14HOST_1_LIST_0.bin  → Slot 0 (Analog, 16 channels)
14HOST_1_LIST_1.bin  → Slot 1 (Analog, 16 channels)
14HOST_1_LIST_2.bin  → Slot 2 (Analog, 16 channels)
14HOST_1_LIST_3.bin  → Slot 3 (Analog, 16 channels)
14HOST_1_LIST_4.bin  → Slot 4 (Analog, 16 channels)
14HOST_1_LIST_5.bin  → Slot 5 (Analog, 16 channels)
14HOST_1_LIST_6.bin  → Slot 6 (Digital, 32 channels)
14HOST_1_LIST_7.bin  → Slot 7 (Digital, 32 channels)
```

Total: 8 files/hour × 24 hours = **192 files/day**

## Advanced Usage

### Reading Specific Time Range

```python
# Read data from hour 10, minute 30, to hour 10, minute 35
# Each minute = 600,000 samples at 10 kHz
start_sample = 30 * 600_000
end_sample = 35 * 600_000

# Read hour 10
data = read_hour_file("10HOST_1_LIST_0.bin", "ANA")

# Extract time range
data_range = data[start_sample:end_sample, :]
```

### Memory-Efficient Block Reading

For large files, read block by block:

```python
with open("00HOST_1_LIST_0.bin", 'rb') as f:
    for block_idx in range(72000):  # All blocks in hour
        data, header = read_analog_block(f, block_idx)
        
        # Process block (50 samples)
        # ... your processing code ...
        
        if block_idx % 1000 == 0:
            print(f"Processed {block_idx}/72000 blocks")
```

### Working with Multiple Slots

```python
# Read all analog cards for hour 00
analog_slots = config.get_analog_slots()
all_analog_data = {}

for slot in analog_slots:
    filepath = f"00HOST_1_LIST_{slot}.bin"
    data = read_hour_file(filepath, "ANA", num_blocks=100)
    all_analog_data[slot] = data
    print(f"Loaded slot {slot}: {data.shape}")
```

## Troubleshooting

### File Not Found
- Verify CFG file path
- Check data file naming convention
- Ensure files are in the correct directory

### Memory Issues
- Read fewer blocks: `read_hour_file(path, type, num_blocks=1000)`
- Process block by block using `read_analog_block()` or `read_digital_block()`

### Parsing Errors
- Verify CFG file format matches expected structure
- Check for special characters or encoding issues

## Performance Notes

**Memory requirements:**
- One hour analog file in memory: ~110 MB per slot
- One hour digital file in memory: ~14 MB per slot
- Full FEPC-LNCMI hour (8 cards): ~700 MB

**Reading speed:**
- Full hour file: ~1-3 seconds per slot (depends on disk I/O)
- Single block: <1 ms

## Data Files Summary

| System | Analog Cards | Digital Cards | Files/Day | Total/Day |
|--------|-------------|---------------|-----------|-----------|
| FEPC-LNCMI | 6 | 2 | 192 | ~20 GB |
| FEPC-AUX-LNCMI | 3 | 2 | 120 | ~13 GB |

## Contact

For questions or issues, refer to the original documentation from DRF/Irfu/DIS.

---
*Last updated: December 2025*
