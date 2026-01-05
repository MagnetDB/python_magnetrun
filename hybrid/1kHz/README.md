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

### 3. Extract Specific Variable

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

### `plot_fepc_data.py`
Plot specific variables from FEPC binary data files.

```bash
# Plot a variable with automatic slot detection
python plot_fepc_data.py -c HOST_2_DATA.CFG -v ALIM1_J1

# Specify slot explicitly
python plot_fepc_data.py -c HOST_2_DATA.CFG -v ALIM1_J1 -s 4

# Save plot to file
python plot_fepc_data.py -c HOST_2_DATA.CFG -v ALIM1_J1 -o output.png

# Debug mode with live plotting
python plot_fepc_data.py -c HOST_2_DATA.CFG -v ALIM1_J1 --debug

# Specify endianness and CNV directory
python plot_fepc_data.py -c HOST_2_DATA.CFG -v ALIM1_J1 -e little --cnv-dir /path/to/cnv
```

**Options:**
- `-c, --cfg`: Path to HOST_X_DATA.CFG file (required)
- `-v, --variable`: Variable name to plot (required)
- `-s, --slot`: Card slot number (optional, auto-detected)
- `-o, --output`: Output file for plot (PNG, PDF, etc.)
- `-e, --endian`: Endianness ('big' or 'little', default: big)
- `--debug`: Show live plot while loading data
- `--cnv-dir`: Directory containing CNV calibration files

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
- File naming patterns
- Exported CSV file with all variables

### `example_fepc_usage.py`
Comprehensive examples demonstrating all features:
- Reading configuration
- Reading single blocks
- Reading complete hour files
- Extracting specific variables
- Plotting data

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
    coef_a: float = 1.0    # Coefficient A for linear calibration
    coef_b: float = 0.0    # Coefficient B for linear calibration
    a: float = 1.0         # Scale factor A
    b: float = 0.0         # Offset B
    cnv_file: str = None   # Path to CNV file for piecewise calibration
    unit: str = None       # Physical unit (e.g., 'A', 'V', 'bar')
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
    variable_names: List[str]  # Variable names from CFG lines
    calibrations: List[CalibrationInfo]  # Calibration for each analog channel
    digital_info: List[Dict]   # Additional info for digital variables
    analog_info: List[Dict]    # Additional info for analog variables
```

**`FEPCConfig`**
```python
@dataclass
class FEPCConfig:
    fepc_name: str         # FEPC identifier
    num_cards: int         # Total number of cards
    cards: List[CardInfo]  # List of card configurations
    host_number: str       # HOST number from CFG filename
    digital_variables: Dict[int, List[str]]  # Slot -> variable names
    analog_variables: Dict[int, List[str]]   # Slot -> variable names

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

**`read_hour_file(filepath, card_type, num_blocks=72000, endian='big', debug=False)`**
- Read complete hour file (or specified number of blocks)
- `endian`: Byte order ('big' or 'little')
- `debug`: If True, shows live plot while reading
- Returns: numpy array
  - Analog: shape (N×50, 16), dtype=int16
  - Digital: shape (N×50, 32), dtype=bool

**`apply_calibration(raw_data, calib_info=None, cnv_dict=None)`**
- Apply calibration to convert raw values to physical units
- Linear: `Signal = A * (COEF_A * N + COEF_B) + B`
- Piecewise: Uses CNV file lookup table with interpolation

**`load_calibration(cnv_path)`**
- Load CNV file for piecewise linear calibration
- Returns: dict with 'n_values' and 'physical_values' arrays

**`calibrate_channel(raw_data, card, channel_idx, cnv_directory='.')`**
- Convenience function to calibrate a single channel
- Automatically selects calibration method

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
*Last updated: January 2026*
