# RMS File Reader for FEPC-AUX-LNCMI

A Python library for reading and analyzing RMS (Root Mean Square) data files from the LNCMI (Laboratoire National des Champs Magnétiques Intenses) control system, specifically for FEPC-AUX-LNCMI format.

## Overview

RMS files are binary data files with ASCII headers that contain magnetometer and sensor data from LNCMI experiments. The files include:
- **Header**: 8 lines of ASCII text starting with `#` containing metadata
- **Binary Data**: Time-series data with timestamps and variable values

### File Structure

**FEPC-AUX-LNCMI Configuration:**
- 2 MAD cards (32 channels each = 64 digital signals)
- 3 MIVA cards (16 channels each = 48 analog signals)
- Sample size: 8 bytes (timestamp) + variable data
- Actual width: 257 bytes (7 unnamed digital signals excluded)

## Features

- ✅ Parse ASCII headers with metadata extraction
- ✅ Read binary data with proper type handling (float32 for analog, bit for digital)
- ✅ Support for both analog and digital variables
- ✅ Automatic timestamp conversion
- ✅ Export to pandas DataFrame
- ✅ Variable information extraction
- ✅ Time-based filtering
- ✅ Batch processing support
- ✅ Data export (CSV, Excel, HDF5)

## Installation

### Requirements

```bash
pip install numpy pandas
```

### Optional dependencies

```bash
# For Excel export
pip install openpyxl

# For HDF5 export
pip install tables

# For plotting
pip install matplotlib
```

## Quick Start

### Basic Usage

```python
from rms_reader import read_rms_file

# Read RMS file into a pandas DataFrame
df = read_rms_file('path/to/your/file.rms')

print(df.head())
print(df.info())
```

### Detailed Information

```python
from rms_reader import RMSFileReader

# Create reader instance
reader = RMSFileReader('path/to/your/file.rms')

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

#### `RMSFileReader`

Main class for reading RMS files.

**Methods:**

- `parse_header()`: Parse the ASCII header and extract metadata
- `read_binary_data()`: Read and parse the binary data portion
- `read()`: Complete read operation (header + data)
- `get_variable_info()`: Return DataFrame with variable specifications
- `get_metadata()`: Return dictionary with file metadata
- `print_summary()`: Print formatted summary of file contents

**Properties:**

- `variables`: List of `RMSVariable` objects
- `metadata`: Dictionary containing file metadata
- `data`: pandas DataFrame with the data (after reading)

#### `RMSVariable`

Represents a single variable in the RMS file.

**Attributes:**

- `name`: Variable name
- `var_type`: Type ('float32' or 'bit')
- `unit`: Unit of measurement (for analog variables)
- `min_val`: Minimum value
- `max_val`: Maximum value
- `display_format`: Display format string
- `is_analog`: Boolean indicating if variable is analog
- `byte_size`: Size in bytes (4 for analog, 1 for digital)

### Convenience Functions

```python
# Quick read
df = read_rms_file(filepath)

# Get info without reading data
metadata, var_info = get_rms_info(filepath)
```

## Header Format

The header contains 8 lines with the following information:

1. `# rms data file` - File type identifier
2. `# processed on ...` - Processing information
3. `# header [encoding:...]` - Encoding information
4. `# format = ...` - Data format specification
5. `# variables = ...` - Variable definitions (names, types, units, ranges)
6. `# windows = ...` - Time window (start and end times)
7. `# frequency = ...` - Sampling frequency in Hz
8. `# data-helper [...]` - Binary data structure information

### Variable Definition Format

Each variable is defined as:
```
NAME [type:TYPE|unit:UNIT|min:MIN|max:MAX|df:FORMAT]
```

**Analog variables (float32):**
```
PT205 [type:float32|unit:mBar|min:-4056.256|max:4056.256|df:%.3f]
```

**Digital variables (bit):**
```
MSS_OK_D [type:bit|0:OFF|1:ON]
```

## Data Structure

### DataFrame Structure

The returned DataFrame has:
- **Index**: Timestamps (datetime64[ns, UTC])
- **Columns**: Variable names
- **Values**: 
  - Analog variables: float32 values
  - Digital variables: uint8 (0 or 1)

### Variable Naming Convention

Based on the LNCMI system:

**Temperature sensors:**
- `TT###` or `TT###A`: Temperature readings
- `TT###_D1`: First alarm threshold (SD)
- `TT###_D2`: Second alarm threshold (FD)

**Pressure sensors:**
- `PT###`: Pressure readings
- `PT###_D1`, `PT###_D2`: Alarm thresholds

**Voltage/Power:**
- `PH_V##`: Phase voltages
- `PW_*`: Power-related signals
- `ALIM*`: Power supply signals

**Digital status:**
- `*_D`: Digital signals (status, alarms, controls)
- `*_OK_D`: Status OK indicators
- `*_DEF_D`: Fault indicators

**Magnetic field:**
- `BITTER_V#`: Bitter magnet voltages
- `MAGNET_*`: Magnet-related signals

## Usage Examples

### Example 1: Basic Data Reading

```python
from rms_reader import read_rms_file
import matplotlib.pyplot as plt

# Read data
df = read_rms_file('data.rms')

# Plot a temperature sensor
df['TT200A'].plot()
plt.ylabel('Temperature (K)')
plt.title('Temperature vs Time')
plt.grid(True)
plt.show()
```

### Example 2: Analyze Digital Signals

```python
from rms_reader import RMSFileReader

reader = RMSFileReader('data.rms')
df = reader.read()

# Get all digital variables
var_info = reader.get_variable_info()
digital_vars = var_info[var_info['type'] == 'bit']['name'].tolist()

# Check which alarms were triggered
for var in digital_vars:
    if var.endswith('_D1') or var.endswith('_D2'):
        if df[var].sum() > 0:
            print(f"Alarm triggered: {var} ({df[var].sum()} times)")
```

### Example 3: Time-Based Analysis

```python
from rms_reader import read_rms_file

df = read_rms_file('data.rms')

# Filter by time range
start_time = '2025-03-11 00:30:00'
end_time = '2025-03-11 00:45:00'
df_filtered = df[start_time:end_time]

# Calculate statistics for this period
print(df_filtered[['PT205', 'TT200A']].describe())
```

### Example 4: Temperature Monitoring

```python
from rms_reader import read_rms_file

df = read_rms_file('data.rms')

# Find all temperature sensors
temp_cols = [col for col in df.columns if col.startswith('TT') and not col.endswith(('_D1', '_D2'))]

# Check for temperature excursions
for col in temp_cols:
    temp_range = df[col].max() - df[col].min()
    if temp_range > 10:  # More than 10K variation
        print(f"{col}: Range = {temp_range:.2f} K")
```

### Example 5: Export Data

```python
from rms_reader import read_rms_file

df = read_rms_file('data.rms')

# Export to CSV
df.to_csv('exported_data.csv')

# Export to Excel with multiple sheets
with pd.ExcelWriter('exported_data.xlsx') as writer:
    # Separate analog and digital
    analog_cols = [col for col in df.columns if not col.endswith('_D') and not col.endswith('_D1') and not col.endswith('_D2')]
    digital_cols = [col for col in df.columns if col.endswith('_D') or col.endswith('_D1') or col.endswith('_D2')]
    
    df[analog_cols].to_excel(writer, sheet_name='Analog')
    df[digital_cols].to_excel(writer, sheet_name='Digital')
```

### Example 6: Batch Processing

```python
from rms_reader import read_rms_file
from pathlib import Path

# Process all RMS files in a directory
rms_files = Path('./data').glob('*.rms')

results = []
for filepath in rms_files:
    df = read_rms_file(filepath)
    
    # Extract summary statistics
    results.append({
        'file': filepath.name,
        'start': df.index[0],
        'end': df.index[-1],
        'duration': (df.index[-1] - df.index[0]).total_seconds(),
        'samples': len(df),
        'max_temp': df[[c for c in df.columns if c.startswith('TT')]].max().max()
    })

import pandas as pd
summary_df = pd.DataFrame(results)
print(summary_df)
```

## File Format Specification

### Header Example

```
# rms data file
# processed on LNCMI-controlPC [10.10.0.5] (asnet) offline at 03/11/2025-01:02:00
# header [encoding:US-ASCII - line-ending:unix]
# format = binary,asnet-vgen-1.0
# variables = ALIM1_J1 [type:float32|unit:A|min:-19800.000|max:19800.000|df:%.3f];...
# windows = [UTC] 03/11/2025-00:00:00.000 -> 03/11/2025-01:00:00.000
# frequency = 10.000 Hz
# data-helper [offset:0x144d - time:8(B),absolute - width:257(B)]
```

### Binary Data Format

Each sample consists of:
1. **Timestamp**: 8 bytes (double precision float, Unix timestamp)
2. **Variables**: In alphabetical order
   - Analog (float32): 4 bytes each
   - Digital (bit): 1 byte each

**Total sample width for FEPC-AUX-LNCMI**: 257 bytes

## Troubleshooting

### Issue: "Offset not found" error

**Solution**: Ensure the file has a valid header with the `# data-helper` line containing offset information.

### Issue: Wrong number of samples

**Solution**: Check that the file isn't truncated. The number of samples should be:
```
num_samples = (file_size - data_offset) / sample_width
```

### Issue: Incorrect data values

**Solution**: 
1. Verify variable ordering (alphabetical)
2. Check for unnamed variables that affect byte positions
3. Ensure correct endianness (<, little-endian)

### Issue: Timestamp conversion errors

**Solution**: Verify the timestamp format. LNCMI uses Unix timestamps (seconds since 1970-01-01).

## Performance Considerations

- **Memory usage**: A 1-hour file at 10 Hz (~36,000 samples) uses approximately:
  - 257 bytes/sample × 36,000 samples ≈ 9.3 MB (raw)
  - ~15-20 MB in pandas DataFrame (with overhead)

- **Reading speed**: Typically 100-200 MB/s on modern systems

- **Optimization tips**:
  - For large files, consider reading header first to check size
  - Use time-based filtering after reading to reduce memory
  - For analysis of specific variables, select columns after reading

## Contributing

Contributions are welcome! Areas for improvement:
- Support for FEPC-LNCMI format (454 bytes width)
- Streaming data reader for very large files
- More analysis utilities
- Plotting templates

## License

This code is provided for use with LNCMI data analysis. Please check with LNCMI for specific usage restrictions.

## Authors

Created for LNCMI-G (Laboratoire National des Champs Magnétiques Intenses - Grenoble)

## References

- LNCMI Documentation: Internal documentation on FEPC system
- Data format based on ASNET-VGEN-1.0 specification

## Support

For issues related to:
- **File format**: Contact LNCMI-G data acquisition team
- **This reader**: Open an issue on the repository
- **Data interpretation**: Contact experiment PI or LNCMI scientific staff
