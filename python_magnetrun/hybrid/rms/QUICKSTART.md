# RMS Reader - Quick Start Guide

## Installation

1. **Install dependencies:**
```bash
pip install numpy pandas

# Optional for additional features:
pip install matplotlib openpyxl tables
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Read an RMS file (simplest method)

```python
from rms_reader import read_rms_file

# Read entire file into DataFrame
df = read_rms_file('your_file.rms')

print(df.head())
print(df.info())
```

### 2. Get file information without reading data

```python
from rms_reader import RMSFileReader

reader = RMSFileReader('your_file.rms')
reader.parse_header()

# Print summary
reader.print_summary()

# Get variable details
var_info = reader.get_variable_info()
print(var_info)

# Get metadata
metadata = reader.get_metadata()
print(metadata)
```

### 3. Validate a file

```bash
python validate_rms.py validate your_file.rms
```

### 4. Quick plotting

```bash
# Overview plot with key variables
python plot_rms.py overview your_file.rms

# Plot specific variables
python plot_rms.py vars your_file.rms PT205 TT200A PH_V11 -o plot.png

# Plot all temperatures
python plot_rms.py temps your_file.rms temps.png

# Show digital signals timeline
python plot_rms.py digital your_file.rms alarms.png
```

## Common Tasks

### Extract specific variables

```python
from rms_reader import read_rms_file

df = read_rms_file('your_file.rms')

# Select specific columns
pressure = df['PT205']
temp = df['TT200A']

# Multiple variables
subset = df[['PT205', 'TT200A', 'PH_V11']]
```

### Filter by time

```python
df = read_rms_file('your_file.rms')

# Filter by time range
filtered = df['2025-03-11 00:30:00':'2025-03-11 00:45:00']

# Or using boolean indexing
start_time = pd.Timestamp('2025-03-11 00:30:00', tz='UTC')
end_time = pd.Timestamp('2025-03-11 00:45:00', tz='UTC')
filtered = df[(df.index >= start_time) & (df.index <= end_time)]
```

### Find alarm triggers

```python
from rms_reader import RMSFileReader

reader = RMSFileReader('your_file.rms')
df = reader.read()

# Get all alarm variables (ending with _D1 or _D2)
alarm_vars = [col for col in df.columns if col.endswith(('_D1', '_D2'))]

# Check which alarms were triggered
for var in alarm_vars:
    if df[var].sum() > 0:
        print(f"{var}: triggered {df[var].sum()} times")
        # Find when it was triggered
        trigger_times = df.index[df[var] == 1]
        print(f"  First trigger: {trigger_times[0]}")
        print(f"  Last trigger: {trigger_times[-1]}")
```

### Export data

```python
df = read_rms_file('your_file.rms')

# CSV
df.to_csv('exported_data.csv')

# Excel
df.to_excel('exported_data.xlsx')

# HDF5
df.to_hdf('exported_data.h5', key='rms_data')
```

### Calculate statistics

```python
df = read_rms_file('your_file.rms')

# Overall statistics
print(df.describe())

# Statistics for specific variables
print(df[['PT205', 'TT200A']].describe())

# Custom statistics
mean_pressure = df['PT205'].mean()
max_temp = df['TT200A'].max()
temp_range = df['TT200A'].max() - df['TT200A'].min()
```

## File Structure Reference

### FEPC-AUX-LNCMI Format
- **Configuration**: 2 MAD cards + 3 MIVA cards
- **Sample width**: 257 bytes (8 byte timestamp + 249 byte data)
- **Sampling frequency**: Typically 10 Hz
- **Variables**: ~48 analog + ~57 digital (7 unnamed excluded)

### Variable Naming
- `TT###`: Temperature sensors
- `PT###`: Pressure sensors  
- `PH_V##`: Phase voltages
- `ALIM*`: Power supply currents
- `*_D`, `*_D1`, `*_D2`: Digital signals and alarms
- `MAGNET_*`: Magnet-related signals
- `BITTER_*`: Bitter magnet signals

## Troubleshooting

### Problem: ImportError for numpy or pandas
**Solution**: Install required packages
```bash
pip install numpy pandas
```

### Problem: "No module named 'rms_reader'"
**Solution**: Make sure rms_reader.py is in the same directory or in your PYTHONPATH

### Problem: Wrong data values
**Solution**: Verify file format is FEPC-AUX-LNCMI. The reader expects:
- Sample width: 257 bytes
- Variables in alphabetical order
- Little-endian byte order

### Problem: File validation fails
**Solution**: Run the validation script for detailed diagnostics
```bash
python validate_rms.py inspect your_file.rms
```

## Support Files

- **rms_reader.py**: Main reader library
- **rms_examples.py**: 10 detailed usage examples
- **validate_rms.py**: File validation and inspection tool
- **plot_rms.py**: Quick plotting utilities
- **README.md**: Full documentation
- **requirements.txt**: Python dependencies

## Next Steps

1. Try the validation script on your RMS file
2. Run the overview plot to see your data
3. Check the examples in rms_examples.py
4. Read the full README.md for advanced features

## Quick Reference Card

```python
# Import
from rms_reader import read_rms_file, RMSFileReader

# Read
df = read_rms_file('file.rms')                    # Quick read
reader = RMSFileReader('file.rms')                # Detailed access
reader.parse_header()                             # Parse header only
df = reader.read()                                # Read all data

# Inspect
reader.print_summary()                            # Print summary
var_info = reader.get_variable_info()             # Variable details
metadata = reader.get_metadata()                  # File metadata

# Filter
df['variable_name']                               # One variable
df[['var1', 'var2']]                              # Multiple variables
df['2025-03-11 00:30:00':'2025-03-11 01:00:00']  # Time range

# Analyze
df.describe()                                     # Statistics
df['var'].mean(), df['var'].std()                 # Mean, std dev
df['var'].min(), df['var'].max()                  # Min, max

# Export
df.to_csv('data.csv')                             # CSV
df.to_excel('data.xlsx')                          # Excel
df.to_hdf('data.h5', key='rms')                   # HDF5
```

## Questions?

Check the full README.md for comprehensive documentation and examples.
