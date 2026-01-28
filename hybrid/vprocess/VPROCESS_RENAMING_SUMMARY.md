# VProcess Scripts Renamed - Aligned with kHz/RMS Pattern

## Summary of Changes

The vprocess module has been reorganized to match the naming patterns used in your existing `hybrid/kHz` and `hybrid/rms` modules, and updated to support the actual vprocess filename format.

## Script Renaming

### Before → After

| Old Name | New Name | Pattern Match |
|----------|----------|---------------|
| `vprocess_reader.py` | `vprocess_reader.py` | ✓ Same as `rms_reader.py` |
| `vprocess_plot.py` | `plot_vprocess.py` | ✓ Same pattern as `plot_fepc_data.py` |
| `vprocess_examples.py` | `vprocess_examples.py` | ✓ Same as `rms_examples.py` |
| `vprocess_validate.py` | `validate.py` | ✓ Simpler (in vprocess/ folder) |
| `vprocess_batch.py` | `batch.py` | ✓ Simpler (in vprocess/ folder) |
| `vprocess_test.py` | `test.py` | ✓ Simpler (in vprocess/ folder) |
| `vprocess_cli.py` | `cli.py` | ✓ Simpler (in vprocess/ folder) |

### Rationale

1. **Core scripts keep data type prefix** (matches `rms_reader.py`, `rms_examples.py`)
2. **Plot script follows fepc pattern** (`plot_vprocess.py` like `plot_fepc_data.py`)
3. **Utilities use simple names** since they're already in `vprocess/` folder

## New Directory Structure

```
vprocess/
├── __init__.py                 # Module exports
├── vprocess_reader.py          # Core reader (like rms_reader.py)
├── plot_vprocess.py            # Plotting (like plot_fepc_data.py)
├── vprocess_examples.py        # Examples (like rms_examples.py)
├── validate.py                 # Validation utility
├── batch.py                    # Batch processing
├── test.py                     # Testing with mock data
├── cli.py                      # Unified CLI
├── README.md                   # Documentation
└── INTEGRATION_GUIDE.md        # HybridRun integration
```

This matches the kHz/RMS pattern:
```
kHz/                            rms/
├── fepc_reader.py             ├── rms_reader.py
├── plot_fepc_data.py          ├── rms_examples.py
└── README.md                  └── README.md
```

## Filename Format Support

### Added Features

Two new utility functions for handling the actual vprocess filename format:

#### 1. `parse_vprocess_filename()`

Parses the standardized filename format: `YYYYMMDD_HHMMSS__YYYYMMDD_HHMMSS.vprocess`

```python
from vprocess import parse_vprocess_filename

# Parse filename
filename = "20251105_000000__20251105_005959.vprocess"
start_time, end_time = parse_vprocess_filename(filename)

print(start_time)  # 2025-11-05 00:00:00
print(end_time)    # 2025-11-05 00:59:59
```

**Format Details**:
- `YYYYMMDD`: Year, month, day (8 digits)
- `HHMMSS`: Hour, minute, second (6 digits)
- `__`: Double underscore separator
- `.vprocess`: File extension

**Examples**:
- `20251105_000000__20251105_005959.vprocess` - 00:00:00 to 00:59:59
- `20251105_060000__20251105_065959.vprocess` - 06:00:00 to 06:59:59
- `20251105_230000__20251105_235959.vprocess` - 23:00:00 to 23:59:59

#### 2. `find_vprocess_files_for_date()`

Finds all vprocess files for a specific date:

```python
from datetime import datetime
from vprocess import find_vprocess_files_for_date

# Find all files for November 5, 2025
date = datetime(2025, 11, 5)
files = find_vprocess_files_for_date('./data', date)

print(f"Found {len(files)} files")  # Expected: 24 (one per hour)
for file in files:
    print(f"  - {file.name}")
```

This automatically:
- Parses filenames
- Filters by date
- Returns sorted list
- Handles edge cases

## Usage Updates

### Command-Line Tools

All tools now use the new script names:

**Validation**:
```bash
python validate.py data.vprocess --check-data
```

**Plotting**:
```bash
python plot_vprocess.py data.vprocess --vars TT115A TT508A
python plot_vprocess.py data.vprocess --overview
python plot_vprocess.py data.vprocess --compare TT115A TT508A
```

**Batch Processing**:
```bash
python batch.py --dir ./data --merge --output merged.csv
python batch.py --dir ./data --analyze
```

**Testing**:
```bash
python test.py
python test.py --create-mock --samples 1000
```

**Unified CLI**:
```bash
python cli.py info data.vprocess
python cli.py validate data.vprocess --check-data
python cli.py plot data.vprocess --vars TT115A TT508A
python cli.py batch --dir ./data --merge
```

### Python API

Import statements remain clean:

```python
# Core functionality
from vprocess import (
    VProcessFileReader,
    read_vprocess_file,
    parse_vprocess_filename,
    find_vprocess_files_for_date
)

# Quick read
df = read_vprocess_file('20251105_000000__20251105_005959.vprocess')

# Parse filename
start, end = parse_vprocess_filename('20251105_060000__20251105_065959.vprocess')

# Find daily files
from datetime import datetime
files = find_vprocess_files_for_date('./data', datetime(2025, 11, 5))
```

### Utilities Can Be Imported Directly

```python
# From within vprocess directory
from validate import validate_vprocess_file
from batch import process_batch, export_data
from plot_vprocess import plot_variables, plot_comparison
```

## Integration with HybridRun

The renaming makes integration cleaner:

```python
# hybrid/vprocess/ structure matches hybrid/rms/ structure
hybrid/
├── kHz/
│   ├── fepc_reader.py
│   └── plot_fepc_data.py
├── rms/
│   ├── rms_reader.py
│   └── rms_examples.py
└── vprocess/                    ← NEW
    ├── vprocess_reader.py      ← Core (matches rms_reader.py)
    ├── plot_vprocess.py        ← Plotting (matches plot_fepc_data.py)
    ├── vprocess_examples.py    ← Examples (matches rms_examples.py)
    └── ...utilities...
```

## Backward Compatibility

If you have existing code using old names, here's the migration:

```python
# Old imports (still work from same directory)
from vprocess_reader import VProcessFileReader
from vprocess_examples import example_quick_read

# New imports (same, no change needed for core)
from vprocess_reader import VProcessFileReader
from vprocess_examples import example_quick_read

# Old utility scripts
# python vprocess_validate.py file.vprocess
# python vprocess_plot.py file.vprocess --vars A B
# python vprocess_batch.py --dir ./data

# New utility scripts (cleaner names)
python validate.py file.vprocess
python plot_vprocess.py file.vprocess --vars A B
python batch.py --dir ./data
```

## Benefits of Renaming

### 1. Consistency with Hybrid Module
- Matches kHz/RMS naming patterns
- Easier to navigate codebase
- Clear organizational structure

### 2. Cleaner Names
- Less redundant (no `vprocess_` prefix for utilities in `vprocess/` folder)
- Shorter commands
- More intuitive

### 3. Professional Structure
- Follows established conventions
- Matches scientific software patterns
- Ready for integration

### 4. Filename Support
- Handles actual LNCMI file naming
- Automatic date-based file discovery
- Proper timestamp parsing

## Testing the Changes

All functionality remains the same, just with cleaner names:

```bash
# 1. Test basic reading
python -c "from vprocess import read_vprocess_file; print('✓ Import works')"

# 2. Test filename parsing
python -c "from vprocess import parse_vprocess_filename; \
    print(parse_vprocess_filename('20251105_000000__20251105_005959.vprocess'))"

# 3. Test utilities
python test.py --create-mock
python validate.py mock_data.vprocess --check-data
python plot_vprocess.py mock_data.vprocess --overview --no-show

# 4. Test CLI
python cli.py info mock_data.vprocess
```

## Updated Documentation

All documentation has been updated:
- ✅ README.md - New script names and filename format
- ✅ __init__.py - Exports new utility functions
- ✅ All script docstrings - Updated usage examples
- ✅ INTEGRATION_GUIDE.md - Ready for HybridRun

## File Organization for Daily Data

With the new filename support, organizing daily data is straightforward:

```python
from datetime import datetime
from vprocess import find_vprocess_files_for_date, read_vprocess_file
import pandas as pd

# Load all data for November 5, 2025
date = datetime(2025, 11, 5)
files = find_vprocess_files_for_date('./vprocess/2025-11-05', date)

# Read and concatenate
dfs = [read_vprocess_file(str(f)) for f in files]
daily_data = pd.concat(dfs, axis=0).sort_index()

print(f"Loaded {len(daily_data)} samples for {date.date()}")
print(f"Time range: {daily_data.index[0]} to {daily_data.index[-1]}")
```

## Migration Checklist

If updating existing code:

- [ ] Update script calls: `vprocess_*.py` → new names
- [ ] Add filename parsing for date-based file discovery
- [ ] Update any shell scripts or automation
- [ ] Update documentation/notebooks
- [ ] Test with actual LNCMI data files

## Next Steps

1. **Test with Real Data**: Use actual vprocess files with format `YYYYMMDD_HHMMSS__YYYYMMDD_HHMMSS.vprocess`
2. **Integrate into HybridRun**: Follow INTEGRATION_GUIDE.md
3. **Add to Hybrid CLI**: Optionally integrate utilities into main `hybrid/cli.py`
4. **Daily Processing**: Use `find_vprocess_files_for_date()` for automated workflows

## Summary

✅ **Scripts renamed** to match kHz/RMS patterns  
✅ **Filename parsing** for actual LNCMI format  
✅ **Date-based discovery** for daily file processing  
✅ **All imports updated** and working  
✅ **Documentation updated** with new names  
✅ **Cleaner structure** for integration  
✅ **Professional organization** matching scientific software standards  

The vprocess module is now fully aligned with your hybrid module architecture and ready for integration!
