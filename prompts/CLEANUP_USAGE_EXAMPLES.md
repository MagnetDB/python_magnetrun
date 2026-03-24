# Data Cleanup and Preparation - Usage Examples

This document demonstrates how to use the new flexible `prepareData()` and `cleanupData()` methods.

## Overview

Both methods have been refactored to accept optional configuration dictionaries for more flexibility while maintaining backward compatibility through legacy versions.

**Key Enhancement:** The methods now intelligently detect your intent from the configuration parameters:
- If you specify Icoil columns in `keys_to_rename`, it will use those explicitly (skipping complex auto-detection)
- If you specify UH/UB in `keys_to_add`, it will skip Ucoil auto-detection
- Otherwise, falls back to the original auto-detection behavior for backward compatibility

## Backward Compatibility

Legacy versions are available for existing code:
- `prepareData_legacy()` in `MagnetRun.py`
- `cleanupData_legacy()` in `magnetdata.py`

## New Flexible Methods

### 1. MagnetData.cleanupData()

Located in `magnetdata.py`, this method now accepts optional parameters:

```python
data.cleanupData(
    keys_to_remove=None,      # list of column names to remove
    keys_to_rename=None,      # dict mapping {old_name: new_name}
    keys_to_add=None,         # dict mapping {new_key: formula}
    debug=False
)
```

#### Smart Detection Features:

1. **Icoil column selection via `keys_to_rename`**: If you include any `Icoil\d+` keys in `keys_to_rename`, only those columns will be kept, skipping the complex auto-detection logic.

2. **UH/UB calculation via `keys_to_add`**: If you define "UH" or "UB" in `keys_to_add`, the Ucoil auto-detection is skipped entirely.

#### Example 1: Basic usage (backward compatible)
```python
from python_magnetrun.magnetdata import MagnetData

data = MagnetData.fromcsv("myfile.csv")
data.cleanupData(debug=True)  # Uses default auto-detection behavior
```

#### Example 2: Explicitly specify which Icoil columns to keep (RECOMMENDED)
```python
data = MagnetData.fromcsv("myfile.csv")
data.cleanupData(
    keys_to_rename={
        "Icoil1": "IB",    # Only Icoil1 and Icoil20 will be kept
        "Icoil20": "IH"
    },
    debug=True
)
# This skips all the complex Icoil auto-detection logic!
```

#### Example 3: Manually compute UH/UB and keep specific Icoils  
```python
data = MagnetData.fromcsv("myfile.csv")
data.cleanupData(
    keys_to_add={
        "UH": "UH = Ucoil1 + Ucoil2 + Ucoil3 + Ucoil4 + Ucoil5 + Ucoil6 + Ucoil7 + Ucoil8 + Ucoil9 + Ucoil10",
        "UB": "UB = Ucoil11 + Ucoil12 + Ucoil13 + Ucoil14 + Ucoil15 + Ucoil16 + Ucoil17 + Ucoil18 + Ucoil19 + Ucoil20"
    },
    keys_to_rename={
        "Icoil1": "IB",
        "Icoil20": "IH"
    },
    keys_to_remove=[
        "Ucoil1", "Ucoil2", "Ucoil3", "Ucoil4", "Ucoil5",
        "Ucoil6", "Ucoil7", "Ucoil8", "Ucoil9", "Ucoil10",
        "Ucoil11", "Ucoil12", "Ucoil13", "Ucoil14", "Ucoil15",
        "Ucoil16", "Ucoil17", "Ucoil18", "Ucoil19", "Ucoil20"
    ],
    debug=True
)
# This completely bypasses BOTH Icoil and Ucoil auto-detection!
```

#### Example 4: Add computed columns
```python
data = MagnetData.fromcsv("myfile.csv")
data.cleanupData(
    keys_to_add={
        "IH_ref": "IH_ref = Idcct1 + Idcct2",
        "IB_ref": "IB_ref = Idcct3 + Idcct4",
        "Ptotal": "Ptotal = Pmagnet + Q"
    },
    debug=True
)
```

#### Example 5: Rename columns
```python
data = MagnetData.fromcsv("myfile.csv")
data.cleanupData(
    keys_to_rename={
        "Flow1": "FlowH",
        "Flow2": "FlowB",
        "Rpm1": "RpmH",
        "Rpm2": "RpmB"
    },
    debug=True
)
```

#### Example 6: Full explicit control (RECOMMENDED for clarity)
```python
data = MagnetData.fromcsv("myfile.csv")
data.cleanupData(
    keys_to_add={
        "UH": "UH = Ucoil1 + Ucoil2 + Ucoil3 + Ucoil4 + Ucoil5",
        "UB": "UB = Ucoil6 + Ucoil7 + Ucoil8 + Ucoil9 + Ucoil10",
        "IH_ref": "IH_ref = Idcct1 + Idcct2",
        "IB_ref": "IB_ref = Idcct3 + Idcct4"
    },
    keys_to_rename={
        "Icoil5": "IH",
        "Icoil10": "IB",
        "Flow1": "FlowH",
        "Flow2": "FlowB"
    },
    keys_to_remove=["Idcct1", "Idcct2", "Idcct3", "Idcct4"] + 
                   [f"Ucoil{i}" for i in range(1, 11)],
    debug=True
)
# Complete explicit control - no auto-detection needed!
```

### 2. prepareData() function

Located in `MagnetRun.py`, this function now accepts optional parameters:

```python
from python_magnetrun.MagnetRun import prepareData

prepareData(
    data,                    # MagnetData object
    housing,                 # Housing name (e.g., "M9", "M8", "M10")
    keys_to_remove=None,     # list of column names to remove
    keys_to_rename=None,     # dict mapping {old_name: new_name}
    keys_to_add=None,        # dict mapping {new_key: formula}
    debug=False
)
```

**Note:** The `prepareData` function now internally calls `cleanupData()`, which means the smart detection features work here too! If you specify Icoil columns in `keys_to_rename` or UH/UB in `keys_to_add`, the complex auto-detection is bypassed.

#### Example 1: Basic usage (backward compatible)
```python
from python_magnetrun.MagnetRun import prepareData
from python_magnetrun.magnetdata import MagnetData

data = MagnetData.fromcsv("M9_data.csv")
prepareData(data, "M9", debug=True)  # Uses default behavior for M9
```

#### Example 2: Override Icoil detection for M9 (RECOMMENDED)
```python
data = MagnetData.fromcsv("M9_data.csv")
prepareData(
    data,
    "M9",
    keys_to_rename={
        "Icoil1": "IH",    # Explicitly specify which Icoils to keep
        "Icoil20": "IB"
    },
    debug=True
)
# The standard M9 operations still run, but Icoil detection is explicit!
```

#### Example 3: Full manual control for M9
```python
data = MagnetData.fromcsv("M9_data.csv")
prepareData(
    data,
    "M9",
    keys_to_add={
        "UH": "UH = Ucoil1 + Ucoil2 + Ucoil3 + Ucoil4 + Ucoil5 + Ucoil6 + Ucoil7 + Ucoil8 + Ucoil9 + Ucoil10",
        "UB": "UB = Ucoil11 + Ucoil12 + Ucoil13 + Ucoil14 + Ucoil15 + Ucoil16 + Ucoil17 + Ucoil18 + Ucoil19 + Ucoil20",
        "efficiency": "efficiency = Pmagnet / (Pmagnet + Q) * 100"
    },
    keys_to_rename={
        "Icoil1": "IH",
        "Icoil20": "IB"
    },
    keys_to_remove=["Idcct1", "Idcct2", "Idcct3", "Idcct4"] +
                   [f"Ucoil{i}" for i in range(1, 21)],
    debug=True
)
# Complete explicit control - bypasses all auto-detection!
```

#### Example 4: Add custom computed fields only
```python
data = MagnetData.fromcsv("M9_data.csv")
prepareData(
    data,
    "M9",
    keys_to_add={
        "Ptotal": "Ptotal = Pmagnet + Q",
        "efficiency": "efficiency = Pmagnet / Ptotal * 100"
    },
    debug=True
)
```

#### Example 5: Remove intermediate values after standard operations
```python
data = MagnetData.fromcsv("M9_data.csv")
prepareData(
    data,
    "M9",
    keys_to_remove=["Idcct1", "Idcct2", "Idcct3", "Idcct4", "IH_ref", "IB_ref"],
    debug=True
)
```

#### Example 6: Custom renaming after standard operations
```python
data = MagnetData.fromcsv("M9_data.csv")
prepareData(
    data,
    "M9",
    keys_to_rename={
        "TinH": "T_in_helix",
        "TinB": "T_in_bitter",
        "Icoil2": "IH",      # This also controls which Icoils to keep!
        "Icoil18": "IB"
    },
    debug=True
)
```

## Operation Order

### cleanupData() execution order:
1. Standard cleanup operations (remove empty columns, handle duplicates, etc.)
2. **keys_to_add** - Add computed columns
3. **keys_to_rename** - Rename columns
4. **keys_to_remove** - Remove columns

### prepareData() execution order:
1. Add timestamp
2. **keys_to_add** - Add custom computed columns (before standard operations)
3. Standard housing-specific operations (IH_ref, IB_ref, Flow/Rpm/Tin/HP renaming)
4. Call `cleanupData()`
5. Rename Icoil columns to IH/IB
6. **keys_to_rename** - Apply custom renames (after standard operations)
7. **keys_to_remove** - Remove custom columns (at the end)

## Formula Syntax

Formulas use pandas DataFrame.eval() syntax:
- Simple arithmetic: `"result = col1 + col2"`
- Multiple operations: `"result = (col1 + col2) * col3 / 100"`
- Functions: `"result = abs(col1 - col2)"`

See [pandas.DataFrame.eval documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.eval.html) for more details.

## Key Benefits of Explicit Configuration (NEW!)

### ✨ Smart Detection

The refactored methods now intelligently detect your intent:

1. **Specify Icoil columns via `keys_to_rename`**:
   ```python
   keys_to_rename={"Icoil1": "IB", "Icoil20": "IH"}
   ```
   → Automatically keeps only Icoil1 and Icoil20, **skipping 100+ lines of complex auto-detection logic!**

2. **Specify UH/UB via `keys_to_add`**:
   ```python
   keys_to_add={"UH": "UH = Ucoil1 + ... + Ucoil10", "UB": "UB = Ucoil11 + ... + Ucoil20"}
   ```
   → Automatically **skips Ucoil groupby auto-detection logic!**

3. **No explicit config?**
   → Falls back to original auto-detection for **full backward compatibility**

### 🎯 Best Practices

#### ✅ RECOMMENDED: Use explicit configuration
```python
# Clear, explicit, maintainable
data.cleanupData(
    keys_to_add={
        "UH": "UH = Ucoil1 + Ucoil2 + Ucoil3 + Ucoil4 + Ucoil5",
        "UB": "UB = Ucoil6 + Ucoil7 + Ucoil8 + Ucoil9 + Ucoil10"
    },
    keys_to_rename={
        "Icoil5": "IH",
        "Icoil10": "IB"
    },
    keys_to_remove=[f"Ucoil{i}" for i in range(1, 11)],
    debug=True
)
```

**Benefits:**
- ✅ **Much faster** - skips complex auto-detection
- ✅ **More predictable** - explicit is better than implicit
- ✅ **Easier to debug** - you know exactly what columns to expect
- ✅ **Self-documenting** - clear what the code does

#### ⚠️ LEGACY: Rely on auto-detection
```python
# Works, but uses complex heuristics
data.cleanupData(debug=True)
```

**Use when:**
- Working with legacy code
- Dataset structure is unknown/varies
- Quick prototyping

### 🚀 Performance Comparison

| Approach | Auto-Detection Code | Performance | Clarity |
|----------|-------------------|-------------|---------|
| **Explicit Config** | Skipped entirely | ⚡ Fast | ✨ Crystal clear |
| **Auto-Detection** | ~200 lines executed | 🐌 Slower | 🤔 Heuristic-based |

## Notes

- All operations are performed **in-place** on the MagnetData object
- Keys must exist in the DataFrame before being referenced in formulas
- The order of operations matters - plan your operations accordingly
- Use `debug=True` to see detailed logging of operations
- **For new code, prefer explicit configuration over auto-detection** for better performance and clarity
