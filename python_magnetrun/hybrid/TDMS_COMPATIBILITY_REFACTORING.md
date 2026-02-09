# HybridData TDMS Compatibility Refactoring

## Overview

Refactor `HybridData` and `HybridRun` classes to fully match the MagnetData/MagnetRun TDMS-compatible interface. This enables seamless comparison between hybrid magnet data (kHz/RMS) and traditional pupitre/TDMS data.

**Key Goals:**
1. Organize data using TDMS-like `group/channel` structure
2. Support multiple FEPC systems in a single HybridData instance
3. Implement lazy loading to handle large kHz files efficiently
4. Maintain backward compatibility with existing code

**Implementation Phases:**
- **Phase 1 (Current):** ANALOG data only (kHz and RMS)
  - 16 channels per card (ANA type)
  - Float/uint16 data types
  - Primary measurement data (currents, voltages, etc.)
- **Phase 2 (Future):** Add DIGITAL data support
  - 32 channels per card (DIG type)
  - Boolean data types
  - Status/control signals

---

## Current State vs. Target State

### Current Implementation

```python
# Groups are top-level (type/system only)
Groups = {
    "kHz/FEPC-LNCMI": {"type": "kHz", "system": "FEPC-LNCMI", "files": [...]},
    "rms/FEPC-LNCMI": {"type": "rms", "system": "FEPC-LNCMI", "files": [...]},
}

Keys = [
    "kHz/FEPC-LNCMI",      # Group only
    "rms/FEPC-LNCMI",      # Group only
]

# Access pattern
data, time = hrun.getData("kHz/FEPC-LNCMI/I_H1")  # Works but inconsistent with Groups
```

### Target Implementation (TDMS-Compatible)

```python
# Groups contain channels with metadata (like TDMS)
Groups = {
    "kHz/FEPC-LNCMI": {
        "I_H1": {
            "slot": 1,
            "card_type": "ANA",
            "wf_increment": 0.001,      # 1 kHz sampling
            "wf_samples": 86400000,     # Samples in dataset
            "wf_start_time": datetime(...),
            "wf_start_offset": 0.0,
            "files": [Path("00HOST_...LIST_1.bin"), ...],
        },
        "I_H2": {...},
        # ... more channels
    },
    "kHz/FEPC-AUX-LNCMI": {
        "I_AUX": {...},
    },
    "rms/FEPC-LNCMI": {
        "I_H1": {
            "type": "float32",
            "unit": "A",
            "wf_increment": 0.1,        # 10 Hz sampling
            "wf_samples": 864000,
            "wf_start_time": datetime(...),
            "files": [Path("FEPC-LNCMI_2025-01-06_0000—2025-01-06_0100.rms"), ...],
        },
        # ... more channels
    },
}

Keys = [
    "kHz/FEPC-LNCMI/I_H1",
    "kHz/FEPC-LNCMI/I_H2",
    "kHz/FEPC-AUX-LNCMI/I_AUX",
    "rms/FEPC-LNCMI/I_H1",
    # ... full group/channel paths
]

# Lazy-loaded Data dictionary
Data = {
    "kHz/FEPC-LNCMI": LazyDataFrame(...),   # Loads channels on access
    "rms/FEPC-LNCMI": LazyDataFrame(...),
}

# Access patterns (matching MagnetData)
df = hdata.getData("kHz/FEPC-LNCMI/I_H1")           # Single channel
df = hdata.getData("kHz/FEPC-LNCMI")                # All channels in group
df = hdata.getTdmsData("kHz/FEPC-LNCMI", "I_H1")    # Direct TDMS-style access
```

---

## Implementation Steps

### Step 1: Create Lazy Loading Infrastructure

**File:** `hybrid/hybrid_data.py`

**Tasks:**
1. Create `LazyDataFrame` class that mimics pandas DataFrame but loads data on column access
2. Create `LazyGroupData` class that manages lazy loading for a specific group
3. Implement `__getitem__` to load channels on demand
4. Add caching to avoid reloading same channel

**Expected behavior:**
```python
# Data dict contains lazy loaders
Data["kHz/FEPC-LNCMI"]["I_H1"]  # Triggers load of I_H1 channel
Data["kHz/FEPC-LNCMI"][["I_H1", "I_H2"]]  # Loads multiple channels

# Support pandas-like operations
Data["kHz/FEPC-LNCMI"].columns  # List available channels (without loading)
```

**Key considerations:**
- kHz files can be 2-3 GB per day per slot (ANALOG only in Phase 1)
- Only load data when explicitly accessed
- Cache loaded channels to avoid repeated disk I/O
- Support both single channel and multi-channel access

**Phase 1 scope:**
- ANALOG cards only (card_type == "ANA")
- Skip DIGITAL cards during group building
- 16 channels per ANALOG card
- uint16 raw data, float32 after calibration

---

### Step 2: Refactor `_build_groups()` Method

**File:** `hybrid/hybrid_data.py`

**Tasks:**
1. Modify `_build_groups()` to populate channel-level metadata
2. For kHz data: parse FEPCConfig to get all variables and their metadata
3. For RMS data: read first RMS file header to get variable list
4. Build complete `Groups` dict with channel metadata
5. Populate `Keys` list with full `group/channel` paths

**Implementation notes:**
```python
def _build_groups(self) -> None:
    """Build Groups and Keys from discovered data (TDMS-compatible)"""
    self.Groups = {}
    self.Keys = []
    
    for system in self._info.fepc_systems:
        # kHz groups
        if system in self._info.khz_files:
            group_name = f"kHz/{system}"
            self._build_khz_group(group_name, system)
        
        # RMS groups
        if system in self._info.rms_files:
            group_name = f"rms/{system}"
            self._build_rms_group(group_name, system)
        
        # Trigger groups (metadata only, no channels)
        if system in self._info.trigger_files:
            group_name = f"trigger/{system}"
            self._build_trigger_group(group_name, system)

def _build_khz_group(self, group_name: str, system: str) -> None:
    """Build kHz group with channel metadata (ANALOG only for now)"""
    self.Groups[group_name] = {}
    
    # Load config to get channel information
    config = self.load_khz_config(system)
    if config is None:
        logger.warning(f"No config found for {system}, skipping kHz group")
        return
    
    # Get file list for this system
    bin_files = self._info.khz_files[system]
    
    # Iterate through all cards and variables
    for card in config.cards:
        # Phase 1: ANALOG data only
        if card.card_type != "ANA":
            logger.debug(f"Skipping DIGITAL card in slot {card.slot} (Phase 2)")
            continue
        
        for i, var_name in enumerate(card.variable_names):
            channel_key = f"{group_name}/{var_name}"
            self.Keys.append(channel_key)
            
            # Build metadata (similar to TDMS properties)
            self.Groups[group_name][var_name] = {
                "slot": card.slot,
                "card_type": card.card_type,
                "channel_index": i,
                "wf_increment": 1.0 / 1000.0,  # 1 kHz
                "wf_samples": self._estimate_khz_samples(bin_files, card.card_type),
                "wf_start_time": self._get_khz_start_time(bin_files[0]),
                "wf_start_offset": 0.0,
                "files": [f for f in bin_files if f"LIST_{card.slot}.bin" in f.name],
            }

def _build_rms_group(self, group_name: str, system: str) -> None:
    """Build RMS group with channel metadata"""
    self.Groups[group_name] = {}
    
    # Get RMS file list
    rms_files = self._info.rms_files[system]
    if not rms_files:
        return
    
    # Read variable info from first file
    var_info = self.get_rms_variable_info(system, file_idx=0)
    
    for _, row in var_info.iterrows():
        var_name = row["name"]
        channel_key = f"{group_name}/{var_name}"
        self.Keys.append(channel_key)
        
        self.Groups[group_name][var_name] = {
            "type": row["type"],
            "unit": row.get("unit", ""),
            "wf_increment": 1.0 / 10.0,  # Typical RMS: 10 Hz
            "wf_samples": self._estimate_rms_samples(rms_files),
            "wf_start_time": self._get_rms_start_time(rms_files[0]),
            "wf_start_offset": 0.0,
            "files": rms_files,
        }
```

**Helper methods to add:**
- `_estimate_khz_samples(bin_files, card_type)` - Calculate total samples from file sizes
- `_get_khz_start_time(bin_file)` - Extract start time from filename
- `_estimate_rms_samples(rms_files)` - Calculate total samples from RMS files
- `_get_rms_start_time(rms_file)` - Extract start time from RMS filename

---

### Step 3: Implement `getTdmsData()` Method

**File:** `hybrid/hybrid_data.py`

**Tasks:**
1. Add `getTdmsData(group, channel)` method matching MagnetData signature
2. Support single channel: `getTdmsData("kHz/FEPC-LNCMI", "I_H1")`
3. Support multi-channel: `getTdmsData("kHz/FEPC-LNCMI", ["I_H1", "I_H2"])`
4. Support all channels: `getTdmsData("kHz/FEPC-LNCMI", None)`
5. Return pandas DataFrame with appropriate index

**Implementation:**
```python
def getTdmsData(self, group: str, channel: str | List[str] | None) -> pd.DataFrame:
    """
    Get data for TDMS-style group/channel access
    
    Parameters
    ----------
    group : str
        Group name (e.g., "kHz/FEPC-LNCMI")
    channel : str, list[str], or None
        Channel name(s) or None for all channels
    
    Returns
    -------
    pd.DataFrame
        Data with time index
    """
    if group not in self.Groups:
        raise ValueError(f"Group '{group}' not found. Available: {list(self.Groups.keys())}")
    
    # Parse group to get type and system
    parts = group.split("/")
    data_type, system = parts[0], parts[1]
    
    # Get or create lazy data loader for this group
    if group not in self.Data:
        self.Data[group] = LazyGroupData(self, group, data_type, system)
    
    # Access channels (triggers loading)
    if channel is None:
        # Return all channels as DataFrame
        return self.Data[group].as_dataframe()
    elif isinstance(channel, str):
        # Single channel
        return pd.DataFrame({channel: self.Data[group][channel]})
    else:
        # Multiple channels
        return pd.DataFrame({ch: self.Data[group][ch] for ch in channel})
```

---

### Step 4: Update `getData()` Method

**File:** `hybrid/hybrid_data.py`

**Tasks:**
1. Modify `getData(key)` to handle both formats:
   - `"kHz/FEPC-LNCMI/I_H1"` - full path (current)
   - `"kHz/FEPC-LNCMI"` - group only (new)
2. Delegate to `getTdmsData()` for consistency
3. Support list of keys: `getData(["kHz/FEPC-LNCMI/I_H1", "rms/FEPC-LNCMI/I_H1"])`

**Implementation:**
```python
def getData(self, key: str | List[str] | None = None) -> pd.DataFrame:
    """
    Get data for a specific key (MagnetData-compatible)
    
    Parameters
    ----------
    key : str, list[str], or None
        - "group/channel": single channel
        - "group": all channels from group
        - ["group1/channel1", "group2/channel2"]: multiple channels
        - None: return Data dict
    
    Returns
    -------
    pd.DataFrame or dict
    """
    if key is None:
        return self.Data
    
    if isinstance(key, str):
        parts = key.split("/")
        
        if len(parts) == 3:
            # Full path: "kHz/FEPC-LNCMI/I_H1"
            group = f"{parts[0]}/{parts[1]}"
            channel = parts[2]
            return self.getTdmsData(group, channel)
        
        elif len(parts) == 2:
            # Group only: "kHz/FEPC-LNCMI"
            group = key
            return self.getTdmsData(group, None)
        
        else:
            raise ValueError(f"Invalid key format: {key}")
    
    elif isinstance(key, list):
        # Multiple keys
        dfs = []
        for k in key:
            dfs.append(self.getData(k))
        return pd.concat(dfs, axis=1)
    
    else:
        raise TypeError(f"Invalid key type: {type(key)}")
```

---

### Step 5: Add Units Support

**File:** `hybrid/hybrid_data.py`

**Tasks:**
1. Implement `Units()` method to populate `self.units` dict
2. Implement `getUnitKey(key)` method to return unit for a specific channel
3. Use pint for unit handling (consistent with MagnetData)

**Implementation:**
```python
def Units(self, debug: bool = False):
    """
    Populate units dictionary for all channels
    Similar to MagnetData.Units()
    """
    from pint import UnitRegistry
    ureg = UnitRegistry()
    
    for key in self.Keys:
        parts = key.split("/")
        data_type = parts[0]
        group = f"{parts[0]}/{parts[1]}"
        channel = parts[2]
        
        # Get unit from group metadata
        if channel in self.Groups[group]:
            metadata = self.Groups[group][channel]
            unit_str = metadata.get("unit", "")
            
            # Map common units
            if data_type == "kHz":
                # kHz variables are typically currents or voltages
                if "I_" in channel:
                    self.units[key] = ("Current", ureg.ampere)
                elif "U_" in channel or "V_" in channel:
                    self.units[key] = ("Voltage", ureg.volt)
                else:
                    self.units[key] = ("Unknown", ureg.dimensionless)
            
            elif data_type == "rms":
                # RMS files contain unit information
                if unit_str:
                    try:
                        self.units[key] = (channel, ureg(unit_str))
                    except:
                        self.units[key] = (channel, ureg.dimensionless)
                else:
                    self.units[key] = (channel, ureg.dimensionless)

def getUnitKey(self, key: str) -> Tuple:
    """
    Get unit for a specific key
    
    Parameters
    ----------
    key : str
        Full key path (e.g., "kHz/FEPC-LNCMI/I_H1")
    
    Returns
    -------
    tuple
        (symbol, pint.Unit)
    """
    if not self.units:
        self.Units()
    
    if key not in self.units:
        raise KeyError(f"Key '{key}' not found in units dict")
    
    return self.units[key]
```

---

### Step 6: Update HybridRun.getData()

**File:** `hybrid/hybrid_run.py`

**Tasks:**
1. Simplify `HybridRun.getData()` to delegate to `HybridData.getData()`
2. Keep downsampling and caching logic in HybridRun
3. Remove parsing logic (now handled by HybridData)

**Implementation:**
```python
def getData(
    self,
    key: Optional[str] = None,
    downsample: Optional[int] = None,
    options: Optional[LoadOptions] = None,
) -> Union[Dict, pd.DataFrame]:
    """
    Get data for a specific key (MagnetRun-compatible)
    
    Parameters
    ----------
    key : str, optional
        Data key in format 'type/system/variable' or 'type/system'
        Examples:
        - 'kHz/FEPC-LNCMI/I_H1' - specific kHz variable
        - 'rms/FEPC-LNCMI' - all RMS variables
    downsample : int, optional
        Target number of points for downsampling
    options : LoadOptions, optional
        Additional loading options
    
    Returns
    -------
    pd.DataFrame or dict
    """
    if self.HybridData is None:
        raise RuntimeError("HybridRun.getData: no HybridData associated")
    
    opts = options or self.default_options
    if downsample is not None:
        opts = LoadOptions(
            lazy=opts.lazy,
            cache=opts.cache,
            downsample=downsample,
            downsample_method=opts.downsample_method,
            start_time=opts.start_time,
            end_time=opts.end_time,
            hours=opts.hours,
            apply_calib=opts.apply_calib,
            cnv_dir=opts.cnv_dir,
        )
    
    # Check cache
    cache_key = f"{key}:{opts.downsample}:{opts.hours}"
    if opts.cache and cache_key in self._cache:
        entry = self._cache[cache_key]
        logger.debug(f"Cache hit for {cache_key}")
        return entry.data
    
    # Delegate to HybridData (now handles all parsing)
    df = self.HybridData.getData(key)
    
    # Apply time filtering if specified
    if opts.hours is not None:
        # TODO: implement time filtering based on hours
        pass
    
    # Apply downsampling if requested
    if opts.downsample and len(df) > opts.downsample:
        df = self._downsample_dataframe(df, opts.downsample, opts.downsample_method)
    
    # Cache result
    if opts.cache:
        self._add_to_cache(cache_key, df, opts)
    
    return df
```

---

### Step 7: Add Compatibility Methods

**File:** `hybrid/hybrid_data.py`

**Tasks:**
1. Add `extractData(keys)` method (used by MagnetData)
2. Add `getDuration(group)` method
3. Add `getStartDate(group)` method
4. Add `info()` method for debugging

**Implementation:**
```python
def extractData(self, keys: List[str]) -> pd.DataFrame:
    """
    Extract columns for specified keys (MagnetData-compatible)
    
    Parameters
    ----------
    keys : list[str]
        List of keys to extract
    
    Returns
    -------
    pd.DataFrame
    """
    return self.getData(keys)

def getDuration(self, group: str = None) -> float:
    """
    Compute duration of the run in seconds
    
    Parameters
    ----------
    group : str, optional
        Group name (default: first available group)
    
    Returns
    -------
    float
        Duration in seconds
    """
    if group is None:
        group = list(self.Groups.keys())[0]
    
    if group not in self.Groups:
        raise ValueError(f"Group '{group}' not found")
    
    # Get first channel from group
    channel = list(self.Groups[group].keys())[0]
    metadata = self.Groups[group][channel]
    
    dt = metadata["wf_increment"]
    samples = metadata["wf_samples"]
    duration = dt * samples
    
    return duration

def getStartDate(self, group: str = None) -> Tuple:
    """
    Get start timestamp
    
    Parameters
    ----------
    group : str, optional
        Group name
    
    Returns
    -------
    tuple
        (start_date, start_time, end_date, end_time)
    """
    if group is None:
        group = list(self.Groups.keys())[0]
    
    channel = list(self.Groups[group].keys())[0]
    start_t = self.Groups[group][channel]["wf_start_time"]
    
    # Calculate end time
    duration = self.getDuration(group)
    end_t = start_t + timedelta(seconds=duration)
    
    dformat = "%Y.%m.%d"
    tformat = "%H:%M:%S"
    
    start_date = start_t.strftime(dformat)
    start_time = start_t.strftime(tformat)
    end_date = end_t.strftime(dformat)
    end_time = end_t.strftime(tformat)
    
    return (start_date, start_time, end_date, end_time)

def info(self):
    """Print information about HybridData (similar to MagnetData.info())"""
    from tabulate import tabulate
    
    print(f"HybridData: {self.FileName}, Type={self.Type}")
    
    headers = ["Group", "Channel", "Samples", "Increment", "Start Time", "Start Offset"]
    tables = []
    
    for group, channels in self.Groups.items():
        for channel, metadata in channels.items():
            if isinstance(metadata, dict):
                table = [
                    group,
                    channel,
                    metadata.get("wf_samples", "N/A"),
                    metadata.get("wf_increment", "N/A"),
                    metadata.get("wf_start_time", "N/A"),
                    metadata.get("wf_start_offset", 0.0),
                ]
                tables.append(table)
    
    print(tabulate(tables, headers=headers, tablefmt="grid"))
```

---

### Step 8: Testing and Validation

**Tasks:**
1. Create test script that compares MagnetRun and HybridRun interfaces
2. Test lazy loading (verify data not loaded until accessed)
3. Test memory usage with large datasets
4. Validate all methods work with both single and multiple FEPC systems

**Test script template:**
```python
#!/usr/bin/env python3
"""Test HybridData TDMS compatibility"""

from python_magnetrun.hybrid.hybrid_data import HybridData
from python_magnetrun.hybrid.hybrid_run import HybridRun
from python_magnetrun.MagnetRun import MagnetRun
import pandas as pd

def test_interface_compatibility():
    """Test that HybridData and MagnetData have compatible interfaces"""
    
    # Load hybrid data
    hdata = HybridData.fromdir("/path/to/data", "2025-01-06", fepc_system=None)
    
    # Load TDMS data
    mrun = MagnetRun.fromtdms("M10", "insert", "/path/to/file.tdms")
    mdata = mrun.getMData()
    
    # Test common methods
    print("Testing getKeys()...")
    hkeys = hdata.getKeys()
    mkeys = mdata.getKeys()
    print(f"  HybridData keys: {len(hkeys)}")
    print(f"  MagnetData keys: {len(mkeys)}")
    
    print("\nTesting getData() with group/channel...")
    # Assuming both have similar channel
    hdf = hdata.getData("kHz/FEPC-LNCMI/I_H1")
    mdf = mdata.getData("Courants_Alimentations/I_H1")
    print(f"  HybridData: {type(hdf)}, shape={hdf.shape}")
    print(f"  MagnetData: {type(mdf)}, shape={mdf.shape}")
    
    print("\nTesting getTdmsData()...")
    hdf2 = hdata.getTdmsData("kHz/FEPC-LNCMI", "I_H1")
    mdf2 = mdata.getTdmsData("Courants_Alimentations", "I_H1")
    print(f"  HybridData: {type(hdf2)}")
    print(f"  MagnetData: {type(mdf2)}")
    
    print("\nTesting Units()...")
    hdata.Units()
    mdata.Units()
    print(f"  HybridData units: {len(hdata.units)} defined")
    print(f"  MagnetData units: {len(mdata.units)} defined")
    
    print("\nTesting info()...")
    hdata.info()
    mdata.info()

def test_lazy_loading():
    """Verify lazy loading works"""
    import sys
    
    hdata = HybridData.fromdir("/path/to/data", "2025-01-06")
    
    # Check memory before access
    print("Before accessing data:")
    print(f"  Data dict keys: {list(hdata.Data.keys())}")
    
    # Access one channel
    print("\nAccessing first channel...")
    df = hdata.getData("kHz/FEPC-LNCMI/I_H1")
    print(f"  Loaded: {df.shape}")
    
    # Check memory after
    print("\nAfter accessing data:")
    print(f"  Data dict keys: {list(hdata.Data.keys())}")

def test_multiple_systems():
    """Test handling multiple FEPC systems"""
    hdata = HybridData.fromdir("/path/to/data", "2025-01-06", fepc_system=None)
    
    print("Available groups:")
    for group in hdata.Groups.keys():
        print(f"  {group}: {len(hdata.Groups[group])} channels")
    
    # Access from different systems
    if "kHz/FEPC-LNCMI" in hdata.Groups and "kHz/FEPC-AUX-LNCMI" in hdata.Groups:
        df1 = hdata.getData("kHz/FEPC-LNCMI/I_H1")
        df2 = hdata.getData("kHz/FEPC-AUX-LNCMI/I_AUX")
        print(f"\nFEPC-LNCMI: {df1.shape}")
        print(f"FEPC-AUX-LNCMI: {df2.shape}")

if __name__ == "__main__":
    test_interface_compatibility()
    test_lazy_loading()
    test_multiple_systems()
```

---

## Migration Notes

### Backward Compatibility

The refactoring should maintain backward compatibility:

✅ **Still works:**
```python
hrun.getData("kHz/FEPC-LNCMI/I_H1")  # Existing code unchanged
```

✅ **New features:**
```python
hdata.getData("kHz/FEPC-LNCMI")      # Get all channels
hdata.getTdmsData("kHz/FEPC-LNCMI", "I_H1")  # TDMS-style
hdata.Units()  # Unit handling
hdata.info()   # Detailed info
```

### Performance Considerations

1. **First access may be slow** - Loading metadata from CFG/RMS files
2. **Subsequent access is fast** - Metadata cached
3. **Memory usage controlled** - Only loaded channels consume memory
4. **Disk I/O optimized** - Memory mapping where possible

### Breaking Changes

Minimal breaking changes expected:

⚠️ **Changed:**
- `Groups` structure now contains channels (was just metadata)
- `Keys` list contains full paths (was just group names)
- `Data` dict contains lazy loaders (was empty)

✅ **Compatible:**
- `getData(key)` still works with full paths
- `getKeys()` returns full list
### Phase 1: ANALOG Data Only

- **Step 1** (Lazy loading): 2-3 hours
- **Step 2** (Build groups): 3-4 hours
- **Step 3** (getTdmsData): 1-2 hours
- **Step 4** (getData): 1 hour
- **Step 5** (Units): 1-2 hours
- **Step 6** (HybridRun): 1 hour
- **Step 7** (Compatibility): 2-3 hours
- **Step 8** (Testing): 2-3 hours

**Phase 1 Total: 13-18 hours** of development time

### Phase 2: DIGITAL Data Support (Future)

- Extend lazy loading for boolean arrays: 1-2 hours
- Update group building for DIG cards: 1-2 hours
- Add digital data reading logic: 2-3 hours
- Update tests: 1-2 hours

**Phase 2 Total: 5-9 hours** additional hour
### Phase 1 (ANALOG Data)

✅ HybridData has identical interface to MagnetData for ANALOG channels  
✅ Can use HybridRun and MagnetRun interchangeably in analysis code  
✅ Lazy loading prevents memory issues with large kHz files  
✅ Multiple FEPC systems supported in single instance  
✅ Units handling compatible with pint  
✅ All existing tests pass  
✅ New tests validate TDMS compatibility  
✅ ANALOG kHz and RMS data fully functional  

### Phase 2 (DIGITAL Data - Future)

⏳ DIGITAL cards discoverable in Groups  
⏳ DIGITAL channels accessible via getData()  
⏳ Boolean data types handled correctly  
⏳ Mixed ANALOG/DIGITAL analysis supported
---

## Success Criteria

✅ HybridData has identical interface to MagnetData  
✅ Can use HybridRun and MagnetRun interchangeably in analysis code  
✅ Lazy loading prevents memory issues with large kHz files  
✅ Multiple FEPC systems supported in single instance  
✅ Units handling compatible with pint  
✅ All existing tests pass  
✅ New tests validate TDMS compatibility  

---

## References

- [MagnetData implementation](../python_magnetrun/magnetdata.py)
- [MagnetRun implementation](../python_magnetrun/MagnetRun.py)
- [Current HybridData](hybrid_data.py)
- [Current HybridRun](hybrid_run.py)
- [TDMS file format docs](https://www.ni.com/en-us/support/documentation/supplemental/06/the-ni-tdms-file-format.html)
