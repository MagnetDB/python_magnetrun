# Examples

This directory contains example scripts demonstrating various features of the `python_magnetrun` package.

## Hybrid Data Plotting Examples

### 1. Minimal Example: `plot_hybrid_minimal.py`

A simplified example showing the core concepts of loading and plotting hybrid kHz data alongside pupitre and TDMS data.

**Features:**
- Simple, easy-to-understand code
- Hardcoded configuration (edit the script to customize)
- Demonstrates field name mapping using a dictionary
- Shows date-based file discovery

**Usage:**
```bash
# Edit the script to set your data directories and preferences
python plot_hybrid_minimal.py
```

**Good for:** Learning the basics, quick prototyping

### 2. Full-Featured Example: `plot_hybrid_with_pupitre_tdms.py`

A complete command-line tool with extensive options and error handling.

**Features:**
- Full command-line argument parsing
- Configurable data directories
- Hour-based filtering
- Save and/or show plots
- Comprehensive error handling
- Automatic file discovery

**Usage:**
```bash
# Basic usage
python plot_hybrid_with_pupitre_tdms.py \
    -d 2025-01-27 \
    -s FEPC-AUX-LNCMI \
    -k ALIM1_J1 \
    --site M10 \
    --show

# With custom directories
python plot_hybrid_with_pupitre_tdms.py \
    -d 2025-01-27 \
    -s FEPC-LNCMI \
    -k I_H1 \
    --site M9 \
    --hybrid-dir /path/to/hybrid/data \
    --pupitre-dir /path/to/pupitre \
    --pigbrother-dir /path/to/pigbrother \
    --hours 10,11,12 \
    --save output.png \
    --show
```

**Good for:** Production use, automation, detailed analysis

See [README_hybrid_plotting.md](README_hybrid_plotting.md) for detailed documentation.

## Key Concepts Demonstrated

### 1. Field Name Mapping

Different data sources use different field names for the same physical quantity. Use dictionaries to map between them:

```python
FIELD_MAPPING = {
    # hybrid_key -> (pupitre_field, tdms_field)
    "kHz/FEPC-AUX-LNCMI/ALIM1_J1": ("IH", "Référence_GR1"),
    "kHz/FEPC-LNCMI/I_H1": ("IH", "Référence_GR1"),
}
```

### 2. Date-Based File Discovery

Find corresponding files across different data sources using the date:

```python
# Pupitre files: M10/2025.01.27---*.txt
pupitre_pattern = f"{pupitre_dir}/{site}/{year}.{month:02d}.{day:02d}*.txt"

# TDMS files: M10/Overview/M10_Overview_250127-*.tdms
tdms_pattern = f"{pigbrother_dir}/{site}/Overview/{site}_Overview_{yy:02d}{mm:02d}{dd:02d}-*.tdms"
```

### 3. Loading Different Data Sources

```python
# Hybrid data
hrun = HybridRun.fromdir(base_dir, date_str, fepc_system, site)

# Pupitre data
pupitre = MagnetRun.fromtxt(site, insert, pupitre_file)

# TDMS data
tdms = MagnetRun.fromtdms(site, insert, tdms_file)
```

### 4. Data Access and Plotting

```python
# Hybrid: high-frequency with downsampling
data, time = hrun.getData(hybrid_key, downsample=10000, hours=[10, 11, 12])

# Pupitre and TDMS: direct access
pupitre_values = pupitre.getMData().getData(pupitre_field)
pupitre_time = pupitre.getMData().getData('t')
```

## Directory Structure Expected

```
workspace/
├── hybrid_data/                    # Hybrid kHz/RMS/Trigger data
│   ├── kHz/
│   │   └── YYYY-MM-DD/
│   │       ├── FEPC-LNCMI/
│   │       └── FEPC-AUX-LNCMI/
│   ├── rms/
│   └── trigger/
│
├── pupitre_datadir/                # Pupitre text files
│   ├── M9/
│   │   └── YYYY.MM.DD---HH:MM:SS.txt
│   ├── M10/
│   └── M8/
│
└── pigbrother_datadir/             # TDMS files
    └── Fichiers_Data/
        ├── M9/
        │   └── Overview/
        │       └── M9_Overview_YYMMDD-HHMM.tdms
        ├── M10/
        └── M8/
```

## Configuration

Default data directories are defined in `python_magnetrun/analysis/config.py`:

```python
DEFAULT_DATA_DIR = "/home/LNCMI-G/christophe.trophime/LNCMIG-Data/srv-data-install"
DEFAULT_PIGBROTHER_DATA_DIR = "/home/.../pigbrotherdata/Fichiers_Data"
```

You can override these using command-line arguments or by editing the scripts.

## Site-Specific Configurations

Different sites (M8, M9, M10) have different channel naming conventions:

| Site | GR1 Current | GR2 Current | Note            |
| ---- | ----------- | ----------- | --------------- |
| M9   | IH          | IB          | H=High, B=Bas   |
| M10  | IB          | IH          | Swapped from M9 |
| M8   | IB          | IH          | Same as M10     |

See `python_magnetrun/analysis/config.py` for complete site configurations.

## Troubleshooting

### Import Errors

Make sure the hybrid module is in your Python path:

```bash
export PYTHONPATH=/path/to/python_magnetrun:$PYTHONPATH
```

Or run from the repository root:

```bash
cd /path/to/python_magnetrun
python examples/plot_hybrid_minimal.py
```

### No Files Found

- Check your date format
- Verify data directories exist and contain files for the specified date
- Check site name matches directory structure (M9, M10, M8)

### Field Not Found

- Use `print(data.getMData().getKeys())` to see available fields
- Update the mapping dictionaries to match your actual field names
- Check site-specific configurations

## Further Reading

- [Hybrid Module Documentation](../python_magnetrun/hybrid/README.md)
- [Analysis Configuration](../python_magnetrun/analysis/config.py)
- [File Discovery](../python_magnetrun/analysis/loaders.py)
- [Main README](../README.md)

## Contributing

To add new examples:

1. Create a descriptive filename (e.g., `plot_xyz.py`)
2. Include docstrings and comments
3. Add usage examples to this README
4. Consider creating both minimal and full-featured versions

## License

Same as parent project.
