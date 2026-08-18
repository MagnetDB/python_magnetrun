# VProcess Example Scripts

Command-line tools for batch processing and unified CLI access to VProcess data files.
These scripts live in `examples/` and rely on the `python_magnetrun` package being installed.

## Scripts

| Script | Purpose |
|--------|---------|
| `vprocess_validate.py` | Validate a VProcess file structure and content |
| `vprocess_batch.py` | Batch-process or merge multiple VProcess files |
| `vprocess_cli.py` | Unified CLI combining all operations |
| `vprocess_args.py` | Argument parser definitions (imported by `vprocess_cli.py`) |

## File Validation

```bash
# Basic validation
python vprocess_validate.py data.vprocess

# Full validation with data checking
python vprocess_validate.py data.vprocess --check-data

# Quiet mode (only errors)
python vprocess_validate.py data.vprocess --quiet
```

## Batch Processing

```bash
# Merge all files in directory to CSV
python vprocess_batch.py --dir ./data --output merged.csv --merge

# Export specific variables to HDF5
python vprocess_batch.py --dir ./data --vars TT115A TT508A --format hdf5 --merge

# List common variables across all files
python vprocess_batch.py --dir ./data --list-common-vars

# Analyze files and create summary
python vprocess_batch.py --dir ./data --analyze --output summary.csv
```

## Unified CLI

All operations available through a single interface:

```bash
python vprocess_cli.py info data.vprocess
python vprocess_cli.py validate data.vprocess --check-data
python vprocess_cli.py plot data.vprocess --vars TT115A TT508A
python vprocess_cli.py batch --dir ./data --merge --output merged.csv
```

## See Also

- [vprocess/README.md](../python_magnetrun/hybrid/vprocess/README.md) — file format, API reference, validation, and library usage
- [README_hybrid_plotting.md](README_hybrid_plotting.md) — plotting hybrid data
