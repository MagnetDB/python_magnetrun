# Magnetrun Analysis Module

A comprehensive Python package for analyzing magnetrun experimental data from LNCMI (Laboratoire National des Champs Magnétiques Intenses). This module processes TDMS and text files from different measurement sites (M8, M9, M10), synchronizes data from multiple sources, and provides visualization and analysis tools.

## Features

- **Multi-source data loading**: Load and merge data from overview, archive, pupitre, and incident files
- **Automatic file discovery**: Find related files based on naming conventions and timestamps
- **Time synchronization**: Synchronize clocks between different data sources with lag detection
- **Distance metrics**: Compute similarity metrics including Euclidean, MAPE, correlation, and DTW
- **Visualization**: Plot multi-source comparisons with downsampling for large datasets
- **Structured logging**: Comprehensive logging with console colors, file output, and JSON format
- **CLI interface**: Full command-line interface for batch processing

## Installation

The analysis module is part of the `python_magnetrun` package:

```bash
# Install dependencies
pip install numpy pandas scipy matplotlib natsort

# Optional: for DTW support
pip install fastdtw
```

## Module Structure

```
python_magnetrun/analysis/
├── __init__.py          # Public API exports
├── config.py            # Configuration dataclasses & site settings
├── loaders.py           # File discovery & data loading
├── synchronization.py   # Time sync & lag computation
├── metrics.py           # Distance metrics, DTW, correlation
├── plotting.py          # Visualization with downsampling
├── processing.py        # Main orchestration & workflow
├── cli.py               # Logging infrastructure & CLI
└── py.typed             # PEP 561 type checking marker
```

## Quick Start

### Basic Usage

```python
from python_magnetrun.analysis import (
    process_overview_file,
    ProcessingConfig,
    setup_logging,
)

# Setup logging
logger = setup_logging(debug=True)

# Configure processing
config = ProcessingConfig(
    pupitre_datadir="/path/to/pupitre/data",
    synchronize=True,
    compute_lag=True,
    downsample_percent=10.0,  # Plot 10% of points
)

# Process an overview file
record = process_overview_file("M9_Overview_241106-091500.tdms", config)

# Access results
print(f"Site: {record.site}")
print(f"Duration: {record.duration} seconds")
print(f"Signatures: {list(record.signatures.keys())}")

# Get DataFrames
df_overview = record.get_overview()
df_archive = record.get_archive()
df_pupitre = record.get_pupitre()
```

### Command-Line Interface

```bash
# Basic processing with plot display
python -m python_magnetrun.analysis.cli M9_Overview_*.tdms --show

# Save plots to output directory
python -m python_magnetrun.analysis.cli input.tdms --save --output-dir ./plots

# Compute distance metrics (Euclidean, MAPE, Correlation, DTW)
python -m python_magnetrun.analysis.cli input.tdms --distance

# Full analysis: synchronize, compute lag, compute metrics, save plots
python -m python_magnetrun.analysis.cli input.tdms --synchronize --lag --distance --save

# With debug logging to file
python -m python_magnetrun.analysis.cli input.tdms --debug --log-file analysis.log

# Downsample large datasets for faster plotting (plot only 10% of points)
python -m python_magnetrun.analysis.cli input.tdms --show --downsample 10

# Dry run (discover files without loading/processing)
python -m python_magnetrun.analysis.cli input.tdms --dry-run
```

The CLI performs the following operations based on flags:
- **--show/--save**: Generates multi-source comparison plots for each current key
- **--distance**: Computes Euclidean, MAPE, Correlation, and DTW metrics
- **--synchronize**: Synchronizes pupitre timestamps with overview
- **--lag**: Computes lag correlation between data sources
- **--downsample**: Reduces plotted points for large datasets (auto-detected if not specified)

[ ] Basic processing with plot display
[ ] Save plots to output directory
[ ] Compute distance metrics (Euclidean, MAPE, Correlation, DTW)
[ ] Full analysis: synchronize, compute lag, compute metrics, save plots
[ ] With debug logging to file
[ ] Downsample large datasets for faster plotting
[ ] Dry run (discover files without loading/processing)

## Module Documentation

### config.py - Configuration

Analysis parameters and site configuration.  Housing-specific role
assignments (`SiteConfig`) live in `python_magnetrun/site_config.py` and
are re-exported from `config.py` for backward compatibility.

```python
from python_magnetrun.analysis import (
    SiteConfig,
    AnalysisConfig,
    SITE_CONFIGS,
    get_site_config,
)

# Get built-in default for a housing
config = get_site_config("M9")
print(f"GR1 current channel: {config.reference_gr1_current}")  # "IH"
print(f"GR2 current channel: {config.reference_gr2_current}")  # "IB"
print(f"Supports hybrid: {config.supports_format('hybrid')}")  # False

# Load from a custom per-housing JSON file
config = get_housing_config("M9", json_file="M9-housing-config.json")

# Runtime override (e.g. GR1/GR2 swapped for an atypical run)
config = get_housing_config("M9", overrides={"gr1_current": "IB", "gr2_current": "IH"})

# Create full analysis configuration
analysis = AnalysisConfig.for_housing("M9")
```

**Key Classes:**
- `HousingConfig` - Housing-dependent sensor role assignments (defined in `housing_config.py`)
- `AnalysisConfig` - Full analysis configuration (housing + thresholds + channels + colours)
- `ColorConfig` - Plot color configuration
- `ThresholdConfig` - Regime detection thresholds

**Constants:**
- `SAMPLING_RATE_OVERVIEW` = 1 Hz
- `SAMPLING_RATE_ARCHIVE` = 120 Hz
- `SAMPLING_RATE_INCIDENTS` = 4800 Hz

### loaders.py - File Discovery & Loading

Automatic file discovery and data loading utilities.

```python
from python_magnetrun.analysis import (
    FileDiscovery,
    FileSet,
    FileMetadata,
    load_data,
    merge_data,
)

# Discover related files for an overview file
discovery = FileDiscovery(
    base_dir="/path/to/data",
    pupitre_datadir="/path/to/pupitre",
)

file_set = discovery.discover_for_overview("M9_Overview_241106-091500.tdms")
print(f"Archive files: {file_set.archive}")
print(f"Pupitre files: {file_set.pupitre}")
print(f"Incident files: {file_set.default + file_set.trigger + file_set.spike}")

# Load data from files
df_list = load_data(file_set.archive, site="M9", insert="", group="Courants", keys=["Courant_GR1"])
df_merged = merge_data(df_list)
```

**Key Classes:**
- `FileDiscovery` - Orchestrates file discovery
- `FileSet` - Container for related file paths
- `FileMetadata` - Metadata extracted from filenames

### synchronization.py - Time Synchronization

Synchronize timestamps between different data sources.

```python
from python_magnetrun.analysis import (
    synchronize_data,
    compute_lag,
    find_best_matching_regime,
    SyncResult,
    LagResult,
)

# Synchronize pupitre data with overview reference time
timeshift, df_synced = synchronize_data(df_pupitre, reference_t0)
print(f"Applied timeshift: {timeshift.total_seconds()} seconds")

# Compute lag using cross-correlation
df1_data = {"df": df_overview, "field": "Courant_GR1", "range": (0, 100)}
df2_data = {"df": df_pupitre, "field": "IH", "range": (0, 100)}
lag = compute_lag("timestamp", df1_data, df2_data)
print(f"Computed lag: {lag.total_seconds()} seconds")

# Find matching regimes between signatures
matches = find_best_matching_regime(overview_signature, pupitre_signature)
for regime, match, score, lags, indices in matches:
    print(f"Regime {regime}: score={score}, lag={lags}")
```

**Key Functions:**
- `synchronize_data()` - Apply timestamp synchronization
- `apply_lag_correction()` - Apply lag correction to DataFrame
- `compute_lag()` - Compute lag using cross-correlation
- `find_best_matching_regime()` - Match regimes between signatures
- `check_lag_reliability()` - Validate lag computation reliability

### metrics.py - Distance Metrics

Compute similarity and distance metrics between time series.

```python
from python_magnetrun.analysis import (
    calc_euclidean,
    calc_mape,
    calc_correlation,
    calc_dtw,
    cross_correlation,
    find_optimal_lag,
    MetricResult,
)

# Calculate various distance metrics
euclidean = calc_euclidean(series1, series2)
mape = calc_mape(series1, series2)
correlation = calc_correlation(series1, series2)

print(f"Euclidean distance: {euclidean.value}")
print(f"MAPE: {mape.value}%")
print(f"Correlation: {correlation.value}")

# Dynamic Time Warping
dtw_result = calc_dtw(series1, series2, radius=10)
print(f"DTW distance: {dtw_result.value}")
print(f"Warping path length: {len(dtw_result.metadata['path'])}")

# Cross-correlation for lag detection
correlation_values = cross_correlation(signal1, signal2)
optimal_lag = find_optimal_lag(signal1, signal2)
```

**Key Functions:**
- `calc_euclidean()` - Euclidean distance
- `calc_mape()` - Mean Absolute Percentage Error
- `calc_correlation()` - Pearson correlation coefficient
- `calc_dtw()` - Dynamic Time Warping distance
- `cross_correlation()` - Cross-correlation array
- `find_optimal_lag()` - Find optimal lag via cross-correlation

### plotting.py - Visualization

Plotting utilities with support for large datasets.

```python
from python_magnetrun.analysis import (
    plot_data,
    plot_comparison,
    plot_time_series,
    plot_regimes,
    downsample_for_plot,
    estimate_downsample_percent,
    PlotStyle,
    PlotColors,
)

# Estimate appropriate downsampling for large datasets
n_points = len(df)
downsample_pct = estimate_downsample_percent(n_points, target_points=10000)

# Plot multi-source comparison
fig = plot_data(
    df_overview, df_archive, df_pupitre, df_incidents,
    channels_dict, pupitre_dict,
    site="M9",
    tkey="t",
    key="Courant_GR1",
    title="M9 Run Analysis",
    msg="(synchronized)",
    downsample_percent=downsample_pct,
    show=True,
    save=True,
    output_path="output.png",
)

# Simple two-series comparison
plot_comparison(
    df1, df2,
    x_col="t",
    y_col1="Courant_GR1",
    y_col2="IH",
    label1="Overview",
    label2="Pupitre",
    downsample_percent=10.0,
)

# Manual downsampling for custom plots
x_ds, y_ds = downsample_for_plot(x, y, percent=5.0)
plt.plot(x_ds, y_ds)
```

**Downsampling Functions:**
- `downsample_for_plot()` - Uniform step-based downsampling
- `downsample_dataframe()` - Downsample entire DataFrame
- `downsample_minmax()` - Preserve min/max values (better for peaks)
- `estimate_downsample_percent()` - Auto-suggest downsampling percentage

**Plot Functions:**
- `plot_data()` - Multi-source comparison plot
- `plot_comparison()` - Two-series comparison
- `plot_time_series()` - One or more time series
- `plot_regimes()` - Add colored regime spans
- `plot_incidents_markers()` - Add incident markers

### processing.py - Main Orchestration

High-level processing workflow.

```python
from python_magnetrun.analysis import (
    process_overview_file,
    process_experiment,
    ProcessingConfig,
    OverviewRecord,
    ProcessingResult,
    create_overview_dict,
    summarize_record,
    print_record_summary,
)

# Configure processing
config = ProcessingConfig(
    pupitre_datadir="/data/pupitre",
    synchronize=True,
    compute_lag=True,
    compute_distance=False,
    downsample_percent=10.0,
    debug=True,
    show=False,
    save=True,
)

# Process single file
record = process_overview_file("M9_Overview_241106-091500.tdms", config)

# Print summary
print_record_summary(record)

# Process multiple files
result = process_experiment(
    ["M9_Overview_241106-091500.tdms", "M9_Overview_241106-093000.tdms"],
    config,
)
print(f"Processed {len(result)} files")
print(f"Errors: {result.errors}")

# Convert to legacy dict format (backward compatibility)
overview_dict = create_overview_dict(result)
```

**Key Classes:**
- `ProcessingConfig` - Workflow configuration
- `OverviewRecord` - Complete record for one overview file
- `ProcessingResult` - Results for multiple files

### cli.py - Logging & CLI

Comprehensive logging infrastructure and command-line interface.

```python
from python_magnetrun.analysis import (
    setup_logging,
    get_logger,
    set_log_level,
    LogConfig,
    ProgressTracker,
    timed_operation,
    LogContext,
)

# Setup logging with multiple outputs
logger = setup_logging(
    debug=True,
    log_file="analysis.log",      # Text log file
    json_file="analysis.json",    # Structured JSON log
    use_colors=True,              # Colored console output
)

# Get module-specific logger
proc_logger = get_logger("processing")
proc_logger.info("Starting processing")

# Change log level at runtime
set_log_level("WARNING")

# Time an operation
with timed_operation("Loading large dataset"):
    df = pd.read_csv("large_file.csv")
# Output: "Loading large dataset... Loading large dataset completed in 2.34s"

# Track progress
tracker = ProgressTracker(total=100, description="Processing files", log_interval=10)
for i in range(100):
    process_item(i)
    tracker.update()
tracker.finish()
# Output: "Processing files: 10/100 (10.0%) - 5.2/s - ETA: 17.3s"

# Add context to log records (useful for JSON logging)
with LogContext(file="data.tdms", site="M9"):
    logger.info("Processing file")  # JSON log includes file and site
```

**Logging Functions:**
- `setup_logging()` - Configure logging system
- `get_logger()` - Get module-specific logger
- `set_log_level()` - Change log level at runtime

**Progress Utilities:**
- `ProgressTracker` - Track progress with rate/ETA
- `timed_operation()` - Context manager for timing
- `LogContext` - Add metadata to log records

## CLI Reference

```
usage: python -m python_magnetrun.analysis.cli [-h] [--pupitre-datadir DIR]
                                                [--output-dir DIR] [--tkey {t,timestamp}]
                                                [--synchronize] [--lag] [--distance]
                                                [--flow-params] [--dry-run] [--bins N]
                                                [--window N] [--levels N]
                                                [--downsample PERCENT] [--show] [--save]
                                                [--debug] [--quiet] [--log-file FILE]
                                                [--json-log FILE] [--no-color]
                                                input_file [input_file ...]

Analyze magnetrun data from TDMS and pupitre files

positional arguments:
  input_file            Input TDMS overview files to process

Data directories:
  --pupitre-datadir DIR Directory containing pupitre data files
  --output-dir DIR      Directory for output files

Processing options:
  --tkey {t,timestamp}  Time column to use for plotting
  --synchronize         Synchronize pupitre clock with overview
  --lag                 Compute lag correlation between sources
  --distance            Compute distance/DTW metrics between series
  --flow-params         Compute flow parameters
  --dry-run             Discover files but don't load/process data

Analysis parameters:
  --bins N              Number of bins for histograms (default: 100)
  --window N            Rolling window size for smoothing (default: 10)
  --levels N            Number of levels for piecewise fitting (default: 3)
  --downsample PERCENT  Percentage of data points to plot (default: 100.0)

Output options:
  --show                Display plots interactively (requires X11)
  --save                Save plots to PNG files

Logging options:
  --debug               Enable debug output
  --quiet, -q           Only show warnings and errors
  --log-file FILE       Write logs to file
  --json-log FILE       Write structured JSON logs to file
  --no-color            Disable colored console output
```

## Site Configurations

Housing configurations define which pupitre field plays each GR role.
The canonical source is `python_magnetrun/<Housing>-site-config.json`;
see `python_magnetrun/site_config.py` and the main README for management tools.

### M9 (default: H supply = GR1, B supply = GR2)
| Parameter | GR1 | GR2 |
|-----------|-----|-----|
| Current | IH | IB |
| Flow | FlowH | FlowB |
| RPM | RpmH | RpmB |
| Pressure In | HPH | HPB |
| Voltages | UH, Ucoil1–14 | UB, Ucoil15, Ucoil16 |
| Formats | pupitre, pigbrother | |

### M8 (default: B supply = GR1, H supply = GR2; also runs hybrid)
| Parameter | GR1 | GR2 |
|-----------|-----|-----|
| Current | IB | IH |
| Flow | FlowB | FlowH |
| RPM | RpmB | RpmH |
| Pressure In | HPB | HPH |
| Voltages | UB, Ucoil15, Ucoil16 | UH, Ucoil1–14 |
| Formats | pupitre, pigbrother, hybrid | |

### M10 (same convention as M8)
| Parameter | GR1 | GR2 |
|-----------|-----|-----|
| Current | IB | IH |
| Flow | FlowB | FlowH |
| RPM | RpmB | RpmH |
| Pressure In | HPB | HPH |
| Voltages | UB, Ucoil15, Ucoil16 | UH, Ucoil1–14 |
| Formats | pupitre, pigbrother | |

> **Note:** The default GR1/GR2 assignment can be overridden at runtime for
> atypical runs via `get_site_config("M9", overrides={"gr1_current": "IB", ...})`.

## Data Flow

```
┌─────────────────┐
│  Overview TDMS  │ (1 Hz)
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  FileDiscovery  │────▶│   Archive TDMS  │     │  Pupitre TXT    │
└────────┬────────┘     │    (120 Hz)     │     │  (variable Hz)  │
         │              └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         │              ┌────────┴────────┐              │
         │              │  Incident TDMS  │              │
         │              │   (4800 Hz)     │              │
         │              └────────┬────────┘              │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OverviewRecord                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐│
│  │ overview │  │ archive  │  │ pupitre  │  │ incidents        ││
│  │ DataFrame│  │ DataFrame│  │ DataFrame│  │ (default/trigger/││
│  │          │  │          │  │          │  │  spike)          ││
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘│
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ signatures, sync_info, metrics, flow_params              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│   Visualization │
│   & Analysis    │
└─────────────────┘
```

## Testing

Run the test suite:

```bash
# Run all analysis tests
pytest tests/analysis/ -v

# Run specific test module
pytest tests/analysis/test_processing.py -v

# Run with coverage
pytest tests/analysis/ --cov=python_magnetrun.analysis --cov-report=html
```

## Contributing

When adding new functionality:

1. Follow the existing patterns (frozen dataclasses, type hints, NumPy docstrings)
2. Add comprehensive tests in `tests/analysis/`
3. Update this README with new features
4. Use the module logger: `logger = logging.getLogger("magnetrun.analysis.modulename")`

## License

Part of the python_magnetrun package. See the main repository for license information.
