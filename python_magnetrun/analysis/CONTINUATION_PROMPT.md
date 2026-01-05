# Continuation Prompt: Analysis Module Refactoring

## Project Context

We are refactoring `python_magnetrun/analysis-refactor.py` (1328 lines) into a modular `python_magnetrun/analysis/` package. This is a data analysis tool for magnet run experiments at LNCMI (Laboratoire National des Champs Magnétiques Intenses) that processes TDMS and text files from different data sources (overview, archive, pupitre, incidents) at different sampling rates.

## Original File Location
- Source: `python_magnetrun/analysis-refactor.py`
- Refactoring plan: `REFACTORING_PROMPT.md`

## Completed Steps

### Step 2: config.py ✅ (24KB)
Extracted all configuration with dataclasses:
- Constants: `SAMPLING_RATE_OVERVIEW` (1 Hz), `SAMPLING_RATE_ARCHIVE` (120 Hz), `SAMPLING_RATE_INCIDENTS` (4800 Hz)
- Time offsets: `TIME_OFFSET_*` calculated as (1/rate)/2
- Analysis defaults: `LAG_THRESHOLD_RATIO`, `DEFAULT_WINDOW_SIZE`, `DEFAULT_BINS`, `DEFAULT_LEVELS`
- Dataclasses: `ColorConfig`, `ChannelMapping`, `VoltageChannelMapping`, `SiteConfig`, `ThresholdConfig`, `AnalysisConfig`
- Pre-defined: `SITE_CONFIGS` dict with M8, M9, M10 configurations
- Backward compatibility: `setup()` function returns original 6-tuple format

### Step 4: loaders.py ✅ (27KB)
Extracted file loading and discovery:
- `convert_to_timestamp()` - Parse date/time strings
- `extract_data()` - Extract timestamps from files
- `find_files()` - Build glob patterns for related files
- `select_files()` - Filter files by timestamp range
- `load_df()`, `load_data()`, `merge_data()` - DataFrame loading
- Dataclasses: `FileMetadata`, `FileSet`
- Class: `FileDiscovery` - Main discovery orchestrator
- Backward compatibility: `discover_files()` returns dict format

### Step 6: synchronization.py ✅ (23KB)
Extracted time synchronization logic:
- `synchronize_data()` - Align DataFrame to reference timestamp
- `apply_lag_correction()` - Apply lag correction
- `compute_lag()`, `lag_correlation()` - Cross-correlation based lag detection
- `compute_regime_score()`, `find_best_matching_regime()` - Regime matching
- `check_lag_reliability()` - Validate lag quality
- Dataclasses: `SyncResult`, `LagResult`, `RegimeMatch`
- Utilities: `add_time_column()`, `get_timestamp_info()`

### Step 5: plotting.py ✅ (28KB)
Extracted visualization functions with downsampling support:
- Dataclasses: `PlotStyle`, `PlotColors` for configuration
- Downsampling: `downsample_for_plot()`, `downsample_dataframe()`, `downsample_minmax()`, `estimate_downsample_percent()`
- Main plots: `plot_data()` - multi-source comparison with optional downsampling
- `plot_comparison()` - two-series comparison
- `plot_time_series()` - single/multi-channel time series
- Regime visualization: `plot_regimes()` with colored axvspan
- Incident markers: `plot_incidents_markers()` with vertical lines
- Interactive annotations for incidents (clickable)
- Utilities: `create_figure_grid()`, `save_figure()`, `setup_matplotlib_defaults()`

### Step 7: metrics.py ✅ (24KB)
Extracted distance metrics and correlation functions:
- Distance metrics: `calc_euclidean()`, `calc_mae()`, `calc_mape()`, `calc_correlation()`, `compute_all_distances()`
- Cross-correlation: `crosscorr()`, `compute_tlcc()`, `compute_windowed_tlcc()`, `compute_rolling_tlcc()`, `compute_pearson_correlation()`
- DTW functions: `compute_dtw_distance()`, `compute_dtw_distance_fast()`, `dtw_with_paa()`
- Plotting: `plot_tlcc()`, `plot_dtw_alignment()`
- Dataclasses: `DistanceResult`, `DTWResult`, `CorrelationResult`, `TLCCResult`
- Convenience: `compare_series()` for full analysis

### Step 3: cli.py ✅ (18KB)
Comprehensive logging infrastructure and CLI:
- Logging: `setup_logging()`, `get_logger()`, `set_log_level()`, `LogConfig`
- Formatters: `ColoredFormatter` (ANSI colors), `JSONFormatter` (structured logs)
- Progress: `ProgressTracker` (with rate/ETA), `timed_operation()` context manager
- Context: `LogContext` for adding metadata to log records
- Argument parsing: `create_argument_parser()`, `parse_arguments()`, `args_to_processing_config()`
- Entry point: `main()` with full integration

### Step 8: processing.py ✅ (28KB)
Extracted main orchestration logic:
- Configuration: `ProcessingConfig` for workflow settings
- Main dataclasses: `OverviewRecord` (complete experiment data), `ProcessingResult` (multi-file results)
- Main functions: `process_overview_file()`, `process_experiment()`
- Data loading: `load_overview_data()`, `load_archive_data()`, `load_pupitre_data()`, `load_incidents_data()`
- Utilities: `compute_time_offset()`, `add_time_column_with_offset()`, `get_site_config()`, `summarize_record()`, `print_record_summary()`
- Legacy compatibility: `create_overview_dict()` converts to original format

### Existing: py.typed
PEP 561 marker for type checking support.

## Remaining Steps

**All steps are now complete!** The refactoring is finished.

Optional future enhancements:
- Integration testing with real TDMS files
- Performance profiling and optimization
- Additional CLI subcommands

## Current Module Structure - COMPLETE

```
python_magnetrun/analysis/
├── __init__.py          # Public API exports (7KB)
├── config.py            # ✅ Configuration dataclasses (24KB)
├── loaders.py           # ✅ File loading/discovery (27KB)
├── synchronization.py   # ✅ Time sync logic (23KB)
├── metrics.py           # ✅ Distance metrics/DTW/correlation (24KB)
├── plotting.py          # ✅ Visualization with downsampling (26KB)
├── processing.py        # ✅ Main orchestration (28KB)
├── cli.py               # ✅ Logging & CLI infrastructure (18KB)
└── py.typed             # PEP 561 marker

tests/analysis/
├── __init__.py
├── test_config.py       # ✅ Complete (14KB)
├── test_loaders.py      # ✅ Complete
├── test_synchronization.py  # ✅ Complete
├── test_metrics.py      # ✅ Complete
├── test_plotting.py     # ✅ Complete
├── test_processing.py   # ✅ Complete
└── test_cli.py          # ✅ Complete
```

## Refactoring Summary

The original 1328-line `analysis-refactor.py` has been successfully broken down into:

| Module | Purpose | Size |
|--------|---------|------|
| config.py | Configuration dataclasses & site settings | 24KB |
| loaders.py | File discovery & data loading | 27KB |
| synchronization.py | Time sync & lag computation | 23KB |
| metrics.py | Distance metrics, DTW, correlation | 24KB |
| plotting.py | Visualization with downsampling | 26KB |
| processing.py | Main orchestration & workflow | 28KB |
| cli.py | Logging infrastructure & CLI | 18KB |

**Total: ~170KB of well-structured, tested, documented code**

## Established Patterns

1. **Dataclasses** - Use frozen dataclasses for immutable configuration
2. **Type hints** - Full type annotations with `from __future__ import annotations`
3. **Docstrings** - NumPy style with Parameters, Returns, Examples
4. **Logging** - Module-level logger: `logger = logging.getLogger("magnetrun.analysis.<module>")`
5. **Backward compatibility** - Provide functions that return original formats
6. **Lazy imports** - Import heavy dependencies (MagnetRun) inside functions to avoid circular deps
7. **Testing** - Comprehensive pytest tests with mocking where needed

## Key Data Structures

### overview_dict (from original)
```python
overview_dict[filename] = {
    "mode": mode,
    "signature": {},  # Signature objects per key
    "sources": dict_files,  # FileSet equivalent
    "data": {
        "overview": pd.DataFrame(),
        "pupitre": pd.DataFrame(),
        "archive": pd.DataFrame(),
        "default": [],
        "trigger": [],
        "spike": [],
    },
    "t0": t0,
    "BP": {},
    "teb": {},
    "debitbrut": {},
    "flow_params": {},
}
```

### Signature class (existing in python_magnetrun/signature.py)
Has: name, symbol, unit, t0, timeshift, changes, regimes, times, values

## Dependencies
- pandas, numpy - Data manipulation
- matplotlib - Plotting
- natsort - Natural sorting of files
- scipy.signal - Cross-correlation (correlate, correlation_lags)
- dtaidistance - DTW (optional, for metrics)
- tabulate - Table formatting (used in original)

## Instructions for Continuation

1. **Read the original file** to understand the complete context:
   - `python_magnetrun/analysis-refactor.py`
   - `python_magnetrun/processing/correlations.py` (for metrics)

2. **Follow established patterns** from completed modules

3. **All major steps complete!** The refactoring is finished. Optional enhancements:
   - Step 3 (cli.py) - Add structured logging configuration if needed
   - Integration testing with real TDMS files
   - Performance profiling and optimization

4. **Update `__init__.py`** with new exports after each module

5. **Create tests** in `tests/analysis/test_<module>.py`

6. **Verify imports** work from package level:
   ```python
   from python_magnetrun.analysis import <new_exports>
   ```

## Files to Reference
- Original: `python_magnetrun/analysis-refactor.py`
- Correlations: `python_magnetrun/processing/correlations.py`
- Signature: `python_magnetrun/signature.py`
- Trends: `python_magnetrun/processing/trends.py`
- Completed modules in `python_magnetrun/analysis/`
