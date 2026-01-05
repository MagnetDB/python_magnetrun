# Refactoring Prompt for `analysis-refactor.py`

This document provides a step-by-step guide to refactor the `analysis-refactor.py` script for improved readability, maintainability, and testability.

---

## Overview

**Current state:**
- Single file with 1328 lines
- `main()` function is 858 lines long
- 80+ `print()` statements scattered throughout
- No docstrings on functions
- Hardcoded paths and magic numbers
- Configuration dictionaries returned as tuples

**Target state:**
- Modular structure with separate files for distinct concerns
- Functions under 50 lines each
- Proper logging infrastructure
- Type hints and docstrings on all public functions
- Configuration in dataclasses
- Constants for magic numbers

---

## Step 1: Create the Module Structure

Create the following directory structure:

```
python_magnetrun/
├── analysis/
│   ├── __init__.py
│   ├── cli.py              # Command-line interface and main entry point
│   ├── config.py           # Configuration dataclasses and constants
│   ├── loaders.py          # Data loading and file operations
│   ├── synchronization.py  # Time synchronization logic
│   ├── metrics.py          # Distance and similarity metrics
│   ├── plotting.py         # Visualization functions
│   └── processing.py       # Core data processing logic
```

**Task:** Create empty files with module docstrings for each file listed above.

---

## Step 2: Extract Constants and Configuration (`config.py`)

### 2.1 Define constants for magic numbers

Extract all magic numbers from the code and define them as module-level constants:

```python
"""Configuration constants and dataclasses for magnetrun analysis."""

from dataclasses import dataclass, field
from typing import Dict, List
import os

# =============================================================================
# Sampling rates (Hz)
# =============================================================================
SAMPLING_RATE_ARCHIVE: float = 120.0
SAMPLING_RATE_INCIDENTS: float = 4800.0

# =============================================================================
# Analysis thresholds
# =============================================================================
LAG_THRESHOLD_RATIO: float = 0.2
DEFAULT_WINDOW_SIZE: int = 50
DEFAULT_BINS: int = 10
DEFAULT_LEVELS: int = 4

# =============================================================================
# Default paths (can be overridden via environment variables)
# =============================================================================
DEFAULT_DATA_DIR: str = os.environ.get(
    "MAGNETRUN_DATA_DIR",
    "/home/LNCMI-G/christophe.trophime/LNCMIG-Data/srv-data-install"
)
```

### 2.2 Create dataclasses for configuration

Replace the `setup()` function's tuple return with structured dataclasses:

```python
@dataclass(frozen=True)
class ChannelMapping:
    """Mapping between reference and current channels."""
    reference_gr1: str = "Courant_GR1"
    reference_gr2: str = "Courant_GR2"
    
    def get_current_channel(self, reference_key: str) -> str:
        """Get the current channel name for a reference key."""
        mapping = {
            "Référence_GR1": self.reference_gr1,
            "Référence_GR2": self.reference_gr2,
        }
        return mapping[reference_key]


@dataclass(frozen=True)
class SiteConfig:
    """Configuration specific to a measurement site."""
    name: str
    reference_gr1_current: str
    reference_gr2_current: str
    reference_gr1_flow: str
    reference_gr2_flow: str
    reference_gr1_rpm: str
    reference_gr2_rpm: str
    reference_gr1_pin: str
    reference_gr2_pin: str
    voltage_channels_gr1: List[str] = field(default_factory=list)
    voltage_channels_gr2: List[str] = field(default_factory=list)


# Pre-defined site configurations
SITE_CONFIGS: Dict[str, SiteConfig] = {
    "M9": SiteConfig(
        name="M9",
        reference_gr1_current="IH",
        reference_gr2_current="IB",
        reference_gr1_flow="FlowH",
        reference_gr2_flow="FlowB",
        reference_gr1_rpm="RpmH",
        reference_gr2_rpm="RpmB",
        reference_gr1_pin="HPH",
        reference_gr2_pin="HPB",
        voltage_channels_gr1=["UH"],
        voltage_channels_gr2=["UB", "Ucoil15", "Ucoil16"],
    ),
    "M8": SiteConfig(
        name="M8",
        reference_gr1_current="IB",
        reference_gr2_current="IH",
        # ... etc
    ),
    # ... M10
}


@dataclass
class ThresholdConfig:
    """Threshold values for signal detection."""
    thresholds: Dict[str, float] = field(default_factory=dict)
    
    def get(self, key: str, default: float = 0.1) -> float:
        """Get threshold for a given key."""
        return self.thresholds.get(key, default)
    
    @classmethod
    def default(cls) -> "ThresholdConfig":
        """Create default threshold configuration."""
        return cls(thresholds={
            "Référence_GR1": 0.5,
            "Courant_GR1": 0.5,
            "Référence_GR2": 0.5,
            "Courant_GR2": 0.5,
            "IH": 1.0,
            "IB": 1.0,
            "UH": 0.1,
            "UB": 0.1,
            "debitbrut": 25.0,
            "Pmagnet": 0.1,
            # Internal coils default to 1.0e-2
            **{f"Interne{i}": 1.0e-2 for i in range(1, 8)},
            **{f"Ucoil{i}": 1.0e-2 for i in range(1, 17)},
        })


@dataclass
class AnalysisConfig:
    """Complete analysis configuration."""
    site: SiteConfig
    channels: ChannelMapping = field(default_factory=ChannelMapping)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig.default)
    color_map: Dict[str, str] = field(
        default_factory=lambda: {"U": "red", "P": "green", "D": "blue"}
    )
    
    @classmethod
    def for_site(cls, site_name: str) -> "AnalysisConfig":
        """Create configuration for a specific site."""
        if site_name not in SITE_CONFIGS:
            raise ValueError(f"Unknown site: {site_name}. Available: {list(SITE_CONFIGS.keys())}")
        return cls(site=SITE_CONFIGS[site_name])
```

**Task:** 
1. Create `config.py` with all constants and dataclasses
2. Update `setup()` to return an `AnalysisConfig` instance or remove it entirely
3. Test that the configuration works identically to the original dictionaries

---

## Step 3: Set Up Logging Infrastructure (`cli.py`)

### 3.1 Create a logging setup function

```python
"""Command-line interface for magnetrun analysis."""

import argparse
import logging
from pathlib import Path
from typing import Optional

from .config import DEFAULT_DATA_DIR, DEFAULT_BINS, DEFAULT_WINDOW_SIZE, DEFAULT_LEVELS


def setup_logging(debug: bool = False, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Configure logging for the analysis module.
    
    Parameters
    ----------
    debug : bool
        If True, set log level to DEBUG; otherwise INFO
    log_file : Path, optional
        If provided, also log to this file
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    level = logging.DEBUG if debug else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Configure root logger
    logger = logging.getLogger("magnetrun.analysis")
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always verbose in file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
```

### 3.2 Refactor argument parsing

```python
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the analysis script.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Analyze magnetrun data from TDMS and pupitre files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "input_files",
        nargs="+",
        type=Path,
        help="Input TDMS overview files to process",
    )
    
    # Data directories
    dir_group = parser.add_argument_group("Data directories")
    dir_group.add_argument(
        "--pupitre-datadir",
        type=Path,
        default=Path(DEFAULT_DATA_DIR),
        help="Directory containing pupitre data files",
    )
    dir_group.add_argument(
        "--log-datadir",
        type=Path,
        default=Path(DEFAULT_DATA_DIR),
        help="Directory containing log files",
    )
    
    # Input files
    input_group = parser.add_argument_group("Additional input files")
    input_group.add_argument(
        "--logs",
        nargs="+",
        type=Path,
        help="Log files from ACQ_ENET",
    )
    
    # Analysis options
    analysis_group = parser.add_argument_group("Analysis options")
    analysis_group.add_argument(
        "--tkey",
        choices=["t", "timestamp"],
        default="t",
        help="Time key to use for plotting",
    )
    analysis_group.add_argument(
        "--synchronize",
        action="store_true",
        help="Synchronize pupitre/pigbrother clocks",
    )
    analysis_group.add_argument(
        "--lag",
        action="store_true",
        help="Compute lag between pupitre and pigbrother data",
    )
    analysis_group.add_argument(
        "--distance",
        action="store_true",
        help="Compute distance metrics between series",
    )
    analysis_group.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of bins for histograms",
    )
    analysis_group.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Rolling window size",
    )
    analysis_group.add_argument(
        "--levels",
        type=int,
        default=DEFAULT_LEVELS,
        help="Number of levels for analysis",
    )
    
    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files without processing data",
    )
    output_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    output_group.add_argument(
        "--save",
        action="store_true",
        help="Save graphs as PNG files",
    )
    output_group.add_argument(
        "--show",
        action="store_true",
        help="Display graphs (requires X11)",
    )
    
    return parser.parse_args()
```

**Task:**
1. Create `cli.py` with logging setup and argument parsing
2. Replace all `print()` statements with appropriate `logger.info()`, `logger.debug()`, or `logger.warning()` calls
3. Use `logger.debug()` for verbose output that should only appear with `--debug`

---

## Step 4: Extract Data Loading Functions (`loaders.py`)

### 4.1 Move and document existing loader functions

```python
"""Data loading and file discovery utilities."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from natsort import natsorted

from ..MagnetRun import MagnetRun
from ..utils.convert import convert_to_timestamp
from .config import AnalysisConfig

logger = logging.getLogger(__name__)


def extract_file_metadata(
    file: Path,
    site: str,
    insert: str,
    key: Optional[str] = None,
    dry_run: bool = False,
) -> Tuple[str, str, bool]:
    """
    Extract metadata from a data file without fully loading it.
    
    Parameters
    ----------
    file : Path
        Path to the input file (.txt or .tdms)
    site : str
        Site identifier (e.g., 'M9', 'M10')
    insert : str
        Insert identifier
    key : str, optional
        Data key to validate existence of
    dry_run : bool
        If True, only extract timestamps without loading data
        
    Returns
    -------
    tuple[str, str, bool]
        - start_timestamp: Formatted start time
        - end_timestamp: Formatted end time  
        - skip: True if file should be skipped (key not found)
        
    Raises
    ------
    RuntimeError
        If file extension is not supported
    """
    # ... implementation
```

### 4.2 Create a FileDiscovery class

```python
@dataclass
class FileSet:
    """Collection of related data files for an analysis run."""
    overview: Path
    archive: List[Path]
    pupitre: List[Path]
    default: List[Path]
    trigger: List[Path]
    spike: List[Path]
    
    def __post_init__(self):
        """Validate that required files exist."""
        if not self.overview.exists():
            raise FileNotFoundError(f"Overview file not found: {self.overview}")


class FileDiscovery:
    """Discover related data files based on an overview file."""
    
    def __init__(self, pupitre_datadir: Path, log_datadir: Path):
        self.pupitre_datadir = pupitre_datadir
        self.log_datadir = log_datadir
        self._logger = logging.getLogger(__name__)
    
    def find_related_files(
        self,
        overview_file: Path,
        site: str,
        start_time: str,
        end_time: str,
    ) -> FileSet:
        """
        Find all files related to an overview file.
        
        Parameters
        ----------
        overview_file : Path
            The overview TDMS file
        site : str
            Site identifier
        start_time : str
            Start timestamp for filtering
        end_time : str
            End timestamp for filtering
            
        Returns
        -------
        FileSet
            Collection of discovered file paths
        """
        # ... implementation
```

**Task:**
1. Create `loaders.py` with all file loading functions
2. Add proper docstrings and type hints
3. Replace tuple returns with named tuples or dataclasses where appropriate
4. Use `pathlib.Path` consistently instead of string paths

---

## Step 5: Extract Synchronization Logic (`synchronization.py`)

### 5.1 Create synchronization functions

```python
"""Time synchronization between data sources."""

import logging
from datetime import datetime, timedelta
from typing import Tuple

import pandas as pd

from ..processing.correlations import compute_lag
from ..signature import Signature

logger = logging.getLogger(__name__)


def synchronize_dataframe(
    df: pd.DataFrame,
    reference_t0: datetime,
) -> Tuple[pd.DataFrame, float]:
    """
    Synchronize a DataFrame's timestamps to a reference time.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'timestamp' column
    reference_t0 : datetime
        Reference start time
        
    Returns
    -------
    tuple[pd.DataFrame, float]
        - Synchronized DataFrame with 't' column added
        - Time offset applied (seconds)
    """
    df = df.copy()
    t_offset = (df["timestamp"].iloc[0] - reference_t0).total_seconds()
    df["t"] = (df["timestamp"] - reference_t0).dt.total_seconds()
    logger.debug("Applied time offset: %.3f seconds", t_offset)
    return df, t_offset


def compute_cross_correlation_lag(
    series1: pd.Series,
    series2: pd.Series,
    max_lag: int = 1000,
) -> timedelta:
    """
    Compute the time lag between two series using cross-correlation.
    
    Parameters
    ----------
    series1 : pd.Series
        Reference time series
    series2 : pd.Series
        Series to align
    max_lag : int
        Maximum lag to consider (samples)
        
    Returns
    -------
    timedelta
        Computed lag (positive means series2 is delayed)
    """
    # ... implementation using compute_lag
```

**Task:**
1. Create `synchronization.py` with all sync-related functions
2. Extract the signature matching logic into separate functions
3. Add unit tests for synchronization functions

---

## Step 6: Extract Metrics Functions (`metrics.py`)

### 6.1 Consolidate distance calculations

```python
"""Distance and similarity metrics for time series comparison."""

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..processing.distance import calc_euclidean, calc_mape, calc_correlation

logger = logging.getLogger(__name__)


@dataclass
class DistanceMetrics:
    """Results from distance calculations between two time series."""
    euclidean: float
    mae: float
    pearson_correlation: float
    mean_difference: float
    min_difference: float
    max_difference: float
    variance: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for tabulation."""
        return {
            "Euclidean": self.euclidean,
            "MAE": self.mae,
            "Pearson": self.pearson_correlation,
            "Mean": self.mean_difference,
            "Min": self.min_difference,
            "Max": self.max_difference,
            "Variance": self.variance,
        }


def compute_distance_metrics(
    series1: np.ndarray,
    series2: np.ndarray,
) -> DistanceMetrics:
    """
    Compute comprehensive distance metrics between two aligned series.
    
    Parameters
    ----------
    series1 : np.ndarray
        First time series
    series2 : np.ndarray
        Second time series (must be same length as series1)
        
    Returns
    -------
    DistanceMetrics
        Computed metrics
        
    Raises
    ------
    ValueError
        If series have different lengths
    """
    if len(series1) != len(series2):
        raise ValueError(
            f"Series must have same length: {len(series1)} != {len(series2)}"
        )
    
    difference = series2 - series1
    scipy_stats = stats.describe(difference)
    
    return DistanceMetrics(
        euclidean=calc_euclidean(series1, series2),
        mae=calc_mape(series1, series2),
        pearson_correlation=calc_correlation(series1, series2),
        mean_difference=scipy_stats.mean,
        min_difference=scipy_stats.minmax[0],
        max_difference=scipy_stats.minmax[1],
        variance=scipy_stats.variance,
    )


def compute_dtw_similarity(
    series1: np.ndarray,
    series2: np.ndarray,
) -> Tuple[float, list]:
    """
    Compute Dynamic Time Warping similarity between two series.
    
    Parameters
    ----------
    series1 : np.ndarray
        First time series
    series2 : np.ndarray
        Second time series
        
    Returns
    -------
    tuple[float, list]
        - Similarity score (lower is more similar)
        - Best warping path
    """
    from dtaidistance import dtw
    
    distance, paths = dtw.warping_paths(series1, series2, use_c=False)
    best_path = dtw.best_path(paths)
    similarity_score = distance / len(best_path)
    
    return similarity_score, best_path
```

**Task:**
1. Create `metrics.py` with all distance/similarity functions
2. Move DTW-related code from main()
3. Add proper error handling for edge cases

---

## Step 7: Extract Plotting Functions (`plotting.py`)

### 7.1 Refactor plot_data function

```python
"""Visualization functions for magnetrun analysis."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass
class PlotOptions:
    """Options for plot generation."""
    show: bool = False
    save: bool = False
    output_dir: Path = Path(".")
    dpi: int = 300
    figsize: tuple = (12, 5)


def plot_current_comparison(
    df_overview: pd.DataFrame,
    df_archive: pd.DataFrame,
    df_pupitre: pd.DataFrame,
    config: AnalysisConfig,
    reference_key: str,
    tkey: str = "t",
    title: str = "",
    subtitle: str = "",
    incidents: Optional[Dict[str, List[pd.DataFrame]]] = None,
    options: PlotOptions = None,
) -> Optional[plt.Figure]:
    """
    Create a comparison plot of current data from multiple sources.
    
    Parameters
    ----------
    df_overview : pd.DataFrame
        Overview data
    df_archive : pd.DataFrame
        Archive data
    df_pupitre : pd.DataFrame
        Pupitre data
    config : AnalysisConfig
        Analysis configuration
    reference_key : str
        Reference key (e.g., "Référence_GR1")
    tkey : str
        Time key for x-axis
    title : str
        Plot title
    subtitle : str
        Additional subtitle (e.g., sync info)
    incidents : dict, optional
        Dictionary of incident DataFrames to annotate
    options : PlotOptions, optional
        Plot display/save options
        
    Returns
    -------
    plt.Figure or None
        Figure object if created, None if error
    """
    options = options or PlotOptions()
    
    fig, ax = plt.subplots(figsize=options.figsize)
    
    current_channel = config.channels.get_current_channel(reference_key)
    pupitre_channel = config.site.get_current_channel(reference_key)
    
    # Plot data from each source
    df_overview.plot(x=tkey, y=reference_key, color="b", ax=ax, label=f"Overview: {reference_key}")
    df_overview.plot(x=tkey, y=current_channel, marker=".", color="r", ax=ax, label=f"Overview: {current_channel}")
    df_archive.plot(x=tkey, y=current_channel, alpha=0.5, color="r", ax=ax, label=f"Archive: {current_channel}")
    df_pupitre.plot(x=tkey, y=pupitre_channel, color="g", ax=ax, label=f"Pupitre: {pupitre_channel}")
    
    # Add incident annotations if provided
    if incidents:
        _add_incident_annotations(ax, incidents, tkey, current_channel)
    
    ax.set_title(f"{title} {subtitle}".strip())
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if options.save:
        output_path = options.output_dir / f"{title.replace('_Overview', '')}.png"
        fig.savefig(output_path, dpi=options.dpi)
        logger.info("Saved plot to %s", output_path)
    
    if options.show:
        plt.show()
    else:
        plt.close(fig)
        return None
    
    return fig


def _add_incident_annotations(
    ax: plt.Axes,
    incidents: Dict[str, List[pd.DataFrame]],
    tkey: str,
    value_column: str,
) -> None:
    """Add clickable annotations for incidents."""
    # ... implementation
```

**Task:**
1. Create `plotting.py` with refactored visualization functions
2. Extract the `on_pick` callback logic into a separate class or function
3. Create separate functions for DTW visualization plots

---

## Step 8: Create Main Processing Logic (`processing.py`)

### 8.1 Break down main() into focused functions

```python
"""Core processing logic for magnetrun analysis."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import AnalysisConfig
from .loaders import FileDiscovery, FileSet, load_data, merge_data
from .synchronization import synchronize_dataframe, compute_cross_correlation_lag
from .metrics import compute_distance_metrics, compute_dtw_similarity
from .plotting import plot_current_comparison, PlotOptions

logger = logging.getLogger(__name__)


@dataclass
class OverviewResult:
    """Results from processing a single overview file."""
    filename: str
    site: str
    df_overview: pd.DataFrame
    df_archive: pd.DataFrame
    df_pupitre: pd.DataFrame
    incidents: Dict[str, List[pd.DataFrame]]
    signatures: Dict[str, "Signature"]
    lag: Optional[float] = None
    metrics: Optional[Dict] = None


def process_log_files(
    log_files: List[Path],
    base_files: List[Path],
    reference_t0,
) -> Dict[str, Dict]:
    """
    Process log files and extract error timestamps.
    
    Parameters
    ----------
    log_files : list[Path]
        Paths to JSON log files
    base_files : list[Path]
        Base files to match against log entries
    reference_t0 : datetime
        Reference timestamp for computing relative times
        
    Returns
    -------
    dict
        Dictionary mapping filenames to their log data
    """
    import json
    
    tlogs = {}
    basenames = [f.name for f in base_files]
    
    for log_file in log_files:
        logger.debug("Processing log file: %s", log_file)
        
        with open(log_file) as f:
            logs = json.load(f)
        
        for log_key, log_data in logs.items():
            if log_key not in basenames:
                continue
                
            tlogs[log_key] = {"t0": [], "t": []}
            
            for error in log_data.get("errors", []):
                t_log_str = error["error_timestamp"]
                t_log = pd.to_datetime(t_log_str)
                
                tlogs[log_key]["t0"].append(t_log)
                tlogs[log_key]["t"].append((t_log - reference_t0).total_seconds())
                
            logger.debug("Found %d errors in %s", len(tlogs[log_key]["t"]), log_key)
    
    return tlogs


def detect_operating_mode(
    mdata,
    group: str,
    reference_keys: List[str],
) -> Dict:
    """
    Detect the operating mode based on current references.
    
    Parameters
    ----------
    mdata : MagnetData
        Loaded magnet data
    group : str
        Group name in data
    reference_keys : list[str]
        Reference key names
        
    Returns
    -------
    dict
        Mode configuration with name, intercept, slopes, breakpoint
    """
    from ..flow_params import pwlf_fit
    
    mode = {
        "name": "normal",
        "Intercept": 0,
        "Slopes": [1],
        "Breakpoint": None,
    }
    
    if len(reference_keys) != 2:
        return mode
    
    # Load reference data
    gr_data = mdata.getData([
        f"{group}/Référence_GR1",
        f"{group}/Référence_GR2",
    ]).copy()
    
    # Filter to values above threshold
    gr_filtered = gr_data[
        (gr_data["Référence_GR1"] >= 1) & 
        (gr_data["Référence_GR2"] >= 1)
    ]
    
    if gr_filtered.empty:
        logger.warning("No data above threshold for mode detection")
        return mode
    
    # Fit piecewise linear model
    # ... implementation
    
    return mode


def process_overview_file(
    overview_file: Path,
    config: AnalysisConfig,
    file_discovery: FileDiscovery,
    args,
) -> Optional[OverviewResult]:
    """
    Process a single overview file with all associated data.
    
    Parameters
    ----------
    overview_file : Path
        Path to the overview TDMS file
    config : AnalysisConfig
        Analysis configuration
    file_discovery : FileDiscovery
        File discovery helper
    args : Namespace
        Command-line arguments
        
    Returns
    -------
    OverviewResult or None
        Processing results, or None if skipped
    """
    logger.info("Processing overview file: %s", overview_file)
    
    # Extract file metadata
    # ... implementation
    
    # Discover related files
    # ... implementation
    
    # Load data
    # ... implementation
    
    # Synchronize if requested
    # ... implementation
    
    # Compute metrics if requested
    # ... implementation
    
    return OverviewResult(...)
```

**Task:**
1. Create `processing.py` with the main logic broken into functions
2. Each function should be under 50 lines
3. Use dataclasses for structured return values
4. Add comprehensive error handling

---

## Step 9: Update the Main Entry Point (`cli.py`)

### 9.1 Create a clean main() function

```python
def main() -> int:
    """
    Main entry point for the analysis script.
    
    Returns
    -------
    int
        Exit code (0 for success, non-zero for errors)
    """
    args = parse_arguments()
    logger = setup_logging(debug=args.debug)
    
    logger.info("Starting magnetrun analysis")
    logger.debug("Arguments: %s", args)
    
    try:
        # Sort input files
        input_files = natsorted(args.input_files)
        log_files = natsorted(args.logs) if args.logs else []
        
        logger.info("Processing %d input files", len(input_files))
        
        # Initialize file discovery
        file_discovery = FileDiscovery(
            pupitre_datadir=args.pupitre_datadir,
            log_datadir=args.log_datadir,
        )
        
        # Process each file
        results = []
        for input_file in input_files:
            # Extract site from filename
            site = input_file.stem.split("_")[0]
            config = AnalysisConfig.for_site(site)
            
            result = process_overview_file(
                input_file,
                config,
                file_discovery,
                args,
            )
            
            if result:
                results.append(result)
        
        logger.info("Successfully processed %d/%d files", len(results), len(input_files))
        return 0
        
    except Exception as e:
        logger.exception("Analysis failed: %s", e)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

**Task:**
1. Update `cli.py` with the clean main function
2. Add proper exit codes
3. Ensure all exceptions are logged

---

## Step 10: Add Type Stubs and Documentation

### 10.1 Create `__init__.py` with public API

```python
"""
Magnetrun Analysis Module
=========================

Tools for analyzing magnetrun data from TDMS and pupitre files.

Example usage:
    from python_magnetrun.analysis import AnalysisConfig, process_overview_file
    
    config = AnalysisConfig.for_site("M9")
    result = process_overview_file(path, config, discovery, args)
"""

from .config import (
    AnalysisConfig,
    SiteConfig,
    ChannelMapping,
    ThresholdConfig,
    SITE_CONFIGS,
)
from .loaders import FileDiscovery, FileSet
from .processing import process_overview_file, OverviewResult
from .cli import main

__all__ = [
    "AnalysisConfig",
    "SiteConfig", 
    "ChannelMapping",
    "ThresholdConfig",
    "SITE_CONFIGS",
    "FileDiscovery",
    "FileSet",
    "process_overview_file",
    "OverviewResult",
    "main",
]
```

### 10.2 Add py.typed marker

Create an empty `py.typed` file in the package directory to indicate PEP 561 compliance.

**Task:**
1. Create `__init__.py` with clean public API
2. Add `py.typed` marker
3. Run `mypy` to verify type hints

---

## Step 11: Testing Strategy

### 11.1 Create test structure

```
tests/
├── analysis/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_loaders.py
│   ├── test_synchronization.py
│   ├── test_metrics.py
│   └── conftest.py  # Shared fixtures
```

### 11.2 Example test for config

```python
"""Tests for configuration module."""

import pytest
from python_magnetrun.analysis.config import (
    AnalysisConfig,
    SiteConfig,
    SITE_CONFIGS,
)


class TestSiteConfig:
    def test_available_sites(self):
        """All expected sites should be available."""
        assert "M8" in SITE_CONFIGS
        assert "M9" in SITE_CONFIGS
        assert "M10" in SITE_CONFIGS
    
    def test_m9_configuration(self):
        """M9 should have correct channel mappings."""
        config = SITE_CONFIGS["M9"]
        assert config.reference_gr1_current == "IH"
        assert config.reference_gr2_current == "IB"


class TestAnalysisConfig:
    def test_for_site_valid(self):
        """Should create config for valid site."""
        config = AnalysisConfig.for_site("M9")
        assert config.site.name == "M9"
    
    def test_for_site_invalid(self):
        """Should raise for unknown site."""
        with pytest.raises(ValueError, match="Unknown site"):
            AnalysisConfig.for_site("INVALID")
```

**Task:**
1. Create test files for each module
2. Add fixtures for sample data
3. Achieve >80% code coverage

---

## Step 12: Final Cleanup

### 12.1 Remove deprecated code

- Delete the original `analysis-refactor.py` file
- Remove any remaining commented-out code blocks
- Remove unused imports

### 12.2 Update package metadata

Update `pyproject.toml` or `setup.py`:

```toml
[project.scripts]
magnetrun-analysis = "python_magnetrun.analysis.cli:main"
```

### 12.3 Add pre-commit hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [pandas-stubs, types-tabulate]
```

---

## Verification Checklist

Before considering the refactoring complete, verify:

- [ ] All original functionality is preserved
- [ ] No `print()` statements remain (replaced with logging)
- [ ] All functions have docstrings
- [ ] All functions have type hints
- [ ] No function exceeds 50 lines
- [ ] No file exceeds 300 lines
- [ ] `mypy` passes with no errors
- [ ] `pytest` passes with >80% coverage
- [ ] `black` and `isort` formatting applied
- [ ] Original CLI interface is unchanged (backward compatible)
