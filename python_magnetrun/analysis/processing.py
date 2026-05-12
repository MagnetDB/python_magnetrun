"""
Main processing orchestration for magnetrun analysis.

This module provides the high-level workflow for processing magnetrun
experiments, coordinating file discovery, data loading, synchronization,
analysis, and visualization.

The main data structure is the `OverviewRecord` which holds all data
and metadata for a single overview file and its associated sources.

Example usage::

    from python_magnetrun.analysis.processing import (
        OverviewRecord,
        ProcessingConfig,
        process_overview_file,
        process_experiment,
    )

    # Process a single overview file
    record = process_overview_file(
        overview_file="M9_Overview_241106-091500.tdms",
        config=ProcessingConfig(
            pupitre_datadir="/data/pupitre",
            synchronize=True,
            compute_lag=True,
        ),
    )

    # Access results
    print(f"Housing: {record.housing}, Duration: {record.duration}")
    print(f"Signatures: {list(record.signatures.keys())}")
"""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.downsampling import DownsampleConfig  # noqa: F401 — available for callers
from ..utils.timestamps import add_time_columns, parse_tdms_filename
from ..utils.timezone import utc_naive_to_local
from .config import (
    DEFAULT_DATA_DIR,
    DEFAULT_HYBRID_DATA_DIR,
    DEFAULT_PIGBROTHER_DATA_DIR,
    HOUSING_CONFIGS,
    SAMPLING_RATE_INCIDENTS,
    HousingConfig,
)
from .loaders import (
    FileDiscovery,
    FileSet,
    load_data,
    load_files_data,
)
from .synchronization import (
    compute_lag,
    get_timestamp_info,
    synchronize_data,
)

# Module logger
logger = logging.getLogger("python_magnetrun.analysis.processing")


# =============================================================================
# Configuration dataclasses
# =============================================================================
@dataclass
class ProcessingConfig:
    """
    Configuration for processing workflow.

    Attributes
    ----------
    pupitre_datadir : str
        Directory containing pupitre data files
    pigbrother_datadir : str
        Directory containing pigbrother data files
    hybrid_datadir : str
        Base directory for hybrid data (kHz/rms/trigger/vprocess); used for M8 only
    group : str
        TDMS group name for current channels
    tkey : str
        Time column to use ('t' or 'timestamp')
    synchronize : bool
        Whether to synchronize pupitre with overview
    compute_lag : bool
        Whether to compute lag correlation
    compute_distance : bool
        Whether to compute distance metrics
    compute_flow_params : bool
        Whether to compute flow parameters
    levels : int
        Number of levels for debitbrut analysis
    dry_run : bool
        If True, only discover files without loading
    debug : bool
        Enable debug output
    show : bool
        Show plots interactively
    save : bool
        Save plots to files
    backend : str
        Plotting backend name ('matplotlib', 'plotly', 'plotly-resampler', 'plotly-widget')
    downsample_percent : float
        Percentage of data points to keep when plotting (0–100). 100 disables
        downsampling (default).
    """

    pupitre_datadir: str = DEFAULT_DATA_DIR
    pigbrother_datadir: str = DEFAULT_PIGBROTHER_DATA_DIR
    hybrid_datadir: str = DEFAULT_HYBRID_DATA_DIR
    group: str = "Courants_Alimentations"
    tkey: str = "t"
    synchronize: bool = True
    compute_lag: bool = False
    compute_distance: bool = False
    compute_flow_params: bool = False
    levels: int = 3
    dry_run: bool = False
    debug: bool = False
    show: bool = False
    save: bool = False
    backend: str = "matplotlib"
    downsample_percent: float = 100.0


@dataclass
class OverviewRecord:
    """
    Complete record for an overview file and associated data.

    This is the main data structure holding all information about
    a magnetrun experiment from a single overview file.

    Attributes
    ----------
    filename : str
        Base filename (without extension)
    housing : str
        Housing identifier (M8, M9, M10)
    mode : str
        Measurement mode
    t0 : datetime
        Reference timestamp (start of overview)
    sources : FileSet
        Associated source files
    data : Dict[str, Any]
        Loaded DataFrames for each source type
    signatures : Dict[str, Any]
        Extracted signatures for each key
    sync_info : Dict[str, Any]
        Synchronization information
    flow_params : Dict[str, Any]
        Computed flow parameters
    metrics : Dict[str, Any]
        Computed distance/correlation metrics
    """

    filename: str
    housing: str
    mode: str = ""
    t0: datetime | None = None
    duration: float = 0.0

    # Source files
    sources: FileSet | None = None

    # Loaded data
    data: dict[str, Any] = field(
        default_factory=lambda: {
            "overview": pd.DataFrame(),
            "archive": pd.DataFrame(),
            "pupitre": pd.DataFrame(),
            "default": [],
            "trigger": [],
            "spike": [],
            "hybrid_kHz": pd.DataFrame(),
            "hybrid_rms": pd.DataFrame(),
            "hybrid_vprocess": pd.DataFrame(),
            "hybrid_trigger": [],
        }
    )

    # Analysis results
    signatures: dict[str, Any] = field(default_factory=dict)
    sync_info: dict[str, Any] = field(default_factory=dict)
    flow_params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    # Extra parameters from pupitre
    teb: float = 0.0
    BP: float = 0.0
    debitbrut: dict[str, Any] = field(default_factory=dict)

    def get_overview(self) -> pd.DataFrame:
        """Get overview DataFrame."""
        return self.data.get("overview", pd.DataFrame())

    def get_archive(self) -> pd.DataFrame:
        """Get archive DataFrame."""
        return self.data.get("archive", pd.DataFrame())

    def get_pupitre(self) -> pd.DataFrame:
        """Get pupitre DataFrame."""
        return self.data.get("pupitre", pd.DataFrame())

    def get_hybrid_kHz(self) -> pd.DataFrame:
        """Get hybrid kHz DataFrame."""
        return self.data.get("hybrid_kHz", pd.DataFrame())

    def get_hybrid_rms(self) -> pd.DataFrame:
        """Get hybrid RMS DataFrame."""
        return self.data.get("hybrid_rms", pd.DataFrame())

    def get_hybrid_vprocess(self) -> pd.DataFrame:
        """Get hybrid voltage-process DataFrame."""
        return self.data.get("hybrid_vprocess", pd.DataFrame())

    def get_incidents(self) -> dict[str, list[pd.DataFrame]]:
        """Get incidents as dict of DataFrames."""
        return {
            "default": self.data.get("default", []),
            "trigger": self.data.get("trigger", []),
            "spike": self.data.get("spike", []),
        }

    def get_hybrid_incidents(self) -> dict[str, list[pd.DataFrame]]:
        """Get hybrid incidents as dict of DataFrames."""
        return {
            "hybrid_trigger": self.data.get("hybrid_trigger", []),
        }

    def has_data(self, source: str) -> bool:
        """Check if data is loaded for a source."""
        data = self.data.get(source)
        logger.debug(f'has_data check for source="{source}", data type={type(data)}')
        if isinstance(data, pd.DataFrame):
            return not data.empty
        if isinstance(data, list):
            return len(data) > 0
        return False


@dataclass
class ProcessingResult:
    """
    Result of processing multiple overview files.

    Attributes
    ----------
    records : Dict[str, OverviewRecord]
        Processed records keyed by filename
    housing : str
        Housing identifier
    keys : List[str]
        Current keys processed (e.g., ['Courant_GR1', 'Courant_GR2'])
    errors : List[str]
        Any errors encountered during processing
    """

    records: dict[str, OverviewRecord] = field(default_factory=dict)
    housing: str = ""
    keys: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def get_record(self, filename: str) -> OverviewRecord | None:
        """Get record by filename."""
        return self.records.get(filename)


def add_time_column_with_offset(
    df: pd.DataFrame,
    reference_t0: datetime,
    sampling_rate: float,
    timestamp_col: str = "timestamp",
    time_col: str = "t",
) -> pd.DataFrame:
    """
    Add time column with appropriate offset for sampling rate.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timestamp column
    reference_t0 : datetime
        Reference time (t=0)
    sampling_rate : float
        Sampling rate in Hz
    timestamp_col : str
        Name of timestamp column
    time_col : str
        Name of time column to create

    Returns
    -------
    pd.DataFrame
        DataFrame with time column added
    """
    return add_time_columns(
        df, reference_t0, sampling_rate=sampling_rate,
        timestamp_col=timestamp_col, time_col=time_col,
    )


# =============================================================================
# Data loading functions
# =============================================================================
def load_overview_data(
    record: OverviewRecord,
    group: str,
    keys: list[str] | None,
) -> pd.DataFrame:
    """
    Load overview data for a record.

    Delegates to :func:`load_files_data` which handles per-file ``t`` shifting
    so the merged result has a continuous time axis.

    Parameters
    ----------
    record : OverviewRecord
        Record whose ``sources.overview`` list is loaded
    group : str
        TDMS group name
    keys : list of str, optional
        Channel names to extract

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with ``t``, ``timestamp``, and requested channels.
    """
    if record.sources is None:
        raise ValueError("No sources defined for record")

    logger.info(f"Loading overview data: {len(record.sources.overview)} files")
    return load_files_data(record.sources.overview, record.housing, group, keys)


def load_archive_data(
    record: OverviewRecord,
    housing_config: HousingConfig,
    group: str,
    keys: list[str],
) -> pd.DataFrame:
    """
    Load archive data for a record.

    Parameters
    ----------
    record : OverviewRecord
        Record to load data into
    housing_config : HousingConfig
        Housing configuration
    group : str
        TDMS group name
    keys : List[str]
        Keys to load

    Returns
    -------
    pd.DataFrame
        Loaded archive DataFrame
    """
    if record.sources is None:
        raise ValueError("No archive sources defined for record")
    if not record.sources.archive:
        logger.warning("No archive sources defined for record")
        return pd.DataFrame()

    logger.info(f"Loading archive data: {len(record.sources.archive)} files")
    return load_files_data(record.sources.archive, record.housing, group, keys)


def load_pupitre_data(
    record: OverviewRecord,
    housing_config: HousingConfig,
    group: str,
    keys: list[str],
    extra_keys: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load pupitre data for a record.

    Parameters
    ----------
    record : OverviewRecord
        Record to load data into
    housing_config : HousingConfig
        Housing configuration
    group : str
        TDMS group name
    keys : List[str]
        Current keys to load
    extra_keys : List[str], optional
        Additional keys (flow params, etc.)

    Returns
    -------
    pd.DataFrame
        Loaded pupitre DataFrame
    """
    if record.sources is None:
        raise ValueError("No sources defined for record")
    if not record.sources.pupitre:
        logger.warning("No pupitre sources defined for record")
        return pd.DataFrame()

    # Map keys to pupitre channel names
    pupitre_keys = housing_config.get_pupitre_group_keys(group)
    logger.debug(f"pupitre_keys={pupitre_keys} from group={group}")

    logger.debug(f"Mapping keys to pupitre channels for keys={keys}")
    for key in keys:
        pupitre_channel = housing_config.get_pupitre_current_channel(key)
        logger.debug(f"pupitre_channel={pupitre_channel}, key={key}")
        if pupitre_channel:
            pupitre_keys.append(pupitre_channel)

    # Add flow parameter keys
    pupitre_keys += housing_config.get_pupitre_flow_keys()
    logger.debug(f"Added flow keys, pupitre_keys={pupitre_keys}")

    # Add extra keys
    if extra_keys:
        pupitre_keys.extend(extra_keys)

    # Add standard pupitre fields
    standard_keys = ["BP", "teb", "tsb", "debitbrut", "Pmagnet", "Ptot"]
    pupitre_keys.extend([k for k in standard_keys if k not in pupitre_keys])

    logger.info(
        f"Loading pupitre data: {len(record.sources.pupitre)} files, "
        f"group={group}, keys={keys}, extra_keys={extra_keys}"
    )
    return load_files_data(record.sources.pupitre, record.housing, group, pupitre_keys)


def load_incidents_data(
    record: OverviewRecord,
    housing_config: HousingConfig,
    group: str,
    keys: list[str],
    reference_t0: datetime,
) -> dict[str, list[pd.DataFrame]]:
    """
    Load incidents data (default, trigger, spike).

    Parameters
    ----------
    record : OverviewRecord
        Record to load data into
    housing_config : HousingConfig
        Housing configuration
    group : str
        TDMS group name
    keys : List[str]
        Keys to load
    reference_t0 : datetime
        Reference timestamp for time column

    Returns
    -------
    Dict[str, List[pd.DataFrame]]
        Incidents data by type
    """
    if record.sources is None:
        return {"default": [], "trigger": [], "spike": []}

    incidents = {}

    for incident_type in ["default", "trigger", "spike"]:
        files = getattr(record.sources, incident_type, [])

        if not files:
            incidents[incident_type] = []
            continue

        logger.info(f"Loading {incident_type} data: {len(files)} files")
        # load_files_data(files, record.housing, group, keys)
        df_list = load_data(files, record.housing, "", group, keys)

        # Add time column to each incident DataFrame
        for i, df in enumerate(df_list):
            if not df.empty and "timestamp" in df.columns:
                df_list[i] = add_time_columns(
                    df, reference_t0, sampling_rate=SAMPLING_RATE_INCIDENTS
                )

        incidents[incident_type] = df_list

    return incidents


def load_hybrid_data(
    record: OverviewRecord,
    housing_config: HousingConfig,
    group: str,
    keys: list[str],
    extra_keys: list[str] | None = None,
    htype: str = "kHz",
) -> pd.DataFrame:
    """
    Load hybrid data for a record.

    Parameters
    ----------
    record : OverviewRecord
        Record to load data into
    housing_config : HousingConfig
        Housing configuration
    group : str
        TDMS group name
    keys : List[str]
        Current keys to load
    extra_keys : List[str], optional
        Additional keys (flow params, etc.)

    Returns
    -------
    pd.DataFrame
        Loaded hybrid DataFrame
    """
    if record.sources is None:
        raise ValueError("No sources defined for record")
    if htype == "kHz" and not record.sources.hybrid_kHz:
        logger.warning("No hybrid kHz sources defined for record")
        return pd.DataFrame()
    if htype == "rms" and not record.sources.hybrid_rms:
        logger.warning("No hybrid RMS sources defined for record")
        return pd.DataFrame()

    if htype == "kHz":
        logger.info(
            f"Loading hybrid kHz data: {len(record.sources.hybrid_kHz)} files, group={group}, keys={keys}, extra_keys={extra_keys}"
        )
    elif htype == "rms":
        logger.info(
            f"Loading hybrid RMS data: {len(record.sources.hybrid_rms)} files, group={group}, keys={keys}, extra_keys={extra_keys}"
        )

    # Map keys to hybrid channel names
    hybrid_keys = housing_config.get_hybrid_group_keys(group)
    logger.debug(f"hybrid_keys={hybrid_keys} from group={group}")

    # Path structure: <base_dir>/kHz/<date_str>/<fepc_system>/<HH>HOST_*.bin
    if htype == "kHz":
        first_file = Path(record.sources.hybrid_kHz[0])
    elif htype == "rms":
        first_file = Path(record.sources.hybrid_rms[0])
    base_dir = str(first_file.parents[3])
    date_str = first_file.parents[1].name  # "YYYY-MM-DD"

    hours_set: set[int] = set()
    if htype == "kHz":
        for f in record.sources.hybrid_kHz:
            with contextlib.suppress(ValueError):
                hours_set.add(int(Path(f).name[:2]))
    elif htype == "rms":
        for f in record.sources.hybrid_rms:
            with contextlib.suppress(ValueError):
                hours_set.add(int(Path(f).name[:2]))
    hours_list: list[int] | None = sorted(hours_set) or None

    try:
        from ..hybrid.hybrid_run import HybridRun

        hrun = HybridRun.fromdir(
            base_dir=base_dir, date_str=date_str, housing=record.housing
        )

        for key in hybrid_keys:
            data, time = hrun.getData(key, hours=hours_list)
            logger.debug(
                f"Loaded hybrid data for key={key}, data type={type(data)}, time shape={time.shape}"
            )

    except (OSError, ValueError, ImportError) as e:
        logger.error(
            f"load_hybrid_data: failed to create HybridRun for {date_str}: {e}"
        )
        return pd.DataFrame()
    pass


def load_hybrid_incidents_data(
    record: OverviewRecord,
    housing_config: HousingConfig,
    group: str,
    keys: list[str],
    reference_t0: datetime,
) -> dict[str, list[pd.DataFrame]]:
    """
    Load incidents data (trigger).

    Parameters
    ----------
    record : OverviewRecord
        Record to load data into
    housing_config : HousingConfig
        Housing configuration
    group : str
        TDMS group name
    keys : List[str]
        Keys to load
    reference_t0 : datetime
        Reference timestamp for time column

    Returns
    -------
    Dict[str, List[pd.DataFrame]]
        Incidents data by type
    """
    pass


# =============================================================================
# Main processing functions
# =============================================================================
def _check_overview_end_state(
    df_overview: pd.DataFrame,
    record: OverviewRecord,
    time_zone: str,
) -> None:
    """Warn when the magnet appears still running at the end of the overview.

    Logs a warning for each ``Courant*`` column whose last value exceeds 10 A
    and logs a guess for the next overview filename.
    """
    last_values = {
        col: df_overview[col].iloc[-1]
        for col in df_overview.columns
        if col.startswith("Courant")
    }
    for key, value in last_values.items():
        if abs(value) <= 10.0:
            continue
        last_overview = record.sources.overview[-1]
        logger.warning(
            f"{last_overview}: {key} last value is above 10.0 A: {value}"
        )
        if "timestamp" in df_overview.columns:
            last_ts = df_overview["timestamp"].iloc[-1]
            last_ts_local = utc_naive_to_local(last_ts, time_zone)
        elif "t" in df_overview.columns:
            start_dt = parse_tdms_filename(last_overview)
            if start_dt is None:
                logger.warning(
                    f"Cannot derive end time from {last_overview}, skipping"
                )
                continue
            last_ts_local = start_dt + timedelta(
                seconds=float(df_overview["t"].iloc[-1])
            )
        else:
            logger.warning(
                f"No timestamp or t column in df_overview for {last_overview}, skipping"
            )
            continue
        _housing = os.path.basename(last_overview).split("_")[0]
        _dirname = os.path.dirname(last_overview)
        new_ts_str = last_ts_local.strftime("%y%m%d-%H%M")
        new_overview = os.path.join(_dirname, f"{_housing}_Overview_{new_ts_str}.tdms")
        logger.info(f"Guessed next overview file: {new_overview}")
        break


def _load_all_sources(
    record: OverviewRecord,
    housing_config: HousingConfig,
    config: ProcessingConfig,
    keys: list[str],
) -> None:
    """Load archive, pupitre, hybrid and incident data into *record.data*.

    Also applies pupitre synchronisation when ``config.synchronize`` is set.
    Mutates *record* in place.
    """
    # Archive
    if record.sources.archive:
        df_archive = load_archive_data(record, housing_config, config.group, keys)
        if not df_archive.empty:
            record.data["archive"] = df_archive
        logger.info(
            f"{record.filename}: loaded archive data columns={df_archive.columns.tolist()}"
        )

    # Pupitre
    if record.sources.pupitre:
        df_pupitre = load_pupitre_data(record, housing_config, config.group, keys)
        if not df_pupitre.empty:
            if "teb" in df_pupitre.columns:
                record.teb = float(df_pupitre["teb"].mean())
            if "BP" in df_pupitre.columns:
                record.BP = float(df_pupitre["BP"].mean())
            record.data["pupitre"] = df_pupitre
        logger.info(
            f"{record.filename}: loaded pupitre data columns={df_pupitre.columns.tolist()}"
        )

    # Hybrid kHz
    if record.sources.hybrid_kHz:
        logger.info(f"hybrid_kHz={record.sources.hybrid_kHz}")
        df_hybrid_kHz = load_hybrid_data(record, housing_config, config.group, keys)
        if not df_hybrid_kHz.empty:
            record.data["hybrid_kHz"] = df_hybrid_kHz
        logger.info(
            f"{record.filename}: loaded hybrid kHz columns={df_hybrid_kHz.columns.tolist()}"
        )

    # Hybrid RMS
    if record.sources.hybrid_rms:
        logger.info(f"hybrid_rms={record.sources.hybrid_rms}")
        df_hybrid_rms = load_hybrid_data(
            record, housing_config, config.group, keys, htype="rms"
        )
        if not df_hybrid_rms.empty:
            record.data["hybrid_rms"] = df_hybrid_rms
        logger.info(
            f"{record.filename}: loaded hybrid rms columns={df_hybrid_rms.columns.tolist()}"
        )

    # Hybrid trigger
    if record.sources.hybrid_trigger:
        logger.info(f"hybrid_trigger={record.sources.hybrid_trigger}")
        reference_t0 = (
            record.data["archive"]["timestamp"].iloc[0]
            if record.has_data("archive")
            else record.t0
        )
        load_hybrid_incidents_data(
            record, housing_config, config.group, keys, reference_t0
        )
        logger.debug(f"{record.filename}: loaded hybrid incidents reference_t0={reference_t0}")

    # Synchronise pupitre
    if config.synchronize and record.has_data("pupitre"):
        _synchronize_pupitre(record, config)
        logger.info(f"{record.filename}: synchronization done")

    # Incidents (default / trigger / spike)
    if any([record.sources.default, record.sources.trigger, record.sources.spike]):
        reference_t0 = (
            record.data["archive"]["timestamp"].iloc[0]
            if record.has_data("archive")
            else record.t0
        )
        incidents = load_incidents_data(
            record, housing_config, config.group, keys, reference_t0
        )
        logger.debug(f"{record.filename}: loaded incidents reference_t0={reference_t0}")
        record.data.update(incidents)


def _extract_analysis(
    record: OverviewRecord,
    housing_config: HousingConfig,
    keys: list[str],
    config: ProcessingConfig,
) -> None:
    """Run post-load analysis: signatures and optional lag correlation."""
    _extract_signatures(record, housing_config, keys, config)
    logger.info(f"{record.filename}: extracted signatures for keys={keys}")

    if config.compute_lag and record.has_data("pupitre"):
        _compute_lag_correlation(record, housing_config, keys, config)
        logger.info(f"{record.filename}: computed lag correlation")


def process_overview_file(
    overview_file: str,
    config: ProcessingConfig,
    housing_config: HousingConfig | None = None,
    dry_run: bool = False,
    time_zone: str = "Europe/Paris",
) -> OverviewRecord:
    """Process a single overview file with all associated data.

    This is the main entry point for processing a magnetrun experiment.
    It discovers related files, loads all data sources, synchronizes
    timestamps, extracts signatures, and optionally computes metrics.

    Parameters
    ----------
    overview_file : str
        Path to overview TDMS file
    config : ProcessingConfig
        Processing configuration
    housing_config : HousingConfig, optional
        Housing-specific configuration (auto-detected if not provided)
    time_zone : str, optional
        IANA timezone for filename timestamps (default ``"Europe/Paris"``)

    Returns
    -------
    OverviewRecord
        Complete record with all loaded and processed data
    """
    basename = os.path.basename(overview_file)
    filename = basename.replace(".tdms", "")
    parts = filename.split("_")
    housing = parts[0]
    mode = parts[1] if len(parts) > 1 else "Overview"
    logger.info(f"Processing {filename} (housing={housing}, mode={mode}), config={config}")

    if housing_config is None:
        if housing not in HOUSING_CONFIGS:
            raise ValueError(
                f"Unknown housing: {housing}. Available: {list(HOUSING_CONFIGS.keys())}"
            )
        housing_config = HOUSING_CONFIGS[housing]

    record = OverviewRecord(filename=filename, housing=housing, mode=mode)

    discovery = FileDiscovery(
        pigbrother_datadir=config.pigbrother_datadir,
        pigbrother_runlog_dir=os.path.dirname(overview_file) or None,
        hybrid_datadir=config.hybrid_datadir if housing == "M8" else None,
    )
    record.sources = discovery.discover(overview_file, housing=housing, dry_run=dry_run)
    logger.info(
        f"Discovered: archive={len(record.sources.archive)}, "
        f"pupitre={len(record.sources.pupitre)}, "
        f"incidents={len(record.sources.default) + len(record.sources.trigger) + len(record.sources.spike)}, "
        f"hybrid_kHz={len(record.sources.hybrid_kHz)}, "
        f"hybrid_rms={len(record.sources.hybrid_rms)}, "
        f"hybrid_trigger={len(record.sources.hybrid_trigger)}"
    )

    if config.dry_run:
        logger.info("Dry run — skipping data loading")
        return record

    # Load overview and set record metadata
    df_overview = load_overview_data(record, config.group, keys=None)
    if df_overview.empty:
        raise ValueError(f"Failed to load overview data from {overview_file}")
    logger.info(f"{overview_file}: overview loaded")

    _check_overview_end_state(df_overview, record, time_zone)

    keys = [
        key.replace(config.group, "")
        for key in df_overview.columns.tolist()
        if key.startswith(config.group)
    ]
    logger.info(f"{overview_file}: group={config.group}, keys={keys}")

    record.t0 = df_overview["timestamp"].iloc[0]
    record.data["overview"] = df_overview
    record.duration = get_timestamp_info(df_overview)["duration"]
    logger.info(f"{overview_file}: t0={record.t0}, duration={record.duration}s")

    _load_all_sources(record, housing_config, config, keys)
    _extract_analysis(record, housing_config, keys, config)

    logger.info(f"Processing complete for {filename}")
    return record


def process_experiment(
    overview_files: list[str],
    config: ProcessingConfig,
    dry_run: bool = False,
) -> ProcessingResult:
    """
    Process multiple overview files for an experiment.

    Parameters
    ----------
    overview_files : List[str]
        List of overview file paths
    config : ProcessingConfig
        Processing configuration

    Returns
    -------
    ProcessingResult
        Results for all processed files
    """
    from natsort import natsorted

    result = ProcessingResult()

    # Sort files naturally
    sorted_files = natsorted(overview_files)

    # Determine housing from first file
    if sorted_files:
        first_basename = os.path.basename(sorted_files[0])
        result.housing = first_basename.split("_")[0]

    for overview_file in sorted_files:
        try:
            record = process_overview_file(overview_file, config, dry_run=dry_run)
            result.records[record.filename] = record

            # Track keys from first successful record
            if not result.keys and record.signatures:
                result.keys = list(record.signatures.keys())

        except (OSError, ValueError, RuntimeError) as e:
            error_msg = f"Error processing {overview_file}: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

            if config.debug:
                import traceback

                traceback.print_exc()

    logger.info(
        f"Processed {len(result.records)}/{len(sorted_files)} files successfully"
    )

    return result


# =============================================================================
# Helper functions
# =============================================================================


def _synchronize_pupitre(record: OverviewRecord, config: ProcessingConfig) -> None:
    """Synchronize pupitre data with overview reference time."""
    df_pupitre = record.data["pupitre"]
    ot0 = record.t0

    logger.info("Synchronizing pupitre data with overview")

    # Apply synchronization
    timeshift, df_synced = synchronize_data(df_pupitre, ot0)

    record.data["pupitre"] = df_synced
    record.sync_info["timeshift"] = timeshift
    record.sync_info["timeshift_seconds"] = timeshift.total_seconds()

    logger.info(f"Applied timeshift: {timeshift.total_seconds():.3f} s")


def _extract_signatures(
    record: OverviewRecord,
    housing_config: HousingConfig,
    keys: list[str],
    config: ProcessingConfig,
) -> None:
    """Extract signatures from overview data."""
    # This would integrate with the Signature class
    # For now, store placeholder info

    logger.info(f"Extracting signatures for keys={keys} from overview data")
    df_overview = record.data["overview"]

    for key in keys:
        if key in df_overview.columns:
            record.signatures[key] = {
                "key": key,
                "min": float(df_overview[key].min()),
                "max": float(df_overview[key].max()),
                "mean": float(df_overview[key].mean()),
            }

            logger.debug(
                f"Signature for {key}: min={record.signatures[key]['min']:.2f}, max={record.signatures[key]['max']:.2f}"
            )


def _compute_lag_correlation(
    record: OverviewRecord,
    housing_config: HousingConfig,
    keys: list[str],
    config: ProcessingConfig,
) -> None:
    """Compute lag correlation between overview and pupitre."""
    df_overview = record.data["overview"]
    df_pupitre = record.data["pupitre"]

    for key in keys:
        overview_key = key
        pupitre_key = housing_config.get_pupitre_current_channel(key)

        if not pupitre_key:
            continue

        if overview_key not in df_overview.columns:
            continue
        if pupitre_key not in df_pupitre.columns:
            continue

        logger.info(f"Computing lag for {overview_key} vs {pupitre_key}")

        try:
            # Prepare data for lag computation
            df1_data = {
                "df": df_overview[["timestamp", overview_key]].copy(),
                "field": overview_key,
                "range": (0, None),
            }
            df2_data = {
                "df": df_pupitre[["timestamp", pupitre_key]].copy(),
                "field": pupitre_key,
                "range": (0, None),
            }

            lag = compute_lag(
                "timestamp",
                df1_data,
                df2_data,
                show=config.show,
                save=config.save,
                debug=config.debug,
            )

            record.sync_info[f"lag_{key}"] = lag
            record.sync_info[f"lag_{key}_seconds"] = lag.total_seconds()

            logger.info(f"Lag for {key}: {lag.total_seconds():.3f} s")

        except (ValueError, RuntimeError) as e:
            logger.warning(f"Failed to compute lag for {key}: {e}")


# =============================================================================
# Convenience functions
# =============================================================================
def create_overview_dict(result: ProcessingResult) -> dict[str, dict[str, Any]]:
    """
    Convert ProcessingResult to legacy overview_dict format.

    This provides backward compatibility with the original
    analysis-refactor.py structure.

    Parameters
    ----------
    result : ProcessingResult
        Processing result

    Returns
    -------
    dict
        Legacy overview_dict format
    """
    overview_dict = {}

    for filename, record in result.records.items():
        overview_dict[filename] = {
            "mode": record.mode,
            "signature": record.signatures,
            "sources": {
                "overview": record.sources.overview if record.sources else [],
                "archive": record.sources.archive if record.sources else [],
                "pupitre": record.sources.pupitre if record.sources else [],
                "default": record.sources.default if record.sources else [],
                "trigger": record.sources.trigger if record.sources else [],
                "spike": record.sources.spike if record.sources else [],
            },
            "data": record.data,
            "t0": record.t0,
            "BP": record.BP,
            "teb": record.teb,
            "debitbrut": record.debitbrut,
            "flow_params": record.flow_params,
        }

    return overview_dict


def summarize_record(record: OverviewRecord) -> dict[str, Any]:
    """
    Create a summary of an OverviewRecord.

    Parameters
    ----------
    record : OverviewRecord
        Record to summarize

    Returns
    -------
    dict
        Summary information
    """
    return {
        "filename": record.filename,
        "housing": record.housing,
        "mode": record.mode,
        "t0": str(record.t0) if record.t0 else None,
        "duration": record.duration,
        "has_overview": record.has_data("overview"),
        "has_archive": record.has_data("archive"),
        "has_pupitre": record.has_data("pupitre"),
        "has_hybrid_kHz": record.has_data("hybrid_kHz"),
        "has_hybrid_rms": record.has_data("hybrid_rms"),
        "has_hybrid_vprocess": record.has_data("hybrid_vprocess"),
        "n_hybrid_incidents": len(record.data.get("hybrid_trigger", [])),
        "n_default": len(record.data.get("default", [])),
        "n_trigger": len(record.data.get("trigger", [])),
        "n_spike": len(record.data.get("spike", [])),
        "signatures": list(record.signatures.keys()),
        "sync_info": record.sync_info,
        "teb": record.teb,
        "BP": record.BP,
    }


def print_record_summary(record: OverviewRecord) -> None:
    """Print a formatted summary of a record."""
    summary = summarize_record(record)

    print("=" * 60)
    print(f"Overview Record: {summary['filename']}")
    print("=" * 60)
    print(f"Housing: {summary['housing']}")
    print(f"Mode: {summary['mode']}")
    print(f"Start time: {summary['t0']}")
    print(f"Duration: {summary['duration']:.1f} s")
    print("Data sources:")
    print(f"  - Overview: {'yes' if summary['has_overview'] else 'no'}")
    print(f"  - Archive: {'yes' if summary['has_archive'] else 'no'}")
    print(f"  - Pupitre: {'yes' if summary['has_pupitre'] else 'no'}")

    if record.housing == "M8":
        print(f"  - Hybrid kHz: {'yes' if summary['has_hybrid_kHz'] else 'no'}")
        print(f"  - Hybrid RMS: {'yes' if summary['has_hybrid_rms'] else 'no'}")
        print(
            f"  - Hybrid Vprocess: {'yes' if summary['has_hybrid_vprocess'] else 'no'}"
        )

    print(
        f"  - Incidents: {summary['n_default']} default, {summary['n_trigger']} trigger, {summary['n_spike']} spike"
    )
    if record.housing == "M8":
        print(f"  - Hybrid Incidents: {summary['n_hybrid_incidents']}")

    print(f"Signatures: {', '.join(summary['signatures']) or 'None'}")

    if summary["sync_info"]:
        print("Sync info:")
        for key, value in summary["sync_info"].items():
            print(f"  - {key}: {value}")

    print(f"Pupitre params: teb={summary['teb']:.2f}, BP={summary['BP']:.2f}")
    print("=" * 60, flush=True)
