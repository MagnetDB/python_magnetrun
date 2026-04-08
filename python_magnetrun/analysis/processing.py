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
    print(f"Site: {record.site}, Duration: {record.duration}")
    print(f"Signatures: {list(record.signatures.keys())}")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from .config import (
    DEFAULT_DATA_DIR,
    DEFAULT_PIGBROTHER_DATA_DIR,
    HOUSING_CONFIGS,
    SAMPLING_RATE_ARCHIVE,
    SAMPLING_RATE_INCIDENTS,
    SAMPLING_RATE_OVERVIEW,
    HousingConfig,
)
from .loaders import (
    FileDiscovery,
    FileSet,
    load_data,
    merge_data,
)
from .synchronization import (
    compute_lag,
    get_timestamp_info,
    synchronize_data,
)

# Module logger
logger = logging.getLogger("magnetrun.analysis.processing")


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
    downsample_percent : float
        Downsampling percentage for plots
    """

    pupitre_datadir: str = DEFAULT_DATA_DIR
    pigbrother_datadir: str = DEFAULT_PIGBROTHER_DATA_DIR
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
    site : str
        Site identifier (M8, M9, M10)
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
    site: str
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

    def get_incidents(self) -> dict[str, list[pd.DataFrame]]:
        """Get incidents as dict of DataFrames."""
        return {
            "default": self.data.get("default", []),
            "trigger": self.data.get("trigger", []),
            "spike": self.data.get("spike", []),
        }

    def has_data(self, source: str) -> bool:
        """Check if data is loaded for a source."""
        data = self.data.get(source)
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
    site : str
        Site identifier
    keys : List[str]
        Current keys processed (e.g., ['Courant_GR1', 'Courant_GR2'])
    errors : List[str]
        Any errors encountered during processing
    """

    records: dict[str, OverviewRecord] = field(default_factory=dict)
    site: str = ""
    keys: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def get_record(self, filename: str) -> OverviewRecord | None:
        """Get record by filename."""
        return self.records.get(filename)


# =============================================================================
# Time offset utilities
# =============================================================================
def compute_time_offset(sampling_rate: float) -> float:
    """
    Compute time offset for a given sampling rate.

    The offset is half the sampling period, used to center
    timestamps within their sampling window.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate in Hz

    Returns
    -------
    float
        Time offset in seconds
    """
    return (1.0 / sampling_rate) / 2.0


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
    t_offset = compute_time_offset(sampling_rate)

    df = df.copy()
    df[time_col] = df.apply(
        lambda row: (row[timestamp_col] - reference_t0).total_seconds() + t_offset,
        axis=1,
    )

    return df


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

    Parameters
    ----------
    record : OverviewRecord
        Record to load data into
    site_config : HousingConfig
        Site configuration
    group : str
        TDMS group name
    keys : List[str]
        Keys to load

    Returns
    -------
    pd.DataFrame
        Loaded overview DataFrame
    """
    if record.sources is None:
        raise ValueError("No sources defined for record")

    logger.info(f"Loading overview data: {len(record.sources.overview)} files")

    df_list = load_data(
        record.sources.overview,
        record.site,
        "",  # insert placeholder
        group,
        keys,
    )

    if not df_list:
        logger.warning("No overview data loaded")
        return pd.DataFrame()

    # Use first file (overview is typically single file)
    df = df_list[0] if len(df_list) == 1 else merge_data(df_list)

    return df


def load_archive_data(
    record: OverviewRecord,
    site_config: HousingConfig,
    group: str,
    keys: list[str],
) -> pd.DataFrame:
    """
    Load archive data for a record.

    Parameters
    ----------
    record : OverviewRecord
        Record to load data into
    site_config : HousingConfig
        Site configuration
    group : str
        TDMS group name
    keys : List[str]
        Keys to load

    Returns
    -------
    pd.DataFrame
        Loaded archive DataFrame
    """
    if record.sources is None or not record.sources.archive:
        raise ValueError("No archive sources defined for record")

    logger.info(f"Loading archive data: {len(record.sources.archive)} files")
    print("record.sources.archive:", record.sources.archive)
    df_list = load_data(
        record.sources.archive,
        record.site,
        "",
        group,
        keys,
    )
    print(f"Loaded {len(df_list)} archive DataFrames")
    for _df in df_list:
        print(f"Archive DataFrame columns: {_df.columns.tolist()}")

    if not df_list:
        logger.warning("No archive data loaded")
        return pd.DataFrame()

    return merge_data(df_list)


def load_pupitre_data(
    record: OverviewRecord,
    site_config: HousingConfig,
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
    site_config : HousingConfig
        Site configuration
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
    if record.sources is None or not record.sources.pupitre:
        logger.warning("No pupitre sources defined for record")
        return pd.DataFrame()

    logger.info(
        f"Loading pupitre data: {len(record.sources.pupitre)} files, group={group}, keys={keys}, extra_keys={extra_keys}"
    )
    # Map keys to pupitre channel names using helper
    pupitre_keys = _get_pupitre_group(site_config, group) or []
    print(f"pupitre_keys={pupitre_keys} from group={group}", flush=True)

    print(f"Mapping keys to pupitre channels for keys={keys}", flush=True)
    for key in keys:
        pupitre_channel = _get_pupitre_channel(site_config, key)
        print(f"pupitre_channel={pupitre_channel}, key={key}", flush=True)
        if pupitre_channel:
            pupitre_keys.append(pupitre_channel)

    # Add flow parameter keys
    pupitre_keys += _get_pupitre_flow(site_config) or []
    print(f"Added flow keys, pupitre_keys={pupitre_keys}", flush=True)

    # Add extra keys
    if extra_keys:
        pupitre_keys.extend(extra_keys)

    # Add standard pupitre fields
    standard_keys = ["BP", "teb", "tsb", "debitbrut", "Pmagnet", "Ptot"]
    pupitre_keys.extend([k for k in standard_keys if k not in pupitre_keys])

    logger.info(f"Loading pupitre data: {len(record.sources.pupitre)} files")

    df_list = load_data(
        record.sources.pupitre,
        record.site,
        "",
        group,
        pupitre_keys,
    )

    if not df_list:
        logger.warning("No pupitre data loaded")
        return pd.DataFrame()

    return merge_data(df_list)


def load_incidents_data(
    record: OverviewRecord,
    site_config: HousingConfig,
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
    site_config : HousingConfig
        Site configuration
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
    t_offset = compute_time_offset(SAMPLING_RATE_INCIDENTS)

    for incident_type in ["default", "trigger", "spike"]:
        files = getattr(record.sources, incident_type, [])

        if not files:
            incidents[incident_type] = []
            continue

        logger.info(f"Loading {incident_type} data: {len(files)} files")

        df_list = load_data(files, record.site, "", group, keys)

        # Add time column to each incident DataFrame
        for df in df_list:
            if not df.empty and "timestamp" in df.columns:
                df["t"] = df.apply(
                    lambda row: (row["timestamp"] - reference_t0).total_seconds()
                    + t_offset,
                    axis=1,
                )

        incidents[incident_type] = df_list

    return incidents


# =============================================================================
# Main processing functions
# =============================================================================
def process_overview_file(
    overview_file: str,
    config: ProcessingConfig,
    site_config: HousingConfig | None = None,
) -> OverviewRecord:
    """
    Process a single overview file with all associated data.

    This is the main entry point for processing a magnetrun experiment.
    It discovers related files, loads all data sources, synchronizes
    timestamps, extracts signatures, and optionally computes metrics.

    Parameters
    ----------
    overview_file : str
        Path to overview TDMS file
    config : ProcessingConfig
        Processing configuration
    site_config : HousingConfig, optional
        Site-specific configuration (auto-detected if not provided)

    Returns
    -------
    OverviewRecord
        Complete record with all loaded and processed data

    Examples
    --------
    >>> record = process_overview_file(
    ...     "M9_Overview_241106-091500.tdms",
    ...     ProcessingConfig(synchronize=True),
    ... )
    >>> print(f"Duration: {record.duration} s")
    """
    # Extract site and filename info
    dirname = os.path.dirname(overview_file)
    basename = os.path.basename(overview_file)
    filename = basename.replace(".tdms", "")
    parts = filename.split("_")
    site = parts[0]
    mode = parts[1] if len(parts) > 1 else "Overview"

    logger.info(
        f"Processing overview file: {filename} (dirname={dirname}, site={site}, mode={mode})"
    )

    # Get site configuration
    if site_config is None:
        if site not in HOUSING_CONFIGS:
            raise ValueError(
                f"Unknown site: {site}. Available: {list(HOUSING_CONFIGS.keys())}"
            )
        site_config = HOUSING_CONFIGS[site]

    # Create record
    record = OverviewRecord(
        filename=filename,
        site=site,
        mode=mode,
    )

    # Discover associated files
    discovery = FileDiscovery(
        pigbrother_datadir=config.pigbrother_datadir,
        log_datadir=os.path.dirname(overview_file) or None,
    )

    # Use the new `discover` API which returns a FileSet
    # tdms is loaded at this point
    record.sources = discovery.discover(overview_file, site=site)

    logger.info(
        f"Discovered files - archive: {len(record.sources.archive)}, "
        f"pupitre: {len(record.sources.pupitre)}, "
        f"incidents: {len(record.sources.default) + len(record.sources.trigger) + len(record.sources.spike)}"
    )

    if config.dry_run:
        logger.info("Dry run - skipping data loading")
        return record

    # Load overview data
    df_overview = load_overview_data(record, config.group, keys=None)
    if df_overview.empty:
        raise ValueError(f"Failed to load overview data from {overview_file}")
    logger.info(f"{overview_file}: loaded done")

    keys = [
        key.replace(config.group, "")
        for key in df_overview.columns.tolist()
        if key.startswith(config.group)
    ]
    logger.info(
        f"{overview_file}: group={config.group}, keys={keys}, df_overview.columns={df_overview.columns.tolist()}"
    )

    # Set reference timestamp
    record.t0 = df_overview["timestamp"].iloc[0]
    record.data["overview"] = df_overview

    # Add time column to overview
    t_offset = compute_time_offset(SAMPLING_RATE_OVERVIEW)
    df_overview["t"] = df_overview.apply(
        lambda row: (row["timestamp"] - record.t0).total_seconds() + t_offset,
        axis=1,
    )

    # Calculate duration
    timestamp_info = get_timestamp_info(df_overview)
    record.duration = timestamp_info["duration"]

    # Load archive data
    if record.sources.archive:
        df_archive = load_archive_data(record, site_config, config.group, keys)
        if not df_archive.empty:
            at0 = df_archive["timestamp"].iloc[0]
            t_offset = compute_time_offset(SAMPLING_RATE_ARCHIVE)
            df_archive["t"] = df_archive.apply(
                lambda row: (row["timestamp"] - at0).total_seconds() + t_offset,
                axis=1,
            )
            record.data["archive"] = df_archive

    # Load pupitre data
    if record.sources.pupitre:
        print(f"site_config: {site_config}")
        print(f"config.group: {config.group}")
        print(f"keys: {keys}")
        df_pupitre = load_pupitre_data(record, site_config, config.group, keys)
        if not df_pupitre.empty:
            pt0 = df_pupitre["timestamp"].iloc[0]
            df_pupitre["t"] = df_pupitre.apply(
                lambda row: (row["timestamp"] - pt0).total_seconds(),
                axis=1,
            )

            # Extract pupitre parameters
            if "teb" in df_pupitre.columns:
                record.teb = float(df_pupitre["teb"].mean())
            if "BP" in df_pupitre.columns:
                record.BP = float(df_pupitre["BP"].mean())

            record.data["pupitre"] = df_pupitre

    # Synchronize pupitre with overview if requested
    if config.synchronize and record.has_data("pupitre"):
        _synchronize_pupitre(record, config)

    # Load incidents data
    if any([record.sources.default, record.sources.trigger, record.sources.spike]):
        reference_t0 = (
            record.data["archive"]["timestamp"].iloc[0]
            if record.has_data("archive")
            else record.t0
        )
        incidents = load_incidents_data(
            record, site_config, config.group, keys, reference_t0
        )
        record.data.update(incidents)

    # Extract signatures for each key
    _extract_signatures(record, site_config, keys, config)

    # Compute lag if requested
    if config.compute_lag and record.has_data("pupitre"):
        _compute_lag_correlation(record, site_config, keys, config)

    logger.info(f"Processing complete for {filename}")

    return record


def process_experiment(
    overview_files: list[str],
    config: ProcessingConfig,
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

    # Determine site from first file
    if sorted_files:
        first_basename = os.path.basename(sorted_files[0])
        result.site = first_basename.split("_")[0]

    for overview_file in sorted_files:
        try:
            record = process_overview_file(overview_file, config)
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


def _get_pupitre_channel(site_config: HousingConfig, key: str) -> str | None:
    """Get pupitre channel name for a key."""
    if key == "Courant_GR1":
        return site_config.reference_gr1_current
    elif key == "Courant_GR2":
        return site_config.reference_gr2_current
    return None


def _get_pupitre_group(site_config: HousingConfig, group: str) -> list[str] | None:
    """Get pupitre list of keys for a group."""
    keys = []
    if group == "Courants_Alimentations":
        if site_config.reference_gr1_current:
            keys.append(site_config.reference_gr1_current)
        if site_config.reference_gr2_current:
            keys.append(site_config.reference_gr2_current)
    elif group == "Tensions_Aimants":
        if site_config.voltage_channels_gr1 and site_config.voltage_channels_gr2:
            keys = site_config.voltage_channels_gr1 + site_config.voltage_channels_gr2

    return keys if keys else None


def _get_pupitre_flow(site_config: HousingConfig) -> list[str] | None:
    """Get pupitre keys for flow."""
    keys = []
    if site_config.reference_gr1_flow:
        keys.append(site_config.reference_gr1_flow)
    if site_config.reference_gr2_flow:
        keys.append(site_config.reference_gr2_flow)
    if site_config.reference_gr1_rpm:
        keys.append(site_config.reference_gr1_rpm)
    if site_config.reference_gr2_rpm:
        keys.append(site_config.reference_gr2_rpm)
    if site_config.reference_gr1_pin:
        keys.append(site_config.reference_gr1_pin)
    if site_config.reference_gr2_pin:
        keys.append(site_config.reference_gr2_pin)
    return keys if keys else None


def _get_archive_channel(key: str) -> str:
    """Get archive channel name for a key (typically same as key)."""
    return key


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
    site_config: HousingConfig,
    keys: list[str],
    config: ProcessingConfig,
) -> None:
    """Extract signatures from overview data."""
    # This would integrate with the Signature class
    # For now, store placeholder info
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
    site_config: HousingConfig,
    keys: list[str],
    config: ProcessingConfig,
) -> None:
    """Compute lag correlation between overview and pupitre."""
    df_overview = record.data["overview"]
    df_pupitre = record.data["pupitre"]

    for key in keys:
        # Get channel mappings using helper functions
        overview_key = _get_archive_channel(key)
        pupitre_key = _get_pupitre_channel(site_config, key)

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
        "site": record.site,
        "mode": record.mode,
        "t0": str(record.t0) if record.t0 else None,
        "duration": record.duration,
        "has_overview": record.has_data("overview"),
        "has_archive": record.has_data("archive"),
        "has_pupitre": record.has_data("pupitre"),
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

    logger.info("=" * 60)
    logger.info(f"Overview Record: {summary['filename']}")
    logger.info("=" * 60)
    logger.info(f"Site: {summary['site']}, Mode: {summary['mode']}")
    logger.info(f"Start time: {summary['t0']}")
    logger.info(f"Duration: {summary['duration']:.1f} s")
    logger.info("Data sources:")
    logger.info(f"  - Overview: {'yes' if summary['has_overview'] else 'no'}")
    logger.info(f"  - Archive: {'yes' if summary['has_archive'] else 'no'}")
    logger.info(f"  - Pupitre: {'yes' if summary['has_pupitre'] else 'no'}")
    logger.info(
        f"  - Incidents: {summary['n_default']} default, {summary['n_trigger']} trigger, {summary['n_spike']} spike"
    )
    logger.info(f"Signatures: {', '.join(summary['signatures']) or 'None'}")

    if summary["sync_info"]:
        logger.info("Sync info:")
        for key, value in summary["sync_info"].items():
            logger.info(f"  - {key}: {value}")

    logger.info(f"Pupitre params: teb={summary['teb']:.2f}, BP={summary['BP']:.2f}")
    logger.info("=" * 60)
