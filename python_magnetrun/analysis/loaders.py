"""
Data loading and file operations for magnetrun analysis.

This module provides utilities for:
- Discovering related files (overview, archive, pupitre, incidents)
- Loading TDMS and text data files
- Extracting timestamps and metadata from filenames
- Merging multiple data sources

The file discovery process works as follows:
1. Parse the overview filename to extract site, mode, date, time
2. Build glob patterns for related files (archive, pupitre, incidents)
3. Filter files by timestamp range to match the overview period
4. Return a FileSet containing all related files

Example usage::

    from python_magnetrun.analysis.loaders import FileDiscovery, load_data, merge_data

    # Discover files related to an overview file
    discovery = FileDiscovery(pupitre_datadir="/path/to/pupitre")
    file_set = discovery.discover("M9_Overview_241106-1643.tdms")

    # Load and merge data
    dfs = load_data(file_set.archive, site="M9", insert="", group="Courants_Alimentations", keys=["Courant_GR1"])
    df = merge_data(dfs)
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from natsort import natsorted

from .config import (
    DEFAULT_DATA_DIR,
    DEFAULT_PIGBROTHER_DATA_DIR,
    DEFAULT_GROUP,
    TIME_OFFSET_ARCHIVE,
    TIME_OFFSET_INCIDENTS,
    TIME_OFFSET_OVERVIEW,
)

# Module logger
logger = logging.getLogger("magnetrun.analysis.loaders")


# =============================================================================
# Timestamp format constants
# =============================================================================
TIMESTAMP_FORMAT: str = "%Y-%m-%d %H:%M:%S"
"""Standard timestamp format used for file selection."""


# =============================================================================
# Utility functions
# =============================================================================
def convert_to_timestamp(
    date_str: str,
    time_str: str,
    date_format: str = "%y%m%d",
    time_format: str = "%H%M",
) -> Tuple[float, str]:
    """
    Convert date and time strings to timestamp and formatted string.

    Parameters
    ----------
    date_str : str
        Date string (e.g., "241106" for 2024-11-06)
    time_str : str
        Time string (e.g., "1643" for 16:43)
    date_format : str, optional
        Format for parsing date string (default: "%y%m%d")
    time_format : str, optional
        Format for parsing time string (default: "%H%M")

    Returns
    -------
    tuple[float, str]
        (unix_timestamp, formatted_datetime_string)

    Examples
    --------
    >>> ts, fmt = convert_to_timestamp("241106", "1643")
    >>> print(fmt)
    2024-11-06 16:43:00

    >>> ts, fmt = convert_to_timestamp("2024.11.06", "16:43:00",
    ...                                 date_format="%Y.%m.%d",
    ...                                 time_format="%H:%M:%S")
    >>> print(fmt)
    2024-11-06 16:43:00
    """
    date_time_str = date_str + time_str
    date_time_format = date_format + time_format
    date_time_obj = datetime.strptime(date_time_str, date_time_format)

    timestamp = date_time_obj.timestamp()
    formatted = date_time_obj.strftime(TIMESTAMP_FORMAT)

    return (timestamp, formatted)


# =============================================================================
# File metadata dataclass
# =============================================================================
@dataclass
class FileMetadata:
    """
    Metadata extracted from a data file.

    Attributes
    ----------
    filepath : str
        Full path to the file
    site : str
        Site identifier (M8, M9, M10)
    mode : str
        File mode/type (Overview, Archive, Default, etc.)
    start_time : str
        Start timestamp as formatted string
    end_time : str
        End timestamp as formatted string
    start_timestamp : float
        Start time as Unix timestamp
    duration : float
        Duration in seconds
    skip : bool
        Whether this file should be skipped (e.g., missing key)
    """

    filepath: str
    site: str = ""
    mode: str = ""
    start_time: str = ""
    end_time: str = ""
    start_timestamp: float = 0.0
    duration: float = 0.0
    skip: bool = False

    @property
    def filename(self) -> str:
        """Get filename without extension."""
        return Path(self.filepath).stem

    @property
    def extension(self) -> str:
        """Get file extension."""
        return Path(self.filepath).suffix

    def overlaps(self, start: str, end: str) -> bool:
        """
        Check if this file's time range overlaps with given range.

        Parameters
        ----------
        start : str
            Start time in TIMESTAMP_FORMAT
        end : str
            End time in TIMESTAMP_FORMAT

        Returns
        -------
        bool
            True if ranges overlap
        """
        if not self.start_time or not self.end_time:
            return False

        file_start = datetime.strptime(self.start_time, TIMESTAMP_FORMAT)
        file_end = datetime.strptime(self.end_time, TIMESTAMP_FORMAT)
        range_start = datetime.strptime(start, TIMESTAMP_FORMAT)
        range_end = datetime.strptime(end, TIMESTAMP_FORMAT)

        return file_start >= range_start and file_end <= range_end


# =============================================================================
# File set dataclass
# =============================================================================
@dataclass
class FileSet:
    """
    Container for a set of related files.

    Groups all files associated with a single overview file:
    overview, archive, pupitre, and incident files (default, trigger, spike).

    Attributes
    ----------
    overview : List[str]
        Overview TDMS files (typically just one)
    archive : List[str]
        Archive TDMS files (120 Hz data)
    pupitre : List[str]
        Pupitre text files (control system data)
    default : List[str]
        Default incident files (4800 Hz)
    trigger : List[str]
        Manual trigger incident files
    spike : List[str]
        Spike incident files
    """

    overview: List[str] = field(default_factory=list)
    archive: List[str] = field(default_factory=list)
    pupitre: List[str] = field(default_factory=list)
    default: List[str] = field(default_factory=list)
    trigger: List[str] = field(default_factory=list)
    spike: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        """Convert to dictionary format (backward compatibility)."""
        return {
            "overview": self.overview,
            "archive": self.archive,
            "pupitre": self.pupitre,
            "default": self.default,
            "trigger": self.trigger,
            "spike": self.spike,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, List[str]]) -> "FileSet":
        """Create from dictionary."""
        return cls(
            overview=d.get("overview", []),
            archive=d.get("archive", []),
            pupitre=d.get("pupitre", []),
            default=d.get("default", []),
            trigger=d.get("trigger", []),
            spike=d.get("spike", []),
        )

    def __len__(self) -> int:
        """Total number of files."""
        return (
            len(self.overview)
            + len(self.archive)
            + len(self.pupitre)
            + len(self.default)
            + len(self.trigger)
            + len(self.spike)
        )

    @property
    def has_archive(self) -> bool:
        """Check if archive files are available."""
        return len(self.archive) > 0

    @property
    def has_pupitre(self) -> bool:
        """Check if pupitre files are available."""
        return len(self.pupitre) > 0

    @property
    def has_incidents(self) -> bool:
        """Check if any incident files are available."""
        return len(self.default) > 0 or len(self.trigger) > 0 or len(self.spike) > 0


# =============================================================================
# Extract data function
# =============================================================================
def extract_data(
    file: str,
    site: str,
    insert: str,
    key: Optional[str] = None,
    dry_run: bool = False,
) -> Tuple[str, str, bool]:
    """
    Extract timestamp range and metadata from a data file.

    Parses the filename to determine timestamps, and optionally loads
    the file to verify keys and get exact duration.

    Parameters
    ----------
    file : str
        Path to the data file (.txt or .tdms)
    site : str
        Site identifier (used for loading)
    insert : str
        Insert identifier (used for loading)
    key : str, optional
        Key to verify exists in the file
    dry_run : bool, optional
        If True, only parse filename without loading data

    Returns
    -------
    tuple[str, str, bool]
        (start_timestamp, end_timestamp, skip_flag)
        Timestamps are formatted as TIMESTAMP_FORMAT strings.
        skip_flag is True if the file should be skipped.

    Raises
    ------
    RuntimeError
        If file extension is not supported

    Examples
    --------
    >>> start, end, skip = extract_data("M9_Overview_241106-1643.tdms", "M9", "")
    >>> print(start)
    2024-11-06 16:43:00
    """
    # Lazy import to avoid circular dependencies
    from python_magnetrun.MagnetRun import MagnetRun

    extension = os.path.splitext(file)[-1]
    filename = os.path.basename(file).replace(extension, "")

    start_timestamp: float = 0.0
    start_ftimestamp: str = ""
    mrun = None

    if extension == ".txt":
        # Pupitre file format: "2024.11.06 - 16:43:00.txt"
        date, time = filename.replace(".txt", "").split(" - ")
        start_timestamp, start_ftimestamp = convert_to_timestamp(
            date, time, date_format="%Y.%m.%d", time_format="%H:%M:%S"
        )
        if not dry_run:
            mrun = MagnetRun.fromtxt(site, insert, file)

    elif extension == ".tdms":
        res = filename.split("_")

        # Regular case: M9_Overview_241106-1643
        if len(res) == 3:
            site_from_file, mode, timestamp = res
            date, time = timestamp.split("-")
            # Time may have seconds (6 chars) or just HHMM (4 chars)
            start_timestamp, start_ftimestamp = convert_to_timestamp(
                date,
                time[:4],  # Use first 4 chars for HHMM
                date_format="%y%m%d",
                time_format="%H%M",
            )
        # Special case for default files: M9_Default_241106-164300_HT
        elif len(res) == 4:
            site_from_file, mode, timestamp, dmode = res
            date, time = timestamp.split("-")
            start_timestamp, start_ftimestamp = convert_to_timestamp(
                date, time, date_format="%y%m%d", time_format="%H%M%S"
            )
        else:
            logger.warning("Unexpected filename format: %s", filename)
            # Try to parse anyway
            start_ftimestamp = ""

        if not dry_run:
            mrun = MagnetRun.fromtdms(site, insert, file)
    else:
        raise RuntimeError(f"{file}: unsupported extension {extension}")

    skip = False
    end_ftimestamp = ""

    if not dry_run and mrun is not None:
        mdata = mrun.getMData()

        # Check if required key exists
        if key is not None:
            if key not in mdata.getKeys():
                logger.debug("%s: key %s not found", file, key)
                skip = True

        # Calculate end timestamp from duration
        duration = mdata.getDuration()
        end_dt = datetime.fromtimestamp(start_timestamp) + timedelta(seconds=duration)
        end_ftimestamp = end_dt.strftime(TIMESTAMP_FORMAT)

    return (start_ftimestamp, end_ftimestamp, skip)


# =============================================================================
# Find files function
# =============================================================================
def find_files(
    overview_file: str,
    site: str,
    date: str,
    time: str,
    pupitre_datadir: Union[str, Path] = DEFAULT_DATA_DIR,
) -> Tuple[str, str, str, str, str]:
    """
    Build glob patterns to find files related to an overview file.

    Parameters
    ----------
    overview_file : str
        Path to the overview TDMS file
    site : str
        Site identifier (M8, M9, M10)
    date : str
        Date string from filename (e.g., "241106")
    time : str
        Time string from filename (e.g., "1643")
    pupitre_datadir : str or Path, optional
        Base directory for pupitre files

    Returns
    -------
    tuple[str, str, str, str, str]
        (pupitre_filter, archive_filter, default_filter, trigger_filter, spike_filter)
        Each is a glob pattern for finding related files.

    Examples
    --------
    >>> filters = find_files("data/M9_Overview_241106-1643.tdms", "M9", "241106", "1643")
    >>> pupitre, archive, default, trigger, spike = filters
    """
    pupitre_datadir = Path(pupitre_datadir)

    # Pupitre pattern: /datadir/M9/2024.11.06*.txt
    pupitre_site_dir = pupitre_datadir / site
    pupitre_filter = str(
        pupitre_site_dir / f"20{date[0:2]}.{date[2:4]}.{date[4:]}*.txt"
    )

    # Get base paths from overview file
    extension = os.path.splitext(overview_file)[-1]
    filename = os.path.basename(overview_file).replace(extension, "")
    overview_dir = os.path.dirname(overview_file)

    # Archive pattern
    pigbrother = filename.replace("Overview", "Archive")
    archive_datadir = overview_dir.replace("Overview", "Fichiers_Archive")
    archive_filter = f"{archive_datadir}/{pigbrother.replace(time, '*.tdms')}"

    # Incident patterns
    default_datadir = overview_dir.replace("Overview", "Fichiers_Default")
    trigger_datadir = overview_dir.replace("Overview", "Fichiers_Manuel_Trig")
    spike_datadir = overview_dir.replace("Overview", "Fichiers_Spike")

    default_name = filename.replace("Overview", "Default")
    default_filter = f"{default_datadir}/{default_name.replace(time, '*.tdms')}"

    trigger_name = filename.replace("Overview", "ManuelTrig")
    trigger_filter = f"{trigger_datadir}/{trigger_name.replace(time, '*.tdms')}"

    spike_name = filename.replace("Overview", "Spikes")
    spike_filter = f"{spike_datadir}/{spike_name.replace(time, '*.tdms')}"

    return (
        pupitre_filter,
        archive_filter,
        default_filter,
        trigger_filter,
        spike_filter,
    )


# =============================================================================
# Select files function
# =============================================================================
def select_files(
    files: List[str],
    site: str,
    start: str,
    end: str,
) -> List[str]:
    """
    Filter files by timestamp range.

    Selects files whose time range falls within the specified range.

    Parameters
    ----------
    files : List[str]
        List of file paths to filter
    site : str
        Site identifier
    start : str
        Start timestamp (TIMESTAMP_FORMAT)
    end : str
        End timestamp (TIMESTAMP_FORMAT)

    Returns
    -------
    List[str]
        Filtered and naturally sorted list of files

    Examples
    --------
    >>> files = glob.glob("data/M9_Archive_241106-*.tdms")
    >>> selected = select_files(files, "M9", "2024-11-06 16:00:00", "2024-11-06 18:00:00")
    """
    if not files:
        return []

    start_time = datetime.strptime(start, TIMESTAMP_FORMAT)
    end_time = datetime.strptime(end, TIMESTAMP_FORMAT)

    selected = []
    for file in files:
        try:
            file_start, file_end, skip = extract_data(
                file, site=site, insert="", key=None, dry_run=False
            )

            if not file_start or not file_end:
                continue

            file_start_time = datetime.strptime(file_start, TIMESTAMP_FORMAT)
            file_end_time = datetime.strptime(file_end, TIMESTAMP_FORMAT)

            # Check if file time range is within selection range
            if file_start_time >= start_time and file_end_time <= end_time:
                selected.append(file)

        except Exception as e:
            logger.warning("Error processing %s: %s", file, e)
            continue

    return natsorted(selected) if selected else []


# =============================================================================
# Load DataFrame functions
# =============================================================================
def load_df(
    file: str,
    site: str,
    insert: str,
    group: str,
    keys: Optional[List[str]],
) -> Tuple[pd.DataFrame, Optional[datetime]]:
    """
    Load a single file into a pandas DataFrame.

    Handles both .txt (pupitre) and .tdms (pigbrother) files.
    Adds timestamp column for time alignment.

    Parameters
    ----------
    file : str
        Path to the data file
    site : str
        Site identifier
    insert : str
        Insert identifier
    group : str
        TDMS group name (for .tdms files)
    keys : List[str]
        Column/channel names to load

    Returns
    -------
    tuple[pd.DataFrame, datetime | None]
        (dataframe, start_time)
        DataFrame contains requested columns plus 'timestamp'.
        Returns (empty DataFrame, None) if loading fails.

    Examples
    --------
    >>> df, t0 = load_df("M9_Archive_241106-1643.tdms", "M9", "",
    ...                   "Courants_Alimentations", ["Courant_GR1", "Courant_GR2"])
    """
    # Lazy import
    from python_magnetrun.MagnetRun import MagnetRun

    logger.info(f"load_df: file={file}, group={group}, keys={keys}")

    extension = os.path.splitext(file)[-1]
    df = pd.DataFrame()
    t0: Optional[datetime] = None

    try:
        if extension == ".txt":
            mrun = MagnetRun.fromtxt(site, insert, file)
            mdata = mrun.getMData()
            t0 = mdata.Data["timestamp"].iloc[0]
            selected_keys = ["t", "timestamp"]
            if keys is not None:
                selected_keys += keys
            df = pd.DataFrame(mdata.getData(selected_keys))

        elif extension == ".tdms":
            mrun = MagnetRun.fromtdms(site, insert, file)
            mdata = mrun.getMData()

            # Load data
            channels = list(mdata.Data[group].keys())
            print(f"load_df: channels={channels}", flush=True)
            df = pd.DataFrame(mdata.getTdmsData(group, keys))

            # Check if first key exists
            first_key = channels[0] if keys is None or not keys else keys[0]
            print(f"first_key: {first_key}", flush=True)
            if keys is not None:
                if keys[0] not in mdata.Groups.get(group, {}):
                    logger.debug(
                        "load_df: %s/%s not found in %s", group, keys[0], mdata.FileName
                    )
                    return df, t0

            # Get timing information
            t0 = mdata.Groups[group][first_key]["wf_start_time"]
            dt = mdata.Groups[group][first_key]["wf_increment"]
            t_offset = mdata.Groups[group][first_key]["wf_start_offset"]

            logger.debug("%s: t0=%s, dt=%s, t_offset=%s", file, t0, dt, t_offset)

            # Add timestamp column
            df["timestamp"] = [
                np.datetime64(t0).astype(datetime)
                + timedelta(seconds=i * dt + t_offset)
                for i in df.index.to_list()
            ]
        else:
            logger.warning("Unsupported file extension: %s", extension)

    except Exception as e:
        logger.error("Failed to load %s: %s", file, e)

    return df, t0


def load_data(
    files: List[str],
    site: str,
    insert: str,
    group: str,
    keys: Optional[List[str]],
) -> List[pd.DataFrame]:
    """
    Load multiple files and return list of DataFrames.

    Parameters
    ----------
    files : List[str]
        List of file paths to load
    site : str
        Site identifier
    insert : str
        Insert identifier
    group : str
        TDMS group name
    keys : List[str]
        Column/channel names to load

    Returns
    -------
    List[pd.DataFrame]
        List of loaded DataFrames (empty DataFrames are excluded)

    Examples
    --------
    >>> files = ["archive1.tdms", "archive2.tdms"]
    >>> dfs = load_data(files, "M9", "", "Courants_Alimentations", ["Courant_GR1"])
    """
    logger.info(f"load_data: files={files}, group={group}, keys={keys}")

    df_list = []
    for file in files:
        df, t0 = load_df(file, site, insert, group, keys)
        if not df.empty:
            df_list.append(df)
    return df_list


def merge_data(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple DataFrames into one.

    Concatenates DataFrames vertically, preserving column structure.

    Parameters
    ----------
    df_list : List[pd.DataFrame]
        List of DataFrames to merge

    Returns
    -------
    pd.DataFrame
        Merged DataFrame

    Raises
    ------
    ValueError
        If df_list is empty

    Examples
    --------
    >>> merged = merge_data([df1, df2, df3])
    """
    if not df_list:
        raise ValueError("Cannot merge empty list of DataFrames")

    if len(df_list) == 1:
        return df_list[0]

    return pd.concat(df_list, ignore_index=True)


# =============================================================================
# FileDiscovery class
# =============================================================================
class FileDiscovery:
    """
    Discovers and filters related data files for an overview file.

    This class encapsulates the logic for finding all files related to
    a single overview TDMS file, including archive, pupitre, and incident files.

    Parameters
    ----------
    pupitre_datadir : str or Path
        Directory containing pupitre data files
    log_datadir : str or Path, optional
        Directory containing log files (defaults to pupitre_datadir)

    Attributes
    ----------
    pupitre_datadir : Path
        Pupitre data directory
    log_datadir : Path
        Log data directory

    Examples
    --------
    >>> discovery = FileDiscovery(pupitre_datadir="/data/pupitre")
    >>> file_set = discovery.discover("M9_Overview_241106-1643.tdms")
    >>> print(f"Found {len(file_set.archive)} archive files")
    """

    def __init__(
        self,
        pupitre_datadir: Union[str, Path] = DEFAULT_DATA_DIR,
        pigbrother_datadir: Union[str, Path] = DEFAULT_PIGBROTHER_DATA_DIR,
        log_datadir: Optional[Union[str, Path]] = None,
    ):
        self.pupitre_datadir = Path(pupitre_datadir)
        self.pigbrother_datadir = Path(pigbrother_datadir)
        self.log_datadir = Path(log_datadir) if log_datadir else self.pupitre_datadir

    def discover(
        self,
        overview_file: str,
        site: Optional[str] = None,
    ) -> FileSet:
        """
        Discover all files related to an overview file.

        Parameters
        ----------
        overview_file : str
            Path to the overview TDMS file
        site : str, optional
            Site identifier (extracted from filename if not provided)

        Returns
        -------
        FileSet
            Container with all discovered related files
        """
        # Resolve overview path: if dirname is empty, look under pigbrother_datadir/<site>/Overview
        extension = os.path.splitext(overview_file)[-1]
        basename = os.path.basename(overview_file)
        filename = basename.replace(extension, "")
        logger.info(
            f"discover overview file: {overview_file} (filename={filename}, extension={extension})"
        )

        # If an explicit dirname wasn't provided, try to locate the overview
        # file under the pigbrother data directory structure.
        overview_dir = os.path.dirname(overview_file)
        resolved_overview = overview_file
        if not overview_dir:
            # Try pigbrother_datadir/<site>/Overview (site parsed below)
            logger.debug("No directory in overview_file, attempting to resolve...")
            parts_tmp = filename.split("_")
            if parts_tmp:
                file_site_tmp = site if site else parts_tmp[0]
                candidate_dir = self.pigbrother_datadir / file_site_tmp / "Overview"
                candidate_path = candidate_dir / basename
                logger.debug(f"candidate_path={candidate_path}")
                if candidate_path.exists():
                    resolved_overview = str(candidate_path)
                    overview_dir = str(candidate_dir)
                    logger.debug(
                        "Resolved overview %s -> %s", overview_file, resolved_overview
                    )
                else:
                    # Keep original (may be relative to cwd)
                    overview_dir = ""
        logger.info(
            f"discover overview_dir={overview_dir}, resolved_overview={resolved_overview}"
        )

        # Extract site, mode, timestamp from filename
        parts = filename.split("_")
        if len(parts) < 3:
            logger.error("Cannot parse filename: %s", filename)
            return FileSet(overview=[f"{overview_dir}/{overview_file}"])

        file_site = parts[0]
        timestamp = parts[2]

        if site is None:
            site = file_site

        # Parse date and time
        date, time = timestamp.split("-")

        # Use resolved overview path (may be under pigbrother_datadir)
        overview_path_for_extract = resolved_overview

        # Get time range from overview file
        start, end, skip = extract_data(
            overview_path_for_extract, site, insert="", key=None, dry_run=False
        )

        if skip or not start or not end:
            logger.warning("Could not extract time range from %s", overview_file)
            return FileSet(overview=[f"{overview_dir}/{overview_file}"])

        # Get file patterns (pass a path that includes the directory)
        overview_for_patterns = overview_path_for_extract
        filters = find_files(
            overview_for_patterns,
            site,
            date,
            time,
            pupitre_datadir=self.pupitre_datadir,
        )
        pupitre_filter, archive_filter, default_filter, trigger_filter, spike_filter = (
            filters
        )

        logger.debug("File patterns:")
        logger.debug("  pupitre: %s", pupitre_filter)
        logger.debug("  archive: %s", archive_filter)
        logger.debug("  default: %s", default_filter)
        logger.debug("  trigger: %s", trigger_filter)
        logger.debug("  spike: %s", spike_filter)

        # Find and filter files
        file_set = FileSet(overview=[f"{overview_dir}/{overview_file}"])

        file_set.pupitre = select_files(glob.glob(pupitre_filter), site, start, end)
        file_set.archive = select_files(glob.glob(archive_filter), site, start, end)
        file_set.default = select_files(glob.glob(default_filter), site, start, end)
        file_set.trigger = select_files(glob.glob(trigger_filter), site, start, end)
        file_set.spike = select_files(glob.glob(spike_filter), site, start, end)
        logger.info("file_set.pupitre: %s", file_set.pupitre)
        logger.info("file_set.archive: %s", file_set.archive)
        logger.info("file_set.default: %s", file_set.default)
        logger.info("file_set.spike: %s", file_set.spike)
        logger.info("file_set.trigger: %s", file_set.trigger)

        logger.info(
            "Discovered files for %s: %d archives, %d pupitres, %d incidents",
            filename,
            len(file_set.archive),
            len(file_set.pupitre),
            len(file_set.default) + len(file_set.trigger) + len(file_set.spike),
        )

        return file_set

    def discover_batch(
        self,
        overview_files: List[str],
    ) -> Dict[str, FileSet]:
        """
        Discover files for multiple overview files.

        Parameters
        ----------
        overview_files : List[str]
            List of overview file paths

        Returns
        -------
        Dict[str, FileSet]
            Dictionary mapping filenames to FileSets
        """
        results = {}
        for overview_file in overview_files:
            filename = Path(overview_file).stem
            results[filename] = self.discover(overview_file)
        return results


# =============================================================================
# Convenience function for backward compatibility
# =============================================================================
def discover_files(
    overview_file: str,
    pupitre_datadir: Union[str, Path] = DEFAULT_DATA_DIR,
    pigbrother_datadir: Union[str, Path] = DEFAULT_PIGBROTHER_DATA_DIR,
    site: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Discover files related to an overview file (backward compatible).

    This function provides backward compatibility with code that expects
    the dict_files format from the original analysis-refactor.py.

    Parameters
    ----------
    overview_file : str
        Path to overview TDMS file
    pupitre_datadir : str or Path
        Pupitre data directory
    site : str, optional
        Site identifier

    Returns
    -------
    Dict[str, List[str]]
        Dictionary with keys: overview, archive, pupitre, default, trigger, spike
    """
    discovery = FileDiscovery(pupitre_datadir=pupitre_datadir)
    file_set = discovery.discover(overview_file, site=site)
    return file_set.to_dict()
