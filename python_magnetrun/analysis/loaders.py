"""
Data loading and file operations for magnetrun analysis.

This module provides utilities for:
- Discovering related files (overview, archive, pupitre, incidents)
- Loading TDMS and text data files
- Extracting timestamps and metadata from filenames
- Merging multiple data sources

The file discovery process works as follows:
1. Parse the overview filename to extract housing, mode, date, time
2. Build glob patterns for related files (archive, pupitre, incidents)
3. Filter files by timestamp range to match the overview period
4. Return a FileSet containing all related files

Example usage::

    from python_magnetrun.analysis.loaders import FileDiscovery, load_data, merge_data

    # Discover files related to an overview file
    discovery = FileDiscovery(pupitre_datadir="/path/to/pupitre")
    file_set = discovery.discover("M9_Overview_241106-1643.tdms")

    # Load and merge data
    dfs = load_data(file_set.archive, housing="M9", site="", group="Courants_Alimentations", keys=["Courant_GR1"])
    df = merge_data(dfs)
"""

from __future__ import annotations

import glob
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from natsort import natsorted

from ..magnetdata_base import DataType
from ..magnetdata_pandas import _open_text_with_fallback
from ..runlogs.pigbrother import PIGBROTHER_LOG_FILENAME
from .config import (
    DEFAULT_DATA_DIR,
    DEFAULT_PIGBROTHER_DATA_DIR,
)

# Module logger
logger = logging.getLogger("python_magnetrun.analysis.loaders")


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
) -> tuple[float, str]:
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
    housing : str
        Housing identifier (M8, M9, M10)
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
    housing: str = ""
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
    overview, archive, pupitre, incident files (default, trigger, spike),
    hybrid files (kHz, rms, trigger, vprocess),
    and run-log files (pigbrother ACQ_ENET, pupitre Cirrus).

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
    hybrid_kHz : List[str]
        Hybrid kHz acquisition files
    hybrid_rms : List[str]
        Hybrid RMS files
    hybrid_trigger : List[str]
        Hybrid trigger files
    hybrid_vprocess : List[str]
        Hybrid voltage-process files
    pigbrother_runlog : List[str]
        Pigbrother run-log files (``LOG_ACQ_ENET.txt``)
    pupitre_runlog : List[str]
        Pupitre Cirrus run-log files (``cirrus/A[1-4]/YYYY-MM-DD_cirrus_out.log``)
    """

    overview: list[str] = field(default_factory=list)
    archive: list[str] = field(default_factory=list)
    pupitre: list[str] = field(default_factory=list)
    default: list[str] = field(default_factory=list)
    trigger: list[str] = field(default_factory=list)
    spike: list[str] = field(default_factory=list)
    hybrid_kHz: list[str] = field(default_factory=list)
    hybrid_rms: list[str] = field(default_factory=list)
    hybrid_trigger: list[str] = field(default_factory=list)
    hybrid_vprocess: list[str] = field(default_factory=list)
    pigbrother_runlog: list[str] = field(default_factory=list)
    pupitre_runlog: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        """Convert to dictionary format (backward compatibility)."""
        return {
            "overview": self.overview,
            "archive": self.archive,
            "pupitre": self.pupitre,
            "default": self.default,
            "trigger": self.trigger,
            "spike": self.spike,
            "hybrid_kHz": self.hybrid_kHz,
            "hybrid_rms": self.hybrid_rms,
            "hybrid_trigger": self.hybrid_trigger,
            "hybrid_vprocess": self.hybrid_vprocess,
            "pigbrother_runlog": self.pigbrother_runlog,
            "pupitre_runlog": self.pupitre_runlog,
        }

    @classmethod
    def from_dict(cls, d: dict[str, list[str]]) -> FileSet:
        """Create from dictionary."""
        return cls(
            overview=d.get("overview", []),
            archive=d.get("archive", []),
            pupitre=d.get("pupitre", []),
            default=d.get("default", []),
            trigger=d.get("trigger", []),
            spike=d.get("spike", []),
            hybrid_kHz=d.get("hybrid_kHz", []),
            hybrid_rms=d.get("hybrid_rms", []),
            hybrid_trigger=d.get("hybrid_trigger", []),
            hybrid_vprocess=d.get("hybrid_vprocess", []),
            pigbrother_runlog=d.get("pigbrother_runlog", []),
            pupitre_runlog=d.get("pupitre_runlog", []),
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
            + len(self.hybrid_kHz)
            + len(self.hybrid_rms)
            + len(self.hybrid_trigger)
            + len(self.hybrid_vprocess)
            + len(self.pigbrother_runlog)
            + len(self.pupitre_runlog)
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

    @property
    def has_hybrid_kHz(self) -> bool:
        """Check if hybrid kHz files are available."""
        return len(self.hybrid_kHz) > 0

    @property
    def has_hybrid_rms(self) -> bool:
        """Check if hybrid RMS files are available."""
        return len(self.hybrid_rms) > 0

    @property
    def has_hybrid_vprocess(self) -> bool:
        """Check if hybrid voltage-process files are available."""
        return len(self.hybrid_vprocess) > 0

    @property
    def has_hybrid_incidents(self) -> int:
        """Return number of hybrid trigger files."""
        return len(self.hybrid_trigger)


# =============================================================================
# Metadata-only helpers (no data arrays loaded)
# =============================================================================


def _tdms_end_from_properties(
    file: str,
    start_timestamp: float,
    start_ftimestamp: str,
    check_all_groups: bool = False,
) -> str:
    """Return end timestamp by reading actual sample counts from TDMS data.

    For each usable channel (skipping the ``Infos`` group), counts the actual
    number of samples written via ``read_data_chunks()`` and multiplies by
    ``wf_increment`` from channel properties.  ``wf_samples`` is NOT trusted
    because it is written by the DAQ at acquisition *start* and may differ from
    the data actually recorded (e.g. truncated files).

    Parameters
    ----------
    file:
        Path to the TDMS file.
    start_timestamp:
        Unix timestamp of the acquisition start (from the filename).
    start_ftimestamp:
        Formatted start timestamp string (unused, kept for API compatibility).
    check_all_groups:
        When ``True``, check every group instead of stopping after the first
        usable channel.  A warning is emitted if groups report different
        durations (indicating a corrupt or truncated file).  The duration
        computed from the first usable channel is still returned.

    Returns an empty string if no usable channel is found.
    """
    from nptdms import TdmsFile

    try:
        with TdmsFile.open(file) as tdms:
            first_duration: float | None = None
            for group in tdms.groups():
                if group.name == "Infos":
                    continue
                # Scan ALL channels in the group and keep the longest duration.
                # Spike/trigger files often have a single-sample trigger channel
                # first; using only the first channel would give a misleadingly
                # short duration.  The waveform channels (many samples) follow.
                group_duration: float | None = None
                for ch in group.channels():
                    p = ch.properties
                    logger.debug(f"checking channel {ch.path} with properties {p}")
                    if "wf_increment" not in p:
                        continue
                    actual = sum(len(chunk) for chunk in ch.data_chunks())
                    wf_prop = p.get("wf_samples")
                    if wf_prop is not None and int(wf_prop) != actual:
                        logger.warning(
                            f"{file} {ch.path}: wf_samples={wf_prop} != actual={actual} — using actual count"
                        )
                    duration = actual * float(p["wf_increment"])
                    logger.debug(
                        f"{file} {ch.path}: actual={actual} samples, duration={duration:.3f}s"
                    )
                    if group_duration is None or duration > group_duration:
                        group_duration = duration

                if group_duration is None:
                    continue  # no channel with wf_increment in this group

                if first_duration is None:
                    first_duration = group_duration
                    if not check_all_groups:
                        break  # fast path: first usable group is enough
                elif abs(group_duration - first_duration) > 1.0:
                    logger.warning(
                        f"{file} group {group.name!r}: duration {group_duration:.3f}s differs from first group {first_duration:.3f}s"
                    )

            if first_duration is not None:
                end_dt = datetime.fromtimestamp(start_timestamp) + timedelta(
                    seconds=first_duration
                )
                return end_dt.strftime(TIMESTAMP_FORMAT)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.warning(f"{file}: {exc}")
    return ""


def _pupitre_end_from_last_line(file: str, keys: list[str]) -> str:
    """Return end timestamp from the last data row without loading the full file.

    Seeks to the end of *file*, walks back to find the last non-empty line,
    then parses ``Date`` and ``Time`` fields by column position.

    Returns an empty string on any parse or I/O failure.
    """
    if "Date" not in keys or "Time" not in keys:
        return ""
    date_idx = keys.index("Date")
    time_idx = keys.index("Time")
    try:
        with open(file, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            if pos == 0:
                return ""
            # skip trailing newline/carriage-return bytes
            pos -= 1
            f.seek(pos)
            while pos > 0 and f.read(1) in (b"\n", b"\r", b" "):
                pos -= 1
                f.seek(pos)
            # walk back to the start of the last data line
            while pos > 0:
                pos -= 1
                f.seek(pos)
                if f.read(1) == b"\n":
                    break
            last_line = f.readline().decode(errors="replace").strip()
        fields = last_line.split()
        if len(fields) <= max(date_idx, time_idx):
            return ""
        end_dt = datetime.strptime(
            f"{fields[date_idx]} {fields[time_idx]}", "%Y.%m.%d %H:%M:%S"
        )
        logger.debug(f"end_dt={end_dt} from last line: {last_line}")
        return end_dt.strftime(TIMESTAMP_FORMAT)
    except (OSError, ValueError, IndexError, UnicodeDecodeError) as exc:
        logger.warning(f"{file}: {exc}")
        return ""


# =============================================================================
# Extract data function
# =============================================================================
def extract_data(
    file: str,
    housing: str,
    site: str = "",
    key: str | None = None,
    dry_run: bool = False,
) -> tuple[str, str, bool]:
    """
    Extract timestamp range and metadata from a data file.

    Parses the filename to determine timestamps, and optionally loads
    the file to verify keys and get exact duration.

    Parameters
    ----------
    file : str
        Path to the data file (.txt or .tdms)
    housing : str, optional
        Housing identifier (used for loading)
    site : str, optional
        Site identifier (used for loading)
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

    logger.info(
        f"extract_data: file={file}, housing={housing}, site={site}, key={key}, dry_run={dry_run}"
    )
    extension = os.path.splitext(file)[-1]
    filename = os.path.basename(file).replace(extension, "")

    start_timestamp: float = 0.0
    start_ftimestamp: str = ""
    end_ftimestamp: str = ""
    skip: bool = False
    mrun = None

    if extension == ".txt":
        # Pupitre file format: "2024.11.06 - 16:43:00.txt"
        date, time = filename.replace(".txt", "").split(" - ")
        start_timestamp, start_ftimestamp = convert_to_timestamp(
            date, time, date_format="%Y.%m.%d", time_format="%H:%M:%S"
        )
        logger.info(
            f"Parsed pupitre filename: date={date}, time={time}, start_ftimestamp={start_ftimestamp}"
        )
        if not dry_run:
            if key is None:
                # metadata-only path: read header to get column positions, then
                # seek last line for end timestamp — no full DataFrame loaded
                with _open_text_with_fallback(file) as _f:
                    _hdr = pd.read_csv(
                        _f, sep=r"\s+", engine="python", skiprows=1, nrows=0
                    )
                _keys = _hdr.columns.tolist()
                end_ftimestamp = _pupitre_end_from_last_line(file, _keys)
            else:
                mrun = MagnetRun.fromtxt(housing, site, file)
                logger.info(f"Loaded pupitre file: {file}")

    elif extension == ".tdms":
        res = filename.split("_")

        mode = None
        date, time = None, None

        # Regular case: M9_Overview_241106-1643
        if len(res) == 3:
            housing_from_file, mode, timestamp = res
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
            housing_from_file, mode, timestamp, dmode = res
            if housing_from_file != housing:
                logger.warning(
                    f"Housing mismatch in filename: expected {housing}, got {housing_from_file}"
                )
            date, time = timestamp.split("-")
            start_timestamp, start_ftimestamp = convert_to_timestamp(
                date, time, date_format="%y%m%d", time_format="%H%M%S"
            )
        else:
            logger.warning(f"Unexpected filename format: {filename}")
            # Try to parse anyway
            start_ftimestamp = ""
        logger.info(
            f"Parsed TDMS filename: housing={housing}, mode={mode}, date={date}, time={time}, start_ftimestamp={start_ftimestamp}"
        )
        if not dry_run:
            if key is None:
                # metadata-only path: read wf_samples/wf_increment from
                # channel properties — no data arrays loaded
                end_ftimestamp = _tdms_end_from_properties(
                    file, start_timestamp, start_ftimestamp
                )
            else:
                mrun = MagnetRun.fromtdms(housing, site, file)
                logger.info(f"Loaded TDMS file: {file}")
    else:
        raise RuntimeError(f"{file}: unsupported extension {extension}")

    if not dry_run and mrun is not None:
        mdata = mrun.getMData()

        # Check if required key exists
        if key is not None and key not in mdata.getKeys():
            logger.debug(f"{file}: key {key} not found")
            skip = True

        # Calculate end timestamp from duration
        duration = mdata.getDuration()
        end_dt = datetime.fromtimestamp(start_timestamp) + timedelta(seconds=duration)
        end_ftimestamp = end_dt.strftime(TIMESTAMP_FORMAT)
        logger.info(
            f"{file}: start={start_ftimestamp}, end={end_ftimestamp}, duration={duration}s"
        )
    # dry_run=True: filename-derived start only, end unknown
    # (mrun is None because we skipped all load paths above)

    return (start_ftimestamp, end_ftimestamp, skip)


# =============================================================================
# Find files function
# =============================================================================
def find_files(
    overview_file: str,
    housing: str,
    date: str,
    time: str,
    pupitre_datadir: str | Path = DEFAULT_DATA_DIR,
) -> tuple[str, str, str, str, str]:
    """
    Build glob patterns to find files related to an overview file.

    Parameters
    ----------
    overview_file : str
        Path to the overview TDMS file
    housing : str
        Housing identifier (M8, M9, M10)
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
    logger.info(
        f"find_files: overview_file={overview_file}, housing={housing}, date={date}, time={time}, pupitre_datadir={pupitre_datadir}"
    )
    pupitre_datadir = Path(pupitre_datadir)

    # Pupitre pattern: /datadir/M9/2024.11.06*.txt
    pupitre_site_dir = pupitre_datadir / housing
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
    files: list[str],
    housing: str,
    start: str,
    end: str,
    min_duration_seconds: float = 30.0,
) -> list[str]:
    """
    Filter files by timestamp range.

    Selects files whose time range falls within the specified range.

    Parameters
    ----------
    files : List[str]
        List of file paths to filter
    housing : str
        Housing identifier
    start : str
        Start timestamp (TIMESTAMP_FORMAT)
    end : str
        End timestamp (TIMESTAMP_FORMAT)
    min_duration_seconds : float, optional
        Files shorter than this threshold are discarded (default 30 s).
        Pass 0.0 for incident files (default, trigger, spike) which are
        legitimately short captures.

    Returns
    -------
    List[str]
        Filtered and naturally sorted list of files

    Examples
    --------
    >>> files = glob.glob("data/M9_Archive_241106-*.tdms")
    >>> selected = select_files(files, "M9", "2024-11-06 16:00:00", "2024-11-06 18:00:00")
    """

    natsortedfiles = natsorted(files)
    logger.info(
        f"select_files: files={natsortedfiles}, housing={housing}, start={start}, end={end}"
    )
    if not natsortedfiles:
        return []

    start_time = datetime.strptime(start, TIMESTAMP_FORMAT)
    end_time = datetime.strptime(end, TIMESTAMP_FORMAT)

    selected = []
    for file in natsortedfiles:
        try:
            file_start, file_end, skip = extract_data(
                file, housing, site="", key=None, dry_run=False
            )
            logger.info(
                f"File {file}: extracted start={file_start}, end={file_end}, skip={skip}"
            )

            if not file_start or not file_end:
                continue

            file_start_time = datetime.strptime(file_start, TIMESTAMP_FORMAT)
            file_end_time = datetime.strptime(file_end, TIMESTAMP_FORMAT)

            # Check if file time range is within selection range
            logger.debug(
                f"File {file} (file_start_time={file_start_time}, file_end_time={file_end_time}) "
                f"within the selection range (start_time={start_time}, end_time={end_time})?"
            )
            if file_start_time < end_time and file_end_time > start_time:
                actual_duration = (file_end_time - file_start_time).total_seconds()
                if actual_duration <= min_duration_seconds:
                    logger.warning(
                        f"{file}: duration {actual_duration:.3f}s <= min {min_duration_seconds:.1f}s, skipping"
                    )
                    continue
                logger.debug(f"Selected file: {file}")
                selected.append(file)

        except (OSError, ValueError, RuntimeError, UnicodeDecodeError) as e:
            logger.warning(f"Error processing {file}: {e}")
            continue

    return natsorted(selected) if selected else []


# =============================================================================
# Load DataFrame functions
# =============================================================================
def load_df(
    file: str,
    housing: str,
    site: str,
    group: str,
    keys: list[str] | None,
) -> tuple[pd.DataFrame, datetime | None]:
    """
    Load a single file into a pandas DataFrame.

    Handles both .txt (pupitre) and .tdms (pigbrother) files.
    Adds timestamp column for time alignment.

    Parameters
    ----------
    file : str
        Path to the data file
    housing : str
        Housing identifier
    site : str
        Site identifier
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
    t0: datetime | None = None

    try:
        if extension == ".txt":
            mrun = MagnetRun.fromtxt(housing, site, file)
            mdata = mrun.getMData()
            logger.debug(f"load_df --pupitre -- {file}: mdata keys={mdata.getKeys()}")
            t0 = mdata.start_timestamp
            selected_keys = ["t", "timestamp"]
            if keys is not None:
                selected_keys += keys
            logger.debug(f"load_df: selected_keys={selected_keys}")
            df = pd.DataFrame(mdata.getData(selected_keys))

        elif extension == ".tdms":
            mrun = MagnetRun.fromtdms(housing, site, file)
            mdata = mrun.getMData()
            logger.debug(f"load_df --tdms -- {file}: mdata keys={mdata.getKeys()}")

            # Load data
            channels = list(mdata.getData(group).keys())
            logger.debug(f"channels={channels}")
            df = mdata.getTdmsData(group, keys)

            # Check if first key exists
            first_key = channels[0] if keys is None or not keys else keys[0]
            logger.debug(f"first_key: {first_key}")
            if keys is not None and keys and keys[0] not in mdata.Groups.get(group, {}):
                logger.debug(f"{group}/{keys[0]} not found in {mdata.FileName}")
                return df, t0

            # Use t and timestamp already computed by prepareData → addTime()
            t0 = mdata.start_timestamp
            logger.debug(f"{file}: t0={t0}")
            df["t"] = mdata.Data[group]["t"]
            df["timestamp"] = mdata.Data[group]["timestamp"]
        else:
            logger.warning(f"Unsupported file extension: {extension}")

    except (OSError, ValueError, RuntimeError, KeyError) as e:
        logger.error(f"Failed to load {file}: {e}")

    return df, t0


def load_data(
    files: list[str],
    housing: str,
    site: str,
    group: str,
    keys: list[str] | None,
) -> list[pd.DataFrame]:
    """
    Load multiple files and return list of DataFrames.

    Parameters
    ----------
    files : List[str]
        List of file paths to load
    housing : str
        Housing identifier
    site : str
        Site identifier
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
    logger.info(
        f"load_data: files={files}, housing={housing}, site={site}, group={group}, keys={keys}"
    )

    df_list = []
    for file in files:
        df, t0 = load_df(file, housing, site, group, keys)
        if not df.empty:
            df_list.append(df)
    return df_list


def merge_data(df_list: list[pd.DataFrame]) -> pd.DataFrame:
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


def load_files_data(
    files: list[str],
    housing: str,
    group: str,
    keys: list[str] | None,
) -> pd.DataFrame:
    """
    Load and merge a list of data files into a single DataFrame with a
    continuous ``t`` column.

    Each file is loaded via :func:`~python_magnetrun.MagnetRun.load_mrun`.
    The ``t`` column of every file is shifted so it is relative to the first
    file's ``start_timestamp``, giving a monotonically increasing time axis
    across file boundaries after concatenation.

    For TDMS files the key list is qualified with the group name
    (``"group/key"``) so that :meth:`~TdmsMagnetData.getData` can resolve the
    correct channels.  For text-backed files plain key names are used.

    Parameters
    ----------
    files : list of str
        Paths to the data files (mixed extensions are supported).
    housing : str
        Housing identifier forwarded to :func:`load_mrun`.
    group : str
        TDMS group name used to qualify channel keys for ``.tdms`` files.
    keys : list of str, optional
        Channel names to extract.  ``t`` and ``timestamp`` are always
        included automatically.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with columns ``t``, ``timestamp``, and all
        requested channels.  Returns an empty DataFrame when no file could
        be loaded.
    """
    from python_magnetrun.MagnetRun import load_mrun

    if not files:
        return pd.DataFrame()

    first_t0: pd.Timestamp | None = None
    df_list: list[pd.DataFrame] = []

    for file in files:
        try:
            mrun = load_mrun(file, housing=housing)
            mdata = mrun.getMData()
            t0_file = mdata.start_timestamp

            if first_t0 is None:
                first_t0 = pd.Timestamp(t0_file) if t0_file is not None else None

            shift = (
                (pd.Timestamp(t0_file) - first_t0).total_seconds()
                if first_t0 is not None and t0_file is not None
                else 0.0
            )

            # Dispatch on data type rather than file extension.
            if mdata.Type == DataType.TDMS:
                df = pd.DataFrame(mdata.getTdmsData(group=group, channel=None))
            else:
                desired = (keys or []) + ["t", "timestamp"]
                df = pd.DataFrame(mdata.getData(desired))

            df["t"] = df["t"] + shift

            df_list.append(df)
            logger.debug(f"{file} t0={t0_file}, shift={shift:.3f}s, rows={len(df)}")

        except (OSError, ValueError, RuntimeError, KeyError, UnicodeDecodeError) as e:
            logger.error(f"load_files_data: failed to load {file}: {e}")

    if not df_list:
        return pd.DataFrame()

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
    pigbrother_datadir : str or Path
        Root directory for pigbrother ``.tdms`` files
    pigbrother_runlog_dir : str or Path, optional
        Directory that contains ``LOG_ACQ_ENET.txt``.
        Defaults to ``pigbrother_datadir`` when not set.
    pupitre_runlog_dir : str or Path, optional
        Root directory for pupitre Cirrus run-log files
        (``cirrus/A[1-4]/YYYY-MM-DD_cirrus_out.log``).
        No default — leave ``None`` to skip pupitre run-log discovery.

    Attributes
    ----------
    pupitre_datadir : Path
        Pupitre data directory
    pigbrother_datadir : Path
        Pigbrother data directory
    pigbrother_runlog_dir : Path
        Directory searched for ``LOG_ACQ_ENET.txt``
    pupitre_runlog_dir : Path or None
        Root for Cirrus run-log discovery, or ``None`` if not configured

    Examples
    --------
    >>> discovery = FileDiscovery(pupitre_datadir="/data/pupitre")
    >>> file_set = discovery.discover("M9_Overview_241106-1643.tdms")
    >>> print(f"Found {len(file_set.archive)} archive files")
    >>> print(f"Pigbrother runlog: {file_set.pigbrother_runlog}")
    """

    def __init__(
        self,
        pupitre_datadir: str | Path = DEFAULT_DATA_DIR,
        pigbrother_datadir: str | Path = DEFAULT_PIGBROTHER_DATA_DIR,
        pigbrother_runlog_dir: str | Path | None = None,
        pupitre_runlog_dir: str | Path | None = None,
        hybrid_datadir: str | Path | None = None,
    ):
        self.pupitre_datadir = Path(pupitre_datadir)
        self.pigbrother_datadir = Path(pigbrother_datadir)
        self.pigbrother_runlog_dir = (
            Path(pigbrother_runlog_dir)
            if pigbrother_runlog_dir
            else self.pigbrother_datadir
        )
        self.pupitre_runlog_dir = (
            Path(pupitre_runlog_dir) if pupitre_runlog_dir else None
        )
        self.hybrid_datadir = Path(hybrid_datadir) if hybrid_datadir else None

    def discover(
        self,
        overview_file: str,
        housing: str | None = None,
        dry_run: bool = False,
    ) -> FileSet:
        """
        Discover all files related to an overview file.

        Parameters
        ----------
        overview_file : str
            Path to the overview TDMS file
        housing : str, optional
            Housing identifier (extracted from filename if not provided)
        dry_run : bool, optional
        Returns
        -------
        FileSet
            Container with all discovered related files
        """
        logger.info(
            f"Discovering files for overview: {overview_file}, housing={housing}"
        )
        # Resolve overview path: if dirname is empty, look under pigbrother_datadir/<housing>/Overview
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
            # Try pigbrother_datadir/<housing>/Overview (housing parsed below)
            logger.debug("No directory in overview_file, attempting to resolve...")
            parts_tmp = filename.split("_")
            if parts_tmp:
                file_housing_tmp = housing if housing else parts_tmp[0]
                candidate_dir = self.pigbrother_datadir / file_housing_tmp / "Overview"
                candidate_path = candidate_dir / basename
                logger.debug(f"candidate_path={candidate_path}")
                if candidate_path.exists():
                    resolved_overview = str(candidate_path)
                    overview_dir = str(candidate_dir)
                    logger.debug(
                        f"Resolved overview {overview_file} -> {resolved_overview}"
                    )
                else:
                    # Keep original (may be relative to cwd)
                    overview_dir = ""
        logger.info(
            f"discover overview_dir={overview_dir}, resolved_overview={resolved_overview}"
        )

        # Extract housing, mode, timestamp from filename
        parts = filename.split("_")
        logger.info(f"Filename parts: {parts}")
        if len(parts) < 3:
            logger.error(f"Cannot parse filename: {filename}")
            return FileSet(overview=[f"{overview_dir}/{overview_file}"])

        file_housing = parts[0]
        timestamp = parts[2]

        if housing is None:
            housing = file_housing
            logger.info(f"Extracted housing from filename: {housing}")

        # Parse date and time
        date, time = timestamp.split("-")
        logger.info(f"Extracted date={date}, time={time} from filename")

        # Use resolved overview path (may be under pigbrother_datadir)
        overview_path_for_extract = resolved_overview

        # Get time range from overview file
        start, end, skip = extract_data(
            overview_path_for_extract,
            housing,
            site="",
            key=None,
            dry_run=dry_run,
        )
        logger.info(f"Overview file time range: start={start}, end={end}, skip={skip}")
        if housing == "M8":
            logger.info(
                f"Overview file {overview_file} has housing M8, looking for hybrid data to be implemented"
            )

        if skip or not start or not end:
            logger.warning(f"Could not extract time range from {overview_file}")
            return FileSet(overview=[f"{overview_dir}/{overview_file}"])

        # Get file patterns (pass a path that includes the directory)
        overview_for_patterns = overview_path_for_extract
        filters = find_files(
            overview_for_patterns,
            housing,
            date,
            time,
            pupitre_datadir=self.pupitre_datadir,
        )
        pupitre_filter, archive_filter, default_filter, trigger_filter, spike_filter = (
            filters
        )

        logger.info("File patterns:")
        logger.info(f"  pupitre: {pupitre_filter}")
        logger.info(f"  archive: {archive_filter}")
        logger.info(f"  default: {default_filter}")
        logger.info(f"  trigger: {trigger_filter}")
        logger.info(f"  spike: {spike_filter}")

        # Find and filter files
        file_set = FileSet(overview=[resolved_overview])

        file_set.pupitre = select_files(glob.glob(pupitre_filter), housing, start, end)
        file_set.archive = select_files(glob.glob(archive_filter), housing, start, end)
        # Incident files (default, trigger, spike) are intentionally short captures;
        # disable the minimum-duration guard that is appropriate for archive files.
        file_set.default = select_files(
            glob.glob(default_filter), housing, start, end, min_duration_seconds=0.0
        )
        file_set.trigger = select_files(
            glob.glob(trigger_filter), housing, start, end, min_duration_seconds=0.0
        )
        file_set.spike = select_files(
            glob.glob(spike_filter), housing, start, end, min_duration_seconds=0.0
        )
        logger.info(f"file_set.pupitre: {file_set.pupitre}")
        logger.info(f"file_set.archive: {file_set.archive}")
        logger.info(f"file_set.default: {file_set.default}")
        logger.info(f"file_set.spike: {file_set.spike}")
        logger.info(f"file_set.trigger: {file_set.trigger}")

        # --- Pigbrother run-log (LOG_ACQ_ENET.txt) ---
        pb_log = self.pigbrother_runlog_dir / PIGBROTHER_LOG_FILENAME
        if pb_log.exists():
            file_set.pigbrother_runlog = [str(pb_log)]
            logger.info(f"file_set.pigbrother_runlog: {file_set.pigbrother_runlog}")
        else:
            logger.debug(f"Pigbrother runlog not found at {pb_log}")

        # --- Pupitre run-log (Cirrus cirrus/A[1-4]/YYYY-MM-DD_cirrus_out.log) ---
        logger.debug(
            f"Checking for pupitre run-log: pupitre_runlog_dir={self.pupitre_runlog_dir}, start={start}, end={end}"
        )
        if self.pupitre_runlog_dir is not None and start and end:
            from ..runlogs.pupitre import discover_pupitre_runlogs

            file_set.pupitre_runlog = discover_pupitre_runlogs(
                self.pupitre_runlog_dir,
                start_date=start[:10],
                end_date=end[:10],
            )
            logger.info(f"file_set.pupitre_runlog: {file_set.pupitre_runlog}")

        # --- Hybrid data (M8 only) ---
        logger.debug(
            f"Checking for hybrid data: housing={housing}, hybrid_datadir={self.hybrid_datadir}, start={start}, end={end}"
        )
        if housing == "M8" and self.hybrid_datadir is not None and start and end:
            _ts_part = filename.split("_")[2]  # e.g. "241106-1643"
            _date_part, _time_part = _ts_part.split("-")  # "241106", "1643"
            _local_dt = datetime.strptime(_date_part + _time_part, "%y%m%d%H%M")
            date_str = _local_dt.strftime("%Y-%m-%d")
            time_str = _local_dt.strftime("%H:%M")
            # Load overview to get accurate duration
            from python_magnetrun.MagnetRun import MagnetRun as _MR

            _mrun = _MR.fromtdms(housing, "", resolved_overview)
            _duration = _mrun.getMData().getDuration()
            _local_end_dt = _local_dt + timedelta(seconds=_duration)
            end_time_str = _local_end_dt.strftime("%H:%M")
            hours = range(_local_dt.hour, _local_end_dt.hour + 1)
            try:
                from ..hybrid.hybrid_data import HybridData

                hdata = HybridData(str(self.hybrid_datadir), date_str)

                def _khz_hour(p: Path) -> int | None:
                    try:
                        return int(p.name[:2])
                    except ValueError:
                        return None

                def _rms_hour(p: Path) -> int | None:
                    m = re.search(r"\d{4}-\d{2}-\d{2}_(\d{2})\d{2}[—-]", p.stem)
                    return int(m.group(1)) if m else None

                def _trigger_hour(p: Path) -> int | None:
                    # parent dir: TRIGGER__YYYY-MM-DD__HH-MM
                    m = re.search(r"__(\d{2})-\d{2}$", p.parent.name)
                    return int(m.group(1)) if m else None

                # kHz: skip CFG entries, filter by hours
                file_set.hybrid_kHz = natsorted(
                    str(f)
                    for key, files in hdata._info.khz_files.items()
                    if not key.endswith("_cfg")
                    for f in files
                    if _khz_hour(f) in hours
                )
                # rms: filter by hours
                file_set.hybrid_rms = natsorted(
                    str(f)
                    for files in hdata._info.rms_files.values()
                    for f in files
                    if _rms_hour(f) in hours
                )
                # trigger: filter by hours
                file_set.hybrid_trigger = natsorted(
                    str(f)
                    for files in hdata._info.trigger_files.values()
                    for f in files
                    if _trigger_hour(f) in hours
                )
                logger.info(
                    f"Hybrid data for {date_str} {time_str}–{end_time_str} (local)"
                    f" hours={list(hours)}: "
                    f"{len(file_set.hybrid_kHz)} kHz, "
                    f"{len(file_set.hybrid_rms)} rms, "
                    f"{len(file_set.hybrid_trigger)} trigger"
                )
            except (OSError, ValueError, ImportError) as e:
                logger.warning(f"Could not discover hybrid data for {date_str}: {e}")

        logger.info(
            f"Discovered files for {filename}: {len(file_set.archive)} archives, "
            f"{len(file_set.pupitre)} pupitres, "
            f"{len(file_set.default) + len(file_set.trigger) + len(file_set.spike)} incidents, "
            f"{len(file_set.pigbrother_runlog)} pigbrother runlog, "
            f"{len(file_set.pupitre_runlog)} pupitre runlog, "
            f"{len(file_set.hybrid_kHz)} kHz, "
            f"{len(file_set.hybrid_rms)} rms, "
            f"{len(file_set.hybrid_trigger)} trigger"
        )

        return file_set

    def discover_batch(
        self,
        overview_files: list[str],
        dry_run: bool = False,
    ) -> dict[str, FileSet]:
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
            results[filename] = self.discover(overview_file, dry_run=dry_run)
        return results


# =============================================================================
# Convenience function for backward compatibility
# =============================================================================
def discover_files(
    overview_file: str,
    pupitre_datadir: str | Path = DEFAULT_DATA_DIR,
    pigbrother_datadir: str | Path = DEFAULT_PIGBROTHER_DATA_DIR,
    housing: str | None = None,
    dry_run: bool = False,
) -> dict[str, list[str]]:
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
    housing : str, optional
        Housing identifier
    dry_run : bool, optional
        If True, perform a dry run without loading data

    Returns
    -------
    Dict[str, List[str]]
        Dictionary with keys: overview, archive, pupitre, default, trigger, spike
    """
    discovery = FileDiscovery(pupitre_datadir=pupitre_datadir)
    file_set = discovery.discover(overview_file, housing=housing, dry_run=dry_run)
    return file_set.to_dict()
