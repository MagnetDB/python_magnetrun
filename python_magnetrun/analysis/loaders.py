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
from typing import cast

import pandas as pd
from natsort import natsorted

from ..magnetdata_base import DataType
from ..runlogs.pigbrother import PIGBROTHER_LOG_FILENAME
from ..utils.files import (
    TIMESTAMP_FORMAT,
    extract_data,
    find_files,
    select_files,
)
from .config import (
    DEFAULT_DATA_DIR,
    DEFAULT_PIGBROTHER_DATA_DIR,
)

# Module logger
logger = logging.getLogger("python_magnetrun.analysis.loaders")


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
                from python_magnetrun.magnetdata_tdms import TdmsMagnetData

                df = pd.DataFrame(
                    cast(TdmsMagnetData, mdata).getTdmsData(group=group, channel=None)
                )
            else:
                available = set(mdata.Keys)
                desired = [k for k in (keys or []) + ["t", "timestamp"] if k in available]
                missing = [k for k in (keys or []) if k not in available]
                if missing:
                    logger.warning(f"load_files_data: {file}: requested keys not available: {missing}")
                df = pd.DataFrame(mdata.getData(desired))

            df["t"] = df["t"] + shift

            df_list.append(df)
            logger.debug(f"{file} t0={t0_file}, shift={shift:.3f}s, rows={len(df)}")

        except (OSError, ValueError, RuntimeError, KeyError, UnicodeDecodeError) as e:
            logger.error(f"load_files_data: failed to load {file}: {e}")

    if not df_list:
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)


def merge_data(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge a list of DataFrames into one.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        DataFrames to merge. Must be non-empty.

    Returns
    -------
    pd.DataFrame
        The single DataFrame unchanged, or a concatenation of all inputs.

    Raises
    ------
    ValueError
        When *dfs* is empty.
    """
    if not dfs:
        raise ValueError("merge_data: list is empty")
    if len(dfs) == 1:
        return dfs[0]
    return pd.concat(dfs, ignore_index=True)


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

    # ------------------------------------------------------------------
    # Private helpers for discover()
    # ------------------------------------------------------------------

    def _resolve_overview_path(
        self,
        overview_file: str,
        housing: str | None,
        filename: str,
        basename: str,
    ) -> tuple[str, str]:
        """Resolve a bare filename to a full path under pigbrother_datadir.

        Returns (resolved_overview, overview_dir).  When *overview_file*
        already contains a directory component both values are returned
        unchanged.
        """
        overview_dir = os.path.dirname(overview_file)
        resolved_overview = overview_file
        if not overview_dir:
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
                    overview_dir = ""
        return resolved_overview, overview_dir

    def _parse_overview_filename(
        self,
        filename: str,
        housing: str | None,
    ) -> tuple[str, str, str] | None:
        """Extract (housing, date, time) from an overview filename stem.

        Returns ``None`` and logs an error when the stem cannot be parsed.
        """
        parts = filename.split("_")
        logger.info(f"Filename parts: {parts}")
        if len(parts) < 3:
            logger.error(f"Cannot parse filename: {filename}")
            return None
        file_housing = parts[0]
        timestamp = parts[2]
        resolved_housing = housing if housing is not None else file_housing
        if housing is None:
            logger.info(f"Extracted housing from filename: {resolved_housing}")
        try:
            date, time = timestamp.split("-")
        except ValueError:
            logger.error(f"Cannot split timestamp {timestamp!r} in {filename}")
            return None
        return resolved_housing, date, time

    def _select_related_files(
        self,
        overview_path: str,
        housing: str,
        date: str,
        time: str,
        start: str,
        end: str,
    ) -> FileSet:
        """Glob and time-filter all file types related to *overview_path*.

        Returns a :class:`FileSet` with ``overview`` left empty (the caller
        fills it after resolving the path).
        """
        filters = find_files(
            overview_path,
            housing,
            date,
            time,
            pupitre_datadir=self.pupitre_datadir,
        )
        pupitre_f, archive_f, default_f, trigger_f, spike_f = filters
        logger.info(
            f"File patterns: pupitre={pupitre_f} archive={archive_f} "
            f"default={default_f} trigger={trigger_f} spike={spike_f}"
        )

        file_set = FileSet()
        file_set.pupitre = select_files(glob.glob(pupitre_f), housing, start, end)
        file_set.archive = select_files(glob.glob(archive_f), housing, start, end)
        # Incident files are intentionally short; disable the min-duration guard.
        file_set.default = select_files(
            glob.glob(default_f), housing, start, end, min_duration_seconds=0.0
        )
        file_set.trigger = select_files(
            glob.glob(trigger_f), housing, start, end, min_duration_seconds=0.0
        )
        file_set.spike = select_files(
            glob.glob(spike_f), housing, start, end, min_duration_seconds=0.0
        )
        logger.info(
            f"Selected: pupitre={file_set.pupitre} archive={file_set.archive} "
            f"default={file_set.default} trigger={file_set.trigger} spike={file_set.spike}"
        )
        return file_set

    def _discover_runlogs(
        self,
        file_set: FileSet,
        start: str,
        end: str,
    ) -> None:
        """Populate *file_set* pigbrother and pupitre run-log fields in place."""
        pb_log = self.pigbrother_runlog_dir / PIGBROTHER_LOG_FILENAME
        if pb_log.exists():
            file_set.pigbrother_runlog = [str(pb_log)]
            logger.info(f"file_set.pigbrother_runlog: {file_set.pigbrother_runlog}")
        else:
            logger.debug(f"Pigbrother runlog not found at {pb_log}")

        if self.pupitre_runlog_dir is not None and start and end:
            from ..runlogs.pupitre import discover_pupitre_runlogs

            file_set.pupitre_runlog = discover_pupitre_runlogs(
                self.pupitre_runlog_dir,
                start_date=start[:10],
                end_date=end[:10],
            )
            logger.info(f"file_set.pupitre_runlog: {file_set.pupitre_runlog}")

    def _discover_hybrid_data(
        self,
        file_set: FileSet,
        housing: str,
        filename: str,
        resolved_overview: str,
        start: str,
        end: str,
    ) -> None:
        """Populate *file_set* hybrid kHz/RMS/trigger fields for M8. No-op for other housings."""
        if housing != "M8" or self.hybrid_datadir is None or not start or not end:
            return

        _ts_part = filename.split("_")[2]
        _date_part, _time_part = _ts_part.split("-")
        _local_dt = datetime.strptime(_date_part + _time_part, "%y%m%d%H%M")
        date_str = _local_dt.strftime("%Y-%m-%d")
        time_str = _local_dt.strftime("%H:%M")

        from python_magnetrun.MagnetRun import MagnetRun as _MR

        _mrun = _MR.fromtdms(housing, "", resolved_overview)
        _duration = _mrun.getMData().getDuration()
        _local_end_dt = _local_dt + timedelta(seconds=_duration)
        end_time_str = _local_end_dt.strftime("%H:%M")
        hours = range(_local_dt.hour, _local_end_dt.hour + 1)

        # kHz/RMS/trigger filenames encode UTC hours; convert French local hours
        # to UTC so the file-name filter matches correctly (handles CET/CEST).
        from zoneinfo import ZoneInfo

        _tz_paris = ZoneInfo("Europe/Paris")
        _tz_utc = ZoneInfo("UTC")
        _local_date = _local_dt.date()
        utc_hours = {
            datetime(_local_date.year, _local_date.month, _local_date.day,
                     h, 0, 0, tzinfo=_tz_paris).astimezone(_tz_utc).hour
            for h in hours
        }

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

            def _trigger_dir_hour(p: Path) -> int | None:
                m = re.search(r"__(\d{2})-\d{2}$", p.name)
                return int(m.group(1)) if m else None

            def _vprocess_hour(p: Path) -> int | None:
                m = re.match(r"^\d{8}_(\d{2})", p.name)
                return int(m.group(1)) if m else None

            file_set.hybrid_kHz = cast(
                list[str],
                natsorted(
                    str(f)
                    for key, files in hdata._info.khz_files.items()
                    if not key.endswith("_cfg")
                    for f in files
                    if _khz_hour(f) in utc_hours
                ),
            )
            file_set.hybrid_rms = cast(
                list[str],
                natsorted(
                    str(f)
                    for files in hdata._info.rms_files.values()
                    for f in files
                    if _rms_hour(f) in utc_hours
                ),
            )
            file_set.hybrid_trigger = cast(
                list[str],
                natsorted(
                    str(d)
                    for dirs in hdata._info.trigger_dirs.values()
                    for d in dirs
                    if _trigger_dir_hour(d) in utc_hours
                ),
            )
            file_set.hybrid_vprocess = cast(
                list[str],
                natsorted(
                    str(f)
                    for f in hdata._info.vprocess_files
                    if _vprocess_hour(f) in utc_hours
                ),
            )
            logger.info(
                f"Hybrid data for {date_str} {time_str}–{end_time_str} (local)"
                f" hours={list(hours)}: "
                f"{len(file_set.hybrid_kHz)} kHz, "
                f"{len(file_set.hybrid_rms)} rms, "
                f"{len(file_set.hybrid_trigger)} trigger dirs, "
                f"{len(file_set.hybrid_vprocess)} vprocess"
            )
        except (OSError, ValueError, ImportError) as e:
            logger.warning(f"Could not discover hybrid data for {date_str}: {e}")

    # ------------------------------------------------------------------
    # Public discovery methods
    # ------------------------------------------------------------------

    def discover(
        self,
        overview_file: str,
        housing: str | None = None,
        dry_run: bool = False,
    ) -> FileSet:
        """Discover all files related to an overview file.

        Parameters
        ----------
        overview_file : str
            Path to the overview TDMS file
        housing : str, optional
            Housing identifier (extracted from filename if not provided)
        dry_run : bool, optional
            When True skip full file loads (timestamps from filename only)

        Returns
        -------
        FileSet
            Container with all discovered related files
        """
        logger.info(f"Discovering files for overview: {overview_file}, housing={housing}")

        extension = os.path.splitext(overview_file)[-1]
        basename = os.path.basename(overview_file)
        filename = basename.replace(extension, "")
        logger.info(f"discover overview file: {overview_file} (filename={filename})")

        resolved_overview, overview_dir = self._resolve_overview_path(
            overview_file, housing, filename, basename
        )
        logger.info(f"discover overview_dir={overview_dir}, resolved_overview={resolved_overview}")

        parsed = self._parse_overview_filename(filename, housing)
        if parsed is None:
            return FileSet(overview=[f"{overview_dir}/{overview_file}"])
        housing, date, time = parsed
        logger.info(f"Extracted date={date}, time={time} from filename")

        if housing == "M8":
            logger.info(f"Overview file {overview_file} has housing M8, hybrid data will be discovered")

        start, end, skip = extract_data(
            resolved_overview, housing, site="", key=None, dry_run=dry_run
        )
        logger.info(f"Overview file time range: start={start}, end={end}, skip={skip}")

        if skip or not start or not end:
            logger.warning(f"Could not extract time range from {overview_file}")
            return FileSet(overview=[resolved_overview])

        file_set = self._select_related_files(
            resolved_overview, housing, date, time, start, end
        )
        file_set.overview = [resolved_overview]

        self._discover_runlogs(file_set, start, end)
        self._discover_hybrid_data(file_set, housing, filename, resolved_overview, start, end)

        logger.info(
            f"Discovered files for {filename}: {len(file_set.archive)} archives, "
            f"{len(file_set.pupitre)} pupitres, "
            f"{len(file_set.default) + len(file_set.trigger) + len(file_set.spike)} incidents, "
            f"{len(file_set.pigbrother_runlog)} pigbrother runlog, "
            f"{len(file_set.pupitre_runlog)} pupitre runlog, "
            f"{len(file_set.hybrid_kHz)} kHz, "
            f"{len(file_set.hybrid_rms)} rms, "
            f"{len(file_set.hybrid_trigger)} trigger dirs, "
            f"{len(file_set.hybrid_vprocess)} vprocess"
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
