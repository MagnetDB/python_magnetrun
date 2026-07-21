from __future__ import annotations

import contextlib
import glob
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import pandas as pd
from natsort import natsorted

from .timestamps import parse_filename_timestamp

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".tdms", ".txt", ".csv"})

TIMESTAMP_FORMAT: str = "%Y-%m-%d %H:%M:%S"
"""Standard timestamp format used for file selection."""

# Subdirectory names used in the TDMS data tree.
DIR_ARCHIVE: str = "Fichiers_Archive"
DIR_DEFAULT: str = "Fichiers_Default"
DIR_TRIGGER: str = "Fichiers_Manuel_Trig"
DIR_SPIKE: str = "Fichiers_Spike"


def get_extension(path: str) -> str:
    """Return the lower-cased file extension of *path* (e.g. ``".tdms"``)."""
    return os.path.splitext(path)[-1].lower()


@contextlib.contextmanager
def _open_text_with_fallback(path: str | Path):
    """Open a text file as UTF-8, falling back to Latin-1 on decode error."""
    try:
        with open(path, encoding="utf-8") as probe:
            probe.read(1)
    except UnicodeDecodeError:
        with open(path, encoding="latin-1", errors="replace") as f:
            yield f
    else:
        with open(path, encoding="utf-8") as f:
            yield f


def expand_input_files(
    input_patterns: list[str], datadir: dict[str, str], housing: str | None = None
) -> list[str]:
    """Expand glob patterns in input file arguments.

    Search order for patterns without an explicit directory component:
    1. Current working directory.
    2. ``datadir[extension]`` (base data directory for the extension).
    3. ``datadir[extension]/housing`` (housing subdirectory, when *housing* is given).

    :param input_patterns: List of file patterns to expand
    :type input_patterns: list
    :param datadir: Dictionary mapping file extensions to their base directories
    :type datadir: dict
    :param housing: Optional housing identifier (e.g. ``'M9'``) appended to the
        extension-specific data directory as a fallback search location
    :type housing: str or None
    :return: List of expanded file paths
    :rtype: list
    """
    # Mapping from TDMS mode name (2nd underscore-part of filename) to subdirectory name.
    _TDMS_MODE_DIRS: dict[str, str] = {
        "Overview": "Overview",
        "Archive": DIR_ARCHIVE,
        "Default": DIR_DEFAULT,
        "Spikes": DIR_SPIKE,
        "ManuelTrig": DIR_TRIGGER,
    }

    logger.debug(f"Expanding input files ({input_patterns})...")
    expanded_files = []
    for pattern in input_patterns:
        extension = os.path.splitext(pattern)[-1]
        logger.debug(f"pattern: {pattern}, extension: {extension}")

        # Pattern has an explicit directory — use it directly.
        if os.path.dirname(pattern):
            search_pattern = pattern
            matches = glob.glob(search_pattern)
            if matches:
                logger.debug(f"matches: {matches}")
                expanded_files.extend(matches)
            else:
                logger.warning(f"No matches found for pattern: {pattern}")
                expanded_files.append(pattern)
            continue

        # Build candidate search patterns in priority order.
        candidates: list[str] = []

        # 1. Current working directory.
        candidates.append(os.path.join(os.getcwd(), pattern))

        # 2 & 3. Extension-specific datadir, with and without housing/site subdir.
        if extension in datadir:
            base_datadir = datadir[extension]
            if base_datadir:
                if extension == ".tdms":
                    # Special handling: extract housing and mode from filename.
                    # Mode maps to a subdirectory (e.g. Overview→Overview,
                    # Archive→DIR_ARCHIVE, …).
                    parts = os.path.basename(pattern).split("_")
                    if len(parts) >= 2:
                        housing_from_file = parts[0]
                        mode = parts[1]
                        mode_dir = _TDMS_MODE_DIRS.get(mode, mode)
                        candidates.append(
                            os.path.join(
                                base_datadir, housing_from_file, mode_dir, pattern
                            )
                        )
                        if (
                            housing_from_file != housing
                            and housing
                            and housing not in ("notdefined", "")
                        ):
                            logger.warning(
                                f"Housing mismatch for pattern '{pattern}': "
                                f"extracted '{housing_from_file}' vs provided '{housing}'"
                            )
                    else:
                        candidates.append(os.path.join(base_datadir, pattern))
                else:
                    # For .txt pupitre files: try base_datadir, then base_datadir/<housing>
                    # where housing is extracted from the filename (e.g. M10 from M10_…).
                    candidates.append(os.path.join(base_datadir, pattern))
                    basename = os.path.basename(pattern)
                    if basename.startswith("M") and "_" in basename:
                        housing_from_name = basename.split("_")[0]
                        candidates.append(
                            os.path.join(base_datadir, housing_from_name, pattern)
                        )
                        if (
                            housing_from_name != housing
                            and housing
                            and housing not in ("notdefined", "")
                        ):
                            logger.warning(
                                f"Housing mismatch for pattern '{pattern}': "
                                f"extracted '{housing_from_name}' vs provided '{housing}'"
                            )
                    else:
                        if housing and housing not in ("notdefined", ""):
                            candidates.append(
                                os.path.join(base_datadir, housing, pattern)
                            )

        logger.debug(f"candidates: {candidates}")

        matched = False
        for candidate in candidates:
            matches = glob.glob(candidate)
            if matches:
                logger.debug(f"matched '{candidate}': {matches}")
                expanded_files.extend(matches)
                matched = True
                break

        if not matched:
            logger.warning(
                f"No matches found for pattern: {pattern} (tried: {candidates})"
            )
            expanded_files.append(pattern)

    logger.debug(f"expanded_files: {expanded_files}")
    return expanded_files


# =============================================================================
# Metadata-only helpers (no data arrays loaded)
# =============================================================================


def _tdms_end_from_properties(
    file: str,
    start_timestamp: float,
    start_ftimestamp: str,
    check_all_groups: bool = False,
) -> str:
    """Return end timestamp by reading actual sample counts from TDMS channel properties.

    Counts actual samples via ``read_data_chunks()`` (not the ``wf_samples``
    property, which may be wrong for truncated files) and multiplies by
    ``wf_increment``.  Returns an empty string when no usable channel is found.
    """
    from nptdms import TdmsFile

    try:
        with TdmsFile.open(file) as tdms:
            first_duration: float | None = None
            for group in tdms.groups():
                if group.name == "Infos":
                    continue
                group_duration: float | None = None
                for ch in group.channels():
                    p = ch.properties
                    logger.debug(f"checking channel {ch.path} with properties {p}")
                    if "wf_increment" not in p:
                        continue
                    actual = sum(len(chunk) for chunk in ch.data_chunks())
                    wf_prop = p.get("wf_samples")
                    if wf_prop is not None and int(wf_prop) != actual:
                        logger.info(
                            f"{file} {ch.path}: wf_samples={wf_prop} != actual={actual} — using actual count"
                        )
                    duration = actual * float(p["wf_increment"])
                    logger.debug(
                        f"{file} {ch.path}: actual={actual} samples, duration={duration:.3f}s"
                    )
                    if group_duration is None or duration > group_duration:
                        group_duration = duration

                if group_duration is None:
                    continue

                if first_duration is None:
                    first_duration = group_duration
                    if not check_all_groups:
                        break
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
            pos -= 1
            f.seek(pos)
            while pos > 0 and f.read(1) in (b"\n", b"\r", b" "):
                pos -= 1
                f.seek(pos)
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
    """Extract timestamp range and metadata from a data file.

    Parses the filename to determine the start timestamp, and optionally loads
    the file to verify keys and get exact end timestamp.

    When *key* is ``None`` and *dry_run* is ``False``, uses lightweight
    metadata-only paths (TDMS channel properties / last line of pupitre file)
    to compute the end timestamp without loading full data arrays.

    Parameters
    ----------
    file : str
        Path to the data file (.txt or .tdms)
    housing : str
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
    """
    from ..MagnetRun import MagnetRun  # lazy import to avoid circular dependency

    logger.info(
        f"extract_data: file={file}, housing={housing}, site={site}, key={key}, dry_run={dry_run}"
    )
    extension = os.path.splitext(file)[-1]

    start_timestamp: float = 0.0
    start_ftimestamp: str = ""
    end_ftimestamp: str = ""
    skip: bool = False
    mrun = None

    dt = parse_filename_timestamp(file)
    if dt is not None:
        start_timestamp = dt.timestamp()
        start_ftimestamp = dt.strftime(TIMESTAMP_FORMAT)

    if extension == ".txt":
        logger.info(f"Parsed pupitre filename: start_ftimestamp={start_ftimestamp}")
        if not dry_run:
            if key is None:
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
        logger.info(f"Parsed TDMS filename: start_ftimestamp={start_ftimestamp}")
        if not dry_run:
            if key is None:
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
        if key is not None and key not in mdata.getKeys():
            skip = True
        duration = mdata.getDuration()
        end_dt = datetime.fromtimestamp(start_timestamp) + timedelta(seconds=duration)
        end_ftimestamp = end_dt.strftime(TIMESTAMP_FORMAT)
        logger.info(
            f"{file}: start={start_ftimestamp}, end={end_ftimestamp}, duration={duration}s"
        )

    return (start_ftimestamp, end_ftimestamp, skip)


# =============================================================================
# Find files function
# =============================================================================
def find_files(
    overview_file: str,
    housing: str,
    date: str,
    time: str,
    pupitre_datadir: str | Path = ".",
) -> tuple[str, str, str, str, str]:
    """Build glob patterns to find files related to an overview file.

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
    """
    logger.info(
        f"find_files: overview_file={overview_file}, housing={housing}, date={date}, time={time}"
    )
    pupitre_datadir = Path(pupitre_datadir)

    pupitre_site_dir = pupitre_datadir / housing
    pupitre_filter = str(
        pupitre_site_dir / f"20{date[0:2]}.{date[2:4]}.{date[4:]}*.txt"
    )

    extension = os.path.splitext(overview_file)[-1]
    filename = os.path.basename(overview_file).replace(extension, "")
    overview_dir = os.path.dirname(overview_file)

    pigbrother = filename.replace("Overview", "Archive")
    archive_datadir = overview_dir.replace("Overview", DIR_ARCHIVE)
    archive_filter = f"{archive_datadir}/{pigbrother.replace(time, '*.tdms')}"

    default_datadir = overview_dir.replace("Overview", DIR_DEFAULT)
    trigger_datadir = overview_dir.replace("Overview", DIR_TRIGGER)
    spike_datadir = overview_dir.replace("Overview", DIR_SPIKE)

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
    """Filter files by timestamp range.

    Selects files whose time range overlaps with the specified range.

    Parameters
    ----------
    files : list[str]
        List of file paths to filter
    housing : str
        Housing identifier
    start : str
        Start timestamp (TIMESTAMP_FORMAT)
    end : str
        End timestamp (TIMESTAMP_FORMAT)
    min_duration_seconds : float, optional
        Files shorter than this threshold are discarded (default 30 s).
        Pass 0.0 for incident files which are legitimately short captures.

    Returns
    -------
    list[str]
        Filtered and naturally sorted list of files
    """
    natsortedfiles = cast("list[str]", natsorted(files))
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

    return cast("list[str]", natsorted(selected)) if selected else []


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
    """Load a single file into a pandas DataFrame.

    Handles both .txt (pupitre) and .tdms (pigbrother) files.
    Adds ``t`` and ``timestamp`` columns for time alignment.

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
    keys : list[str] or None
        Column/channel names to load

    Returns
    -------
    tuple[pd.DataFrame, datetime | None]
        (dataframe, start_time). Returns (empty DataFrame, None) if loading fails.
    """
    from ..MagnetRun import MagnetRun  # lazy import to avoid circular dependency

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

            channels = list(mdata.getData(group).keys())
            logger.debug(f"channels={channels}")
            from ..magnetdata_tdms import TdmsMagnetData

            df = cast(TdmsMagnetData, mdata).getTdmsData(group, keys)

            first_key = channels[0] if keys is None or not keys else keys[0]
            logger.debug(f"first_key: {first_key}")
            if keys is not None and keys and keys[0] not in mdata.Groups.get(group, {}):
                logger.debug(f"{group}/{keys[0]} not found in {mdata.FileName}")
                return df, t0

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
    """Load multiple files and return list of DataFrames (empty frames excluded).

    Parameters
    ----------
    files : list[str]
        List of file paths to load
    housing : str
        Housing identifier
    site : str
        Site identifier
    group : str
        TDMS group name
    keys : list[str] or None
        Column/channel names to load

    Returns
    -------
    list[pd.DataFrame]
        List of loaded DataFrames
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
    """Merge multiple DataFrames into one by vertical concatenation.

    Parameters
    ----------
    df_list : list[pd.DataFrame]
        List of DataFrames to merge

    Returns
    -------
    pd.DataFrame
        Merged DataFrame

    Raises
    ------
    ValueError
        If df_list is empty
    """
    if not df_list:
        raise ValueError("Cannot merge empty list of DataFrames")
    if len(df_list) == 1:
        return df_list[0]
    return pd.concat(df_list, ignore_index=True)
