import glob
import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from natsort import natsorted

from ..MagnetRun import MagnetRun
from .convert import convert_to_timestamp

logger = logging.getLogger(__name__)


def expand_input_files(
    input_patterns: list, datadir: dict, housing: str | None = None
) -> list:
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
        "Archive": "Fichiers_Archives",
        "Default": "Fichiers_Defaults",
        "Spikes": "Fichiers_Spike",
        "ManuelTrig": "Fichiers_Manuel_Trig",
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
                    # Special handling: extract site and mode from filename.
                    # Mode maps to a subdirectory (e.g. Overview→Overview,
                    # Archive→Fichiers_Archives, …).
                    parts = os.path.basename(pattern).split("_")
                    if len(parts) >= 2:
                        site = parts[0]
                        mode = parts[1]
                        mode_dir = _TDMS_MODE_DIRS.get(mode, mode)
                        candidates.append(
                            os.path.join(base_datadir, site, mode_dir, pattern)
                        )
                    else:
                        candidates.append(os.path.join(base_datadir, pattern))
                else:
                    # For .txt pupitre files: try base_datadir, then base_datadir/<site>
                    # where site is extracted from the filename (e.g. M10 from M10_…).
                    candidates.append(os.path.join(base_datadir, pattern))
                    basename = os.path.basename(pattern)
                    if basename.startswith("M") and "_" in basename:
                        site_from_name = basename.split("_")[0]
                        candidates.append(
                            os.path.join(base_datadir, site_from_name, pattern)
                        )
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
            logger.warning(f"No matches found for pattern: {pattern} (tried: {candidates})")
            expanded_files.append(pattern)

    logger.debug(f"expanded_files: {expanded_files}")
    return expanded_files


def extract_data(
    file: str, site: str, insert: str, key: str | None, dry_run: bool = False
) -> tuple:
    """Extract start and end timestamps from a data file.

    :param file: Path to the data file (.txt, .tdms, or .csv)
    :type file: str
    :param site: Site identifier (e.g., M8, M9, M10)
    :type site: str
    :param insert: Insert identifier
    :type insert: str
    :param key: Optional key to validate existence in the file
    :type key: str | None
    :param dry_run: If True, skip loading actual data and only parse timestamps from filename
    :type dry_run: bool
    :return: Tuple of (start_timestamp, end_timestamp, skip_flag) as formatted strings
    :rtype: tuple
    """
    skip = False
    extension = os.path.splitext(file)[-1]
    filename = os.path.basename(file).replace(extension, "")

    start_timestamp = 0.0
    start_ftimestamp = ""
    mrun = MagnetRun()
    match extension:
        case ".txt":
            # (site, timestamp) = filename.split("_")
            # date, time = timestamp.split("---")
            date, time = filename.replace(".txt", "").split(" - ")
            # convert ddate and dtime into a timestamp
            (start_timestamp, start_ftimestamp) = convert_to_timestamp(
                date, time, date_format="%Y.%m.%d", time_format="%H:%M:%S"
            )
            if not dry_run:
                mrun = MagnetRun.fromtxt(site, insert, file)
        case ".tdms":
            site = ""
            timestamp = ""
            res = filename.split("_")

            # regular case
            if len(res) == 3:
                (site, mode, timestamp) = res
                date, time = timestamp.split("-")
                # print(f"data={date}, time={time} (type={type(time)})")
                (start_timestamp, start_ftimestamp) = convert_to_timestamp(date, time[0:4])
            # special for default files
            elif len(res) == 4:
                (site, mode, timestamp, dmode) = res
                # print(f"mode={mode}, dmode={dmode}")
                date, time = timestamp.split("-")
                (start_timestamp, start_ftimestamp) = convert_to_timestamp(
                    date, time, "%y%m%d", "%H%M%S"
                )

            if not dry_run:
                try:
                    mrun = MagnetRun.fromtdms(site, insert, file)
                except RuntimeError as e:
                    logger.error(f"Error loading tdms file {file}: {e}")
                    skip = True
        case _:
            raise RuntimeError(f"{file}: unsupported {extension}")

    end_ftimestamp = ""
    if not dry_run and not skip:
        mdata = mrun.getMData()
        if key is not None and key not in mdata.getKeys():
            logger.warning(f"{file}: {key} not found")
            skip = True

        duration = mdata.getDuration()
        end_timestamp = datetime.fromtimestamp(start_timestamp) + pd.to_timedelta(
            duration, unit="s"
        )
        end_ftimestamp = end_timestamp.strftime("%Y-%m-%d %H:%M:%S")

    return (start_ftimestamp, end_ftimestamp, skip)


def find_files(args, file, site, date, time):
    """Generate file filter patterns for different data types.

    :param args: Command line arguments containing data directory paths
    :type args: argparse.Namespace
    :param file: Overview file path used as reference
    :type file: str
    :param site: Site identifier (e.g., M8, M9, M10)
    :type site: str
    :param date: Date string in YYMMDD format
    :type date: str
    :param time: Time string in HHMM format
    :type time: str
    :return: Tuple of filter patterns (pupitre, archive, default, trigger, spike)
    :rtype: tuple
    """
    logger.debug(f"find_files: file={file}, site={site}")

    pupitre_datadir = f"{args.pupitre_datadir}/{site}"
    pupitre_filter = f"{pupitre_datadir}/20{date[0:2]}.{date[2:4]}.{date[4:]}*.txt"
    logger.debug(f"find_files: pupitre_datadir: {pupitre_datadir}")
    logger.debug(f"find_files: pupitre_filter: {pupitre_filter}")
    logger.debug(f"find_files: file: {file}")

    extension = os.path.splitext(file)[-1]
    filename = os.path.basename(file).replace(extension, "")
    pigbrother = filename.replace("Overview", "Archive")
    archive_datadir = os.path.dirname(file).replace("Overview", "Fichiers_Archive")
    archive_filter = f"{archive_datadir}/{pigbrother.replace(time, '*.tdms')}"

    default_datadir = os.path.dirname(file).replace("Overview", "Fichiers_Default")
    trigger_datadir = os.path.dirname(file).replace("Overview", "Fichiers_Manuel_Trig")
    spike_datadir = os.path.dirname(file).replace("Overview", "Fichiers_Spike")

    default = filename.replace("Overview", "Default")
    default_filter = f"{default_datadir}/{default.replace(time, '*.tdms')}"

    trigger = filename.replace("Overview", "ManuelTrig")
    trigger_filter = f"{trigger_datadir}/{trigger.replace(time, '*.tdms')}"

    spike = filename.replace("Overview", "Spikes")
    spike_filter = f"{spike_datadir}/{spike.replace(time, '*.tdms')}"

    return pupitre_filter, archive_filter, default_filter, trigger_filter, spike_filter


def select_files(files: list, site: str, start: str, end: str):
    """Select files that fall within a specified time range.

    :param files: List of file paths to filter
    :type files: list
    :param site: Site identifier (e.g., M8, M9, M10)
    :type site: str
    :param start: Start timestamp in format '%Y-%m-%d %H:%M:%S'
    :type start: str
    :param end: End timestamp in format '%Y-%m-%d %H:%M:%S'
    :type end: str
    :return: Naturally sorted list of files within the time range
    :rtype: list
    """
    tformat = "%Y-%m-%d %H:%M:%S"
    start_time = datetime.strptime(start, tformat)
    end_time = datetime.strptime(end, tformat)
    selected = []
    for file in files:
        extension = os.path.splitext(file)[-1]
        filename = os.path.basename(file).replace(extension, "")
        match extension:
            case ".txt":
                # (site, timestamp) = filename.split("_")
                # date, time = timestamp.split("---")
                date, time = filename.replace(".txt", "").split(" - ")
                # convert ddate and dtime into a timestamp
                (start_timestamp, start_ftimestamp) = convert_to_timestamp(
                    date, time, date_format="%Y.%m.%d", time_format="%H:%M:%S"
                )

            case ".tdms":
                site = ""
                timestamp = ""
                res = filename.split("_")

                # regular case
                if len(res) == 3:
                    (site, mode, timestamp) = res
                    date, time = timestamp.split("-")
                    # print(f"data={date}, time={time} (type={type(time)})")
                    (start_timestamp, start_ftimestamp) = convert_to_timestamp(date, time[0:4])
                # special for default files
                elif len(res) == 4:
                    (site, mode, timestamp, dmode) = res
                    # print(f"mode={mode}, dmode={dmode}")
                    date, time = timestamp.split("-")
                    (start_timestamp, start_ftimestamp) = convert_to_timestamp(
                        date, time, "%y%m%d", "%H%M%S"
                    )

        # extra treatment for pupitre in case pupitre ends before end_time but starts before start_time
        if extension == ".txt":
            res = extract_data(file, site=site, insert=None, key=None)
            start_time_file = datetime.strptime(res[0], tformat)
            end_time_file = datetime.strptime(res[1], tformat)

            if start_time >= start_time_file and end_time_file < end_time:
                logger.debug(f"tdms overlap txt file: start {file}")
                # how to get timerange for pupitre that starts at start_time?
            if start_time < start_time_file and end_time_file >= end_time:
                logger.debug(f"tdms overlap txt file: end {file}")
                # how to get timerange for pupitre that starts at start_time?
            if start_time >= start_time_file and end_time < end_time_file:
                logger.debug(f"tdms included into txt file: {file}")

        if datetime.strptime(start_ftimestamp, tformat) >= start_time:
            res = extract_data(file, site=site, insert=None, key=None)
            start_time_file = datetime.strptime(res[0], tformat)
            end_time_file = datetime.strptime(res[1], tformat)
            # print(
            #     f"{file}: start_time_file={start_time_file} end_time_file={end_time_file}, start_time={start_time}, end_time={end_time}",
            #     flush=True,
            # )
            if start_time_file >= start_time and end_time_file < end_time:
                selected.append(file)
                if extension == ".txt":
                    logger.debug(f"selected tdms file: {file}")
            # print(f"Difference: {timestamp - itimestamp} seconds")

    # print(f"selected: {selected}", flush=True)
    if selected:
        return natsorted(selected)
    return selected


def load_df(file, site, insert, group, keys) -> tuple:
    """Load data from a file into a pandas DataFrame.

    :param file: Path to the data file (.txt or .tdms)
    :type file: str
    :param site: Site identifier (e.g., M8, M9, M10)
    :type site: str
    :param insert: Insert identifier
    :type insert: str
    :param group: Data group name for TDMS files
    :type group: str
    :param keys: List of data keys to extract
    :type keys: list
    :return: Tuple of (DataFrame with data, start timestamp)
    :rtype: tuple
    """
    extension = os.path.splitext(file)[-1]

    df = pd.DataFrame()
    # t0 = datetime.now()
    match extension:
        case ".txt":
            mrun = MagnetRun.fromtxt(site, insert, file)
            mdata = mrun.getMData()
            t0 = mdata.Data["timestamp"].iloc[0]
            df = pd.DataFrame(mdata.getData(["t", "timestamp"] + keys))
        case ".tdms":
            mrun = MagnetRun.fromtdms(site, insert, file)
            mdata = mrun.getMData()
            if keys[0] not in mdata.Groups[group]:
                logger.warning(f"load_df tdms {group}/{keys[0]} not found in {mdata.FileName}")
                """
                print(f"available keys are: {mdata.Groups[group].keys()}")
                for key in mdata.Groups[group]:
                    print(f"{group}/{key}: {mdata.Groups[group][key]}")
                # raise RuntimeError(f"{group}/{keys[0]} not found in {mdata.FileName}")
                """
                return df, t0
            t0 = mdata.Groups[group][keys[0]]["wf_start_time"]
            dt = mdata.Groups[group][keys[0]]["wf_increment"]
            t_offset = mdata.Groups[group][keys[0]]["wf_start_offset"]
            logger.debug(f"{file}: t0: {t0}, dt: {dt}, t_offset: {t_offset}")
            df = pd.DataFrame(mdata.getTdmsData(group, keys))
            df["timestamp"] = [
                np.datetime64(t0).astype(datetime) + timedelta(0, i * dt + t_offset)
                for i in df.index.to_list()
            ]
    return df, t0


def load_data(files, site, insert, group, keys) -> list[pd.DataFrame]:
    """Load data from multiple files into a list of DataFrames.

    :param files: List of file paths to load
    :type files: list
    :param site: Site identifier (e.g., M8, M9, M10)
    :type site: str
    :param insert: Insert identifier
    :type insert: str
    :param group: Data group name for TDMS files
    :type group: str
    :param keys: List of data keys to extract
    :type keys: list
    :return: List of DataFrames containing data from each file
    :rtype: list[pd.DataFrame]
    """
    df_ = []
    for file in files:
        df, t0 = load_df(file, site, insert, group, keys)
        if not df.empty:
            df_.append(df)
    return df_


def merge_data(df_list: list) -> pd.DataFrame:
    """Merge multiple DataFrames into a single DataFrame.

    :param df_list: List of DataFrames to merge
    :type df_list: list
    :return: Concatenated DataFrame if multiple DataFrames, otherwise the single DataFrame
    :rtype: pd.DataFrame
    """
    if len(df_list) > 1:
        return pd.concat(df_list)
    return df_list[0]
