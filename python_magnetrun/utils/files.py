from __future__ import unicode_literals
import os
from datetime import datetime, timedelta
from numpy.random import f
import pandas as pd
import numpy as np
from natsort import natsorted

from ..MagnetRun import MagnetRun
from .convert import convert_to_timestamp


def extract_data(
    file: str, site: str, insert: str, key: str | None, dry_run: bool = False
) -> tuple:
    skip = False
    extension = os.path.splitext(file)[-1]
    filename = os.path.basename(file).replace(extension, "")

    start_timestamp = float()
    start_ftimestamp = str()
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
            site = str()
            timestamp = str()
            res = filename.split("_")

            # regular case
            if len(res) == 3:
                (site, mode, timestamp) = res
                date, time = timestamp.split("-")
                # print(f"data={date}, time={time} (type={type(time)})")
                (start_timestamp, start_ftimestamp) = convert_to_timestamp(
                    date, time[0:4]
                )
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
                    print(f"Error loading tdms file {file}: {e}")
                    skip = True
        case _:
            raise RuntimeError(f"{file}: unsupported {extension}")

    end_ftimestamp = str()
    if not dry_run and not skip:
        mdata = mrun.getMData()
        if key is not None:
            if key not in mdata.getKeys():
                print(f"{file}: {key} not found")
                skip = True

        duration = mdata.getDuration()
        end_timestamp = datetime.fromtimestamp(start_timestamp) + pd.to_timedelta(
            duration, unit="s"
        )
        end_ftimestamp = end_timestamp.strftime("%Y-%m-%d %H:%M:%S")

    # print(
    #     f"extract_data: file={file}, start={start_ftimestamp}, end={end_ftimestamp}, skip={skip }",
    #     flush=True,
    # )
    return (start_ftimestamp, end_ftimestamp, skip)


def find_files(args, file, site, date, time):
    print(f"find_files: file={file}, site={site}", flush=True)

    pupitre_datadir = f"{args.pupitre_datadir}/{site}"
    pupitre_filter = f"{pupitre_datadir}/20{date[0:2]}.{date[2:4]}.{date[4:]}*.txt"
    print(f"find_files: pupitre_datadir: {pupitre_datadir}")
    print(f"find_files: pupitre_filter: {pupitre_filter}", flush=True)
    print(f"find_files: file: {file}", flush=True)

    extension = os.path.splitext(file)[-1]
    filename = os.path.basename(file).replace(extension, "")
    pigbrother = filename.replace("Overview", "Archive")
    archive_datadir = os.path.dirname(file).replace("Overview", "Fichiers_Archive")
    archive_filter = f"{archive_datadir}/{pigbrother.replace(time,'*.tdms')}"

    default_datadir = os.path.dirname(file).replace("Overview", "Fichiers_Default")
    trigger_datadir = os.path.dirname(file).replace("Overview", "Fichiers_Manuel_Trig")
    spike_datadir = os.path.dirname(file).replace("Overview", "Fichiers_Spike")

    default = filename.replace("Overview", "Default")
    default_filter = f"{default_datadir}/{default.replace(time,'*.tdms')}"

    trigger = filename.replace("Overview", "ManuelTrig")
    trigger_filter = f"{trigger_datadir}/{trigger.replace(time,'*.tdms')}"

    spike = filename.replace("Overview", "Spikes")
    spike_filter = f"{spike_datadir}/{spike.replace(time,'*.tdms')}"

    return pupitre_filter, archive_filter, default_filter, trigger_filter, spike_filter


def select_files(files: list, site: str, start: str, end: str):
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
                site = str()
                timestamp = str()
                res = filename.split("_")

                # regular case
                if len(res) == 3:
                    (site, mode, timestamp) = res
                    date, time = timestamp.split("-")
                    # print(f"data={date}, time={time} (type={type(time)})")
                    (start_timestamp, start_ftimestamp) = convert_to_timestamp(
                        date, time[0:4]
                    )
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
                print("tdms overlap txt file: start", file, flush=True)
                # how to get timerange for pupitre that starts at start_time?
            if start_time < start_time_file and end_time_file >= end_time:
                print("tdms overlap txt file: end", file, flush=True)
                # how to get timerange for pupitre that starts at start_time?
            if start_time >= start_time_file and end_time < end_time_file:
                print("tdms included into txt file:", file, flush=True)

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
                    print(f"selected tdms file: {file}", flush=True)
            # print(f"Difference: {timestamp - itimestamp} seconds")

    # print(f"selected: {selected}", flush=True)
    if selected:
        return natsorted(selected)
    return selected


def load_df(file, site, insert, group, keys) -> tuple:
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
                print(f"load_df tdms {group}/{keys[0]} not found in {mdata.FileName}")
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
            print(f"{file}: t0: {t0}, dt: {dt}, t_offset: {t_offset}")
            df = pd.DataFrame(mdata.getTdmsData(group, keys))
            df["timestamp"] = [
                np.datetime64(t0).astype(datetime) + timedelta(0, i * dt + t_offset)
                for i in df.index.to_list()
            ]
    return df, t0


def load_data(files, site, insert, group, keys) -> list[pd.DataFrame]:
    df_ = []
    for file in files:
        df, t0 = load_df(file, site, insert, group, keys)
        if not df.empty:
            df_.append(df)
    return df_


def merge_data(df_list: list) -> pd.DataFrame:
    if len(df_list) > 1:
        return pd.concat(df_list)
    return df_list[0]


# TODO use MagnetData instead of files
