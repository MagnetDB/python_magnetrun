import os
from datetime import datetime, timedelta
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsorted
from .MagnetRun import MagnetRun

from .processing.trends import trends
from .utils.convert import convert_to_timestamp

from .processing.correlations import lag_correlation
from .signature import Signature


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="+", help="enter input file tdms")
    parser.add_argument(
        "--pupitre_datadir",
        help="enter pupitre datadir (default srvdata)",
        type=str,
        default="srvdata",
    )
    parser.add_argument(
        "--key",
        help="choose key",
        choices=["Référence_GR1", "Référence_GR2"],
        type=str,
        default="Référence_GR1",
    )
    parser.add_argument(
        "--tkey",
        help="choose tkey",
        choices=["t", "timestamp"],
        type=str,
        default="t",
    )
    parser.add_argument("--debug", help="activate debug", action="store_true")
    parser.add_argument("--save", help="save graphs (png format)", action="store_true")
    parser.add_argument(
        "--show", help="display graphs (X11 required)", action="store_true"
    )
    parser.add_argument(
        "--window", help="stopping criteria for nlopt", type=int, default=10
    )
    parser.add_argument(
        "--synchronize", help="rolling window size", action="store_true"
    )
    return parser.parse_args()


def setup():
    color_dict = {"U": "red", "P": "green", "D": "blue"}
    channels_dict = {"Référence_GR1": "Courant_GR1", "Référence_GR2": "Courant_GR2"}
    uchannels_dict = {
        "Référence_GR1": [
            "ALL_internes",
            "Interne1",
            "Interne2",
            "Interne3",
            "Interne4",
            "Interne5",
            "Interne6",
            "Interne7",
        ],
        "Référence_GR2": ["ALL_externes", "Externe1", "Externe2"],
    }
    pupitre_dict = {
        "M9": {"Référence_GR1": "IH", "Référence_GR2": "IB"},
        "M8": {"Référence_GR1": "IB", "Référence_GR2": "IH"},
        "M10": {"Référence_GR1": "IB", "Référence_GR2": "IH"},
    }
    upupitre_dict = {
        "M9": {"Référence_GR1": ["UH"], "Référence_GR2": ["UB", "Ucoil15", "Ucoil16"]},
        "M8": {"Référence_GR2": ["UH"], "Référence_GR1": ["UB", "Ucoil15", "Ucoil16"]},
        "M10": {"Référence_GR2": ["UH"], "Référence_GR1": ["UB", "Ucoil15", "Ucoil16"]},
    }
    threshold_dict = {
        "Référence_GR1": 0.5,
        "Courant_GR1": 1,
        "ALL_internes": 0.1,
        "Interne1": 1.0e-2,
        "Interne2": 1.0e-2,
        "Interne3": 1.0e-2,
        "Interne4": 1.0e-2,
        "Interne5": 1.0e-2,
        "Interne6": 1.0e-2,
        "Interne7": 1.0e-2,
        "Référence_GR2": 0.5,
        "Courant_GR2": 1,
        "ALL_externes": 0.1,
        "Externe1": 0.1,
        "Externe2": 0.1,
        "IH": 1,
        "UH": 0.1,
        "Ucoil1": 1.0e-2,
        "Ucoil2": 1.0e-2,
        "Ucoil3": 1.0e-2,
        "Ucoil4": 1.0e-2,
        "Ucoil5": 1.0e-2,
        "Ucoil6": 1.0e-2,
        "Ucoil7": 1.0e-2,
        "Ucoil8": 1.0e-2,
        "Ucoil9": 1.0e-2,
        "Ucoil10": 1.0e-2,
        "Ucoil11": 1.0e-2,
        "Ucoil12": 1.0e-2,
        "Ucoil13": 1.0e-2,
        "Ucoil14": 1.0e-2,
        "IB": 1,
        "UB": 0.1,
        "Ucoil15": 0.1,
        "Ucoil16": 0.1,
    }
    return (
        color_dict,
        channels_dict,
        uchannels_dict,
        pupitre_dict,
        upupitre_dict,
        threshold_dict,
    )


def extract_data(file: str, insert: str):
    extension = os.path.splitext(file)[-1]
    filename = os.path.basename(file).replace(extension, "")

    mrun = MagnetRun()
    match extension:
        case ".txt":
            (site, timestamp) = filename.split("_")
            date, time = timestamp.split("---")
            # convert ddate and dtime into a timestamp
            (start_timestamp, start_ftimestamp) = convert_to_timestamp(
                date, time, date_format="%Y.%m.%d", time_format="%H:%M:%S"
            )
            mrun = MagnetRun.fromtxt(site, insert, file)
        case ".tdms":
            (site, mode, timestamp) = filename.split("_")
            date, time = timestamp.split("-")
            (start_timestamp, start_ftimestamp) = convert_to_timestamp(date, time)
            mrun = MagnetRun.fromtdms(site, insert, file)
        case _:
            raise RuntimeError(f"{file}: unsupported {extension}")

    mdata = mrun.getMData()
    duration = mdata.getDuration()
    end_timestamp = datetime.fromtimestamp(start_timestamp) + pd.to_timedelta(
        duration, unit="s"
    )
    end_ftimestamp = end_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return (start_ftimestamp, end_ftimestamp)


def find_files(args, site, time):
    pupitre_datadir = args.pupitre_datadir
    pupitre_filter = f"{pupitre_datadir}/{site}_20{time[0][0:2]}.{time[0][2:4]}.{time[0][4:]}---*.txt"

    file = args.input_file[0]
    extension = os.path.splitext(file)[-1]
    filename = os.path.basename(file).replace(extension, "")
    pigbrother = filename.replace("Overview", "Archive")
    archive_datadir = os.path.dirname(file).replace("Overview", "Fichiers_Archive")
    archive_filter = f"{archive_datadir}/{pigbrother.replace(time[1],'*.tdms')}"
    default_datadir = os.path.dirname(file).replace("Overview", "Fichiers_Default")

    default_filefilter = f"{file.replace('_Overview_','_Default_')}"
    default_filter = f"{default_datadir}/{default_filefilter.replace(time[1],'*.tdms')}"
    return pupitre_filter, archive_filter, default_filter


def load_df(file, site, insert, group, keys) -> tuple:
    extension = os.path.splitext(file)[-1]

    df = pd.DataFrame()
    t0 = datetime.now()
    match extension:
        case ".txt":
            mrun = MagnetRun.fromtxt(site, insert, file)
            mdata = mrun.getMData()
            t0 = mdata.Data["timestamp"].iloc[0]
            df = pd.DataFrame(mdata.getData(["t", "timestamp"] + keys))
        case ".tdms":
            mrun = MagnetRun.fromtdms(site, insert, file)
            mdata = mrun.getMData()
            t0 = mdata.Groups[group][keys[0]]["wf_start_time"]
            dt = mdata.Groups[group][keys[0]]["wf_increment"]
            df = pd.DataFrame(mdata.getTdmsData(group, keys))
            df["timestamp"] = [
                np.datetime64(t0).astype(datetime) + timedelta(0, i * dt)
                for i in df.index.to_list()
            ]
    return df, t0


def load_data(files, site, insert, group, keys) -> list[pd.DataFrame]:
    df_ = []
    for file in files:
        df, t0 = load_df(file, site, insert, group, keys)
        df_.append(df)
    return df_


def merge_data(df_list: list) -> pd.DataFrame:
    if len(df_list) > 1:
        return pd.concat(df_list)
    return df_list[0]


def synchronize_data(df: pd.DataFrame, t0: datetime) -> tuple:
    t0_ = df["timestamp"].iloc[0]
    timeshift = t0 - t0_
    df["timestamp"] = df["timestamp"] + pd.to_timedelta(timeshift)

    pt0 = df.iloc[0]["timestamp"]
    df.drop(["t"], axis=1, inplace=True)
    df["t"] = df.apply(lambda row: (row.timestamp - pt0).total_seconds(), axis=1)
    return timeshift, df


def plot_data(
    df_overview,
    df_archive,
    df_pupitre,
    df_incidents,
    channels_dict,
    pupitre_dict,
    site,
    tkey,
    key,
    title,
    msg,
    args,
):
    my_ax = plt.gca()
    legends = [f"{key}"]
    df_overview.plot(x=tkey, y=key, ax=my_ax)
    df_archive.plot(x=tkey, y=channels_dict[key], ax=my_ax)
    legends.append(f"{channels_dict[key]}")
    df_pupitre.plot(x=tkey, y=pupitre_dict[site][key], ax=my_ax)
    legends.append(f"pupitre {pupitre_dict[site][key]}")
    plt.legend(labels=legends)

    if not df_incidents.empty:
        for incident in df_incidents[tkey].to_list():
            plt.axvline(incident, color="red", alpha=0.2, label="Incident")

    plt.title(f"{title}: {key} {msg}")
    plt.grid()
    if args.show:
        plt.show()
    if args.save:
        plt.savefig(f"{title}-{channels_dict[key]}-timestamp-concat.png", dpi=300)
    plt.close()


def main():
    args = parse_arguments()

    (
        color_dict,
        channels_dict,
        uchannels_dict,
        pupitre_dict,
        upupitre_dict,
        threshold_dict,
    ) = setup()

    input_files = natsorted(args.input_file)

    insert = "tututu"
    extension = os.path.splitext(input_files[0])[-1]
    filename = os.path.basename(input_files[0]).replace(extension, "")
    (site, mode, timestamp) = filename.split("_")
    date, time = timestamp.split("-")
    print(f"site={site}, date={date}, time={time}, insert={insert}", flush=True)

    # select files
    pupitre_filter, archive_filter, default_filter = find_files(
        args, site, (date, time)
    )
    pupitre_files = natsorted(glob.glob(pupitre_filter))
    archive_files = natsorted(glob.glob(archive_filter))
    default_files = natsorted(glob.glob(default_filter))
    print(
        f"pfilter: {pupitre_filter}, afilter: {archive_filter}, dfilter: {default_filter}",
        flush=True,
    )
    print(f"pupitre_files={pupitre_files}", flush=True)
    print(f"archive_files={archive_files}", flush=True)
    print(f"default_files={default_files}", flush=True)

    overview_dict = {}
    for file in input_files:
        extension = os.path.splitext(file)[-1]
        # dirname = os.path.dirname(file)
        filename = os.path.basename(file).replace(extension, "")
        start, end = extract_data(file, insert)
        print(f"{file}: start={start}, end={end}", flush=True)

        dict_files = {"pupitre": [], "archive": [], "default": []}

        for pfile in pupitre_files:
            pstart, pend = extract_data(pfile, insert)
            if pstart >= start and pend <= end:
                dict_files["pupitre"].append(pfile)
        for pfile in archive_files:
            pstart, pend = extract_data(pfile, insert)
            if pstart >= start and pend <= end:
                dict_files["archive"].append(pfile)
        for pfile in default_files:
            pstart, pend = extract_data(pfile, insert)
            if pstart >= start and pend <= end:
                dict_files["default"].append(pfile)

        print(f"dict_files[{filename}]: ", dict_files, flush=True)
        overview_dict[filename] = {
            "sources": dict_files,
            "data": {
                "overview": pd.DataFrame(),
                "pupitre": pd.DataFrame(),
                "archive": pd.DataFrame(),
                "default": pd.DataFrame(),
            },
            "t0": datetime.now(),
        }

    # print("overview_dict:", overview_dict, flush=True)

    # Load Overview data
    df_overview_list = load_data(
        input_files,
        site,
        insert,
        "Courants_Alimentations",
        [args.key, channels_dict[args.key]],
    )
    df_overview = merge_data(df_overview_list)
    ot0 = df_overview["timestamp"].iloc[0]
    df_overview["t"] = df_overview.apply(
        lambda row: (row.timestamp - ot0).total_seconds(), axis=1
    )

    # Load Archive data
    for ofile in overview_dict:
        df_archive_list = load_data(
            overview_dict[ofile]["sources"]["archive"],
            site,
            insert,
            "Courants_Alimentations",
            [channels_dict[args.key]],
        )
        df_archive = merge_data(df_archive_list)
        at0 = df_archive.iloc[0]["timestamp"]
        df_archive["t"] = df_archive.apply(
            lambda row: (row.timestamp - at0).total_seconds(), axis=1
        )
        overview_dict[ofile]["data"]["archive"] = df_archive

    # Load pupitre data
    for ofile in overview_dict:
        df_pupitre_list = load_data(
            overview_dict[ofile]["sources"]["pupitre"],
            site,
            insert,
            "Courants_Alimentations",
            [pupitre_dict[site][args.key]],
        )
        df_pupitre = merge_data(df_pupitre_list)
        pt0 = df_pupitre.iloc[0]["timestamp"]
        df_pupitre["t"] = df_pupitre.apply(
            lambda row: (row.timestamp - pt0).total_seconds(), axis=1
        )
        overview_dict[ofile]["data"]["pupitre"] = df_pupitre

    # Load incidents data
    for ofile in overview_dict:
        incidents = []
        print(overview_dict[ofile]["sources"]["default"])
        for ifile in overview_dict[ofile]["sources"]["default"]:
            mrun = MagnetRun.fromtdms(site, insert, ofile)
            mdata = mrun.getMData()
            t0 = mdata.Groups["Courants_Alimentations"][args.key]["wf_start_time"]
            incidents.append(t0)

        df = pd.DataFrame(incidents, columns=["timestamp"])
        df["t"] = df.apply(lambda row: (row.timestamp - ot0).total_seconds(), axis=1)
        overview_dict[ofile]["data"]["defaut"] = df

    # save signature per overview file
    for i, file in enumerate(input_files):
        extension = os.path.splitext(input_files[0])[-1]
        filename = os.path.basename(input_files[0]).replace(extension, "")

        mrun = MagnetRun.fromtdms(site, insert, file)
        mdata = mrun.getMData()
        t0 = mdata.Groups["Courants_Alimentations"][args.key]["wf_start_time"]

        # get mode
        bitter_only = True
        if (
            "Courants_Alimentations/Référence_GR1" in mdata.getKeys()
            and "Courants_Alimentations/Référence_GR2" in mdata.getKeys()
        ):
            GR = mdata.getData(
                [
                    "Courants_Alimentations/Référence_GR1",
                    "Courants_Alimentations/Référence_GR2",
                ]
            ).copy()
            GR.loc[:, "diff"] = GR.loc[:, "Référence_GR2"] - GR.loc[:, "Référence_GR1"]

            if GR["diff"].abs().max() > 5 and not (
                GR["Référence_GR2"].abs().max() <= 1
                or GR["Référence_GR1"].abs().max() <= 1
            ):
                print(f"{mdata.FileName}: mode=ECOmode")
            del GR

        if bitter_only:
            uprobes = ["ALL_externes", "Externe1", "Externe2"]
            print("\n!!! Selecting only U probes for Bitter !!!\n")

        osignature = Signature.from_mdata(
            mdata, f"Courants_Alimentations/{args.key}", "t", threshold_dict[args.key]
        )
        # osignature.dump(filename)

        # get pandas dataframe
        df_overview = df_overview_list[i]
        df_archive = overview_dict[filename]["data"]["archive"]
        df_incidents = overview_dict[filename]["data"]["default"]

        # synchronize data ad get timeshit
        timeshift, df_pupitre = synchronize_data(
            overview_dict[filename]["data"]["pupitre"], ot0
        )

        # get lag correlation
        # lag_correlation(
        print("\nLag correlation: pupitre/pigbrother overview")
        print("t0 (overview):", df_overview["timestamp"].iloc[0])
        print("t0 (pupitre):", df_pupitre["timestamp"].iloc[0])

        def compute_lag(
            df1,
            key1: str,
            istart1: int,
            iend1: int,
            df2,
            key2: str,
            istart2: int = None,
            iend2: int = None,
            show: bool = False,
            save: bool = False,
        ):
            ts1 = df1.loc[:, ["timestamp", key1]]
            ts1.set_index("timestamp", inplace=True)
            ts1 = ts1.iloc[:, 0]
            ts1 = ts1 - ts1.min()

            ts1_index = ts1.index.to_list()
            otstart = ts1_index[istart1]
            otend = ts1_index[iend1]
            print(f"ts1 range: [{istart1}, {iend1}]-> [{otstart}, {otend}]")

            ts2 = df2.loc[:, ["timestamp", key2]]
            ts2.set_index("timestamp", inplace=True)
            ts2 = ts2.iloc[:, 0]

            pstart = ts2.index.get_indexer([pd.Timestamp(otstart)], method="nearest")
            pend = ts2.index.get_indexer([pd.Timestamp(otend)], method="nearest")
            ptstart = ts2.index[pstart[0]]
            ptend = ts2.index[pend[0]]
            print(f"ts2 range:  [{pstart[0]}, {pend[0]}] -> [{ptstart}, {ptend}]")

            # resample to make sure that timeseries share the same index
            ts2_index = ts2.index.to_list()
            ts2_resampled = ts2.resample("1s", origin=ts2_index[0]).asfreq()

            # Interpolate missing values (optional, depending on your use case)
            ts2_resampled = ts2_resampled.interpolate(method="linear")
            pstart = ts2_resampled.index.get_indexer(
                [pd.Timestamp(otstart)], method="nearest"
            )
            pend = ts2_resampled.index.get_indexer(
                [pd.Timestamp(otend)], method="nearest"
            )
            ptstart = ts2_resampled.index[pstart[0]]
            ptend = ts2_resampled.index[pend[0]]

            # pupitre data
            ts2_data = {
                "field": key2,
                "df": ts2_resampled,
                "range": {"start": ptstart, "end": ptend},
            }

            # overview data
            ts1_data = {
                "field": key1,
                "df": ts1,
                "range": {"start": otstart, "end": otend},
            }
            lag = lag_correlation(
                ts2_data,
                ts1_data,
                show=show,
                save=save,
            )

            return lag

        lag = compute_lag(
            df_overview,
            channels_dict[args.key],
            osignature.changes[1] - 2,
            osignature.changes[2] + 2,
            df_pupitre,
            pupitre_dict[site][args.key],
            show=args.show,
            save=args.save,
        )
        """
        ts_overview_field = channels_dict[args.key]
        ts_overview = df_overview.loc[:, ["timestamp", ts_overview_field]]
        ts_overview.set_index("timestamp", inplace=True)
        ts_overview = ts_overview.iloc[:, 0]
        ts_overview = ts_overview - ts_overview.min()

        ts_overview_index = ts_overview.index.to_list()
        ostart = osignature.changes[1] - 2
        oend = osignature.changes[2] + 2
        otstart = ts_overview_index[ostart]
        otend = ts_overview_index[oend]
        print(f"ts_overview overview range: [{ostart}, {oend}]-> [{otstart}, {otend}]")

        ts_pupitre_field = pupitre_dict[site][args.key]
        ts_pupitre = df_pupitre.loc[:, ["timestamp", ts_pupitre_field]]
        ts_pupitre.set_index("timestamp", inplace=True)
        ts_pupitre = ts_pupitre.iloc[:, 0]

        ts_pupitre_index = ts_pupitre.index.to_list()
        pstart = ts_pupitre.index.get_indexer([pd.Timestamp(otstart)], method="nearest")
        pend = ts_pupitre.index.get_indexer([pd.Timestamp(otend)], method="nearest")
        ptstart = ts_pupitre.index[pstart[0]]
        ptend = ts_pupitre.index[pend[0]]
        print(
            f"ts_pupitre timestamp range:  [{pstart}, {pend}] -> [{ptstart}, {ptend}]"
        )

        # resample to make sure that timeseries share the same index
        ts_pupitre_resampled = ts_pupitre.resample(
            "1s", origin=ts_pupitre_index[0]
        ).asfreq()

        # Interpolate missing values (optional, depending on your use case)
        ts_pupitre_resampled = ts_pupitre_resampled.interpolate(method="linear")
        pstart = ts_pupitre_resampled.index.get_indexer(
            [pd.Timestamp(otstart)], method="nearest"
        )
        pend = ts_pupitre_resampled.index.get_indexer(
            [pd.Timestamp(otend)], method="nearest"
        )
        ptstart = ts_pupitre_resampled.index[pstart[0]]
        ptend = ts_pupitre_resampled.index[pend[0]]

        # pupitre data
        ts_pupitre_data = {
            "field": ts_pupitre_field,
            "df": ts_pupitre_resampled,
            "range": {"start": ptstart, "end": ptend},
        }

        # overview data
        ts_overview_data = {
            "field": ts_overview_field,
            "df": ts_overview,
            "range": {"start": otstart, "end": otend},
        }
        lag = lag_correlation(
            ts_pupitre_data,
            ts_overview_data,
            show=args.show,
            save=args.save,
        )
        """

        df_pupitre["timestamp"] = df_pupitre["timestamp"] - pd.to_timedelta(f"{lag}s")

        pt0 = df_pupitre["timestamp"].iloc[0]
        print("new t0 (pupitre):", df_pupitre["timestamp"].iloc[0])
        df_pupitre.drop(["t"], axis=1, inplace=True)
        df_pupitre["t"] = df_pupitre.apply(
            lambda row: (row.timestamp - ot0).total_seconds(), axis=1
        )

        # plots
        timeshift -= lag
        msg = f"(sync with pigbrother {timeshift.total_seconds()} s)"

        # plot sync data vs t or timestamp
        plot_data(
            df_overview_list[i],
            df_archive,
            df_pupitre,
            df_incidents,
            channels_dict,
            pupitre_dict,
            site,
            args.tkey,
            args.key,
            filename,
            msg,
            args,
        )


if __name__ == "__main__":
    main()
