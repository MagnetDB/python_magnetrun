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

from .processing.correlations import compute_lag
from .signature import Signature

from scipy import stats
from .processing.distance import calc_euclidean, calc_mape, calc_correlation
from tabulate import tabulate


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
        "M9": {
            "Référence_GR1": "IH",
            "Référence_GR2": "IB",
            "Référence_GR1_Q": "FlowH",
            "Référence_GR2_Q": "FlowB",
        },
        "M8": {
            "Référence_GR1": "IB",
            "Référence_GR2": "IH",
            "Référence_GR1_Q": "FlowB",
            "Référence_GR2_Q": "FlowH",
        },
        "M10": {
            "Référence_GR1": "IB",
            "Référence_GR2": "IH",
            "Référence_GR1_Q": "FlowB",
            "Référence_GR2_Q": "FlowH",
        },
    }
    upupitre_dict = {
        "M9": {"Référence_GR1": ["UH"], "Référence_GR2": ["UB", "Ucoil15", "Ucoil16"]},
        "M8": {"Référence_GR2": ["UH"], "Référence_GR1": ["UB", "Ucoil15", "Ucoil16"]},
        "M10": {"Référence_GR2": ["UH"], "Référence_GR1": ["UB", "Ucoil15", "Ucoil16"]},
    }
    threshold_dict = {
        "Référence_GR1": 0.5,
        "Courant_GR1": 0.5,
        "ALL_internes": 0.1,
        "Interne1": 1.0e-2,
        "Interne2": 1.0e-2,
        "Interne3": 1.0e-2,
        "Interne4": 1.0e-2,
        "Interne5": 1.0e-2,
        "Interne6": 1.0e-2,
        "Interne7": 1.0e-2,
        "Référence_GR2": 0.5,
        "Courant_GR2": 0.5,
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


def extract_data(file: str, insert: str, key: str | None) -> tuple:
    extension = os.path.splitext(file)[-1]
    filename = os.path.basename(file).replace(extension, "")

    start_timestamp = float()
    start_ftimestamp = str()
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
            site = str()
            timestamp = str()
            res = filename.split("_")

            # regular case
            if len(res) == 3:
                (site, mode, timestamp) = res
                date, time = timestamp.split("-")
                (start_timestamp, start_ftimestamp) = convert_to_timestamp(date, time)
            # special for default files
            elif len(res) == 4:
                (site, mode, timestamp, dmode) = res
                date, time = timestamp.split("-")
                (start_timestamp, start_ftimestamp) = convert_to_timestamp(
                    date, time, "%y%m%d", "%H%M%S"
                )

            mrun = MagnetRun.fromtdms(site, insert, file)
        case _:
            raise RuntimeError(f"{file}: unsupported {extension}")

    skip = False
    mdata = mrun.getMData()
    if key is not None:
        if not key in mdata.getKeys():
            print(f"{file}: {key} not found")
            skip = True

    duration = mdata.getDuration()
    end_timestamp = datetime.fromtimestamp(start_timestamp) + pd.to_timedelta(
        duration, unit="s"
    )
    end_ftimestamp = end_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return (start_ftimestamp, end_ftimestamp, skip)


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

    default = filename.replace("Overview", "Default")
    default_filter = f"{default_datadir}/{default.replace(time[1],'*.tdms')}"
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
            if not keys[0] in mdata.Groups[group]:
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
    df_overview.plot(x=tkey, y=key, color="b", ax=my_ax)
    legends = [f"Overview: {key}"]
    df_overview.plot(x=tkey, y=channels_dict[key], marker="o", color="r", ax=my_ax)
    legends.append(f"Overview: {channels_dict[key]}")
    df_archive.plot(x=tkey, y=channels_dict[key], alpha=0.5, color="r", ax=my_ax)
    legends.append(f"Archive: {channels_dict[key]}")
    df_pupitre.plot(x=tkey, y=pupitre_dict[site][key], marker=".", color="g", ax=my_ax)
    legends.append(f"Pupitre: {pupitre_dict[site][key]}")
    plt.legend(labels=legends)

    if not df_incidents.empty:
        for incident in df_incidents[tkey].to_list():
            plt.axvline(incident, color="red", alpha=0.2, label="Incident")

    plt.title(f'{title.replace("_Overview","")}: {key} {msg}')
    plt.grid()
    if args.show:
        plt.show()
    if args.save:
        (label, igroup) = key.split("_")
        plt.savefig(f'{title.replace("_Overview","")}-{igroup}.png', dpi=300)
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
    print(f"input_files: {input_files}", flush=True)

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
    print("\nfilters:")
    print(
        f"pfilter: {pupitre_filter},\nafilter: {archive_filter},\ndfilter: {default_filter}",
        flush=True,
    )
    print("\nfiles:")
    print(f"pupitre_files={pupitre_files}", flush=True)
    print(f"archive_files={archive_files}", flush=True)
    print(f"default_files={default_files}", flush=True)
    print("\n")

    symbol = str()
    unit = str()
    group = "Courants_Alimentations"
    overview_dict = {}
    for file in input_files:
        extension = os.path.splitext(file)[-1]
        # dirname = os.path.dirname(file)
        filename = os.path.basename(file).replace(extension, "")
        start, end, skip = extract_data(file, insert, f"{group}/{args.key}")
        print(f"{filename}: file={file}, start={start}, end={end}", flush=True)

        dict_files = {"pupitre": [], "archive": [], "default": []}

        for pfile in pupitre_files:
            pstart, pend, pskip = extract_data(
                pfile, insert, pupitre_dict[site][args.key]
            )
            if pstart >= start and pend <= end:
                dict_files["pupitre"].append(pfile)
        for pfile in archive_files:
            pstart, pend, pskip = extract_data(
                pfile, insert, f"{group}/{channels_dict[args.key]}"
            )
            if not pskip:
                if pstart >= start and pend <= end:
                    dict_files["archive"].append(pfile)
        for pfile in default_files:
            pstart, pend, pskip = extract_data(
                pfile, insert, f"{group}/{channels_dict[args.key]}"
            )
            if not pskip:
                if pstart >= start and pend <= end:
                    dict_files["default"].append(pfile)

        if not skip:
            mrun = MagnetRun.fromtdms(site, insert, file)
            mdata = mrun.getMData()
            t0 = mdata.Groups[group][args.key]["wf_start_time"]

            symbol, unit = mdata.getUnitKey(f"{group}/{channels_dict[args.key]}")

            # get mode
            # TODO make sure to screen current about a certain threshold
            bitter_only = True
            if (
                f"{group}/Référence_GR1" in mdata.getKeys()
                and f"{group}/Référence_GR2" in mdata.getKeys()
            ):
                GR = mdata.getData(
                    [
                        f"{group}/Référence_GR1",
                        f"{group}/Référence_GR2",
                    ]
                ).copy()
                GR.loc[:, "diff"] = (
                    GR.loc[:, "Référence_GR2"] - GR.loc[:, "Référence_GR1"]
                )

                if GR["diff"].abs().max() > 5 and not (
                    GR["Référence_GR2"].abs().max() <= 1
                    or GR["Référence_GR1"].abs().max() <= 1
                ):
                    print(f"{filename}: mode=ECOmode")

                    # compute current factor (corr_Ih_Ib.py)
                    import piecewise_regression

                    x = GR["Référence_GR1"].to_numpy()
                    y = GR["Référence_GR2"].to_numpy()
                    """
                    my_ax = plt.gca()
                    my_ax.plot(x, y, "o")
                    plt.xlabel("Référence_GR1 [A]")
                    plt.ylabel("Référence_GR2 [A]")
                    plt.grid()
                    plt.show()
                    plt.close()
                    """

                    pw_fit = piecewise_regression.Fit(x, y, n_breakpoints=1)
                    pw_fit.summary()
                    # Plot the data, fit, breakpoints and confidence intervals
                    pw_fit.plot_data(color="grey", s=2)
                    # Pass in standard matplotlib keywords to control any of the plots
                    pw_fit.plot_fit(color="red", linewidth=4)
                    pw_fit.plot_breakpoints()
                    pw_fit.plot_breakpoint_confidence_intervals()
                    plt.xlabel("Référence_GR1 [A]")
                    plt.ylabel("Référence_GR2 [A]")
                    plt.grid()
                    plt.show()
                    plt.close()
                del GR

            if bitter_only:
                uprobes = ["ALL_externes", "Externe1", "Externe2"]
                print("\n!!! Selecting only U probes for Bitter !!!\n")

            signature = Signature.from_mdata(
                mdata,
                f"{group}/{args.key}",
                "t",
                threshold_dict[args.key],
            )

            print(f"dict_files[{filename}]: ", dict_files, flush=True)
            overview_dict[filename] = {
                "signature": signature,
                "sources": dict_files,
                "data": {
                    "overview": pd.DataFrame(),
                    "pupitre": pd.DataFrame(),
                    "archive": pd.DataFrame(),
                    "default": pd.DataFrame(),
                },
                "t0": t0,
            }

    # Load Overview data
    df_overview_list = load_data(
        input_files,
        site,
        insert,
        group,
        [args.key, channels_dict[args.key]],
    )
    # print("df_overview_list:", len(df_overview_list), flush=True)

    for i, ofile in enumerate(overview_dict):
        # print(i, ofile)
        overview_dict[ofile]["data"]["overview"] = df_overview_list[i]

    # Load Archive data
    print("\nLoad Archive data")
    for ofile in overview_dict:
        df_archive_list = load_data(
            overview_dict[ofile]["sources"]["archive"],
            site,
            insert,
            group,
            [channels_dict[args.key]],
        )
        df_archive = merge_data(df_archive_list)
        at0 = df_archive.iloc[0]["timestamp"]
        t_offset = (1 / 120.0) / 2.0
        df_archive["t"] = df_archive.apply(
            lambda row: (row.timestamp - at0).total_seconds() + t_offset, axis=1
        )
        overview_dict[ofile]["data"]["archive"] = df_archive

    # Load pupitre data
    print("\nLoad Pupitre data")
    for ofile in overview_dict:
        df_pupitre_list = load_data(
            overview_dict[ofile]["sources"]["pupitre"],
            site,
            insert,
            group,
            [
                pupitre_dict[site][args.key],
                pupitre_dict[site][f"{args.key}_Q"],
                "teb",
                "debitbrut",
            ],
        )
        df_pupitre = merge_data(df_pupitre_list)
        pt0 = df_pupitre.iloc[0]["timestamp"]
        df_pupitre["t"] = df_pupitre.apply(
            lambda row: (row.timestamp - pt0).total_seconds(), axis=1
        )
        overview_dict[ofile]["data"]["pupitre"] = df_pupitre

        # IH/FlowH or IB/FlowB
        x = df_pupitre[pupitre_dict[site][args.key]].to_numpy()
        y = df_pupitre[pupitre_dict[site][f"{args.key}_Q"]].to_numpy()
        my_ax = plt.gca()
        my_ax.plot(x, y, "o")
        plt.xlabel(f"{pupitre_dict[site][args.key]} [A]")
        plt.ylabel(f'{pupitre_dict[site][f"{args.key}_Q"]} [l/s]')
        plt.grid()
        plt.title(overview_dict[ofile]["sources"]["pupitre"])
        plt.show()
        plt.close()

    # Load incidents data
    print("\nLoad Incidents data")
    for ofile in overview_dict:
        incidents = []
        print(f'{ofile}: {overview_dict[ofile]["sources"]["default"]}')
        for ifile in overview_dict[ofile]["sources"]["default"]:
            mrun = MagnetRun.fromtdms(site, insert, ifile)
            mdata = mrun.getMData()
            t0 = mdata.Groups[group][args.key]["wf_start_time"]
            incidents.append(t0)

        ot0 = overview_dict[ofile]["data"]["overview"].iloc[0]["timestamp"]
        df = pd.DataFrame(incidents, columns=["timestamp"])
        df["t"] = df.apply(lambda row: (row.timestamp - ot0).total_seconds(), axis=1)
        overview_dict[ofile]["data"]["defaut"] = df

    # save signature per overview file
    print("\nProcess Overview Files (signature, lag)")
    for i, ofile in enumerate(overview_dict):
        # get pandas dataframe
        df_overview = overview_dict[ofile]["data"]["overview"]  # df_overview_list[i]
        ot0 = df_overview["timestamp"].iloc[0]

        # for overview files
        t_offset = 1 / 2.0
        df_overview["t"] = df_overview.apply(
            lambda row: (row.timestamp - ot0).total_seconds() + t_offset, axis=1
        )
        df_archive = overview_dict[ofile]["data"]["archive"]
        df_incidents = overview_dict[ofile]["data"]["default"]

        # synchronize data ad get timeshit
        timeshift, df_pupitre = synchronize_data(
            overview_dict[ofile]["data"]["pupitre"], ot0
        )

        # get lag correlation
        # lag_correlation(
        print("\nLag correlation: pupitre/pigbrother overview")
        print("t0 (overview):", df_overview["timestamp"].iloc[0])
        print("t0 (pupitre):", df_pupitre["timestamp"].iloc[0])

        tables = []
        headers = ["P", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        signature = overview_dict[ofile]["signature"]
        t0 = overview_dict[ofile]["t0"]
        for i, regime in enumerate(signature.regimes):
            istart = signature.changes[i]
            iend = df_overview.index.to_list()[-1]
            if i < len(signature.changes) - 1:
                iend = signature.changes[i + 1]

            if regime == "P":
                table = [i] + df_overview[args.key].iloc[
                    istart:iend
                ].describe().to_list()
                tables.append(table)
        print(tabulate(tables, headers=headers, tablefmt="psql"))
        print("regimes (overview/reference):", signature.regimes)

        osignature = Signature.from_df(
            ofile,
            t0,
            df_overview,
            channels_dict[args.key],
            symbol,
            unit,
            tkey="t",
            threshold=threshold_dict[channels_dict[args.key]],
            timeshift=0,
        )

        print("regimes (overview/current):", osignature.regimes)

        psignature = Signature.from_df(
            ofile,
            t0,
            df_pupitre,
            pupitre_dict[site][args.key],
            symbol,
            unit,
            tkey="t",
            threshold=threshold_dict[pupitre_dict[site][args.key]],
            timeshift=0,
        )
        print("regimes (pupitre/current):", psignature.regimes)

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
        legends = [pupitre_dict[site][args.key]]
        df_pupitre.plot(
            x="t", y=pupitre_dict[site][args.key], color="red", marker=".", ax=ax1
        )

        for x in psignature.times:
            ax1.axvline(x=x, color="red")
        ax1.legend(labels=legends)
        ax1.grid(True)

        legends = [channels_dict[args.key]]
        df_overview.plot(
            x="t",
            y=channels_dict[args.key],
            color="blue",
            marker="o",
            alpha=0.2,
            ax=ax2,
        )
        for x in osignature.times:
            ax2.axvline(x=x, color="blue")
        ax2.legend(labels=legends)
        ax2.grid(True)

        legends = [args.key]
        df_overview.plot(
            x="t", y=args.key, color="green", marker="*", alpha=0.2, ax=ax3
        )
        for x in signature.times:
            ax3.axvline(x=x, color="green")
        ax3.legend(labels=legends)
        ax3.grid(True)

        fig.suptitle(f"{filename}: Decompose {pupitre_dict[site][args.key]}")
        plt.show()
        plt.close()

        """
        # Check consistend data with plateaux
        from .utils.plateaux import nplateaus

        symbol, unit = mdata.getUnitKey(f"{group}/{args.key}")
        pdata = nplateaus(
            mdata,
            xField=("t", "t", "s"),
            yField=(f"{group}/{args.key}", symbol, unit),
            threshold=threshold_dict[args.key],
            num_points_threshold=10,
            save=args.save,
            show=args.show,
            verbose=False,
        )
        print("pdata:", pdata)
        for plateau in pdata:
            print(f'plateau: {plateau["start"]} -> {plateau["end"]}')
        for i, regime in enumerate(signature.regimes):
            if regime == "P":
                end = df_overview["t"].iloc[-1]
                if i < len(signature.times) - 2:
                    end = signature.times[i + 1]
                print(f"regime P: {signature.times[i]} -> {end}")
        """
        # plot sync data vs t or timestamp
        msg = "vs timestamp"
        plot_data(
            df_overview,
            df_archive,
            df_pupitre,
            df_incidents,
            channels_dict,
            pupitre_dict,
            site,
            "timestamp",
            args.key,
            filename,
            msg,
            args,
        )
        msg = "vs t"
        plot_data(
            df_overview,
            df_archive,
            df_pupitre,
            df_incidents,
            channels_dict,
            pupitre_dict,
            site,
            "t",
            args.key,
            filename,
            msg,
            args,
        )

        # find the latest big change in signature for overview and pupitre
        def compute_regime_score(
            regime: str,
            value: tuple,
            time: tuple,
            reference_regime: str,
            reference_value: tuple,
            reference_time: tuple,
        ):
            score = float("inf")
            lags = (float("inf"), float("inf"))
            if reference_regime == regime:
                start_diff = abs(value[0] - reference_value[0])
                end_diff = abs(value[1] - reference_value[1])
                score = start_diff + end_diff

                start_lag = time[0] - reference_time[0]
                end_lag = time[1] - reference_time[1]
                lags = (start_lag, end_lag)

            return (score, lags)

        def find_best_matching_regime(signature, reference_signature):
            best_matches = []
            for i, regime in enumerate(signature.regimes):
                best_score = float("inf")
                best_lags = (float("inf"), float("inf"))
                best_index = (0, 0)
                best_match = None

                if regime in ["U", "D"] and i < len(signature.times) - 2:
                    values = (signature.values[i], signature.values[i + 1])
                    times = (signature.times[i], signature.times[i + 1])
                    for j, ref_regime in enumerate(reference_signature.regimes):
                        # get value and time range
                        if (
                            ref_regime in ["U", "D"]
                            and j < len(reference_signature.times) - 2
                        ):
                            ref_values = (
                                reference_signature.values[j],
                                reference_signature.values[j + 1],
                            )
                            ref_times = (
                                reference_signature.times[j],
                                reference_signature.times[j + 1],
                            )

                            score, lags = compute_regime_score(
                                regime,
                                values,
                                times,
                                ref_regime,
                                ref_values,
                                ref_times,
                            )

                            if score < best_score:
                                best_score = score
                                best_match = ref_regime
                                best_lags = lags
                                best_index = (i, j)
                    best_matches.append(
                        (regime, best_match, best_score, best_lags, best_index)
                    )
            return best_matches

        best_matches = find_best_matching_regime(osignature, psignature)

        for regime, best_match, score, lags, best_index in best_matches:
            print(
                f"Best match for regime {regime} [{best_index[0]}] in overview: {best_match} with score {score} and lags {lags} [{best_index[1]}] in pupitre"
            )

        # get lag from 1st U sequence
        print("1st lag")
        df1_data = {
            "df": df_overview.loc[:, ["timestamp", channels_dict[args.key]]],
            "field": channels_dict[args.key],
            "range": (osignature.changes[0], osignature.changes[2] + 2),
        }
        df2_data = {
            "df": df_pupitre.loc[:, ["timestamp", pupitre_dict[site][args.key]]],
            "field": pupitre_dict[site][args.key],
            "range": (psignature.changes[0], psignature.changes[2] + 2),
        }

        lag = compute_lag(
            "timestamp",
            df1_data,
            df2_data,
            show=args.show,
            save=args.save,
            debug=args.debug,
        )
        print(f"1st lag: {lag.total_seconds()} s")
        df_pupitre["timestamp"] = df_pupitre["timestamp"] - pd.to_timedelta(f"{lag}s")

        # update timestamp and t
        pt0 = df_pupitre["timestamp"].iloc[0]
        print("new t0 (pupitre):", df_pupitre["timestamp"].iloc[0])
        df_pupitre.drop(["t"], axis=1, inplace=True)
        df_pupitre["t"] = df_pupitre.apply(
            lambda row: (row.timestamp - ot0).total_seconds(), axis=1
        )
        print(df_pupitre.head())

        # update times for psignature
        for j in range(len(psignature.times)):
            psignature.times[j] = psignature.times[j] - lag.total_seconds()
        best_matches = find_best_matching_regime(osignature, psignature)

        for regime, best_match, score, lags, best_index in best_matches:
            print(
                f"1st lag: Best match for regime {regime} [{best_index[0]}] in overview: {best_match} with score {score} and lags {lags} [{best_index[1]}] in pupitre"
            )

        # plots
        msg = f"(sync, 1st lag with pigbrother {lag.total_seconds()} s)"

        # plot sync data vs t or timestamp
        plot_data(
            df_overview,
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

        # get lag from last D
        print("last lag: ", flush=True)

        """
        print(
            f"overview: {osignature.regimes[-4]}{osignature.regimes[-2]}: istart={osignature.changes[-3] - 2}, iend={osignature.changes[-2] + 2}"
        )
        print(
            f'overview: {osignature.regimes[-3]}{osignature.regimes[-2]}: ostart={df_overview["t"].iloc[osignature.changes[-3] - 2]}, oend={df_overview["t"].iloc[osignature.changes[-2] + 2]}'
        )
        print(
            f"pupitre: {psignature.regimes[-3]}{psignature.regimes[-2]}: istart={psignature.changes[-3] - 2}, iend={psignature.changes[-2] + 2}"
        )
        print(
            f'pupitre: {psignature.regimes[-3]}{psignature.regimes[-2]}: pstart={df_pupitre["t"].iloc[psignature.changes[-3] - 2]}, pend={df_pupitre["t"].iloc[psignature.changes[-2] + 2]}'
        )

        df1_data["range"] = (None, None)
        df2_data["range"] = (None, None)

        compute_lag(
            "timestamp",
            df1_data,
            df2_data,
            show=args.show,
            save=args.save,
            debug=args.debug,
        )
        print(f"{lag.total_seconds()} s")

        # plots
        msg = f"(sync, lag with pigbrother {lag.total_seconds()} s)"

        # plot sync data vs t or timestamp
        plot_data(
            df_overview,
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
        """
        # compute distance between pupitre and pigbrother
        # print("df_pupitre:", df_pupitre)
        xdata = df_pupitre.copy()
        xdata.set_index("timestamp", inplace=True)
        xdata_index = xdata.index.to_list()
        xdata_resampled = xdata.resample("1s", origin=ot0).asfreq()
        # Interpolate missing values (optional, depending on your use case)
        xdata_resampled = xdata_resampled.interpolate(method="linear")
        xdata_resampled = xdata_resampled + xdata_resampled.min()

        print("xdata_resampled:", xdata_resampled.head())
        xdata_resampled.set_index("t", inplace=True)
        # print("after resample xdata_resampled:", xdata_resampled)
        end_index = xdata_resampled.index.values[-1].astype(int)
        print(
            f"\nDistance between pupitre {pupitre_dict[site][args.key]} and pigbrother {channels_dict[args.key]} from t=0 to t={end_index} s"
        )

        x = (
            xdata_resampled[pupitre_dict[site][args.key]]
            .loc[0:end_index]
            .to_numpy()
            .reshape(-1)
        )
        y = df_overview[channels_dict[args.key]].loc[0:end_index].to_numpy().reshape(-1)

        plt.plot(x, label="pupitre", marker=".", color="g")
        plt.plot(y, label="overview", marker="o", color="r", alpha=0.2)
        plt.title("distance")
        plt.legend()
        plt.grid()
        plt.show()
        plt.close()

        # print('Ib:', x, type(x), x.shape)
        scipy_stats = stats.describe(y - x)

        (label, igroup) = args.key.split("_")
        tables = []
        headers = [
            "Euclidean",
            "MAE",
            "Pearson",
            "Image",
            "mean",
            "min",
            "max",
            "var",
        ]
        table = [
            calc_euclidean(x, y),
            calc_mape(x, y),
            calc_correlation(x, y),
            f'{filename.replace("_Overview","")}-{igroup}.png',
            scipy_stats.mean,
            scipy_stats.minmax[0],
            scipy_stats.minmax[1],
            scipy_stats.variance,
        ]
        tables.append(table)
        print(tabulate(tables, headers, tablefmt="simple"), "\n")

        # Calculate DTW distance and obtain the warping paths (no need for the C library)
        from dtaidistance import dtw
        from scipy.stats import pearsonr

        ts_x = df_overview.loc[:, ["t", channels_dict[args.key]]]
        ts_x.set_index("t", inplace=True)
        ts_x.index = ts_x.index.astype(int)
        print("dtw ts_x:", ts_x)
        ts_y = xdata_resampled
        # drop negative index in ts_y
        ts_y = ts_y[ts_y.index >= 0]
        ts_y.index = ts_y.index.astype(int)
        print("dtw ts_y:", ts_y)

        ts_x = ts_x.to_numpy().reshape(-1)
        ts_y = ts_y.to_numpy().reshape(-1)

        distance, paths = dtw.warping_paths(ts_x, ts_y, use_c=False)
        best_path = dtw.best_path(paths)
        similarity_score = distance / len(best_path)

        # Create a DataFrame to display the similarity score and correlation coefficient
        results_df = pd.DataFrame(
            {"Metric": ["DTW Similarity Score"], "Value": [similarity_score]}
        )

        # Add descriptions for the results
        results_df["Description"] = [
            "Lower scores indicate greater similarity between the time series."
        ]
        print(results_df)

        plt.figure(figsize=(12, 8))

        # Original Time Series Plot
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax1.plot(ts_x, label=channels_dict[args.key], color="blue")
        ax1.plot(
            ts_y, label=pupitre_dict[site][args.key], linestyle="--", color="orange"
        )
        ax1.set_title("Original Time Series")
        ax1.legend()
        ax1.grid(True)

        # Shortest Path Plot (Cost Matrix with the path)
        # In this example, only the path is plotted, not the entire cost matrix.

        ax2 = plt.subplot2grid((2, 2), (0, 1))
        ax2.plot(
            np.array(best_path)[:, 0],
            np.array(best_path)[:, 1],
            "green",
            marker="o",
            linestyle="-",
        )
        ax2.set_title("Shortest Path (Best Path)")
        ax2.set_xlabel(channels_dict[args.key])
        ax2.set_ylabel(pupitre_dict[site][args.key])
        ax2.grid(True)

        # Point-to-Point Comparison Plot
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        ax3.plot(ts_x, label=channels_dict[args.key], color="blue", marker="o")
        ax3.plot(
            ts_y,
            label=pupitre_dict[site][args.key],
            color="orange",
            marker="x",
            linestyle="--",
        )
        for a, b in best_path:
            ax3.plot(
                [a, b],
                [ts_x[a], ts_y[b]],
                color="grey",
                linestyle="-",
                linewidth=1,
                alpha=0.5,
                sharex=ax1,
            )
        ax3.set_title("Point-to-Point Comparison After DTW Alignment")
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
