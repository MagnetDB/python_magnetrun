import os
from datetime import datetime, timedelta
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsorted
from .MagnetRun import MagnetRun

from .utils.convert import convert_to_timestamp

from .processing.correlations import compute_lag
from .signature import Signature

from scipy import stats
from .processing.distance import calc_euclidean, calc_mape, calc_correlation
from .flow_params import pwlf_fit
from sympy import Symbol
from tabulate import tabulate


# new default for pupitre-datadir: srvdata -> /home/LNCMI-G/christophe.trophime/LNCMIG-Data/srv-data-install
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="+", help="enter input file tdms")
    parser.add_argument(
        "--pupitre_datadir",
        help="enter pupitre datadir (default srvdata)",
        type=str,
        default="/home/LNCMI-G/christophe.trophime/LNCMIG-Data/srv-data-install",
    )

    """
    parser.add_argument(
        "--key",
        help="choose key",
        choices=["Référence_GR1", "Référence_GR2"],
        type=str,
        default="Référence_GR1",
    )
    """
    parser.add_argument(
        "--tkey",
        help="choose tkey",
        choices=["t", "timestamp"],
        type=str,
        default="t",
    )

    parser.add_argument("--dry_run", help="dry_run mode", action="store_true")
    parser.add_argument("--debug", help="activate debug", action="store_true")
    parser.add_argument("--save", help="save graphs (png format)", action="store_true")
    parser.add_argument(
        "--show", help="display graphs (X11 required)", action="store_true"
    )
    parser.add_argument(
        "--synchronize",
        help="synchronize clock pupitre/pigbrother files",
        action="store_true",
    )
    parser.add_argument(
        "--lag",
        help="compute lag between pupitre and pigbrother data",
        action="store_true",
    )
    parser.add_argument(
        "--distance", help="compute distance between series", action="store_true"
    )
    parser.add_argument("--bins", help="set bins for histograms", type=int, default=10)
    parser.add_argument(
        "--window", help="set rolling window size", type=int, default=50
    )
    parser.add_argument("--levels", help="set levels", type=int, default=4)
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
            "Référence_GR1_Rpm": "RpmH",
            "Référence_GR2_Rpm": "RpmB",
            "Référence_GR1_Pin": "HPH",
            "Référence_GR2_Pin": "HPB",
        },
        "M8": {
            "Référence_GR1": "IB",
            "Référence_GR2": "IH",
            "Référence_GR1_Q": "FlowB",
            "Référence_GR2_Q": "FlowH",
            "Référence_GR1_Rpm": "RpmB",
            "Référence_GR2_Rpm": "RpmH",
            "Référence_GR1_Pin": "HPB",
            "Référence_GR2_Pin": "HPH",
        },
        "M10": {
            "Référence_GR1": "IB",
            "Référence_GR2": "IH",
            "Référence_GR1_Q": "FlowB",
            "Référence_GR2_Q": "FlowH",
            "Référence_GR1_Rpm": "RpmB",
            "Référence_GR2_Rpm": "RpmH",
            "Référence_GR1_Pin": "HPB",
            "Référence_GR2_Pin": "HPH",
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
        "debitbrut": 25,
        "Pmagnet": 0.1,
    }
    return (
        color_dict,
        channels_dict,
        uchannels_dict,
        pupitre_dict,
        upupitre_dict,
        threshold_dict,
    )


def extract_data(
    file: str, site: str, insert: str, key: str | None, dry_run: bool = False
) -> tuple:
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
                mrun = MagnetRun.fromtdms(site, insert, file)
        case _:
            raise RuntimeError(f"{file}: unsupported {extension}")

    skip = False
    end_ftimestamp = str()
    if not dry_run:
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
    # LNCMIG-Data/srv-data-install/M9/2025.12.08 - 08:58:59.txt
    pupitre_datadir = f"{args.pupitre_datadir}/{site}"
    pupitre_filter = f"{pupitre_datadir}/20{date[0:2]}.{date[2:4]}.{date[4:]}*.txt"

    # pupitre_datadir = args.pupitre_datadir
    # pupitre_filter = (
    #     f"{pupitre_datadir}/{site}_20{date[0:2]}.{date[2:4]}.{date[4:]}---*.txt"
    # )

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
        res = extract_data(file, site=site, insert=None, key=None)
        start_time_file = datetime.strptime(res[0], tformat)
        end_time_file = datetime.strptime(res[1], tformat)
        # print(
        #     f"{file}: start_time_file={start_time_file} end_time_file={end_time_file}, start_time={start_time}, end_time={end_time}",
        #     flush=True,
        # )
        if start_time_file >= start_time and end_time_file < end_time:
            selected.append(file)
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


def synchronize_data(df: pd.DataFrame, t0: datetime) -> tuple:
    t0_ = df["timestamp"].iloc[0]
    timeshift = t0 - t0_
    df["timestamp"] = df["timestamp"] + pd.to_timedelta(timeshift)

    pt0 = df.iloc[0]["timestamp"]
    df.drop(["t"], axis=1, inplace=True)
    df["t"] = df.apply(lambda row: (row.timestamp - pt0).total_seconds(), axis=1)
    return timeshift, df


def plot_data(
    df_overview: pd.DataFrame,
    df_archive: pd.DataFrame,
    df_pupitre: pd.DataFrame,
    df_incidents: dict | None,
    channels_dict: dict,
    pupitre_dict: dict,
    site: str,
    tkey: str,
    key: str,
    title: str,
    msg: str,
    args,
):
    # my_ax = plt.gca()
    fig, my_ax = plt.subplots(figsize=(12, 5))
    df_overview.plot(x=tkey, y=key, color="b", ax=my_ax)
    legends = [f"Overview: {key}"]
    df_overview.plot(x=tkey, y=channels_dict[key], marker=".", color="r", ax=my_ax)
    legends.append(f"Overview: {channels_dict[key]}")
    df_archive.plot(x=tkey, y=channels_dict[key], alpha=0.5, color="r", ax=my_ax)
    legends.append(f"Archive: {channels_dict[key]}")
    df_pupitre.plot(x=tkey, y=pupitre_dict[site][key], color="g", ax=my_ax)
    legends.append(f"Pupitre: {pupitre_dict[site][key]}")
    plt.legend(labels=legends)

    annotation_dict = {}
    if df_incidents is not None:
        for itype, incident in df_incidents.items():
            # print(f"plot_data: itype={itype}", flush=True)
            for i, idf in enumerate(incident):
                t_mid = idf[tkey].median()
                f_mid = idf[channels_dict[key]].median()
                (point,) = my_ax.plot(t_mid, f_mid, "yo", markersize=8)

                # Add annotation with arrow
                annot = my_ax.annotate(
                    rf"{itype} \#{i+1}",
                    xy=(t_mid, f_mid),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                )

                # Make annotation clickable
                annot.set_picker(True)

                # Store metadata
                annotation_dict[annot] = {
                    "anomaly": rf"{itype} \#{i+1}",
                    "idx": i,
                    "df": idf,
                    "pupitre": (df_pupitre, pupitre_dict[site][key]),
                    "archive": (df_archive, channels_dict[key]),
                }

    # Track open subplot figures
    open_figures = {}

    def on_pick(event):
        if event.artist in annotation_dict:
            annot = event.artist
            metadata = annotation_dict[annot]
            anomaly = metadata["anomaly"]
            idx = metadata["idx"]
            idf = metadata["df"]
            pupitre, pupitre_key = metadata["pupitre"]
            archive, archive_key = metadata["archive"]

            # Close previous subplot if it exists
            if idx in open_figures:
                plt.close(open_figures[idx])

            # Create subplot
            fig_sub, ax_sub = plt.subplots(figsize=(8, 5))
            # ax_sub.plot(idf["t"], idf[channels_dict[key]], "y-", linewidth=2)
            idf.plot(x="t", y=archive_key, color="y", ax=ax_sub)
            pupitre.plot(x="t", y=pupitre_key, color="g", ax=ax_sub)
            archive.plot(x="t", y=archive_key, color="r", alpha=0.5, ax=ax_sub)
            ax_sub.set_xlabel("t")
            ax_sub.set_xlim(idf.iloc[0]["t"], idf.iloc[-1]["t"])
            ax_sub.set_title(anomaly)
            ax_sub.grid(True, alpha=0.3)

            fig_sub.tight_layout()
            fig_sub.show()

            # Store figure reference
            open_figures[idx] = fig_sub

    # Connect pick event
    fig.canvas.mpl_connect("pick_event", on_pick)
    plt.title(f'{title.replace("_Overview","")}: {key} {msg}')
    plt.grid()
    plt.tight_layout()
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
    # print(f"input_files: {input_files}", flush=True)

    insert = "tututu"

    symbol = str()
    unit = str()
    group = "Courants_Alimentations"
    Ikeys_ref = []
    overview_dict = {}
    for file in input_files:
        print(f"*** processing file: {file} ***", flush=True)
        extension = os.path.splitext(file)[-1]
        filename = os.path.basename(file).replace(extension, "")
        (site, mode, timestamp) = filename.split("_")
        date, time = timestamp.split("-")
        # print(f"site={site}, date={date}, time={time}, insert={insert}", flush=True)
        (start_ftimestamp, end_ftimestamp, skip) = extract_data(
            file, site, insert=insert, key=None
        )

        # select files
        pupitre_filter, archive_filter, default_filter, trigger_filter, spike_filter = (
            find_files(args, file, site, date, time)
        )
        pupitre_files = select_files(
            glob.glob(pupitre_filter), site, start_ftimestamp, end_ftimestamp
        )
        archive_files = select_files(
            glob.glob(archive_filter), site, start_ftimestamp, end_ftimestamp
        )
        default_files = select_files(
            glob.glob(default_filter), site, start_ftimestamp, end_ftimestamp
        )
        trigger_files = select_files(
            glob.glob(trigger_filter), site, start_ftimestamp, end_ftimestamp
        )
        spike_files = select_files(
            glob.glob(spike_filter), site, start_ftimestamp, end_ftimestamp
        )
        print("\nfilters:")
        print(
            f"pfilter: {pupitre_filter},\nafilter: {archive_filter},\ndfilter: {default_filter},\ntfilter: {trigger_filter},\ntfilter: {spike_filter}",
            flush=True,
        )

        print("\nfiles:")
        print(f"pupitre_files={pupitre_files}", flush=True)
        print(f"archive_files={archive_files}", flush=True)
        print(f"default_files={default_files}", flush=True)
        print(f"trigger_files={trigger_files}", flush=True)
        print(f"spike_files={spike_files}", flush=True)
        print("\n")

        if not args.dry_run:
            mrun = MagnetRun.fromtdms(site, insert, file)
            mdata = mrun.getMData()
            Ikeys_ref = [
                key.replace(f"{group}/", "")
                for key in mdata.getKeys()
                if "Référence_GR" in key
            ]
            start, end, skip = extract_data(
                file, site, insert, f"{group}/{Ikeys_ref[0]}"
            )
            # print(f"{filename}: file={file}, start={start}, end={end}", flush=True)

            dict_files = {
                "overview": [file],
                "pupitre": [],
                "archive": [],
                "default": [],
                "trigger": [],
                "spike": [],
            }

            for pfile in pupitre_files:
                pstart, pend, pskip = extract_data(
                    pfile, site, insert, pupitre_dict[site][Ikeys_ref[0]]
                )
                if pstart >= start and pend <= end:
                    dict_files["pupitre"].append(pfile)
            for pfile in archive_files:
                pstart, pend, pskip = extract_data(
                    pfile, site, insert, f"{group}/{channels_dict[Ikeys_ref[0]]}"
                )
                if not pskip:
                    if pstart >= start and pend <= end:
                        dict_files["archive"].append(pfile)
            for pfile in default_files:
                pstart, pend, pskip = extract_data(
                    pfile, site, insert, f"{group}/{channels_dict[Ikeys_ref[0]]}"
                )
                if not pskip:
                    if pstart >= start and pend <= end:
                        dict_files["default"].append(pfile)

            for pfile in trigger_files:
                pstart, pend, pskip = extract_data(
                    pfile, site, insert, f"{group}/{channels_dict[Ikeys_ref[0]]}"
                )
                if not pskip:
                    if pstart >= start and pend <= end:
                        dict_files["trigger"].append(pfile)

            for pfile in spike_files:
                pstart, pend, pskip = extract_data(
                    pfile, site, insert, f"{group}/{channels_dict[Ikeys_ref[0]]}"
                )
                if not pskip:
                    if pstart >= start and pend <= end:
                        dict_files["spike"].append(pfile)

            print(f"{file}: dict_files: {dict_files}", flush=True)
            t0 = mdata.Groups[group][Ikeys_ref[0]]["wf_start_time"]

            symbol, unit = mdata.getUnitKey(f"{group}/{channels_dict[Ikeys_ref[0]]}")

            # get mode
            # TODO make sure to screen current about a certain threshold
            print("find mode:")
            bitter_only = True
            mode = {"name": "normal", "Intercept": 0, "Slopes": [1], "Breakpoint": None}
            if len(Ikeys_ref) == 2:
                GR = mdata.getData(
                    [
                        f"{group}/Référence_GR1",
                        f"{group}/Référence_GR2",
                    ]
                ).copy()

                GR = GR.query("`Référence_GR1` >= 300 and `Référence_GR2` >= 300")

                x = GR["Référence_GR1"].to_numpy()
                y = GR["Référence_GR2"].to_numpy()
                for segment in [1, 2]:
                    (mfit, eqns) = pwlf_fit(
                        "Référence_GR1",
                        x,
                        "Référence_GR2",
                        y,
                        degree=1,
                        segment=segment,
                    )
                    # TODO if error ?my_pwlf.standard_errors()? on brkpoints is big, try with 1 segment
                    I0 = eqns[0].evalf(subs={Symbol("x"): 0})
                    if abs(I0) <= 10:
                        break

                # mode: Intercept, slopes, brkpts
                if segment == 1:
                    mode = {
                        "name": "normal",
                        "Intercept": eqns[0].evalf(subs={Symbol("x"): 0}),
                        "Slopes": [float(mfit.beta[1])],
                        "Breakpoint": None,
                    }
                else:
                    mode = {
                        "name": "ecomode",
                        "Intercept": eqns[0].evalf(subs={Symbol("x"): 0}),
                        "Slopes": [
                            float(mfit.beta[1]),
                            float(mfit.beta[1] + mfit.beta[2]),
                        ],
                        "Breakpoint": float(mfit.fit_breaks[1]),
                    }
                print(f"mode:{mode}")
                del GR

            if bitter_only:
                uprobes = ["ALL_externes", "Externe1", "Externe2"]
                print("\n!!! Selecting only U probes for Bitter !!!\n")

            print(f"dict_files[{filename}]: ", dict_files, flush=True)
            overview_dict[filename] = {
                "mode": mode,
                "signature": {},
                "sources": dict_files,
                "data": {
                    "overview": pd.DataFrame(),
                    "pupitre": pd.DataFrame(),
                    "archive": pd.DataFrame(),
                    "default": [],
                    "trigger": [],
                    "spike": [],
                },
                "t0": t0,
                "BP": {},
                "teb": {},
                "debitbrut": {},
                "flow_params": {},
            }

            for key in Ikeys_ref:
                overview_dict[filename]["signature"][key] = Signature.from_mdata(
                    mdata,
                    f"{group}/{key}",
                    "t",
                    threshold_dict[key],
                )

            # Load Overview data
            Ikeys = Ikeys_ref + [channels_dict[key] for key in Ikeys_ref]
            print("\nLoad Overview data")
            # print(i, filename)
            df_overview_list = load_data(
                overview_dict[filename]["sources"]["overview"],
                site,
                insert,
                group,
                Ikeys,
            )
            # print(f"df_overview_list: {len(df_overview_list)} files")
            overview_dict[filename]["data"]["overview"] = df_overview_list[0]
            # print(
            #     f"{filename}: overview \n {overview_dict[filename]['data']['overview']}"
            # )

            df_overview = overview_dict[filename]["data"][
                "overview"
            ]  # df_overview_list[i]
            ot0 = df_overview["timestamp"].iloc[0]

            print("\nLoad Archive data")
            if len(overview_dict[filename]["sources"]["archive"]) == 0:
                raise RuntimeError(f"no archive file associated with {filename}")

            df_archive_list = load_data(
                overview_dict[filename]["sources"]["archive"],
                site,
                insert,
                group,
                Ikeys,
            )

            df_archive = merge_data(df_archive_list)

            # true if at0 is roughly equal to ot0
            at0 = df_archive.iloc[0]["timestamp"]
            t_offset = (1 / 120.0) / 2.0
            df_archive["t"] = df_archive.apply(
                lambda row: (row.timestamp - at0).total_seconds() + t_offset, axis=1
            )
            overview_dict[filename]["data"]["archive"] = df_archive

            # Load pupitre data
            print("\nLoad Pupitre data")
            Ikeys_p = [pupitre_dict[site][key] for key in Ikeys_ref]
            Flowkeys = []
            for key in Ikeys_ref:
                Flowkeys += [
                    pupitre_dict[site][f"{key}_Q"],
                    pupitre_dict[site][f"{key}_Rpm"],
                    pupitre_dict[site][f"{key}_Pin"],
                ]

            df_pupitre_list = load_data(
                overview_dict[filename]["sources"]["pupitre"],
                site,
                insert,
                group,
                Ikeys_p
                + Flowkeys
                + [
                    "BP",
                    "teb",
                    "debitbrut",
                    "Pmagnet",
                ],
            )
            df_pupitre = merge_data(df_pupitre_list)
            print(f"{filename}: pupitre \n {df_pupitre.keys()}")
            pt0 = df_pupitre.iloc[0]["timestamp"]
            df_pupitre["t"] = df_pupitre.apply(
                lambda row: (row.timestamp - pt0).total_seconds(), axis=1
            )

            # TODO: save a better approx for teb
            overview_dict[filename]["teb"] = df_pupitre["teb"].mean()
            overview_dict[filename]["BP"] = df_pupitre["BP"].mean()

            from .flow_params import debitbrut

            for nlevel in range(1, args.levels + 1):
                (thresholds, high_values, low_values) = debitbrut(
                    df_pupitre, filename, nlevels=nlevel
                )

            overview_dict[filename]["debitbrut"] = {
                "thresholds": thresholds,
                "high": high_values,
                "low": low_values,
            }

            overview_dict[filename]["data"]["pupitre"] = df_pupitre
            # synchronize data ad get timeshit
            timeshift, df_pupitre = synchronize_data(
                overview_dict[filename]["data"]["pupitre"], ot0
            )

            # just a few check on extra_key
            for extra_key in ["teb", "debitbrut", "Pmagnet", "BP"]:
                print(f"{extra_key}: {df_pupitre[extra_key].describe()}")
                df_pupitre.plot(x="t", y=extra_key)
                plt.grid()
                plt.show()
                plt.close()

            # IH/FlowH or IB/FlowB
            print(f"list key df_pupitre: {list(df_pupitre.keys())}")
            for key in Ikeys_ref:
                print(
                    f'flow-params for {key}: Ikey={pupitre_dict[site][key]}, RpmKey={pupitre_dict[site][f"{key}_Rpm"]}'
                )
                # TODO: how to find Imax automatically
                # use cjekel/piecewise_linear_fit_py??
                from .flow_params import compute as flow_params

                flow_params(
                    df_pupitre,
                    Ikey=pupitre_dict[site][key],
                    RpmKey=pupitre_dict[site][f"{key}_Rpm"],
                    QKey=pupitre_dict[site][f"{key}_Q"],
                    PinKey=pupitre_dict[site][f"{key}_Pin"],
                    PoutKey="BP",
                    name=f"{filename}",
                    show=args.show,
                    debug=False,
                )

            # Load incidents data
            print("\nLoad Incidents data")
            at0 = overview_dict[filename]["data"]["archive"].iloc[0]["timestamp"]
            for type in ["default", "trigger", "spike"]:
                print(f'{filename}: {overview_dict[filename]["sources"][type]}')
                overview_dict[filename]["data"][type] = load_data(
                    overview_dict[filename]["sources"][type],
                    site,
                    insert,
                    group,
                    Ikeys,
                )

                for i, df in enumerate(overview_dict[filename]["data"][type]):
                    it0 = df["timestamp"].iloc[0]
                    t_offset = (1 / 4800.0) / 2.0
                    print(
                        f'{type} file {overview_dict[filename]["sources"][type][i]}: it0={it0}, at0={at0}, t0={(it0 - at0).total_seconds() + t_offset}, len={len(df)}'
                    )
                    print(df["timestamp"])
                    df["t"] = df.apply(
                        lambda row: (row.timestamp - at0).total_seconds() + t_offset,
                        axis=1,
                    )

            # save signature per overview file
            print(f"\nProcess Overview Files (signature, lag): {filename} ****")

            # for overview files
            t_offset = 1 / 2.0
            df_overview["t"] = df_overview.apply(
                lambda row: (row.timestamp - ot0).total_seconds() + t_offset, axis=1
            )
            df_archive = overview_dict[filename]["data"]["archive"]
            df_incidents = {
                "default": overview_dict[filename]["data"]["default"],
                "spike": overview_dict[filename]["data"]["spike"],
                "trigger": overview_dict[filename]["data"]["trigger"],
            }

            # get lag correlation
            # lag_correlation(
            print("\nLag correlation: pupitre/pigbrother overview")
            print("t0 (overview):", df_overview["timestamp"].iloc[0])
            print("t0 (pupitre):", df_pupitre["timestamp"].iloc[0])

            lag_done = False
            msg = "(nosync)"
            for key in Ikeys_ref:
                print(f"**** {key} ****", flush=True)
                tables = []
                headers = [
                    "P",
                    "count",
                    "mean",
                    "std",
                    "min",
                    "25%",
                    "50%",
                    "75%",
                    "max",
                ]
                signature = overview_dict[filename]["signature"][key]
                t0 = overview_dict[filename]["t0"]
                for i, regime in enumerate(signature.regimes):
                    istart = signature.changes[i]
                    iend = df_overview.index.to_list()[-1]
                    if i < len(signature.changes) - 1:
                        iend = signature.changes[i + 1]

                    if regime == "P":
                        table = [i] + df_overview[key].iloc[
                            istart:iend
                        ].describe().to_list()
                        tables.append(table)
                print(tabulate(tables, headers=headers, tablefmt="psql"))
                if args.debug:
                    print(f"regimes (overview[{key}]):", signature.regimes)

                plot_data(
                    df_overview,
                    df_archive,
                    df_pupitre,
                    df_incidents,
                    channels_dict,
                    pupitre_dict,
                    site,
                    args.tkey,
                    key,
                    filename,
                    msg,
                    args,
                )

                # get lag from 1st U sequence
                if args.lag:
                    osignature = Signature.from_df(
                        filename,
                        t0,
                        df_overview,
                        channels_dict[key],
                        symbol,
                        unit,
                        tkey="t",
                        threshold=threshold_dict[channels_dict[key]],
                        timeshift=0,
                        show=False,
                        debug=args.debug,
                    )
                    psignature = Signature.from_df(
                        filename,
                        t0,
                        df_pupitre,
                        pupitre_dict[site][key],
                        symbol,
                        unit,
                        tkey="t",
                        threshold=threshold_dict[pupitre_dict[site][key]],
                        timeshift=0,
                        show=False,
                        debug=args.debug,
                    )
                    if args.debug:
                        print(
                            f"regimes (pupitre[{pupitre_dict[site][key]}]):",
                            psignature.regimes,
                        )
                        print(
                            f"times (overview[{channels_dict[key]}]):", osignature.times
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
                        tscore = float("inf")
                        lags = (float("inf"), float("inf"))
                        if reference_regime == regime:
                            start_diff = abs(value[0] - reference_value[0])
                            end_diff = abs(value[1] - reference_value[1])
                            score = start_diff + end_diff
                            score = abs(
                                abs(value[1] - value[0])
                                - abs(reference_value[1] - reference_value[0])
                            )

                            tscore = abs(
                                abs(time[1] - time[0])
                                - abs(reference_time[1] - reference_time[0])
                            )
                            start_lag = time[0] - reference_time[0]
                            end_lag = time[1] - reference_time[1]
                            lags = (start_lag, end_lag)

                        return (score, tscore, lags)

                    def find_best_matching_regime(signature, reference_signature):
                        best_matches = []
                        for i, regime in enumerate(signature.regimes):
                            best_score = float("inf")
                            best_lags = (float("inf"), float("inf"))
                            best_index = (0, 0)
                            best_match = None

                            if regime in ["U", "D"] and i <= len(signature.times) - 2:
                                values = (signature.values[i], signature.values[i + 1])
                                times = (signature.times[i], signature.times[i + 1])
                                for j, ref_regime in enumerate(
                                    reference_signature.regimes
                                ):
                                    # get value and time range
                                    if (
                                        ref_regime in ["U", "D"]
                                        and j <= len(reference_signature.times) - 2
                                    ):
                                        ref_values = (
                                            reference_signature.values[j],
                                            reference_signature.values[j + 1],
                                        )
                                        ref_times = (
                                            reference_signature.times[j],
                                            reference_signature.times[j + 1],
                                        )

                                        score, tscore, lags = compute_regime_score(
                                            regime,
                                            values,
                                            times,
                                            ref_regime,
                                            ref_values,
                                            ref_times,
                                        )
                                        # print(i, regime, j, ref_regime, score, tscore, lags)

                                        if score < best_score:
                                            best_score = score
                                            best_match = ref_regime
                                            best_lags = lags
                                            best_index = (i, j)
                                            best_matches.append(
                                                (
                                                    regime,
                                                    best_match,
                                                    best_score,
                                                    best_lags,
                                                    best_index,
                                                )
                                            )
                        return best_matches

                    best_matches = find_best_matching_regime(osignature, psignature)

                    duration = df_pupitre.index.values[-1] - df_pupitre.index.values[0]
                    for regime, best_match, score, lags, best_index in best_matches:
                        rmsg = ""
                        if abs(lags[0]) / duration >= 0.2:
                            rmsg = "!! unlikely !!"

                        """
                        print(f"{key}  Best match for regime {regime} [{best_index[0]}, t={osignature.times[best_index[0]]}] in overview: {best_match} with score {score} and lags {lags} [{best_index[1]}, t={psignature.times[best_index[1]]}] in pupitre (duration={duration}) {rmsg}"
                        )
                        """

                    if not lag_done:
                        print(
                            f"1st lag for {key}: between {channels_dict[key]} - {pupitre_dict[site][key]}"
                        )
                        df1_data = {
                            "df": df_overview.loc[:, ["timestamp", channels_dict[key]]],
                            "field": channels_dict[key],
                            "range": (osignature.changes[0], osignature.changes[2] + 2),
                        }
                        df2_data = {
                            "df": df_pupitre.loc[
                                :, ["timestamp", pupitre_dict[site][key]]
                            ],
                            "field": pupitre_dict[site][key],
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
                        df_pupitre["timestamp"] = df_pupitre[
                            "timestamp"
                        ] - pd.to_timedelta(f"{lag}s")

                        # update timestamp and t
                        pt0 = df_pupitre["timestamp"].iloc[0]
                        print("new t0 (pupitre):", pt0)
                        df_pupitre.drop(["t"], axis=1, inplace=True)
                        df_pupitre["t"] = df_pupitre.apply(
                            lambda row: (row.timestamp - ot0).total_seconds(), axis=1
                        )
                        if args.debug:
                            print(f"df_pupitre with lag:\n{df_pupitre.head()}")

                        # update times for pupitre signature
                        for j in range(len(psignature.times)):
                            psignature.times[j] = (
                                psignature.times[j] - lag.total_seconds()
                            )
                            lag_done = True

                        # find lags at U and D
                        best_matches = find_best_matching_regime(osignature, psignature)
                        for regime, best_match, score, lags, best_index in best_matches:
                            rmsg = ""
                            if abs(lags[0]) / duration >= 0.2:
                                rmsg = "!! unlikely !!"
                            """
                            print(
                            f"{key} 1st lag: Best match for regime {regime} [{best_index[0]}, t={osignature.times[best_index[0]]}] in overview: {best_match} with score {score} and lags {lags} [{best_index[1]}, t={psignature.times[best_index[1]]}] in pupitre (duration={duration}) {rmsg}"
                        )
                            """

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
                        key,
                        filename,
                        msg,
                        args,
                    )

                if args.distance:
                    # compute distance between pupitre and pigbrother
                    # print("df_pupitre:", df_pupitre)
                    xdata = df_pupitre.copy()
                    xdata.set_index("timestamp", inplace=True)
                    print("xdata:\n", xdata.head())
                    xdata_index = xdata.index.to_list()
                    xdata_resampled = xdata.resample("1s", origin=ot0).asfreq()
                    # Interpolate missing values (optional, depending on your use case)
                    xdata_resampled = xdata_resampled.interpolate(method="linear")
                    # xdata_resampled = xdata_resampled + xdata_resampled.min()

                    print("xdata_resampled:\n", xdata_resampled.head())
                    xdata_resampled.set_index("t", inplace=True)
                    # print("after resample xdata_resampled:", xdata_resampled)
                    end_index = xdata_resampled.index.values[-1].astype(int)
                    print(
                        f"\nDistance between pupitre {pupitre_dict[site][key]} and pigbrother {channels_dict[key]} from t=0 to t={end_index} s"
                    )

                    x = (
                        xdata_resampled[pupitre_dict[site][key]]
                        .loc[0:end_index]
                        .to_numpy()
                        .reshape(-1)
                    )
                    y = (
                        df_overview[channels_dict[key]]
                        .loc[0:end_index]
                        .to_numpy()
                        .reshape(-1)
                    )

                    plt.plot(x, label="pupitre", marker=".", color="g")
                    plt.plot(y, label="overview", marker="o", color="r", alpha=0.2)
                    plt.title(f"distance: {key}")
                    plt.legend()
                    plt.grid()
                    plt.show()
                    plt.close()

                    # print('Ib:', x, type(x), x.shape)
                    scipy_stats = stats.describe(y - x)

                    (label, igroup) = key.split("_")
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
                    # see https://medium.com/@markstent/dynamic-time-warping-a8c5027defb6
                    from dtaidistance import dtw
                    from scipy.stats import pearsonr

                    ts_x = df_overview.loc[:, ["t", channels_dict[key]]]
                    ts_x.set_index("t", inplace=True)
                    # TODO no longer working since t_offset on overview data
                    # ts_x.index = ts_x.index.astype(int)
                    print("dtw ts_x:", ts_x)
                    ts_y = xdata_resampled.loc[:, [pupitre_dict[site][key]]]
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
                        {
                            "Metric": ["DTW Similarity Score"],
                            "Value": [similarity_score],
                        }
                    )

                    # Add descriptions for the results
                    results_df["Description"] = [
                        "Lower scores indicate greater similarity between the time series."
                    ]
                    print(results_df)

                    plt.figure(figsize=(12, 8))

                    # Original Time Series Plot
                    ax1 = plt.subplot2grid((2, 2), (0, 0))
                    ax1.plot(ts_x, label=channels_dict[key], color="blue")
                    ax1.plot(
                        ts_y,
                        label=pupitre_dict[site][key],
                        linestyle="--",
                        color="orange",
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
                    ax2.set_xlabel(channels_dict[key])
                    ax2.set_ylabel(pupitre_dict[site][key])
                    ax2.grid(True)

                    # Point-to-Point Comparison Plot
                    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2, sharex=ax1)
                    ax3.plot(ts_x, label=channels_dict[key], color="blue", marker="o")
                    ax3.plot(
                        ts_y,
                        label=pupitre_dict[site][key],
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
                        )
                    ax3.set_title("Point-to-Point Comparison After DTW Alignment")
                    ax3.legend()
                    ax3.grid(True)

                    plt.tight_layout()
                    plt.show()


if __name__ == "__main__":
    main()
