"""Main module."""

import os
import datetime
import numpy as np

from tabulate import tabulate

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten

matplotlib.rcParams["text.usetex"] = True

from .MagnetRun import MagnetRun
from .utils.plateaux import nplateaus

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="enter input file tdms")
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
    parser.add_argument("--debug", help="acticate debug", action="store_true")

    parser.add_argument("--save", help="save graphs (png format)", action="store_true")
    parser.add_argument(
        "--show", help="display graphs (X11 required)", action="store_true"
    )
    parser.add_argument(
        "--threshold",
        help="specify threshold for regime detection",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--dthreshold",
        help="specify duration threshold for regime detection",
        type=float,
        default=10,
    )
    parser.add_argument(
        "--window", help="stopping criteria for nlopt", type=int, default=10
    )
    args = parser.parse_args()
    print(f"args: {args}", flush=True)

    file = args.input_file
    f_extension = os.path.splitext(file)[-1]
    if f_extension != ".tdms":
        raise RuntimeError("so far file with tdms extension are implemented")

    dirname = os.path.dirname(file)
    print(f"dirname: {dirname}")
    filename = os.path.basename(file).replace(f_extension, "")
    result = filename.startswith("M")
    insert = "tututu"
    index = filename.index("_")
    site = filename[0:index]

    mrun = MagnetRun.fromtdms(site, insert, file)
    mdata = mrun.getMData()

    # Channels
    channels_dict = {
        "Référence_GR1": [
            ["Référence_A1", "Courant_A1"],
            ["Référence_A2", "Courant_A2"],
        ],
        "Référence_GR2": [
            ["Référence_A3", "Courant_A3"],
            ["Référence_A4", "Courant_A4"],
        ],
    }

    # check ECO mode: "Référence_GR1" != "Référence_GR2"
    GR = mdata.getData(
        ["Courants_Alimentations/Référence_GR1", "Courants_Alimentations/Référence_GR2"]
    )
    GR["diff"] = GR["Référence_GR2"] - GR["Référence_GR1"]

    if GR["diff"].abs().max() > 5 and not (
        GR["Référence_GR2"].abs().max() <= 1 or GR["Référence_GR1"].abs().max() <= 1
    ):
        print(f"{filename}: ECOmode")
    del GR

    # perform for selected key
    key = f"Courants_Alimentations/{args.key}"
    (symbol, unit) = mdata.getUnitKey(key)
    (group, channel) = key.split("/")
    period = mdata.Groups[group][channel]["wf_increment"]
    num_points_threshold = int(args.dthreshold / period)

    print(f"display plateaus for {key}")
    pdata = nplateaus(
        mdata,
        xField=("t", "t", "s"),
        yField=(key, symbol, unit),
        threshold=args.threshold,
        num_points_threshold=num_points_threshold,
        save=args.save,
        show=args.show,
        verbose=False,
    )

    df_plateaux = pd.DataFrame()
    for entry in ["start", "end", "value"]:
        df_plateaux[entry] = [plateau[entry] for plateau in pdata]
    df_plateaux["duration"] = df_plateaux["end"] - df_plateaux["start"]

    # print only if plateaux
    (nrows, ncols) = df_plateaux.shape
    print(f"df_plateaux: {df_plateaux.shape}")
    if nrows != 0:
        # rename column value using symbol and unit
        df_plateaux.rename(
            columns={
                "start": "start [s]",
                "end": "end [s]",
                "duration": "duration [s]",
                "value": f"value [{unit:~P}]",
            },
            inplace=True,
        )
        columns = list(df_plateaux.keys())
        print(
            tabulate(
                df_plateaux,
                headers="keys",
                tablefmt="psql",
                showindex=False,
            )
        )

        # create a signature of the B profile
    else:
        print(
            f'{file.replace(f_extension,"")}: no peaks detected - duration={mdata.getDuration()}, {mdata.getData(key).describe()}'
        )

    # create signature for Overview
    # U, P, D
    t0 = 0
    signature = str()
    tables = []
    headers = ["plateau", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    for index, row in df_plateaux.iterrows():
        t_start = row["start [s]"]
        t_end = row["end [s]"]
        # print("keys:", mdata.Data[group].keys())
        b = pd.DataFrame(mdata.getTdmsData(group, "Champ_magn"))
        dt = mdata.Groups[group]["Champ_magn"]["wf_increment"]
        b["t"] = b.index * dt
        selected_b = b[(b["t"] > t_start) & (b["t"] < t_end)]

        # NB: Champ_magn in gauss (1.e-4 tesla)
        if args.show:
            selected_b["Champ_magn"].plot()
            (symbol, unit) = mdata.PigBrotherUnits("Champ_magn")
            plt.ylabel(f"{symbol} [{unit:~P}]")
            plt.grid()
            plt.show()

        tables.append([index] + selected_b["Champ_magn"].describe().to_list())

    print("\nMagnetic Field [Gauss]")
    print(
        tabulate(
            tables,
            headers=headers,
            tablefmt="psql",
            showindex=False,
        ),
    )

    # search for Pupitre / Archive files
    pupitre = filename.replace("_Overview_", "_")
    date = pupitre.split("_")
    time = date[1].split("-")
    pupitre_datadir = args.pupitre_datadir
    pupitre_filter = f"{pupitre_datadir}/{site}_20{time[0][0:2]}.{time[0][2:4]}.{time[0][4:]}---{time[1][0:2]}:{time[1][2:]}:*.txt"
    print(f"\npupitre: {pupitre_filter}")

    pigbrother = filename.replace("Overview", "Archive")
    time = filename.split("-")
    archive_datadir = dirname.replace("Overview", "Fichiers_Archive")
    archive_filter = f"{archive_datadir}/{pigbrother.replace(time[1],'*.tdms')}"
    print(f"pigbrother: {archive_filter}")

    default_datadir = dirname.replace("Overview", "Fichiers_Default")
    default_filter = f"{default_datadir}/{pigbrother.replace(time[1],'*.tdms')}"
    print(f"pigbrother default: {default_filter}")
    import glob

    pupitre_files = glob.glob(pupitre_filter)
    archive_files = glob.glob(archive_filter)
    default_files = glob.glob(default_filter)
    print(f"Pupitre files: {pupitre_files}")
    print(f"Archive files: {archive_files}")
    print(f"Default files: {default_files}")

    # merge Archive data
    from natsort import natsorted
    from matplotlib.cbook import flatten

    print("\nMerge Archive files")
    df_dict = {}
    for channel in flatten(channels_dict[args.key]):
        print(channel)
        df_ = []
        for i, file in enumerate(natsorted(archive_files)):
            mrun = MagnetRun.fromtdms(site, insert, file)
            filename = os.path.basename(file).replace(f_extension, "")
            mdata = mrun.getMData()
            df = pd.DataFrame(mdata.getTdmsData(group, channel))
            print(filename)
            t0 = mdata.Groups[group][channel]["wf_start_time"]
            dt = mdata.Groups[group][channel]["wf_increment"]
            df["timestamp"] = [
                np.datetime64(t0).astype(datetime.datetime)
                + datetime.timedelta(0, i * dt)
                for i in df.index.to_list()
            ]
            # print(df.head())
            # print(df.tail())
            df_.append(df)

        df_archive = pd.concat(df_)
        t0 = df_archive.iloc[0]["timestamp"]
        df_archive["t"] = df_archive.apply(
            lambda row: (row.timestamp - t0).total_seconds(),
            axis=1,
        )
        df_archive.drop(["timestamp"], axis=1, inplace=True)

        # print(df_archive.head())
        # print(df_archive.tail())
        df_archive.plot(x="t", y=channel)
        plt.grid()
        if args.show:
            plt.show()
        if args.save:
            plt.savefig(f"{filename}-{channel}-concat.png", dpi=300)
        plt.close()
        df_dict[channel] = df_archive

    # extract plateau data and perform stats on plateau
    print(f"Stats per plateaux - group={group}")
    tables = []
    headers = [
        "plateau",
        "field",
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]

    for index, row in df_plateaux.iterrows():
        if row["duration [s]"] >= 60:
            t_start = row["start [s]"]
            t_end = row["end [s]"]

            my_ax = plt.gca()

            for channel in channels_dict[args.key]:
                for item in channel:
                    df = df_dict[item]
                    selected_df = df[(df["t"] > t_start) & (df["t"] < t_end)]
                    tables.append(
                        [index, item] + selected_df[item].describe().to_list()
                    )
                    # print(selected_df.head())

                    selected_df[item].plot.hist(bins=20, alpha=0.5, ax=my_ax)

                plt.title(
                    f"{filename}: {channel} - plateau: from {t_start} to {t_end} s"
                )
                plt.grid()
                if args.show:
                    plt.show()
                if args.save:
                    plt.savefig(f"{filename}-{'-'.join(channel)}-histo.png", dpi=300)
                plt.close()

    print("\nCourants stats over plateau [A]")
    print(
        tabulate(
            tables,
            headers=headers,
            tablefmt="psql",
            showindex=False,
        ),
    )
