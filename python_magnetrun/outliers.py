"""Main module."""

import os
import traceback

from matplotlib.cbook import flatten

from python_magnetrun.processing import breakingpoints

from .MagnetRun import MagnetRun
from .processing.smoothers import savgol
from scipy.signal import find_peaks

from tabulate import tabulate

# import logging
from natsort import natsorted

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available


def mean_mad(x):
    return np.fabs(x - x.mean()).mean()


def mad(x):
    # return np.fabs(x - x.mean()).mean()
    return np.median(np.fabs(x - np.median(x)))  # .median()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="+", help="enter input file")
    parser.add_argument(
        "--site", help="specify a site (ex. M8, M9,...)", default="M9"
    )  # use housing instead
    parser.add_argument("--insert", help="specify an insert", default="notdefined")
    parser.add_argument("--debug", help="acticate debug", action="store_true")
    parser.add_argument("--plot", help="acticate plot", action="store_true")
    parser.add_argument("--save", help="activate plot", action="store_true")
    parser.add_argument("--normalize", help="normalize data", action="store_true")

    parser.add_argument(
        "--threshold",
        help="specify threshold for outliers detection",
        type=float,
        default=2,
    )
    parser.add_argument("--window", help="rolling window size", type=int, default=120)
    args = parser.parse_args()
    print(f"args: {args}", flush=True)

    # load df pandas from input_file
    # check extension
    supported_formats = [".tdms"]

    inputs = {}
    extensions = {}
    for i, file in enumerate(args.input_file):
        f_extension = os.path.splitext(file)[-1]
        if f_extension not in extensions:
            extensions[f_extension] = [i]
        else:
            extensions[f_extension].append(i)
    print(f"extensions: {extensions}")

    input_files = natsorted(args.input_file)
    for file in input_files:
        f_extension = os.path.splitext(file)[-1]
        if f_extension not in supported_formats:
            raise RuntimeError(
                f"so far file with extension in {supported_formats} are implemented"
            )

        filename = os.path.basename(file)
        result = filename.startswith("M")
        insert = "tututu"
        site = "tttt"

        try:
            index = filename.index("_")
            site = filename[0:index]

            match f_extension:
                case ".tdms":
                    mrun = MagnetRun.fromtdms(site, insert, file)
                case _:
                    raise RuntimeError(
                        f"so far file with extension in {supported_formats} are implemented"
                    )
        except Exception as error:
            print(f"{file}: an error occurred when loading:", error)
            continue

        mdata = mrun.getMData()

        for item in [
            (
                "Courants_Alimentations/Référence_GR1",
                "Courants_Alimentations/Courant_GR1",
            ),
            (
                "Courants_Alimentations/Référence_GR2",
                "Courants_Alimentations/Courant_GR2",
            ),
        ]:
            # plot
            if args.plot:
                fig = plt.figure(figsize=(16, 12))

                ax0 = plt.subplot(211)
                ax0.set_title(
                    f'{filename.replace(f_extension,"")}: {item[0]}/{item[1]}'
                )
                ax0.set_xlabel("t [s]")

                ax1 = plt.subplot(212, sharex=ax0)
                ax1.set_title(f"MAD - windows={args.window}")
                ax1.set_xlabel("t [s]")

            ts_outliers = []
            ax1legend = []
            ax0legend = []

            for key in item:
                print(f"pigbrother: stats for {key}", flush=True)
                (symbol, unit) = mdata.getUnitKey(key)

                # compute num_points_threshold from dthresold
                (group, channel) = key.split("/")
                period = mdata.Groups[group][channel]["wf_increment"]

                ts = mdata.Data[group][channel]
                # print(ts, type(ts))

                """
                from hampel import hampel

                result = hampel(ts, window_size=args.window, n_sigma=5.0)
                filtered_data = result.filtered_data
                outlier_indices = result.outlier_indices
                print("outlier_indices: ", outlier_indices)
                medians = result.medians
                mad_values = result.median_absolute_deviations
                print("mad_values: ", mad_values, type(mad_values))
                thresholds = result.thresholds

                outliers = ts[outlier_indices]
                """

                tmax = ts.max()
                if args.normalize:
                    ts /= tmax
                    ts -= ts.mean()
                freq = 1 / mdata.Groups[group][channel]["wf_increment"]
                print(f"{group}/{channel}: freq={freq} Hz", flush=True)

                ts_median = ts.rolling(window=args.window).median()
                ts_mean = ts.rolling(window=args.window).mean()
                ts_std = ts.rolling(window=args.window).std()

                ts_mad = ts.rolling(window=args.window).apply(mad, raw=True)
                ts_mean_mad = ts.rolling(window=args.window).apply(mean_mad, raw=True)

                # find outliers MAD and mean MAD
                selector = ts - ts_median / ts_mad  # Faux
                outliers = ts[selector > args.threshold]

                # selector = ts - ts_mean / ts_mean_mad  # Faux
                # meanoutliers = ts[selector > args.threshold]

                ts_outliers.append(outliers)

                # plot
                if args.plot:
                    ts.plot(ax=ax0)
                    outliers.plot(ax=ax0, marker="x", linestyle="none")
                    # meanoutliers.plot(ax=ax0, marker="o", linestyle="none", mfc="none")
                    ax0.set_ylabel(f"{symbol} [{unit:~P}]")
                    ax0legend.append(f"{key}: mean={tmax} [{unit:~P}]")
                    ax0legend.append("Median MAD")
                    # ax0legend.append("Mean MAD")

                    # ts_mad.plot(ax=ax1)
                    ts_mad.plot(ax=ax1)
                    ts_mad[outliers.index].plot(ax=ax1, marker="x", linestyle="none")
                    # ts_mean_mad[meanoutliers.index].plot(
                    #    ax=ax1, marker="o", linestyle="none", mfc="none"
                    # )
                    ax1legend.append("Estimator Median AD")
                    ax1legend.append("Median AD")
                    # ax1legend.append("Mean AD")

            if args.plot:
                ax0.grid()
                ax1.grid()
                ax0.legend(ax0legend)
                ax1.legend(ax1legend)
                if not args.save:
                    plt.show()
                plt.close()

            # check outliers in actual than are not present in ref
            # shall add some fuzzy params
            ts_ref = ts_outliers[0].index
            ts_actual = ts_outliers[1].index
            res = ts_actual[~ts_actual.isin(ts_ref)]
            print("diff:\n", res)
            ts.plot()
            ts[res].plot(marker="o", linestyle="none", mfc="none")
            plt.grid()
            plt.title(
                f'{filename.replace(f_extension,"")}: {key}: mean={tmax} [{unit:~P}]'
            )
            plt.show()
            plt.close()
