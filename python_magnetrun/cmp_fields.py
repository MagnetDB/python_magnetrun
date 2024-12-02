"""Main module."""

import os
from platform import mac_ver
import traceback

from .MagnetRun import MagnetRun
from .processing.smoothers import savgol
from scipy.signal import find_peaks
from scipy import stats

from tabulate import tabulate

# import logging

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tabulate import tabulate

# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available


def calc_euclidean(actual, predic):
    return np.sqrt(np.sum((actual - predic) ** 2))


def calc_mape(actual, predic):
    return np.mean(np.abs((actual - predic) / actual))


def calc_correlation(actual, predic):
    a_diff = actual - np.mean(actual)
    p_diff = predic - np.mean(predic)
    numerator = np.sum(a_diff * p_diff)
    denominator = np.sqrt(np.sum(a_diff**2)) * np.sqrt(np.sum(p_diff**2))
    return numerator / denominator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="+", help="enter input file")
    parser.add_argument(
        "--xkey",
        type=str,
        help="select xkey",
        default="IH",
    )
    parser.add_argument(
        "--ykey",
        type=str,
        help="select ykey",
        default="IB",
    )
    parser.add_argument("--save", help="save graphs (png format)", action="store_true")
    args = parser.parse_args()
    print(f"args: {args}", flush=True)

    print(f"Compare: {args.xkey} and {args.ykey}")
    print("File Euclidean MAE Pearson")
    tables = []
    headers = ["Name", "Euclidean", "MAE", "Pearson", "Image"]

    supported_formats = [".txt"]
    for file in args.input_file:
        f_extension = os.path.splitext(file)[-1]
        if f_extension not in supported_formats:
            raise RuntimeError(
                f"so far file with extension in {supported_formats} are implemented"
            )

        filename = os.path.basename(file)
        result = filename.startswith("M")
        insert = "tututu"
        if result:
            try:
                index = filename.index("_")
                site = filename[0:index]
                print(f"site detected: {site}")
            except Exception:
                print("no site detected - use args.site argument instead")
                pass

        match f_extension:
            case ".txt":
                mrun = MagnetRun.fromtxt(site, insert, file)
            case _:
                raise RuntimeError(
                    f"so far file with extension in {supported_formats} are implemented"
                )

        x = mrun.getData(args.xkey).to_numpy().reshape(-1)
        y = mrun.getData(args.ykey).to_numpy().reshape(-1)
        # print('Ib:', x, type(x), x.shape)
        scipy_stats = stats.describe(y - x)
        print(scipy_stats.minmax) # minmax: tuple, mean, variance

        my_ax = plt.gca()
        mrun.getMData().plotData(x="t", y=args.xkey, ax=my_ax)
        mrun.getMData().plotData(x="t", y=args.ykey, ax=my_ax)

        imagefile = f"{os.path.splitext(file)[0]}-{args.xkey}-{args.ykey}"
        table = [
            file,
            calc_euclidean(x, y),
            calc_mape(x, y),
            calc_correlation(x, y),
            f"{imagefile}_vs_time.png",
        ]
        tables.append(table)

        plt.title(f"{os.path.basename(file)}: {args.xkey}, {args.ykey}")
        if args.save:
            print(f"saveto: {imagefile}_vs_time.png", flush=True)
            plt.savefig(f"{imagefile}_vs_time.png", dpi=300)
        else:
            plt.show()
        plt.close()

        """
        # dtw algo
        from dtaidistance import dtw
        from dtaidistance import dtw_visualisation as dtwvis
        d, paths = dtw.warping_paths(x, y, window=25, psi=2)
        best_path = dtw.best_path(paths)
        dtwvis.plot_warpingpaths(x, y, paths, best_path)
        print(d)
        plt.show()
        """

    print(tabulate(tables, headers, tablefmt="simple"), "\n")
