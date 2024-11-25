"""Main module."""

import os
import traceback

from .MagnetRun import MagnetRun
from .processing.smoothers import savgol
from scipy.signal import find_peaks

from tabulate import tabulate

# import logging

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="enter input file")
    args = parser.parse_args()
    print(f"args: {args}", flush=True)

    file = args.input_file

    # load df pandas from input_file
    # check extension
    supported_formats = [".txt"]

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

    Ib = mrun.getData('IB')
    x = Ib.to_numpy().reshape(-1)
    y = mrun.getData('IH').to_numpy().reshape(-1)
    # print('Ib:', x, type(x), x.shape)

    import piecewise_regression
    pw_fit = piecewise_regression.Fit(x, y, n_breakpoints=2)
    pw_fit.summary()
    
    # Plot the data, fit, breakpoints and confidence intervals
    pw_fit.plot_data(color="grey", s=20)
    # Pass in standard matplotlib keywords to control any of the plots
    pw_fit.plot_fit(color="red", linewidth=4)
    pw_fit.plot_breakpoints()
    pw_fit.plot_breakpoint_confidence_intervals()
    plt.xlabel(r'$I_B$ [A]')
    plt.ylabel(r'$I_H$ [A]')
    plt.grid()
    plt.show()
    plt.close()