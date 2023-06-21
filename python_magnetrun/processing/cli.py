"""
Locally Weighted Linear Regression (Loess)

see:
https://xavierbourretsicotte.github.io/loess.html
"""

import os
import sys

from math import ceil
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

##from IPython.display import Image
##from IPython.display import display
# plt.style.use('seaborn-white')
## if jupyter: %matplotlib inline

import statsmodels.api as sm
from .filters import filterpikes
from .correlations import lagged_correlation
from .smoothers import lowess_ag, lowess_sm, lowess_bell_shape_kern, kernel_function
from ..MagnetRun import MagnetRun

if __name__ == "__main__":
    import argparse
    from .. import python_magnetrun

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument(
        "--keys", help="specify keys to select (eg: Tin1;Tin2)", default="Tin1"
    )
    parser.add_argument(
        "--show",
        help="display graphs (requires X11 server active)",
        action="store_true",
    )
    parser.add_argument("--debug", help="activate debug mode", action="store_true")

    # define subparser: filter, smooth, lag_correlation
    subparsers = parser.add_subparsers(
        title="commands", dest="command", help="sub-command help"
    )

    # parser_plot = subparsers.add_parser('plot', help='plot help')
    parser_filter = subparsers.add_parser("filter", help="filter help")
    parser_smooth = subparsers.add_parser("smooth", help="smooth help")
    parser_lag = subparsers.add_parser("lag", help="lag help")

    parser_filter.add_argument(
        "--threshold", help="specify a threshold for filter", type=float, default=0.5
    )
    parser_filter.add_argument(
        "--twindows", help="specify a window length", type=int, default=10
    )

    parser_smooth.add_argument(
        "--method",
        help="select a smoother for data",
        type=str,
        choices=["ag", "bell_kernel", "statsmodel_sm", "all"],
        default="bell_kernel",
    )
    parser_smooth.add_argument(
        "--smooth_params",
        help='pass param for smoother method (eg "f;iter")',
        type=str,
        default="400",
    )
    # parser.add_argument("--smoothing_f", help="specify smoothing f param", type=float, default=0.25)
    # parser.add_argument("--smoothing_tau", help="specify smoothing tau param", type=float, default=0.005)
    # parser.add_argument("--smoothing_iter", help="specify smoothing iter param", type=int, default=5)

    parser_lag.add_argument(
        "--target", help="specify a target field", type=str, default="tsb"
    )
    parser_lag.add_argument(
        "--trange", help="specify a range for t", type=int, default=100
    )

    args = parser.parse_args()

    threshold = 0.5
    twindows = 10
    if args.command == "filter":
        threshold = args.threshold
        twindows = args.twindows

    smoothing_f = 0.7
    smoothing_tau = 400
    smoothing_iter = 3
    if args.command == "smooth":
        params = args.smooth_params.split(";")
        if args.method == "ag":
            smoothing_f = float(params[0])
            smoothing_iter = int(params[1])
        elif args.method == "bell_kernel":
            smoothing_tau = float(params[0])
        elif args.method == "statsmodel_sm":
            smoothing_tau = float(params[0])
        else:
            smoothing_tau = float(params[0])
            smoothing_f = float(params[1])
            smoothing_iter = int(params[2])

    f_extension = os.path.splitext(args.input_file)[-1]
    if f_extension != ".txt":
        print("so far only txt file support is implemented")
        sys.exit(0)

    housing = args.site
    filename = os.path.basename(args.input_file)
    result = filename.startswith("M")
    if result:
        try:
            index = filename.index("_")
            args.site = filename[0:index]
            print("site detected: %s" % args.site)
        except:
            print("no site detected - use args.site argument instead")
            pass
    mrun = MagnetRun.fromtxt(housing, args.site, filename)
    mdata = mrun.getMData()
    mdata.addTime()
    start_timestamp = mdata.getStartDate()
    dkeys = mrun.getKeys()

    inplace = False
    skeys = args.keys.split(";")
    if args.command == "filter":
        for key in skeys:
            filterpikes(
                mrun,
                key,
                inplace,
                threshold,
                twindows,
                args.debug,
                args.show,
                args.input_file,
            )

    if args.command == "smooth":
        for key in skeys:
            selected_df = mrun.getMData().extractData(["t", key])
            Meanval = selected_df[key].mean()

            # Initializing noisy non linear data
            x = selected_df["t"].to_numpy()  # np.linspace(0,1,100)
            y = selected_df[key].to_numpy()  # np.sin(x * 1.5 * np.pi )

            print("display Weighted Linear Regression")
            plt.figure(figsize=(10, 6))
            plt.scatter(x, y, facecolors="none", edgecolor="darkblue", label=key)

            #
            print("compute Locally Weighted Linear Regression %s" % args.method)
            if args.method == "ag":
                try:
                    yest = lowess_ag(x, y, f=smoothing_f, iter=smoothing_iter)
                    plt.plot(x, yest, color="orange", label="Loess: A. Gramfort")
                except:
                    print("Failed to build lowess_ag")

            if args.method == "bell_kernel":
                try:
                    yest_bell = lowess_bell_shape_kern(x, y, smoothing_tau)
                    if args.debug:
                        x0 = (x[0] + x[40]) / 2.0
                        plt.fill(
                            x[:40],
                            Meanval * kernel_function(x[:40], x0, args.smoothing_tau),
                            color="lime",
                            alpha=0.5,
                            label="Bell shape kernel",
                        )
                        plt.plot(
                            x, yest_bell, color="red", label="Loess: bell shape kernel"
                        )
                except:
                    print("Failed to build bell")

            if args.method == "statsmodel_sm":
                try:
                    # f = 0.7 # between 0 and 1, by default: 2./3.
                    yest_sm = lowess_sm(x, y, f=smoothing_f, iter=smoothing_iter)
                    plt.plot(
                        x, yest_sm, color="magenta", label="Loess: statsmodel"
                    )  # marker="o",
                except:
                    print("Failed to build sm")

            plt.legend()
            plt.title("Loess regression comparisons")
            if args.show:
                plt.show()
            else:
                imagefile = mrun.getSite()
                start_date = ""
                start_time = ""
                if "Date" in dkeys and "Time" in dkeys:
                    tformat = "%Y.%m.%d %H:%M:%S"
                    start_date = mrun.getMData().getData("Date").iloc[0]
                    start_time = mrun.getMData().getData("Time").iloc[0]

                plt.savefig(
                    "%s_%s---%s-smoothed-%s.png"
                    % (imagefile, str(start_date), str(start_time), key),
                    dpi=300,
                )
            plt.close()

    if args.command == "lag":
        for key in skeys:
            df = mrun.getData()
            for t in range(args.trange):
                lagged_correlation(df, args.target, key, t)
