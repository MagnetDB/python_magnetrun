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
    parser.add_argument(
        "--breakpoints",
        type=int,
        help="set number of breakpoints",
        default=2,
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["pwlf", "piecewise_regression", "ruptures"],
        help="set breakpoint algo (pwfl, piecewise)",
        default="pwfl",
    )
    parser.add_argument(
        "--save", help="save graphs (png format)", action="store_true"
    )
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

    x = mrun.getData(args.xkey).to_numpy().reshape(-1)
    y = mrun.getData(args.ykey).to_numpy().reshape(-1)
    # print('Ib:', x, type(x), x.shape)

    # perform fit while pw_fit.summary['converged'] == True
    if args.algo == "piecewise_regression":
        import piecewise_regression
        breakpoints=args.breakpoints
    
        converged = True
        while converged:
            ms = piecewise_regression.ModelSelection(x, y, max_breakpoints=breakpoints, n_boot=20,)
            for i,model_summary in enumerate(ms.model_summaries):
                if model_summary['converged'] == False:
                    breakpoints = i
                    converged = False
                    break
        
            if converged:
                # min_distance_between_breakpoints=0.01,min_distance_to_edge=0.02
                pw_fit = piecewise_regression.Fit(x, y, n_breakpoints=breakpoints)
                pw_fit.summary()
                    
                # Plot the data, fit, breakpoints and confidence intervals
                pw_fit.plot_data(color="grey", s=20)
                # Pass in standard matplotlib keywords to control any of the plots
                pw_fit.plot_fit(color="red", linewidth=4)
                pw_fit.plot_breakpoints()
                pw_fit.plot_breakpoint_confidence_intervals()

                mdata = mrun.getMData()
                (symbol, unit) = mdata.getUnitKey(args.xkey)
                plt.xlabel(f"{symbol} [{unit:~P}]")

                (symbol, unit) = mdata.getUnitKey(args.ykey)
                plt.ylabel(f"{symbol} [{unit:~P}]")

                plt.title(f"{file}: {args.ykey} vs {args.xkey} (breakpoints={breakpoints})")
                plt.grid()

                if not args.save:
                    plt.show()
                else:
                    imagefilename = "image"
                    print(f"saveto: {imagefilename}_{args.ykey}_vs_{args.xkey}.png", flush=True)
                    plt.savefig(f"{imagefilename}_{args.ykey}_vs_{args.xkey}.png", dpi=300)

                plt.close()
            
            breakpoints += 1

    # use pwlf
    if args.algo == "pwlf":
        import pwlf

        myPWLF = pwlf.PiecewiseLinFit(x,y)  
        #   fit the data for four line segments
        res = myPWLF.fit(args.breakpoints)

        #   predict for the determined points
        xHat = np.linspace(min(x), max(x), num=10000)
        yHat = myPWLF.predict(xHat)

        # plot the results
        plt.figure()
        plt.plot(x, y, 'o')
        plt.plot(xHat, yHat, '-')

        mdata = mrun.getMData()
        (symbol, unit) = mdata.getUnitKey(args.xkey)
        plt.xlabel(f"{symbol} [{unit:~P}]")

        (symbol, unit) = mdata.getUnitKey(args.ykey)
        plt.ylabel(f"{symbol} [{unit:~P}]")

        plt.title(f"{file}: {args.ykey} vs {args.xkey} (breakpoints={args.breakpoints})")
        plt.grid()
        plt.show()

    if args.algo == "ruptures":
        import ruptures as rpt
        #algo = rpt.Dynp(model="l2").fit(y)
        algo_c = rpt.KernelCPD(kernel="rbf", min_size=2).fit(y)
        result = algo_c.predict(pen=100)

        print(result)
        rpt.display(y, result)
        plt.show()