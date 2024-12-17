"""Main module."""

import os
import traceback

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
    parser.add_argument(
        "--breakpoints",
        type=int,
        help="set number of breakpoints",
        default=1,
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["piecewise_aggregation", "pwlf", "piecewise_regression", "ruptures"],
        help="set breakpoint algo (pwfl, piecewise)",
        default="pwfl",
    )
    parser.add_argument(
        "--normalize", help="normalize data before plot", action="store_true"
    )
    parser.add_argument("--save", help="save graphs (png format)", action="store_true")
    parser.add_argument("--debug", help="activate debug mode", action="store_true")
    parser.add_argument(
        "--find", help="try to automatically find breakpoints", action="store_true"
    )
    args = parser.parse_args()
    print(f"args: {args}", flush=True)

    # load df pandas from input_file
    # check extension
    supported_formats = [".txt", ".tdms"]

    xkey = args.xkey
    ykey = args.ykey

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
        if result:
            try:
                index = filename.index("_")
                site = filename[0:index]
                # print(f"site detected: {site}")
            except Exception:
                print("no site detected - use args.site argument instead")
                pass

        match f_extension:
            case ".txt":
                mrun = MagnetRun.fromtxt(site, insert, file)
            case ".tdms":
                mrun = MagnetRun.fromtdms(site, insert, file)

                # check keys
                (group, channel) = ykey.split("/")
                if "/" not in xkey:
                    xkey = f"{group}/{xkey}"
            case _:
                raise RuntimeError(
                    f"so far file with extension in {supported_formats} are implemented"
                )

        ydata = mrun.getData(ykey)
        (ysymbol, yunit) = mrun.getUnit(ykey)
        ylabel = f"{ysymbol} [{yunit:~P}]"
        y = ydata.to_numpy().reshape(-1)

        xdata = mrun.getData(xkey)
        (xsymbol, xunit) = mrun.getUnit(xkey)
        xlabel = f"{xsymbol} [{xunit:~P}]"
        x = xdata.to_numpy().reshape(-1)

        if xsymbol == ysymbol:
            ylabel = f"{ykey} [{yunit:~P}]"
            xlabel = f"{xkey} [{xunit:~P}]"

        if args.normalize:
            ymax = y.max()
            y /= abs(ymax)
            ylabel = f"{ymax} [{yunit:~P}]"
            if "/" in xkey:
                (group, channel) = xkey.split("/")
                if channel != "t":
                    xmax = x.max()
                    x /= abs(xmax)
                    xlabel = f"{xmax} [{xunit:~P}]"
                    break
            else:
                if xkey != "t":
                    xmax = x.max()
                    x /= abs(xmax)
                    xlabel = f"{xmax} [{xunit:~P}]"

        # y as f(x)
        if args.debug:
            plt.figure()
            plt.plot(x, y, "o")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid()
            plt.show()
            plt.close()

        if args.algo == "piecewise_aggregation":
            from pyts.approximation import PiecewiseAggregateApproximation
            from scipy import stats

            # Parameters
            # n_samples, n_timestamps = 100, 48 for toy pb
            n_samples, n_timestamps = 1, y.shape[0]

            # Toy dataset
            # rng = np.random.RandomState(41)
            # X = rng.randn(n_samples, n_timestamps)
            # X = mrun.getData(ykey)
            X = np.array([y])
            print(f"X: {X.shape}")

            # PAA transformation
            window_size = 4
            paa = PiecewiseAggregateApproximation(
                window_size=window_size, overlapping=False
            )
            X_paa = paa.transform(X)
            print(f"X_paa: {X_paa[0].shape}")

            # Show the results for the first time series
            plt.figure(figsize=(6, 4))
            plt.plot(X[0], "o--", ms=2, label="Original")
            plt.plot(
                np.arange(
                    window_size // 2, n_timestamps + window_size // 2, window_size
                ),
                X_paa[0],
                "o--",
                ms=2,
                label="PAA",
            )
            """
            plt.vlines(
                np.arange(0, n_timestamps, window_size) - 0.5,
                X[0].min(),
                X[0].max(),
                color="g",
                linestyles="--",
                linewidth=0.5,
            )
            """
            plt.legend(loc="best", fontsize=10)
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.title("Piecewise Aggregate Approximation", fontsize=16)
            plt.grid()
            plt.show()

            # DFT transformation
            from pyts.approximation import DiscreteFourierTransform

            n_coefs = y.shape[0] // 4
            dft = DiscreteFourierTransform(
                n_coefs=n_coefs, norm_mean=False, norm_std=False
            )
            X_dft = dft.fit_transform(X)

            # Compute the inverse transformation
            if n_coefs % 2 == 0:
                real_idx = np.arange(1, n_coefs, 2)
                imag_idx = np.arange(2, n_coefs, 2)
                X_dft_new = np.c_[
                    X_dft[:, :1],
                    X_dft[:, real_idx]
                    + 1j * np.c_[X_dft[:, imag_idx], np.zeros((n_samples,))],
                ]
            else:
                real_idx = np.arange(1, n_coefs, 2)
                imag_idx = np.arange(2, n_coefs + 1, 2)
                X_dft_new = np.c_[
                    X_dft[:, :1], X_dft[:, real_idx] + 1j * X_dft[:, imag_idx]
                ]
            X_irfft = np.fft.irfft(X_dft_new, n_timestamps)

            # Show the results for the first time series
            plt.figure(figsize=(6, 4))
            plt.plot(X[0], "o--", ms=4, label="Original")
            plt.plot(X_irfft[0], "o--", ms=4, label="DFT - {0} coefs".format(n_coefs))
            plt.legend(loc="best", fontsize=10)
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.title("Discrete Fourier Transform", fontsize=16)
            plt.show()

            # SAX
            import matplotlib.lines as mlines
            from scipy.stats import norm
            from pyts.approximation import SymbolicAggregateApproximation

            X -= abs(X.mean())
            # SAX transformation
            n_bins = 3
            # strategy in ['uniform', 'quantile', 'normal']
            sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy="quantile")
            X_sax = sax.fit_transform(X)

            # Compute gaussian bins (for strategy==normal)
            print(stats.describe(X[0]))
            print(np.linspace(0, 1, n_bins + 1)[1:-1])
            bins = norm.ppf(np.linspace(0, 1, n_bins + 1)[1:-1])
            print(bins.tolist())

            # Show the results for the first time series
            bottom_bool = np.r_[True, X_sax[0, 1:] > X_sax[0, :-1]]

            plt.figure(figsize=(6, 4))
            plt.plot(X[0], "o--", label="Original")
            for x, y, s, bottom in zip(
                range(n_timestamps), X[0], X_sax[0], bottom_bool
            ):
                va = "bottom" if bottom else "top"
                plt.text(x, y, s, ha="center", va=va, fontsize=14, color="#ff7f0e")
            plt.hlines(bins, 0, n_timestamps, color="g", linestyles="--", linewidth=0.5)
            sax_legend = mlines.Line2D(
                [],
                [],
                color="#ff7f0e",
                marker="*",
                label="SAX - {0} bins".format(n_bins),
            )
            first_legend = plt.legend(
                handles=[sax_legend], fontsize=8, loc=(0.76, 0.86)
            )
            ax = plt.gca().add_artist(first_legend)
            plt.legend(loc=(0.81, 0.93), fontsize=8)
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.title("Symbolic Aggregate approXimation", fontsize=16)
            plt.show()

        # perform fit while pw_fit.summary['converged'] == True
        if args.algo == "piecewise_regression":
            import piecewise_regression

            breakpoints = args.breakpoints

            converged = True
            while converged:
                if args.find:
                    ms = piecewise_regression.ModelSelection(
                        x,
                        y,
                        max_breakpoints=breakpoints,
                        n_boot=100,
                    )
                    for i, model_summary in enumerate(ms.model_summaries):
                        if model_summary["converged"] == False:
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
                    (symbol, unit) = mdata.getUnitKey(xkey)
                    plt.xlabel(f"{symbol} [{unit:~P}]")

                    (symbol, unit) = mdata.getUnitKey(ykey)
                    plt.ylabel(f"{symbol} [{unit:~P}]")

                    plt.title(
                        f"{file}: {args.ykey} vs {xkey} (breakpoints={breakpoints})"
                    )
                    plt.grid()

                    if not args.save:
                        plt.show()
                    else:
                        imagefilename = file.replace(f_extension, "")
                        print(
                            f"saveto: {imagefilename}_{ykey}_vs_{xkey}-piecewise.png",
                            flush=True,
                        )
                        plt.savefig(
                            f"{imagefilename}_{ykey}_vs_{xkey}-piecewise.png",
                            dpi=300,
                        )

                    plt.close()

                breakpoints += 1
                if not args.find:
                    converged = False

        # use pwlf
        if args.algo == "pwlf":
            import pwlf

            myPWLF = pwlf.PiecewiseLinFit(x, y)
            #   fit the data for four line segments
            res = myPWLF.fit(args.breakpoints)

            #   predict for the determined points
            xHat = np.linspace(min(x), max(x), num=10000)
            yHat = myPWLF.predict(xHat)

            # plot the results
            plt.figure()
            plt.plot(x, y, "o")
            plt.plot(xHat, yHat, "-")

            mdata = mrun.getMData()
            (symbol, unit) = mdata.getUnitKey(xkey)
            plt.xlabel(f"{symbol} [{unit:~P}]")

            (symbol, unit) = mdata.getUnitKey(ykey)
            plt.ylabel(f"{symbol} [{unit:~P}]")

            plt.title(f"{file}: {args.ykey} vs {xkey} (breakpoints={args.breakpoints})")
            plt.grid()
            if not args.save:
                plt.show()
            else:
                imagefilename = file.replace(f_extension, "")
                print(
                    f"saveto: {imagefilename}_{ykey}_vs_{xkey}-pwlf.png",
                    flush=True,
                )
                plt.savefig(f"{imagefilename}_{ykey}_vs_{xkey}-pwlf.png", dpi=300)
            plt.close()

        if args.algo == "ruptures":
            from scipy import stats
            import ruptures as rpt

            # ?? replace y index by x if xkay != t??

            # change point detection
            model = "l2"  # ""l2", l1", "rbf", "linear", "normal", "ar"
            sigma = ydata.rolling(window=10).std()
            print(f"sigma: {sigma.describe()}")
            sigma_min = float(sigma.mean().iloc[0])
            print(f"sigma: {sigma_min}")

            n = len(y)
            pen = np.log(n) * sigma_min
            epsilon = 3 * sigma_min
            print(f"pen={pen}")
            print(f"epsilon={epsilon}")
            # algo = rpt.KernelCPD(kernel="linear", min_size=20).fit(y)
            # algo = rpt.Window(width=100, model=model).fit(y)
            # algo = rpt.Dynp(model="l2").fit(y) # breakingpoints must be none
            algo = rpt.Pelt(model="l2", min_size=10, jump=1).fit(y)
            # algo = rpt.Binseg(model="linear").fit(y) # fails (signal.ndim > 1, "Not enough dimensions")
            # algo = rpt.BottomUp(model="linear").fit(y) # fails (signal.ndim > 1, "Not enough dimensions")
            result = algo.predict(pen=pen)
            # result = algo.predict(epsilon=epsilon)

            print(f"result: {len(result)} breakpoints ({result})")
            rpt.display(y, result)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if not args.save:
                plt.show()
            else:
                imagefilename = file.replace(f_extension, "")
                print(
                    f"saveto: {imagefilename}_{ykey}_vs_{xkey}-kernelcpd.png",
                    flush=True,
                )
                plt.savefig(f"{imagefilename}_{ykey}_vs_{xkey}-kernelcpd.png", dpi=300)
            plt.close()
