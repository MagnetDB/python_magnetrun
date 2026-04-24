"""Compare 2 timeseries."""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from scipy import stats
from tabulate import tabulate

from python_magnetrun.cli_args import create_base_parser
from python_magnetrun.log_utils import get_logger, setup_logging
from python_magnetrun.MagnetRun import MagnetRun
from python_magnetrun.processing.smoothers import lowess_bell_shape_kern

logger = get_logger()

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
    base_parser = create_base_parser()
    parser = argparse.ArgumentParser(parents=[base_parser])
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
        "--lagcorrelation", help="estimate lagcorrelation", action="store_true"
    )
    parser.add_argument(
        "--range", help="select data from (index value)", type=int, nargs=2
    )
    parser.add_argument("--to", help="select data to (index value)", type=int)
    parser.add_argument("--dtw", help="use dtw", action="store_true")
    parser.add_argument("--save", help="save graphs (png format)", action="store_true")
    parser.add_argument("--outputdir", type=str, help="enter output directory")
    args = parser.parse_args()
    setup_logging(level=args.log_level, log_file=args.log_file)
    logger.debug("args: %s", args)

    xkey = args.xkey
    ykey = args.ykey

    import re

    # Todo change title and ... if tdms data

    labels = [xkey, ykey]
    for i, label in enumerate(labels):
        regex = r"(.*?)/"  # before
        data = re.findall(regex, label)
        if data:
            labels[i] = label.replace(f"{data[0]}/", "")

    print(f"Compare: {xkey} and {ykey}")
    tables = []
    headers = [
        "Name",
        "duration [s]",
        "Euclidean",
        "MAE",
        "Pearson",
        "Image",
        f"<{labels[1]} - {labels[0]}>",
        "min",
        "max",
        "var",
    ]

    if args.outputdir:
        output = args.outputdir

    supported_formats = [".txt", ".tdms"]
    input_files = natsorted(args.input_file)
    for file in input_files:
        f_extension = os.path.splitext(file)[-1]
        if f_extension not in supported_formats:
            raise RuntimeError(
                f"so far file with extension in {supported_formats} are implemented"
            )

        filename = os.path.basename(file)
        result = filename.startswith("M")
        insert = args.insert if args.insert else None
        housing = args.housing if args.housing else None
        site = args.site if args.site else None
        if result:
            try:
                index = filename.index("_")
                housing = filename[0:index] if housing is None else housing
                print(f"housing detected: {housing}")
            except ValueError:
                print("no housing detected - use args.housing argument instead")
                pass

        match f_extension:
            case ".txt":
                mrun = MagnetRun.fromtxt(housing=housing, site=site, filename=filename)
            case ".tdms":
                mrun = MagnetRun.fromtdms(housing=housing, site=site, filename=filename)
            case _:
                raise RuntimeError(
                    f"so far file with extension in {supported_formats} are implemented"
                )

        fig, my_ax = plt.subplots()
        mrun.getMData().plotData(x="t", y=xkey, ax=my_ax)
        mrun.getMData().plotData(x="t", y=ykey, ax=my_ax)

        dirname = os.path.dirname(file)
        imagefile = f"{os.path.splitext(file)[0].replace(f_extension, '')}-{labels[0]}-{labels[1]}"
        if args.outputdir:
            print(f"imagefile: {imagefile}, ", end="")
            print(f"replace {dirname} by {args.outputdir}, ", end="")
            imagefile = imagefile.replace(dirname, args.outputdir)
            print(f"-> {imagefile}")

        my_ax.set_title(f"{filename.replace(f_extension, '')}: {labels[0]}, {labels[1]}")
        if args.save:
            print(f"saveto: {imagefile}_vs_time.png", flush=True)
            fig.savefig(f"{imagefile}_vs_time.png", dpi=300)
        else:
            plt.show()
        plt.close(fig)

        xdata = mrun.getData(xkey)
        ydata = mrun.getData(ykey)

        print(f"xdata={xkey}: keys={xdata.keys()}")
        if args.range:
            (start, end) = args.range
            print(f"range=[start={start}, end={end}]")
            if f_extension == ".txt":
                xdata = xdata[start:end]
                ydata = ydata[start:end]

                timesteps = np.arange(start, end, 1)
                print(type(timesteps), type(ydata))
                ysmoothed = lowess_bell_shape_kern(
                    xdata.to_numpy().reshape(-1), ydata.to_numpy().reshape(-1)
                )

                fig_r, ax_r = plt.subplots()
                ax_r.plot(xdata / xdata.max(), "-")
                ax_r.plot(ysmoothed / ysmoothed.max(), "o")
                ax_r.plot(ydata / ydata.max(), "+")
                ax_r.grid()
                plt.show()
                plt.close(fig_r)

            else:
                raise RuntimeError("range features: not impemented for tdms data")

        x = xdata.to_numpy().reshape(-1)
        y = ydata.to_numpy().reshape(-1)
        # print('Ib:', x, type(x), x.shape)
        scipy_stats = stats.describe(y - x)
        # print(scipy_stats.minmax) # minmax: tuple, mean, variance

        table = [
            filename.replace(f_extension, ""),
            mrun.getMData().getDuration(),
            calc_euclidean(x, y),
            calc_mape(x, y),
            calc_correlation(x, y),
            f"{imagefile}_vs_time.png",
            scipy_stats.mean,
            scipy_stats.minmax[0],
            scipy_stats.minmax[1],
            scipy_stats.variance,
        ]
        tables.append(table)

        if args.lagcorrelation:
            (ysymbol, yunit) = mrun.getUnit(ykey)
            ylabel = f"{ysymbol} [{yunit:~P}]"
            ymax = y.max()

            (xsymbol, xunit) = mrun.getUnit(xkey)
            xlabel = f"{xsymbol} [{xunit:~P}]"
            xmax = x.max()

            y /= abs(ymax)
            ylabel = f"{labels[1]}: mean={np.mean(y):.3f}, {ymax:.3f} [{yunit:~P}]"
            if "/" in xkey:
                (group, channel) = xkey.split("/")
                if channel != "t":
                    x /= abs(xmax)
                    xlabel = f"{labels[0]}: mean={np.mean(x):.3f}, max={xmax:.3f} [{xunit:~P}]"
            else:
                if xkey != "t":
                    x /= abs(xmax)
                    xlabel = f"{labels[0]}: mean={np.mean(x):.3f}, max={xmax:.3f} [{xunit:~P}]"

            # investigate lag correlation
            from scipy import signal

            correlation = signal.correlate(x - np.mean(x), y - np.mean(y), mode="full")
            lags = signal.correlation_lags(len(x), len(y), mode="full")
            lag = lags[np.argmax(abs(correlation))]
            print(
                f"cross-correlation lag={lag}",
                flush=True,
            )

            fig_lag, ax_lag = plt.subplots()
            ax_lag.plot(x, "-")
            ax_lag.plot(np.roll(y, lag), "-")
            ax_lag.legend([xlabel, ylabel])
            ax_lag.set_xlabel("t [s]")
            ax_lag.grid()
            ax_lag.set_title(
                f"{os.path.basename(file)}: {labels[0]}, {labels[1]} - cross-correlation lag={lag}"
            )
            if args.save:
                print(f"saveto: {imagefile}_vs_lagcorrelation.png", flush=True)
                fig_lag.savefig(f"{imagefile}_vs_lagcorrelation.png", dpi=300)
            else:
                plt.show()
            plt.close(fig_lag)

        if args.dtw:
            # dtw algo
            from dtaidistance import dtw
            from dtaidistance import dtw_visualisation as dtwvis

            try:
                d, paths = dtw.warping_paths(x, y, window=25, psi=2)
                best_path = dtw.best_path(paths)
                dtwvis.plot_warpingpaths(x, y, paths, best_path)
                fig_dtw = plt.gcf()
                print(f"dtw distance: {d}")
                if args.save:
                    print(f"saveto: {imagefile}_vs_dtw.png", flush=True)
                    fig_dtw.savefig(f"{imagefile}_vs_dtw.png", dpi=300)
                else:
                    plt.show()
            except MemoryError:
                print("!!! retry pyts piecewise aggregate approx !!!")

                # PAA transformation
                from pyts.approximation import PiecewiseAggregateApproximation

                n_samples, n_timestamps = 1, y.shape[0]
                window_size = 125
                paa = PiecewiseAggregateApproximation(
                    window_size=window_size, overlapping=False
                )
                try:
                    x_paa = paa.transform(np.array([x]))
                    y_paa = paa.transform(np.array([y]))

                    # Show the results for the first time series
                    fig_paa, ax_paa = plt.subplots()
                    ax_paa.plot(x, "o--", ms=4, label="Original")
                    ax_paa.plot(
                        np.arange(
                            window_size // 2,
                            n_timestamps + window_size // 2,
                            window_size,
                        ),
                        x_paa[0],
                        "o--",
                        ms=4,
                        label="x PAA",
                    )
                    ax_paa.set_ylabel(xlabel)
                    ax_paa.set_xlabel("t [s]")
                    ax_paa.grid()
                    ax_paa.set_title(
                        f"{os.path.basename(file)}: {labels[0]} - PiecewiseAggegateApprox (windows=125)"
                    )
                    if args.save:
                        print(f"saveto: {imagefile}_paa.png", flush=True)
                        fig_paa.savefig(f"{imagefile}_paa.png", dpi=300)
                    else:
                        plt.show()
                    plt.close(fig_paa)

                    d, paths = dtw.warping_paths(x_paa[0], y_paa[0], window=125, psi=2)
                    best_path = dtw.best_path(paths)
                    dtwvis.plot_warpingpaths(x_paa[0], y_paa[0], paths, best_path)
                    fig_dtw_paa = plt.gcf()
                    print(f"dtw distance: {d}")
                    if args.save:
                        print(f"saveto: {imagefile}_vs_dtw-paa.png", flush=True)
                        fig_dtw_paa.savefig(f"{imagefile}_vs_dtw-paa.png", dpi=300)
                    else:
                        plt.show()
                except (ValueError, RuntimeError) as e:
                    print(f"dtw using paa exception : {e}, type={type(e)}")
                    pass

            except (ValueError, RuntimeError, ImportError) as e:
                print(f"dtw exception: {e}, type={type(e)}")
                pass

    print(tabulate(tables, headers, tablefmt="simple"), "\n")
