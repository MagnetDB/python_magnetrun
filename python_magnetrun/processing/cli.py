"""
Locally Weighted Linear Regression (Loess)

see:
https://xavierbourretsicotte.github.io/loess.html
"""

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

from ..log_utils import SIMPLE_FORMAT, setup_logging
from ..magnetdata_base import DataType, MagnetDataBase
from ..MagnetRun import MagnetRun

##from IPython.display import Image
##from IPython.display import display
# plt.style.use('seaborn-white')
## if jupyter: %matplotlib inline
from .filters import filterpikes
from .smoothers import kernel_function, lowess_ag, lowess_bell_shape_kern, lowess_sm

logger = logging.getLogger(__name__)


def addtime(mdata: MagnetDataBase, group: str, channel: str) -> pd.DataFrame:
    logger.debug("addtime")

    df = pd.DataFrame(mdata.getData(f"{group}/{channel}"))
    t0 = mdata.Groups[group][channel]["wf_start_time"]
    dt = mdata.Groups[group][channel]["wf_increment"]
    df["t"] = [i * dt for i in df.index.to_list()]

    df = df.set_index("t")
    logger.debug(df.head())
    return df


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
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

    parser_filter.add_argument(
        "--threshold", help="specify a threshold for filter", type=float, default=0.5
    )
    parser_filter.add_argument(
        "--twindows", help="specify a window length", type=int, default=10
    )
    parser_filter.add_argument(
        "--keys",
        nargs="+",
        help="specify keys to select (eg: Tin1;Tin2)",
        default="Tin1",
    )

    # smoother
    parser_smooth = subparsers.add_parser("smooth", help="smooth help")
    parser_smooth.add_argument(
        "--keys",
        nargs="+",
        help="specify keys to select (eg: Tin1;Tin2)",
        default="Tin1",
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
        help='pass param for smoother method (eg "tau")',
        nargs="?",
        default="400",
    )
    # parser.add_argument("--smoothing_f", help="specify smoothing f param", type=float, default=0.25)
    # parser.add_argument("--smoothing_tau", help="specify smoothing tau param", type=float, default=0.005)
    # parser.add_argument("--smoothing_iter", help="specify smoothing iter param", type=int, default=5)

    parser_lag = subparsers.add_parser("lag", help="lag help")
    parser_lag.add_argument(
        "--keys",
        nargs="+",
        help="specify keys to select (eg: Tin1;Tin2)",
        default="Tin1",
    )
    parser_lag.add_argument(
        "--target", help="specify a target field", type=str, default="tsb"
    )
    parser_lag.add_argument(
        "--trange", help="specify a range for t", type=int, default=100
    )

    args = parser.parse_args()
    setup_logging(
        level=logging.DEBUG if args.debug else logging.WARNING,
        fmt=SIMPLE_FORMAT,
    )
    logger.debug(f"args: {args}")

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
        elif args.method == "bell_kernel" or args.method == "statsmodel_sm":
            smoothing_tau = float(params[0])
        else:
            smoothing_tau = float(params[0])
            smoothing_f = float(params[1])
            smoothing_iter = int(params[2])

    supported_formats = [".txt", ".tdms"]
    f_extension = os.path.splitext(args.input_file)[-1]
    if f_extension not in supported_formats:
        logger.error("so far only txt file support is implemented")
        sys.exit(0)

    site = "tutut"  # shall be an id from magnetdb
    housing = args.housing if args.housing is not None else "notdefined"
    filename = os.path.basename(args.input_file)
    result = filename.startswith("M")
    if result:
        try:
            index = filename.index("_")
            housing = filename[0:index]
            logger.info(
                f"housing detected: {housing}  -- overwrite args.housing={args.housing}"
            )
        except ValueError:
            logger.warning(
                "no housing detected - use args.housing={args.housing} argument instead"
            )

    match f_extension:
        case ".txt":
            mrun = MagnetRun.fromtxt(housing, site, args.input_file)
        case ".tdms":
            mrun = MagnetRun.fromtdms(housing, site, args.input_file)
        case _:
            raise RuntimeError(
                f"so far file with extension in {supported_formats} are implemented"
            )

    mdata = mrun.getMData()
    start_timestamp = mdata.getStartDate()
    dkeys = mrun.getKeys()

    inplace = False
    skeys = args.keys
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
        logger.info(f"smooth: {skeys}")
        for key in skeys:
            # TODO fix for tdms
            if mdata.Type == DataType.PUPITRE:
                selected_df = mrun.getMData().extractData(["t", key])
            else:
                (group, channel) = key.split("/")
                selected_df = addtime(mdata, group, channel)

            logger.debug(f"{selected_df.head()}")
            Meanval = selected_df[key].mean()
            logger.debug(f"{Meanval}")

            # Initializing noisy non linear data
            x = selected_df["t"].to_numpy()  # np.linspace(0,1,100)
            y = selected_df[key].to_numpy()  # np.sin(x * 1.5 * np.pi )

            logger.info("display Weighted Linear Regression")
            plt.figure(figsize=(10, 6))
            plt.scatter(x, y, facecolors="none", edgecolor="darkblue", label=key)

            #
            logger.info(f"compute Locally Weighted Linear Regression {args.method}")
            if args.method == "ag":
                try:
                    logger.info(f"f={smoothing_f}, iter={smoothing_iter}")
                    yest = lowess_ag(x, y, f=smoothing_f, iter=smoothing_iter)
                    plt.plot(x, yest, color="orange", label="Loess: A. Gramfort")
                except (ValueError, RuntimeError) as e:
                    logger.error(f"Failed to build lowess_ag: {e}")

            if args.method == "bell_kernel":
                try:
                    logger.info(f"tau={smoothing_tau}")
                    yest_bell = lowess_bell_shape_kern(x, y, smoothing_tau)
                    if args.debug:
                        x0 = (x[0] + x[40]) / 2.0
                        plt.fill(
                            x[:40],
                            Meanval * kernel_function(x[:40], x0, smoothing_tau),
                            color="lime",
                            alpha=0.5,
                            label="Bell shape kernel",
                        )
                    plt.plot(
                        x, yest_bell, color="red", label="Loess: bell shape kernel"
                    )
                except (ValueError, RuntimeError) as e:
                    logger.error(f"Failed to build bell: {e}")

            if args.method == "statsmodel_sm":
                try:
                    logger.info(f"f={smoothing_f}, iter={smoothing_iter}")
                    yest_sm = lowess_sm(x, y, f=smoothing_f, iter=smoothing_iter)
                    plt.plot(
                        x, yest_sm, color="magenta", label="Loess: statsmodel"
                    )  # marker="o",
                except (ValueError, RuntimeError) as e:
                    logger.error(f"Failed to build sm: {e}")

            plt.grid()
            plt.legend()
            plt.title(f"Loess regression comparisons {args.method}")
            if args.show:
                plt.show()
            else:
                imagefile = filename + "-" + "-".join(args.keys)
                start_date = ""
                start_time = ""
                if "Date" in dkeys and "Time" in dkeys:
                    tformat = "%Y.%m.%d %H:%M:%S"
                    start_date = mrun.getMData().getData("Date").iloc[0]
                    start_time = mrun.getMData().getData("Time").iloc[0]

                plt.savefig(
                    f"{imagefile}_{start_date}---{start_time}-smoothed-{key}.png",
                    dpi=300,
                )
            plt.close()

    if args.command == "lag":
        logger.warning(
            "lag: not implemented yet -- see analysis-refactor.py for proper use"
        )
        # for key in skeys:
        #     df = mrun.getData()
        #     for t in range(args.trange):
        #         lag_correlation(df, args.target, key, t)


def _run(args: "argparse.Namespace") -> int:
    """Dispatcher-compatible entry: receives already-parsed Namespace."""
    setup_logging(
        level=logging.DEBUG if getattr(args, "debug", False) else logging.WARNING,
        fmt=SIMPLE_FORMAT,
    )
    logger.debug(f"args: {args}")

    threshold = 0.5
    twindows = 10
    command = getattr(args, "proc_command", None) or getattr(args, "command", None)
    if command == "filter":
        threshold = args.threshold
        twindows = args.twindows

    smoothing_f = 0.7
    smoothing_tau = 400
    smoothing_iter = 3
    if command == "smooth":
        params = args.smooth_params.split(";")
        if args.method == "ag":
            smoothing_f = float(params[0])
            smoothing_iter = int(params[1])
        elif args.method in ("bell_kernel", "statsmodel_sm"):
            smoothing_tau = float(params[0])
        else:
            smoothing_tau = float(params[0])
            smoothing_f = float(params[1])
            smoothing_iter = int(params[2])

    supported_formats = [".txt", ".tdms"]
    f_extension = os.path.splitext(args.input_file)[-1]
    if f_extension not in supported_formats:
        logger.error("so far only txt file support is implemented")
        return 1

    site = "tutut"
    housing = getattr(args, "housing", None) or "notdefined"
    filename = os.path.basename(args.input_file)
    if filename.startswith("M"):
        try:
            idx = filename.index("_")
            housing = filename[:idx]
        except ValueError:
            logger.warning("no housing detected — use --housing")

    match f_extension:
        case ".txt":
            mrun = MagnetRun.fromtxt(housing, site, args.input_file)
        case ".tdms":
            mrun = MagnetRun.fromtdms(housing, site, args.input_file)
        case _:
            raise RuntimeError(f"unsupported extension '{f_extension}'")

    mdata = mrun.getMData()
    dkeys = mrun.getKeys()
    inplace = False
    skeys = getattr(args, "keys", []) or []

    if command == "filter":
        for key in skeys:
            filterpikes(mrun, key, inplace, threshold, twindows, args.debug, args.show, args.input_file)

    if command == "smooth":
        logger.info(f"smooth: {skeys}")
        for key in skeys:
            if mdata.Type == DataType.PUPITRE:
                selected_df = mrun.getMData().extractData(["t", key])
            else:
                (group, channel) = key.split("/")
                selected_df = addtime(mdata, group, channel)

            Meanval = selected_df[key].mean()
            x = selected_df["t"].to_numpy()
            y = selected_df[key].to_numpy()

            plt.figure(figsize=(10, 6))
            plt.scatter(x, y, facecolors="none", edgecolor="darkblue", label=key)

            if args.method == "ag":
                try:
                    yest = lowess_ag(x, y, f=smoothing_f, iter=smoothing_iter)
                    plt.plot(x, yest, color="orange", label="Loess: A. Gramfort")
                except (ValueError, RuntimeError) as e:
                    logger.error(f"Failed to build lowess_ag: {e}")
            if args.method == "bell_kernel":
                try:
                    yest_bell = lowess_bell_shape_kern(x, y, smoothing_tau)
                    plt.plot(x, yest_bell, color="red", label="Loess: bell shape kernel")
                except (ValueError, RuntimeError) as e:
                    logger.error(f"Failed to build bell: {e}")
            if args.method == "statsmodel_sm":
                try:
                    yest_sm = lowess_sm(x, y, f=smoothing_f, iter=smoothing_iter)
                    plt.plot(x, yest_sm, color="magenta", label="Loess: statsmodel")
                except (ValueError, RuntimeError) as e:
                    logger.error(f"Failed to build sm: {e}")

            plt.grid()
            plt.legend()
            plt.title(f"Loess regression comparisons {args.method}")
            if args.show:
                plt.show()
            else:
                imagefile = filename + "-" + "-".join(skeys)
                start_date = ""
                start_time = ""
                if "Date" in dkeys and "Time" in dkeys:
                    start_date = mrun.getMData().getData("Date").iloc[0]
                    start_time = mrun.getMData().getData("Time").iloc[0]
                plt.savefig(f"{imagefile}_{start_date}---{start_time}-smoothed-{key}.png", dpi=300)
            plt.close()

    if command == "lag":
        logger.warning("lag: not implemented yet")

    return 0


def register(sub: "argparse._SubParsersAction") -> None:
    """Register the ``processing`` subcommand on *sub*."""

    p = sub.add_parser("processing", help="filter/smooth/lag a single run file")
    p.add_argument("input_file", help="input file (.txt or .tdms)")
    p.add_argument("--show", action="store_true", help="display graphs")
    p.add_argument("--debug", action="store_true", help="activate debug mode")
    p.add_argument("--housing", help="housing (e.g. M9)", default=None)

    sub2 = p.add_subparsers(title="processing commands", dest="proc_command")

    pf = sub2.add_parser("filter", help="filter spikes")
    pf.add_argument("--threshold", type=float, default=0.5)
    pf.add_argument("--twindows", type=int, default=10)
    pf.add_argument("--keys", nargs="+", default="Tin1")

    ps = sub2.add_parser("smooth", help="smooth data")
    ps.add_argument("--keys", nargs="+", default="Tin1")
    ps.add_argument(
        "--method", choices=["ag", "bell_kernel", "statsmodel_sm", "all"], default="bell_kernel"
    )
    ps.add_argument("--smooth_params", nargs="?", default="400")

    pl = sub2.add_parser("lag", help="lag correlation (not yet implemented)")
    pl.add_argument("--keys", nargs="+", default="Tin1")
    pl.add_argument("--target", type=str, default="tsb")
    pl.add_argument("--trange", type=int, default=100)

    p.set_defaults(_handler=_run)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
