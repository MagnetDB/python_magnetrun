"""Plot commands: visualise MagnetRun data."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.cbook import flatten

from ..hybrid import HybridRun
from ..MagnetRun import MagnetRun
from ..utils.downsampling import DownsampleConfig

logger = logging.getLogger(__name__)


def plot_bkpts(
    file: str,
    channel: str,
    symbol: str,
    unit: str,
    ts: pd.DataFrame,
    smoothed: np.ndarray,
    smoothed_der1: np.ndarray,
    smoothed_der2: np.ndarray,
    quantiles_der: float,
    peaks: np.ndarray,
    ignore_peaks: list[int],
    anomalies: list[int],
    level: int,
    window: int,
    save: bool = False,
):
    """_summary_

    :param file: _description_
    :type file: str
    :param channel: _description_
    :type channel: str
    :param symbol: _description_
    :type symbol: str
    :param unit: _description_
    :type unit: str
    :param ts: _description_
    :type ts: pd.tseries
    :param smoothed: _description_
    :type smoothed: np.ndarray
    :param smoothed_der1: _description_
    :type smoothed_der1: np.ndarray
    :param smoothed_der2: _description_
    :type smoothed_der2: np.ndarray
    :param quantiles_der: _description_
    :type quantiles_der: float
    :param peaks: _description_
    :type peaks: np.ndarray
    :param ignore_peaks: _description_
    :type ignore_peaks: list[int]
    :param anomalies: _description_
    :type anomalies: list[int]
    :param save: _description_, defaults to False
    :type save: bool, optional
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 1)

    ax0 = plt.subplot(gs[0])
    ax0.plot(ts.to_numpy(), label=channel, color="blue", marker="o", linestyle="None")
    ax0.plot(smoothed, label="smoothed", color="red")
    ax0.legend()
    ax0.grid()
    # ax0.set_xlabel('t [s]')
    ax0.set_ylabel(f"{symbol} [{unit:~P}]")
    ax0.set_title(f"{file}: {channel}")

    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.plot(smoothed_der2, label=channel, color="red")
    ax1.legend()
    ax1.grid()
    # ax1.set_xlabel('t [s]')
    ax1.set_title(f"Savgo filter [2nd order der]: ({level}%: {quantiles_der:.3e})")

    ax2 = plt.subplot(gs[2], sharex=ax0)
    std_ts = ts.rolling(window=window).std()
    ax2.plot(std_ts.to_numpy(), label="rolling std", color="blue")
    ax2.legend()
    ax2.grid()
    ax2.set_xlabel("t [s]")
    ax2.set_title("Rolling std")

    if peaks.shape[0]:
        ax0.plot(peaks, smoothed[peaks], "go", label="peaks")
        ax0.legend()

        ax1.plot(peaks, smoothed_der2[peaks], "go", label="peaks")
        ax1.legend()

    if ignore_peaks:
        ax0.plot(ignore_peaks, smoothed[ignore_peaks], "yo", label="ignore peaks")
        ax0.legend()

        ax1.plot(ignore_peaks, smoothed_der2[ignore_peaks], "yo", label="ignore peaks")
        ax1.legend()

    if anomalies:
        ax0.plot(anomalies, smoothed[anomalies], "ro", label="anomalies")
        ax0.legend()

        ax1.plot(anomalies, smoothed_der2[anomalies], "ro", label="anomalies")
        ax1.legend()

    if save:
        f_extension = os.path.splitext(file)[-1]
        plt.savefig(
            f"{file.replace(f_extension, '')}-{channel}-detect_bkpts.png", dpi=300
        )
    else:
        plt.show()
    plt.close()


def plot_vs_time(input_files, inputs, extensions, args):
    """Plot data versus time for selected keys.

    :param input_files: List of input file paths
    :type input_files: list
    :param inputs: Dictionary containing MagnetRun data for each file
    :type inputs: dict
    :param extensions: Dictionary mapping file extensions to indices
    :type extensions: dict
    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    """
    my_ax = plt.gca()

    items = args.vs_time
    logger.debug(f"items={items}")
    title = os.path.basename(input_files[0]) if input_files else ""
    if len(input_files) > 1:
        klabels = flatten(items)
        title = f"{'-'.join(klabels)}"

    legends = []
    t0 = []
    symbol, unit = None, None
    f_extension = ""
    file = ""
    key = ""
    for i, file in enumerate(input_files):
        f_extension = os.path.splitext(file)[-1]
        plot_args = items[list(extensions.keys()).index(f_extension)]
        logger.debug(
            f"field: {file}, plot_args: {plot_args}, f_extension:{f_extension}"
        )
        if args.log_level == "DEBUG":
            logger.debug(
                f"plot_args: {plot_args}, f_extension:{f_extension}, {extensions[f_extension]}"
            )
        mrun: MagnetRun = inputs[file]["data"]
        t0.append(mrun.StartTime)
        mdata = mrun.getMData()
        delta_t = 0.0
        if i >= 1:
            # align time axis
            delta_t = (mrun.StartTime - t0[0]).total_seconds()
            logger.info(f"align time axis: delta_t={delta_t} s")
            mdata.shiftTime(delta_t)

        for key in plot_args:
            try:
                (symbol, unit) = mdata.getUnitKey(key)
                logger.debug(f"plot {key} [{symbol} {unit:~P}]")

                mdata.plotData(
                    x="t", y=key, ax=my_ax, normalize=args.normalize, offset=delta_t
                )
                legends.append(
                    f"{os.path.basename(file).replace(f_extension, '')}: {key}"
                )
                if args.normalize:
                    legends[
                        -1
                    ] += f" max={float(mdata.getData([key]).max().iloc[0]):.3f} [{unit:~P}]"
                    logger.debug("normalize")
            except RuntimeError:
                logger.error(f"key: {key} not found in {file}")
                logger.info(f"available keys: {mdata.getKeys()}")
                continue

    # -- Hybrid data ----------------------------------------------------------
    vs_time_hybrid = getattr(args, "vs_time_hybrid", None)
    if vs_time_hybrid and "hybrid" in inputs:
        hrun: HybridRun = inputs["hybrid"]["data"]
        time_offset = (hrun.StartTime - t0[0]).total_seconds() if t0 else 0.0
        logger.info(f"hybrid time offset: {time_offset} s")

        for key in vs_time_hybrid:
            try:
                data, time = hrun.getData(
                    key, downsample=DownsampleConfig(n_out=args.hybrid_downsample)
                )
                data = np.asarray(data, dtype=float)
                time = np.asarray(time, dtype=float)

                data_max = float(data.max())
                if args.normalize:
                    data_min = float(data.min())
                    data = (data - data_min) / (data_max - data_min)

                my_ax.plot(time + time_offset, data, alpha=0.7, linewidth=0.5)
                label = f"{args.fepc_system}:{args.hybrid_date}: {key}"
                legends.append(label)

                if args.normalize:
                    unit_info = hrun.getMData().getUnitKey(key)
                    unit_str = f" [{unit_info[1]:~P}]" if len(unit_info) == 2 else ""
                    legends[-1] += f" max={data_max:.3f}{unit_str}"

                # Use hybrid unit for axis label if not already set by file data
                if symbol is None:
                    unit_info = hrun.getMData().getUnitKey(key)
                    if len(unit_info) == 2:
                        symbol, unit = unit_info

            except (KeyError, ValueError, RuntimeError) as e:
                logger.error(f"key: {key} not found in hybrid data: {e}")
                continue
    # -------------------------------------------------------------------------

    if symbol is not None and unit is not None:
        plt.ylabel(f"{symbol} [{unit:~P}]")
    if args.normalize:
        plt.ylabel("normalized")

    if len(legends) > 1:
        my_ax.legend(labels=legends)

    if t0:
        (t_symbol, t_unit) = inputs[input_files[0]]["data"].getMData().getUnitKey("t")
        plt.xlabel(f"{t_symbol} [{t_unit:~P}]")
    else:
        plt.xlabel("t [s]")

    plt.title(title)
    if not args.save:
        plt.show()
    else:
        imagefile = (
            f"{file.replace(f_extension, '')}-{key}" if file else f"hybrid-{key}"
        )
        logger.info(f"saveto: {imagefile}_vs_time.png")
        plt.savefig(f"{imagefile}_vs_time.png", dpi=300)
    plt.close()


def plot_key_vs_key(input_files, inputs, extensions, args):
    """Plot key versus key pairs.

    :param input_files: List of input file paths
    :type input_files: list
    :param inputs: Dictionary containing MagnetRun data for each file
    :type inputs: dict
    :param extensions: Dictionary mapping file extensions to indices
    :type extensions: dict
    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    """
    my_ax = plt.gca()

    items = args.key_vs_key
    file = ""
    f_extension = ""
    title = os.path.basename(input_files[0])
    if len(input_files) > 1:
        klabels = flatten(items)
        title = f"{'-'.join(klabels)}"

    legends = []
    # split pairs in key1, key2
    print(f"key_vs_key={args.key_vs_key}", flush=True)
    print(f"extensions={extensions}", flush=True)
    pairs = args.key_vs_key
    for file in input_files:
        f_extension = os.path.splitext(file)[-1]
        legends.append(os.path.basename(file).replace(f_extension, ""))
        plot_args = pairs[list(extensions.keys()).index(f_extension)]
        logger.debug(
            f"field: {file}, plot_args: {plot_args}, f_extension:{f_extension}"
        )
        mrun: MagnetRun = inputs[file]["data"]
        mdata = mrun.getMData()

        for pair in plot_args:
            # print(f"pair={pair}")
            # print("pair=", pair, " type=", type(pair))
            items = pair.split("-")
            if len(items) != 2:
                raise RuntimeError(f"invalid pair of keys:{pair}")
            key1 = items[0]
            key2 = items[1]
            print(f"plotting {key1} vs {key2} from {file}", flush=True)
            try:
                mdata.plotData(x=key1, y=key2, ax=my_ax)
            except RuntimeError as e:
                logger.error(f"pair {pair!r}: key not found in {file}: {e}")
                logger.info(f"available keys: {mdata.getKeys()}")
                continue

    if len(legends) > 1:
        plt.legend(labels=legends)
    plt.title(title)

    if not args.save:
        plt.show()
    else:
        imagefilename = f"{file.replace(f_extension, '')}-{'_'.join(items)}"
        logger.info(f"saveto: {imagefilename}.png")
        plt.savefig(f"{imagefilename}.png", dpi=300)
    plt.close()
