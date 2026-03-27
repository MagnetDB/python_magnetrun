"""Main module."""

import logging
import os
import sys
import traceback

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.cbook import flatten
from natsort import natsorted
from scipy.signal import find_peaks
from tabulate import tabulate

from .hybrid import HybridRun
from .MagnetRun import MagnetRun
from .processing.smoothers import savgol
from .utils.files import expand_input_files

# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available

# Setup logger
logger = logging.getLogger(__name__)


def format_exception_location() -> str:
    """Get a concise string with file:line:function where exception occurred"""
    exc_type, exc_value, exc_tb = sys.exc_info()
    if exc_tb is not None:
        tb = traceback.extract_tb(exc_tb)
        if tb:
            frame = tb[-1]
            from pathlib import Path

            filename = Path(frame.filename).name
            return f"{filename}:{frame.lineno}:{frame.name}"
    return "unknown:?:?"


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
        plt.savefig(f"{file.replace(f_extension, '')}-{channel}-detect_bkpts.png", dpi=300)
    else:
        plt.show()
    plt.close()


def output_keys(file, inputs, extensions, args):
    """Extract and save selected keys to CSV file with optional smoothing.

    :param file: Input file path
    :type file: str
    :param inputs: Dictionary containing MagnetRun data for each file
    :type inputs: dict
    :param extensions: Dictionary mapping file extensions to indices
    :type extensions: dict
    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    """
    f_extension = os.path.splitext(file)[-1]
    mrun: MagnetRun = inputs[file]["data"]
    mdata = mrun.getMData()
    selected_keys = args.output_key[extensions[f_extension][0]]
    # print(f"selected_keys[{file}]: {selected_keys}")
    if "t" not in selected_keys:
        selected_keys.insert(0, "t")
    logger.info(f"selected keys: {selected_keys}")

    file_name = file.replace(f_extension, "")
    for key in selected_keys:
        if key != "t":
            file_name += f"_{key.replace('/', '_')}"
    file_name = file_name + "_vs_Time.csv"

    selected_df = mdata.extractData(selected_keys)
    if selected_df is not None:
        if args.smoother is not None:
            from .processing.smoothers import (
                lowess_ag,
                lowess_bell_shape_kern,
                lowess_sm,
            )

            for key in selected_keys:
                if key != "t":
                    logger.debug(f"smooth {key}")

                    y = selected_df[key].to_numpy()
                    x = selected_df["t"].to_numpy()
                    y_smoothed = None
                    smoother = args.smoother
                    match smoother:
                        case "savgol":
                            y_smoothed = savgol(
                                y=y,
                                window=args.window,
                                polyorder=3,
                                deriv=0,
                            )
                        case "ag":
                            y_smoothed = lowess_ag(
                                x,
                                y,
                                f=args.smoothing_f,
                                iter=args.smoothing_iter,
                            )
                        case "bell_kernel":
                            y_smoothed = lowess_bell_shape_kern(x, y, args.smoothing_tau)
                        case "statsmodel_sm":
                            y_smoothed = lowess_sm(
                                x,
                                y,
                                f=args.smoothing_f,
                                iter=args.smoothing_iter,
                            )
                        case _:
                            logger.error(f"{key}: unknow smoother {smoother}")

                    if args.log_level == "DEBUG":
                        selected_df[f"{key}_smoothed"] = y_smoothed
                        ax = selected_df.plot("t", key)
                        selected_df.plot("t", f"{key}_smoothed", ax=ax)
                        plt.show()
                        plt.close()
                        selected_df = selected_df.drop(f"{key}_smoothed", axis=1)

                    selected_df[key] = y_smoothed

        selected_df.to_csv(file_name, sep="\t", index=False, header=True)


def extract_pairkeys(file, inputs, extensions, args):
    """Extract and save pair of keys to CSV files.

    :param file: Input file path
    :type file: str
    :param inputs: Dictionary containing MagnetRun data for each file
    :type inputs: dict
    :param extensions: Dictionary mapping file extensions to indices
    :type extensions: dict
    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    """
    f_extension = os.path.splitext(file)[-1]
    mrun: MagnetRun = inputs[file]["data"]
    mdata = mrun.getMData()
    select_args = args.extract_pairkeys[extensions[f_extension][0]]
    for item in select_args:
        pairs = item.split(";")
        for pair in pairs:
            items = pair.split("-")
            if len(items) != 2:
                raise RuntimeError(f"invalid pair of keys: {pair}")
            key1 = items[0]
            key2 = items[1]
            if mdata is not None:
                newdf = mdata.extractData([key1, key2])
                if newdf is not None:
                    # Remove line with I=0
                    newdf = newdf[newdf[key1] != 0]
                    newdf = newdf[newdf[key2] != 0]

                    file_name = f"{file.replace(f_extension, '')}-{str(pair)}.csv"
                    newdf.to_csv(file_name, sep="\t", index=False, header=False)


def convert_to_csv(file, inputs):
    """Convert MagnetRun data to CSV format.

    :param file: Input file path
    :type file: str
    :param inputs: Dictionary containing MagnetRun data for each file
    :type inputs: dict
    """
    mrun: MagnetRun = inputs[file]["data"]
    mdata = mrun.getMData()

    extension = os.path.splitext(file)[-1]
    file_name = file.replace(extension, ".csv")
    if mdata.Type == 0:
        mdata.to_csv(file_name, sep="\t", index=False, header=True)


def output_timerange(file, inputs, extensions, args):
    """Extract and save data for a specified time range to CSV file.

    :param file: Input file path
    :type file: str
    :param inputs: Dictionary containing MagnetRun data for each file
    :type inputs: dict
    :param extensions: Dictionary mapping file extensions to indices
    :type extensions: dict
    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    """
    f_extension = os.path.splitext(file)[-1]
    mrun: MagnetRun = inputs[file]["data"]
    mdata = mrun.getMData()
    select_args = args.output_timerange[extensions[f_extension][0]]
    for item in select_args:
        timerange = item.split(";")

        file_name = file.replace(f_extension, "")
        file_name = file_name + "_from" + str(timerange[0].replace(":", "-"))
        file_name = file_name + "_to" + str(timerange[1].replace(":", "-")) + ".csv"

        if mdata.Type == 0:
            selected_df = mdata.extractTimeData(timerange)
            if selected_df is not None:
                selected_df.to_csv(file_name, sep="\t", index=False, header=True)
        elif mdata.Type == 1:
            for group in mdata.Data:
                selected_df = mdata.extractTimeData(timerange, group)
                if selected_df is not None:
                    selected_df.to_csv(file_name, sep="\t", index=False, header=True)


def add_field(mrun, args):
    """Add computed or formula-based fields to MagnetRun data.

    :param mrun: MagnetRun instance
    :type mrun: MagnetRun
    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    """
    mdata = mrun.getMData()
    logger.debug(mdata.getKeys())

    if args.compute:
        from pint import UnitRegistry

        from python_magnetcooling.water_properties import get_rho

        ureg = UnitRegistry()
        nkey = "rho"
        nkey_unit = ("rho", ureg.kilogram / ureg.meter**3)
        nkey_params = ["HPH", "TinH"]
        nkey_method = get_rho

        mdata.computeData(nkey_method, nkey, nkey_params, nkey_unit)
        logger.debug(mdata.getKeys())
        print(mdata.getData("rho").describe())

        if args.plot:
            my_ax = plt.gca()
            mdata.plotData(x="t", y=nkey, ax=my_ax, normalize=args.normalize)
            for param in nkey_params:
                mdata.plotData(x="t", y=param, ax=my_ax, normalize=args.normalize)

            if not args.save:
                plt.show()
            else:
                imagefile = nkey
                logger.info(f"saveto: {imagefile}_vs_time.png")
                plt.savefig(f"{imagefile}_vs_time.png", dpi=300)
            plt.close()

    if args.formula:
        logger.debug(f"add {args.formula}, plot={args.plot}")

        nkey = args.formula.split(" = ")[0]
        nunit = ""

        # self.units[key] = ("U", ureg.volt)
        logger.debug(f"try to add nkey={nkey} (formula={args.formula[1:]})")
        mdata.addData(key=nkey, formula=args.formula, unit=nunit)
        logger.debug(mdata.getKeys())
        if args.plot:
            my_ax = plt.gca()
            mdata.plotData(x="t", y=nkey, ax=my_ax, normalize=args.normalize)

            logger.debug(f"args.vs_time: {args.vs_time}")
            if args.vs_time:
                for key in args.vs_time[0]:
                    logger.debug(key)
                    mdata.plotData(x="t", y=key, ax=my_ax, normalize=args.normalize)

            if not args.save:
                plt.show()
            else:
                imagefile = nkey
                logger.info(f"saveto: {imagefile}_vs_time.png")
                plt.savefig(f"{imagefile}_vs_time.png", dpi=300)
            plt.close()


def output_time(file, inputs, extensions, times):
    """Extract and save data at specific time points to CSV file.

    :param file: Input file path
    :type file: str
    :param inputs: Dictionary containing MagnetRun data for each file
    :type inputs: dict
    :param extensions: Dictionary mapping file extensions to indices
    :type extensions: dict
    :param times: List of time points to extract
    :type times: list
    """
    f_extension = os.path.splitext(file)[-1]
    mrun: MagnetRun = inputs[file]["data"]
    mdata = mrun.getMData()
    select_args = times[extensions[f_extension][0]]
    select_args_str = "_at"
    for item in select_args:
        select_args_str += f"-{item:.3f}s"

    if mdata.Type == 0:
        data = mdata.Data
        df = data[data["timestamp"].isin(times)]
        file_name = file.replace(f_extension, "")
        file_name = file_name + select_args_str + ".csv"
        df.to_csv()

    elif mdata.Type == 1:
        data = mdata.Data
        for group in data:
            df = data[data.index.isin(times)]
            file_name = file.replace(f_extension, f"-{group}")
            file_name = file_name + select_args_str + ".csv"
            df.to_csv()


def display_stats(file, inputs, args, multiindex, columns, data):
    """Display and calculate statistics for MagnetRun data.

    :param file: Input file path
    :type file: str
    :param inputs: Dictionary containing MagnetRun data for each file
    :type inputs: dict
    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    :param multiindex: Multi-index for DataFrame
    :type multiindex: list
    :param columns: Column names for DataFrame
    :type columns: list
    :param data: Data rows for DataFrame
    :type data: list
    :return: Updated columns and data
    :rtype: tuple
    """
    from .processing import stats

    extension = os.path.splitext(file)[-1]
    mrun: MagnetRun = inputs[file]["data"]
    mdata = mrun.getMData()

    multiindex[0].append(os.path.basename(file).replace(extension, ""))

    if not args.plateau and not args.detect_bkpts and not args.localmax:
        result = stats.stats(mdata, display=False)
        # print("headers: ", result[1])
        # print("data: ", result[0])

        if not multiindex[1]:
            multiindex[1] = [table[0] for table in result[0]]
            columns = result[1][1:]

        for table in result[0]:
            data.append(table[1:])

    try:
        # print(f"args.keys: {args.keys}")

        if args.keys:
            multiindex[1] = args.keys
            for key in args.keys:
                if mdata.Type == 0:
                    logger.info(f"pupitre: stats for {key}")
                    (symbol, unit) = mdata.getUnitKey(key)

                    period = 1
                    num_points_threshold = int(args.dthreshold / period)
                    tkey = "t"
                    channel = key
                elif mdata.Type == 1:
                    logger.info(f"pigbrother: stats for {key}")
                    (symbol, unit) = mdata.getUnitKey(key)

                    # compute num_points_threshold from dthresold
                    (group, channel) = key.split("/")
                    period = mdata.Groups[group][channel]["wf_increment"]
                    num_points_threshold = int(args.dthreshold / period)

                    tkey = f"{group}/t"

                logger.debug(f"num_points_threshold: {num_points_threshold}")

                if args.localmax:
                    # find local maximum
                    from scipy.signal import argrelextrema

                    # create a sample series
                    Field = mdata.getData([tkey, key])
                    # print(Field.keys())

                    # use shift() function
                    local_max_indices = argrelextrema(
                        Field[channel].values, np.greater, mode="clip"
                    )
                    # print the results
                    # print(f"local_max_indices: {local_max_indices}")
                    """
                    for local in local_max_indices[0]:
                        print(local, end=": ", flush=True)
                        local_max = s["Field"].iat[int(local)]
                        print(local_max)
                    """

                    my_ax = plt.gca()
                    mdata.plotData(x="t", y=key, ax=my_ax)

                    local_max = Field.iloc[local_max_indices[0]]
                    # print(local_max, "type=", type(local_max))
                    local_max.plot(x="t", y=channel, ax=my_ax, marker="*")
                    plt.grid()
                    plt.show()

                if args.plateau:
                    from .processing.plateaux import nplateaus

                    logger.info(f"display plateaus for {key}")
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
                    logger.debug(f"df_plateaux: {df_plateaux.shape}")
                    if nrows != 0:
                        data.append(
                            df_plateaux.loc[df_plateaux["duration"].idxmax()].to_numpy().tolist()
                        )
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
                        data.append(
                            [
                                stats.numpy_NaN,
                                stats.numpy_NaN,
                                stats.numpy_NaN,
                                stats.numpy_NaN,
                            ]
                        )
                        print(
                            f"{file.replace(extension, '')}: no peaks detected - duration={mdata.getDuration()}, {mdata.getData(key).describe()}"
                        )
                    # print(f"data: {len(data)}")
                    # print(f"data: {data[-1]}")

                if args.detect_bkpts:
                    ts = None
                    if mdata.Type == 0:
                        ts = mdata.Data[key]
                        freq = 1
                        logger.info(f"{key}: freq={freq} Hz")
                    elif mdata.Type == 1:
                        ts = mdata.Data[group][channel]
                        freq = 1 / mdata.Groups[group][channel]["wf_increment"]
                        logger.info(f"{group}/{channel}: freq={freq} Hz")

                    #
                    smoothed = savgol(
                        y=ts.to_numpy(),
                        window=args.window,
                        polyorder=3,
                        deriv=0,
                    )
                    logger.debug(f"{file}: stats for smoothed")
                    logger.debug(f"min: {abs(smoothed).min()}")
                    logger.debug(f"mean: {abs(smoothed).mean()}")
                    logger.debug(f"max: {abs(smoothed).max()}")
                    logger.debug(f"std: {abs(smoothed).std()}")
                    quantiles = {}
                    for level in range(5, 100, 5):
                        quantiles[str(level)] = np.quantile(abs(smoothed), level / 100.0)

                    level = args.level
                    max_level_50 = abs(1 - abs(smoothed).max() / quantiles["50"]) * 100.0
                    max_level_75 = abs(1 - abs(smoothed).max() / quantiles["75"]) * 100.0
                    logger.debug(f"max_level_50={max_level_50}, max_level_75={max_level_75}")
                    if max_level_75 >= 5000:
                        logger.debug(f"overwrite level: {level} -> 40")
                        level = 40
                    if max_level_75 >= 1000:
                        logger.debug(f"overwrite level: {level} -> 80")
                        level = 80
                    if max_level_75 <= 500:
                        logger.debug(f"overwrite level: {level} -> 95")
                        level = 95
                    if max_level_75 <= 60:
                        logger.debug(f"overwrite level: {level} -> 96")
                        level = 96
                    if max_level_75 <= 20:
                        logger.debug(f"overwrite level: {level} -> 97")
                        level = 97
                    if max_level_75 <= 10:
                        logger.debug(f"overwrite level: {level} -> 98")
                        level = 98
                    if max_level_75 <= 0.1:
                        logger.debug(f"overwrite level: {level} -> 99")
                        level = 99
                    if max_level_75 <= 0.02:
                        logger.debug(f"overwrite level: {level} -> 99.7")
                        level = 99.7
                    # print(f'{file}: {max_level}%', flush=True)

                    smoothed_der1 = savgol(
                        y=ts.to_numpy(),
                        window=args.window,
                        polyorder=3,
                        deriv=1,
                    )
                    smoothed_der2 = savgol(
                        y=ts.to_numpy(),
                        window=args.window,
                        polyorder=3,
                        deriv=2,
                    )
                    logger.debug(f"{file}: stats for smoother 2nd order derivate")
                    logger.debug(f"min: {abs(smoothed_der2).min()}")
                    logger.debug(f"mean: {abs(smoothed_der2).mean()}")
                    logger.debug(f"max: {abs(smoothed_der2).max()}")
                    logger.debug(f"std: {abs(smoothed_der2).std()}")
                    quantiles_der = np.quantile(abs(smoothed_der2), level / 100.0)

                    # find peak of der2
                    peaks, peaks_properties = find_peaks(abs(smoothed_der2), height=quantiles_der)

                    # get peaks where std is above a giventhresold
                    # filtered_std_df = std_ts.gt(10)
                    ignore_peaks = []
                    """
                    for peak in peaks:
                        # print(f'peak={peak}', flush=True)
                        num = peak-1  # args.window
                        before = smoothed_der1[num]
                        num = peak+1 #args.window
                        after = smoothed_der1[num]
                        diff = abs(before-after)
                        # print(f'{peak}: before={before} after={after} diff={diff}', end="")
                        if diff <= 1:
                            # print(' **')
                            ignore_peaks.append(peak)
                        print(flush=True)
                    """
                    print(f"{channel}: peaks={peaks.shape[0]}, ignore_peaks={len(ignore_peaks)}")

                    plot_bkpts(
                        file,
                        channel,
                        symbol,
                        unit,
                        ts,
                        smoothed,
                        smoothed_der1,
                        smoothed_der2,
                        quantiles_der,
                        peaks,
                        ignore_peaks,
                        [],
                        level,
                        args.window,
                        args.save,
                    )

                    if mdata.Type == 1:
                        # select key from GR1 or GR2
                        selected = [
                            t for t in mdata.Keys if t.startswith("Tensions_Aimant/Interne")
                        ]
                        logger.debug(f"selected: {selected}")
                        for key in selected:
                            (symbol, unit) = mdata.getUnitKey(key)

                            (group, channel) = key.split("/")
                            period = mdata.Groups[group][channel]["wf_increment"]
                            num_points_threshold = int(args.dthreshold / period)

                            ts = mdata.Data[group][channel]
                            smoothed = savgol(
                                y=ts.to_numpy(),
                                window=args.window,
                                polyorder=3,
                                deriv=0,
                            )
                            smoothed_der1 = savgol(
                                y=ts.to_numpy(),
                                window=args.window,
                                polyorder=3,
                                deriv=1,
                            )
                            smoothed_der2 = savgol(
                                y=ts.to_numpy(),
                                window=args.window,
                                polyorder=3,
                                deriv=2,
                            )
                            quantiles_der = np.quantile(abs(smoothed_der2), level / 100.0)
                            cpeaks, cpeaks_properties = find_peaks(
                                abs(smoothed_der2), height=quantiles_der
                            )

                            if cpeaks.shape[0] != peaks.shape[0]:
                                # print(f'{channel}: peaks={cpeaks.shape[0]}')
                                # print(f'peaks: {peaks}')
                                # print(f'cpeaks: {cpeaks}')
                                isin = np.isin(cpeaks, peaks)
                                anomalies = []
                                real_anomalies = []
                                for i, item in enumerate(isin):
                                    if not item:
                                        first = np.isin([cpeaks[i] - 1], peaks)
                                        last = np.isin([cpeaks[i] + 1], peaks)
                                        # print(cpeaks[i], first, last, item)
                                        if not first[0] and not last[0]:
                                            anomalies.append(cpeaks[i])

                                            # calculate the difference array
                                            difference_array = np.absolute(peaks - cpeaks[i])

                                            # find the index of minimum element from the array
                                            index = difference_array.argmin()
                                            msg = f"{i}: closest values in peaks={peaks[index]}, cpeaks[{i}]={cpeaks[i]}"

                                            if abs(peaks[index] - cpeaks[i]) >= args.window:
                                                msg += " **"
                                                real_anomalies.append(cpeaks[i])
                                            logger.debug(f"{msg}")

                                print(f"anomalies: {len(anomalies)} - likely {len(real_anomalies)}")
                                if real_anomalies:
                                    # if args.verbose:
                                    #     print(f"anomalies: {anomalies}")
                                    plot_bkpts(
                                        file,
                                        channel,
                                        symbol,
                                        unit,
                                        ts,
                                        smoothed,
                                        smoothed_der1,
                                        smoothed_der2,
                                        quantiles_der,
                                        cpeaks,
                                        [],
                                        real_anomalies,
                                        level,
                                        args.window,
                                        args.save,
                                    )

    except (OSError, ValueError, RuntimeError, KeyError):
        logger.error(traceback.format_exc())
        pass

    return columns, data


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
        logger.debug(f"field: {file}, plot_args: {plot_args}, f_extension:{f_extension}")
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

                mdata.plotData(x="t", y=key, ax=my_ax, normalize=args.normalize, offset=delta_t)
                legends.append(f"{os.path.basename(file).replace(f_extension, '')}: {key}")
                if args.normalize:
                    legends[-1] += (
                        f" max={float(mdata.getData([key]).max().iloc[0]):.3f} [{unit:~P}]"
                    )
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
                data, time = hrun.getData(key, downsample=args.hybrid_downsample)
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
        imagefile = f"{file.replace(f_extension, '')}-{key}" if file else f"hybrid-{key}"
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
    # print(f"key_vs_key={args.key_vs_key}")
    pairs = args.key_vs_key
    for file in input_files:
        f_extension = os.path.splitext(file)[-1]
        legends.append(os.path.basename(file).replace(f_extension, ""))
        plot_args = pairs[list(extensions.keys()).index(f_extension)]
        logger.debug(f"field: {file}, plot_args: {plot_args}, f_extension:{f_extension}")
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
            mdata.plotData(x=key1, y=key2, ax=my_ax)

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


def main():
    from .cli_args import create_main_parser, get_datadir_mapping

    parser = create_main_parser()
    args = parser.parse_args()

    # Configure logging level
    log_level = getattr(logging, args.log_level.upper(), logging.WARNING)
    logging_config = {
        "level": log_level,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }
    if args.log_file:
        logging_config["filename"] = args.log_file
        logging_config["filemode"] = "a"
    logging.basicConfig(**logging_config)
    logger.setLevel(log_level)

    logger.debug(f"args: {args}")

    # load df pandas from input_file
    # check extension
    supported_formats = [".txt", ".tdms", ".csv"]

    # Create datadir dictionary mapping extensions to data directories
    datadir = get_datadir_mapping(args)

    # Expand glob patterns in input_file arguments
    expanded_files = expand_input_files(args.input_file, datadir)
    input_files = natsorted(expanded_files)
    logger.debug(f"input_files: {input_files}")

    inputs = {}
    extensions = {}
    for i, file in enumerate(input_files):
        f_extension = os.path.splitext(file)[-1]
        if f_extension not in extensions:
            extensions[f_extension] = [i]
        else:
            extensions[f_extension].append(i)
    # print(f"extensions: {extensions}, entries: {len(extensions.keys())}")
    # print(f"args.vs_time: {args.vs_time}, entries: {len(args.vs_time)}")

    for file in input_files:
        f_extension = os.path.splitext(file)[-1]
        if f_extension not in supported_formats:
            raise RuntimeError(f"so far file with extension in {supported_formats} are implemented")

        filename = os.path.basename(file)
        insert = args.insert if args.insert else "notdefined"
        housing = args.housing if args.housing else "notdefined"
        site = "tttt"
        if args.site:
            site = args.site
        else:
            result = filename.startswith("M")
            if result:
                try:
                    index = filename.index("_")
                    site = filename[0:index]
                    # print(f"site detected: {site}")
                except ValueError:
                    logger.warning(f"{file}: no site detected - use args.site argument instead")
                    continue
                if args.log_level == "DEBUG":
                    logger.debug(f"site={site}")

        try:
            match f_extension:
                case ".txt":
                    mrun = MagnetRun.fromtxt(site, insert, file)
                case ".tdms":
                    mrun = MagnetRun.fromtdms(site, insert, file)
                case ".csv":
                    mrun = MagnetRun.fromcsv(site, insert, file)
                case _:
                    raise RuntimeError(
                        f"so far file with extension in {supported_formats} are implemented"
                    )
        except (OSError, ValueError, RuntimeError) as error:
            # Print detailed error information with traceback
            tb_str = "".join(traceback.format_exception(*sys.exc_info()))
            print(f"{file}: an error occurred when loading at {format_exception_location()}")
            logger.error(f"Error: {error}")
            logger.error(f"Traceback:\n{tb_str}")
            continue

        mrun.setHousing(housing)
        inputs[file] = {"data": mrun}

        if args.command == "add":
            add_field(mrun, args)

        if args.command == "info":
            mdata = mrun.getMData()
            if args.list:
                logger.info(f"{file}: valid keys")
                mdata.info()

            if args.convert:
                data = mdata.Data
                if mdata.Type == 0:
                    csvfile = file.replace(f_extension, ".csv")
                    data.to_csv(csvfile, sep="\t", index=True, header=True)
                elif mdata.Type == 1:
                    for key, df in data.items():
                        logger.debug(f"convert: key={key}")
                        csvfile = file.replace(f_extension, f"-{key}.csv")
                        df.to_csv(csvfile, sep="\t", index=True, header=True)

    # Load hybrid data if requested
    hybrid_datadir = getattr(args, "hybrid_datadir", None)
    hybrid_date = getattr(args, "hybrid_date", None)
    if hybrid_datadir and hybrid_date:
        try:
            hrun = HybridRun.fromdir(
                base_dir=hybrid_datadir,
                date_str=hybrid_date,
                fepc_system=args.fepc_system,
                site=args.site or "",
            )
            inputs["hybrid"] = {"data": hrun}
            logger.info(
                f"loaded hybrid data: {hybrid_datadir}, date={hybrid_date}, "
                f"fepc_system={args.fepc_system}"
            )
        except (OSError, ValueError, RuntimeError) as error:
            tb_str = "".join(traceback.format_exception(*sys.exc_info()))
            print(
                f"hybrid data ({hybrid_datadir}, {hybrid_date}): "
                f"an error occurred at {format_exception_location()}"
            )
            logger.error(f"Error: {error}")
            logger.error(f"Traceback:\n{tb_str}")

    # perform operations defined by options
    if args.command == "plot":
        logger.info("subcommands: plot")
        if args.vs_time:
            assert len(args.vs_time) == len(
                extensions.keys()
            ), f"expected {len(extensions.keys())} vs_time arguments - got {len(args.vs_time)} "

            plot_vs_time(input_files, inputs, extensions, args)

        if args.key_vs_key:
            assert len(args.key_vs_key) == len(
                extensions
            ), f"expected {len(extensions)} key_vs_key arguments - got {len(args.key_vs_key)} "

            plot_key_vs_key(input_files, inputs, extensions, args)

    if args.command == "select":
        # plot_args = items[extensions[f_extension][0]]
        if args.output_time:
            assert (
                len(args.output_time) == len(extensions.keys())
            ), f"expected {len(extensions.keys())} output_time arguments - got {len(args.output_time)} "

            times = args.output_time.split(";")
            logger.info(f"Select data at {times}")
            for file in inputs:
                output_time(file, inputs, extensions, times)

        if args.output_timerange:
            assert (
                len(args.output_timerange) == len(extensions.keys())
            ), f"expected {len(extensions.keys())} output_timerange arguments - got {len(args.output_timerange)} "

            timerange = args.output_timerange.split(";")
            for file in inputs:
                output_timerange(file, inputs, extensions, args)

        if args.output_key:
            assert (
                len(args.output_key) == len(extensions.keys())
            ), f"expected {len(extensions.keys())} output_key arguments - got {len(args.output_key)} "

            for file in inputs:
                output_keys(file, inputs, extensions, args)

        if args.extract_pairkeys:
            assert (
                len(args.extract_pairkeys) == len(extensions.keys())
            ), f"expected {len(extensions.keys())} extract_pairkeys arguments - got {len(args.extract_pairkeys)} "
            for file in inputs:
                extract_pairkeys(file, inputs, extensions, args)

        if args.convert:
            for file in inputs:
                convert_to_csv(file, inputs)

    if args.command == "stats":
        if args.plateau:
            pass

        logger.info("Stats:")

        # to display stats
        multiindex = [[], []]
        columns = []
        data = []
        df_data = []
        output = "stats"
        if args.plateau:
            if not args.keys:
                args.keys = ["Field"]
            output += "-plateau"
        if args.localmax:
            if not args.keys:
                args.keys = ["Field"]
            output += "-localmax"
        if args.detect_bkpts:
            if not args.keys:
                args.keys = ["Field"]
            output += "-bkpts"

        for file in inputs:
            columns, data = display_stats(file, inputs, args, multiindex, columns, data)

        print(
            "concat tabs:",
            f"multi_index={len(multiindex[0]), len(multiindex[1])}",
            f"columns={len(columns)}",
            f"data={len(data)}",
        )
        logger.debug(f"multiindex: {multiindex}")
        logger.debug(f"columns: {columns}")
        logger.debug(f"data: {data}")

        df = pd.DataFrame(
            data,
            pd.MultiIndex.from_product(multiindex),
            columns=columns,
        )
        print(df.to_markdown(tablefmt="simple"))

        # save df to csv
        print("head:", df.head())
        df.to_csv(f"{output}.csv")


if __name__ == "__main__":
    main()
