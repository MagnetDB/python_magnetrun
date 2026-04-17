"""Stats command: compute and display statistics for MagnetRun data."""

import logging
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from tabulate import tabulate

from ..magnetdata_base import DataType
from ..MagnetRun import MagnetRun
from ..processing.smoothers import savgol
from .plot import plot_bkpts

logger = logging.getLogger(__name__)


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
    from ..processing import stats

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
                if mdata.Type == DataType.PUPITRE:
                    logger.info(f"pupitre: stats for {key}")
                    (symbol, unit) = mdata.getUnitKey(key)

                    period = 1
                    num_points_threshold = int(args.dthreshold / period)
                    tkey = "t"
                    channel = key
                elif mdata.Type == DataType.TDMS:
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
                    from ..processing.plateaux import nplateaus

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
                            df_plateaux.loc[df_plateaux["duration"].idxmax()]
                            .to_numpy()
                            .tolist()
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
                        logger.info(
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
                        logger.warning(
                            f"{file.replace(extension, '')}: no peaks detected - duration={mdata.getDuration()}, {mdata.getData(key).describe()}"
                        )
                    # print(f"data: {len(data)}")
                    # print(f"data: {data[-1]}")

                if args.detect_bkpts:
                    ts = None
                    if mdata.Type == DataType.PUPITRE:
                        ts = mdata.getData([key])[key]
                        freq = 1
                        logger.info(f"{key}: freq={freq} Hz")
                    elif mdata.Type == DataType.TDMS:
                        ts = mdata.getData(f"{group}/{channel}")[channel]
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
                        quantiles[str(level)] = np.quantile(
                            abs(smoothed), level / 100.0
                        )

                    level = args.level
                    max_level_50 = (
                        abs(1 - abs(smoothed).max() / quantiles["50"]) * 100.0
                    )
                    max_level_75 = (
                        abs(1 - abs(smoothed).max() / quantiles["75"]) * 100.0
                    )
                    logger.debug(
                        f"max_level_50={max_level_50}, max_level_75={max_level_75}"
                    )
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
                    peaks, peaks_properties = find_peaks(
                        abs(smoothed_der2), height=quantiles_der
                    )

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
                    logger.info(
                        f"{channel}: peaks={peaks.shape[0]}, ignore_peaks={len(ignore_peaks)}"
                    )

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

                    if mdata.Type == DataType.TDMS:
                        # select key from GR1 or GR2
                        selected = [
                            t
                            for t in mdata.Keys
                            if t.startswith("Tensions_Aimant/Interne")
                        ]
                        logger.debug(f"selected: {selected}")
                        for key in selected:
                            (symbol, unit) = mdata.getUnitKey(key)

                            (group, channel) = key.split("/")
                            period = mdata.Groups[group][channel]["wf_increment"]
                            num_points_threshold = int(args.dthreshold / period)

                            ts = mdata.getData(f"{group}/{channel}")[channel]
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
                            quantiles_der = np.quantile(
                                abs(smoothed_der2), level / 100.0
                            )
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
                                            difference_array = np.absolute(
                                                peaks - cpeaks[i]
                                            )

                                            # find the index of minimum element from the array
                                            index = difference_array.argmin()
                                            msg = f"{i}: closest values in peaks={peaks[index]}, cpeaks[{i}]={cpeaks[i]}"

                                            if (
                                                abs(peaks[index] - cpeaks[i])
                                                >= args.window
                                            ):
                                                msg += " **"
                                                real_anomalies.append(cpeaks[i])
                                            logger.debug(f"{msg}")

                                logger.info(
                                    f"anomalies: {len(anomalies)} - likely {len(real_anomalies)}"
                                )
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
