"""Main module."""

import math
import datetime
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available

from ..magnetdata import MagnetData
from ..utils.sequence import list_sequence, list_duplicates_of


def stats(Data: MagnetData, fields: list[str] | None = None, debug: bool = False):
    """compute stats from the actual run"""

    # TODO:
    # add teb,... to list
    # add duration
    # add duration per Field above certain values
    # add \int Power over time

    from tabulate import tabulate

    # see https://github.com/astanin/python-tabulate for tablefmt
    if isinstance(Data.Data, pd.DataFrame):
        print("Statistics:\n")
        # print(f"data keys: {Data.getKeys()}", flush=True)
        tables = []
        headers = ["Name", "Mean", "Max", "Min", "Std", "Median", "Mode"]
        selected_fields = ["Field", "Pmagnet", "Ptot", "teb", "tsb", "debitbrut"]
        if fields is not None:
            selected_fields = fields

        for f in fields:
            table = [
                f"{f}[N/A]",
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
            ]
            if f in Data.getKeys():
                fname, unit = Data.getUnitKey(f)
                df = Data.Data[f]
                if debug:
                    print(f"get stats for {f} ({Data.getKeys()})", flush=True)
                    print(f"{f}: {df.head()}", flush=True)
                v_min = float(df.min())
                v_max = float(df.max())
                v_mean = float(df.mean())
                v_var = float(df.var())
                v_median = float(df.median())
                v_mode = np.NaN  # Most frequent value in a data set
                try:
                    v_mode = float(df.mode())
                except:
                    # print(f"{f}: failed to compute df.mode()")
                    pass
                table = [
                    f"{f}[{unit:~P}]",
                    v_mean,
                    v_max,
                    v_min,
                    math.sqrt(v_var),
                    v_median,
                    v_mode,
                ]

            tables.append(table)

        print(tabulate(tables, headers, tablefmt="simple"), "\n")
    else:
        raise RuntimeError("stats: not supported for TdmsFile data")
    return 0


def nplateaus(
    Data: MagnetData,
    xField: tuple,
    yField: tuple,
    threshold: float = 2.0e-2,
    num_points_threshold: int = 600,
    show: bool = False,
    save: bool = False,
    verbose: bool = False,
) -> list:
    """
    detect plateau vs index aka time
    """

    df = Data.getData()
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError(
            f"nplateaux: {Data.FileName}, unexpectype type of data: {type(df)}"
        )

    # filter and group plateaus
    max_difference = threshold
    min_number_points = num_points_threshold
    # group by maximum difference
    group_ids = (abs(df[yField[0]].diff(1)) > max_difference).cumsum()
    plateau_idx = 0

    plt.plot(
        df[xField[0]],
        df[yField[0]],
        label="measured",
        marker="x",
        lw=0.5,
        ms=2.0,
        color="black",
    )

    plateau_data = []
    for group_idx, group_data in df.groupby(group_ids):
        # filter non-plateaus by min number of points
        if len(group_data) < min_number_points:
            continue

        _start = min(group_data[xField[0]].iloc[0], group_data[xField[0]].iloc[-1])
        _end = max(group_data[xField[0]].iloc[0], group_data[xField[0]].iloc[-1])
        _time = group_data[xField[0]].mean()
        _value = group_data[yField[0]].mean()
        pdata = {"start": _start, "end": _end, "value": _value}
        if abs(_start - _end) <= 0.1:
            print(f"ignore plateau: {pdata}")
            continue

        plateau_idx += 1

        plt.plot(
            group_data[xField[0]],
            group_data[yField[0]],
            label=f"Plateau-{plateau_idx} {yField[0]}={_value:.2e} {yField[1]}",
            marker="x",
            lw=1.5,
            ms=5.0,
        )

        plateau_data.append(pdata)
        if verbose:
            print(
                f"plateau[{plateau_idx}]: {plateau_data[-1]}, duration={abs(_start - _end)} {xField[1]}",
                flush=True,
            )
        plt.annotate(
            f"{_value:.2e}",
            (_time, _value * (1 + 0.01)),
            ha="center",
        )
    if verbose:
        print(f"detected plateaux: {plateau_idx}", flush=True)

    plt.legend()
    plt.grid(b=True)

    lname = Data.FileName.replace("_", "-")
    lname = lname.replace(".txt", "")
    lname = lname.split("/")
    plt.title(lname[-1])
    plt.ylabel(f"{yField[0]} [{yField[1]}]")
    plt.xlabel(f"{xField[0]} [{xField[1]}]")
    plt.grid(True)

    if show:
        plt.show()
    if save:
        plt.savefig(f"{yField[0]}-{xField[0]}.png")
    plt.close()

    return plateau_data


def plateaus(
    Data: MagnetData,
    twindows=6,
    threshold=1.0e-4,
    b_threshold=1.0e-3,
    duration=5,
    show=False,
    save=True,
    debug=False,
):
    """get plateaus, pics from the actual run"""
    # print(f'plateaus: show={show}, save={save}, debug={debug}')

    if not isinstance(Data.Data, pd.DataFrame):
        raise RuntimeError("plateaus not available for Tdms data")

    df = Data.Data
    if show or save:
        ax = plt.gca()

    # TODO:
    # pass b_thresold as input param
    # b_threshold = 1.e-3

    if debug:
        print("Search for plateaux:", "Type:", Data.Type)

    B_min = float(df["Field"].min())
    B_max = float(df["Field"].max())
    B_mean = float(df["Field"].mean())
    B_var = float(df["Field"].var())

    Bz = Data.Data["Field"]
    regime = Bz.to_numpy()
    df_ = pd.DataFrame(regime)
    df_["regime"] = pd.Series(regime)

    diff = np.diff(regime)  # scale by B_max??
    df_["diff"] = pd.Series(diff)

    ndiff = np.where(abs(diff) >= threshold, diff, 0)
    df_["ndiff"] = pd.Series(ndiff)
    if debug:
        print("gradient: ", df_)

    # TODO:
    # check gradient:
    #     if 0 in between two 1 (or -1), 0 may be replaced by 1 or -1 depending on ndiff values
    #     same for small sequense of 0 (less than 2s)
    gradient = np.sign(df_["ndiff"].to_numpy())
    # TODO check runtime error from time to time ??
    gradkey = "gradient-Field"
    df_[gradkey] = pd.Series(gradient)

    # # Try to remove spikes
    # ref: https://ocefpaf.github.io/python4oceanographers/blog/2015/03/16/outlier_detection/

    df_["pandas"] = df_[gradkey].rolling(window=twindows, center=True).median()

    difference = np.abs(df_[gradkey] - df_["pandas"])
    outlier_idx = difference > threshold
    # print("median[%d]:" % df_[gradkey][outlier_idx].size, df_[gradkey][outlier_idx])

    if show or save:
        kw = dict(
            marker="o", linestyle="none", color="g", label=str(threshold), legend=True
        )
        df_[gradkey][outlier_idx].plot(**kw)

    # not needed if center=True
    # df_['shifted\_pandas'] =  df_['pandas'].shift(periods=-twindows//2)
    df_.rename(columns={0: "Field"}, inplace=True)

    del df_["ndiff"]
    del df_["diff"]
    del df_["regime"]
    # del df_['pandas']

    if show or save:
        ax = plt.gca()
        df_.plot(ax=ax, grid=True)

        if show:
            plt.show()

        if save:
            # imagefile = self.Site + "_" + self.Insert
            imagefile = "plateaux"
            start_date = ""
            start_time = ""
            if "timestamp" in Data.getKeys():
                start_timestamp = Data.Data["timestamp"].iloc[0]
                start_date = start_timestamp.strftime("%Y.%m.%d")
                start_time = start_timestamp.strftime("%H:%M:%S")

            plt.savefig(
                f"{imagefile}_{str(start_date)}---{str(start_time)}.png",
                dpi=300,
            )
            plt.close()

    # convert panda column to a list
    # print("df_:", df_.columns.values.tolist())
    B_list = df_["pandas"].values.tolist()

    from functools import partial

    regimes_in_source = partial(list_duplicates_of, B_list)
    if debug:
        for c in [1, 0, -1]:
            print(c, regimes_in_source(c))

    # # To get timedelta in mm or millseconds
    # time_d_min = time_d / datetime.timedelta(minutes=1)
    # time_d_ms  = time_d / datetime.timedelta(milliseconds=1)
    plateaux = regimes_in_source(0)
    print(f"Field plateaus(thresold={threshold}: {len(plateaux)})")
    actual_plateaux = []
    for p in plateaux:
        t0 = Data.Data["timestamp"].iloc[p[0]]
        t1 = Data.Data["timestamp"].iloc[p[1]]
        dt = t1 - t0

        # b0=Data.getData('Field').values.tolist()[p[0]]
        b0 = float(Data.Data["Field"].iloc[p[0]])
        b1 = float(Data.Data["Field"].iloc[p[1]])

        tformat = "%Y.%m.%d %H:%M:%S"
        start_time = t0.strftime(tformat)
        end_time = t1.strftime(tformat)

        if debug:
            msg = f"\t{start_time}\t{end_time}"
            print(f"{msg}\t{dt.total_seconds():8.6g}\t{b0:8.4g}\t{b1:8.4g}")

        # if (b1-b0)/b1 > b_thresold: reject plateau
        # if abs(b1) < b_thresold and abs(b0) < b_thresold: reject plateau
        if (dt / datetime.timedelta(seconds=1)) >= duration:
            if abs(b1) >= b_threshold and abs(b0) >= b_threshold:
                actual_plateaux.append(
                    [start_time, end_time, dt.total_seconds(), b0, b1]
                )

    print(
        "%s plateaus(threshold=%g, b_threshold=%g, duration>=%g s): %d over %d"
        % (
            "Field",
            threshold,
            b_threshold,
            duration,
            len(actual_plateaux),
            len(plateaux),
        )
    )
    tables = []
    for p in actual_plateaux:
        b_diff = abs(1.0 - p[3] / p[4])
        tables.append([p[0], p[1], p[2], p[3], p[4], b_diff * 100.0])

    pics = list_sequence(B_list, [1.0, -1.0])
    print(" \nField pics (aka sequence[1,-1]): {len(pics)}")
    pics = list_sequence(B_list, [1.0, 0, -1.0, 0, 1.0])
    print(" \nField pics (aka sequence[1,0,-1,0,1]): {len(pics)}")

    # remove adjacent duplicate
    import itertools

    B_ = [x[0] for x in itertools.groupby(B_list)]
    if debug:
        print("B_=", B_, B_.count(0))
    print(
        "Field commisionning ? (aka sequence [1.0,0,-1.0,0.0,-1.0]): %d"
        % len(list_sequence(B_, [1.0, 0, -1.0, 0.0, -1.0]))
    )
    print("\n\n")

    from tabulate import tabulate

    headers = ["start", "end", "duration", "B0[T]", "B1[T]", "\u0394B/B[%]"]
    print(tabulate(tables, headers, tablefmt="simple"), "\n")

    return 0


def pearson(
    Data: MagnetData,
    fields: list[str],
    save: bool = False,
    show: bool = False,
    debug: bool = False,
):
    """
    compute Pearson correlation for fields

    ref: https://www.kaggle.com/code/adepvenugopal/time-series-correlation-pearson-tlcc-dtw
    """

    nFields = len(fields)
    if isinstance(Data, pd.DataFrame):
        for i in range(nFields):
            for j in range(i + 1, nFields):
                df = Data.getData(["t", fields[i], fields[j]])
                overall_pearson_r = df.corr().iloc[0, 1]
                print(f"Pandas computed Pearson r: {overall_pearson_r}")

                r, p = stats.pearsonr(df.dropna()[fields[i]], df.dropna()[fields[j]])
                print(f"Scipy computed Pearson r: {r} and p-value: {p}")

                # Compute rolling window synchrony
                f, ax = plt.subplots(figsize=(7, 3))
                df.rolling(window=30, center=True).median().plot(ax=ax)
                ax.set(xlabel="Time", ylabel="Pearson r")
                ax.set(title=f"Overall Pearson r = {np.round(overall_pearson_r,2)}")

                if show:
                    plt.show()
                if save:
                    outputfile = f"{fields[i]}-{fields[j]}-pearson.png"
                    plt.savefig(f"{outputfile}.png", dpi=300)
                plt.close()

    else:
        raise RuntimeError(f"stats/pearson: {Data.FileName} not a panda dataframe")


def crosscorr(datax, datay, lag=0, wrap=False):
    """Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))


def tlcc(
    Data: MagnetData,
    xfield: str,
    yfield: str,
    save: bool = False,
    show: bool = False,
    debug: bool = False,
):

    d1 = Data.getData(xfield)
    d2 = Data.getData(yfield)
    seconds = 5
    fps = 30
    rs = [
        crosscorr(d1, d2, lag)
        for lag in range(-int(seconds * fps), int(seconds * fps + 1))
    ]
    offset = np.floor(len(rs) / 2) - np.argmax(rs)
    f, ax = plt.subplots(figsize=(14, 3))
    ax.plot(rs)
    ax.axvline(np.ceil(len(rs) / 2), color="k", linestyle="--", label="Center")
    ax.axvline(np.argmax(rs), color="r", linestyle="--", label="Peak synchrony")
    # ylim=[0.1, 0.31],
    # xlim=[0, 301],
    ax.set(
        title=f"Offset = {offset} frames\n{xfield} leads <> {yfield} leads",
        xlabel="Offset",
        ylabel="Pearson r",
    )
    # ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
    # ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150])
    plt.legend()

    if show:
        plt.show()
    if save:
        pfile = f"{xfield}-{yfield}-TLCC"
        plt.savefig(f"{pfile}.png", dpi=300)
    plt.close()


def wtlcc(
    Data: MagnetData,
    xfield: str,
    yfield: str,
    save: bool = False,
    show: bool = False,
    debug: bool = False,
):
    import seaborn as sns

    df = Data.getData([xfield, yfield])
    # Windowed time lagged cross correlation
    seconds = 5
    fps = 30
    no_splits = 20
    samples_per_split = df.shape[0] / no_splits
    rss = []
    for t in range(0, no_splits):
        d1 = df[xfield].loc[(t) * samples_per_split : (t + 1) * samples_per_split]
        d2 = df[yfield].loc[(t) * samples_per_split : (t + 1) * samples_per_split]
        rs = [
            crosscorr(d1, d2, lag)
            for lag in range(-int(seconds * fps), int(seconds * fps + 1))
        ]
        rss.append(rs)
    rss = pd.DataFrame(rss)
    f, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(rss, cmap="RdBu_r", ax=ax)
    ax.set(
        title=f"Windowed Time Lagged Cross Correlation",
        xlim=[0, 301],
        xlabel="Offset",
        ylabel="Window epochs",
    )
    ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
    ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150])

    if show:
        plt.show()
    if save:
        pfile = f"{xfield}-{yfield}-WTLCC"
        plt.savefig(f"{pfile}.png", dpi=300)
    plt.close()


def rwtlcc(
    Data: MagnetData,
    xfield: str,
    yfield: str,
    save: bool = False,
    show: bool = False,
    debug: bool = False,
):
    import seaborn as sns

    df = Data.getData([xfield, yfield])
    # Rolling window time lagged cross correlation
    seconds = 5
    fps = 30
    window_size = 300  # samples
    t_start = 0
    t_end = t_start + window_size
    step_size = 30
    rss = []
    while t_end < 5400:
        d1 = df[xfield].iloc[t_start:t_end]
        d2 = df[yfield].iloc[t_start:t_end]
        rs = [
            crosscorr(d1, d2, lag, wrap=False)
            for lag in range(-int(seconds * fps), int(seconds * fps + 1))
        ]
        rss.append(rs)
        t_start = t_start + step_size
        t_end = t_end + step_size
    rss = pd.DataFrame(rss)

    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(rss, cmap="RdBu_r", ax=ax)
    ax.set(
        title=f"Rolling Windowed Time Lagged Cross Correlation",
        xlim=[0, 301],
        xlabel="Offset",
        ylabel="Epochs",
    )
    ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
    ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150])

    if show:
        plt.show()
    if save:
        pfile = f"{xfield}-{yfield}-RWTLCC"
        plt.savefig(f"{pfile}.png", dpi=300)
    plt.close()
