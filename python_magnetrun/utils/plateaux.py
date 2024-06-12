#! /usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ..magnetdata import MagnetData
from .sequence import list_duplicates_of, list_sequence

from datetime import timedelta


def tuple_type(strings: str) -> tuple:
    strings = strings.replace("(", "").replace(")", "")
    mapped_str = map(str, strings.split(","))
    return tuple(mapped_str)


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

    print(f"nplateaus: xField={xField}, yField={yField}", flush=True)

    df = pd.DataFrame()
    if not isinstance(Data.Data, pd.DataFrame):
        if xField[0] == "t":
            (group, channel) = yField[0].split("/")
            df = Data.getData([f"{group}/t", yField[0]])
            ykey = channel
    else:
        df = Data.getData()
        ykey = yField[0]

    # filter and group plateaus
    max_difference = threshold
    min_number_points = num_points_threshold
    # group by maximum difference
    group_ids = (abs(df[ykey].diff(1)) > max_difference).cumsum()
    plateau_idx = 0

    plt.plot(
        df[xField[0]],
        df[ykey],
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
        _value = group_data[ykey].mean()
        pdata = {"start": _start, "end": _end, "value": _value}
        if abs(_start - _end) <= 0.1:
            print(f"ignore plateau: {pdata}")
            continue

        plateau_idx += 1

        plt.plot(
            group_data[xField[0]],
            group_data[ykey],
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
    plt.ylabel(f"{ykey} [{yField[1]}]")
    plt.xlabel(f"{xField[0]} [{xField[1]}]")
    plt.grid(True)

    if show:
        plt.show()
    if save:
        plt.savefig(f"{ykey}-{xField[0]}.png")

    plt.close()

    return plateau_data


def plateaus(
    Data: MagnetData,
    yField: tuple = ("Field", "T"),
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

    group = None
    df = pd.DataFrame()
    if not isinstance(Data.Data, pd.DataFrame):
        (group, channel) = yField[0].split("/")
        df = Data.getData([f"{group}/t", yField[0]])
        ykey = channel
    else:
        df = Data.getData()
        ykey = yField[0]

    ax = plt.gca()

    # TODO:
    # pass b_thresold as input param
    # b_threshold = 1.e-3

    if debug:
        print("Search for plateaux:", "Type:", Data.Type)

    Bz = df[ykey]
    B_min = float(Bz.min())
    B_max = float(Bz.max())
    B_mean = float(Bz.mean())
    B_std = float(Bz.std())

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

    ax = plt.gca()
    df_.plot(ax=ax, grid=True)

    if save:
        # imagefile = self.Site + "_" + self.Insert
        imagefile = "plateaux"
        (start_date, start_time, end_date, end_time) = Data.getStartDate(group)

        plt.savefig(
            f"{imagefile}_{str(start_date)}---{str(start_time)}.png",
            dpi=300,
        )
    if show:
        plt.show()

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
        if Data.Type == 0:
            t0 = Data.Data["timestamp"].iloc[p[0]]
            t1 = Data.Data["timestamp"].iloc[p[1]]
        else:
            t0 = df.index[0]
            t1 = df.index[-1]
        dt = t1 - t0

        # b0=Data.getData('Field').values.tolist()[p[0]]
        b0 = float(Bz.iloc[p[0]])
        b1 = float(Bz.iloc[p[1]])

        tformat = "%Y.%m.%d %H:%M:%S"
        start_time = t0.strftime(tformat)
        end_time = t1.strftime(tformat)

        if debug:
            msg = f"\t{start_time}\t{end_time}"
            print(f"{msg}\t{dt.total_seconds():8.6g}\t{b0:8.4g}\t{b1:8.4g}")

        # if (b1-b0)/b1 > b_thresold: reject plateau
        # if abs(b1) < b_thresold and abs(b0) < b_thresold: reject plateau
        if (dt / timedelta(seconds=1)) >= duration:
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
    print(f" \nField pics (aka sequence[1,-1]): {len(pics)}")
    pics = list_sequence(B_list, [1.0, 0, -1.0, 0, 1.0])
    print(f" \nField pics (aka sequence[1,0,-1,0,1]): {len(pics)}")

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


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_files", help="input txt file (ex. HL31_2018.04.13.txt)", nargs="+"
    )
    parser.add_argument(
        "--difference", help="specify difference", type=float, default=2.0e-2
    )
    parser.add_argument(
        "--min_num_points",
        help="specify minimum number of points",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--xField",
        help="specify xField (name, unit)",
        type=tuple_type,
        default="(t,s)",
    )
    parser.add_argument(
        "--yField",
        help="specify yField (name, unit)",
        type=tuple_type,
        default="(Field,T)",
    )

    parser.add_argument(
        "--show",
        help="display graphs (default save in png format)",
        action="store_true",
    )
    args = parser.parse_args()

    print(f"input_files: {args.input_files}")
    xField = args.xField
    yField = args.yField

    show = args.show
    save = not show
    threshold = args.difference
    num_points_threshold = args.min_num_points

    for name in args.input_files:
        Data = MagnetData.fromtxt(name)
        nplateaus(Data, xField, yField, threshold, num_points_threshold, show, save)


if __name__ == "__main__":
    main()
