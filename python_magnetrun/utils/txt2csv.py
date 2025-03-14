from __future__ import unicode_literals
import sys
import argparse
import datetime
import pandas as pd
import numpy as np
import matplotlib

# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from .files import concat_files
from .plots import plot_vs_time
from .plots import plot_key_vs_key

from ..cooling import water


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_files",
        help="input txt file (ex. HL31_2018.04.13.txt)",
        nargs="+",
        metavar="Current",
        type=str,
        default="HL31_2018.04.13.txt",
    )
    parser.add_argument(
        "--plot_vs_time", help='select key(s) to plot (ex. "Field[;Ucoil1]")'
    )
    parser.add_argument(
        "--plot_key_vs_key", help='select pair(s) of keys to plot (ex. "Field-Icoil1'
    )
    parser.add_argument("--output_time", help="output key(s) for time")
    parser.add_argument(
        "--output_timerange", help="set time range to extract (start;end)"
    )
    parser.add_argument("--output_key", help="output key(s) for time")
    parser.add_argument("--extract_pairkeys", help="dump key(s) to file")
    parser.add_argument(
        "--show",
        help="display graphs (default save in png format)",
        action="store_true",
    )
    parser.add_argument("--list", help="list key in csv", action="store_true")
    parser.add_argument("--convert", help="convert file to csv", action="store_true")
    parser.add_argument(
        "--missing", help="detect eventually missong probes", action="store_true"
    )
    parser.add_argument(
        "--nhelices", help="specify number of helices", type=int, default=14
    )
    parser.add_argument(
        "--check",
        help="returns True if active voltage taps==nhelices",
        action="store_true",
    )
    args = parser.parse_args()

    df = concat_files(args.input_files)

    # Get Name of columns
    keys = df.columns.values.tolist()
    # print(f"keys={keys}")

    max_tap = 0
    for i in range(1, args.nhelices + 1):
        ukey = "Ucoil%d" % i
        # print ("Ukey=%s" % ukey, (ukey in keys) )
        if ukey in keys:
            max_tap = i
    if args.check:
        # print ("max_tap=%d" % max_tap)
        print(max_tap == args.nhelices)
        sys.exit(0)

    missing_probes = []
    for i in range(1, max_tap + 1):
        ukey = "Ucoil%d" % i
        if ukey not in keys:
            # print ("Ukey=%s" % ukey, (ukey in keys) )
            df[ukey] = 0
            missing_probes.append(i)
            # Add an empty column

    if args.missing:
        print(f"missing_probes: {missing_probes}")

    # Get Name of columns
    keys = df.columns.values.tolist()
    if args.list:
        print(f"keys={keys}")
        sys.exit(0)

    # Add some more columns
    # Helix Magnet
    if "Uh" not in keys:
        # print("Create extra columns [ie: U1,U2,...]")
        df["Uh"] = 0
        for i in range(15):
            ukey = "Ucoil%d" % i
            if ukey in keys:
                df["Uh"] += df[ukey]

        df["Pe1"] = df.apply(lambda row: row.Uh * row.Icoil1 / 1.0e6, axis=1)
        df["DP1"] = df["HP1"] - df["BP"]

        # Get Water property
        df["rho1"] = df.apply(lambda row: water.getRho(row.BP, row.Tin1), axis=1)
        df["cp1"] = df.apply(lambda row: water.getCp(row.BP, row.Tin1), axis=1)

        for i in range(1, 6):
            df["DT1"] = df.apply(
                lambda row: (
                    row.Pe1 * 1.0e6 / (row.rho1 * row.cp1 * row.Flow1 * 1.0e-3)
                    if (row.Flow1 != 0)
                    else row.Tin1
                ),
                axis=1,
            )

            # Water Property at BP bar, Tin+DT1
            df["rho1"] = df.apply(
                lambda row: water.getRho(
                    (row.BP + row.DP1 / 2.0), (row.Tin1 + row.DT1 / 2.0)
                ),
                axis=1,
            )
            df["cp1"] = df.apply(
                lambda row: water.getCp(
                    (row.BP + row.DP1 / 2.0), (row.Tin1 + row.DT1 / 2.0)
                ),
                axis=1,
            )

        df["Tout1"] = df["Tin1"] + df["DT1"]
        df["Tw1"] = df.apply(lambda row: row.Tin1 + row.DT1 / 2.0, axis=1)
        df["P1"] = df.apply(lambda row: (row.HP1 + row.BP) / 2.0, axis=1)

    # Helix Magnet
    if "Ub" not in keys:
        # Bitter
        if "Ucoil15" in keys:
            df["Ub"] = df["Ucoil15"] + df["Ucoil16"]
            df["Pe2"] = df.apply(lambda row: row.Ub * row.Icoil15 / 1.0e6, axis=1)

            # Water Property at BP bar, Tin+DT1/2.
            df["rho2"] = df.apply(lambda row: water.getRho(row.BP, row.Tin2), axis=1)
            df["cp2"] = df.apply(lambda row: water.getCp(row.BP, row.Tin2), axis=1)

            df["DP2"] = df["HP2"] - df["BP"]
            for i in range(1, 6):
                df["DT2"] = df.apply(
                    lambda row: (
                        row.Pe2 * 1.0e6 / (row.rho2 * row.cp2 * row.Flow2 * 1.0e-3)
                        if (row.Flow2 != 0)
                        else row.Tin2
                    ),
                    axis=1,
                )
                df["rho2"] = df.apply(
                    lambda row: water.getRho(
                        (row.BP + row.DP2 / 2.0),
                        (row.Tin2 + row.DT2 / 2.0),
                    ),
                    axis=1,
                )
                df["cp2"] = df.apply(
                    lambda row: water.getCp(
                        (row.BP + row.DP2 / 2.0),
                        (row.Tin2 + row.DT2 / 2.0),
                    ),
                    axis=1,
                )

            df["Tout2"] = df["Tin2"] + df["DT2"]
            df["Tw2"] = df.apply(lambda row: row.Tin2 + row.DT2 / 2.0, axis=1)
            df["P2"] = df.apply(lambda row: (row.HP2 + row.BP) / 2.0, axis=1)

    # If M9: Icoil1=Idcct1+Iddct2, If M10/M8: Icoil2=Idcct3+Iddct4
    df["I1"] = df["Idcct1"] + df["Idcct2"]
    df["I2"] = df["Idcct3"] + df["Idcct4"]

    del df["Idcct1"]
    del df["Idcct2"]
    del df["Idcct3"]
    del df["Idcct4"]

    for i in range(2, max_tap):
        ikey = "Icoil%d" % i
        del df[ikey]

    if "Icoil16" in keys:
        del df["Icoil16"]

    df["Power"] = df["Pe1"] + df["Pe2"]

    df["rho"] = df.apply(lambda row: water.getRho(row.BP, row.Tout), axis=1)
    df["cp"] = df.apply(lambda row: water.getCp(row.BP, row.Tout), axis=1)
    df["DT"] = df.apply(
        lambda row: (
            row.Pmagnet * 1.0e6 / (row.rho * row.cp * (row.Flow1 + row.Flow2) * 1.0e-3)
            if (row.Flow1 * row.Flow2 != 0)
            else row.Tin2
        ),
        axis=1,
    )

    df["Toutg"] = (
        df["rho1"] * df["cp1"] * df["Flow1"] * df["Tout1"]
        + df["rho2"] * df["cp2"] * df["Flow2"] * df["Tout2"]
    )
    df["Toutg"] = df["Toutg"] / (
        df["rho1"] * df["cp1"] * df["Flow1"] + df["rho2"] * df["cp2"] * df["Flow2"]
    )

    # update keys
    keys = df.columns.values.tolist()

    # Check data type
    # print pd.api.types.is_string_dtype(df['Icoil1'])
    # print pd.api.types.is_numeric_dtype(df['Icoil1'])
    # df[['Field', 'Icoil1']] = df[['Field', 'Icoil1']].apply(pd.to_numeric)

    if args.plot_vs_time:
        print("plot_vs_time=", args.plot_vs_time)
        # split into keys
        items = args.plot_vs_time.split(";")
        plot_vs_time(df, items, args.show)

    if args.plot_key_vs_key:
        pairs = args.plot_key_vs_key.split(";")
        plot_key_vs_key(df, pairs, args.show)

    if args.output_time:
        times = args.output_time.split(";")
        print("Select data at %s " % (times))
        if args.output_key:
            keys = args.output_key.split(";")
            print(df[df["Time"].isin(times)][keys])
        else:
            print(df[df["Time"].isin(times)])

    if args.output_timerange:
        timerange = args.output_timerange.split(";")
        print("Select data from %s to %s" % (timerange[0], timerange[1]))
        file_name = "timerange"  # input_file.replace(".txt", "")
        file_name = file_name + "_from" + str(timerange[0].replace(":", "-"))
        file_name = file_name + "_to" + str(timerange[1].replace(":", "-")) + ".csv"

        selected_df = df[df["Time"].between(timerange[0], timerange[1], inclusive=True)]
        if args.output_key:
            keys = args.output_key.split(";")
            keys.insert(0, "Time")
            # keys.append('Time')
            # print(selected_df[keys])
            selected_df[keys].to_csv(file_name, sep=str("\t"), index=False, header=True)
        else:
            selected_df.to_csv(file_name, sep=str("\t"), index=False, header=True)

    if args.extract_pairkeys:
        pairs = args.extract_pairkeys.split(";")
        for pair in pairs:
            items = pair.split("-")
            if len(items) != 2:
                print("invalid pair of keys: %s" % pair)
                sys.exit(1)
            key1 = items[0]
            key2 = items[1]
            newdf = pd.concat([df[key1], df[key2]], axis=1)  # , keys=['df1', 'df2'])

            # Remove line with I=0
            newdf = newdf[newdf[key1] != 0]
            newdf = newdf[newdf[key2] != 0]

            file_name = str(pair) + ".csv"
            newdf.to_csv(file_name, sep=str("\t"), index=False, header=False)

    # Save to CSV
    if args.convert:
        df.to_csv("concatdf.csv", index=False, header=True, date_format=str)


if __name__ == "__main__":
    main()
