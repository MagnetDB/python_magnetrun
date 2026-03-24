from __future__ import unicode_literals
import logging
import sys
import argparse
import pandas as pd

logger = logging.getLogger(__name__)

import matplotlib

matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt

from .plots import plot_vs_time
from .plots import plot_key_vs_key

from ..magnetdata import MagnetData
from ..MagnetRun import prepareData_legacy
from ..thermal_pipeline import CoolingCircuitColumns, compute_circuit_thermal, compute_global_thermal


def concat_files(
    input_files: list,
    housing: str,
    keys: list = [],
    debug: bool = False,
) -> pd.DataFrame:
    """Load, concatenate and prepare pupitre txt files.

    Reads one or more pupitre .txt files (or .csv files), concatenates them,
    and applies prepareData_legacy() to add the time column and rename/cleanup
    voltage/current columns (UH/UB, IH/IB, TinH/TinB, etc.).

    :param input_files: list of input file paths
    :param housing: housing name for prepareData_legacy (e.g. "M9", "M8", "M10")
    :param keys: optional subset of columns to keep before concatenation
    :param debug: activate debug logging
    :return: prepared DataFrame
    """
    logger.debug(f"concat_files: input_files={input_files}, housing={housing}, keys={keys}")

    dfs: list[pd.DataFrame] = []
    for f in input_files:
        logger.debug(f"concat_files: process {f}")
        try:
            if f.endswith(".txt"):
                _df = pd.read_csv(f, sep=r"\s+", engine="python", skiprows=1)
            else:
                _df = pd.read_csv(f, sep=",", engine="python", skiprows=0)
            if keys:
                dfs.append(_df[keys])
            else:
                dfs.append(_df)
        except Exception as e:
            raise RuntimeError(f"concat_files: failed to load {f}") from e

    raw_df = pd.concat(dfs, axis=0).dropna(axis=1, how="all").reset_index(drop=True)

    # Wrap in MagnetData and apply prepareData_legacy:
    #   - adds t and timestamp columns (via addTime)
    #   - renames Flow/Rpm/Tin/HP with H/B suffixes per housing
    #   - computes UH / UB from Ucoil groups (via cleanupData_legacy)
    #   - renames Icoil1 -> IH, Icoil[-1] -> IB
    data = MagnetData(
        filename=input_files[0],
        Groups={},
        Keys=raw_df.columns.tolist(),
        Type=0,
        Data=raw_df,
    )
    prepareData_legacy(data, housing, debug=debug)
    return data.getData()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_files",
        help="input txt file (ex. HL31_2018.04.13.txt)",
        nargs="+",
        metavar="Current",
        type=str,
    )
    parser.add_argument(
        "--housing",
        help="housing name (M9, M8, M10) - controls Flow/Tin/HP column mapping",
        type=str,
        default="M9",
    )
    parser.add_argument(
        "--plot_vs_time", help='select key(s) to plot (ex. "Field[;Ucoil1]")'
    )
    parser.add_argument(
        "--plot_key_vs_key", help='select pair(s) of keys to plot (ex. "Field-Icoil1")'
    )
    parser.add_argument("--output_time", help="output key(s) for time")
    parser.add_argument(
        "--output_timerange", help="set time range to extract (start;end)"
    )
    parser.add_argument("--output_key", help="output key(s)")
    parser.add_argument("--extract_pairkeys", help="dump key(s) to file")
    parser.add_argument(
        "--show",
        help="display graphs (default save in png format)",
        action="store_true",
    )
    parser.add_argument("--list", help="list keys in csv", action="store_true")
    parser.add_argument("--convert", help="convert file to csv", action="store_true")
    parser.add_argument("--debug", help="activate debug", action="store_true")
    args = parser.parse_args()

    df = concat_files(args.input_files, args.housing, debug=args.debug)
    keys = df.columns.values.tolist()

    if args.list:
        print(f"keys={keys}")
        sys.exit(0)

    # Helix cooling circuit thermal computations
    # After prepareData_legacy: UH, IH, FlowH, TinH, HPH are present
    if "UH" in keys and "PeH" not in keys:
        circuit_H = CoolingCircuitColumns(
            voltage="UH",
            current="IH",
            flow="FlowH",
            temp_in="TinH",
            pressure_in="HPH",
            back_pressure="BP",
        )
        df = compute_circuit_thermal(df, circuit_H)

    # Bitter cooling circuit (only present for hybrid magnets)
    if "UB" in keys and "PeB" not in keys:
        circuit_B = CoolingCircuitColumns(
            voltage="UB",
            current="IB",
            flow="FlowB",
            temp_in="TinB",
            pressure_in="HPB",
            back_pressure="BP",
        )
        df = compute_circuit_thermal(df, circuit_B)

    # Global thermal quantities (only when both circuits are present)
    if "PeH" in df.columns and "PeB" in df.columns:
        df = compute_global_thermal(df, circuit_H, circuit_B)

    keys = df.columns.values.tolist()

    if args.plot_vs_time:
        print("plot_vs_time=", args.plot_vs_time)
        items = args.plot_vs_time.split(";")
        plot_vs_time(df, items, args.show)

    if args.plot_key_vs_key:
        pairs = args.plot_key_vs_key.split(";")
        plot_key_vs_key(df, pairs, args.show)

    if args.output_time:
        times = args.output_time.split(";")
        print("Select data at %s " % (times))
        if args.output_key:
            cols = args.output_key.split(";")
            print(df[df["Time"].isin(times)][cols])
        else:
            print(df[df["Time"].isin(times)])

    if args.output_timerange:
        timerange = args.output_timerange.split(";")
        print("Select data from %s to %s" % (timerange[0], timerange[1]))
        file_name = "timerange"
        file_name += "_from" + str(timerange[0].replace(":", "-"))
        file_name += "_to" + str(timerange[1].replace(":", "-")) + ".csv"

        selected_df = df[df["Time"].between(timerange[0], timerange[1], inclusive="both")]
        if args.output_key:
            cols = args.output_key.split(";")
            cols.insert(0, "Time")
            selected_df[cols].to_csv(file_name, sep="\t", index=False, header=True)
        else:
            selected_df.to_csv(file_name, sep="\t", index=False, header=True)

    if args.extract_pairkeys:
        pairs = args.extract_pairkeys.split(";")
        for pair in pairs:
            items = pair.split("-")
            if len(items) != 2:
                print("invalid pair of keys: %s" % pair)
                sys.exit(1)
            key1, key2 = items
            newdf = pd.concat([df[key1], df[key2]], axis=1)
            newdf = newdf[(newdf[key1] != 0) & (newdf[key2] != 0)]
            newdf.to_csv(f"{pair}.csv", sep="\t", index=False, header=False)

    if args.convert:
        df.to_csv("concatdf.csv", index=False, header=True, date_format=str)


if __name__ == "__main__":
    main()
