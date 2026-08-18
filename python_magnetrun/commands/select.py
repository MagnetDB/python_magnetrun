"""Select command: extract and export MagnetRun data to CSV."""

import argparse
import logging
import os

import matplotlib.pyplot as plt

from ..magnetdata_base import DataType
from ..MagnetRun import MagnetRun
from ..processing.smoothers import savgol

logger = logging.getLogger(__name__)


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
            from ..processing.smoothers import (
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
                            y_smoothed = lowess_bell_shape_kern(
                                x, y, args.smoothing_tau
                            )
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
    if mdata.Type == DataType.PUPITRE:
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
        file_name = file_name + "_from" + timerange[0].replace(":", "-").replace(" ", "T")
        file_name = file_name + "_to" + timerange[1].replace(":", "-").replace(" ", "T") + ".csv"

        if mdata.Type == DataType.PUPITRE:
            selected_df = mdata.extractTimeData(item)
            if selected_df is not None:
                selected_df.to_csv(file_name, sep="\t", index=False, header=True)
        elif mdata.Type == DataType.TDMS:
            for group in mdata.Groups:
                selected_df = mdata.extractTimeData(item, group)
                if selected_df is not None:
                    selected_df.to_csv(file_name, sep="\t", index=False, header=True)


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

    if mdata.Type == DataType.PUPITRE:
        data = mdata.getData()
        df = data[data["t"].isin(times)]
        if mdata.start_timestamp is not None:
            import pandas as pd

            df = df.copy()
            df["timestamp"] = mdata.start_timestamp + pd.to_timedelta(df["t"], unit="s")
        file_name = file.replace(f_extension, "")
        file_name = file_name + select_args_str + ".csv"
        df.to_csv()

    elif mdata.Type == DataType.TDMS:
        for group in mdata.Groups:
            df = mdata.getData(group)
            df = df[df.index.isin(times)]
            file_name = file.replace(f_extension, f"-{group}")
            file_name = file_name + select_args_str + ".csv"
            df.to_csv()


def _run(args: "argparse.Namespace") -> int:
    import logging

    from ..log_utils import setup_logging
    from ._shared import load_inputs

    log_level = getattr(logging, getattr(args, "log_level", "WARNING").upper(), logging.WARNING)
    setup_logging(level=log_level, log_file=getattr(args, "log_file", None))

    input_files, inputs, extensions = load_inputs(args)
    if not inputs:
        logger.error("No files loaded.")
        return 1

    if getattr(args, "output_time", None):
        times = args.output_time
        for file in input_files:
            if file in inputs:
                output_time(file, inputs, extensions, times)

    if getattr(args, "output_timerange", None):
        for file in input_files:
            if file in inputs:
                output_timerange(file, inputs, extensions, args)

    if getattr(args, "output_key", None):
        for file in input_files:
            if file in inputs:
                output_keys(file, inputs, extensions, args)

    if getattr(args, "extract_pairkeys", None):
        for file in input_files:
            if file in inputs:
                extract_pairkeys(file, inputs, extensions, args)

    if getattr(args, "convert", False):
        for file in input_files:
            if file in inputs:
                convert_to_csv(file, inputs)

    return 0


def register(sub: "argparse._SubParsersAction") -> None:
    from ..cli_args import create_base_parser, create_common_smoothing_parser

    base = create_base_parser(add_input_file=False)
    smoothing_parser = create_common_smoothing_parser()

    p = sub.add_parser(
        "select",
        parents=[base, smoothing_parser],
        help="extract or export run data",
    )
    p.add_argument("input_file", nargs="+", help="input file(s)")
    p.add_argument("--output_time", nargs="+", help="output key(s) for time")
    p.add_argument(
        "--output_timerange",
        help="time range to extract (YYYY-MM-DD HH:MM:SS;YYYY-MM-DD HH:MM:SS, local)",
        action="append",
    )
    p.add_argument("--output_key", nargs="+", help="key(s) to extract", action="append")
    p.add_argument("--extract_pairkeys", nargs="+", help="key pair(s) to dump", action="append")
    p.add_argument("--convert", action="store_true", help="convert file to CSV")
    p.set_defaults(_handler=_run)
