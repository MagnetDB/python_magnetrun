"""Main module."""

import logging
import os
import sys
import traceback

import matplotlib
import pandas as pd
from natsort import natsorted

from .commands.add import add_field
from .commands.plot import plot_key_vs_key, plot_vs_time
from .commands.select import (
    convert_to_csv,
    extract_pairkeys,
    output_keys,
    output_time,
    output_timerange,
)
from .commands.stats import display_stats
from .hybrid import HybridRun
from .log_utils import format_exception_location, setup_logging
from .magnetdata_base import DataType
from .MagnetRun import MagnetRun
from .utils.files import expand_input_files

# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available

# Setup logger — must NOT use __name__ here because when this module is
# executed as `python -m python_magnetrun.cli`, __name__ == "__main__" which
# is outside the "python_magnetrun.*" hierarchy and therefore bypasses the
# handler configured by setup_logging().
logger = logging.getLogger("python_magnetrun.cli")


def main():
    """Deprecated entry point.  Use ``magnetrun <subcommand>`` instead."""
    import warnings

    warnings.warn(
        "python-magnetrun is deprecated. Use 'magnetrun <subcommand>' instead.",
        DeprecationWarning,
        stacklevel=1,
    )

    from .args import create_argument_parser, get_datadir_mapping

    parser = create_argument_parser()
    args = parser.parse_args(sys.argv[1:])

    # Configure logging level
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(level=log_level, log_file=args.log_file if args.log_file else None)

    logger.debug(f"args: {args}")

    # load df pandas from input_file
    # check extension
    supported_formats = [".txt", ".tdms", ".csv"]

    # Create datadir dictionary mapping extensions to data directories
    datadir = get_datadir_mapping(args)

    # Expand glob patterns in input_file arguments
    expanded_files = expand_input_files(
        args.input_file, datadir, housing=getattr(args, "housing", None)
    )
    # Group files by extension, natsort within each group, preserving first-seen order
    extensions = {}
    for file in expanded_files:
        ext = os.path.splitext(file)[-1]
        extensions.setdefault(ext, []).append(file)
    for ext in extensions:
        extensions[ext] = natsorted(extensions[ext])

    input_files = [f for files in extensions.values() for f in files]
    # Remap extension -> list of indices into input_files
    extensions = {}
    for i, file in enumerate(input_files):
        ext = os.path.splitext(file)[-1]
        extensions.setdefault(ext, []).append(i)
    logger.debug(f"input_files: {input_files}")
    logger.debug(f"first file: {input_files[0]}, last file: {input_files[-1]}")
    logger.debug(f"extensions: {extensions}, entries: {len(extensions.keys())}")
    # logger.debug(f"args.vs_time: {args.vs_time}, entries: {len(args.vs_time)}")

    inputs = {}
    for file in input_files:
        f_extension = os.path.splitext(file)[-1]
        if f_extension not in supported_formats:
            raise RuntimeError(
                f"so far file with extension in {supported_formats} are implemented"
            )

        filename = os.path.basename(file)
        insert = args.insert if args.insert else "notdefined"
        housing = args.housing if args.housing else "notdefined"
        site = args.site if args.site else "notdefined"
        result = filename.startswith("M")
        if result:
            try:
                index = filename.index("_")
                housing = filename[0:index] if housing == "notdefined" else housing
                # print(f"site detected: {site}")
            except ValueError:
                logger.warning(
                    f"{file}: no site detected - use args.site argument instead"
                )
                continue
        if args.log_level == "DEBUG":
            logger.debug(f"housing={housing}, site={site}")

        try:
            match f_extension:
                case ".txt":
                    mrun = MagnetRun.fromtxt(housing=housing, site=site, filename=file)
                case ".tdms":
                    mrun = MagnetRun.fromtdms(housing=housing, site=site, filename=file)
                case ".csv":
                    mrun = MagnetRun.fromcsv(housing=housing, site=site, filename=file)
                case _:
                    raise RuntimeError(
                        f"so far file with extension in {supported_formats} are implemented"
                    )
        except (OSError, ValueError, RuntimeError) as error:
            # Print detailed error information with traceback
            tb_str = "".join(traceback.format_exception(*sys.exc_info()))
            logger.error(
                f"{file}: an error occurred when loading at {format_exception_location()}"
            )
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
                print(f"{file}: valid keys")
                mdata.info()

            if args.convert:
                if mdata.Type == DataType.PUPITRE:
                    data = mdata.getData()
                    csvfile = file.replace(f_extension, ".csv")
                    data.to_csv(csvfile, sep="\t", index=True, header=True)
                elif mdata.Type == DataType.TDMS:
                    for group in mdata.Groups:
                        logger.debug(f"convert: group={group}")
                        df = mdata.getData(group)
                        csvfile = file.replace(f_extension, f"-{group}.csv")
                        df.to_csv(csvfile, sep="\t", index=True, header=True)

    # Load hybrid data if requested
    if args.hybrid_datadir and args.hybrid_date:
        try:
            hrun = HybridRun.fromdir(
                base_dir=args.hybrid_datadir,
                date_str=args.hybrid_date,
                fepc_system=args.fepc_system,
                site=args.site or "",
            )
            inputs["hybrid"] = {"data": hrun}
            logger.info(
                f"loaded hybrid data: {args.hybrid_datadir}, date={args.hybrid_date}, "
                f"fepc_system={args.fepc_system}"
            )
        except (OSError, ValueError, RuntimeError) as error:
            tb_str = "".join(traceback.format_exception(*sys.exc_info()))
            logger.error(
                f"hybrid data ({args.hybrid_datadir}, {args.hybrid_date}): "
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
            ), f"expected {len(extensions.keys())} vs_time arguments - got {len(args.vs_time)} - extensions={list(extensions.keys())}"

            plot_vs_time(input_files, inputs, extensions, args)

        if args.key_vs_key:
            assert len(args.key_vs_key) == len(
                extensions.keys()
            ), f"expected {len(extensions.keys())} key_vs_key arguments - got {len(args.key_vs_key)} - extensions={list(extensions.keys())} "

            plot_key_vs_key(input_files, inputs, extensions, args)

    if args.command == "select":
        # plot_args = items[extensions[f_extension][0]]
        if args.output_time:
            assert len(args.output_time) == len(
                extensions.keys()
            ), f"expected {len(extensions.keys())} output_time arguments - got {len(args.output_time)} "

            times = args.output_time.split(";")
            logger.info(f"Select data at {times}")
            for file in inputs:
                output_time(file, inputs, extensions, times)

        if args.output_timerange:
            assert len(args.output_timerange) == len(
                extensions.keys()
            ), f"expected {len(extensions.keys())} output_timerange arguments - got {len(args.output_timerange)} "

            timerange = args.output_timerange.split(";")
            for file in inputs:
                output_timerange(file, inputs, extensions, args)

        if args.output_key:
            assert len(args.output_key) == len(
                extensions.keys()
            ), f"expected {len(extensions.keys())} output_key arguments - got {len(args.output_key)} "

            for file in inputs:
                output_keys(file, inputs, extensions, args)

        if args.extract_pairkeys:
            assert len(args.extract_pairkeys) == len(
                extensions.keys()
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
        plateau_index = []
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
            columns, data = display_stats(
                file, inputs, args, multiindex, columns, data, plateau_index
            )

        logger.debug(
            f"concat tabs: "
            f"multi_index={len(multiindex[0]), len(multiindex[1])}, "
            f"columns={len(columns)}, "
            f"data={len(data)}"
        )
        logger.debug(f"multiindex: {multiindex}")
        logger.debug(f"columns: {columns}")
        logger.debug(f"data: {data}")

        if args.plateau:
            index = pd.MultiIndex.from_tuples(plateau_index, names=["file", "key"])
        else:
            index = pd.MultiIndex.from_product(multiindex)
        df = pd.DataFrame(
            data,
            index,
            columns=columns,
        )
        print(df.to_markdown(tablefmt="simple"))

        # save df to csv
        logger.debug(f"head: {df.head()}")
        df.to_csv(f"{output}.csv")


if __name__ == "__main__":
    main()
