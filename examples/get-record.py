"""Console script."""

import argparse
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

from python_magnetrun.cli_args import create_base_parser, get_datadir_mapping
from python_magnetrun.log_utils import get_logger, setup_logging
from python_magnetrun.magnetdata_base import MagnetDataBase
from python_magnetrun.MagnetRun import MagnetRun
from python_magnetrun.processing.correlations import pearson
from python_magnetrun.processing.stats import stats
from python_magnetrun.utils.files import expand_input_files
from python_magnetrun.utils.timestamps import parse_filename_timestamp

logger = get_logger(__name__)


def load_record(file: str, args, show: bool = False) -> MagnetDataBase:
    """Load record from a pupitre .txt or pigbrother .tdms file."""
    filename = os.path.basename(file)
    extension = os.path.splitext(filename)[-1]
    housing = args.housing if args.housing != "notdefined" else filename.split("_")[0]
    site = args.site

    try:
        if extension == ".txt":
            mrun = MagnetRun.fromtxt(housing, site, file)
        elif extension == ".tdms":
            mrun = MagnetRun.fromtdms(site, housing, file)
        else:
            raise RuntimeError(f"unsupported file extension: {extension}")
    except RuntimeError:
        raise
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"{file}: failed to load ({type(e).__name__}: {e})") from e

    data = mrun.MagnetData
    if not isinstance(data, MagnetDataBase):
        raise RuntimeError(
            f"{file}: cannot load data as MagnetDataBase (got {type(data).__name__})"
        )

    return data


def select_data(data, args) -> dict | None:
    """Return a result dict if the record passes selection criteria, else None."""
    duration = data.getDuration()
    if duration > args.duration:
        bdata = data.extractDataThreshold("Field", args.field)
        if not bdata.empty:
            bfield = data.getData("Field").squeeze()
            return {
                "file": data.FileName,
                "duration (s)": f"{duration:.1f}",
                "Field min (T)": f"{bfield.min():.3f}",
                "Field mean (T)": f"{bfield.mean():.3f}",
                "Field max (T)": f"{bfield.max():.3f}",
            }
    return None


def _sort_key(file: str) -> datetime:
    """Return the start timestamp for *file*, raising RuntimeError if unparseable."""
    ts = parse_filename_timestamp(file)
    if ts is None:
        raise RuntimeError(f"cannot parse timestamp from filename: {file!r}")
    return ts


def main():
    """Console script."""
    base_parser = create_base_parser()
    parser = argparse.ArgumentParser(parents=[base_parser])

    subparsers = parser.add_subparsers(
        title="commands", dest="command", help="sub-command help"
    )
    parser_select = subparsers.add_parser("select", help="select help")
    parser_stats = subparsers.add_parser("stats", help="stats help")
    parser_plot = subparsers.add_parser("plot", help="select help")
    parser_aggregate = subparsers.add_parser("aggregate", help="select help")

    # subcommand select
    parser_select.add_argument(
        "--duration",
        help="select record with a duration more than",
        type=float,
        default="60",
    )
    parser_select.add_argument(
        "--fields", help="select fields to plot", type=str, nargs="+"
    )
    parser_select.add_argument(
        "--field",
        help="select field with a value more than",
        type=float,
        default="18.",
    )

    # subcommand plot
    parser_plot.add_argument(
        "--fields", help="select fields to plot", type=str, nargs="+"
    )
    parser_plot.add_argument(
        "--xfield", help="select x to plot", type=str, default="timestamp"
    )
    parser_plot.add_argument("--show", help="enable show mode", action="store_true")
    parser_plot.add_argument("--save", help="enable save mode", action="store_true")
    parser_plot.add_argument("--title", type=str, default=None, metavar="TEXT", help="override plot title")

    # subcommand aggregate
    parser_aggregate.add_argument(
        "--fields", help="select fields to aggregate", type=str, nargs="+"
    )
    parser_aggregate.add_argument(
        "--name", help="set basename of file to be saved", type=str
    )
    parser_aggregate.add_argument(
        "--show", help="enable show mode", action="store_true"
    )
    parser_aggregate.add_argument(
        "--save", help="enable save mode", action="store_true"
    )
    parser_aggregate.add_argument("--title", type=str, default=None, metavar="TEXT", help="override plot title")

    # subcommand stats
    parser_stats.add_argument("--fields", help="select fields", type=str, nargs="+")
    parser_stats.add_argument(
        "--pairplot", help="enable save mode", action="store_true"
    )
    parser_stats.add_argument(
        "--pearson", help="enable Pearson correlation calculation", action="store_true"
    )
    parser_stats.add_argument(
        "--tlcc", help="enable TLCC correlation calculation", action="store_true"
    )
    parser_stats.add_argument(
        "--dtw", help="enable DTW correlation calculation", action="store_true"
    )
    parser_stats.add_argument("--show", help="enable show mode", action="store_true")
    parser_stats.add_argument("--save", help="enable save mode", action="store_true")
    parser_stats.add_argument("--title", type=str, default=None, metavar="TEXT", help="override plot title")

    args = parser.parse_args()
    logger.error(f"get-record: Arguments={args}, pwd={os.getcwd()}")

    setup_logging(level=args.log_level, log_file=args.log_file)
    debug = args.log_level == "DEBUG"

    # Always print input parameters for traceability
    logger.info(f"getrecords: Arguments={args}, pwd={os.getcwd()}")

    # Expand glob patterns and search configured data directories
    datadir = get_datadir_mapping(args)
    files = expand_input_files(args.input_file, datadir, args.housing)
    print(f"files: {len(files)} records, datadir={datadir}")

    # need to be sorted by time??
    files = sorted(files, key=_sort_key, reverse=False)
    logger.debug(f"sort by time: {files}")

    selected_keys = []
    if args.command == "plot":
        if args.xfield:
            selected_keys += [args.xfield]
        if args.fields:
            logger.debug(f"args.fields={args.fields}")
            selected_keys += args.fields
    elif args.command == "aggregate":
        selected_keys += args.fields
    else:
        selected_keys += ["Field"]
    logger.debug(f"selected_keys={selected_keys}")

    min_duration = 60
    if args.command == "select":
        min_duration = args.duration
    elif args.command == "stats":
        logger.warning(f"Overwriting min duration for stats: {min_duration} -> 1000")
        min_duration = 1000

    if "timestamp" not in selected_keys:
        selected_keys.append("timestamp")
    logger.debug(f"selected_keys (with timestamp)={selected_keys}")

    ax = plt.gca()

    if args.command == "aggregate":
        logger.info(f"aggregate: fields={selected_keys}")

        df_: list[pd.DataFrame] = []

        for file in files:
            try:
                logger.info(f"record: {file}")
                data = load_record(file, args)
                # logger.info(f"record: {file} - duration: {data.getDuration():.1f} s")
                if data.getDuration() >= min_duration:
                    try:
                        df_.append(data.getData(selected_keys))
                    except (KeyError, IndexError) as error:
                        logger.error(
                            f"record: {file} - ignored: {selected_keys} not all in {data.getKeys()} (error={error})",
                        )
                else:
                    logger.info(
                        f"record: {file} - skipped (duration < {min_duration} s)"
                    )

            except (OSError, ValueError, RuntimeError) as error:
                logger.error(f"record: {file} - fail to load ({error})")

        logger.info(f"aggregate: {len(df_)} dataframes collected")

        try:
            df = pd.concat(df_, axis=0)
        except ValueError as _exc:
            if "different registries" not in str(_exc):
                raise
            col_regs: dict[str, list[tuple[str, object]]] = {}
            for frame in df_:
                for col, entry in (frame.attrs.get("units") or {}).items():
                    unit = entry[1] if isinstance(entry, tuple) else entry
                    if unit is None:
                        continue
                    col_regs.setdefault(col, []).append(
                        (str(unit), id(getattr(unit, "_REGISTRY", None)))
                    )
            lines = []
            for col, entries in col_regs.items():
                reg_ids = {reg_id for _, reg_id in entries}
                if len(reg_ids) > 1:
                    unit_strs = list(dict.fromkeys(u for u, _ in entries))
                    lines.append(
                        f"  column {col!r}: unit representations {unit_strs} "
                        f"span {len(reg_ids)} different pint registries"
                    )
            detail = "\n".join(lines) if lines else "  (no specific column identified)"
            raise ValueError(
                f"pd.concat failed — pint Unit registry mismatch.\n"
                f"Columns with conflicting registries:\n{detail}\n"
                "Ensure all records use the same pint UnitRegistry instance."
            ) from _exc
        output = f"aggregate-{'-'.join(args.fields)}.csv"
        df.to_csv(output)
        logger.debug(f"concat dataframe:\n{df.head()}")
        logger.info(f"saved {df.columns.values.tolist()} to {os.getcwd()}/{output}")

        df["month"] = df["timestamp"].dt.month
        df["year"] = df["timestamp"].dt.year
        df["day"] = df["timestamp"].dt.day
        logger.debug(f"concat df:\n{df.head()}")

        n_months = df["month"].nunique()
        n_years = df["year"].nunique()
        if n_months > 1:
            x_col, hue_col = "month", "year"
            x_label = "Month"
        else:
            x_col = "day"
            x_label = "Day"
            hue_col = "year" if n_years > 1 else None
        logger.info(
            f"aggregate plot: x={x_col}, hue={hue_col} ({n_months} month(s), {n_years} year(s))"
        )

        # Build housing label
        if args.housing != "notdefined":
            housing_label = f"housing={args.housing}"
        else:
            housings = sorted({os.path.basename(f).split("_")[0] for f in files})
            housing_label = (
                f"housing={housings[0]}"
                if len(housings) == 1
                else f"housings={housings}"
            )

        # Build date label from the actual data range
        if n_months == 1:
            date_label = datetime(
                int(df["year"].iloc[0]), int(df["month"].iloc[0]), 1
            ).strftime("%B %Y")
        elif n_years == 1:
            date_label = str(int(df["year"].iloc[0]))
        else:
            date_label = f"{int(df['year'].min())}-{int(df['year'].max())}"

        if args.fields:
            import seaborn as sns

            for key in args.fields:
                logger.info(f"seaborn plot for {key} per {x_col}")
                ax = sns.lineplot(x=x_col, y=key, hue=hue_col, data=df)

                (symbol, unit) = data.getUnitKey(key)
                ax.set_ylabel(f"{symbol} [{unit:~P}]")
                ax.set_xlabel(x_label)
                ax.set_title(args.title or f"{key}: {housing_label}, {date_label}")
                plt.grid()
                if args.show:
                    plt.show()
                if args.save:
                    logger.info(
                        f"seaborn plot for {key} saved to {os.getcwd()}/{key}-seaborn.png"
                    )
                    plt.savefig(f"{key}-seaborn.png", dpi=300)
                plt.close()

        return 0

    # other commands
    legends = {}
    select_rows: list[dict] = []

    for file in files:
        try:
            logger.info(f"record: {file}")
            data = load_record(file, args)
            logger.info(f"record: {file} - duration: {data.getDuration():.1f} s")
        except (OSError, ValueError, RuntimeError) as error:
            logger.error(f"record: {file} - fail to load ({error})")
            # is it possible to curate txt files
            # reading csv line by line with import csv??

        else:
            data.Units()
            if args.command == "select":
                row = select_data(data, args)
                if row is not None:
                    logger.info(f"record: {file} - selected")
                    select_rows.append(row)
                else:
                    logger.info(f"record: {file} - skipped")

            elif args.command == "stats":
                if (
                    args.pearson and data.getDuration() >= min_duration
                ):  # previous limit 1000:
                    pearson(data, args.fields, args.save, args.show, debug)
                elif (
                    args.pairplot and data.getDuration() >= min_duration
                ):  # previous limit 1000:
                    import seaborn as sns

                    selected_keys = [
                        "Field",
                        "IH",
                        "IB",
                        "TinH",
                        "TinB",
                        "Tout",
                        "HPH",
                        "HPB",
                        "BP",
                        "FlowH",
                        "FlowB",
                        "RpmH",
                        "RpmB",
                        "Pmagnet",
                        "Ptot",
                        "teb",
                        "tsb",
                        "debitbrut",
                    ]

                    if args.fields:
                        selected_keys = args.fields

                    selected_df = data.getData(selected_keys)
                    logger.info(f"pairplot: selected_keys={len(selected_keys)}")
                    ax = sns.pairplot(selected_df)
                    if args.show:
                        plt.show()
                    if args.save:
                        pfile = f"{file}-pairplot"
                        plt.savefig(f"{pfile}.png", dpi=300)
                    plt.close()

                if args.fields:
                    logger.info(f"stats for {args.fields}")
                    stats(data, args.fields, debug)

            elif args.command == "plot":
                if args.xfield not in data.Keys:
                    logger.warning(
                        f"missing xfield={args.xfield} in {data.Keys} - ignored dataset"
                    )
                else:
                    if data.getDuration() >= min_duration:
                        if args.fields:
                            for key in args.fields:
                                if key not in data.Keys:
                                    logger.warning(
                                        f"missing field={key} - ignored dataset"
                                    )
                                else:
                                    bfield = data.getData(key).to_numpy()
                                    (symbol, unit) = data.getUnitKey(key)
                                    logger.info(
                                        f"{key}[{unit:~P}]: min={bfield.min()}, mean={bfield.mean()}, max={bfield.max()}",
                                    )
                                    data.plotData(args.xfield, key, ax)

                                    # overwrite legend
                                    if key in legends:
                                        legends[key].append(
                                            f"{data.FileName.replace('.txt','')}"
                                        )
                                    else:
                                        legends[key] = [
                                            data.FileName.replace(".txt", "")
                                        ]

                    else:
                        logger.info(
                            f"record: {file} - duration < {min_duration} s, ignored"
                        )

    if args.command == "select":
        if select_rows:
            print(tabulate(select_rows, headers="keys", tablefmt="simple"))
        else:
            print("No records matched selection criteria.")

    if args.command == "plot":
        logger.info(f"plot: {len(legends)} subplots")
        if not legends:
            logger.warning("no field to plot")
        else:
            leg = plt.legend()
            ax.get_legend().set_visible(False)

            if args.show:
                plt.show()
            if args.save:
                pfile = ""
                for field in args.fields:
                    pfile += f"{field}-"
                pfile = pfile[:-1] + "-vs-" + args.xfield

                plt.savefig(f"{pfile}.png", dpi=300)
            plt.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
