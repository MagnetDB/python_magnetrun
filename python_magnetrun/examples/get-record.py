"""Console script."""

import argparse
import sys
import os
import matplotlib.pyplot as plt

from ..MagnetRun import MagnetRun
from ..magnetdata import MagnetData
from ..processing.stats import stats
from ..processing.correlations import pearson

from datetime import datetime
import pandas as pd


def load_record(file: str, args, show: bool = False) -> MagnetData:
    """Load record."""
    # print(f'load_record: {file}')

    filename = os.path.basename(file)
    (housing, timestamp) = filename.split("_")
    site = "blbl"

    mrun = MagnetRun.fromtxt(housing, site, file)
    data = mrun.MagnetData
    if not isinstance(data, MagnetData):
        raise RuntimeError(f"{file}: cannot load data as MagnetData")

    return data


def select_data(data, args) -> bool:
    duration = data.getDuration()
    if duration > args.duration:
        bdata = data.extractDataThreshold("Field", args.field)
        if not bdata.empty:
            bfield = data.getData("Field")
            print(
                f"record: {data.FileName}, duration: {data.getDuration()} s, Field: min={bfield.min()}, mean={bfield.mean()}, max={bfield.max()}"
            )
            return True
    return False


"""
    # print(f"stats: {data.stats('Field')}")

        if show:
            ax = plt.gca()
            data.plotData("t", "Field", ax)
            plt.title(f"{file}: Magnet Field")
            plt.show()

            ax = plt.gca()
            data.plotData("t", "IH", ax)
            data.plotData("t", "IB", ax)
            plt.title(f"{file}: current")
            plt.show()

    return data
    # print(f"stats: {mrun.getStats()}")
"""


def getTimestamp(file: str, debug: bool = False) -> datetime:
    """
    extract timestamp from file
    """
    # print(f"getTime({file}):", flush=True)

    filename = ""
    if "/" in file:
        filename = file.split("/")
    res = filename[-1].split("_")
    if debug:
        print(f"getTime({file})={res}", flush=True)

    try:
        (site, date_string) = res
        date_string = date_string.replace(".txt", "")
        tformat = "%Y.%m.%d---%H:%M:%S"
        timestamp = datetime.strptime(date_string, tformat)
        if debug:
            print(f"{site}: timestamp={timestamp}")
    except:
        raise RuntimeError(f"getTimestamp: {file} failed -unexpected filename")

    return timestamp


def main():
    """Console script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile", help="specify inputfile", nargs="+")
    parser.add_argument("--debug", help="enable debug mode", action="store_true")

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

    args = parser.parse_args()

    print(f"getrecords: Arguments={args}, pwd={os.getcwd()}")

    # check if input_file is a string or a list
    files = args.inputfile

    # need to be sorted by time??
    files = sorted(files, key=lambda x: getTimestamp(x, args.debug), reverse=False)
    if args.debug:
        print(f"sort by time: {files}")

    selected_keys = []
    if args.command == "plot":
        if args.xfield:
            selected_keys += [args.xfield]
        if args.fields:
            print(f"args.fields={args.fields}")
            selected_keys += args.fields
    elif args.command == "aggregate":
        selected_keys += args.fields
    else:
        selected_keys += ["Field"]
    print(f"selected_keys={selected_keys}", flush=True)

    min_duration = 60
    if args.command == "select":
        min_duration = args.duration
    elif args.command == "stats":
        print(f"!!! OverWrite min duration for stats: {min_duration} -> 1000 !!!")
        min_duration = 1000

    if "timestamp" not in selected_keys:
        selected_keys.append("timestamp")
    print(f"selected_keys={selected_keys}", flush=True)

    """
    https://stackoverflow.com/questions/57601552/how-to-plot-timeseries-using-pandas-with-monthly-groupby
    https://gist.github.com/vincentarelbundock/3485014

    need to concat magnetdata
    build 'month' and 'year' column in resulting dataframe

    import pandas as pd
    import statsmodels.api as sm
    import seaborn as sns

    df = sm.datasets.co2.load(as_pandas=True).data
    df['month'] = pd.to_datetime(df.index).month
    df['year'] = pd.to_datetime(df.index).year
    sns.lineplot(x='month',y='co2',hue='year',data=df.query('year>1995')) # filtered over 1995 to make the plot less cluttered
    """

    ax = plt.gca()

    if args.command == "aggregate":
        print(f"aggregate: fields={selected_keys}")

        df_ = []

        for file in files:
            try:
                print(
                    f"record: {file}",
                    end=" ",
                    flush=True,
                )
                data = load_record(file, args)
                print(
                    f", duration: {data.getDuration()} s",
                    end=" ",
                    flush=True,
                )
                if data.getDuration() >= min_duration:
                    try:
                        df_.append(data.Data[selected_keys])
                        print(f"- extract {selected_keys}", flush=True)
                    except Exception as error:
                        print(
                            f"- ignored dataset: {selected_keys} not all in {data.getKeys()} (error={error})"
                        )
                        pass
                else:
                    print("- skipped", flush=True)

            except Exception as error:
                print(f"- fail to load (error={error})", flush=True)
                pass

        print(f"plot over time with seaborn: {len(df_)} dataframes", flush=True)

        df = pd.concat(df_, axis=0)
        output = f"aggregate-{'-'.join(args.fields)}.csv"
        df.to_csv(output)
        print(f"concat dataframe: {df.head()}", flush=True)
        print(f"{df.columns.values.tolist()} to {os.getcwd()}/{output}", flush=True)

        # pd.DatetimeIndex(df['InsertedDate']).month

        df["month"] = df["timestamp"].dt.month
        df["year"] = df["timestamp"].dt.year
        print(f"concat df: {df.head()}")

        if args.fields:
            import seaborn as sns

            for key in args.fields:
                print(f"seaborn plot for {key} per months over years", flush=True)
                ax = sns.lineplot(x="month", y=key, hue="year", data=df)

                (symbol, unit) = data.getUnitKey(key)
                ax.set_ylabel(f"{symbol} [{unit:~P}]")
                ax.set_title(f"{file}: {key}")
                """
                # filtered over 1995 to make the plot less cluttered
                sns.lineplot(
                    x="month", y=key, hue="year", data=df.query("year>1995")
                )
                """
                plt.grid()
                if args.show:
                    plt.show()
                if args.save:
                    print(
                        f"seaborn plot for {key} per months over years saved to {os.getcwd()}/{key}-seaborn.png",
                        flush=True,
                    )
                    plt.savefig(f"{key}-seaborn.png", dpi=300)
                plt.close()

        return 0

    # other commands
    legends = {}

    for file in files:
        # print(f'file={file}', flush=True)
        try:
            print(
                f"record: {file}",
                end=" ",
                flush=True,
            )
            data = load_record(file, args)
            print(
                f", duration: {data.getDuration()} s",
                end=" ",
                flush=True,
            )
        except:
            print("- fail to load")
            # is it possible to curate txt files
            # reading csv line by line with import csv??

        else:
            data.Units()
            if args.command == "select":
                if select_data(data, args):
                    bfield = data.getData("Field")
                    print(
                        f"- Field: min={bfield.min()}, mean={bfield.mean()}, max={bfield.max()}",
                        flush=True,
                    )

            elif args.command == "stats":
                if (
                    args.pearson and data.getDuration() >= min_duration
                ):  # previous limit 1000:
                    pearson(data, args.fields, args.save, args.show, args.debug)
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
                    print(
                        f"pairplot: selected_keys={len(selected_keys)}",
                        flush=True,
                    )
                    ax = sns.pairplot(selected_df)
                    if args.show:
                        plt.show()
                    if args.save:
                        pfile = f"{file}-pairplot"
                        plt.savefig(f"{pfile}.png", dpi=300)
                    plt.close()

                if args.fields:
                    # save to tabular
                    print(f"stats for {args.fields}")
                    # print(data.getData(args.fields).head(20))

                    stats(data, args.fields, args.debug)

            elif args.command == "plot":
                if args.xfield not in data.Keys:
                    print(
                        f"- missing xfield={args.xfield} in {data.Keys}- ignored dataset"
                    )
                else:
                    if data.getDuration() >= min_duration:
                        if args.fields:
                            for key in args.fields:
                                if key not in data.Keys:
                                    print(f"\t- missing field={key} ignored dataset")
                                else:
                                    bfield = data.getData(key).to_numpy()
                                    (symbol, unit) = data.getUnitKey(key)
                                    print(
                                        f"- {key}[{unit:~P}]: min={bfield.min()}, mean={bfield.mean()}, max={bfield.max()}",
                                        flush=True,
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
                        print(f"duration < {min_duration} - ignored dataset")

    if args.command == "plot":
        print(f"plot: {len(legends)} subplots", flush=True)
        if not legends:
            print("no field to plot")
        else:
            # if len(legends) < 10:
            #    ax.legend(legends)

            # if legend
            leg = plt.legend()
            # ax.get_legend().remove()
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
