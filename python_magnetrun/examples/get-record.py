"""Console script."""

import argparse
import sys
import os
import matplotlib.pyplot as plt

from ..MagnetRun import MagnetRun
from ..magnetdata import MagnetData

from datetime import datetime


def load_record(file: str, args, show: bool = False) -> MagnetData:
    """Load record."""
    # print(f'load_record: {file}')

    filename = os.path.basename(file)
    (housing, timestamp) = filename.split("_")
    site = "blbl"

    mrun = MagnetRun.fromtxt(housing, site, file)
    data = mrun.MagnetData
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
    res = file.split("_")
    """
    if len(res) != 2:
        print(f"{file}: skipped")
    """

    (site, date_string) = res
    date_string = date_string.replace(".txt", "")
    tformat = "%Y.%m.%d---%H:%M:%S"
    timestamp = datetime.strptime(date_string, tformat)
    if debug:
        print(f"{site}: timestamp={timestamp}")
    return timestamp


def main():
    """Console script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile", help="specify inputfile", type=str)
    parser.add_argument("--debug", help="enable debug mode", action="store_true")

    subparsers = parser.add_subparsers(
        title="commands", dest="command", help="sub-command help"
    )
    parser_select = subparsers.add_parser("select", help="select help")
    parser_stats = subparsers.add_parser("stats", help="stats help")
    parser_plot = subparsers.add_parser("plot", help="select help")

    # subcommand select
    parser_select.add_argument(
        "--duration",
        help="select record with a duration more than",
        type=float,
        default="60",
    )
    parser_select.add_argument(
        "--field",
        help="select record with a duration more than",
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

    # subcommand stats
    parser_stats.add_argument("--fields", help="select fields", type=str, nargs="+")

    args = parser.parse_args()

    print(f"getrecords: Arguments={args}, pwd={os.getcwd()}")

    # check if input_file is a string or a list
    input_file = args.inputfile
    files = [input_file]
    if "*" in input_file:
        import glob

        directory = input_file.split("/")
        if args.debug:
            print(f"find records with {input_file}")
            print(f"directory={directory}")

        if len(directory) > 1:
            cwd = os.getcwd()
            for d in directory[:-1]:
                os.chdir(d)

            files = [file for file in glob.glob(directory[-1])]
        else:
            # need to be sorted by time??
            files = [file for file in glob.glob(input_file)]

        # need to be sorted by time??
        files = sorted(files, key=lambda x: getTimestamp(x), reverse=True)
        if args.debug:
            print("sort by time")
            for file in files:
                print(file)

    ax = plt.gca()
    legends = {}
    df_ = []
    selected_keys = [args.xfield] + args.fields
    if "timestamp" not in selected_keys:
        selected_keys.append("timestamp")
    print(f'selected_keys={selected_keys}')
    
    for file in files:
        try:
            data = load_record(file, args)
            if args.command == "select":
                if select_data(data, args):
                    bfield = data.getData("Field")
                    print(
                        f"record: {data.FileName}, duration: {data.getDuration()} s, Field: min={bfield.min()}, mean={bfield.mean()}, max={bfield.max()}"
                    )

            elif args.command == "stats":
                if args.fields:
                    for key in args.fields:
                        bfield = data.getData(args.fields)
                        print(
                            f"record: {data.FileName}, duration: {data.getDuration()} s, Field: min={bfield.min()}, mean={bfield.mean()}, max={bfield.max()}"
                        )

            elif args.command == "plot":
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

                if args.xfield not in data.Keys:
                    print(
                        f"{data.FileName}: missing xfield={args.xfield} - ignored dataset"
                    )
                else:
                    if data.getDuration() >= 60:
                        if args.fields:
                            for key in args.fields:
                                if key not in data.Keys:
                                    print(
                                        f"{data.FileName}: missing field={key} ignored dataset"
                                    )
                                else:
                                    bfield = data.getData(key)
                                    print(
                                        f"{data.FileName}: duration: {data.getDuration()} s, {key}: min={bfield.min()}, mean={bfield.mean()}, max={bfield.max()}",
                                        flush=True,
                                    )
                                    data.plotData(args.xfield, key, ax)

                                    # overwrite legend
                                    if key in legends:
                                        legends[key].append(f"{data.FileName.replace('.txt','')}")
                                    else:
                                        legends[key] = [data.FileName.replace('.txt','')]

                            try:
                                df_.append(data.Data[selected_keys])
                            except:
                                pass
                    else:
                        print(
                            f"{data.FileName}: duration={data.getDuration()} s, ignored dataset"
                        )
        except:
            print(f'fail to load {file}')
            pass

    if args.command == "plot":

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
                plt.savefig("tutu.png", dpi=300)
            plt.close()

        print("plot over time with seaborn", flush=True)
        import pandas as pd
        import statsmodels.api as sm
        import seaborn as sns

        df = pd.concat(df_, axis=0)
        df.to_csv('teb.csv')
        print(f"{df.keys()}")

        # pd.DatetimeIndex(df['InsertedDate']).month

        df["month"] = df["timestamp"].dt.month
        df["year"] = df["timestamp"].dt.year
        print(f"concat df: {df.head()}")

        if args.fields:
            for key in args.fields:
                print(f"seaborn plot for {key} per months over years")
                sns.lineplot(x="month", y=key, hue="year", data=df)
                """
                # filtered over 1995 to make the plot less cluttered
                sns.lineplot(
                    x="month", y=key, hue="year", data=df.query("year>1995")
                )
                """
                plt.show()
                plt.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
