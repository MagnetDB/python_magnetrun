"""Main module."""

import os
import sys
import traceback

# import logging

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available


from .MagnetRun import MagnetRun

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="+", help="enter input file")
    parser.add_argument(
        "--site", help="specify a site (ex. M8, M9,...)", default="M9"
    )  # use housing instead
    parser.add_argument("--insert", help="specify an insert", default="notdefined")
    parser.add_argument("--debug", help="acticate debug", action="store_true")

    subparsers = parser.add_subparsers(
        title="commands", dest="command", help="sub-command help"
    )
    parser_info = subparsers.add_parser("info", help="info help")
    parser_select = subparsers.add_parser("select", help="select help")
    parser_stats = subparsers.add_parser("stats", help="stats help")
    parser_plot = subparsers.add_parser("plot", help="select help")

    # add info subcommand
    parser_info.add_argument("--list", help="list key in csv", action="store_true")
    parser_info.add_argument("--convert", help="save to csv", action="store_true")

    # add plot subcommand
    parser_plot.add_argument(
        "--vs_time",
        help='select key(s) to plot (ex. "Field [Ucoil1]")',
        nargs="+",
        action="append",
    )
    parser_plot.add_argument(
        "--key_vs_key",
        help='select pair(s) of keys to plot (ex. "Field-Icoil1")',
        nargs="+",
        action="append",
    )
    parser_plot.add_argument(
        "--save", help="save graphs (png format)", action="store_true"
    )

    # extract subcommand
    parser_select.add_argument(
        "--output_time", nargs="+", help="output key(s) for time"
    )
    parser_select.add_argument(
        "--output_timerange",
        help="set time range to extract (start;end)",
        action="append",
    )
    parser_select.add_argument(
        "--output_key",
        nargs="+",
        help="output key(s) for time",
        action="append",
    )
    parser_select.add_argument(
        "--extract_pairkeys",
        nargs="+",
        help="dump key(s) to file",
        action="append",
    )
    parser_select.add_argument(
        "--convert", help="convert file to csv", action="store_true"
    )

    # add stats subcommand
    parser_stats.add_argument(
        "--save", help="save graphs (png format)", action="store_true"
    )
    parser_stats.add_argument(
        "--show", help="display graphs (require X11)", action="store_true"
    )
    parser_stats.add_argument(
        "--keys",
        help="select key(s) to perform selected stats",
        nargs="+",
    )
    parser_stats.add_argument(
        "--threshold",
        help="specify threshold for regime detection",
        type=float,
        default=1.0e-3,
    )
    parser_stats.add_argument(
        "--bthreshold",
        help="specify b threshold for regime detection",
        type=float,
        default=1.0e-3,
    )
    parser_stats.add_argument(
        "--dthreshold",
        help="specify duration threshold for regime detection",
        type=float,
        default=10,
    )
    parser_stats.add_argument(
        "--window", help="stopping criteria for nlopt", type=int, default=10
    )
    args = parser.parse_args()
    print(f"args: {args}", flush=True)

    # load df pandas from input_file
    # check extension
    supported_formats = [".txt", ".tdms", ".csv"]

    inputs = {}
    extensions = {}
    for i, file in enumerate(args.input_file):
        f_extension = os.path.splitext(file)[-1]
        if f_extension not in extensions:
            extensions[f_extension] = [i]
        else:
            extensions[f_extension].append(i)
    # print(f"extensions: {extensions}")

    for file in args.input_file:
        f_extension = os.path.splitext(file)[-1]
        if f_extension not in supported_formats:
            raise RuntimeError(
                f"so far file with extension in {supported_formats} are implemented"
            )

        filename = os.path.basename(file)
        result = filename.startswith("M")
        insert = "tututu"
        if result:
            try:
                index = filename.index("_")
                site = filename[0:index]
                print(f"site detected: {site}")
            except:
                print("no site detected - use args.site argument instead")
                pass

        match f_extension:
            case ".txt":
                mrun = MagnetRun.fromtxt(site, insert, file)
            case ".tdms":
                mrun = MagnetRun.fromtdms(site, insert, file)
            case ".csv":
                mrun = MagnetRun.fromcsv(site, insert, file)
            case _:
                raise RuntimeError(
                    f"so far file with extension in {supported_formats} are implemented"
            )

        inputs[file] = {"data": mrun}

        if args.command == "info":
            mrun.getMData().info()
            if args.list:
                print(f"{file}: valid keys")
                for key in mrun.getKeys():
                    print('\t', key)

            if args.convert:
                mdata = mrun.getMData()
                data = mdata.Data
                if mdata.Type == 0:
                    csvfile = file.replace(f_extension, ".csv")
                    data.to_csv(csvfile, sep=str("\t"), index=True, header=True)
                elif mdata.Type == 1:
                    for key, df in data.items():
                        print(f'convert: key={key}', flush=True)
                        csvfile = file.replace(f_extension, f"-{key}.csv")
                        df.to_csv(csvfile, sep=str("\t"), index=True, header=True)


    # perform operations defined by options
    if args.command == "plot":

        if args.vs_time:
            assert len(args.vs_time) == len(
                extensions
            ), f"expected {len(extensions)} vs_time arguments - got {len(args.vs_time)} "

            my_ax = plt.gca()

            items = args.vs_time
            # print(f"items={items}", flush=True)
            for file in args.input_file:
                f_extension = os.path.splitext(file)[-1]
                plot_args = items[extensions[f_extension][0]]
                mrun = inputs[file]["data"]
                for key in plot_args:
                    print(f"plot key={key}, type={type(key)}", flush=True)
                    mrun.getMData().plotData(x="t", y=key, ax=my_ax)
            if not args.save:
                plt.show()
            else:
                imagefile = "image"
                print(f"saveto: {imagefile}_vs_time.png", flush=True)
                plt.savefig(f"{imagefile}_vs_time.png", dpi=300)
            plt.close()

        if args.key_vs_key:
            assert len(args.key_vs_key) == len(
                extensions
            ), f"expected {len(extensions)} key_vs_key arguments - got {len(args.key_vs_key)} "

            my_ax = plt.gca()

            # split pairs in key1, key2
            print(f"key_vs_key={args.key_vs_key}")
            pairs = args.key_vs_key
            for file in args.input_file:
                f_extension = os.path.splitext(file)[-1]
                plot_args = pairs[extensions[f_extension][0]]
                mrun = inputs[file]["data"]

                for pair in plot_args:
                    print(f"pair={pair}")
                    # print("pair=", pair, " type=", type(pair))
                    items = pair.split("-")
                    if len(items) != 2:
                        raise RuntimeError(f"invalid pair of keys:{pair}")
                    key1 = items[0]
                    key2 = items[1]
                    mrun.plotData(x=key1, y=key2, ax=my_ax)

            if not args.save:
                plt.show()
            else:
                imagefile = "image"
                print(f"saveto: {imagefile}_key_vs_key.png", flush=True)
            plt.savefig(f"{imagefile}_key_vs_key.png", dpi=300)
            plt.close()

    if args.command == "select":
        # plot_args = items[extensions[f_extension][0]]
        if args.output_time:
            assert len(args.output_time) == len(
                extensions
            ), f"expected {len(extensions)} output_time arguments - got {len(args.output_time)} "

            times = args.output_time.split(";")
            print(f"Select data at {times}")
            for file in inputs:
                f_extension = os.path.splitext(file)[-1]
                mrun = inputs[file]["data"]
                mdata = mrun.getMData()
                select_args = times[extensions[f_extension][0]]
                select_args_str = "_at"
                for item in select_args:
                    select_args_str += f"-{item:.3f}s"

                if mdata.Type == 0:
                    data = mdata.Data
                    df = data[data["timestamp"].isin(times)]
                    file_name = file.replace(f_extension, "")
                    file_name = file_name + select_args_str + ".csv"
                    df.to_csv()

                elif mdata.Type == 1:
                    data = mdata.Data
                    for group in data:
                        df = data[data.index.isin(times)]
                        file_name = file.replace(f_extension, f"-{group}")
                        file_name = file_name + select_args_str + ".csv"
                        df.to_csv()

        if args.output_timerange:
            assert len(args.output_timerange) == len(
                extensions
            ), f"expected {len(extensions)} output_timerange arguments - got {len(args.output_timerange)} "

            timerange = args.output_timerange.split(";")
            for file in inputs:
                f_extension = os.path.splitext(file)[-1]
                mrun = inputs[file]["data"]
                mdata = mrun.getMData()
                select_args = args.output_timerange[extensions[f_extension][0]]
                for item in select_args:
                    timerange = item.split(";")

                    file_name = file.replace(f_extension, "")
                    file_name = (
                        file_name + "_from" + str(timerange[0].replace(":", "-"))
                    )
                    file_name = (
                        file_name + "_to" + str(timerange[1].replace(":", "-")) + ".csv"
                    )

                    if mdata.Type == 0:
                        selected_df = mdata.extractTimeData(timerange)
                        if not selected_df is None:
                            selected_df.to_csv(
                                file_name, sep=str("\t"), index=False, header=True
                            )
                    elif mdata.Type == 1:
                        for group in mdata.Data:
                            selected_df = mdata.extractTimeData(timerange, group)
                            if not selected_df is None:
                                selected_df.to_csv(
                                    file_name, sep=str("\t"), index=False, header=True
                                )

        if args.output_key:
            assert len(args.output_key) == len(
                extensions
            ), f"expected {len(extensions)} output_key arguments - got {len(args.output_key)} "

            for file in inputs:
                f_extension = os.path.splitext(file)[-1]
                mrun = inputs[file]["data"]
                mdata = mrun.getMData()
                select_args = args.output_key[extensions[f_extension][0]]
                for item in select_args:
                    keys = item.split(";")
                    keys.insert(0, "Time")

                    file_name = file.replace(".txt", "")
                    for key in keys:
                        if key != "Time":
                            file_name = file_name + "_" + key
                    file_name = file_name + "_vs_Time.csv"

                    selected_df = mdata.extractData(keys)
                    if not selected_df is None:
                        selected_df.to_csv(
                            file_name, sep=str("\t"), index=False, header=True
                        )

        if args.extract_pairkeys:
            assert len(args.extract_pairkeys) == len(
                extensions
            ), f"expected {len(extensions)} extract_pairkeys arguments - got {len(args.extract_pairkeys)} "
            for file in inputs:
                f_extension = os.path.splitext(file)[-1]
                mrun = inputs[file]["data"]
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
                        if not mdata is None:
                            newdf = mdata.extractData([key1, key2])
                            if not newdf is None:
                                # Remove line with I=0
                                newdf = newdf[newdf[key1] != 0]
                                newdf = newdf[newdf[key2] != 0]

                                file_name = str(pair) + ".csv"
                                newdf.to_csv(
                                    file_name, sep=str("\t"), index=False, header=False
                                )

        if args.convert:
            for file in inputs:
                mrun = inputs[file]["data"]
                mdata = mrun.getMData()

                extension = os.path.splitext(file)[-1]
                file_name = args.input_file.replace(extension, ".csv")
                if mdata.Type == 0:
                    mdata.to_csv(file_name, sep=str("\t"), index=False, header=True)

    if args.command == "stats":
        from .processing import stats
        from .utils.plateaux import plateaus, nplateaus

        print("Stats:")
        for file in inputs:
            print(file)
            extension = os.path.splitext(file)[-1]
            mrun = inputs[file]["data"]
            mdata = mrun.getMData()

            stats.stats(mdata)

            try:
                if mdata.Type == 0:
                    if args.keys:
                        for key in args.keys:
                            print(f"pupitre: stats for {key}", flush=True)
                            symbol = mdata.getUnitKey(key)[0]
                            plateaus(
                                mdata,
                                yField=(key, symbol),
                                twindows=args.window,
                                threshold=args.threshold,
                                b_threshold=args.bthreshold,
                                duration=args.dthreshold,
                                show=args.show,
                                save=args.save,
                                debug=args.debug,
                            )

                            pdata = nplateaus(
                                mdata,
                                xField=("t", "s"),
                                yField=(key, symbol),
                                threshold=args.threshold,
                                num_points_threshold=600,
                                save=args.save,
                                show=args.show,
                                verbose=False,
                            )
                            for plateau in pdata:
                                if plateau["value"] > 1.0:
                                    print(f"{plateau}", flush=True)

                elif mdata.Type == 1:
                    if args.keys:
                        for key in args.keys:
                            print(f"pigbrother: stats for {key}", flush=True)
                            symbol = mdata.getUnitKey(key)[0]
                            plateaus(
                                mdata,
                                yField=(key, symbol),
                                twindows=args.window,
                                threshold=args.threshold,
                                b_threshold=args.bthreshold,
                                duration=args.dthreshold,
                                show=args.show,
                                save=args.save,
                                debug=args.debug,
                            )
                            pdata = nplateaus(
                                mdata,
                                xField=("t", "s"),
                                yField=(key, symbol),
                                threshold=args.threshold,
                                num_points_threshold=600,
                                save=args.save,
                                show=args.show,
                                verbose=False,
                            )
                            for plateau in pdata:
                                if plateau["value"] > 1.0:
                                    print(f"{plateau}", flush=True)
            except Exception as e:
                print(traceback.format_exc())
                pass
