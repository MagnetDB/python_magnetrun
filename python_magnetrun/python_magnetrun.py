"""Main module."""

import os
import sys
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
    parser.add_argument("input_file") # can be a list
    parser.add_argument("--site", help="specify a site (ex. M8, M9,...)", default="M9") # use housing instead
    parser.add_argument("--insert", help="specify an insert", default="notdefined")
    parser.add_argument("--plot_vs_time", help="select key(s) to plot (ex. \"Field [Ucoil1]\")", nargs="+") # use nargs instead
    parser.add_argument("--plot_key_vs_key", help="select pair(s) of keys to plot (ex. \"Field-Icoil1")
    parser.add_argument("--output_time", help="output key(s) for time")
    parser.add_argument(
        "--output_timerange", help="set time range to extract (start;end)"
    )
    parser.add_argument("--output_key", help="output key(s) for time")
    parser.add_argument("--extract_pairkeys", help="dump key(s) to file")
    parser.add_argument(
        "--show",
        help="display graphs (requires X11 server active)",
        action="store_true",
    )
    parser.add_argument("--save", help="save graphs (png format)", action="store_true")
    parser.add_argument("--list", help="list key in csv", action="store_true")
    parser.add_argument("--convert", help="convert file to csv", action="store_true")
    parser.add_argument(
        "--stats", help="display stats and find regimes", action="store_true"
    )
    parser.add_argument(
        "--thresold",
        help="specify thresold for regime detection",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--bthresold",
        help="specify b thresold for regime detection",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--dthresold",
        help="specify duration thresold for regime detection",
        type=float,
        default=10,
    )
    parser.add_argument(
        "--window", help="stopping criteria for nlopt", type=int, default=10
    )
    parser.add_argument("--debug", help="acticate debug", action="store_true")
    args = parser.parse_args()

    # load df pandas from input_file
    # check extension
    supported_formats = [".txt", ".tdms", ".csv"]
    f_extension=os.path.splitext(args.input_file)[-1]
    file_name, file_extension = os.path.splitext(args.input_file)
    print(f'file_name={file_name}, file_extension={file_extension}', flush=True)
    #if f_extension not in supported_formats:
    #    raise RuntimeError(f"so far file with extension in {supported_formats} are implemented")

    filename = os.path.basename(args.input_file)
    result = filename.startswith("M")
    if result:
        try:
            index = filename.index("_")
            args.site = filename[0:index]
            print("site detected: %s" % args.site)
        except:
            print("no site detected - use args.site argument instead")
            pass

    match file_extension:
        case ".txt":
            mrun = MagnetRun.fromtxt(args.site, args.insert, args.input_file)
        case '.tdms':
            mrun = MagnetRun.fromtdms(args.site, args.insert, args.input_file)
        case '.csv':
            mrun = MagnetRun.fromcsv(args.site, args.insert, args.input_file)
        case _:
            raise RuntimeError(f"so far file with extension in {supported_formats} are implemented")        

    dkeys = mrun.getKeys()

    if args.list:
        print("Valid keys are:")
        for key in dkeys:
            print(key)
        sys.exit(0)

    if args.convert:
        extension = os.path.splitext(args.input_file)[-1]
        file_name = args.input_file.replace(extension, ".csv")
        data = mrun.getMData()
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_name, sep=str("\t"), index=False, header=True)
        sys.exit(0)

    # perform operations defined by options
    if args.plot_vs_time:
        my_ax = plt.gca()
        # split into keys
        items = args.plot_vs_time
        print(f"items={items}", flush=True)
        # loop over key
        for key in items:
            print(f"plot key={key}, type={type(key)}")
            mrun.getMData().plotData(x='t', y=key, ax=my_ax)
        if args.show:
            plt.show()
        else:
            imagefile = args.input_file.replace(".txt", "")
            print(f"saveto: {imagefile}", flush=True)
            plt.savefig(f'{imagefile}_vs_time.png', dpi=300 )
        plt.close()

    if args.plot_key_vs_key:
        # split pairs in key1, key2
        print(f"plot_key_vs_key={args.plot_key_vs_key}")
        pairs = args.plot_key_vs_key.split(';')
        for pair in pairs:
            print(f"pair={pair}")
            my_ax = plt.gca()
            # print("pair=", pair, " type=", type(pair))
            items = pair.split("-")
            if len(items) != 2:
                raise RuntimeError(f"invalid pair of keys:{pair}")
            key1= items[0]
            key2 =items[1]
            if key1 in dkeys and key2 in dkeys:
                data = mrun.getMData()
                if isinstance(data, pd.DataFrame):
                    data.getMData().plotData(
                        x=key1, y=key2, ax=my_ax
                    )  # on graph per pair
            else:
                raise Exception(f"unknown keys: {key1} {key2} (Check valid keys with --list option)")
            if args.show:
                plt.show()
            else:
                imagefile = args.input_file.replace(".txt", "")
                plt.savefig(f'{imagefile}_{key1}_vs_{key2}.png', dpi=300 )
            plt.close()

    if args.output_time:
        if mrun.getType() != 0:
            raise RuntimeError("output_time: feature not implemented for tdms format")

        times = args.output_time.split(";")
        print (f"Select data at {times}" )
        df = mrun.getData()
        if args.output_key:
            keys = args.output_key.split(";")
            print(df[df['Time'].isin(times)][keys])
        else:
            print(df[df['Time'].isin(times)])

    if args.output_timerange:
        if mrun.getType() != 0:
            raise RuntimeError("output_time: feature not implemented for tdms/csv format")

        timerange = args.output_timerange.split(";")

        file_name = args.input_file.replace(".txt", "")
        file_name = file_name + "_from" + str(timerange[0].replace(":", "-"))
        file_name = file_name + "_to" + str(timerange[1].replace(":", "-")) + ".csv"
        data = mrun.getMData()
        if not data is None:
            selected_df = data.extractTimeData(timerange)
            if not selected_df is None:
                selected_df.to_csv(file_name, sep=str("\t"), index=False, header=True)

    if args.output_key:
        if mrun.getType() != 0:
            raise RuntimeError("output_time: feature not implemented for tdms format")
            raise RuntimeError("output_time: feature not implemented for tdms format")

        keys = args.output_key.split(";")
        keys.insert(0, "Time")

        file_name = args.input_file.replace(".txt", "")
        for key in keys:
            if key != "Time":
                file_name = file_name + "_" + key
        file_name = file_name + "_vs_Time.csv"

        data = mrun.getMData()
        if not data is None:
            selected_df = data.extractData(keys)
            if not selected_df is None:
                selected_df.to_csv(file_name, sep=str("\t"), index=False, header=True)

    if args.extract_pairkeys:
        if mrun.getType():
            raise RuntimeError("output_time: feature not implemented for tdms format")
            raise RuntimeError("output_time: feature not implemented for tdms format")

        pairs = args.extract_pairkeys.split(";")
        for pair in pairs:
            items = pair.split("-")
            if len(items) != 2:
                raise RuntimeError(f"invalid pair of keys: {pair}")
            key1 = items[0]
            key2 = items[1]
            data = mrun.getMData()
            if not data is None:
                newdf = data.extractData([key1, key2])
                if not newdf is None:
                    # Remove line with I=0
                    newdf = newdf[newdf[key1] != 0]
                    newdf = newdf[newdf[key2] != 0]

                    file_name = str(pair) + ".csv"
                    newdf.to_csv(file_name, sep=str("\t"), index=False, header=False)

    if args.stats:
        from .processing import stats

        data = mrun.getMData()
        if not data is None:
            stats.stats(data)
            stats.plateaus(
                data,
                twindows=args.window,
                threshold=args.thresold,
                b_threshold=args.bthresold,
                duration=args.dthresold,
                show=args.show,
                save=args.save,
                debug=args.debug,
            )
