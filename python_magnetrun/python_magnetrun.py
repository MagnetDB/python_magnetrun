"""Main module."""

import math
import os
import sys
import datetime
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available
from .magnetdata import MagnetData


from .MagnetRun import MagnetRun

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--site", help="specify a site (ex. M8, M9,...)", default="M9")
    parser.add_argument("--plot_vs_time", help="select key(s) to plot (ex. \"Field[;Ucoil1]\")")
    parser.add_argument("--plot_key_vs_key", help="select pair(s) of keys to plot (ex. \"Field-Icoil1")
    parser.add_argument("--output_time", help="output key(s) for time")
    parser.add_argument("--output_timerange", help="set time range to extract (start;end)")
    parser.add_argument("--output_key", help="output key(s) for time")
    parser.add_argument("--extract_pairkeys", help="dump key(s) to file")
    parser.add_argument("--show", help="display graphs (requires X11 server active)", action='store_true')
    parser.add_argument("--save", help="save graphs (png format)", action='store_true')
    parser.add_argument("--list", help="list key in csv", action='store_true')
    parser.add_argument("--convert", help="convert file to csv", action='store_true')
    parser.add_argument("--stats", help="display stats and find regimes", action='store_true')
    parser.add_argument("--thresold", help="specify thresold for regime detection", type=float, default=1.e-3)
    parser.add_argument("--bthresold", help="specify b thresold for regime detection", type=float, default=1.e-3)
    parser.add_argument("--dthresold", help="specify duration thresold for regime detection", type=float, default=10)
    parser.add_argument("--window", help="stopping criteria for nlopt", type=int, default=10)
    parser.add_argument("--debug", help="acticate debug", action='store_true')
    args = parser.parse_args()

    # load df pandas from input_file
    # check extension
    f_extension=os.path.splitext(args.input_file)[-1]
    if f_extension != ".txt":
        raise RuntimeError("so far only txt file support is implemented")

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
    mrun = MagnetRun.fromtxt(args.site, args.input_file)
    dkeys = mrun.getKeys()

    if args.list:
        print("Valid keys are:")
        for key in dkeys:
            print(key)
        sys.exit(0)

    if args.convert:
        extension = os.path.splitext(args.input_file)[-1]
        file_name = args.input_file.replace(extension, ".csv")
        mrun.getData().to_csv(file_name, sep=str('\t'), index=False, header=True)
        sys.exit(0)

    # perform operations defined by options
    if args.plot_vs_time:
        my_ax = plt.gca()
        # split into keys
        items = args.plot_vs_time.split(';')
        print("items=", items)
        # loop over key
        for key in items:
            print("plot key=", key, "type=", type(key))
            mrun.getMData().plotData(x='Time', y=key, ax=my_ax)
        if args.show:
            plt.show()
        else:
            imagefile = args.input_file.replace(".txt", "")
            plt.savefig('%s_vs_time.png' % imagefile, dpi=300 )
            plt.close()

    if args.plot_key_vs_key:
        # split pairs in key1, key2
        print("plot_key_vs_key=", args.plot_key_vs_key)
        pairs = args.plot_key_vs_key.split(';')
        for pair in pairs:
            print("pair=", pair)
            my_ax = plt.gca()
            #print("pair=", pair, " type=", type(pair))
            items = pair.split('-')
            if len(items) != 2:
                raise RuntimeError("invalid pair of keys: %s" % pair)
            key1= items[0]
            key2 =items[1]
            if key1 in dkeys and key2 in dkeys:
                mrun.getMData().plotData(x=key1, y=key2, ax=my_ax) # on graph per pair
            else:
                raise Exception("unknown keys: %s %s" % (key1, key2), " (Check valid keys with --list option)")
            if args.show:
                plt.show()
            else:
                imagefile = args.input_file.replace(".txt", "")
                plt.savefig('%s_%s_vs_%s.png' % (imagefile, key1, key2), dpi=300 )
                plt.close()

    if args.output_time:
        if mrun.getType() != 0:
            raise RuntimeError("output_time: feature not implemented for tdms format")

        times = args.output_time.split(";")
        print ("Select data at %s " % (times) )
        df = mrun.getData()
        if args.output_key:
            keys = args.output_key.split(";")
            print(df[df['Time'].isin(times)][keys])
        else:
            print(df[df['Time'].isin(times)])

    if args.output_timerange:
        if mrun.getType() != 0:
            raise RuntimError("output_time: feature not implemented for tdms format")

        timerange = args.output_timerange.split(";")

        file_name = args.input_file.replace(".txt", "")
        file_name = file_name + "_from" + str(timerange[0].replace(":", "-"))
        file_name = file_name + "_to" + str(timerange[1].replace(":", "-")) + ".csv"
        selected_df = mrun.getMData().extractTimeData(timerange)
        selected_df.to_csv(file_name, sep=str('\t'), index=False, header=True)

    if args.output_key:
        if mrun.getType() != 0:
            raise RuntimError("output_time: feature not implemented for tdms format")

        keys = args.output_key.split(";")
        keys.insert(0, 'Time')

        file_name = args.input_file.replace(".txt", "")
        for key in keys:
            if key != 'Time':
                file_name = file_name + "_" + key
        file_name = file_name + "_vs_Time.csv"

        selected_df = mrun.getMData().extractData(keys)
        selected_df.to_csv(file_name, sep=str('\t'), index=False, header=True)

    if args.extract_pairkeys:
        if mrun.getType():
            raise RuntimError("output_time: feature not implemented for tdms format")

        pairs = args.extract_pairkeys.split(';')
        for pair in pairs:
            items = pair.split('-')
            if len(items) != 2:
                raise RuntimeError("invalid pair of keys: %s" % pair)
            key1= items[0]
            key2 =items[1]
            newdf = mrun.getMData().extractData([key1, key2])

            # Remove line with I=0
            newdf = newdf[newdf[key1] != 0]
            newdf = newdf[newdf[key2] != 0]

            file_name=str(pair)+".csv"
            newdf.to_csv(file_name, sep=str('\t'), index=False, header=False)

    if args.stats:
        mrun.stats()
        mrun.plateaus(twindows=args.window,
                      thresold=args.thresold,
                      bthresold=args.bthresold,
                      duration=args.dthresold,
                      show=args.show,
                      save=args.save,
                      debug=args.debug)
