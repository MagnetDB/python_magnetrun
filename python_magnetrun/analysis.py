"""Main module."""

import os
import datetime
from natsort import natsorted
import numpy as np

from tabulate import tabulate

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten

matplotlib.rcParams["text.usetex"] = True

from .MagnetRun import MagnetRun
# from .utils.plateaux import nplateaus
from .processing.trends import trends

def convert_to_timestamp(date_str, time_str):
    from datetime import datetime
    # Format the date and time strings
    date_format = "%y%m%d"
    time_format = "%H%M%S"

    # Parse the date and time strings into a datetime object
    date_time_str = date_str + time_str
    date_time_format = date_format + time_format
    date_time_obj = datetime.strptime(date_time_str, date_time_format)

    # Convert the datetime object to a timestamp
    timestamp = date_time_obj.timestamp()

    # Format the datetime object to the desired string format
    formatted_date_time = date_time_obj.strftime("%Y-%m-%d %H:%M:%S")
    
    return (timestamp, formatted_date_time)

def lag_correlation(series, trend_component, start_index: int=0, end_index: int | None = None, show: bool=False, save: bool=False): 
    from scipy.signal import correlate

    # Select a slice of the time series
    if start_index != 0:
        if end_index is not None:
            time_series_slice = series[start_index:end_index]
            trend_slice = trend_component[start_index:end_index]
        else:
            time_series_slice = series[start_index:]
            trend_slice = trend_component[start_index:]
    else:
        if end_index is not None:
            time_series_slice = series[:end_index]
            trend_slice = trend_component[:end_index]
        else:
            time_series_slice = series
            trend_slice = trend_component    

    # Drop NaN values from trend slice
    trend_slice = trend_slice[~np.isnan(trend_slice)]
    time_series_slice = time_series_slice[-len(trend_slice):]

    # Compute cross-correlation
    correlation = correlate(time_series_slice - np.mean(time_series_slice), trend_slice - np.mean(trend_slice))
    lags = np.arange(-len(correlation)//2 + 1, len(correlation)//2 + 1)

    # Find the lag with maximum correlation
    lag = lags[np.argmax(correlation)]

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_series_slice, label='Time Series Slice')
    plt.plot(trend_slice, label='Trend Slice', color='red')
    plt.title('Time Series Slice and Trend Slice')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(lags, correlation, label='Cross-Correlation')
    plt.axvline(lag, color='red', linestyle='--', label=f'Lag = {lag}')
    plt.title('Cross-Correlation between Time Series Slice and Trend Slice')
    plt.xlabel('Lag')
    plt.ylabel('Cross-Correlation')
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig("lag_correlation.png", dpi=300)       
    if show:
        plt.show()
    plt.close()

    print(f"Estimated lag: {lag} for {start_index} to {end_index}") # what unit for lag???
    return lag  

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="+", help="enter input file tdms")
    parser.add_argument(
        "--pupitre_datadir",
        help="enter pupitre datadir (default srvdata)",
        type=str,
        default="srvdata",
    )
    parser.add_argument(
        "--key",
        help="choose key",
        choices=["Référence_GR1", "Référence_GR2"],
        type=str,
        default="Référence_GR1",
    )
    parser.add_argument("--debug", help="acticate debug", action="store_true")

    parser.add_argument("--save", help="save graphs (png format)", action="store_true")
    parser.add_argument(
        "--show", help="display graphs (X11 required)", action="store_true"
    )
    parser.add_argument(
        "--threshold",
        help="specify threshold for regime detection",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--dthreshold",
        help="specify duration threshold for regime detection",
        type=float,
        default=10,
    )
    parser.add_argument(
        "--window", help="stopping criteria for nlopt", type=int, default=10
    )
    parser.add_argument("--synchronize", help="synchronize pigbrother and pupitre data", action="store_true")
    args = parser.parse_args()
    print(f"args: {args}", flush=True)

    # Channels
    channels_dict = {
        "Référence_GR1": "Courant_GR1",
        "Référence_GR2": "Courant_GR2",
    }
    uchannels_dict = {
        "Référence_GR1": ["ALL_internes", 'Interne1', 'Interne2', 'Interne3', 'Interne4', 'Interne5', 'Interne6', 'Interne7'],
        "Référence_GR2": ["ALL_externes", 'Externe1', 'Externe2'],
    }
    pupitre_dict = {
        "M9": {
            "Référence_GR1": "IH",
            "Référence_GR2": "IB",
        },
        "M8": {
            "Référence_GR1": "IB",
            "Référence_GR2": "IH",
        },
        "M10": {
            "Référence_GR1": "IB",
            "Référence_GR2": "IH",
        }
    }
    upupitre_dict = {
        "M9": {
            "Référence_GR1": ["UH"],
            "Référence_GR2": ["UB", "Ucoil15", "Ucoil16"],
        },
        "M8": {
            "Référence_GR2": ["UH"],
            "Référence_GR1": ["UB", "Ucoil15", "Ucoil16"],
        },
        "M10": {
            "Référence_GR2": ["UH"],
            "Référence_GR1": ["UB", "Ucoil15", "Ucoil16"],
        }
    }
    
    filenames = []
    mdatas = []
    for i,file in enumerate(natsorted(args.input_file)):
        f_extension = os.path.splitext(file)[-1]
        if f_extension != ".tdms":
            raise RuntimeError("so far file with tdms extension are implemented")

        dirname = os.path.dirname(file)
        print(f"dirname: {dirname}")
        filename = os.path.basename(file).replace(f_extension, "")
        filenames.append(filename)

        result = filename.startswith("M")
        insert = "tututu"
        index = filename.index("_")
        site = filename[0:index]

        mrun = MagnetRun.fromtdms(site, insert, file)
        mdata = mrun.getMData()
        mdatas.append(mdata)

        (psite, pmode, ptimestamp) = filename.split("_") 
        pdate, ptime = ptimestamp.split('-')
        (timestamp, formatted_timestamp) = convert_to_timestamp(pdate, ptime)
        print(f'timestamp={formatted_timestamp}')
        group = "Courants_Alimentations"
        pkey = f"{group}/{args.key}"
        print(f't0 = { mdata.Groups[group][args.key]["wf_start_time"]}')


        # check ECO mode: "Référence_GR1" != "Référence_GR2"
        GR = mdata.getData(
            ["Courants_Alimentations/Référence_GR1", "Courants_Alimentations/Référence_GR2"]
        )
        GR["diff"] = GR["Référence_GR2"] - GR["Référence_GR1"]

        if GR["diff"].abs().max() > 5 and not (
            GR["Référence_GR2"].abs().max() <= 1 or GR["Référence_GR1"].abs().max() <= 1
        ):
            print(f"{filename}: ECOmode")
        del GR

        # perform for selected key
        (regimes, times, values, components) = trends(mdata, "t", key=f"{group}/{args.key}", window=1, threshold=0.5, show=False, save=args.save, debug=args.debug)
        print(f"{filename} {pkey}: regimes={regimes}")    

        # perform for all uchannels    
        ugroup = "Tensions_Aimant"
        for uchannel in uchannels_dict[args.key]:
            print(f"{filename}: {uchannel}", end="", flush=True)
            (regimes, times, values, components) = trends(mdata, "t", key=f"{ugroup}/{uchannel}", window=1, threshold=0.05, show=False, save=args.save, debug=args.debug)
            print(f": regimes={regimes}")

    # search for Pupitre / Archive files
    pupitre = filenames[0].replace("_Overview_", "_")
    date = pupitre.split("_")
    time = date[1].split("-")
    pupitre_datadir = args.pupitre_datadir
    #pupitre_filter = f"{pupitre_datadir}/{site}_20{time[0][0:2]}.{time[0][2:4]}.{time[0][4:]}---{time[1][0:2]}:{time[1][2:]}:*.txt"
    pupitre_filter = f"{pupitre_datadir}/{site}_20{time[0][0:2]}.{time[0][2:4]}.{time[0][4:]}---*.txt"
    if args.debug: print(f"\npupitre: {pupitre_filter}")

    pigbrother = filenames[0].replace("Overview", "Archive")
    time = filenames[0].split("-")
    archive_datadir = dirname.replace("Overview", "Fichiers_Archive")
    archive_filter = f"{archive_datadir}/{pigbrother.replace(time[1],'*.tdms')}"
    if args.debug: print(f"pigbrother archive_filter: {archive_filter}, time={time}")

    default_datadir = dirname.replace("Overview", "Fichiers_Default")
    default_filefilter = f"{filenames[0].replace('_Overview_','_Default_')}"
    default_filter = f"{default_datadir}/{default_filefilter.replace(time[1],'*.tdms')}"
    if args.debug: print(f"pigbrother default_filter: {default_filter}, time={time}")   
    import glob

    pupitre_files = glob.glob(pupitre_filter)
    archive_files = glob.glob(archive_filter)
    default_files = glob.glob(default_filter)
    if args.debug: 
        print(f"Pupitre files: {pupitre_files}")
        print(f"Archive files: {archive_files}")
        print(f"Default files: {default_files}")
    

    my_ax=plt.gca()
    legends = []
    for i, mdata in enumerate(mdatas):
        legends.append(f"{filenames[i]}: {args.key}")
        mdata.plotData(x="timestamp", y=pkey, ax=my_ax)

    print("\nMerge Archive files")
    df_ = []
    channel = channels_dict[args.key]
    for i, afile in enumerate(natsorted(archive_files)):
        mrun = MagnetRun.fromtdms(site, insert, afile)
        afilename = os.path.basename(afile).replace(f_extension, "")
        mdata = mrun.getMData()
        df = pd.DataFrame(mdata.getTdmsData(group, channel))
        print(f'{afilename} group={group}, channel={channel}: ', end="", flush=True)
        t0 = mdata.Groups[group][channel]["wf_start_time"]
        dt = mdata.Groups[group][channel]["wf_increment"]
        df["timestamp"] = [
            np.datetime64(t0).astype(datetime.datetime)
            + datetime.timedelta(0, i * dt)
            for i in df.index.to_list()
        ]
        print(f't0={t0}, dt={dt}, len={len(df)}')
        if args.debug:
            print(df.head())
            print(df.tail())
        df_.append(df)

        """
        (regimes, times, values, components) = trends(mdata, "t", key=f"{group}/{channel}", window=1, threshold=0.5, show=(not args.save), save=args.save, debug=args.debug)
        print(f"{filename} {group}/{channel}: regimes={regimes}")    
        for uchannel in uchannels_dict[args.key]:
            print(f"\n{filename}: {uchannel}")
            (regimes, times, values, components) = trends(mdata, "t", key=f"{ugroup}/{uchannel}", window=1, threshold=0.05, show=(not args.save), save=args.save, debug=args.debug)
            print(f"{filename} {ugroup}/{uchannel}: regimes={regimes}")
        """

    df_archive = pd.concat(df_)
    t0 = df_archive.iloc[0]["timestamp"]
    df_archive["t"] = df_archive.apply(
        lambda row: (row.timestamp - t0).total_seconds(),
        axis=1,
    )
    # df_archive.drop(["timestamp"], axis=1, inplace=True)

    if args.debug:
        print(df_archive.head())
        print(df_archive.tail())
    df_archive.plot(x="timestamp", y=channel, ax=my_ax)
    legends.append(f'{channel}')

    print("\nMerge Defauts")
    for defaultfile in natsorted(default_files):
        dfile = os.path.basename(defaultfile).replace('.tdms', "")
        (dsite, dmode, dtimestamp, dcomment) = dfile.split('_')
        ddate, dtime = dtimestamp.split('-')
        #convert ddate and dtime into a timestamp
        (timestamp, formatted_timestamp) = convert_to_timestamp(ddate, dtime)
        print(f"{defaultfile} -> {dcomment} at {formatted_timestamp}")
        # plot()

    # make sure that site is unique
    site = filenames[0].split("_")[0]

    print("\nMerge Pupitre")
    df_ = []
    for i, pfile in enumerate(natsorted(pupitre_files)):
        mrun = MagnetRun.fromtxt(site, insert, pfile)
        pfilename = os.path.basename(pfile).replace(f_extension, "")
        mdata = mrun.getMData()
        print(f'{pfilename} {pupitre_dict[site][args.key]}: ', end="", flush=True)
        df = pd.DataFrame(mdata.getData(["timestamp", pupitre_dict[site][args.key]]))
        print(f't0={mdata.Data.iloc[0]["timestamp"]}')
        if args.debug:
            print(df.head())
            print(df.tail())
        df_.append(df)

        """
        (regimes, times, values, components) = trends(mdata, "t", key=f"{pupitre_dict[site][args.key]}", window=1, threshold=1, show=False, save=args.save, debug=args.debug)
        print(f"{filename} {pupitre_dict[site][args.key]}: regimes={regimes}")    
        for uchannel in upupitre_dict[site][args.key]:
            print(f"\n{filename}: {uchannel}")
            (regimes, times, values, components) = trends(mdata, "t", key=f"{uchannel}", window=1, threshold=0.5, show=False, save=args.save, debug=args.debug)
            print(f"{filename} {uchannel}: regimes={regimes}")
        """

    msg = "(nosync)"
    if df_:
        pt0 = df_[0].iloc[0]["timestamp"]
        if len(df_) > 1:
            df_parchive = pd.concat(df_)
        else:
            df_parchive = df_[0]  

        # synchronize pupitre with pigbrother overview
        if args.synchronize:
            msg = f'(synchronized on pigbrother {t0})'
            print("t0 (pigbrother):", t0)
            print("t0 (pupitre):", pt0)
            print("diff t0 (pigbrother/pupitre):", t0 - pt0)
            df_parchive["timestamp"] = df_parchive["timestamp"] + pd.to_timedelta(t0 - pt0)

        npt0 = df_parchive.iloc[0]["timestamp"]
        
        df_parchive["t"] = df_parchive.apply(
            lambda row: (row.timestamp - npt0).total_seconds(),
            axis=1,
        )
        # df_parchive.drop(["timestamp"], axis=1, inplace=True)

        # plots
        if args.debug:
            print(df_parchive.head())
            print(df_parchive.tail())
        df_parchive.plot(x="timestamp", y=pupitre_dict[site][args.key], ax=my_ax)
        legends.append(f'pupitre {pupitre_dict[site][args.key]}')
        plt.legend(labels=legends)


    plt.title(f'{filename}: {channel} {msg}')
    plt.grid()
    if args.show:
        plt.show()
    if args.save:
        plt.savefig(f"{filename}-{channel}-concat.png", dpi=300)
    plt.close()

    if df_:
        # compute lag correlation
        print("t0 (pigbrother):", df_archive["timestamp"].iloc[0])
        print("t0 (pupitre):", df_parchive["timestamp"].iloc[0])
        t0_diff =  df_archive["timestamp"].iloc[0] - df_parchive["timestamp"].iloc[0]
        print("diff t0 (pigbrother-pupiptre):", t0_diff.seconds, "s")
        df_parchive.drop(["timestamp"], axis=1, inplace=True)
        df_archive.drop(["timestamp"], axis=1, inplace=True)
        print(f"pigbrother data: {df_archive.keys()}")
        print(f"pupitre data: {df_parchive.keys()}")
        df_parchive.set_index('t', inplace=True)
        df_archive.set_index('t', inplace=True)
        if args.debug:
            print("pupitre:", df_parchive[pupitre_dict[site][args.key]])
            print("pigbrother:", df_archive[channels_dict[args.key]])
        
        # pseries = 
        # series = 
        lag = lag_correlation(df_parchive[pupitre_dict[site][args.key]], df_archive[channels_dict[args.key]], start_index=0, end_index=None, show=args.show, save=args.save)
        print("lag:", df_archive.index[lag], "s")
        print("lag: diff t0", df_archive.index[lag] - df_parchive.index[0])
