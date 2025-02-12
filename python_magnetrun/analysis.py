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

def find_duplicates(df: pd.DataFrame, name: str, key: str, strict: bool = False):
    dups = pd.Index(df[key].to_numpy()).duplicated()
    unique, counts = np.unique(dups, return_counts=True)
    #print(f'duplicated df[t]: {dict(zip(unique, counts))}')
    dups_dict = dict(zip(unique, counts))
    # print(f'dups_dict df[t]: {dups_dict}')
    if np.True_ in dups_dict:
        dups_index = np.where(dups == np.True_)
        drop_index = [i.item() for i in dups_index]
        print(f'duplicated index found in {name} at rows={drop_index}')
        for i in drop_index:
            print(f'rows={i}/{i-1}:\n{df.loc[i-1:i].head()}')
        if strict:
            raise RuntimeError(f"found duplicates time in {name}")
        else:
            print("remove duplicates")
            df.drop(df.index[drop_index], inplace=True)
            return df

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
    
    title = os.path.basename(args.input_file[0]).replace(".tdms", "")
    if len(args.input_file) >1:
        title = title.split('-')[0]

    filenames = []
    mdatas = []
    df_ = []
    for i,file in enumerate(natsorted(args.input_file)):
        f_extension = os.path.splitext(file)[-1]
        if f_extension != ".tdms":
            raise RuntimeError("so far file with tdms extension are implemented")

        dirname = os.path.dirname(file)
        filename = os.path.basename(file).replace(f_extension, "")
        filenames.append(filename)
        print(f"dirname={dirname}, filename={filename}: ", end="", flush=True)
        
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
        print(f'timestamp={formatted_timestamp}, ', end="", flush=True)
        group = "Courants_Alimentations"
        pkey = f"{group}/{args.key}"
        print(f't0 = { mdata.Groups[group][args.key]["wf_start_time"]}')


        # check ECO mode: "Référence_GR1" != "Référence_GR2"
        GR = mdata.getData(
            ["Courants_Alimentations/Référence_GR1", "Courants_Alimentations/Référence_GR2"]
        ).copy()
        GR.loc[:, "diff"] = GR.loc[:,"Référence_GR2"] - GR.loc[:,"Référence_GR1"]

        if GR["diff"].abs().max() > 5 and not (
            GR["Référence_GR2"].abs().max() <= 1 or GR["Référence_GR1"].abs().max() <= 1
        ):
            print(f"{filename}: mode=ECOmode")
        del GR

        # perform for selected key
        (regimes, times, values, components) = trends(mdata, "t", key=f"{pkey}", window=1, threshold=0.5, show=False, save=args.save, debug=args.debug)
        print(f"{filename} {pkey}: regimes={regimes}")    

        (iregimes, itimes, ivalues, icomponents) = trends(mdata, "t", key=f"{group}/{channels_dict[args.key]}", window=1, threshold=0.5, show=False, save=args.save, debug=args.debug)
        print(f"{filename} {group}{channels_dict[args.key]}: iregimes={iregimes}")    

        # perform for all uchannels    
        ugroup = "Tensions_Aimant"
        for uchannel in uchannels_dict[args.key]:
            print(f"{filename} {ugroup}/{uchannel}: ", end="", flush=True)
            (uregimes, utimes, uvalues, ucomponents) = trends(mdata, "t", key=f"{ugroup}/{uchannel}", window=1, threshold=1.e-2, show=False, save=args.save, debug=args.debug)
            print(f"uregimes={uregimes}")

        color_dict = {"U": "red", "P": "green", "D": "blue"}
        my_ax = plt.gca()
        mdata.plotData(x="t", y=pkey, ax=my_ax, normalize=True)
        mdata.plotData(x="t", y=f"{group}/{channels_dict[args.key]}", ax=my_ax, normalize=True)
        for uchannel in uchannels_dict[args.key]:
            mdata.plotData(x="t", y=f"{ugroup}/{uchannel}", ax=my_ax, normalize=True)

        t0 = 0
        for i in range(1, len(regimes)):
            #print(f'axvspan[{i}]: [{t0},{times[i]}], regime={regimes[i-1]}, color={color_dict[regimes[i-1]]}')
            my_ax.axvspan(t0, times[i], facecolor=color_dict[regimes[i-1]], alpha=.5)
            t0 = times[i]
        t0 = 0
        for i in range(1, len(iregimes)):
            #print(f'axvspan[{i}]: [{t0},{itimes[i]}], regime={iregimes[i-1]}, color={color_dict[iregimes[i-1]]}')
            my_ax.axvspan(t0, itimes[i], facecolor=color_dict[iregimes[i-1]], alpha=.5)
            t0 = itimes[i]
        
        t0 = 0
        for i in range(1, len(uregimes)):
            #print(f'axvspan[{i}]: [{t0},{utimes[i]}], regime={uregimes[i-1]}, color={color_dict[uregimes[i-1]]}')
            my_ax.axvspan(t0, utimes[i], facecolor=color_dict[uregimes[i-1]], alpha=.5)
            t0 = utimes[i]

        plt.xlabel('Normalized Field')
        plt.show()
        plt.close()

        # extract data
        df = pd.DataFrame(mdata.getTdmsData(group, [args.key, channels_dict[args.key]]))
        t0 = mdata.Groups[group][args.key]["wf_start_time"]
        dt = mdata.Groups[group][args.key]["wf_increment"]
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

    df_overview = pd.concat(df_)
    ot0 = df_overview.iloc[0]["timestamp"]
    df_overview["t"] = df_overview.apply(
        lambda row: (row.timestamp - ot0).total_seconds(),
        axis=1,
    )
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
    at0 = df_archive.iloc[0]["timestamp"]
    df_archive["t"] = df_archive.apply(
        lambda row: (row.timestamp - at0).total_seconds(),
        axis=1,
    )
    # df_archive.drop(["timestamp"], axis=1, inplace=True)

    if args.debug:
        print(df_archive.head())
        print(df_archive.tail())

    print("\nMerge Defauts")
    incidents = []
    for defaultfile in natsorted(default_files):
        dfile = os.path.basename(defaultfile).replace('.tdms', "")
        (dsite, dmode, dtimestamp, dcomment) = dfile.split('_')
        ddate, dtime = dtimestamp.split('-')
        #convert ddate and dtime into a timestamp
        (timestamp, formatted_timestamp) = convert_to_timestamp(ddate, dtime)
        
        mrun = MagnetRun.fromtdms(site, insert, defaultfile)
        mdata = mrun.getMData()
        t0 = mdata.Groups[group][args.key]["wf_start_time"]
        print(f"{defaultfile} -> {dcomment} at {formatted_timestamp}, t0={t0}")
        
        incidents.append(t0)


    # make sure that site is unique
    site = filenames[0].split("_")[0]

    print("\nMerge Pupitre")
    df_ = []
    for i, pfile in enumerate(natsorted(pupitre_files)):
        mrun = MagnetRun.fromtxt(site, insert, pfile)
        pfilename = os.path.basename(pfile).replace(f_extension, "")
        mdata = mrun.getMData()
        # print(mdata.Keys)
        print(f'{pfilename} {pupitre_dict[site][args.key]}: ', end="", flush=True)
        df = pd.DataFrame(mdata.getData(["timestamp", pupitre_dict[site][args.key]]))
        print(f'{pfilename} t0={mdata.Data.iloc[0]["timestamp"]}, len={df.shape[0]}, duration={mdata.getDuration()} s')

        find_duplicates(df, pfilename, "timestamp")
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

    df_pupitre = pd.concat(df_)
    pt0 = df_pupitre.iloc[0]["timestamp"]
    df_pupitre["t"] = df_pupitre.apply(
        lambda row: (row.timestamp - pt0).total_seconds(),
        axis=1,
    )

    msg = "(nosync)"
    # synchronize pupitre with pigbrother overview
    if args.synchronize:
        print('synchronize pigbrother and pupitre data')
        print("t0 (pigbrother overview):", ot0)
        print("t0 (pigbrother archive):", at0)
        print("t0 (pupitre):", pt0)

        print("diff t0 (pigbrother/pupitre):", pt0 - ot0)
        df_pupitre["timestamp"] = df_pupitre["timestamp"] + pd.to_timedelta(ot0 - pt0)
        msg = f'(sync with pigbrother {(ot0 - pt0).totalseconds} s)'
        
        pt0 = df_pupitre.iloc[0]["timestamp"]
        df_pupitre.drop(["t"], axis=1, inplace=True)
        df_pupitre["t"] = df_pupitre.apply(
           lambda row: (row.timestamp - pt0).total_seconds(),
            axis=1,
        )

    # compute lag correlation
    print('\nLag correlation: pupitre/pigbrother overview')
    print("t0 (overview):", df_overview["timestamp"].iloc[0])
    print("t0 (pupitre):", df_pupitre["timestamp"].iloc[0])
    
    # re-sample to make sure that the two dataframes have the same time index  
    # get index min and max from timeseries
    # create a new index from min(index min) and max(index max) with min() and max() retunrs an int 
    # try to reindex
    # take the series with the greatest number of rows as a reference
    from math import floor, ceil
    tmin = 0
    tmax = 0
    for df_ in [df_overview, df_pupitre]:
        tmin = min(df_['t'].min(), tmin)
        tmax = max(df_['t'].max(), tmax)
        
    new_index = [i for i in range(floor(tmin),ceil(tmax)+1)]
    print(f"new_index: from {new_index[0]} to {new_index[-1]}")

    ts_overview = df_overview.loc[:,["t", channels_dict[args.key]]]
    ts_pupitre = df_pupitre.loc[:,["t", pupitre_dict[site][args.key]]]
    ts_pupitre.set_index('t', inplace=True)
    ts_overview.set_index('t', inplace=True)
    ts_pupitre = ts_pupitre.iloc[:,0]
    ts_overview = ts_overview.iloc[:,0]
    
    # reindex to make sure that timeseries share the same index 
    ts_pupitre.reindex(new_index, method='ffill')
    ts_overview.reindex(new_index, method='ffill')
    if args.debug:
        print('pupitre:', ts_pupitre.size, type(ts_pupitre))
        print(ts_pupitre.head())
        print(ts_pupitre.tail())
        print('overview:', ts_overview.size, type(ts_overview))
        print(ts_overview.head())
        print(ts_overview.tail())
        my_ax=plt.gca()
        ts_pupitre.plot(style='.', ax=my_ax)
        ts_overview.plot(style='o', ax=my_ax)
        plt.grid()
        plt.show()

    lag = lag_correlation(ts_pupitre, ts_overview, start_index=0, end_index=None, show=args.show, save=args.save)

    # plots

    my_ax=plt.gca()
    legends = [f"{args.key}"]
    df_overview.plot(x="timestamp", y=args.key, ax=my_ax)

    df_archive.plot(x="timestamp", y=channels_dict[args.key], ax=my_ax)
    legends.append(f'{channels_dict[args.key]}')

    df_pupitre.plot(x="timestamp", y=pupitre_dict[site][args.key], ax=my_ax)
    legends.append(f'pupitre {pupitre_dict[site][args.key]}')
    plt.legend(labels=legends)

    # plot incidents
    for incident in incidents:
        plt.axvline(incident, color="red", alpha=.2, label="Incident")

    plt.title(f'{title}: {args.key} {msg}')
    plt.grid()
    if args.show:
        plt.show()
    if args.save:
        plt.savefig(f"{filename}-{channel}-concat.png", dpi=300)
    plt.close()

