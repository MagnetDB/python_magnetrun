"""Main module."""

from .MagnetRun import MagnetRun
from .processing.trends import trends

import os
import datetime
from natsort import natsorted

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["text.usetex"] = True

def convert_to_timestamp(date_str, time_str):
    """_summary_

    :param date_str: _description_
    :type date_str: _type_
    :param time_str: _description_
    :type time_str: _type_
    :return: _description_
    :rtype: _type_
    """

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

def lag_correlation(data1: dict, data2: dict, show: bool=False, save: bool=False, debug: bool=False):
    """_summary_

    :param data1: _description_
    :type data1: dict
    :param data2: _description_
    :type data2: dict
    :param show: _description_, defaults to False
    :type show: bool, optional
    :param save: _description_, defaults to False
    :type save: bool, optional
    :param debug: _description_, defaults to False
    :type debug: bool, optional
    :return: _description_
    :rtype: _type_
    """

    from scipy.signal import correlate, correlation_lags

    series = data1["df"]
    name_series = data1["field"]
    start_index1 = data1["range"]["start"]
    end_index1 = data1["range"]["end"]

    trend = data2["df"]
    name_trend = data2["field"]
    start_index2 = data2["range"]["start"]
    end_index2 = data2["range"]["end"]

    # Select a slice of the time series
    if start_index1 != 0:
        if end_index1 is not None:
            time_series_slice = series[start_index1:end_index1]
        else:
            time_series_slice = series[start_index1:]
    else:
        if end_index1 is not None:
            time_series_slice = series[:end_index1]
        else:
            time_series_slice = series

    # Select a slice of the trend series
    if start_index2 != 0:
        if end_index2 is not None:
            trend_slice = trend[start_index2:end_index2]
        else:
            trend_slice = trend[start_index2:]
    else:
        if end_index2 is not None:
            trend_slice = trend[:end_index2]
        else:
            trend_slice = trend

    # Drop NaN values from trend slice
    # trend_slice = trend_slice[~np.isnan(trend_slice)]
    # time_series_slice = time_series_slice[-len(trend_slice):]

    # Compute cross-correlation
    # correlation = correlate(time_series_slice - np.mean(time_series_slice), trend_slice - np.mean(trend_slice))
    correlation = correlate(time_series_slice -time_series_slice.mean(), trend_slice -trend_slice.mean())
    lags = correlation_lags(time_series_slice.size, trend_slice.size, mode="full")

    # Find the lag with maximum correlation
    lag = lags[np.argmax(correlation)]

    time_trend_slice_lag = time_series_slice.copy()
    
    time_shift = pd.to_timedelta(f'{lag}s')
    time_trend_slice_lag.index = time_trend_slice_lag.index+time_shift
    print("after lag")
    print(time_trend_slice_lag.head())

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_series_slice, label=name_series)
    plt.plot(trend_slice, label=name_trend, color='red')
    plt.plot(time_trend_slice_lag, label=f'{name_series} with lag {lag}s', color='green', marker="o", linestyle='None', alpha=0.2)
    plt.title(f'{name_series} and {name_trend}')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(lags, correlation, label='Cross-Correlation')
    plt.axvline(lag, color='red', linestyle='--', label=f'Lag = {lag}')
    plt.title(f'Cross-Correlation between {name_series} and {name_trend}')
    plt.xlabel('Lag')
    plt.ylabel('Cross-Correlation')
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig("lag_correlation.png", dpi=300)
    if show:
        plt.show()
    plt.close()

    return time_shift

if __name__ == "__main__":
    import sys

    # Print all arguments
    print("Script name:", sys.argv[0])
    print("All arguments:", sys.argv[1:])

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
    threshold_dict = {
        "Référence_GR1": 0.5,
        "Courant_GR1": 0.5,
        "ALL_internes": 0.1,
        "Interne1": 1.e-2,
        "Interne2": 1.e-2,
        "Interne3": 1.e-2, 
        "Interne4": 1.e-2, 
        "Interne5": 1.e-2, 
        "Interne6": 1.e-2, 
        "Interne7": 1.e-2,
        "Référence_GR2": 0.5,
        "Courant_GR2": 0.5,
        "ALL_externes": 0.1, 
        "Externe1": 1.e-2, 
        "Externe2": 1.e-2,
        "IH": 0.5,
        "UH": 0.1, 
        "Ucoil1": 1.e-2, 
        "Ucoil2": 1.e-2,
        "Ucoil3": 1.e-2, 
        "Ucoil4": 1.e-2,
        "Ucoil5": 1.e-2, 
        "Ucoil6": 1.e-2,
        "Ucoil7": 1.e-2, 
        "Ucoil8": 1.e-2, 
        "Ucoil9": 1.e-2,
        "Ucoil10": 1.e-2, 
        "Ucoil11": 1.e-2,
        "Ucoil12": 1.e-2, 
        "Ucoil13": 1.e-2,
        "Ucoil14": 1.e-2, 
        "IB": 0.5,
        "UB": 0.1, 
        "Ucoil15": 1.e-2, 
        "Ucoil16": 1.e-2
    }
    print(list(threshold_dict.keys()))

    # Set variables to avoid unbound warning
    insert = ""
    site = ""
    dirname = ""
    filename = ""
    f_extension = ""

    input_files = natsorted(args.input_file)
    filenames = []
    mdatas = []
    df_ = []
    for i,file in enumerate(input_files):
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

    # search for Pupitre / Archive files
    pupitre = filenames[0].replace("_Overview_", "_")
    date = pupitre.split("_")
    time = date[1].split("-")
    pupitre_datadir = args.pupitre_datadir
    #pupitre_filter = f"{pupitre_datadir}/{site}_20{time[0][0:2]}.{time[0][2:4]}.{time[0][4:]}---{time[1][0:2]}:{time[1][2:]}:*.txt"
    pupitre_filter = f"{pupitre_datadir}/{site}_20{time[0][0:2]}.{time[0][2:4]}.{time[0][4:]}---*.txt"
    if args.debug:
        print(f"\npupitre: {pupitre_filter}")

    pigbrother = filenames[0].replace("Overview", "Archive")
    time = filenames[0].split("-")
    archive_datadir = dirname.replace("Overview", "Fichiers_Archive")
    archive_filter = f"{archive_datadir}/{pigbrother.replace(time[1],'*.tdms')}"
    if args.debug:
        print(f"pigbrother archive_filter: {archive_filter}, time={time}")

    default_datadir = dirname.replace("Overview", "Fichiers_Default")
    default_filefilter = f"{filenames[0].replace('_Overview_','_Default_')}"
    default_filter = f"{default_datadir}/{default_filefilter.replace(time[1],'*.tdms')}"
    if args.debug:
        print(f"pigbrother default_filter: {default_filter}, time={time}")
    import glob

    pupitre_files = natsorted(glob.glob(pupitre_filter))
    archive_files = natsorted(glob.glob(archive_filter))
    default_files = natsorted(glob.glob(default_filter))
    if args.debug:
        print(f"Overview files: {natsorted(args.input_file)}")
        print(f"Pupitre files: {pupitre_files}")
        print(f"Archive files: {archive_files}")
        print(f"Default files: {default_files}")

    title = os.path.basename(args.input_file[0]).replace(".tdms", "")
    if len(args.input_file) >1:
        title = title.split('-')[0]

    group = "Courants_Alimentations"
    ugroup = "Tensions_Aimant"
    mdatas = []
    df_ = []
    for i,file in enumerate(input_files):
        f_extension = os.path.splitext(file)[-1]

        dirname = os.path.dirname(file)
        filename = os.path.basename(file).replace(f_extension, "")

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
        (regimes, times, values, components) = trends(mdata, "t", key=f"{pkey}", window=1, threshold=threshold_dict[args.key], show=False, save=args.save, debug=args.debug)
        print(f"{filename} {pkey}: regimes={regimes}")

        (iregimes, itimes, ivalues, icomponents) = trends(mdata, "t", key=f"{group}/{channels_dict[args.key]}", window=1, threshold=threshold_dict[channels_dict[args.key]], show=False, save=args.save, debug=args.debug)
        print(f"{filename} {group}{channels_dict[args.key]}: iregimes={iregimes}")
        print(f"{filename} {group}{channels_dict[args.key]}: itimes={itimes}")

        # perform for all uchannels
        uregimes = []
        utimes = []
        uvalues = []
        ucomponents = []
        for uchannel in uchannels_dict[args.key]:
            print(f"{filename} {ugroup}/{uchannel}: ", end="", flush=True)
            (uregime, utime, uvalue, ucomponent) = trends(mdata, "t", key=f"{ugroup}/{uchannel}", window=1, threshold=threshold_dict[uchannel], show=False, save=args.save, debug=args.debug)
            uregimes.append(uregime)
            utimes.append(utime)
            uvalues.append(uvalue)
            ucomponents.append(ucomponent)
            print(uregime)
            # print(f'{uchannel}: utime={utime}')
            
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

        for i,uregime in enumerate(uregimes):
            t0 = 0
            utime = utimes[i]
            for j in range(1, len(uregime)):
                # print(f'axvspan[{i}][{j}]: [{t0},{utime[j]}], regime={uregime[j-1]}, color={color_dict[uregime[j-1]]}')
                my_ax.axvspan(t0, utime[j], facecolor=color_dict[uregime[j-1]], alpha=.5)
                t0 = utime[j]

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

    print("\nMerge Archive files")
    df_ = []
    channel = channels_dict[args.key]
    for i, afile in enumerate(archive_files):
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
    for defaultfile in default_files:
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
        df_.append(df)

        (pregimes, ptimes, pvalues, pcomponents) = trends(mdata, "t", key=pupitre_dict[site][args.key], window=1, threshold=threshold_dict[pupitre_dict[site][args.key]], show=False, save=args.save, debug=args.debug)
        print(f"{pfilename} {pupitre_dict[site][args.key]}: pregimes={pregimes}")
        print(f"{pfilename} {pupitre_dict[site][args.key]}: ptimes={ptimes}")

    df_pupitre = pd.concat(df_)
    pt0 = df_pupitre.iloc[0]["timestamp"]
    df_pupitre["t"] = df_pupitre.apply(
        lambda row: (row.timestamp - pt0).total_seconds(),
        axis=1,
    )

    msg = "(nosync)"
    # synchronize pupitre with pigbrother overview
    if args.synchronize:
        print('\nSynchronize pigbrother and pupitre data')
        print("t0 (pigbrother overview):", ot0)
        print("t0 (pigbrother archive):", at0)
        print("t0 (pupitre):", pt0)

        print("diff t0 (pigbrother/pupitre):", pt0 - ot0)
        df_pupitre["timestamp"] = df_pupitre["timestamp"] + pd.to_timedelta(ot0 - pt0)
        msg = f'(sync with pigbrother {(ot0 - pt0).total_seconds()} s)'

        pt0 = df_pupitre.iloc[0]["timestamp"]
        df_pupitre.drop(["t"], axis=1, inplace=True)
        df_pupitre["t"] = df_pupitre.apply(
           lambda row: (row.timestamp - pt0).total_seconds(),
            axis=1,
        )

    # compute lag correlation
    # algo seems to be very sensitive
    # have to focus on a steep change
    # ex: pigbrotherdata/Fichiers_Data/M9/Overview/M9_Overview_240509-*.tdms from 0 to 50
    # lag: estimated by hand 5s found 5s
    # NB: index range is hardcoded right now
    #
    # the alternative is to use signal signature

    print('\nLag correlation: pupitre/pigbrother overview')
    print("t0 (overview):", df_overview["timestamp"].iloc[0])
    print("t0 (pupitre):", df_pupitre["timestamp"].iloc[0])

    ts_overview_field = channels_dict[args.key]
    ts_overview = df_overview.loc[:,["timestamp", ts_overview_field]]
    ts_pupitre_field = pupitre_dict[site][args.key]
    ts_pupitre = df_pupitre.loc[:,["timestamp",ts_pupitre_field]]
    ts_pupitre.set_index('timestamp', inplace=True)
    ts_overview.set_index('timestamp', inplace=True)
    ts_pupitre = ts_pupitre.iloc[:,0]
    ts_overview = ts_overview.iloc[:,0]
    if args.debug:
        print('before reindex')
        print('pupitre:', ts_pupitre.size, type(ts_pupitre))
        print(ts_pupitre.head())
        print(ts_pupitre.tail())
        print('overview:', ts_overview.size, type(ts_overview))
        print(ts_overview.head())
        print(ts_overview.tail())
        my_ax=plt.gca()
        ts_pupitre.plot(style='.', ax=my_ax)
        ts_overview.plot(style='o', alpha=0.5, ax=my_ax)
        plt.title('before reindex')
        plt.grid()
        plt.show()

    # reindex to make sure that timeseries share the same index
    ts_pupitre.resample('1s') #.reindex(new_index, method='ffill')
    ts_overview.resample('1s') #.reindex(new_index, method='ffill')
    if args.debug:
        print('after reindex')
        print('pupitre:', ts_pupitre.size, type(ts_pupitre))
        print(ts_pupitre.head())
        print(ts_pupitre.tail())
        print('overview:', ts_overview.size, type(ts_overview))
        print(ts_overview.head())
        print(ts_overview.tail())
        my_ax=plt.gca()
        ts_pupitre.plot(style='.', ax=my_ax)
        ts_overview.plot(style='o', alpha=0.5, ax=my_ax)
        plt.title('after reindex')
        plt.grid()
        plt.show()

    # return the index of the element whci equals 7 in myseries: Index(myseries).get_loc(7)
    # get index from its positions?
    ts_pupitre_data = {
        "field": ts_pupitre_field,
        "df": ts_pupitre,
        "range" : {
            "start":  ts_pupitre_field.index[0],
            "end": 500
        }
    }
    ts_overview_data = {
        "field": ts_overview_field,
        "df": ts_overview,
        "range" : {
            "start": 1000,
            "end": 1500
        }
    }
    lag = lag_correlation(ts_pupitre_data, ts_overview_data, show=args.show, save=args.save,)


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
        plt.savefig(f"{filename}-{channel}-timestamp-concat.png", dpi=300)
    plt.close()

    my_ax=plt.gca()
    legends = [f"{args.key}"]
    df_overview.plot(x="t", y=args.key, ax=my_ax)

    df_archive.plot(x="t", y=channels_dict[args.key], ax=my_ax)
    legends.append(f'{channels_dict[args.key]}')

    df_pupitre.plot(x="t", y=pupitre_dict[site][args.key], ax=my_ax)
    legends.append(f'pupitre {pupitre_dict[site][args.key]}')
    plt.legend(labels=legends)

    plt.title(f'{title}: {args.key} {msg}')
    plt.grid()
    if args.show:
        plt.show()
    if args.save:
        plt.savefig(f"{filename}-{channel}-concat.png", dpi=300)
    plt.close()
    