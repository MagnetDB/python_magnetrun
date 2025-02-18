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

from .utils.convert import convert_to_timestamp
from .processing.correlations import lag_correlation
from .signature import Signature
from math import floor, ceil

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
    parser.add_argument("--synchronize", help="rolling window size", action="store_true")
    args = parser.parse_args()
    print(f"args: {args}", flush=True)

    # Colors for signature
    color_dict = {"U": "red", "P": "green", "D": "blue"}
    
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
        "Courant_GR1": 1,
        "ALL_internes": 0.1,
        "Interne1": 1.e-2,
        "Interne2": 1.e-2,
        "Interne3": 1.e-2,
        "Interne4": 1.e-2,
        "Interne5": 1.e-2,
        "Interne6": 1.e-2,
        "Interne7": 1.e-2,
        "Référence_GR2": 0.5,
        "Courant_GR2": 1,
        "ALL_externes": 0.1,
        "Externe1": 0.1, #1.e-2,
        "Externe2": 0.1, #1.e-2,
        "IH": 1,
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
        "IB": 1,
        "UB": 0.1,
        "Ucoil15": 0.1,
        "Ucoil16": 0.1
    }
    # print(list(threshold_dict.keys()))

    # Set variables to avoid unbound warning
    insert = ""
    site = ""
    dirname = ""
    filename = ""
    f_extension = ""

    input_files = natsorted(args.input_file)
    pupitre_files_selected = []
    archive_files_selected = []
    
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
        print(f"dirname={dirname}, filename={filename}: ", flush=True)

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
        print(f"\npupitre_filter: {pupitre_filter}")

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

    ochanges = []
    oregimes = []
    otimes = []
    ovalues = []
    ocomponents = []

    oichanges = []
    oiregimes = []
    oitimes = []
    oivalues = []
    oicomponents = []

    pupitre_files_dict = {}
    archive_files_dict = {}
    pkey = f"{group}/{args.key}"

    # if only one reference, Bitter only exp
    uprobes = uchannels_dict[args.key]
        
    print(f"\nMerge Overview: {len(input_files)}")
    for i,file in enumerate(input_files):
        f_extension = os.path.splitext(file)[-1]

        dirname = os.path.dirname(file)
        filename = os.path.basename(file).replace(f_extension, "")
        print(f'{filename} {group}/{args.key}: ', end="", flush=True)

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
        print(f'time={timestamp}, timestamp={formatted_timestamp}, ', end="", flush=True)
        ot0 = mdata.Groups[group][args.key]["wf_start_time"]
        print(f'ot0 = {ot0}')

        pupitre_files_dict[file] = []
        for pfile in pupitre_files:
            pfilename = os.path.basename(pfile).replace('.txt', "")
            (psite, ptimestamp) = pfilename.split("_")
            pdate, ptime = ptimestamp.split('---')
            # remove seconds
            (ptimestamp, formatted_ptimestamp) = convert_to_timestamp(pdate, ptime[:-3], date_format= "%Y.%m.%d", time_format="%H:%M")
            # print(f"{pfilename} timestamp={formatted_timestamp}, timestamp={formatted_ptimestamp}", end=" ", flush=True)
            if formatted_ptimestamp >= formatted_timestamp:
                pupitre_files_dict[file].append(pfile)

        for value in pupitre_files_dict.values():
            pupitre_files_selected += value
        pupitre_files_selected = natsorted(pupitre_files_selected)
        print(f"\nSelected Pupitre: {pupitre_files_selected}")

        archive_files_dict[file] = []
        for afile in archive_files:
            afilename = os.path.basename(afile).replace('.tdms', "")
            (asite, amode, atimestamp) = afilename.split("_")
            adate, atime = atimestamp.split('-')
            # remove seconds
            (atimestamp, formatted_atimestamp) = convert_to_timestamp(adate, atime)
            # print(f"{afilename} timestamp={formatted_timestamp}, timestamp={formatted_atimestamp}", end=" ", flush=True)
            if formatted_atimestamp >= formatted_timestamp:
                archive_files_dict[file].append(afile)

        archive_files_selected = []
        for value in archive_files_dict.values():
            archive_files_selected += value
        archive_files_selected = natsorted(archive_files_selected)
        print(f"Selected archive: {archive_files_selected}")

        # check ECO mode: "Référence_GR1" != "Référence_GR2"
        bitter_only = True
        if "Courants_Alimentations/Référence_GR1" in mdata.getKeys() and "Courants_Alimentations/Référence_GR2" in mdata.getKeys():
            GR = mdata.getData(
                ["Courants_Alimentations/Référence_GR1", "Courants_Alimentations/Référence_GR2"]
            ).copy()
            GR.loc[:, "diff"] = GR.loc[:,"Référence_GR2"] - GR.loc[:,"Référence_GR1"]

            if GR["diff"].abs().max() > 5 and not (
                GR["Référence_GR2"].abs().max() <= 1 or GR["Référence_GR1"].abs().max() <= 1
            ):
                print(f"{filename}: mode=ECOmode")
            del GR

        if bitter_only:
            uprobes = ["ALL_externes", 'Externe1', 'Externe2']
            print('\n!!! Selecting only U probes for Bitter !!!\n')

        # perform for selected key
        osignature = Signature.from_mdata(mdata, pkey, "t", threshold_dict[args.key])
        # print("osignature:", osignature.regimes)
        osignature.dump(filename)

        (ochange, oregime, otime, ovalue, ocomponent) = trends(mdata, "t", key=f"{pkey}", window=1, threshold=threshold_dict[args.key], show=False, save=args.save, debug=args.debug)
        ochanges.append(ochange)
        oregimes.append(oregime)
        otimes.append(otime)
        ovalues.append(ovalue)
        ocomponents.append(ocomponent)
        #if args.debug:
        print(f"{filename} {pkey}: regimes={oregime}")
        # print(f"{filename} {pkey}: ochanges={ochange}")

        (ichange, iregime, itime, ivalue, icomponent) = trends(mdata, "t", key=f"{group}/{channels_dict[args.key]}", window=1, threshold=threshold_dict[channels_dict[args.key]], show=False, save=args.save, debug=args.debug)
        oichanges.append(ichange)
        oiregimes.append(iregime)
        oitimes.append(itime)
        oivalues.append(ivalue)
        oicomponents.append(icomponent)
        if args.debug:
            print(f"{filename} {group}{channels_dict[args.key]}: ichanges={ichange}")
            print(f"{filename} {group}{channels_dict[args.key]}: iregimes={iregime}")
            print(f"{filename} {group}{channels_dict[args.key]}: itimes={itime}")
        print(f"{filename} {pkey}: oiregime={oregime}")
        # print(f"{filename} {pkey}: oichange={ichange}")

        # perform for all uchannels
        uchanges = []
        uregimes = []
        utimes = []
        uvalues = []
        ucomponents = []

        for uchannel in uprobes :
            print(f'uchannel={uchannel}', end="")
            if f"{ugroup}/{uchannel}" in mdata.getKeys():
                if args.debug: print(f"{filename} {ugroup}/{uchannel}: ", end="", flush=True)
                (uchange, uregime, utime, uvalue, ucomponent) = trends(mdata, "t", key=f"{ugroup}/{uchannel}", window=1, threshold=threshold_dict[uchannel], show=False, save=args.save, debug=args.debug)
                uchanges.append(uchange)
                uregimes.append(uregime)
                utimes.append(utime)
                uvalues.append(uvalue)
                ucomponents.append(ucomponent)
                if args.debug: print(uregime)
                print(f' regimes: {uregime}', flush=True)
            else:
                print(f" - not present in {file}", flush=True)

        my_ax = plt.gca()
        mdata.plotData(x="t", y=pkey, ax=my_ax, normalize=True)
        mdata.plotData(x="t", y=f"{group}/{channels_dict[args.key]}", ax=my_ax, normalize=True)
        for uchannel in uprobes:
            if f"{ugroup}/{uchannel}" in mdata.getKeys():
                mdata.plotData(x="t", y=f"{ugroup}/{uchannel}", ax=my_ax, normalize=True)

        t0 = 0
        for i in range(1, len(oregime)):
            #print(f'axvspan[{i}]: [{t0},{otime[i]}], regime={oregime[i-1]}, color={color_dict[oregime[i-1]]}')
            my_ax.axvspan(t0, otime[i], facecolor=color_dict[oregime[i-1]], alpha=.2)
            t0 = otime[i]
        t0 = 0
        for i in range(1, len(iregime)):
            #print(f'axvspan[{i}]: [{t0},{itimes[i]}], regime={iregimes[i-1]}, color={color_dict[iregimes[i-1]]}')
            my_ax.axvspan(t0, itime[i], facecolor=color_dict[iregime[i-1]], alpha=.2)
            t0 = itime[i]

        for i,uregime in enumerate(uregimes):
            t0 = 0
            utime = utimes[i]
            for j in range(1, len(uregime)):
                # print(f'axvspan[{i}][{j}]: [{t0},{utime[j]}], regime={uregime[j-1]}, color={color_dict[uregime[j-1]]}')
                my_ax.axvspan(t0, utime[j], facecolor=color_dict[uregime[j-1]], alpha=.2)
                t0 = utime[j]

        plt.ylabel('Normalized Field')
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
        print(f'{filename} t0={t0}, dt={dt}, len={len(df)}')
        if args.debug:
            print(df.head())
            print(df.tail())
        df_.append(df)

    if len(df_) > 1:
        df_overview = pd.concat(df_)
    else:
        df_overview = df_[0]

    ot0 = df_overview.iloc[0]["timestamp"]
    df_overview["t"] = df_overview.apply(
        lambda row: (row.timestamp - ot0).total_seconds(),
        axis=1,
    )

    print(f"\nMerge Archive ({len(archive_files_selected)})")
    df_ = []

    at0 = datetime.datetime.now()
    df_archive = pd.DataFrame()

    achanges = []
    aregimes = []
    atimes = []
    avalues = []
    acomponents = []

    channel = channels_dict[args.key]
    for i, afile in enumerate(archive_files_selected):
        afilename = os.path.basename(afile).replace(f_extension, "")
        sizeOfFile = os.path.getsize(afile)
        print(f'{afilename} group={group}, channel={channel}: ', end="", flush=True)
        mrun = MagnetRun.fromtdms(site, insert, afile)
        mdata = mrun.getMData()
        t0 = mdata.Groups[group][channel]["wf_start_time"]
        dt = mdata.Groups[group][channel]["wf_increment"]
        print(f't0={t0}, dt={dt}')
        
        (ichange, iregime, itime, ivalue, icomponent) = trends(mdata, "t", key=f"{group}/{channels_dict[args.key]}", window=480, threshold=threshold_dict[channels_dict[args.key]], show=False, save=args.save, debug=args.debug)
        achanges.append(ichange)
        aregimes.append(iregime)
        atimes.append(itime)
        avalues.append(ivalue)
        acomponents.append(icomponent)
        if args.debug:
            print(f"{afilename} {group}{channels_dict[args.key]}: achange={ichange}")
            print(f"{afilename} {group}{channels_dict[args.key]}: aregimes={iregime}")
            print(f"{afilename} {group}{channels_dict[args.key]}: atime={itime}")

        df = pd.DataFrame(mdata.getTdmsData(group, [channel, args.key]))
        df["timestamp"] = [
            np.datetime64(t0).astype(datetime.datetime)
            + datetime.timedelta(0, i * dt)
            for i in df.index.to_list()
        ]
        # print(f't0={t0}, dt={dt}, len={len(df)}')
        if args.debug:
            print(df.head())
            print(df.tail())
        df_.append(df)

        # Plot trends (works only when 1 archive attached to overview)
        if args.debug:
            my_ax = plt.gca()
            legends = [channels_dict[args.key]]
            mdata.plotData(x="t", y=f"{group}/{channels_dict[args.key]}", ax=my_ax, alpha=0.2)
            legends.append('trend')
            icomponent.plot(ax=my_ax)
            
            # shall get the overview associated to the archive
            legends.append('Overview')
            df_overview.plot(x="t", y=channels_dict[args.key], marker='.', linestyle='None', alpha=0.4, color="r", ax=my_ax)
            my_ax.legend(labels=legends)
            plt.title(f"{afilename}:{channels_dict[args.key]}" )
            # limit to mdata trange
            plt.grid()
            plt.show()
            plt.close()
        

    if df_:
        if len(df_) > 1:
            df_archive = pd.concat(df_)
        else:
            df_archive = df_[0]

        at0 = df_archive.iloc[0]["timestamp"]
        df_archive["t"] = df_archive.apply(
            lambda row: (row.timestamp - at0).total_seconds(),
            axis=1,
        )
        # df_archive.drop(["timestamp"], axis=1, inplace=True)
        if args.debug:
            my_ax = plt.gca()
            df_overview.plot(x="t", y=channels_dict[args.key], label="Overview", color="r", ax=my_ax)
            df_archive.plot(x="t", y=channels_dict[args.key], label="Archive", color="r", alpha=0.2, ax=my_ax)
            plt.title(channels_dict[args.key])
            plt.grid()
            plt.show()
            plt.close()

        if args.debug:
            print(df_archive.head())
            print(df_archive.tail())

    print(f"\nMerge Defauts ({len(default_files)})")
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
    
    print(f"\nMerge Pupitre ({len(pupitre_files_selected)})")
    pchanges = []
    pregimes = []
    ptimes = []
    pvalues = []
    pcomponents = []

    df_ = []
    for i, pfile in enumerate(natsorted(pupitre_files_selected)):
        mrun = MagnetRun.fromtxt(site, insert, pfile)
        pfilename = os.path.basename(pfile).replace(f_extension, "")
        mdata = mrun.getMData()

        # print(mdata.Keys)
        print(f'{pfilename} {pupitre_dict[site][args.key]}: ', end="", flush=True)
        df = pd.DataFrame(mdata.getData(["timestamp", pupitre_dict[site][args.key]]))
        print(f't0={mdata.Data.iloc[0]["timestamp"]}, len={df.shape[0]}, duration={mdata.getDuration()} s')
        df_.append(df)

        (pchange, pregime, ptime, pvalue, pcomponent) = trends(mdata, "t", key=pupitre_dict[site][args.key], window=1, threshold=threshold_dict[pupitre_dict[site][args.key]], show=False, save=args.save, debug=args.debug)
        pchanges.append(pchange)
        pregimes.append(pregime)
        ptimes.append(ptime)
        pvalues.append(pvalue)
        pcomponents.append(pcomponent)
        if args.debug:
            print(f"{pfilename} {pupitre_dict[site][args.key]}: pchange={pchange}")
            print(f"{pfilename} {pupitre_dict[site][args.key]}: pregime={pregime}")
            print(f"{pfilename} {pupitre_dict[site][args.key]}: ptime={ptime}")
        print(f"{pfilename} {pupitre_dict[site][args.key]}: pregime={pregime}")
        # print(f"{pfilename} {pupitre_dict[site][args.key]}: pchange={pchange}")

    if df_:
        if len(df_) > 1:
            df_pupitre = pd.concat(df_)
        else:
            df_pupitre = df_[0]

        pt0 = df_pupitre.iloc[0]["timestamp"]
        df_pupitre["t"] = df_pupitre.apply(
            lambda row: (row.timestamp - pt0).total_seconds(),
            axis=1,
        )

        timeshift = 0
        msg = "(nosync)"
        # synchronize pupitre with pigbrother overview
        if args.synchronize:
            print('\nSynchronize pigbrother and pupitre data')
            print("t0 (pigbrother overview):", ot0)
            print("t0 (pigbrother archive):", at0)
            print("t0 (pupitre):", pt0)

            print(f"timeshift (pigbrother/pupitre): {(ot0 - pt0).total_seconds()} s")
            df_pupitre["timestamp"] = df_pupitre["timestamp"] + pd.to_timedelta(ot0 - pt0)
            timeshift = (ot0 - pt0).total_seconds()
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

        # make a plot to check algo
        # use trends result for overview(channels_dict[args.key]) and pupitre(pupitre_dict[site][args.key])
        my_ax=plt.gca()
        legends = []
        legends.append(f'Overview {args.key}')
        df_overview.plot(x="t", y=args.key, ax=my_ax, marker=".", color='r') # , linestyle='None'
        legends.append(f'Overview {channels_dict[args.key]}')
        df_overview.plot(x="t", y=channels_dict[args.key], ax=my_ax, marker="*", color='b') # , linestyle='None'
        legends.append(f'pupitre {pupitre_dict[site][args.key]}')
        df_pupitre.plot(x="t", y=pupitre_dict[site][args.key], ax=my_ax, marker="o", alpha=0.2, color='b')
        legends.append(f'Archive {channels_dict[args.key]}')
        df_archive.plot(x="t", y=channels_dict[args.key], alpha=0.2, color='b', ax=my_ax)
        legends.append(f'Archive {args.key}')
        df_archive.plot(x="t", y=args.key, label="Archive", alpha=0.2, color='r', ax=my_ax)
        plt.title(channels_dict[args.key])
        plt.title(msg)
        plt.legend(labels=legends)
        plt.grid()
        plt.show()
        plt.close()

        print('\nLag correlation: pupitre/pigbrother overview')
        print("t0 (overview):", df_overview["timestamp"].iloc[0])
        print("t0 (pupitre):", df_pupitre["timestamp"].iloc[0])

        ts_overview_field = channels_dict[args.key]
        ts_overview = df_overview.loc[:,["timestamp", ts_overview_field]]
        ts_overview.set_index('timestamp', inplace=True)
        ts_overview = ts_overview.iloc[:,0]
        ts_overview = ts_overview - ts_overview.min()
        
        # find the biggest U within the shortest time in 1st Overview ???

        ts_overview_index = ts_overview.index.to_list()
        ostart = ochanges[0][1] - 2 
        oend = ochanges[0][2] + 2
        otstart = ts_overview_index[ostart]
        otend = ts_overview_index[oend]
        print(f"ts_overview overview range: [{ostart}, {oend}]-> [{otstart}, {otend}]")

        ts_pupitre_field = pupitre_dict[site][args.key]
        ts_pupitre = df_pupitre.loc[:,["timestamp",ts_pupitre_field]]
        ts_pupitre.set_index('timestamp', inplace=True)
        ts_pupitre = ts_pupitre.iloc[:,0]

        ts_pupitre_index = ts_pupitre.index.to_list()
        pstart = ts_pupitre.index.get_indexer([pd.Timestamp(otstart)], method='nearest')
        pend = ts_pupitre.index.get_indexer([pd.Timestamp(otend)], method='nearest')
        ptstart = ts_pupitre.index[pstart[0]]
        ptend = ts_pupitre.index[pend[0]]
        print(f"ts_pupitre timestamp range:  [{pstart}, {pend}] -> [{ptstart}, {ptend}]")

        # resample to make sure that timeseries share the same index
        ts_pupitre_resampled = ts_pupitre.resample('1s', origin=ts_pupitre_index[0]).asfreq()
        
        # Interpolate missing values (optional, depending on your use case)
        ts_pupitre_resampled = ts_pupitre_resampled.interpolate(method='linear')
        pstart = ts_pupitre_resampled.index.get_indexer([pd.Timestamp(otstart)], method='nearest')
        pend = ts_pupitre_resampled.index.get_indexer([pd.Timestamp(otend)], method='nearest')
        ptstart = ts_pupitre_resampled.index[pstart[0]]
        ptend = ts_pupitre_resampled.index[pend[0]]
        if args.debug:
            print('pupitre_resampled:', ts_pupitre_resampled.size, type(ts_pupitre_resampled))
            print(ts_pupitre_resampled.head())
            print(ts_pupitre_resampled.tail())
            my_ax=plt.gca()
            ts_pupitre_resampled.plot(style='.', color='b', ax=my_ax)
            ts_overview.plot(style='o', alpha=0.5, color='b', ax=my_ax)
            plt.title('after resampling at 1 Hz')
            plt.grid()
            plt.show()


        # pupitre data
        ts_pupitre_data = {
            "field": ts_pupitre_field,
            "df": ts_pupitre_resampled,
            "range" : {
                "start":  ptstart,
                "end": ptend
            }
        }

        # overview data
        ts_overview_data = {
            "field": ts_overview_field,
            "df": ts_overview,
            "range" : {
                "start": otstart,
                "end": otend
            }
        }
        lag = lag_correlation(ts_pupitre_data, ts_overview_data, show=args.show, save=args.save,)

        df_pupitre["timestamp"] = df_pupitre["timestamp"] - pd.to_timedelta(f"{lag}s")

        # plots
        msg = f'(sync with pigbrother {timeshift} - {lag.total_seconds()} s)'

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

        """
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
        """