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


def list_sequence(lst, seq):
    """Return sequences of seq in lst"""
    sequences = []
    count = 0
    len_seq = len(seq)
    upper_bound = len(lst)-len_seq+1
    for i in range(upper_bound):
        if lst[i:i+len_seq] == seq:
            count += 1
            sequences.append([i,i+len_seq])
    return sequences

# see:  https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list
#from collections import defaultdict

def list_duplicates_of(seq,item):
    """Return sequences of duplicate adjacent item in seq"""
    start_at = -1
    locs = []
    sequences = []
    start_index = -1
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            end_index = locs[-1]
            sequences.append([start_index, end_index])
            # print("break end_index=%d" % end_index)
            break
        else:
            if not locs:
                # seq=[loc,0]
                start_index = loc
                # print( "item=%d, start: %d" % (item, loc) )
            else:
                if (loc-locs[-1]) != 1:
                    end_index = locs[-1]
                    sequences.append([start_index, end_index])
                    start_index = loc
                    # print( "item=%d, end: %d, new_start: %d" % (item, locs[-1], loc) )
            locs.append(loc)
            start_at = loc
    return sequences #locs

class MagnetRun:
    """
    Magnet Run

    Site: name of the site
    Insert: list of the MagnetIDs composing the insert
    MagnetData: pandas dataframe or tdms file
    """

    def __init__(self, site="unknown", insert="", data=None):
        """default constructor"""
        self.Site = site
        self.Insert = insert
        self.MagnetData = data

        start_date = None
        try:
            if "Date" in self.MagnetData.getKeys() and "Time" in self.MagnetData.getKeys():
                start_date=self.MagnetData.getData("Date").iloc[0]
                start_time=self.MagnetData.getData("Time").iloc[0]
                end_date=self.MagnetData.getData("Date").iloc[-1]
                end_time = self.MagnetData.getData('Time').iloc[-1]

                tformat="%Y.%m.%d %H:%M:%S"
                t0 = datetime.datetime.strptime(start_date+" "+start_time, tformat)
                t1 = datetime.datetime.strptime(end_date+" "+end_time, tformat)
                dt = (t1-t0)
                duration = dt / datetime.timedelta(seconds=1)

                print("* Site: %s, Insert: %s" % (self.Site, self.Insert),
                      "MagnetData.Type: %d" % self.MagnetData.Type,
                      "start_date=%s" % start_date,
                      "start_time=%s" % start_time,
                      "duration=%g s" % duration)
                
        except:
            print("MagnetRun.__init__: trouble loading data")
            try:
                file_name = "%s_%s_%s-wrongdata.txt" % (self.Site, self.Insert,start_date)
                self.MagnetData.to_csv(file_name, sep=str('\t'), index=False, header=True)
            except:
                print("MagnetRun.__init__: trouble loading data - fail to save csv file")
                pass
            pass
            
    @classmethod
    def fromtxt(cls, site, filename):
        """create from a txt file"""
        with open(filename, 'r') as f:
            insert=f.readline().split()[-1]
            data = MagnetData.fromtxt(filename)

            if site == "M9":
                data.addData("IH", "IH = Idcct1 + Idcct2")
                data.addData("IB", "IB = Idcct3 + Idcct4")
            elif site in ["M8", "M10"]:
                data.addData("IH", "IH = Idcct3 + Idcct4")
                data.addData("IB", "IB = Idcct1 + Idcct2")
            # what about M1, M5 and M7???

        # print("magnetrun.fromtxt: data=", data)
        return cls(site, insert, data)

    @classmethod
    def fromcsv(cls, site, insert, filename):
        """create from a csv file"""
        data = MagnetData.fromcsv(filename)
        return cls(site, insert, data)

    @classmethod
    def fromStringIO(cls, site, name):
        """create from a stringIO"""
        from io import StringIO

        # try:
        ioname = StringIO(name)
        # TODO rework: get item 2 otherwise set to unknown
        insert = "Unknown"
        headers = ioname.readline().split()
        if len(headers) >=2:
            insert = headers[1]
        data = MagnetData.fromStringIO(name)
        # except:
        #      print("cannot read data for %s insert, %s site" % (insert, site) )
        #      fo = open("wrongdata.txt", "w", newline='\n')
        #      fo.write(ioname)
        #      fo.close()
        #      sys.exit(1)
        return cls(site, insert, data)

    def __repr__(self):
        return "%s(Site=%r, Insert=%r, MagnetData=%r)" % \
             (self.__class__.__name__,
              self.Site,
              self.Insert,
              self.MagnetData)

    def getSite(self):
        """returns Site"""
        return self.Site

    def getInsert(self):
        """returns Insert"""
        return self.Insert

    def setSite(self, site):
        """set Site"""
        self.Site = site

    def getType(self):
        """returns Data Type"""
        return self.MagnetData.Type

    def getMData(self):
        """return Magnet Data object"""
        return self.MagnetData

    def getData(self, key=""):
        """return Data"""
        return self.MagnetData.getData(key)

    def getKeys(self):
        """return list of Data keys"""
        return self.MagnetData.Keys

    def getDuration(self):
        """compute duration of the run in seconds"""
        duration = None
        if "Date" in self.MagnetData.getKeys() and "Time" in self.MagnetData.getKeys():
            start_date=self.MagnetData.getData("Date").iloc[0]
            start_time=self.MagnetData.getData("Time").iloc[0]
            end_date=self.MagnetData.getData("Date").iloc[-1]
            end_time = self.MagnetData.getData('Time').iloc[-1]

            tformat="%Y.%m.%d %H:%M:%S"
            t0 = datetime.datetime.strptime(start_date+" "+start_time, tformat)
            t1 = datetime.datetime.strptime(end_date+" "+end_time, tformat)
            dt = (t1-t0)
            duration = dt / datetime.timedelta(seconds=1)
        return duration
    
    def stats(self):
        """compute stats from the actual run"""

        # TODO:
        # add teb,... to list
        # add duration
        # add duration per Field above certain values
        # add \int Power over time

        from tabulate import tabulate
        # see https://github.com/astanin/python-tabulate for tablefmt

        print("Statistics:\n")
        tables = []
        headers = ["Name", "Mean", "Max", "Min", "Std", "Median", "Mode"]
        for (f,unit) in zip(['Field', 'Pmagnet', 'teb', 'debitbrut'],["T", "MW", "C","m\u00B3/h"]):
            v_min = float(self.MagnetData.getData(f).min())
            v_max = float(self.MagnetData.getData(f).max())
            v_mean = float(self.MagnetData.getData(f).mean())
            v_var = float(self.MagnetData.getData(f).var())
            v_median = float(self.MagnetData.getData(f).median())
            v_mode = float(self.MagnetData.getData(f).mode())
            table = ["%s[%s]" % (f,unit), v_mean, v_max, v_min, math.sqrt(v_var), v_median, v_mode]
            tables.append(table)

        print(tabulate(tables, headers, tablefmt="simple"), "\n")
        return 0

    def plateaus(self, twindows=6, threshold=1.e-4, b_threshold=1.e-3, duration=5, show=False, save=True, debug=False):
        """get plateaus, pics from the actual run"""

        # TODO:
        # pass b_thresold as input param
        # b_threshold = 1.e-3
        
        if debug:
            print("Search for plateaux:", "Type:", self.MagnetData.Type)

        B_min = float(self.MagnetData.getData('Field').min())
        B_max = float(self.MagnetData.getData('Field').max())
        B_mean = float(self.MagnetData.getData('Field').mean())
        B_var = float(self.MagnetData.getData('Field').var())

        Bz = self.MagnetData.getData('Field')
        regime = Bz.to_numpy()
        df_ = pd.DataFrame(regime)
        df_['regime']=pd.Series(regime)

        diff = np.diff(regime) # scale by B_max??
        df_['diff']=pd.Series(diff)

        ndiff = np.where(abs(diff) >= threshold, diff, 0)
        df_['ndiff']=pd.Series(ndiff)
        if debug:
            print("gradient: ", df_)

        # TODO:
        # check gradient:
        #     if 0 in between two 1 (or -1), 0 may be replaced by 1 or -1 depending on ndiff values
        #     same for small sequense of 0 (less than 2s)
        gradient = np.sign(df_["ndiff"].to_numpy())
        gradkey = 'gradient-%s' % 'Field'
        df_[gradkey] = pd.Series(gradient)

        # # Try to remove spikes
        # ref: https://ocefpaf.github.io/python4oceanographers/blog/2015/03/16/outlier_detection/
        
        df_['pandas'] = df_[gradkey].rolling(window=twindows, center=True).median()

        difference = np.abs(df_[gradkey] - df_['pandas'])
        outlier_idx = difference > threshold
        # print("median[%d]:" % df_[gradkey][outlier_idx].size, df_[gradkey][outlier_idx])

        kw = dict(marker='o', linestyle='none', color='g',label=str(threshold), legend=True)
        df_[gradkey][outlier_idx].plot(**kw)
        # not needed if center=True
        # df_['shifted\_pandas'] =  df_['pandas'].shift(periods=-twindows//2)
        df_.rename(columns={0:'Field'}, inplace=True)

        del df_['ndiff']
        del df_['diff']
        del df_['regime']
        # del df_['pandas']

        if show or save:
            ax = plt.gca()
            df_.plot(ax=ax, grid=True)

            if show:
                plt.show()

            if save:
                # imagefile = self.Site + "_" + self.Insert
                imagefile = self.Site
                start_date = ""
                start_time = ""
                if "Date" in self.MagnetData.getKeys() and "Time" in self.MagnetData.getKeys():
                    tformat="%Y.%m.%d %H:%M:%S"
                    start_date=self.MagnetData.getData("Date").iloc[0]
                    start_time=self.MagnetData.getData("Time").iloc[0]
            
                plt.savefig('%s_%s---%s.png' % (imagefile,str(start_date),str(start_time)) , dpi=300 )
                plt.close()

        # convert panda column to a list
        # print("df_:", df_.columns.values.tolist())
        B_list = df_['pandas'].values.tolist()

        from functools import partial
        regimes_in_source = partial(list_duplicates_of, B_list)
        if debug:
            for c in [1, 0, -1]:
                print(c, regimes_in_source(c))

        # # To get timedelta in mm or millseconds
        # time_d_min = time_d / datetime.timedelta(minutes=1)
        # time_d_ms  = time_d / datetime.timedelta(milliseconds=1)
        plateaux = regimes_in_source(0)
        print( "%s plateaus(thresold=%g): %d" % ('Field', threshold, len(plateaux)) )
        tformat="%Y.%m.%d %H:%M:%S"
        actual_plateaux = []
        for p in plateaux:
            start=self.MagnetData.getData('Date').iloc[p[0]]
            start_time=self.MagnetData.getData('Time').iloc[p[0]]
            end=self.MagnetData.getData('Date').iloc[p[1]]
            end_time = self.MagnetData.getData('Time').iloc[p[1]]

            t0 = datetime.datetime.strptime(start+" "+start_time, tformat)
            t1 = datetime.datetime.strptime(end+" "+end_time, tformat)
            dt = (t1-t0)

            # b0=self.MagnetData.getData('Field').values.tolist()[p[0]]
            b0 = float(self.MagnetData.getData('Field').iloc[p[0]])
            b1 = float(self.MagnetData.getData('Field').iloc[p[1]])
            if debug:
                print( "\t%s\t%s\t%8.6g\t%8.4g\t%8.4g" % (start_time, end_time, dt.total_seconds(), b0, b1) )

            # if (b1-b0)/b1 > b_thresold: reject plateau
            # if abs(b1) < b_thresold and abs(b0) < b_thresold: reject plateau
            if (dt / datetime.timedelta(seconds=1)) >= duration:
                if abs(b1) >= b_threshold and abs(b0) >= b_threshold:
                    actual_plateaux.append([start_time, end_time, dt.total_seconds(), b0, b1])

        print( "%s plateaus(threshold=%g, b_threshold=%g, duration>=%g s): %d over %d" %
               ('Field', threshold, b_threshold, duration, len(actual_plateaux), len(plateaux)) )
        tables = []
        for p in actual_plateaux:
            b_diff = abs(1. - p[3] / p[4])
            tables.append([ p[0], p[1], p[2], p[3], p[4], b_diff*100.])

        pics = list_sequence(B_list, [1.0,-1.0])
        print( " \n%s pics (aka sequence[1,-1]): %d" % ('Field', len(pics)) )
        pics = list_sequence(B_list, [1.0,0,-1.0,0,1.])
        print( " \n%s pics (aka sequence[1,0,-1,0,1]): %d" % ('Field', len(pics)) )

        # remove adjacent duplicate
        import itertools
        B_ = [x[0] for x in itertools.groupby(B_list)]
        if debug:
            print( "B_=", B_, B_.count(0))
        print( "%s commisionning ? (aka sequence [1.0,0,-1.0,0.0,-1.0]): %d" % ('Field', len(list_sequence(B_, [1.0,0,-1.0,0.0,-1.0]))) )
        print("\n\n")


        from tabulate import tabulate
        headers = ["start", "end", "duration", "B0[T]", "B1[T]", "\u0394B/B[%]" ]
        print( tabulate(tables, headers, tablefmt="simple"), "\n" )

        return 0

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
