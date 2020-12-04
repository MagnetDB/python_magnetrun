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
import magnetdata


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

    def __init__(self, site="M9", insert="", data=None):
        """default constructor"""
        self.Site = site
        self.Insert = insert
        self.MagnetData = data

        if "Date" in self.MagnetData.getKeys() and "Time" in self.MagnetData.getKeys():
            start_date=self.MagnetData.getData("Date").iloc[0]
            start_time=self.MagnetData.getData("Time").iloc[0]
            print("* Site: %s, Insert: %s" % (self.Site, self.Insert),
                  "start_time=", start_time, "start_date=", start_date)

        if self.MagnetData.Type == 0:
            if self.Site == "M9":
                self.MagnetData.addData("IH", "IH = Idcct1 + Idcct2")
                self.MagnetData.addData("IB", "IB = Idcct3 + Idcct4")
            elif self.Site in ["M8", "M10"]:
                self.MagnetData.addData("IH", "IH = Idcct3 + Idcct4")
                self.MagnetData.addData("IB", "IB = Idcct1 + Idcct2")

    @classmethod
    def fromtxt(cls, site, filename):
        """create from a txt file"""
        with open(filename, 'r') as f:
            insert=f.readline().split()[-1]
            data = magnetdata.MagnetData.fromtxt(filename)
        # print("magnetrun.fromtxt: data=", data)
        return cls(site, insert, data)

    @classmethod
    def fromcsv(cls, site, insert, filename):
        """create from a csv file"""
        data = magnetdata.MagnetData.fromcsv(filename)
        return cls(site, insert, data)

    @classmethod
    def fromStringIO(cls, site, name):
        """create from a stringIO"""
        from io import StringIO

        # try:
        ioname = StringIO(name)
        insert = ioname.readline().split()[-1]
        data = magnetdata.MagnetData.fromStringIO(name)
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

    def getData(self):
        """return Data"""
        return self.MagnetData.getData()

    def getKeys(self):
        """return list of Data keys"""
        return self.MagnetData.Keys

    def stats(self):
        """compute stats from the actual run"""

    def plateaus(self, thresold=1.e-4, duration=5, show=False, save=False, ax=None, debug=False):
        """get plateaus, pics from the actual run"""

        if debug:
            print("Search for plateaux:", "Type:", self.MagnetData.Type)

        if not ax and ( show or save):
            ax = plt.gca()

        B_min = float(self.MagnetData.getData('Field').min())
        B_max = float(self.MagnetData.getData('Field').max())
        B_mean = float(self.MagnetData.getData('Field').mean())
        B_var = float(self.MagnetData.getData('Field').var())

        Bz = self.MagnetData.getData('Field')
        regime = Bz.to_numpy()
        df_ = pd.DataFrame(regime)
        df_['regime']=pd.Series(regime)

        diff = np.diff(regime/B_max)
        df_['diff']=pd.Series(diff)

        ndiff = np.where(abs(diff) >= thresold, diff, 0)
        df_['ndiff']=pd.Series(ndiff)

        gradient = np.sign(df_["ndiff"].to_numpy())
        gradkey = 'gradient-%s' % 'Field'
        df_[gradkey]=pd.Series(gradient)
        df_.rename(columns={0:'Field'}, inplace=True)

        del df_['ndiff']
        del df_['diff']
        del df_['regime']

        if show or save:
            df_.plot(ax=ax, grid=True)

        # convert panda column to a list
        # print("df_:", df_.columns.values.tolist())
        B_list = df_[gradkey].values.tolist()

        from functools import partial
        regimes_in_source = partial(list_duplicates_of, B_list)
        if debug:
            for c in [1, 0, -1]:
                print(c, regimes_in_source(c))

        # print ("Bz=", self.MagnetData.getData('Field').values.tolist())
        print( "%s mean=%g T, max=%g T, min=%g T, std=%g T" % ('Field', B_mean, B_max, B_min, math.sqrt(B_var)) )

        # # To get timedelta in mm or millseconds
        # time_d_min = time_d / datetime.timedelta(minutes=1)
        # time_d_ms  = time_d / datetime.timedelta(milliseconds=1)
        plateaux = regimes_in_source(0)
        print( "%s plateaus(thresold=%g): %d" % ('Field', thresold, len(plateaux)) )
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

            b0=self.MagnetData.getData('Field').values.tolist()[p[0]]
            b1=self.MagnetData.getData('Field').iloc[p[1]]
            if debug:
                print( "\t%s\t%s\t%8.6g\t%8.4g\t%8.4g" % (start_time, end_time, dt.total_seconds(), b0, b1) )

            # if (b1-b0)/b1 > b_thresold: reject plateau
            # if abs(b1) < b_thresold and abs(b0) < b_thresold: reject plateau
            if (dt / datetime.timedelta(seconds=1)) >= duration:
                actual_plateaux.append([start_time, end_time, dt.total_seconds(), b0, b1])

        print( "%s plateaus(thresold=%g, duration>=%g s): %d over %d" % ('Field', thresold, duration, len(actual_plateaux), len(plateaux)) )
        print( "\tstart\t\tend\t\tduration[s]\tB0[T]\t\tB1[T]" )
        print( "\t========================================================================" )
        for p in actual_plateaux:
            b_diff = 0
            if b1 != 0:
                b_diff = abs(1.-b0/b1)
            else:
                b_diff = abs(1.-b1/b0)
            print( "\t%s\t%s\t%8.6g\t%8.4g\t%8.4g" % (p[0], p[1], p[2], p[3], p[4]), b_diff*100. )
        print( "\t========================================================================" )

        pics = list_sequence(B_list, [1.0,-1.0])
        print( " \n%s pics (aka sequence[1,-1]): %d" % ('Field', len(pics)) )

        # remove adjacent duplicate
        import itertools
        B_ = [x[0] for x in itertools.groupby(B_list)]
        if debug:
            print( "B_=", B_, B_.count(0))
        print( "%s commisionning ? (aka sequence [-1.0,0.0,-1.0]): %d" % ('Field', len(list_sequence(B_, [-1.0,0.0,-1.0]))) )
        print("\n\n")

        if show:
            plt.show()
        else:
            imagefile = self.Site + "_" + self.Insert
            if "Date" in self.MagnetData.getKeys() and "Time" in self.MagnetData.getKeys():
                tformat="%Y.%m.%d %H:%M:%S"
                start_date=self.MagnetData.getData("Date").iloc[0]
                start_time=self.MagnetData.getData("Time").iloc[0]

            plt.savefig('%s_%s.png' % (imagefile,str(start_date)) , dpi=300 )
            plt.close()

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
    parser.add_argument("--show", help="display graphs (default save in png format)", action='store_true')
    parser.add_argument("--list", help="list key in csv", action='store_true')
    parser.add_argument("--convert", help="convert file to csv", action='store_true')
    parser.add_argument("--stats", help="display stats and find regimes", action='store_true')
    parser.add_argument("--thresold", help="specify thresold for regime detection", type=float, default=1.e-4)
    parser.add_argument("--debug", help="acticate debug", action='store_true')
    args = parser.parse_args()

    # load df pandas from input_file
    # check extension
    f_extension=os.path.splitext(args.input_file)[-1]
    if f_extension != ".txt":
        print("so far only txt file support is implemented")
        sys.exit(0)

    mrun = MagnetRun.fromtxt(args.site, args.input_file)
    dkeys = mrun.getKeys()

    if args.list:
        print("Valid keys are:")
        for key in dkeys:
            print(key)
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
                print("invalid pair of keys: %s" % pair)
                sys.exit(1)
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
            print("output_time: feature not implemented for tdms format")
            sys.exit(0)

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
            print("output_time: feature not implemented for tdms format")
            sys.exit(0)

        timerange = args.output_timerange.split(";")

        file_name = args.input_file.replace(".txt", "")
        file_name = file_name + "_from" + str(timerange[0].replace(":", "-"))
        file_name = file_name + "_to" + str(timerange[1].replace(":", "-")) + ".csv"
        selected_df = mrun.getMData().extractTimeData(timerange)
        selected_df.to_csv(file_name, sep=str('\t'), index=False, header=True)

    if args.output_key:
        if mrun.getType() != 0:
            print("output_time: feature not implemented for tdms format")
            sys.exit(0)

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
            print("output_time: feature not implemented for tdms format")
            sys.exit(0)

        pairs = args.extract_pairkeys.split(';')
        for pair in pairs:
            items = pair.split('-')
            if len(items) != 2:
                print("invalid pair of keys: %s" % pair)
                sys.exit(1)
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
        mrun.plateaus(thresold=args.thresold, show=args.show, debug=args.debug)
