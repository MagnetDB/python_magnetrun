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

from ..magnetdata import MagnetData
from ..utils.sequence import list_sequence, list_duplicates_of

def stats(Data: MagnetData):
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
        v_min = float(Data.getData(f).min())
        v_max = float(Data.getData(f).max())
        v_mean = float(Data.getData(f).mean())
        v_var = float(Data.getData(f).var())
        v_median = float(Data.getData(f).median())
        v_mode = float(Data.getData(f).mode())
        table = ["%s[%s]" % (f,unit), v_mean, v_max, v_min, math.sqrt(v_var), v_median, v_mode]
    tables.append(table)

    print(tabulate(tables, headers, tablefmt="simple"), "\n")
    return 0

def plateaus(Data: MagnetData, twindows=6, threshold=1.e-4, b_threshold=1.e-3, duration=5, show=False, save=True, debug=False):
    """get plateaus, pics from the actual run"""
    print('plateaus: show={show}, save={save}, debug={debug}')

    if show or save:
        ax = plt.gca()

    # TODO:
    # pass b_thresold as input param
    # b_threshold = 1.e-3
    
    if debug:
        print("Search for plateaux:", "Type:", Data.Type)

    B_min = float(Data.getData('Field').min())
    B_max = float(Data.getData('Field').max())
    B_mean = float(Data.getData('Field').mean())
    B_var = float(Data.getData('Field').var())

    Bz = Data.getData('Field')
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

    if show or save:
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
            if "Date" in Data.getKeys() and "Time" in Data.getKeys():
                tformat="%Y.%m.%d %H:%M:%S"
                start_date=Data.getData("Date").iloc[0]
                start_time=Data.getData("Time").iloc[0]
            
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
        start=Data.getData('Date').iloc[p[0]]
        start_time=Data.getData('Time').iloc[p[0]]
        end=Data.getData('Date').iloc[p[1]]
        end_time = Data.getData('Time').iloc[p[1]]

        t0 = datetime.datetime.strptime(start+" "+start_time, tformat)
        t1 = datetime.datetime.strptime(end+" "+end_time, tformat)
        dt = (t1-t0)

        # b0=Data.getData('Field').values.tolist()[p[0]]
        b0 = float(Data.getData('Field').iloc[p[0]])
        b1 = float(Data.getData('Field').iloc[p[1]])
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

