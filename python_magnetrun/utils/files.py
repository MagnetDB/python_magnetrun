from __future__ import unicode_literals
import sys
import argparse
import datetime
import pandas as pd
import freesteam as st
import numpy as np
import matplotlib
# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

# TODO use MagnetData instead of files

def concat_files(input_files: list: list, keys: list, debug: bool=False):
    if debug:
        print(f'input_files: {input_files}')

    df_f = []
    for i, f in enumerate(input_files):
        # if i > 40:
        #     break;
        try:
            if f.endswith(".txt"):
                _df = pd.read_csv(f, sep='\s+', engine='python', skiprows=1)
            else:
                _df = pd.read_csv(f, sep="str(',')", engine='python', skiprows=0)
                
            if keys:
                df_f.append(_df[keys])
            else:
                df_f.append(_df)
        except:
            print(f'load_files: failed to load {f} with pandas')

        # print(f'load_files: {f}')
    df = pd.concat(df_f , axis=0)
        
    # Drop empty columns
    df = df.loc[:, (df != 0.0).any(axis=0)]


    # Add a time column
    tformat="%Y.%m.%d %H:%M:%S"
    start_date=df["Date"].iloc[0]
    start_time=df["Time"].iloc[0]
    end_time=df["Time"].iloc[-1]
    print ("start_time=", start_time, "start_date=", start_date)

    t0 = datetime.datetime.strptime(df['Date'].iloc[0]+" "+df['Time'].iloc[0], tformat)
    try:
        df["t"] = df.apply(lambda row: (datetime.datetime.strptime(row.Date+" "+row.Time, tformat)-t0).total_seconds(), axis=1)
        # # del df['Date']
        # # del df['Time']
    except:
        pass

    return df

