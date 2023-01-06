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

# TODO use MagnetData instead of df
def plot_vs_time(df, items, show: bool=False):
    print(f"plot_vs_time: items={items}")
    keys = df.columns.values.tolist()
    
    ax = plt.gca()
    # loop over key
    for key in items:
        if key in keys:
            df.plot(x='Time', y=key, grid=True, ax=ax)
        else:
            print(f"unknown key: {key}")
            print(f"valid keys: {keys}")
            sys.exit(1)
    if show:
        plt.show()
    else:
        imagefile = "Fields" # input_file.replace(".txt", "")
        plt.savefig(f'{imagefile}_vs_time.png', dpi=300 )
    plt.close()
    
def plot_key_vs_key(df, pairs, show: bool=False):
    keys = df.columns.values.tolist()
    for pair in pairs:
        print(f"pair={pair}")
        ax = plt.gca()
        #print("pair=", pair, " type=", type(pair))
        items = pair.split('-')
        if len(items) != 2:
            print(f"invalid pair of keys: {pair}")
            sys.exit(1)
        key1= items[0]
        key2 =items[1]
        if key1 in keys and key2 in keys:
            df.plot(x=key1, y=key2,kind='scatter',color='red', grid=True, ax=ax) # on graph per pair
        else:
            print(f"unknown pair of keys: {pair}")
            print(f"valid keys: {keys}")
            sys.exit(1)
        if show:
           plt.show()
        else:
            plt.savefig(f'{key1}_vs_{key2}.png', dpi=300 )
        plt.close()

# TODO use MagnetData instead of files
def plot_files(input_files: list, key1: str, key2: str, from_i: int=0, to_i = None, show: bool=False, debug: bool=False):
    if debug:
        print(f'input_files: {input_files}')

    # Import Dataset
    ax = plt.gca()
    colormap = cm.viridis
    if to_i is None:
        colorlist = [colors.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, len(input_files))]
    else:
        colorlist = [colors.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, len(to_i-from_i))]


    df_f = []
    legends = []
    for i, f in enumerate(input_files):
        if i <= from_i:
            continue
        elif not to_i is None:
            if i >= to_i:
                break;
        else:
            try:
                if f.endswith(".txt"):
                    _df = pd.read_csv(f, sep='\s+', engine='python', skiprows=1)
                    df_f.append(_df)
                    keys = _df.columns.values.tolist()
                    if show:
                        if key1 in keys and key2 in keys:
                            lname = f.replace("_","-")
                            lname = lname.replace(".txt","")
                            lname = lname.split('/')
                            legends.append(f'{lname[-1]}')
                            # print(f'rename Flow1 to {lname[-1]}')
                            _df.plot.scatter(x=key1, y=key2, grid=True, label=f'{lname[-1]}', color=colorlist[i], ax=ax)
                            # print(f'tttut')
                else:
                    df_f.append(pd.read_csv(f, sep="str(',')", engine='python', skiprows=0))
            except:
                print(f'load_files: failed to load {f} with pandas')

            # print(f'load_files: {f}')

    # ax.legend()
    plt.legend(loc='best')
    if show:
        plt.show()
    else:
        print('save to file - to be implemented')
        plt.savefig(f'files-{key1}_vs_{key2}.png', dpi=300 )
    plt.close()
