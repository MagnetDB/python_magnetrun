"""
Locally Weighted Linear Regression (Loess)

see:
https://xavierbourretsicotte.github.io/loess.html
"""

import os
import sys

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

##from IPython.display import Image
##from IPython.display import display
# plt.style.use('seaborn-white')
## if jupyter: %matplotlib inline

import statsmodels.api as sm


def filterpikes(mrun, key, inplace, threshold, twindows, debug, show, input_file):
    """
    # filter spikes
    # see: https://ocefpaf.github.io/python4oceanographers/blog/2015/03/16/outlier_detection/
    """

    print("type(mrun):", type(mrun))
    df = mrun.getData()  # .extractData(keys)

    kw = dict(marker="o", linestyle="none", color="r", alpha=0.3)
    mean = df[key].mean()
    # print("%s=" % key, type(df[key]), df[key])

    if mean != 0:
        var = df[key].var()
        # print("mean(%s)=%g" % (key,mean), "std=%g" % math.sqrt(var) )
        filtered = (
            df[key]
            .rolling(window=twindows, center=True)
            .median()
            .fillna(method="bfill")
            .fillna(method="ffill")
        )
        filteredkey = "filtered%s" % key
        # print("** ", filteredkey, type(filtered))

        df = df.assign(**{filteredkey: filtered.values})
        # print("*** ", filteredkey, df[filteredkey])

        difference = np.abs((df[key] - filtered) / mean * 100)
        outlier_idx = difference > threshold

        if debug:
            ax = plt.gca()
            df[key].plot()
            # filtered.plot()
            df[filteredkey].plot()
            df[key][outlier_idx].plot(**kw)

            ax.legend()
            plt.grid(b=True)
            plt.title(mrun.getInsert().replace(r"_", r"\_") + ": Filtered %s" % key)
            if show:
                plt.show()
            else:
                f_extension = os.path.splitext(input_file)[-1]
                imagefile = input_file.replace(f_extension, "-filtered%s.png" % key)
                print("save to %s" % imagefile)
                plt.savefig(imagefile, dpi=300)
            plt.close()

        if inplace:
            # replace key by filtered ones
            del df[key]
            df.rename(columns={filteredkey: key}, inplace=True)

    return mrun


def lagged_correlation(df, target, key, t):
    """
    Compute lag correlation between target and key df column
    """

    lagged_correlation = df[target].corr(df[key].shift(+t))
    print("type(lagged_correlation):", type(lagged_correlation))
    print("lagged_correlation(t=%g):" % t, lagged_correlation)
