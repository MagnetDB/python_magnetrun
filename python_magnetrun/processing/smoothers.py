"""
Locally Weighted Linear Regression (Loess)

see:
https://xavierbourretsicotte.github.io/loess.html
"""

import os
import sys

from math import ceil
import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt
import pandas as pd
##from IPython.display import Image
##from IPython.display import display
#plt.style.use('seaborn-white')
## if jupyter: %matplotlib inline

import statsmodels.api as sm

#Defining the bell shaped kernel function - used for plotting later on
def kernel_function(xi,x0,tau= .005):
    """compute bell kernel"""
    return np.exp( - (xi - x0)**2/(2*tau)   )

def lowess_bell_shape_kern(x, y, tau = .005):
    """lowess_bell_shape_kern(x, y, tau = .005) -> yest
    Locally weighted regression: fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The kernel function is the bell shaped function with parameter tau. Larger tau will result in a
    smoother curve.
    """
    n = len(x)
    yest = np.zeros(n)

    #Initializing all weights from the bell shape kernel function
    w = np.array([np.exp(- (x - x[i])**2/(2*tau)) for i in range(n)])

    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                      [np.sum(weights * x), np.sum(weights * x * x)]])
        theta = linalg.solve(A, b)
        yest[i] = theta[0] + theta[1] * x[i]

    return yest

def lowess_ag(x, y, f=2. / 3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)

    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

            residuals = y - yest
            s = np.median(np.abs(residuals))
            delta = np.clip(residuals / (6.0 * s), -1, 1)
            delta = (1 - delta ** 2) ** 2

    return yest

def lowess_sm(x, y, f=1. / 3., iter=3):
    """
    ref:
    https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
    """
    lowess_sm_ = sm.nonparametric.lowess
    return lowess_sm_(y, x, f, iter, return_sorted = False)

def smooth(mrun, key, inplace, tau, debug, show, input_file):
    """
    Smooth key column data from mrun dataset
    """

    filteredkey = "filtered%s" % key
    df = mrun.getData()
    Meanval = df[key].mean()

    #Initializing noisy non linear data
    x = df["t"].to_numpy()
    y = df[key].to_numpy()

    if debug:
        plt.figure(figsize=(10,6))
        plt.scatter(x,y, facecolors = 'none', edgecolor = 'darkblue', label = key)

    yest_bell = lowess_bell_shape_kern(x,y,tau)
    if debug:
        plt.plot(x,yest_bell,color = 'red', label = 'Loess: bell shape kernel')
        x0 = (x[0] + x[40])/2.
        plt.fill(x[:40],Meanval*kernel_function(x[:40],x0,tau), color = 'lime', alpha = .5, label = 'Bell shape kernel')
        if show:
            plt.show()
        else:
            f_extension = os.path.splitext(input_file)[-1]
            imagefile = input_file.replace(f_extension, "-smoothed%s.png" % key)
            print("save to %s" % imagefile)
            plt.savefig(imagefile, dpi=300)
        plt.close()

    if inplace:
        # replace key by filtered ones
        df[filteredkey] = pd.Series(yest_bell)
        del df[key]
        df.rename(columns={filteredkey: key}, inplace=True)
    # except:
    #     print("smooth: bell_shape_kern smoother failed")
    return mrun

