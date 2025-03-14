"""
Smoothers

* Savitzky-Golay algorithm:
* Locally Weighted Linear Regression (Loess)

see:
https://fr.wikipedia.org/wiki/Algorithme_de_Savitzky-Golay
https://xavierbourretsicotte.github.io/loess.html
"""

import os

from math import ceil
import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt
import pandas as pd

import statsmodels.api as sm

from ..MagnetRun import MagnetRun


def savgol(y: np.array, window: int, polyorder: int = 3, deriv=0) -> np.array:
    """smooth signal using Savitzky-Golay algorithm for derivative of given order

    :param y: signal to be smoothed
    :type y: np.array
    :param window: size of the window
    :type window: float
    :param polyorder: polynomial order of the approx, defaults to 3
    :type polyorder: int, optional
    :param deriv: order of the derivative, defaults to 0
    :type deriv: int, optional
    :return: smooth signal
    :rtype: np.array
    """
    from scipy.signal import savgol_filter

    return savgol_filter(y, window_length=window, polyorder=polyorder, deriv=deriv)


# Defining the bell shaped kernel function - used for plotting later on
def kernel_function(xi, x0, tau: float = 0.005) -> np.array:
    """compute bell kernel

    :param xi: _description_
    :type xi: _type_
    :param x0: _description_
    :type x0: _type_
    :param tau: _description_, defaults to 0.005
    :type tau: float, optional
    :return: _description_
    :rtype: np.array
    """
    return np.exp(-((xi - x0) ** 2) / (2 * tau))


def lowess_bell_shape_kern(x, y, tau: float = 0.005) -> np.array:
    """lowess_bell_shape_kern(x, y, tau = .005) -> yest

    Locally weighted regression: fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The kernel function is the bell shaped function with parameter tau. Larger tau will result in a
    smoother curve.

        :param x: _description_
    :type x: _type_
    :param y: _description_
    :type y: _type_
    :param tau: _description_, defaults to 0.005
    :type tau: float, optional
    :return: _description_
    :rtype: np.array
    """
    n = len(x)
    yest = np.zeros(n)

    # Initializing all weights from the bell shape kernel function
    w = np.array([np.exp(-((x - x[i]) ** 2) / (2 * tau)) for i in range(n)])

    # Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array(
            [
                [np.sum(weights), np.sum(weights * x)],
                [np.sum(weights * x), np.sum(weights * x * x)],
            ]
        )
        theta = linalg.solve(A, b)
        yest[i] = theta[0] + theta[1] * x[i]

    return yest


def lowess_ag(x, y, f: float = 2.0 / 3.0, iter: int = 3) -> np.array:
    """lowess(x, y, f=2./3., iter=3) -> yest

    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.

    :param x: _description_
    :type x: _type_
    :param y: _description_
    :type y: _type_
    :param f: _description_, defaults to 2.0/3.0
    :type f: _type_, optional
    :param iter: _description_, defaults to 3
    :type iter: int, optional
    :return: _description_
    :rtype: np.array
    """
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w**3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)

    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array(
                [
                    [np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)],
                ]
            )
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

            residuals = y - yest
            s = np.median(np.abs(residuals))
            delta = np.clip(residuals / (6.0 * s), -1, 1)
            delta = (1 - delta**2) ** 2

    return yest


def lowess_sm(x, y, f: float = 1.0 / 3.0, iter: int = 3) -> np.array:
    """_summary_

    :param x: _description_
    :type x: _type_
    :param y: _description_
    :type y: _type_
    :param f: _description_, defaults to 1.0/3.0
    :type f: _type_, optional
    :param iter: _description_, defaults to 3
    :type iter: int, optional
    :return: _description_
    :rtype: np.array

    ref:
    https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
    """
    lowess_sm_ = sm.nonparametric.lowess
    return lowess_sm_(y, x, f, iter, return_sorted=False)


def smooth(
    mrun: MagnetRun,
    key: str,
    inplace: bool,
    tau: float,
    debug: bool,
    show: bool,
    input_file: str,
):
    """Smooth key column data from mrun dataset

    :param mrun: _description_
    :type mrun: _type_
    :param key: _description_
    :type key: _type_
    :param inplace: _description_
    :type inplace: _type_
    :param tau: _description_
    :type tau: _type_
    :param debug: _description_
    :type debug: _type_
    :param show: _description_
    :type show: _type_
    :param input_file: _description_
    :type input_file: _type_
    :return: _description_
    :rtype: _type_
    """

    filteredkey = f"filtered{key}"
    print(f"smooth: key={key}, filteredkey={filteredkey}", flush=True)
    df = mrun.getMData().getPandasData(key=None)
    Meanval = df[key].mean()

    # Initializing noisy non linear data
    x = df["t"].to_numpy()
    y = df[key].to_numpy()

    if debug:
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, facecolors="none", edgecolor="darkblue", label=key)

    yest_bell = lowess_bell_shape_kern(x, y, tau)
    if debug:
        plt.plot(x, yest_bell, color="red", label="Loess: bell shape kernel")
        x0 = (x[0] + x[40]) / 2.0
        plt.fill(
            x[:40],
            Meanval * kernel_function(x[:40], x0, tau),
            color="lime",
            alpha=0.5,
            label="Bell shape kernel",
        )
        if show:
            plt.show()
        else:
            f_extension = os.path.splitext(input_file)[-1]
            imagefile = input_file.replace(f_extension, f"-smoothed{key}.png")
            print(f"save to {imagefile}")
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
