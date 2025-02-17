import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..magnetdata import MagnetData

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
    time_trend_slice_lag.index = time_trend_slice_lag.index-time_shift
    print("after lag")
    print(time_trend_slice_lag.head())

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_series_slice, label=name_series, marker="+")
    plt.plot(trend_slice, label=name_trend, color='red', marker="*")
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

# To check

def pearson(
    Data: MagnetData,
    fields: list[str],
    save: bool = False,
    show: bool = False,
    debug: bool = False,
):
    """
    compute Pearson correlation for fields

    ref: https://www.kaggle.com/code/adepvenugopal/time-series-correlation-pearson-tlcc-dtw
    """
    from scipy import stats

    nFields = len(fields)
    if isinstance(Data, pd.DataFrame):
        for i in range(nFields):
            for j in range(i + 1, nFields):
                df = Data.getData(["t", fields[i], fields[j]])
                overall_pearson_r = df.corr().iloc[0, 1]
                print(f"Pandas computed Pearson r: {overall_pearson_r}")

                r, p = stats.pearsonr(df.dropna()[fields[i]], df.dropna()[fields[j]])
                print(f"Scipy computed Pearson r: {r} and p-value: {p}")

                # Compute rolling window synchrony
                f, ax = plt.subplots(figsize=(7, 3))
                df.rolling(window=30, center=True).median().plot(ax=ax)
                ax.set(xlabel="Time", ylabel="Pearson r")
                ax.set(title=f"Overall Pearson r = {np.round(overall_pearson_r,2)}")

                if save:
                    outputfile = f"{fields[i]}-{fields[j]}-pearson.png"
                    plt.savefig(f"{outputfile}.png", dpi=300)
                if show:
                    plt.show
                plt.close()

    else:
        raise RuntimeError(f"stats/pearson: {Data.FileName} not a panda dataframe")


def crosscorr(datax, datay, lag=0, wrap=False):
    """Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))


def tlcc(
    Data: MagnetData,
    xfield: str,
    yfield: str,
    save: bool = False,
    show: bool = False,
    debug: bool = False,
):

    d1 = Data.getData(xfield)
    d2 = Data.getData(yfield)
    seconds = 5
    fps = 30
    rs = [
        crosscorr(d1, d2, lag)
        for lag in range(-int(seconds * fps), int(seconds * fps + 1))
    ]
    offset = np.floor(len(rs) / 2) - np.argmax(rs)
    f, ax = plt.subplots(figsize=(14, 3))
    ax.plot(rs)
    ax.axvline(np.ceil(len(rs) / 2), color="k", linestyle="--", label="Center")
    ax.axvline(np.argmax(rs), color="r", linestyle="--", label="Peak synchrony")
    # ylim=[0.1, 0.31],
    # xlim=[0, 301],
    ax.set(
        title=f"Offset = {offset} frames\n{xfield} leads <> {yfield} leads",
        xlabel="Offset",
        ylabel="Pearson r",
    )
    # ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
    # ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150])
    plt.legend()

    if save:
        pfile = f"{xfield}-{yfield}-TLCC"
        plt.savefig(f"{pfile}.png", dpi=300)
    if show:
        plt.show()
    plt.close()


def wtlcc(
    Data: MagnetData,
    xfield: str,
    yfield: str,
    save: bool = False,
    show: bool = False,
    debug: bool = False,
):
    import seaborn as sns

    df = Data.getData([xfield, yfield])
    # Windowed time lagged cross correlation
    seconds = 5
    fps = 30
    no_splits = 20
    samples_per_split = df.shape[0] / no_splits
    rss = []
    for t in range(0, no_splits):
        d1 = df[xfield].loc[(t) * samples_per_split : (t + 1) * samples_per_split]
        d2 = df[yfield].loc[(t) * samples_per_split : (t + 1) * samples_per_split]
        rs = [
            crosscorr(d1, d2, lag)
            for lag in range(-int(seconds * fps), int(seconds * fps + 1))
        ]
        rss.append(rs)
    rss = pd.DataFrame(rss)
    f, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(rss, cmap="RdBu_r", ax=ax)
    ax.set(
        title=f"Windowed Time Lagged Cross Correlation",
        xlim=[0, 301],
        xlabel="Offset",
        ylabel="Window epochs",
    )
    ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
    ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150])

    if save:
        pfile = f"{xfield}-{yfield}-WTLCC"
        plt.savefig(f"{pfile}.png", dpi=300)
    if show:
        plt.show()
    plt.close()


def rwtlcc(
    Data: MagnetData,
    xfield: str,
    yfield: str,
    save: bool = False,
    show: bool = False,
    debug: bool = False,
):
    import seaborn as sns

    df = Data.getData([xfield, yfield])
    # Rolling window time lagged cross correlation
    seconds = 5
    fps = 30
    window_size = 300  # samples
    t_start = 0
    t_end = t_start + window_size
    step_size = 30
    rss = []
    while t_end < 5400:
        d1 = df[xfield].iloc[t_start:t_end]
        d2 = df[yfield].iloc[t_start:t_end]
        rs = [
            crosscorr(d1, d2, lag, wrap=False)
            for lag in range(-int(seconds * fps), int(seconds * fps + 1))
        ]
        rss.append(rs)
        t_start = t_start + step_size
        t_end = t_end + step_size
    rss = pd.DataFrame(rss)

    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(rss, cmap="RdBu_r", ax=ax)
    ax.set(
        title=f"Rolling Windowed Time Lagged Cross Correlation",
        xlim=[0, 301],
        xlabel="Offset",
        ylabel="Epochs",
    )
    ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
    ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150])

    if save:
        pfile = f"{xfield}-{yfield}-RWTLCC"
        plt.savefig(f"{pfile}.png", dpi=300)
    if show:
        plt.show()

    plt.close()
