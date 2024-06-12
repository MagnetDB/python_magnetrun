import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..magnetdata import MagnetData


def lagged_correlation(df, target, key, t):
    """
    Compute lag correlation between target and key df column
    """

    lagged_correlation = df[target].corr(df[key].shift(+t))
    print("type(lagged_correlation):", type(lagged_correlation))
    print("lagged_correlation(t=%g):" % t, lagged_correlation)


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
