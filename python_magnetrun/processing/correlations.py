import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..magnetdata_base import MagnetDataBase

logger = logging.getLogger(__name__)


def _normalise_range(r: tuple | dict) -> dict:
    """Convert a legacy tuple range ``(start, end)`` to dict form ``{"start": …, "end": …}``."""
    if isinstance(r, tuple):
        return {"start": r[0], "end": r[1]}
    return r


def lag_correlation(
    data1: dict,
    data2: dict,
    show: bool = False,
    save: bool = False,
    debug: bool = False,
) -> pd.Timedelta:
    """Compute lag using scipy cross-correlation.

    .. deprecated::
        Use :func:`python_magnetrun.analysis.synchronization.lag_correlation` instead.

    Parameters
    ----------
    data1 : dict
        First series with keys ``"field"``, ``"df"``, ``"range"`` (dict ``{"start": …, "end": …}``).
    data2 : dict
        Second series (same format as *data1*).
    show : bool, optional
        Display diagnostic plots.
    save : bool, optional
        Save diagnostic plots.
    debug : bool, optional
        Print debug information.

    Returns
    -------
    pd.Timedelta
        Computed lag.
    """
    warnings.warn(
        "processing.correlations.lag_correlation() is deprecated; "
        "use analysis.synchronization.lag_correlation() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from ..analysis.synchronization import lag_correlation as _canonical

    d1 = dict(data1)
    d2 = dict(data2)
    d1["range"] = _normalise_range(d1["range"])
    d2["range"] = _normalise_range(d2["range"])
    return _canonical(d1, d2, show=show, save=save, debug=debug)


def compute_lag(
    tkey: str,
    df1_data: dict,
    df2_data: dict,
    show: bool = False,
    save: bool = False,
    debug: bool = False,
) -> pd.Timedelta:
    """Compute lag between two time series using cross-correlation.

    .. deprecated::
        Use :func:`python_magnetrun.analysis.synchronization.compute_lag` instead.
        Note: the canonical function expects ``"range"`` as a dict
        ``{"start": …, "end": …}``; tuple ranges are converted automatically
        by this shim.

    Parameters
    ----------
    tkey : str
        Name of the time/index column.
    df1_data : dict
        First series with keys ``"df"``, ``"field"``,
        ``"range"`` (tuple ``(start, end)`` or dict ``{"start": …, "end": …}``).
    df2_data : dict
        Second series (same format as *df1_data*).
    show : bool, optional
        Display diagnostic plots.
    save : bool, optional
        Save diagnostic plots.
    debug : bool, optional
        Print debug information.

    Returns
    -------
    pd.Timedelta
        Computed lag.
    """
    warnings.warn(
        "processing.correlations.compute_lag() is deprecated; "
        "use analysis.synchronization.compute_lag() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from ..analysis.synchronization import compute_lag as _canonical

    d1 = dict(df1_data)
    d2 = dict(df2_data)
    d1["range"] = _normalise_range(d1["range"])
    d2["range"] = _normalise_range(d2["range"])
    return _canonical(tkey, d1, d2, show=show, save=save, debug=debug)


# To check


def pearson(
    Data: MagnetDataBase,
    fields: list[str],
    save: bool = False,
    show: bool = False,
    debug: bool = False,
) -> None:
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
                logger.info(f"Pandas computed Pearson r: {overall_pearson_r}")

                r, p = stats.pearsonr(df.dropna()[fields[i]], df.dropna()[fields[j]])
                logger.info(f"Scipy computed Pearson r: {r} and p-value: {p}")

                # Compute rolling window synchrony
                f, ax = plt.subplots(figsize=(7, 3))
                df.rolling(window=30, center=True).median().plot(ax=ax)
                ax.set(xlabel="Time", ylabel="Pearson r")
                ax.set(title=f"Overall Pearson r = {np.round(overall_pearson_r, 2)}")

                if save:
                    outputfile = f"{fields[i]}-{fields[j]}-pearson.png"
                    plt.savefig(f"{outputfile}.png", dpi=300)
                if show:
                    plt.show()
                plt.close()

    else:
        raise RuntimeError(f"stats/pearson: {Data.FileName} not a panda dataframe")


def crosscorr(datax: pd.Series, datay: pd.Series, lag: int = 0, wrap: bool = False) -> float:
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
    Data: MagnetDataBase,
    xfield: str,
    yfield: str,
    save: bool = False,
    show: bool = False,
    debug: bool = False,
) -> None:
    d1 = Data.getData(xfield)
    d2 = Data.getData(yfield)
    seconds = 5
    fps = 30
    rs = [crosscorr(d1, d2, lag) for lag in range(-int(seconds * fps), int(seconds * fps + 1))]
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
    Data: MagnetDataBase,
    xfield: str,
    yfield: str,
    save: bool = False,
    show: bool = False,
    debug: bool = False,
) -> None:
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
        rs = [crosscorr(d1, d2, lag) for lag in range(-int(seconds * fps), int(seconds * fps + 1))]
        rss.append(rs)
    rss = pd.DataFrame(rss)
    f, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(rss, cmap="RdBu_r", ax=ax)
    ax.set(
        title="Windowed Time Lagged Cross Correlation",
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
    Data: MagnetDataBase,
    xfield: str,
    yfield: str,
    save: bool = False,
    show: bool = False,
    debug: bool = False,
) -> None:
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
        title="Rolling Windowed Time Lagged Cross Correlation",
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
