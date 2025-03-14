from inspect import Signature
import pandas as pd

from python_magnetrun.magnetdata import MagnetData
from python_magnetrun.signature import Signature


def piecewise_linear_approximation(
    serie: pd.Series, threshold: float = 0.1
) -> list[str]:
    """
    Perform piecewise linear approximation on a time series and assign a signature.

    Parameters:
    - series: pandas Serie with a 'value' column containing the time series data.
    - threshold: float, the threshold to account for signal noise.

    Returns:
    - signature: list of str.
    """

    # Initialize variables
    signature = []
    previous_value = serie.iloc[0]
    previous_trend = None

    # Iterate over the dataframe
    for i in range(1, len(serie)):
        current_value = serie.iloc[i]
        difference = current_value - previous_value

        if difference >= threshold:
            current_trend = "U"
        elif difference <= -threshold:
            current_trend = "D"
        else:
            current_trend = "P"
        # print(i, current_value, difference, threshold, current_trend)

        # Append the trend to the signature list
        signature.append(current_trend)

        # Update previous values
        previous_value = current_value
        previous_trend = current_trend

    # Add the signature column to the dataframe
    signature.append(previous_trend)

    return signature


def trends_df(
    df: pd.DataFrame,
    tkey: str,
    key: str,
    window: int = 1,
    threshold: float = 1.0e-3,
    filename: str = "",
    show: bool = False,
    save: bool = False,
    debug: bool = False,
) -> tuple:
    """
    Perform seasonal decomposition and piecewise linear approximation on a time series.

    NB: does not work properly if dt is not constant
    """
    from tabulate import tabulate
    from statsmodels.tsa.seasonal import seasonal_decompose
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    if debug:
        print(f"{key}: data({df[tkey].shape})")
        print(df.head())
        print(df.tail())

    stats = df[key].describe()
    describe_list = stats.reset_index().to_dict(orient="records")
    if debug:
        print(f"stats for {key}:")
        print(tabulate(describe_list, headers="keys", tablefmt="psql"))

    # Perform piecewise linear approximation
    # !! use index and not 'key' column !!
    result = seasonal_decompose(df[key], model="additive", period=window)

    fig = result.plot()
    # Add grid to all subplots
    fig.suptitle(f"Seasonal Decomposition of {filename}")
    for ax in fig.axes:
        ax.grid(True)

    if save:
        imagefile = f"{filename}-{key}"
        plt.savefig(f"{imagefile}_seasonal_decompose.png", dpi=300)
    if show:
        plt.show()
    plt.close()

    # Get signature in terms of Up, Plateau, Down
    trend_component = result.trend
    # Drop NaN values from trend slice
    trend_component = trend_component[~np.isnan(trend_component)]
    signature = piecewise_linear_approximation(trend_component, threshold)

    # get index of changes in signature
    changes = [0]
    for i in range(1, len(signature)):
        if signature[i] != signature[i - 1]:
            changes.append(i)

    regimes = [signature[i] for i in changes]
    times = [float(df[tkey].iloc[i]) for i in changes]
    values = [float(df[key].iloc[i]) for i in changes]

    if debug:
        print(f"regimes: {regimes}")
        print(f"{tkey}: {times}")
        print(f"{key}: {values}")

    # TODO for result.trend, result.seasonal and result.resid
    # make sure that trend and df share the same index tkey
    nindex = trend_component.index.values.astype(int)
    new_index = [float(df[tkey].iloc[i]) for i in nindex]
    # print("new_index: ", new_index, flush=True)
    trend_values = trend_component.to_numpy()
    ntrend_component = pd.Series(data=trend_values, index=new_index)
    trend_component = ntrend_component.copy()

    return (changes, regimes, times, values, trend_component)


def trends(
    mdata: MagnetData,
    tkey: str,
    key: str,
    window: int = 1,
    threshold: float = 1.0e-3,
    show: bool = False,
    save: bool = False,
    debug: bool = False,
) -> tuple:
    """
    Perform seasonal decomposition and piecewise linear approximation on a time series.

    NB: does not work properly if dt is not constant
    """
    from tabulate import tabulate
    from statsmodels.tsa.seasonal import seasonal_decompose
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    df = pd.DataFrame()

    (symbol, unit) = mdata.getUnitKey(key)

    dt = 1
    match mdata.Data:
        case pd.DataFrame():
            df = mdata.getData([tkey, key])
            t0 = mdata.Data.iloc[0]["timestamp"]
        case dict():
            (group, channel) = key.split("/")
            t0 = mdata.Groups[group][channel]["wf_start_time"]

            df = mdata.getData(key).copy()
            dt = mdata.Groups[group][channel]["wf_increment"]
            df[tkey] = df.index * dt
            # print(f"trends {mdata.FileName}: t0={t0}, dt={dt}, type={type(dt)}")
            # rename df key
            if debug:
                print("tdms data: ", df.head())
                print(f"rename {channel} to {key}")
            df.rename(columns={channel: key}, inplace=True)

    if debug:
        print(f"{key}: data({df[tkey].shape})")
        print(df.head())
        print(df.tail())

    filename = mdata.FileName
    f_extension = os.path.splitext(filename)[-1]
    file = os.path.basename(filename).replace(f_extension, "")
    (changes, regimes, times, values, trend_component) = trends_df(
        df, tkey, key, window, threshold, filename, show, save, debug
    )

    # Plot piecewise linear approximation
    my_ax = plt.gca()
    legends = [key]
    df.plot(x=tkey, y=key, alpha=0.2, ax=my_ax)
    legends.append("trend")
    trend_component.plot(marker=".", linestyle="None", ax=my_ax)
    for x in times:
        plt.axvline(x=x, color="red")
    my_ax.legend(labels=legends)
    my_ax.grid(True)
    plt.title(f"{file}: Decompose {key}")

    if save:
        imagefile = f"{filename.replace(f_extension,'')}-{key}"
        plt.savefig(f"{imagefile}_signature.png", dpi=300)
    if show:
        plt.show()
    plt.close()

    return (changes, regimes, times, values, trend_component)
