
from inspect import Signature
import pandas as pd

from python_magnetrun.magnetdata import MagnetData
from python_magnetrun.signature import Signature


def piecewise_linear_approximation(serie: pd.Series, threshold: float=0.1) -> list[str]:
    """
    Perform piecewise linear approximation on a time series and assign a signature.

    Parameters:
    - df: pandas DataFrame with a 'value' column containing the time series data.
    - threshold: float, the threshold to account for signal noise.

    Returns:
    - df: pandas DataFrame with an additional 'signature' column.
    """

    # Initialize variables
    signature = []
    previous_value = serie.iloc[0]
    previous_trend = None

    # Iterate over the dataframe
    for i in range(1, len(serie)):
        current_value = serie.iloc[i]
        difference = current_value - previous_value

        if difference > threshold:
            current_trend = "U"
        elif difference < -threshold:
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

def trends(mdata: MagnetData, tkey: str, key: str, window: int=1, threshold: float=1.e-3, show: bool=False, save: bool=False, debug: bool=False) -> tuple:
    """
    Perform seasonal decomposition and piecewise linear approximation on a time series.
    """    
    from tabulate import tabulate
    from statsmodels.tsa.seasonal import seasonal_decompose
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    (symbol, unit) = mdata.getUnitKey(key)
    
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
            # rename df key
            if debug:
                print("tdms data: ", df.head())
                print(f'rename {channel} to {key}')
            df.rename(columns={channel: key}, inplace=True)
    # print(df.head())

    stats = df[key].describe()
    if debug: print(f'stats for {key}:')
    describe_list = stats.reset_index().to_dict(orient='records')
    if debug: 
        print(tabulate(describe_list, headers='keys', tablefmt='psql'))

    # Perform piecewise linear approximation
    # !! use index and not 't' column !!
    result = seasonal_decompose(df[key], model='additive',period=window)

    filename = mdata.FileName
    f_extension = os.path.splitext(filename)[-1]
    file = os.path.basename(filename).replace(f_extension,'')

    fig = result.plot()
    # Add grid to all subplots
    fig.suptitle(f"Seasonal Decomposition of {file}")
    for ax in fig.axes:
        ax.grid(True)

    if save:
        f_extension = os.path.splitext(file)[-1]
        imagefile = f"{file.replace(f_extension,'')}-{key}"
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
        if signature[i] != signature[i-1]:
            changes.append(i)

    regimes = [signature[i] for i in changes]
    times = [float(df[tkey].iloc[i]) for i in changes]
    values = [float(df[key].iloc[i]) for i in changes]

    if debug:
        print(f"Signature: {regimes}")
        print(f"{tkey}: {times}")
        print(f"{key}: {values}")

    # Plot piecewise linear approximation
    my_ax = plt.gca()
    legends = [key]
    mdata.plotData(x=tkey, y=key, ax=my_ax, normalize=False)
    for x in times:
        plt.axvline(x=x, color="red")
    my_ax.legend(labels=legends)
    my_ax.grid(True)
    plt.title(f"Piecewise Linear Approximation of {file}: {key}")

    if save:
        f_extension = os.path.splitext(file)[-1]
        imagefile = f"{file.replace(f_extension,'')}-{key}"
        plt.savefig(f"{imagefile}_signature.png", dpi=300)
    if show:
        plt.show()
    plt.close()

    return (regimes, times, values, result)
