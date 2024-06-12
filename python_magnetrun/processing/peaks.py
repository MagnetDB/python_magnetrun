"""
Find peaks
"""

import matplotlib.pyplot as plt

# from: https://github.com/sztistvan/change_detection/blob/main/change_point_detection_chatgpt.ipynb
# https://stackoverflow.com/questions/47519626/using-numpy-scipy-to-identify-slope-changes-in-digital-signals


def detect_peaks(
    ts,
    max_values,
    group: str,
    channel: str,
    window_size: int,
    threshold: float,
    normalize: bool = False,
    save: bool = False,
    debug: bool = False,
):
    # use rolling() to calculate the rolling minimum
    print(
        f"Rolling windows for {channel}: window_size={window_size}, show={not save}",
        flush=True,
    )
    rolling_mean = ts.rolling(window_size).mean()
    print(f"rolling_mean: {rolling_mean.describe()}")
    rolling_std = ts.rolling(window_size).std()
    print(f"rolling_std: {rolling_std.describe()}")

    # find picks
    from scipy.signal import find_peaks, peak_prominences
    from numpy import argmin

    height = threshold
    if normalize:
        height /= max_values[channel]
    x = rolling_std.to_numpy()
    peaks, peaks_properties = find_peaks(x, height=height)
    if peaks.shape[0] != 0:
        print(f"peaks: {peaks.shape}")
        # print(f'peaks: {peaks.tolist()}')
        # print(f'peaks heights: {peaks_properties["peak_heights"]}, type={type(peaks_properties["peak_heights"])}')
        prominences = peak_prominences(x, peaks)[0]
        print(
            f"prominences:  min={prominences.min()}, mean={prominences.mean()}, max={prominences.max()}, std={prominences.std()}"
        )

    if debug:
        ax = plt.gca()
        ts.plot(ax=ax, label=channel)
        rolling_mean.plot(ax=ax, label="mean")
        rolling_std.plot(ax=ax, label="std")

        plt.title(f"{group}: Rolling window={window_size}")
        plt.legend()
        plt.grid()
        if not save:
            plt.show()
        plt.close()

        if peaks.shape[0] != 0:
            ax = plt.gca()
            plt.title(f"{group}: Peaks for std with a {threshold} threshold")
            plt.plot(peaks, x[peaks], "o", label="peaks")
            plt.plot(x, label="std")
            plt.grid()
            plt.legend()
            plt.show()
            plt.close()

    # how to print peaks on original plot??
    ax = plt.gca()
    ts.plot(ax=ax)

    # get index of outliers
    indices = ts.index.to_list()
    selected_index = []
    for i in peaks.tolist():
        selected_index.append(indices[i])

    outliers = ts[ts.index.isin(selected_index)]
    outliers.plot(ax=ax, marker="o", linestyle="none", label="peaks")
    plt.show()
    plt.close()

    return outliers

    """
    # print(f'outliers: {outliers}')
    # print(f'outliers: {outliers.to_list()}')
    outliers.plot(ax=ax, marker="o", linestyle="none", label="peaks")
    legends.append("peaks")
    """
