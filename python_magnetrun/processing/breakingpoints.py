"""
Find Changing Points in an magnet record (tdms format)

Use kernelCPD algo for performance

ex: python ./test-ruptures.py M9_Overview_240509-1634.tdms --group Tensions_Aimant --channel Interne6 --detect --algo kernelCPD --model rbf --pen 512 --min_size=200
"""

import pandas as pd
import ruptures as rpt

import matplotlib.pyplot as plt

# from: https://github.com/sztistvan/change_detection/blob/main/change_point_detection_chatgpt.ipynb
# https://stackoverflow.com/questions/47519626/using-numpy-scipy-to-identify-slope-changes-in-digital-signals

algo = {
    "Pelt": {"method": rpt.Pelt, "models": ["l1", "l2", "rbf"]},
    "Dynp": {
        "method": rpt.Dynp,
        "models": ["l1", "l2", "rbf", "linear", "normal", "ar"],
    },
    "Binseg": {
        "method": rpt.Binseg,
        "models": ["l1", "l2", "rbf", "linear", "normal", "ar"],
    },
    "Window": {
        "method": rpt.Window,
        "models": ["l1", "l2", "rbf", "linear", "normal", "ar"],
    },
    "BottomUp": {
        "method": rpt.BottomUp,
        "models": ["l1", "l2", "rbf", "linear", "normal", "ar"],
    },
    "kernelCPD": {"method": rpt.KernelCPD, "models": ["linear", "rbf", "cosine"]},
}


def detect_changes(
    time_series: pd.Series,
    algoname: str = "kernelCDP",
    model: str = "rbf",
    min_size: int = 1,
    jump: int = 10,
    pen: float = 2,
    n_bkps: int = 5,
) -> list[int]:
    """
    performs Change Point detection using Pelt algo

    algoname:
    mode:
    min_size:
    jump:

    see:
    model which is the cost function to use (one of: l1, l2, rbf)
    cost which is a custom cost function - should be an instance of BaseCost
    jump which is used to reduce the possible set of change point indexes
    min_size is the minimum number of samples between two change points
    """

    # Convert time series to a numpy array
    signal = time_series.values

    # Perform change point detection using the Pelt algorithm
    # algo = rpt.Pelt(model="rbf", min_size=1, jump=10).fit(signal)
    # is gamma usefull - see https://centre-borelli.github.io/ruptures-docs/code-reference/detection/kernelcpd-reference/
    if algoname == "kernelCDP":
        rpt_algo = rpt.kernelCDP(
            kernel=model, min_size=min_size
        )  # , params={'gamma': 0.001})
    elif algoname == "Window":
        rpt_algo = rpt.Window(model, width=min_size, jump=jump)
    else:
        rpt_algo = algo[algoname]["method"](model, min_size=min_size, jump=jump)

    rpt_algo.fit(signal)
    if algoname in ["Pelt"]:
        result = rpt_algo.predict(pen=pen)
    if algoname in ["kernelCPD"]:
        result = rpt_algo.predict(pen=pen)
    else:
        result = rpt_algo.predict(n_bkps=n_bkps)

    # remove location if equal to len(signal)
    change_points = [i for i in result if i < len(signal)]

    """
    # display
    rpt.display(signal, bkps, result)
    plt.show()
    """

    # Return the list of change point locations
    return change_points


def plot_changes(
    signal: pd.Series,
    changes: list[int],
    method: str,
    name: str,
    show: bool = False,
    ax=None,
):
    """
    display changes
    """

    print(f"plot_changes: method={method}, name={name}, show={show}", flush=True)

    # Plot the time series

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(signal, label=name)

    # Plot the abrupt changes
    # for i in changes:
    #     ax.axvline(signal.index[i], color="r")
    indices = signal.index.to_list()
    selected_index = []
    for i in changes:
        selected_index.append(indices[i])
    outliers = signal[signal.index.isin(selected_index)]
    outliers.plot(ax=ax, marker="+", linestyle="none", label="bkpts")

    # Add labels, grid, and legend
    ax.set_xlabel("Time")
    ax.set_ylabel(name)
    ax.set_title(f"Time Series with Abrupt Changes ({method})")
    ax.legend()
    ax.grid(True)

    # Show the plot
    if show:
        plt.show()
    else:
        print(f"plot_changes: savefig to {name}-changes-matplotlib.png")
        plt.savefig(f"{name}-changes-matplotlib.png", dpi=300)
    plt.close()


def breakingpoints(
    ts: pd.Series,
    channel: str,
    algo: str = "kernelCDP",
    model: str = "rbf",
    min_size: int = 1,
    jump: int = 10,
    pen: float = 2,
    n_bkps: int = 5,
    ax=None,
    save: bool = False,
):
    # display algo
    print(f"algo: {algo}")
    print(f"model: {model}")
    print(f"min_size: {min_size}")
    print(f"jump: {jump}")
    print(f"pen: {pen}", flush=True)

    # Detect abrupt changes
    print(
        f"Detect Changing point for {channel}: show={not save}",
        flush=True,
    )
    changes = detect_changes(
        ts,
        algo,
        model,
        min_size,
        jump,
        pen,
        n_bkps,
    )
    print(f"Changes detected: {len(changes)}", flush=True)

    method = f"{algo}-{model}-pen{pen}"
    print(f"plot: Changing points: method={method}")
    plot_changes(
        ts,
        changes,
        method,
        channel,
        show=(not save),
        ax=ax,
    )
    # legends.append("bkpts")
