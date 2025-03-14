from __future__ import unicode_literals

import sys

import matplotlib
import numpy as np
import pandas as pd

# print("matplotlib=", matplotlib.rcParams.keys())
# matplotlib.rcParams['text.latex.unicode'] = True key not available
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

matplotlib.rcParams["text.usetex"] = True


# TODO add method to use MagnetData instead of df
# TODO add symbol and unit to plot using json dict file


def plot_vs_time(
    df: pd.DataFrame,
    items: list,
    show: bool = False,
    wd: str | None = None,
    ax=None,
    close: bool = False,
):
    print(f"plot_vs_time: items={items}, close={close}", flush=True)
    keys = df.columns.values.tolist()

    ax = plt.gca()
    # loop over key
    for key in items:
        if key in keys:
            df.plot(x="t", y=key, grid=True, ax=ax)
        else:
            raise RuntimeError(f"unknown key: {key}")
    if close:
        if show:
            plt.show()
        else:
            imagefile = "Fields"  # input_file.replace(".txt", "")
            filename = f"{imagefile}_vs_time.png"
            if wd is not None:
                filename = f"{wd}/{filename}"
            print(f"save to file - {filename}")
            plt.savefig(filename, dpi=300)
        plt.close()


def plot_key_vs_key(df, pairs, show: bool = False, wd: str | None = None):
    keys = df.columns.values.tolist()
    for pair in pairs:
        print(f"pair={pair}")
        ax = plt.gca()
        # print("pair=", pair, " type=", type(pair))
        items = pair.split("-")
        if len(items) != 2:
            print(f"invalid pair of keys: {pair}")
            sys.exit(1)
        key1 = items[0]
        key2 = items[1]
        if key1 in keys and key2 in keys:
            df.plot(
                x=key1, y=key2, kind="scatter", color="red", grid=True, ax=ax
            )  # on graph per pair
        else:
            print(f"unknown pair of keys: {pair}")
            print(f"valid keys: {keys}")
            sys.exit(1)
        if show:
            plt.show()
        else:
            filename = f"{key1}_vs_{key2}.png"
            if wd is not None:
                filename = f"{wd}/{filename}"
            print(f"save to file - {filename}")
            plt.savefig(filename, dpi=300)
        plt.close()


# TODO use MagnetData instead of files
def plot_df(
    name: str,
    df: pd.DataFrame,
    key1: str,
    key2: str,
    from_i: int = 0,
    to_i: int | None = None,
    fit: tuple | None = None,
    show: bool = False,
    debug: bool = False,
    wd: str | None = None,
):

    # Import Dataset
    ax = plt.gca()

    df.plot.scatter(
        x=key1,
        y=key2,
        grid=True,
        ax=ax,
    )

    # add fit if present
    if fit:
        (x, y) = fit
        ax.plot(x, y, color="red", linestyle="dashed", linewidth=2, label="fit")

    # add fit if present
    if fit:
        (x, y) = fit
        ax.plot(x, y, color="red", linestyle="dashed", linewidth=2, label="fit")

    plt.ylabel(key2)
    plt.xlabel(key1)
    plt.title(f"{name}: {key1} vs {key2}")

    if not show:
        filename = f"{name}-{key1}_vs_{key2}.png"
        if wd is not None:
            filename = f"{wd}/{filename}"
        # print(f"save to file - {filename}")
        plt.savefig(filename, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_files(
    name: str,
    input_files: list,
    key1: str,
    key2: str,
    from_i: int = 0,
    to_i: int | None = None,
    fit: tuple | None = None,
    show: bool = False,
    debug: bool = False,
    wd: str | None = None,
):
    if debug:
        print(f"plot_files: input_files={input_files}, key1={key1}, key2={key2}")

    # Import Dataset
    ax = plt.gca()
    colormap = cm.viridis
    colorlist = [
        colors.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, len(input_files))
    ]

    legends = []
    for i, f in enumerate(input_files):
        if i < from_i:
            # print(f"plot_files: skip {i}")
            continue
        elif to_i is not None:
            if i >= to_i:
                # print(f"plot_files: break {i}")
                break
        else:
            # print(f"plot_files: try to plot i={i}, f={f}, key1={key1}, key2={key2}")
            try:
                if f.endswith(".txt"):
                    _df = pd.read_csv(f, sep=r"\s+", engine="python", skiprows=1)
                    keys = _df.columns.values.tolist()
                    if key1 in keys and key2 in keys:
                        lname = f.replace("_", "-")
                        lname = lname.replace(".txt", "")
                        lname = lname.split("/")
                        legends.append(f"{lname[-1]}")
                        # print(f'rename Flow1 to {lname[-1]}')
                        _df.plot.scatter(
                            x=key1,
                            y=key2,
                            grid=True,
                            label=f"{lname[-1]}",
                            color=colorlist[i],
                            ax=ax,
                        )
                        # print(f"{f}: displayed")
                    else:
                        print(
                            f"{f}: no displayed - key1={key1} and key2={key2} not in keys"
                        )
            except:
                print(f"plot_files: failed to load {f} with pandas")

    # add fit if present
    if fit:
        (x, y) = fit
        ax.plot(x, y, color="red", linestyle="dashed", linewidth=2, label="fit")

    # add fit if present
    if fit:
        (x, y) = fit
        ax.plot(x, y, color="red", linestyle="dashed", linewidth=2, label="fit")

    # ax.legend()
    plt.legend(loc="best")
    plt.ylabel(key2)
    plt.xlabel(key1)

    if not show:
        filename = f"{name}-{key1}_vs_{key2}.png"
        if wd is not None:
            filename = f"{wd}/{filename}"
        # print(f"save to file - {filename}")
        plt.savefig(filename, dpi=300)
    else:
        plt.show()
    plt.close()
