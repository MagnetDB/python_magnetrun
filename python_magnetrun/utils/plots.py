import logging

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
) -> None:
    logger.debug(f"plot_files: input_files={input_files}, key1={key1}, key2={key2}")

    ax = plt.gca()
    colormap = cm.get_cmap("viridis")
    colorlist = [colors.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, len(input_files))]

    for i, f in enumerate(input_files):
        if i < from_i:
            continue
        elif to_i is not None:
            if i >= to_i:
                break
        else:
            try:
                if f.endswith(".txt"):
                    _df = pd.read_csv(f, sep=r"\s+", engine="python", skiprows=1)
                    keys = _df.columns.values.tolist()
                    if key1 in keys and key2 in keys:
                        lname = f.replace("_", "-")
                        lname = lname.replace(".txt", "")
                        lname = lname.split("/")
                        _df.plot.scatter(
                            x=key1,
                            y=key2,
                            grid=True,
                            label=f"{lname[-1]}",
                            color=colorlist[i],
                            ax=ax,
                        )
                    else:
                        logger.warning(
                            f"{f}: no displayed - key1={key1} and key2={key2} not in keys"
                        )
            except (OSError, pd.errors.ParserError, KeyError, ValueError) as e:
                logger.error(f"plot_files: failed to load {f} with pandas: {e}")

    if fit:
        (x, y) = fit
        ax.plot(x, y, color="red", linestyle="dashed", linewidth=2, label="fit")

    ax.legend(loc="best")
    ax.set_ylabel(key2)
    ax.set_xlabel(key1)

    if not show:
        filename = f"{name}-{key1}_vs_{key2}.png"
        if wd is not None:
            filename = f"{wd}/{filename}"
        plt.savefig(filename, dpi=300)
    else:
        plt.show()
    plt.close()
