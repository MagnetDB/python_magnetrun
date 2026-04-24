from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import pandas as pd

from ..plotting.backend import get_backend

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
    backend: str = "matplotlib",
) -> None:
    logger.debug(f"plot_files: input_files={input_files}, key1={key1}, key2={key2}")

    b = get_backend(backend)
    colormap = cm.get_cmap("viridis")
    colorlist = [colors.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, len(input_files))]

    fig = b.subplots(1, share_x=False)

    for i, f in enumerate(input_files):
        if i < from_i:
            continue
        if to_i is not None and i >= to_i:
            break
        try:
            if f.endswith(".txt"):
                _df = pd.read_csv(f, sep=r"\s+", engine="python", skiprows=1)
                keys = _df.columns.values.tolist()
                if key1 in keys and key2 in keys:
                    lname = f.replace("_", "-").replace(".txt", "").split("/")[-1]
                    b.add_series(
                        fig,
                        0,
                        _df[key1].to_numpy(),
                        _df[key2].to_numpy(),
                        lname,
                        color=colorlist[i],
                        marker="o",
                        linestyle="none",
                        ylabel=key2,
                    )
                else:
                    logger.warning(
                        f"{f}: not displayed — key1={key1} and key2={key2} not in keys"
                    )
        except (OSError, pd.errors.ParserError, KeyError, ValueError) as e:
            logger.error(f"plot_files: failed to load {f}: {e}")

    if fit:
        x, y = fit
        b.add_series(
            fig, 0, np.asarray(x), np.asarray(y), "fit", color="#ff0000", linestyle="--"
        )

    # Set x-axis label via backend-specific duck-typing
    if hasattr(fig, "_magnetrun_axes"):  # MatplotlibBackend
        fig._magnetrun_axes[0].set_xlabel(key1)
    elif hasattr(fig, "update_xaxes"):   # PlotlyBackend
        fig.update_xaxes(title_text=key1)

    if not show:
        filename = f"{name}-{key1}_vs_{key2}.png"
        if wd is not None:
            filename = f"{wd}/{filename}"
        b.save(fig, Path(filename), dpi=300)
    else:
        b.show(fig)
