import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # `magnetrun-processing` — Signal Filtering & Smoothing

    CLI entry point: **`magnetrun-processing`**

    This notebook demonstrates the two main signal-processing sub-commands:

    | Sub-command | What it does |
    |-------------|-------------|
    | `filter`    | Remove spike artefacts from a channel using a sliding-window threshold |
    | `smooth`    | Locally-weighted linear regression (Loess) in several flavours |

    The `lag` sub-command (cross-correlation lag detection) is not yet fully
    implemented in the library and is omitted here.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## File & channel selection
    """)
    return


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import python_magnetrun

    default_path = str(
        Path(python_magnetrun.__file__).parent.parent
        / "tests"
        / "data"
        / "sample_pupitre.txt"
    )
    return (default_path, Path, python_magnetrun)


@app.cell(hide_code=True)
def _(mo, default_path):
    file_input = mo.ui.text(
        value=default_path, label="Path to run file (.txt / .tdms)", full_width=True
    )
    file_input  # noqa: B018
    return (file_input,)


@app.cell(hide_code=True)
def _(mo):
    housing_input = mo.ui.dropdown(
        options=["M8", "M9", "M10", "unknown"],
        value="M9",
        label="Housing",
    )
    housing_input  # noqa: B018
    return (housing_input,)


@app.cell
def _(file_input, housing_input, mo):
    import os

    from python_magnetrun.MagnetRun import MagnetRun

    _path = file_input.value
    _ext = os.path.splitext(_path)[-1]
    _housing = housing_input.value

    mo.stop(
        not os.path.exists(_path),
        mo.callout(mo.md(f"File not found: `{_path}`"), kind="warn"),
    )

    match _ext:
        case ".txt":
            mrun = MagnetRun.fromtxt(housing=_housing, assembly="", filename=_path)
        case ".tdms":
            mrun = MagnetRun.fromtdms(housing=_housing, assembly="", filename=_path)
        case _:
            mo.stop(True, mo.callout(mo.md(f"Unsupported extension: `{_ext}`"), kind="danger"))

    mrun.setHousing(_housing)
    mrun  # noqa: B018
    return (mrun, MagnetRun, os)


@app.cell(hide_code=True)
def _(mo, mrun):
    _keys = mrun.getKeys()
    channel_input = mo.ui.dropdown(
        options=_keys,
        value=_keys[0] if _keys else "",
        label="Channel to process",
    )
    channel_input  # noqa: B018
    return (channel_input,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## `filter` — Spike removal
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The spike filter slides a window of length `w` over the signal and flags
    samples that deviate from the local mean by more than `threshold × std`.
    Flagged samples are replaced by linear interpolation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    threshold_input = mo.ui.slider(
        start=0.1, stop=5.0, step=0.1, value=0.5,
        label="Threshold (× local std)",
        show_value=True,
    )
    window_input = mo.ui.slider(
        start=3, stop=50, step=1, value=10,
        label="Window length",
        show_value=True,
    )
    mo.hstack([threshold_input, window_input], align="start")
    return (threshold_input, window_input)


@app.cell
def _(channel_input, mo, mrun, threshold_input, window_input):
    import numpy as np
    import plotly.graph_objects as go

    from python_magnetrun.processing.filters import filterpikes

    _key = channel_input.value
    _mdata = mrun.getMData()

    mo.stop(
        not _key,
        mo.callout(mo.md("Select a channel above."), kind="info"),
    )

    _df_orig = _mdata.extractData(keys=["t", _key]).copy()
    _x = _df_orig["t"].to_numpy()
    _y = _df_orig[_key].to_numpy()

    _filtered = filterpikes(
        mrun,
        _key,
        inplace=False,
        threshold=threshold_input.value,
        twindows=window_input.value,
        debug=False,
        show=False,
        filename="",
    )

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_x, y=_y, mode="lines", name="original", opacity=0.6))
    if _filtered is not None and hasattr(_filtered, "to_numpy"):
        _fig.add_trace(go.Scatter(x=_x, y=_filtered.to_numpy(), mode="lines", name="filtered"))
    _fig.update_layout(
        title=f"Spike filter — {_key}",
        xaxis_title="t (s)",
        yaxis_title=_key,
        height=400,
    )
    _fig  # noqa: B018
    return (np, go, filterpikes)


@app.cell(hide_code=True)
def _(mo, channel_input, threshold_input, window_input):
    mo.md(f"""
    ### Equivalent CLI command

    ```bash
    magnetrun-processing <file.txt> filter \\
        --threshold {threshold_input.value} \\
        --twindows {window_input.value} \\
        --keys {channel_input.value}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## `smooth` — Loess regression
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Three Loess (locally-weighted regression) variants are available:

    | Method | Parameter | Description |
    |--------|-----------|-------------|
    | `bell_kernel` | `tau` — bandwidth | Default; fast and robust |
    | `ag` | `f` — fraction, `iter` — robustness iterations | A. Gramfort implementation |
    | `statsmodel_sm` | `f`, `iter` | Statsmodels LOWESS |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    method_input = mo.ui.dropdown(
        options=["bell_kernel", "ag", "statsmodel_sm"],
        value="bell_kernel",
        label="Smoother method",
    )
    tau_input = mo.ui.slider(
        start=10, stop=2000, step=10, value=400,
        label="tau (bell_kernel / statsmodel bandwidth)",
        show_value=True,
    )
    f_input = mo.ui.slider(
        start=0.05, stop=0.9, step=0.05, value=0.25,
        label="f — fraction (ag / statsmodel_sm)",
        show_value=True,
    )
    iter_input = mo.ui.slider(
        start=1, stop=10, step=1, value=3,
        label="iter — robustness iterations (ag / statsmodel_sm)",
        show_value=True,
    )
    mo.vstack([method_input, tau_input, mo.hstack([f_input, iter_input])])
    return (method_input, tau_input, f_input, iter_input)


@app.cell
def _(channel_input, f_input, iter_input, method_input, mo, mrun, np, tau_input):
    import plotly.graph_objects as _go

    from python_magnetrun.processing.smoothers import (
        lowess_ag,
        lowess_bell_shape_kern,
        lowess_sm,
    )

    _key = channel_input.value
    _mdata = mrun.getMData()

    mo.stop(
        not _key,
        mo.callout(mo.md("Select a channel above."), kind="info"),
    )

    _df_s = _mdata.extractData(keys=["t", _key])
    _x = _df_s["t"].to_numpy()
    _y = _df_s[_key].to_numpy()

    _method = method_input.value
    try:
        if _method == "bell_kernel":
            _yest = lowess_bell_shape_kern(_x, _y, tau_input.value)
        elif _method == "ag":
            _yest = lowess_ag(_x, _y, f=f_input.value, iter=iter_input.value)
        else:
            _yest = lowess_sm(_x, _y, f=f_input.value, iter=iter_input.value)
        _smooth_ok = True
    except (ValueError, TypeError, np.linalg.LinAlgError) as _e:
        _smooth_ok = False
        _err_msg = str(_e)

    mo.stop(
        not _smooth_ok,
        mo.callout(mo.md(f"Smoother failed: `{_err_msg}`"), kind="danger"),
    )

    _fig2 = _go.Figure()
    _fig2.add_trace(_go.Scatter(x=_x, y=_y, mode="lines", name="original", opacity=0.4))
    _fig2.add_trace(_go.Scatter(x=_x, y=_yest, mode="lines", name=f"Loess ({_method})"))
    _fig2.update_layout(
        title=f"Loess smoothing — {_key} ({_method})",
        xaxis_title="t (s)",
        yaxis_title=_key,
        height=400,
    )
    _fig2  # noqa: B018
    return (lowess_bell_shape_kern, lowess_ag, lowess_sm)


@app.cell(hide_code=True)
def _(mo, channel_input, method_input, tau_input, f_input, iter_input):
    _method = method_input.value
    if _method == "bell_kernel":
        _params = str(tau_input.value)
    else:
        _params = f"{f_input.value};{iter_input.value}"

    mo.md(f"""
    ### Equivalent CLI command

    ```bash
    magnetrun-processing <file.txt> smooth \\
        --method {_method} \\
        --smooth_params "{_params}" \\
        --keys {channel_input.value} \\
        --show
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
