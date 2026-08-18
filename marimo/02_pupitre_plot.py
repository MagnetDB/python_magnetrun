import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 1 — Pupitre Data: Plotting

    Interactive time-series and key-vs-key plots using plotly.

    All plots use plotly so you can zoom, pan, and hover directly in the
    notebook.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Load a file
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
    return (default_path,)


@app.cell(hide_code=True)
def _(default_path, mo):
    file_input = mo.ui.text(
        value=default_path, label="Path to `.txt` file", full_width=True
    )
    housing_input = mo.ui.dropdown(
        options=["M9", "M10", "unknown"],
        value="M9",
        label="Housing",
    )
    mo.vstack([file_input, housing_input])
    return file_input, housing_input


@app.cell
def _(file_input, housing_input):
    from python_magnetrun.MagnetRun import load_mrun

    mrun = load_mrun(
        file_input.value,
        housing=housing_input.value,
        auto_resolve=False,
    )
    keys = mrun.getKeys()
    df = mrun.getDataFrame()
    return df, keys


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Time-series plot
    """)
    return


@app.cell(hide_code=True)
def _(keys, mo):
    channel_select = mo.ui.multiselect(
        options=keys,
        value=keys[:2] if len(keys) >= 2 else keys,
        label="Channels to plot",
    )
    channel_select  # noqa: B018
    return (channel_select,)


@app.cell
def _(channel_select, df, mo):
    import plotly.graph_objects as go

    _selected = channel_select.value
    if not _selected:
        mo.stop(True, mo.md("Select at least one channel."))

    fig_ts = go.Figure()
    for _key in _selected:
        if _key in df.columns and "t" in df.columns:
            fig_ts.add_trace(go.Scatter(x=df["t"], y=df[_key], mode="lines", name=_key))

    fig_ts.update_layout(
        title="Time series",
        xaxis_title="t [s]",
        yaxis_title="value",
        legend_title="Channel",
        hovermode="x unified",
    )
    fig_ts  # noqa: B018
    return (go,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Key-vs-key (X–Y) plot
    """)
    return


@app.cell(hide_code=True)
def _(keys, mo):
    x_select = mo.ui.dropdown(options=keys, value=keys[0], label="X axis")
    y_select = mo.ui.dropdown(
        options=keys,
        value=keys[1] if len(keys) > 1 else keys[0],
        label="Y axis",
    )
    mo.hstack([x_select, y_select])
    return x_select, y_select


@app.cell
def _(df, go, mo, x_select, y_select):
    x_key = x_select.value
    y_key = y_select.value

    if x_key not in df.columns or y_key not in df.columns:
        mo.stop(True, mo.md("Selected column not available."))

    fig_xy = go.Figure(
        go.Scatter(
            x=df[x_key],
            y=df[y_key],
            mode="markers+lines",
            marker=dict(size=4),
            name=f"{y_key} vs {x_key}",
        )
    )
    fig_xy.update_layout(
        title=f"{y_key} vs {x_key}",
        xaxis_title=x_key,
        yaxis_title=y_key,
        hovermode="closest",
    )
    fig_xy  # noqa: B018
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Overlay — multiple channels on one axes

    `plot_overlay` from `python_magnetrun.plotting` places all selected
    channels on a single axes with unit-aware labelling.

    When channels have **different physical units**, tick **Normalise** to
    divide each series by its absolute maximum so shapes can be compared
    on a common [−1, 1] scale.  A dimension-mismatch warning is emitted
    automatically when units are incompatible.
    """)
    return


@app.cell(hide_code=True)
def _(keys, mo):
    _non_time = [k for k in keys if k not in ("t", "timestamp", "Date", "Time")]
    overlay_channels = mo.ui.multiselect(
        options=_non_time,
        value=_non_time[:3] if len(_non_time) >= 3 else _non_time,
        label="Channels",
    )
    overlay_normalize = mo.ui.checkbox(label="Normalise (divide by max)")
    mo.hstack([overlay_channels, overlay_normalize], align="end")
    return (overlay_channels, overlay_normalize)


@app.cell
def _(df, mo, overlay_channels, overlay_normalize):
    from python_magnetrun.plotting import plot_overlay as _plot_overlay
    from python_magnetrun.plotting.backend import get_backend as _get_backend

    _selected = overlay_channels.value
    if not _selected:
        mo.stop(True, mo.md("Select at least one channel above."))

    _b = _get_backend("plotly")
    _fig_ov = _plot_overlay(
        df,
        fields=list(_selected),
        backend=_b,
        normalize=overlay_normalize.value,
        title="Overlay",
    )
    _b.finalize(_fig_ov)
    _fig_ov  # noqa: B018
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Stacked subplots — shared X axis

    `plot_subplots` stacks each channel in its own subplot with a shared
    time axis and per-subplot Y labels — ideal when channels have
    **different units** and normalisation is undesirable.

    Tick **Normalise** to rescale every subplot to [−1, 1] for visual
    shape comparison across channels.
    """)
    return


@app.cell(hide_code=True)
def _(keys, mo):
    _non_time2 = [k for k in keys if k not in ("t", "timestamp", "Date", "Time")]
    subplot_channels = mo.ui.multiselect(
        options=_non_time2,
        value=_non_time2[:3] if len(_non_time2) >= 3 else _non_time2,
        label="Channels",
    )
    subplot_normalize = mo.ui.checkbox(label="Normalise (divide by max)")
    mo.hstack([subplot_channels, subplot_normalize], align="end")
    return (subplot_channels, subplot_normalize)


@app.cell
def _(df, mo, subplot_channels, subplot_normalize):
    from python_magnetrun.plotting import plot_subplots as _plot_subplots
    from python_magnetrun.plotting.backend import get_backend as _get_backend2

    _selected2 = subplot_channels.value
    if not _selected2:
        mo.stop(True, mo.md("Select at least one channel above."))

    _b2 = _get_backend2("plotly")
    _fig_sp = _plot_subplots(
        df,
        fields=list(_selected2),
        backend=_b2,
        normalize=subplot_normalize.value,
        title="Stacked subplots",
    )
    _b2.finalize(_fig_sp)
    _fig_sp  # noqa: B018
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Notes

    - `df["t"]` is elapsed time in seconds since run start.
    - `df["timestamp"]` holds UTC-aware `datetime64` values.
    - All channels are in SI-compatible units as defined in the housing
      defs file (e.g. `Field` in Tesla, currents in Ampere).
    - `plot_overlay` / `plot_subplots` read `df.attrs["units"]` (attached by
      `mdata.getData()`) to build axis labels automatically.
    - Both functions accept `normalize=True` to rescale to [−1, 1] for
      shape comparison across channels with different units.
    - For large files, use `02b_downsampling.py` to reduce the number of
      points before plotting.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## NAS File Browser

    Discover and load a `.txt` pupitre file directly from the NAS.
    **Housing** is taken from the dropdown above; adjust it to switch magnet
    site.  Changing the housing or dates immediately refreshes the file list,
    and the plots below update as soon as a file is selected.

    Root search path: `MAGNETRUN_PUPITRE_DATA_DIR` (or `PUPITRE_DATADIR`);
    default `/mnt/LNCMIG-Data/records/srv-data-install`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import datetime as _dt

    _today = _dt.date.today()
    nas_start_date = mo.ui.date(value=_today - _dt.timedelta(days=30), label="From")
    nas_end_date = mo.ui.date(value=_today, label="To")
    mo.hstack([nas_start_date, nas_end_date], align="start")
    return (nas_end_date, nas_start_date)


@app.cell(hide_code=True)
def _(housing_input, mo, nas_end_date, nas_start_date):
    import glob as _glob
    from pathlib import Path as _Path

    from python_magnetrun.data_dirs import PUPITRE_DATA_DIR as _PUPITRE_DIR
    from python_magnetrun.utils.timestamps import parse_filename_timestamp as _parse_ts

    _housing = housing_input.value
    _site_dir = _Path(_PUPITRE_DIR) / _housing
    _start = nas_start_date.value
    _end = nas_end_date.value

    nas_file_input = None

    if not _site_dir.exists():
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    f"NAS directory **`{_site_dir}`** is not accessible.\n\n"
                    "Mount the NAS or set the `MAGNETRUN_PUPITRE_DATA_DIR`"
                    " environment variable."
                ),
                kind="warn",
            ),
        )

    _txts = sorted(_glob.glob(str(_site_dir / "**" / "*.txt"), recursive=True))
    _in_range = []
    for _f in _txts:
        _dt_f = _parse_ts(_f)
        if _dt_f is not None and _start <= _dt_f.date() <= _end:
            _in_range.append(_f)

    if not _in_range:
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    f"No `.txt` files found under `{_site_dir}` "
                    f"between **{_start}** and **{_end}**.\n\n"
                    "Widen the date range or choose a different housing."
                ),
                kind="info",
            ),
        )

    nas_file_input = mo.ui.dropdown(
        options={_Path(_f).name: _f for _f in _in_range},
        label="Pupitre file",
    )
    nas_file_input  # noqa: B018
    return (nas_file_input,)


@app.cell
def _(housing_input, mo, nas_file_input):
    from python_magnetrun.MagnetRun import load_mrun as _load_mrun

    nas_mrun = None
    nas_df = None
    nas_keys = []

    _no_file = nas_file_input is None or not getattr(nas_file_input, "value", None)
    mo.stop(
        _no_file,
        mo.callout(
            mo.md("Select a file from the browser above to load it."),
            kind="info",
        ),
    )

    nas_mrun = _load_mrun(
        nas_file_input.value,
        housing=housing_input.value,
        auto_resolve=False,
    )
    nas_df = nas_mrun.getDataFrame()
    nas_keys = nas_mrun.getKeys()
    return (nas_df, nas_keys, nas_mrun)


@app.cell
def _(mo, nas_keys, nas_mrun):
    nas_channel_select = mo.ui.multiselect(
        options=nas_keys,
        value=nas_keys[:2] if len(nas_keys) >= 2 else nas_keys,
        label="Channels to plot",
    )

    mo.stop(
        nas_mrun is None,
        mo.callout(
            mo.md("No NAS run loaded — select a file above."),
            kind="info",
        ),
    )
    nas_channel_select  # noqa: B018
    return (nas_channel_select,)


@app.cell
def _(go, mo, nas_channel_select, nas_df, nas_mrun):
    mo.stop(
        nas_mrun is None,
        mo.callout(mo.md("No NAS run loaded."), kind="info"),
    )

    _nas_selected = nas_channel_select.value
    if not _nas_selected:
        mo.stop(True, mo.md("Select at least one channel."))

    _nas_fig_ts = go.Figure()
    for _nas_key in _nas_selected:
        if _nas_key in nas_df.columns and "t" in nas_df.columns:
            _nas_fig_ts.add_trace(
                go.Scatter(
                    x=nas_df["t"], y=nas_df[_nas_key], mode="lines", name=_nas_key
                )
            )

    _nas_fig_ts.update_layout(
        title="Time series (NAS)",
        xaxis_title="t [s]",
        yaxis_title="value",
        legend_title="Channel",
        hovermode="x unified",
    )
    _nas_fig_ts  # noqa: B018
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### Key-vs-key (X–Y) plot
    """)
    return


@app.cell(hide_code=True)
def _(mo, nas_keys, nas_mrun):
    nas_x_select = mo.ui.dropdown(
        options=nas_keys or ["t"], value=(nas_keys or ["t"])[0], label="X axis"
    )
    nas_y_select = mo.ui.dropdown(
        options=nas_keys or ["t"],
        value=(
            (nas_keys or ["t"])[1]
            if len(nas_keys or ["t"]) > 1
            else (nas_keys or ["t"])[0]
        ),
        label="Y axis",
    )

    mo.stop(
        nas_mrun is None,
        mo.callout(mo.md("No NAS run loaded — select a file above."), kind="info"),
    )
    mo.hstack([nas_x_select, nas_y_select])
    return (nas_x_select, nas_y_select)


@app.cell
def _(go, mo, nas_df, nas_mrun, nas_x_select, nas_y_select):
    mo.stop(
        nas_mrun is None,
        mo.callout(mo.md("No NAS run loaded."), kind="info"),
    )

    _nas_xk = nas_x_select.value
    _nas_yk = nas_y_select.value

    if _nas_xk not in nas_df.columns or _nas_yk not in nas_df.columns:
        mo.stop(True, mo.md("Selected column not available."))

    _nas_fig_xy = go.Figure(
        go.Scatter(
            x=nas_df[_nas_xk],
            y=nas_df[_nas_yk],
            mode="markers+lines",
            marker=dict(size=4),
            name=f"{_nas_yk} vs {_nas_xk}",
        )
    )
    _nas_fig_xy.update_layout(
        title=f"{_nas_yk} vs {_nas_xk} (NAS)",
        xaxis_title=_nas_xk,
        yaxis_title=_nas_yk,
        hovermode="closest",
    )
    _nas_fig_xy  # noqa: B018
    return


if __name__ == "__main__":
    app.run()
