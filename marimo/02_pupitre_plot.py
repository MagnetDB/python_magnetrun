import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", title="Part 1 — Pupitre: Plotting")


@app.cell
def __(mo):
    mo.md(
        r"""
        # Part 1 — Pupitre Data: Plotting

        Interactive time-series and key-vs-key plots using plotly.

        All plots use plotly so you can zoom, pan, and hover directly in the
        notebook.
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md("## Load a file")
    return


@app.cell
def __():
    from pathlib import Path

    import python_magnetrun

    _default = str(
        Path(python_magnetrun.__file__).parent.parent / "tests" / "data" / "sample_pupitre.txt"
    )
    return Path, _default, python_magnetrun


@app.cell
def __(mo, _default):
    file_input = mo.ui.text(value=_default, label="Path to `.txt` file", full_width=True)
    housing_input = mo.ui.dropdown(
        options=["M9", "M10", "M14", "M19", "unknown"],
        value="M9",
        label="Housing",
    )
    mo.hstack([file_input, housing_input])
    return file_input, housing_input


@app.cell
def __(file_input, housing_input):
    from python_magnetrun.MagnetRun import load_mrun

    mrun = load_mrun(
        file_input.value,
        housing=housing_input.value,
        auto_resolve=False,
    )
    keys = mrun.getKeys()
    df = mrun.getDataFrame()
    return df, keys, load_mrun, mrun


@app.cell
def __(mo):
    mo.md("---\n## Time-series plot")
    return


@app.cell
def __(keys, mo):
    channel_select = mo.ui.multiselect(
        options=keys,
        value=keys[:2] if len(keys) >= 2 else keys,
        label="Channels to plot",
    )
    channel_select  # noqa: B018
    return (channel_select,)


@app.cell
def __(channel_select, df, mo):
    import plotly.graph_objects as go

    _selected = channel_select.value
    if not _selected:
        mo.stop(True, mo.md("Select at least one channel."))

    fig_ts = go.Figure()
    for _key in _selected:
        if _key in df.columns and "t" in df.columns:
            fig_ts.add_trace(
                go.Scatter(x=df["t"], y=df[_key], mode="lines", name=_key)
            )

    fig_ts.update_layout(
        title="Time series",
        xaxis_title="t [s]",
        yaxis_title="value",
        legend_title="Channel",
        hovermode="x unified",
    )
    fig_ts  # noqa: B018
    return fig_ts, go


@app.cell
def __(mo):
    mo.md("---\n## Key-vs-key (X–Y) plot")
    return


@app.cell
def __(keys, mo):
    x_select = mo.ui.dropdown(options=keys, value=keys[0], label="X axis")
    y_select = mo.ui.dropdown(
        options=keys,
        value=keys[1] if len(keys) > 1 else keys[0],
        label="Y axis",
    )
    mo.hstack([x_select, y_select])
    return x_select, y_select


@app.cell
def __(df, go, mo, x_select, y_select):
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
    return fig_xy, x_key, y_key


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## Notes

        - `df["t"]` is elapsed time in seconds since run start.
        - `df["timestamp"]` holds UTC-aware `datetime64` values.
        - All channels are in SI-compatible units as defined in the housing
          defs file (e.g. `Field` in Tesla, currents in Ampere).
        - For large files, use `02b_downsampling.py` to reduce the number of
          points before plotting.
        """
    )
    return


if __name__ == "__main__":
    app.run()
