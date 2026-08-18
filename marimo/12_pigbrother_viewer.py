import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pigbrother Viewer

    Two-panel oscilloscope-style view of TDMS pigbrother data.
    Select a file, pick a preset group for each panel, then enable or disable
    individual channels. Both panels share a time axis — zoom on one zooms both.
    """)
    return


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import python_magnetrun
    _root = Path(python_magnetrun.__file__).parent.parent / "data"
    _candidates = sorted(_root.glob("*_Overview_*.tdms"))
    default_tdms = str(_candidates[0]) if _candidates else ""
    return (default_tdms,)


@app.cell(hide_code=True)
def _(default_tdms, mo):
    file_input = mo.ui.text(
        value=default_tdms,
        label="Path to `.tdms` file",
        full_width=True,
    )
    housing_input = mo.ui.dropdown(
        options=["M8", "M9", "M10", "unknown"],
        value="M8",
        label="Housing",
    )
    mo.vstack([file_input, housing_input])
    return (file_input, housing_input)


@app.cell
def _(file_input, housing_input, mo):
    from python_magnetrun.MagnetRun import load_mrun

    _mrun = None
    _err = ""
    try:
        _mrun = load_mrun(file_input.value, housing=housing_input.value, auto_resolve=False)
    except (ValueError, FileNotFoundError, OSError, RuntimeError, KeyError, UnicodeDecodeError) as _e:
        _err = str(_e)

    mo.stop(
        _mrun is None,
        mo.callout(mo.md(f"Could not load file: `{_err}`"), kind="danger"),
    )
    assert _mrun is not None
    mrun = _mrun
    mdata = mrun.getMData()
    groups = sorted({k.split("/")[0] for k in mrun.getKeys() if "/" in k})
    return (groups, mdata, mrun)


@app.cell(hide_code=True)
def _(mdata, mrun, mo):
    try:
        _dur = f"{mdata.getDuration():.1f} s"
    except (AttributeError, ValueError, TypeError, IndexError):
        _dur = "—"
    mo.md(
        f"**Loaded:** `{getattr(mdata, 'FileName', '—')}` "
        f"— Housing: `{mrun.getHousing()}` "
        f"— Start: `{mdata.start_timestamp}` "
        f"— Duration: `{_dur}`"
    )
    return


@app.cell(hide_code=True)
def _(groups, mo):
    mo.stop(not groups, mo.callout(mo.md("No TDMS groups found in file."), kind="warn"))
    preset_select = mo.ui.dropdown(options=groups, value=groups[0], label="Pre sets")
    fft_mode = mo.ui.switch(label="FFT MODE")
    mo.hstack([preset_select, fft_mode], align="end")
    return (fft_mode, preset_select)


# --- Panel A: group dropdown (resets when preset changes) ---

@app.cell(hide_code=True)
def _(groups, mo, preset_select):
    panel_a_group = mo.ui.dropdown(
        options=groups,
        value=preset_select.value or groups[0],
        label="",
    )
    return (panel_a_group,)


# --- Panel A: channels (rebuilds when panel_a_group changes) ---

@app.cell(hide_code=True)
def _(mdata, mo, panel_a_group):
    _df = mdata.getTdmsData(panel_a_group.value, None)
    panel_a_channels = [c for c in _df.columns if c not in ("t", "timestamp")]
    panel_a_enable_all = mo.ui.switch(label="Enable ALL", value=True)
    panel_a_checks = mo.ui.array(
        [mo.ui.checkbox(label=c, value=True) for c in panel_a_channels]
    )
    return (panel_a_channels, panel_a_checks, panel_a_enable_all)


# --- Panel B: group dropdown (defaults to next group after preset) ---

@app.cell(hide_code=True)
def _(groups, mo, preset_select):
    _idx = groups.index(preset_select.value) if preset_select.value in groups else 0
    panel_b_group = mo.ui.dropdown(
        options=groups,
        value=groups[(_idx + 1) % len(groups)],
        label="",
    )
    return (panel_b_group,)


# --- Panel B: channels (rebuilds when panel_b_group changes) ---

@app.cell(hide_code=True)
def _(mdata, mo, panel_b_group):
    _df = mdata.getTdmsData(panel_b_group.value, None)
    panel_b_channels = [c for c in _df.columns if c not in ("t", "timestamp")]
    panel_b_enable_all = mo.ui.switch(label="Enable ALL", value=True)
    panel_b_checks = mo.ui.array(
        [mo.ui.checkbox(label=c, value=True) for c in panel_b_channels]
    )
    return (panel_b_channels, panel_b_checks, panel_b_enable_all)


# --- Layout + synchronized figure ---

@app.cell
def _(
    fft_mode,
    mdata,
    mo,
    panel_a_channels,
    panel_a_checks,
    panel_a_enable_all,
    panel_a_group,
    panel_b_channels,
    panel_b_checks,
    panel_b_enable_all,
    panel_b_group,
):
    import numpy as _np
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots

    _sel_a = (
        panel_a_channels
        if panel_a_enable_all.value
        else [c for c, v in zip(panel_a_channels, panel_a_checks.value, strict=False) if v]
    )
    _sel_b = (
        panel_b_channels
        if panel_b_enable_all.value
        else [c for c, v in zip(panel_b_channels, panel_b_checks.value, strict=False) if v]
    )

    _df_a = mdata.getTdmsData(panel_a_group.value, None)
    _df_b = mdata.getTdmsData(panel_b_group.value, None)
    _t_a = "timestamp" if "timestamp" in _df_a.columns else "t"
    _t_b = "timestamp" if "timestamp" in _df_b.columns else "t"

    _fig = _make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[panel_a_group.value, panel_b_group.value],
    )

    def _dt_seconds(df, t_col):
        if len(df) > 1:
            _delta = df[t_col].iloc[1] - df[t_col].iloc[0]
            try:
                return _delta.total_seconds() if hasattr(_delta, "total_seconds") else float(_delta)
            except (TypeError, ValueError):
                pass
        return 1.0

    def _add_traces(df, t_col, channels, row):
        _dt = _dt_seconds(df, t_col)
        for _ch in channels:
            if _ch not in df.columns:
                continue
            if fft_mode.value:
                _vals = df[_ch].fillna(0.0).values
                _y = _np.abs(_np.fft.rfft(_vals))
                _x = _np.fft.rfftfreq(len(_vals), d=_dt)
            else:
                _y = df[_ch]
                _x = df[t_col]
            _fig.add_trace(_go.Scatter(x=_x, y=_y, mode="lines", name=_ch), row=row, col=1)

    _add_traces(_df_a, _t_a, _sel_a, 1)
    _add_traces(_df_b, _t_b, _sel_b, 2)

    _fig.update_layout(
        hovermode="x unified",
        uirevision="shared",
        height=700,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    _fig.update_yaxes(exponentformat="e")
    if fft_mode.value:
        _fig.update_xaxes(row=2, col=1, title_text="Frequency [Hz]")

    _ctrl_a = mo.vstack([
        mo.md("**Panel A**"),
        panel_a_group,
        panel_a_enable_all,
        panel_a_checks,
    ])
    _ctrl_b = mo.vstack([
        mo.md("**Panel B**"),
        panel_b_group,
        panel_b_enable_all,
        panel_b_checks,
    ])

    _layout = mo.hstack(
        [mo.vstack([_ctrl_a, mo.md("---"), _ctrl_b]), mo.ui.plotly(_fig)],
        align="start",
    )
    _layout  # noqa: B018  # pyright: ignore[reportUnusedExpression]
    return


if __name__ == "__main__":
    app.run()
