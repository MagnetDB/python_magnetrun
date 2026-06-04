import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 4 — Overview & Archive: Plotting

    Interactive time-series plots of pigbrother **Overview** (1 Hz) and
    **Archive** (120 Hz) `.tdms` data.

    - **Overview** gives a bird's-eye view of the entire run at 1 Hz.
    - **Archive** provides the same channels at 120 Hz — useful for
      inspecting fast transients that the Overview misses.

    All plots use plotly so you can zoom, pan, and hover directly.
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

    _root = Path(python_magnetrun.__file__).parent.parent / "data"
    default_overview = str(_root / "M8_Overview_251105-0949.tdms")
    default_archive = str(_root / "M8_Archive_251105-0949.tdms")
    return (default_archive, default_overview)


@app.cell(hide_code=True)
def _(default_overview, mo):
    overview_input = mo.ui.text(
        value=default_overview,
        label="Path to Overview `.tdms` file",
        full_width=True,
    )
    housing_input = mo.ui.dropdown(
        options=["M8", "M9", "M10", "unknown"],
        value="M8",
        label="Housing",
    )
    mo.vstack([overview_input, housing_input])
    return (housing_input, overview_input)


@app.cell
def _(housing_input, overview_input):
    from python_magnetrun.MagnetRun import load_mrun

    overview_mrun = load_mrun(
        overview_input.value,
        housing=housing_input.value,
        auto_resolve=False,
    )
    overview_mdata = overview_mrun.getMData()
    overview_keys = overview_mrun.getKeys()
    overview_groups = sorted({
        k.split("/")[0] for k in overview_keys if "/" in k
    })
    return (overview_groups, overview_mdata, overview_mrun)


@app.cell(hide_code=True)
def _(mo, overview_mdata, overview_mrun):
    try:
        _dur_str = f"{overview_mdata.getDuration():.1f} s"
    except (AttributeError, KeyError, IndexError, TypeError):
        _dur_str = "—"

    mo.md(f"""
    **Loaded:** `{getattr(overview_mdata, 'FileName', '—')}`
    — Housing: `{overview_mrun.getHousing()}`
    — Start: `{overview_mdata.start_timestamp}`
    — Duration: `{_dur_str}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Time-series plot — Overview (1 Hz)
    """)
    return


@app.cell(hide_code=True)
def _(mo, overview_groups):
    ov_group_select = mo.ui.dropdown(
        options=overview_groups,
        value=overview_groups[0] if overview_groups else None,
        label="Group",
    )
    ov_group_select  # noqa: B018
    return (ov_group_select,)


@app.cell(hide_code=True)
def _(mo, ov_group_select, overview_mdata):
    mo.stop(
        not ov_group_select.value,
        mo.callout(mo.md("Select a group."), kind="info"),
    )

    _ov_df = overview_mdata.getTdmsData(ov_group_select.value, channel=None)
    _ov_channels = [c for c in _ov_df.columns if c not in ("t", "timestamp")]

    ov_channel_select = mo.ui.multiselect(
        options=_ov_channels,
        value=_ov_channels[:3] if len(_ov_channels) >= 3 else _ov_channels,
        label="Channels",
    )
    ov_channel_select  # noqa: B018
    return (ov_channel_select,)


@app.cell
def _(mo, ov_channel_select, ov_group_select, overview_mdata):
    import plotly.graph_objects as go

    mo.stop(
        not ov_group_select.value,
        mo.callout(mo.md("Select a group above."), kind="info"),
    )
    _selected = ov_channel_select.value
    if not _selected:
        mo.stop(True, mo.md("Select at least one channel."))

    _df = overview_mdata.getTdmsData(ov_group_select.value, channel=None)

    _fig = go.Figure()
    for _ch in _selected:
        if _ch in _df.columns and "t" in _df.columns:
            _fig.add_trace(go.Scatter(
                x=_df["t"],
                y=_df[_ch],
                mode="lines",
                name=_ch,
            ))

    _fig.update_layout(
        title=f"Overview — {ov_group_select.value} (1 Hz)",
        xaxis_title="t [s]",
        yaxis_title="value",
        legend_title="Channel",
        hovermode="x unified",
    )
    _fig  # noqa: B018
    return (go,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Stacked subplots — shared time axis

    Use `plot_subplots` from `python_magnetrun.plotting` to stack each
    channel in its own panel with a shared time axis.  Ideal when channels
    have **different units**.

    Tick **Normalise** to scale each panel to [−1, 1] for shape comparison.
    """)
    return


@app.cell(hide_code=True)
def _(mo, ov_channel_select, ov_group_select, overview_mdata):
    mo.stop(
        not ov_group_select.value,
        mo.callout(mo.md("Select a group above."), kind="info"),
    )

    _df2 = overview_mdata.getTdmsData(ov_group_select.value, channel=None)
    _chs2 = [c for c in _df2.columns if c not in ("t", "timestamp")]

    sp_channels = mo.ui.multiselect(
        options=_chs2,
        value=ov_channel_select.value or _chs2[:3],
        label="Channels",
    )
    sp_normalize = mo.ui.checkbox(label="Normalise (divide by max)")
    mo.hstack([sp_channels, sp_normalize], align="end")
    return (sp_channels, sp_normalize)


@app.cell
def _(mo, ov_group_select, overview_mdata, sp_channels, sp_normalize):
    from python_magnetrun.plotting import plot_subplots as _plot_subplots
    from python_magnetrun.plotting.backend import get_backend as _get_backend

    mo.stop(
        not ov_group_select.value,
        mo.callout(mo.md("Select a group above."), kind="info"),
    )
    _sel = sp_channels.value
    if not _sel:
        mo.stop(True, mo.md("Select at least one channel above."))

    _df_sp = overview_mdata.getTdmsData(ov_group_select.value, channel=None)
    _b = _get_backend("plotly")
    _fig_sp = _plot_subplots(
        _df_sp,
        fields=list(_sel),
        backend=_b,
        normalize=sp_normalize.value,
        title=f"Overview subplots — {ov_group_select.value}",
    )
    _b.finalize(_fig_sp)
    _fig_sp  # noqa: B018
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Overview vs Archive comparison

    Load an Archive file covering the same time window to compare
    **1 Hz Overview** with **120 Hz Archive** on the same channels.
    Zoom into the plot to see the higher-resolution Archive data.
    """)
    return


@app.cell(hide_code=True)
def _(default_archive, mo):
    archive_input = mo.ui.text(
        value=default_archive,
        label="Path to Archive `.tdms` file",
        full_width=True,
    )
    archive_input  # noqa: B018
    return (archive_input,)


@app.cell
def _(archive_input, housing_input, mo):
    from python_magnetrun.MagnetRun import load_mrun as _load_mrun

    archive_mrun = None
    _arch_error = ""

    try:
        archive_mrun = _load_mrun(
            archive_input.value,
            housing=housing_input.value,
            auto_resolve=False,
        )
    except (ValueError, RuntimeError, OSError) as _e:
        _arch_error = str(_e)

    mo.stop(
        archive_mrun is None,
        mo.callout(
            mo.md(f"Could not load Archive: `{_arch_error}`"),
            kind="warn",
        ),
    )
    return (archive_mrun,)


@app.cell(hide_code=True)
def _(mo, ov_group_select, overview_mdata):
    mo.stop(
        not ov_group_select.value,
        mo.callout(mo.md("Select a group above."), kind="info"),
    )

    _df3 = overview_mdata.getTdmsData(ov_group_select.value, channel=None)
    _chs3 = [c for c in _df3.columns if c not in ("t", "timestamp")]

    cmp_channel_select = mo.ui.dropdown(
        options=_chs3,
        value=_chs3[0] if _chs3 else None,
        label="Channel to compare",
    )
    cmp_channel_select  # noqa: B018
    return (cmp_channel_select,)


@app.cell
def _(
    archive_mrun,
    cmp_channel_select,
    go,
    mo,
    ov_group_select,
    overview_mdata,
):
    mo.stop(
        archive_mrun is None,
        mo.callout(mo.md("No Archive loaded — fill in the path above."), kind="info"),
    )
    mo.stop(
        not cmp_channel_select.value,
        mo.callout(mo.md("Select a channel to compare."), kind="info"),
    )

    _group = ov_group_select.value
    _ch = cmp_channel_select.value
    _arch_mdata = archive_mrun.getMData()

    _ov_df = overview_mdata.getTdmsData(_group, channel=None)
    _ar_df = _arch_mdata.getTdmsData(_group, channel=None) if _group in _arch_mdata.Groups else None

    _fig_cmp = go.Figure()

    if "t" in _ov_df.columns and _ch in _ov_df.columns:
        _fig_cmp.add_trace(go.Scatter(
            x=_ov_df["t"],
            y=_ov_df[_ch],
            mode="lines",
            name=f"{_ch} — Overview (1 Hz)",
            line=dict(width=2),
        ))

    if _ar_df is not None and "t" in _ar_df.columns and _ch in _ar_df.columns:
        _fig_cmp.add_trace(go.Scatter(
            x=_ar_df["t"],
            y=_ar_df[_ch],
            mode="lines",
            name=f"{_ch} — Archive (120 Hz)",
            line=dict(width=1, dash="dot"),
            opacity=0.7,
        ))
    elif _ar_df is None:
        mo.callout(
            mo.md(f"Group `{_group}` not found in Archive."),
            kind="warn",
        )

    _fig_cmp.update_layout(
        title=f"{_ch} — Overview vs Archive",
        xaxis_title="t [s]",
        yaxis_title="value",
        legend_title="Source",
        hovermode="x unified",
    )
    _fig_cmp  # noqa: B018
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Key-vs-key (X–Y) plot

    Plot any two channels within the same group against each other.
    """)
    return


@app.cell(hide_code=True)
def _(mo, ov_group_select, overview_mdata):
    mo.stop(
        not ov_group_select.value,
        mo.callout(mo.md("Select a group above."), kind="info"),
    )

    _df4 = overview_mdata.getTdmsData(ov_group_select.value, channel=None)
    _chs4 = [c for c in _df4.columns if c not in ("t", "timestamp")]

    xy_x_select = mo.ui.dropdown(
        options=_chs4,
        value=_chs4[0] if _chs4 else None,
        label="X axis",
    )
    xy_y_select = mo.ui.dropdown(
        options=_chs4,
        value=_chs4[1] if len(_chs4) > 1 else _chs4[0] if _chs4 else None,
        label="Y axis",
    )
    mo.hstack([xy_x_select, xy_y_select])
    return (xy_x_select, xy_y_select)


@app.cell
def _(go, mo, ov_group_select, overview_mdata, xy_x_select, xy_y_select):
    mo.stop(
        not ov_group_select.value,
        mo.callout(mo.md("Select a group above."), kind="info"),
    )

    _df5 = overview_mdata.getTdmsData(ov_group_select.value, channel=None)
    _xk = xy_x_select.value
    _yk = xy_y_select.value

    if _xk not in _df5.columns or _yk not in _df5.columns:
        mo.stop(True, mo.md("Selected column not available."))

    _fig_xy = go.Figure(go.Scatter(
        x=_df5[_xk],
        y=_df5[_yk],
        mode="markers+lines",
        marker=dict(size=3),
        name=f"{_yk} vs {_xk}",
    ))
    _fig_xy.update_layout(
        title=f"{_yk} vs {_xk} — Overview",
        xaxis_title=_xk,
        yaxis_title=_yk,
        hovermode="closest",
    )
    _fig_xy  # noqa: B018
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Notes

    - After `load_mrun()` each TDMS group's DataFrame contains:
      - `t` — elapsed seconds since run start
      - `timestamp` — naive UTC datetime64
    - Access a group DataFrame: `mdata.getTdmsData(group, channel=None)`
    - Keys use `"Group/Channel"` notation: `mrun.getKeys()`
    - For large Archive files (120 Hz, millions of rows), use the
      downsampling notebook `02b_downsampling.py` to reduce points
      before plotting.
    - `FileDiscovery.discover(overview_path)` locates all related
      Archive, Pupitre, and incident files automatically (see
      `03_overview_loading.py`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## NAS File Browser — Overview files

    Discover Overview `.tdms` files from the NAS.  **Housing** is taken
    from the dropdown above; adjust it to switch magnet site.

    Root search path: `MAGNETRUN_PIGBROTHER_DATA_DIR` (or
    `PIGBROTHER_DATADIR`); default `/mnt/LNCMIG-Data/records/pbsurv`.
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
    from pathlib import Path as _Path2

    from python_magnetrun.data_dirs import PIGBROTHER_DATA_DIR as _PB_DIR
    from python_magnetrun.utils.timestamps import parse_filename_timestamp as _parse_ts

    _housing = housing_input.value
    _overview_dir = _Path2(_PB_DIR) / _housing / "Overview"
    _start = nas_start_date.value
    _end = nas_end_date.value

    nas_overview_input = None

    if not _overview_dir.exists():
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    f"NAS directory **`{_overview_dir}`** is not accessible.\n\n"
                    "Mount the NAS or set `MAGNETRUN_PIGBROTHER_DATA_DIR`."
                ),
                kind="warn",
            ),
        )

    _tdms_files = sorted(_glob.glob(str(_overview_dir / "*.tdms")))
    _in_range = []
    for _f in _tdms_files:
        _dt_f = _parse_ts(_f)
        if _dt_f is not None and _start <= _dt_f.date() <= _end:
            _in_range.append(_f)

    if not _in_range:
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    f"No Overview `.tdms` files found under `{_overview_dir}` "
                    f"between **{_start}** and **{_end}**.\n\n"
                    "Widen the date range or choose a different housing."
                ),
                kind="info",
            ),
        )

    nas_overview_input = mo.ui.dropdown(
        options={_Path2(_f).name: _f for _f in _in_range},
        label="Overview file",
    )
    nas_overview_input  # noqa: B018
    return (nas_overview_input,)


@app.cell
def _(housing_input, mo, nas_overview_input):
    from python_magnetrun.MagnetRun import load_mrun as _load_mrun2

    nas_mrun = None
    nas_df = None
    nas_groups = []

    _no_file = nas_overview_input is None or not getattr(nas_overview_input, "value", None)
    mo.stop(
        _no_file,
        mo.callout(
            mo.md("Select an Overview file from the browser above to load it."),
            kind="info",
        ),
    )

    nas_mrun = _load_mrun2(
        nas_overview_input.value,
        housing=housing_input.value,
        auto_resolve=False,
    )
    nas_groups = sorted({
        k.split("/")[0] for k in nas_mrun.getKeys() if "/" in k
    })
    return (nas_groups, nas_mrun)


@app.cell(hide_code=True)
def _(mo, nas_groups, nas_mrun):
    nas_group_select = mo.ui.dropdown(
        options=nas_groups,
        value=nas_groups[0] if nas_groups else None,
        label="Group",
    )

    mo.stop(
        nas_mrun is None,
        mo.callout(mo.md("No NAS run loaded — select a file above."), kind="info"),
    )
    nas_group_select  # noqa: B018
    return (nas_group_select,)


@app.cell(hide_code=True)
def _(mo, nas_group_select, nas_mrun):
    mo.stop(
        nas_mrun is None,
        mo.callout(mo.md("No NAS run loaded."), kind="info"),
    )
    mo.stop(
        not nas_group_select.value,
        mo.callout(mo.md("Select a group above."), kind="info"),
    )

    _nas_df = nas_mrun.getMData().getTdmsData(nas_group_select.value, None)
    _nas_chs = [c for c in _nas_df.columns if c not in ("t", "timestamp")]

    nas_channel_select = mo.ui.multiselect(
        options=_nas_chs,
        value=_nas_chs[:3] if len(_nas_chs) >= 3 else _nas_chs,
        label="Channels",
    )
    nas_channel_select  # noqa: B018
    return (nas_channel_select,)


@app.cell
def _(go, mo, nas_channel_select, nas_group_select, nas_mrun):
    mo.stop(
        nas_mrun is None,
        mo.callout(mo.md("No NAS run loaded."), kind="info"),
    )

    _sel = nas_channel_select.value
    if not _sel:
        mo.stop(True, mo.md("Select at least one channel."))

    _nas_df2 = nas_mrun.getMData().getTdmsData(nas_group_select.value, None)

    _nas_fig = go.Figure()
    for _ch2 in _sel:
        if _ch2 in _nas_df2.columns and "t" in _nas_df2.columns:
            _nas_fig.add_trace(go.Scatter(
                x=_nas_df2["t"],
                y=_nas_df2[_ch2],
                mode="lines",
                name=_ch2,
            ))

    _nas_fig.update_layout(
        title=f"Overview (NAS) — {nas_group_select.value}",
        xaxis_title="t [s]",
        yaxis_title="value",
        legend_title="Channel",
        hovermode="x unified",
    )
    _nas_fig  # noqa: B018
    return


if __name__ == "__main__":
    app.run()
