import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Downsampling: Why It Matters

    Raw magnet-run files can contain millions of points.  Rendering them
    naively freezes the browser.  Downsampling reduces the point count while
    preserving the visual appearance of the signal.

    This notebook lets you compare **raw vs RDP vs M4** side-by-side with
    an interactive slider.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Algorithm overview

    | Method | Type | Parameter | Characteristic |
    |--------|------|-----------|----------------|
    | `stride` | count-based | `n_out` | uniform, fast |
    | `minmax` | count-based | `n_out` | preserves envelope (2 pts/bucket) |
    | `lttb` | count-based | `n_out` | perceptual fidelity (requires tsdownsample) |
    | `minmax_lttb` | count-based | `n_out` | best perceptual fidelity (requires tsdownsample) |
    | **`m4`** | count-based | `n_out` | pixel-perfect line chart, 4 pts/bucket (requires tsdownsample) |
    | **`nan_m4`** | count-based | `n_out` | m4 but preserves NaN gaps (requires tsdownsample) |
    | **`rdp`** | geometry-based | `epsilon` | plateau-aware, fewer pts on flat regions (requires simplification) |
    | **`vw`** | geometry-based | `epsilon` | area-based variant of RDP (requires simplification) |

    Geometry-based methods (`rdp`, `vw`) allocate more points where the
    signal is changing fast and fewer points where it is flat — ideal for
    magnet run data with long constant-field plateaus.
    """)
    return


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
        options=["M8", "M9", "M10", "unknown"],
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
    df_raw = mrun.getDataFrame()
    return df_raw, keys


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Choose channel and target point count
    """)
    return


@app.cell
def _(keys, mo):
    channel_select = mo.ui.dropdown(
        options=keys,
        value=keys[0],
        label="Channel",
    )
    channel_select  # noqa: B018
    return (channel_select,)


@app.cell(hide_code=True)
def _(df_raw, mo):
    n_raw = len(df_raw)
    n_out_slider = mo.ui.slider(
        start=10,
        stop=max(10, n_raw),
        step=max(1, n_raw // 100),
        value=max(10, n_raw // 4),
        label=f"Target n_out (raw has {n_raw} points)",
        full_width=True,
    )
    n_out_slider  # noqa: B018
    return (n_out_slider,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Side-by-side comparison (available methods)
    """)
    return


@app.cell
def _(channel_select, df_raw, mo, n_out_slider):
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from python_magnetrun.utils.downsampling import (
        HAS_SIMPLIFICATION,
        HAS_TSDOWNSAMPLE,
        DownsampleConfig,
        downsample_dataframe,
    )

    _key = channel_select.value
    _n_out = n_out_slider.value

    if _key not in df_raw.columns or "t" not in df_raw.columns:
        mo.stop(True, mo.md("Selected channel not available."))

    # Build the set of configs to compare
    _configs: list[tuple[str, DownsampleConfig]] = [
        ("raw", None),
        ("stride", DownsampleConfig(n_out=_n_out, method="stride")),
        ("minmax", DownsampleConfig(n_out=_n_out, method="minmax")),
    ]

    if HAS_TSDOWNSAMPLE:
        _configs += [
            ("m4", DownsampleConfig(n_out=_n_out, method="m4")),
            ("lttb", DownsampleConfig(n_out=_n_out, method="lttb")),
            ("minmax_lttb", DownsampleConfig(n_out=_n_out, method="minmax_lttb")),
        ]

    # For RDP we need the data to binary-search epsilon
    if HAS_SIMPLIFICATION:
        _t = df_raw["t"].to_numpy(dtype=float)
        _y = df_raw[_key].to_numpy(dtype=float)
        _mask = ~np.isnan(_t) & ~np.isnan(_y)
        if _mask.sum() > 10:
            _rdp_cfg = DownsampleConfig.from_n_out_rdp(
                _y[_mask], _t[_mask], n_out=_n_out, method="rdp", tol=0.15
            )
            _configs.append(("rdp", _rdp_cfg))

    _n_methods = len(_configs)
    _fig = make_subplots(
        rows=_n_methods,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[label for label, _ in _configs],
        vertical_spacing=0.04,
    )

    for _row, (_label, _cfg) in enumerate(_configs, start=1):
        if _cfg is None:
            _t_plot = df_raw["t"].to_numpy()
            _y_plot = df_raw[_key].to_numpy()
            _n_pts = len(_t_plot)
        else:
            _df_ds = downsample_dataframe(df_raw, "t", [_key], _cfg)
            _t_plot = _df_ds["t"].to_numpy()
            _y_plot = _df_ds[_key].to_numpy()
            _n_pts = len(_t_plot)

        _fig.add_trace(
            go.Scatter(
                x=_t_plot,
                y=_y_plot,
                mode="lines",
                name=f"{_label} ({_n_pts} pts)",
                showlegend=True,
            ),
            row=_row,
            col=1,
        )

    _fig.update_layout(
        height=250 * _n_methods,
        title=f"Downsampling comparison — {_key}",
        hovermode="x unified",
    )
    _fig  # noqa: B018
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## NAS File Browser

    Discover and load a `.txt` pupitre file directly from the NAS and run
    the same downsampling comparison.  **Housing** is taken from the dropdown
    above; adjust it to switch magnet site.

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
    from python_magnetrun.MagnetRun import load_mrun as _load_mrun_nas

    nas_mrun = None
    nas_df_raw = None
    nas_keys = []

    _no_file = nas_file_input is None or not getattr(nas_file_input, "value", None)
    mo.stop(
        _no_file,
        mo.callout(
            mo.md("Select a file from the NAS browser above to load it."),
            kind="info",
        ),
    )

    nas_mrun = _load_mrun_nas(
        nas_file_input.value,
        housing=housing_input.value,
        auto_resolve=False,
    )
    nas_df_raw = nas_mrun.getDataFrame()
    nas_keys = nas_mrun.getKeys()
    return (nas_df_raw, nas_keys, nas_mrun)


@app.cell
def _(mo, nas_keys, nas_mrun):
    mo.stop(
        nas_mrun is None,
        mo.callout(mo.md("No NAS run loaded — select a file above."), kind="info"),
    )
    nas_channel_select = mo.ui.dropdown(
        options=nas_keys,
        value=nas_keys[0],
        label="Channel",
    )
    nas_channel_select  # noqa: B018
    return (nas_channel_select,)


@app.cell
def _(mo, nas_df_raw, nas_mrun):
    mo.stop(
        nas_mrun is None,
        mo.callout(mo.md("No NAS run loaded."), kind="info"),
    )
    _n_nas_raw = len(nas_df_raw)
    nas_n_out_slider = mo.ui.slider(
        start=10,
        stop=max(10, _n_nas_raw),
        step=max(1, _n_nas_raw // 100),
        value=max(10, _n_nas_raw // 4),
        label=f"Target n_out (raw has {_n_nas_raw} points)",
        full_width=True,
    )
    nas_n_out_slider  # noqa: B018
    return (nas_n_out_slider,)


@app.cell
def _(mo, nas_channel_select, nas_df_raw, nas_mrun, nas_n_out_slider):
    import numpy as _np
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots

    from python_magnetrun.utils.downsampling import (
        HAS_SIMPLIFICATION as _HAS_SIMPLIFICATION,
    )
    from python_magnetrun.utils.downsampling import (
        HAS_TSDOWNSAMPLE as _HAS_TSDOWNSAMPLE,
    )
    from python_magnetrun.utils.downsampling import (
        DownsampleConfig as _DownsampleConfig,
    )
    from python_magnetrun.utils.downsampling import (
        downsample_dataframe as _downsample_dataframe,
    )

    mo.stop(
        nas_mrun is None,
        mo.callout(mo.md("No NAS run loaded."), kind="info"),
    )

    _nas_key = nas_channel_select.value
    _nas_n_out = nas_n_out_slider.value

    if _nas_key not in nas_df_raw.columns or "t" not in nas_df_raw.columns:
        mo.stop(True, mo.md("Selected channel not available."))

    _nas_configs: list[tuple[str, _DownsampleConfig]] = [
        ("raw", None),
        ("stride", _DownsampleConfig(n_out=_nas_n_out, method="stride")),
        ("minmax", _DownsampleConfig(n_out=_nas_n_out, method="minmax")),
    ]

    if _HAS_TSDOWNSAMPLE:
        _nas_configs += [
            ("m4", _DownsampleConfig(n_out=_nas_n_out, method="m4")),
            ("lttb", _DownsampleConfig(n_out=_nas_n_out, method="lttb")),
            ("minmax_lttb", _DownsampleConfig(n_out=_nas_n_out, method="minmax_lttb")),
        ]

    if _HAS_SIMPLIFICATION:
        _t_nas = nas_df_raw["t"].to_numpy(dtype=float)
        _y_nas = nas_df_raw[_nas_key].to_numpy(dtype=float)
        _mask_nas = ~_np.isnan(_t_nas) & ~_np.isnan(_y_nas)
        if _mask_nas.sum() > 10:
            _rdp_cfg = _DownsampleConfig.from_n_out_rdp(
                _y_nas[_mask_nas],
                _t_nas[_mask_nas],
                n_out=_nas_n_out,
                method="rdp",
                tol=0.15,
            )
            _nas_configs.append(("rdp", _rdp_cfg))

    _n_nas_methods = len(_nas_configs)
    _nas_fig = _make_subplots(
        rows=_n_nas_methods,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[label for label, _ in _nas_configs],
        vertical_spacing=0.04,
    )

    for _row, (_label, _cfg) in enumerate(_nas_configs, start=1):
        if _cfg is None:
            _t_plot = nas_df_raw["t"].to_numpy()
            _y_plot = nas_df_raw[_nas_key].to_numpy()
            _n_pts = len(_t_plot)
        else:
            _df_ds = _downsample_dataframe(nas_df_raw, "t", [_nas_key], _cfg)
            _t_plot = _df_ds["t"].to_numpy()
            _y_plot = _df_ds[_nas_key].to_numpy()
            _n_pts = len(_t_plot)

        _nas_fig.add_trace(
            _go.Scatter(
                x=_t_plot,
                y=_y_plot,
                mode="lines",
                name=f"{_label} ({_n_pts} pts)",
                showlegend=True,
            ),
            row=_row,
            col=1,
        )

    _nas_fig.update_layout(
        height=250 * _n_nas_methods,
        title=f"Downsampling comparison (NAS) — {_nas_key}",
        hovermode="x unified",
    )
    _nas_fig  # noqa: B018
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## When to use each method

    | Scenario | Recommended method |
    |----------|--------------------|
    | Quick preview, any file | `stride` |
    | Preserve min/max envelope | `minmax` |
    | Best visual fidelity, fixed count | `minmax_lttb` or `m4` |
    | NaN gaps must stay visible | `nan_m4` |
    | Plateau-heavy runs, file size matters | `rdp` or `vw` |

    **Rule of thumb for `epsilon` (RDP/VW):** start at 1 % of the signal
    range and adjust until the number of output points meets your target.
    Use `DownsampleConfig.from_n_out_rdp()` to let the library search for
    you.
    """)
    return


if __name__ == "__main__":
    app.run()
