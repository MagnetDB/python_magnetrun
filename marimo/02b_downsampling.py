import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", title="Part 1 — Downsampling Comparison")


@app.cell
def __(mo):
    mo.md(
        r"""
        # Downsampling: Why It Matters

        Raw magnet-run files can contain millions of points.  Rendering them
        naively freezes the browser.  Downsampling reduces the point count while
        preserving the visual appearance of the signal.

        This notebook lets you compare **raw vs RDP vs M4** side-by-side with
        an interactive slider.
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
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
        """
    )
    return


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
    df_raw = mrun.getDataFrame()
    return df_raw, keys, load_mrun, mrun


@app.cell
def __(mo):
    mo.md("---\n## Choose channel and target point count")
    return


@app.cell
def __(keys, mo):
    channel_select = mo.ui.dropdown(
        options=keys,
        value=keys[0],
        label="Channel",
    )
    channel_select
    return (channel_select,)


@app.cell
def __(df_raw, mo):
    n_raw = len(df_raw)
    n_out_slider = mo.ui.slider(
        start=10,
        stop=max(10, n_raw),
        step=max(1, n_raw // 100),
        value=max(10, n_raw // 4),
        label=f"Target n_out (raw has {n_raw} points)",
        full_width=True,
    )
    n_out_slider
    return n_out_slider, n_raw


@app.cell
def __(mo):
    mo.md("---\n## Side-by-side comparison (available methods)")
    return


@app.cell
def __(channel_select, df_raw, mo, n_out_slider):
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from python_magnetrun.utils.downsampling import (
        DownsampleConfig,
        HAS_TSDOWNSAMPLE,
        HAS_SIMPLIFICATION,
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
        rows=_n_methods, cols=1,
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
                x=_t_plot, y=_y_plot, mode="lines",
                name=f"{_label} ({_n_pts} pts)",
                showlegend=True,
            ),
            row=_row, col=1,
        )

    _fig.update_layout(
        height=250 * _n_methods,
        title=f"Downsampling comparison — {_key}",
        hovermode="x unified",
    )
    _fig
    return (
        DownsampleConfig,
        HAS_SIMPLIFICATION,
        HAS_TSDOWNSAMPLE,
        downsample_dataframe,
        go,
        make_subplots,
        np,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
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
        """
    )
    return


if __name__ == "__main__":
    app.run()
