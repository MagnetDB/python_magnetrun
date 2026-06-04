import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # `hybrid-magnetrun` — Hybrid Magnet Data (kHz & RMS)

    CLI entry point: **`hybrid-magnetrun`**

    This notebook demonstrates how to load and explore hybrid magnet data
    acquired by FEPC systems.  Hybrid data lives in a directory tree:

    ```
    <base_dir>/
      kHz/   YYYY-MM-DD/  <FEPC-system>/  *.bin + HOST_*_DATA.CFG
      rms/   YYYY-MM-DD/  <FEPC-system>/  ...
      trigger/  ...
    ```

    | Operation | What it does |
    |-----------|-------------|
    | Discover  | List available dates for kHz, RMS, and trigger data |
    | Variables | Inspect channel names per FEPC system |
    | Load kHz  | Load high-frequency (kHz-rate) channels |
    | Load RMS  | Load RMS-averaged channels |
    | Plot      | Overlay kHz and RMS traces |
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Connection parameters
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    base_dir_input = mo.ui.text(
        value="",
        placeholder="/mnt/LNCMIG-Data/records/hybrid",
        label="Base directory (contains kHz/, rms/, trigger/ sub-dirs)",
        full_width=True,
    )
    base_dir_input  # noqa: B018
    return (base_dir_input,)


@app.cell(hide_code=True)
def _(mo):
    import datetime as _dt
    _today = _dt.date.today()
    date_input = mo.ui.date(value=_today, label="Date")
    fepc_input = mo.ui.dropdown(
        options=["FEPC-LNCMI", "FEPC-AUX-LNCMI", "A1", "A2", "A3", "A4"],
        value="FEPC-LNCMI",
        label="FEPC system",
    )
    mo.hstack([date_input, fepc_input], align="start")
    return (date_input, fepc_input)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Discover available dates
    """)
    return


@app.cell
def _(base_dir_input, mo):
    import os

    mo.stop(
        not base_dir_input.value or not os.path.isdir(base_dir_input.value),
        mo.callout(
            mo.md(
                "Enter the path to the hybrid data base directory above.\n\n"
                "The directory must contain `kHz/`, `rms/`, and/or `trigger/` subdirectories."
            ),
            kind="info",
        ),
    )

    import pandas as pd

    from python_magnetrun.hybrid.utils import list_available_dates

    _rows = []
    for _dtype in ["kHz", "rms", "trigger"]:
        _dates = list_available_dates(base_dir_input.value, _dtype)
        _rows.append({"data_type": _dtype, "count": len(_dates), "first": _dates[0] if _dates else "—", "last": _dates[-1] if _dates else "—"})

    mo.ui.table(pd.DataFrame(_rows), label="Available dates by data type")
    return (list_available_dates, pd, os)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Load HybridData for selected date & system
    """)
    return


@app.cell
def _(base_dir_input, date_input, fepc_input, mo, os):
    mo.stop(
        not base_dir_input.value or not os.path.isdir(base_dir_input.value),
        mo.callout(mo.md("Set a valid base directory above."), kind="info"),
    )

    from python_magnetrun.hybrid.hybrid_data import HybridData

    _date_str = str(date_input.value)
    _system = fepc_input.value

    try:
        hdata = HybridData(
            base_dir=base_dir_input.value,
            date_str=_date_str,
            fepc_system=_system,
        )
        _load_ok = True
    except (ValueError, NotImplementedError, ImportError, OSError) as _e:
        _load_ok = False
        _err = str(_e)

    mo.stop(
        not _load_ok,
        mo.callout(mo.md(f"Failed to load HybridData:\n\n```\n{_err}\n```"), kind="danger"),
    )

    mo.md(f"""
    **HybridData loaded.**

    | Property | Value |
    |----------|-------|
    | Date | `{_date_str}` |
    | FEPC system | `{_system}` |
    | kHz available | `{hdata._info.khz_available}` |
    | RMS available | `{hdata._info.rms_available}` |
    """)
    return (hdata, HybridData)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## kHz channel list
    """)
    return


@app.cell(hide_code=True)
def _(fepc_input, hdata, mo, pd):
    _system = fepc_input.value
    try:
        _vars = hdata.get_khz_variables(_system)
        _rows_v = (
            [{"type": "analog", "channel": c} for c in _vars.get("analog", [])]
            + [{"type": "digital", "channel": c} for c in _vars.get("digital", [])]
        )
        mo.ui.table(pd.DataFrame(_rows_v), label=f"kHz variables — {_system}")
    except (ImportError, OSError) as _e:
        mo.callout(mo.md(f"Cannot list kHz variables: {_e}"), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load and plot a kHz channel
    """)
    return


@app.cell(hide_code=True)
def _(fepc_input, hdata, mo):
    try:
        _vars = hdata.get_khz_variables(fepc_input.value)
        _analog = _vars.get("analog", [])
    except (ImportError, OSError):
        _analog = []

    khz_channel_input = mo.ui.dropdown(
        options=_analog or ["—"],
        value=_analog[0] if _analog else "—",
        label="kHz channel to plot",
    )
    khz_channel_input  # noqa: B018
    return (khz_channel_input,)


@app.cell
def _(fepc_input, hdata, khz_channel_input, mo):
    import plotly.graph_objects as go

    _channel = khz_channel_input.value

    mo.stop(
        not _channel or _channel == "—",
        mo.callout(mo.md("Select a kHz channel above."), kind="info"),
    )

    try:
        _df_khz = hdata.get_khz_data(fepc_input.value, _channel, apply_calibration=True)
        _khz_ok = True
    except (ValueError, ImportError, OSError) as _e:
        _khz_ok = False
        _err = str(_e)

    mo.stop(
        not _khz_ok,
        mo.callout(mo.md(f"Failed to load kHz data: `{_err}`"), kind="danger"),
    )

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_df_khz.index.to_numpy(),
        y=_df_khz.iloc[:, 0].to_numpy(),
        mode="lines",
        name=_channel,
    ))
    _fig.update_layout(
        title=f"kHz — {_channel} ({fepc_input.value})",
        xaxis_title="t (s)",
        height=400,
    )
    _fig  # noqa: B018
    return (go,)


@app.cell(hide_code=True)
def _(mo, base_dir_input, date_input, fepc_input, khz_channel_input):
    mo.md(f"""
    ## Equivalent CLI commands

    List available dates:

    ```bash
    hybrid-magnetrun --base-dir {base_dir_input.value or "<base_dir>"} --list-dates
    ```

    Show kHz variables for a system:

    ```bash
    hybrid-magnetrun \\
        --base-dir {base_dir_input.value or "<base_dir>"} \\
        --date {date_input.value} \\
        --fepc-system {fepc_input.value} \\
        --khz-vars
    ```

    Plot a kHz channel:

    ```bash
    hybrid-magnetrun \\
        --base-dir {base_dir_input.value or "<base_dir>"} \\
        --date {date_input.value} \\
        --fepc-system {fepc_input.value} \\
        --plot-khz "{khz_channel_input.value}" \\
        --save
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
