import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 1 — Pupitre Data: Loading

    This notebook shows how to load a Pupitre `.txt` file and inspect its
    content: metadata, column keys, units, and the underlying DataFrame.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## File picker

    Enter the path to a `.txt` file below.  The sample bundled with the
    test suite is pre-filled so you can run the notebook immediately
    without a NAS connection.
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
def _(mo, default_path):
    file_input = mo.ui.text(
        value=default_path, label="Path to `.txt` file", full_width=True
    )
    file_input  # noqa: B018
    return (file_input,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Housing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    housing_input = mo.ui.dropdown(
        options=["M8", "M9", "M10", "unknown"],
        value="M9",
        label="Housing",
    )
    housing_input  # noqa: B018
    return (housing_input,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Load the run
    """)
    return


@app.cell
def _(file_input, housing_input):
    from python_magnetrun.MagnetRun import load_mrun

    mrun = load_mrun(
        file_input.value,
        housing=housing_input.value,
        auto_resolve=False,
    )
    return (mrun,)


@app.cell(hide_code=True)
def _(mo, mrun):
    mo.md(f"""
    **Run loaded successfully.**

    | Property | Value |
    |----------|-------|
    | Housing | `{mrun.getHousing()}` |
    | File | `{getattr(mrun, 'filename', '—')}` |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Keys and units
    """)
    return


@app.cell
def _(mo, mrun):
    import pandas as pd

    _mdata = mrun.getMData()
    rows = []
    for k in mrun.getKeys():
        meta = _mdata.getFieldMeta(k)
        if meta is not None:
            unit_str = f"{meta.unit:~P}" if meta.unit is not None else "—"
            rows.append({
                "key": k,
                "symbol": meta.symbol,
                "unit": unit_str,
                "label": meta.label or "—",
            })
        else:
            try:
                sym, unit = mrun.getUnit(k)
                unit_str = f"{unit:~P}" if unit is not None else "—"
            except (KeyError, RuntimeError):
                sym, unit_str = k, "—"
            rows.append({"key": k, "symbol": sym, "unit": unit_str, "label": "—"})

    df_keys = pd.DataFrame(rows)
    mo.ui.table(df_keys, label="Available channels")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Raw DataFrame
    """)
    return


@app.cell(hide_code=True)
def _(mo, mrun):
    df = mrun.getDataFrame()
    mo.md(f"""
        Shape: **{df.shape[0]} rows × {df.shape[1]} columns**
        """)
    return (df,)


@app.cell(hide_code=True)
def _(df, mo):
    mo.ui.table(df.head(20), label="First 20 rows")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Data summary statistics
    """)
    return


@app.cell(hide_code=True)
def _(df, mo):
    mo.ui.table(
        df.describe().reset_index().rename(columns={"index": "stat"}),
        label="Descriptive statistics",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MagnetData object

    `mrun.getMData()` returns the underlying `PandasMagnetData` object,
    which exposes the full API for derived fields, stats, extraction, and
    export.
    """)
    return


@app.cell
def _(mo, mrun):
    mdata = mrun.getMData()
    mo.md(f"""
        `type(mdata)` → `{type(mdata).__name__}`

        Available on `mdata`:
        - `mdata.getData()` — returns the raw DataFrame
        - `mdata.stats(key)` — summary stats for one channel
        - `mdata.addData(name, formula=...)` — compute a derived field
        - `mdata.extractTimeData(range_str)` — slice by timestamp
        - `mdata.saveData(keys, filename)` — export to CSV/TSV
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## NAS File Browser

    The cells below let you discover and load a `.txt` pupitre file directly
    from the NAS instead of entering a path manually.

    - **Housing** is taken from the dropdown in the *Housing* section above —
      change it there to switch magnet site.
    - Set the **date range** to narrow the list to the period you care about.
    - Everything reacts automatically: changing the housing or dates immediately
      refreshes the file list.

    The root search path is controlled by the `MAGNETRUN_PUPITRE_DATA_DIR`
    (or `PUPITRE_DATADIR`) environment variable; it defaults to
    `/mnt/LNCMIG-Data/records/srv-data-install`.
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

    # Initialise to None; overwritten below if discovery succeeds.
    # Dependent cells check for None before accessing .value.
    nas_file_input = None

    if not _site_dir.exists():
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    f"NAS directory **`{_site_dir}`** is not accessible.\n\n"
                    "Mount the NAS or override the path with the "
                    "`MAGNETRUN_PUPITRE_DATA_DIR` environment variable."
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
    return (nas_mrun,)


@app.cell(hide_code=True)
def _(mo, nas_file_input, nas_mrun):
    import pandas as _pd

    mo.stop(
        nas_mrun is None,
        mo.callout(
            mo.md("No NAS run loaded yet — select a file in the browser above."),
            kind="info",
        ),
    )

    _nas_df = nas_mrun.getDataFrame()
    _nas_mdata = nas_mrun.getMData()
    _nas_rows = []
    for _k in nas_mrun.getKeys():
        _meta = _nas_mdata.getFieldMeta(_k)
        if _meta is not None:
            _unit_str = f"{_meta.unit:~P}" if _meta.unit is not None else "—"
            _nas_rows.append({
                "key": _k,
                "symbol": _meta.symbol,
                "unit": _unit_str,
                "label": _meta.label or "—",
            })
        else:
            try:
                _sym, _unit = nas_mrun.getUnit(_k)
                _unit_str = f"{_unit:~P}" if _unit is not None else "—"
            except (KeyError, RuntimeError):
                _sym, _unit_str = _k, "—"
            _nas_rows.append({"key": _k, "symbol": _sym, "unit": _unit_str, "label": "—"})

    mo.vstack([
        mo.md(f"""
**NAS run loaded successfully.**

| Property | Value |
|----------|-------|
| Housing | `{nas_mrun.getHousing()}` |
| File | `{nas_file_input.value}` |
| Shape | {_nas_df.shape[0]} rows × {_nas_df.shape[1]} columns |
"""),
        mo.ui.table(_pd.DataFrame(_nas_rows), label="Available channels"),
        mo.ui.table(_nas_df.head(20), label="First 20 rows"),
    ])
    return


if __name__ == "__main__":
    app.run()
