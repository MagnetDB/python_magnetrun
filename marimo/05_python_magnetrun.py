import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # `python-magnetrun` — Data Inspection, Stats & Select

    CLI entry point: **`python-magnetrun`**

    This notebook demonstrates the most common operations exposed by the
    `python-magnetrun` command:

    | Sub-command | What it does |
    |-------------|-------------|
    | `info`      | List available channels and optionally convert to CSV |
    | `stats`     | Descriptive statistics per channel |
    | `select`    | Extract a time slice or a subset of channels to CSV |
    | `add`       | Compute and attach a derived field |

    The `plot` sub-command is covered in the dedicated plotting notebooks
    (`02_pupitre_plot.py` and `04_overview_plot.py`).
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

    Enter the path to a `.txt`, `.tdms`, or `.csv` run file below.
    The sample bundled with the test suite is pre-filled so you can run the
    notebook immediately.
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
        value=default_path, label="Path to run file (.txt / .tdms / .csv)", full_width=True
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## `info` — Load and inspect the run
    """)
    return


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
        case ".csv":
            mrun = MagnetRun.fromcsv(housing=_housing, assembly="", filename=_path)
        case _:
            mo.stop(True, mo.callout(mo.md(f"Unsupported extension: `{_ext}`"), kind="danger"))

    mrun.setHousing(_housing)
    mrun  # noqa: B018
    return (mrun, MagnetRun, os)


@app.cell(hide_code=True)
def _(mo, mrun):
    import pandas as pd

    _mdata = mrun.getMData()
    _rows = []
    for _k in mrun.getKeys():
        _meta = _mdata.getFieldMeta(_k)
        if _meta is not None:
            _unit_str = f"{_meta.unit:~P}" if _meta.unit is not None else "—"
            _rows.append({"key": _k, "symbol": _meta.symbol, "unit": _unit_str, "label": _meta.label or "—"})
        else:
            try:
                _sym, _unit = mrun.getUnit(_k)
                _unit_str = f"{_unit:~P}" if _unit is not None else "—"
            except (KeyError, RuntimeError):
                _sym, _unit_str = _k, "—"
            _rows.append({"key": _k, "symbol": _sym, "unit": _unit_str, "label": "—"})

    mo.ui.table(pd.DataFrame(_rows), label="Available channels (info)")
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Equivalent CLI command

    ```bash
    python-magnetrun <file.txt> --housing M9 info --list
    ```

    To also dump a CSV alongside the original file:

    ```bash
    python-magnetrun <file.txt> --housing M9 info --list --convert
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## `stats` — Descriptive statistics per channel
    """)
    return


@app.cell
def _(mrun, mo, pd):
    from python_magnetrun.processing.stats import stats as _compute_stats

    _mdata = mrun.getMData()
    _result = _compute_stats(_mdata, display=False)

    _columns = _result[1][1:]
    _rows_stats = []
    for _row in _result[0]:
        _entry = dict(zip(["channel"] + _columns, _row, strict=False))
        _rows_stats.append(_entry)

    mo.ui.table(pd.DataFrame(_rows_stats), label="Statistics")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Equivalent CLI command

    ```bash
    python-magnetrun <file.txt> --housing M9 stats
    ```

    With plateau detection and specific channel:

    ```bash
    python-magnetrun <file.txt> --housing M9 stats --plateau --keys Field
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## `select` — Extract a time slice
    """)
    return


@app.cell(hide_code=True)
def _(mrun, mo):
    _df = mrun.getDataFrame()
    _t_min = float(_df.index.min()) if hasattr(_df.index, "min") else 0.0
    _t_max = float(_df.index.max()) if hasattr(_df.index, "max") else 100.0

    mo.md(f"""
    The loaded run has time index in the range **{_t_min:.1f} – {_t_max:.1f} s**.

    Adjust the sliders below to select a sub-range, then inspect the extracted
    DataFrame.
    """)
    return


@app.cell(hide_code=True)
def _(mo, mrun):
    _df = mrun.getDataFrame()
    try:
        _t_min = float(_df.index.min())
        _t_max = float(_df.index.max())
    except (ValueError, TypeError):
        _t_min, _t_max = 0.0, 100.0

    t_start_input = mo.ui.number(
        value=_t_min, start=_t_min, stop=_t_max, step=1.0, label="t start (s)"
    )
    t_end_input = mo.ui.number(
        value=min(_t_min + 60.0, _t_max), start=_t_min, stop=_t_max, step=1.0, label="t end (s)"
    )
    mo.hstack([t_start_input, t_end_input], align="start")
    return (t_start_input, t_end_input)


@app.cell
def _(mo, mrun, t_end_input, t_start_input):
    _mdata = mrun.getMData()
    _t0 = t_start_input.value
    _t1 = t_end_input.value

    mo.stop(
        _t0 >= _t1,
        mo.callout(mo.md("t start must be strictly less than t end."), kind="warn"),
    )

    slice_df = _mdata.extractData(
        keys=mrun.getKeys(),
        trange=[_t0, _t1],
    )
    mo.ui.table(slice_df.head(30), label=f"Slice [{_t0:.1f} – {_t1:.1f} s] — first 30 rows")
    return (slice_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Equivalent CLI command

    ```bash
    python-magnetrun <file.txt> --housing M9 select --output-timerange "0;60"
    ```

    Export a subset of keys:

    ```bash
    python-magnetrun <file.txt> --housing M9 select --output-key Field IH IB
    ```

    Convert the whole file to CSV:

    ```bash
    python-magnetrun <file.txt> --housing M9 select --convert
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## `add` — Compute a derived field
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `add` sub-command computes a new channel from an arbitrary Python
    expression referencing existing key names.

    Below we compute **Field² / IH** as an example.
    """)
    return


@app.cell
def _(mo, mrun):
    _mdata = mrun.getMData()
    _keys = mrun.getKeys()

    mo.stop(
        "Field" not in _keys or "IH" not in _keys,
        mo.callout(
            mo.md(
                "Keys `Field` and `IH` are not present in this file — "
                "the `add` example below requires them."
            ),
            kind="info",
        ),
    )

    _df = _mdata.getData().copy()
    _df["Field2_over_IH"] = _df["Field"] ** 2 / _df["IH"].replace(0, float("nan"))
    mo.ui.table(
        _df[["Field", "IH", "Field2_over_IH"]].head(20),
        label="Derived field: Field² / IH",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Equivalent CLI command

    ```bash
    python-magnetrun <file.txt> --housing M9 add \
        --name Field2_over_IH \
        --formula "Field**2 / IH" \
        --symbol "B²/I" \
        --unit "T²/A"
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
