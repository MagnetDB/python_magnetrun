import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", title="Part 1 — Pupitre: Loading")


@app.cell
def __(mo):
    mo.md(
        r"""
        # Part 1 — Pupitre Data: Loading

        This notebook shows how to load a Pupitre `.txt` file and inspect its
        content: metadata, column keys, units, and the underlying DataFrame.
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
        ## File picker

        Enter the path to a `.txt` file below.  The sample bundled with the
        test suite is pre-filled so you can run the notebook immediately
        without a NAS connection.
        """
    )
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
    file_input
    return (file_input,)


@app.cell
def __(mo):
    mo.md("### Housing")
    return


@app.cell
def __(mo):
    housing_input = mo.ui.dropdown(
        options=["M9", "M10", "M14", "M19", "unknown"],
        value="M9",
        label="Housing",
    )
    housing_input
    return (housing_input,)


@app.cell
def __(mo):
    mo.md("---\n## Load the run")
    return


@app.cell
def __(file_input, housing_input):
    from python_magnetrun.MagnetRun import load_mrun

    mrun = load_mrun(
        file_input.value,
        housing=housing_input.value,
        auto_resolve=False,
    )
    return load_mrun, mrun


@app.cell
def __(mo, mrun):
    mo.md(
        f"""
        **Run loaded successfully.**

        | Property | Value |
        |----------|-------|
        | Housing | `{mrun.getHousing()}` |
        | File | `{getattr(mrun, 'filename', '—')}` |
        """
    )
    return


@app.cell
def __(mo):
    mo.md("## Keys and units")
    return


@app.cell
def __(mo, mrun):
    import pandas as pd

    keys = mrun.getKeys()
    rows = []
    for k in keys:
        try:
            sym, unit = mrun.getUnit(k)
        except Exception:
            sym, unit = k, "—"
        rows.append({"key": k, "symbol": sym, "unit": unit})

    df_keys = pd.DataFrame(rows)
    mo.ui.table(df_keys, label="Available channels")
    return df_keys, k, keys, pd, rows, sym, unit


@app.cell
def __(mo):
    mo.md("## Raw DataFrame")
    return


@app.cell
def __(mo, mrun):
    df = mrun.getDataFrame()
    mo.md(
        f"""
        Shape: **{df.shape[0]} rows × {df.shape[1]} columns**
        """
    )
    return (df,)


@app.cell
def __(df, mo):
    mo.ui.table(df.head(20), label="First 20 rows")
    return


@app.cell
def __(mo):
    mo.md("## Data summary statistics")
    return


@app.cell
def __(df, mo):
    mo.ui.table(
        df.describe().reset_index().rename(columns={"index": "stat"}),
        label="Descriptive statistics",
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## MagnetData object

        `mrun.getMData()` returns the underlying `PandasMagnetData` object,
        which exposes the full API for derived fields, stats, extraction, and
        export.
        """
    )
    return


@app.cell
def __(mo, mrun):
    mdata = mrun.getMData()
    mo.md(
        f"""
        `type(mdata)` → `{type(mdata).__name__}`

        Available on `mdata`:
        - `mdata.getData()` — returns the raw DataFrame
        - `mdata.stats(key)` — summary stats for one channel
        - `mdata.addData(name, formula=...)` — compute a derived field
        - `mdata.extractTimeData(range_str)` — slice by timestamp
        - `mdata.saveData(keys, filename)` — export to CSV/TSV
        """
    )
    return (mdata,)


if __name__ == "__main__":
    app.run()
