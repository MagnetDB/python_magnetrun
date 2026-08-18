import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # `magnetrun-analysis` — Overview TDMS Analysis

    CLI entry point: **`magnetrun-analysis`**

    This notebook demonstrates the analysis workflow for magnet overview TDMS
    files.  It covers:

    | Step | What it does |
    |------|-------------|
    | Discovery | Find archive, pupitre, and log files associated with an overview |
    | Load | Load all data sources and align timestamps |
    | Summary | Print a structured record summary |
    | Metrics | Compute Euclidean, MAPE, and DTW distance metrics |
    | Synchronize | Align pupitre clock to overview via cross-correlation |

    > **Note** — this workflow requires TDMS overview files.  If you only have
    > `.txt` pupitre files, use the `python-magnetrun` notebook instead.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## File selection
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    overview_input = mo.ui.text(
        value="",
        placeholder="/path/to/M9_Overview_2025-01-06.tdms",
        label="Path to Overview TDMS file",
        full_width=True,
    )
    overview_input  # noqa: B018
    return (overview_input,)


@app.cell(hide_code=True)
def _(mo):
    housing_input = mo.ui.dropdown(
        options=["M8", "M9", "M10"],
        value="M9",
        label="Housing",
    )
    housing_input  # noqa: B018
    return (housing_input,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optional: data directories

    Leave blank to auto-discover files next to the overview file.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    pigbrother_dir_input = mo.ui.text(
        value="",
        placeholder="/path/to/pigbrother/logs",
        label="Pigbrother log directory (optional)",
        full_width=True,
    )
    pupitre_dir_input = mo.ui.text(
        value="",
        placeholder="/path/to/pupitre/txt",
        label="Pupitre data directory (optional)",
        full_width=True,
    )
    tdms_dir_input = mo.ui.text(
        value="",
        placeholder="/path/to/archive/tdms",
        label="Archive TDMS directory (optional)",
        full_width=True,
    )
    mo.vstack([pigbrother_dir_input, pupitre_dir_input, tdms_dir_input])
    return (pigbrother_dir_input, pupitre_dir_input, tdms_dir_input)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Processing options
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    synchronize_input = mo.ui.checkbox(label="Synchronize pupitre clock", value=False)
    distance_input = mo.ui.checkbox(label="Compute distance metrics", value=False)
    flow_params_input = mo.ui.checkbox(label="Compute flow parameters", value=False)
    mo.hstack([synchronize_input, distance_input, flow_params_input])
    return (synchronize_input, distance_input, flow_params_input)


@app.cell
def _(
    distance_input,
    flow_params_input,
    housing_input,
    mo,
    overview_input,
    pigbrother_dir_input,
    pupitre_dir_input,
    synchronize_input,
    tdms_dir_input,
):
    import os

    mo.stop(
        not overview_input.value or not os.path.exists(overview_input.value),
        mo.callout(
            mo.md(
                "Enter a valid path to an Overview TDMS file above to run the analysis.\n\n"
                "**Typical location:** `<pupitre_datadir>/<Housing>/Overview/`"
            ),
            kind="info",
        ),
    )

    from python_magnetrun.analysis.processing import (
        ProcessingConfig,
        process_overview_file,
    )

    _config = ProcessingConfig(
        housing=housing_input.value,
        synchronize=synchronize_input.value,
        compute_distance=distance_input.value,
        compute_flow_params=flow_params_input.value,
        pigbrother_datadir=pigbrother_dir_input.value or None,
        pupitre_datadir=pupitre_dir_input.value or None,
        tdms_datadir=tdms_dir_input.value or None,
    )

    try:
        record = process_overview_file(
            overview_file=overview_input.value,
            config=_config,
        )
        _load_ok = True
    except (ValueError, RuntimeError, OSError) as _e:
        _load_ok = False
        _err = str(_e)

    mo.stop(
        not _load_ok,
        mo.callout(mo.md(f"Failed to process file:\n\n```\n{_err}\n```"), kind="danger"),
    )

    mo.md("**Overview file processed successfully.**")
    return (record, ProcessingConfig, process_overview_file, os)


@app.cell(hide_code=True)
def _(mo, record):
    import pandas as pd

    from python_magnetrun.analysis.processing import summarize_record

    _summary = summarize_record(record)
    _rows = [{"property": k, "value": str(v)} for k, v in _summary.items()]
    mo.ui.table(pd.DataFrame(_rows), label="Record summary")
    return (summarize_record, pd)


@app.cell(hide_code=True)
def _(mo, record):
    mo.md(r"""
    ## Data sources loaded

    The record bundles all data sources discovered alongside the overview file.
    """)
    _sources = []
    if record.overview_data is not None:
        _sources.append("overview (TDMS)")
    if record.archive_data is not None:
        _sources.append("archive (TDMS)")
    if record.pupitre_data is not None:
        _sources.append("pupitre (.txt)")
    if record.hybrid_data is not None:
        _sources.append("hybrid (kHz / RMS)")
    if _sources:
        mo.md("Loaded: " + ", ".join(f"**{s}**" for s in _sources))
    else:
        mo.callout(mo.md("No associated data sources found."), kind="info")
    return


@app.cell(hide_code=True)
def _(distance_input, mo, record):
    mo.stop(
        not distance_input.value or record.distances is None,
        mo.callout(
            mo.md("Enable **Compute distance metrics** above to see distance results."),
            kind="info",
        ),
    )
    mo.md("## Distance metrics")
    return


@app.cell(hide_code=True)
def _(distance_input, mo, record, pd):
    mo.stop(
        not distance_input.value or record.distances is None,
        None,
    )

    _dist_rows = []
    for _pair, _d in record.distances.items():
        _dist_rows.append({
            "pair": str(_pair),
            "euclidean": getattr(_d, "euclidean", "—"),
            "mape": getattr(_d, "mape", "—"),
            "dtw": getattr(_d, "dtw", "—"),
        })
    mo.ui.table(pd.DataFrame(_dist_rows), label="Distance metrics")
    return


@app.cell(hide_code=True)
def _(mo, housing_input, overview_input, synchronize_input, distance_input, flow_params_input):
    _flags = []
    if synchronize_input.value:
        _flags.append("--synchronize")
    if distance_input.value:
        _flags.append("--distance")
    if flow_params_input.value:
        _flags.append("--flow-params")

    mo.md(f"""
    ## Equivalent CLI command

    ```bash
    magnetrun-analysis {overview_input.value or "<file.tdms>"} \\
        --housing {housing_input.value} \\
        {" ".join(_flags + ["--save", "--backend plotly"])}
    ```

    For a dry run (discover files only, no data loading):

    ```bash
    magnetrun-analysis {overview_input.value or "<file.tdms>"} --dry-run
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
