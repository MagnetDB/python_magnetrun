import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # `magnetrun-pigbrother-logparser` — Acquisition Log Parser

    CLI entry point: **`magnetrun-pigbrother-logparser`**

    This notebook parses `LOG_ACQ_ENET.txt` — the LabVIEW/DAQmx acquisition
    system log file co-located with the pigbrother TDMS data files.

    The parser extracts:

    | Event type | Description |
    |-----------|-------------|
    | ENET presence tests | OK/KO per device box |
    | Acquisition start/stop | Groupe, aimant, timestamps |
    | File creation | Archive, Overview, Stats, ManuelTrig |
    | DAQmx errors | Error codes, descriptions, solutions |
    | Fault events | SpikeAimant, DefautNums, Courants50Hz |

    **Typical log location:** `/mnt/LNCMIG-Data/records/pbsurv/LOG_ACQ_ENET.txt`
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Log file selection
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    log_input = mo.ui.text(
        value="",
        placeholder="/mnt/LNCMIG-Data/records/pbsurv/LOG_ACQ_ENET.txt",
        label="Path to LOG_ACQ_ENET.txt",
        full_width=True,
    )
    log_input  # noqa: B018
    return (log_input,)


@app.cell
def _(log_input, mo):
    import os

    mo.stop(
        not log_input.value or not os.path.exists(log_input.value),
        mo.callout(
            mo.md(
                "Enter the path to the `LOG_ACQ_ENET.txt` file above.\n\n"
                "This file lives alongside the pigbrother TDMS data on the NAS, "
                "typically at `/mnt/LNCMIG-Data/records/pbsurv/LOG_ACQ_ENET.txt`."
            ),
            kind="info",
        ),
    )

    from python_magnetrun.runlogs.pigbrother import LogParser

    try:
        parser = LogParser(log_input.value)
        parser.parse()
        _parse_ok = True
    except (UnicodeDecodeError, ValueError, OSError) as _e:
        _parse_ok = False
        _err = str(_e)

    mo.stop(
        not _parse_ok,
        mo.callout(mo.md(f"Failed to parse log:\n\n```\n{_err}\n```"), kind="danger"),
    )

    mo.md(f"""
    **Log parsed successfully.**

    | Property | Value |
    |----------|-------|
    | File | `{log_input.value}` |
    | Events found | `{len(parser.events)}` |
    | Files with errors | `{len(parser.files_with_errors)}` |
    | Fault files | `{len(parser.defaut_files)}` |
    """)
    return (parser, LogParser, os)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Acquisition events timeline
    """)
    return


@app.cell
def _(mo, parser):
    import pandas as pd

    from python_magnetrun.runlogs.pigbrother import EventType

    _acq_events = [
        e for e in parser.events
        if e.event_type in (EventType.ACQ_START, EventType.ACQ_STOP, EventType.FILE_CREATED)
    ]

    _rows = []
    for _e in _acq_events[-100:]:
        _rows.append({
            "timestamp": str(_e.timestamp),
            "type": _e.event_type.name,
            "message": _e.message[:80],
        })

    if _rows:
        mo.ui.table(pd.DataFrame(_rows), label="Last 100 acquisition events")
    else:
        mo.callout(mo.md("No acquisition events found."), kind="info")
    return (pd, EventType)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Files with DAQmx errors
    """)
    return


@app.cell
def _(mo, parser, pd):
    _error_rows = []
    for _file, _errors in parser.files_with_errors.items():
        for _err in _errors:
            _error_rows.append({
                "file": _file,
                "error_code": getattr(_err, "code", "—"),
                "description": str(getattr(_err, "description", _err))[:80],
            })

    if _error_rows:
        mo.ui.table(pd.DataFrame(_error_rows), label="Files with DAQmx errors")
    else:
        mo.callout(mo.md("No files with DAQmx errors found."), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fault / default files
    """)
    return


@app.cell
def _(mo, parser, pd):
    _fault_rows = []
    for _file, _info in parser.defaut_files.items():
        _fault_rows.append({
            "file": _file,
            "type": _info.get("type", "—"),
            "description": str(_info.get("description", "—"))[:80],
        })

    if _fault_rows:
        mo.ui.table(pd.DataFrame(_fault_rows), label="Fault TDMS files")
    else:
        mo.callout(mo.md("No fault files recorded in this log."), kind="info")
    return


@app.cell(hide_code=True)
def _(mo, log_input):
    mo.md(f"""
    ## Equivalent CLI command

    ```bash
    magnetrun-pigbrother-logparser {log_input.value or "<LOG_ACQ_ENET.txt>"} --log-level DEBUG
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
