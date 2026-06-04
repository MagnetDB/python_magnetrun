import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 3 — Overview & Archive: Loading

    This notebook shows how to load pigbrother **Overview** (1 Hz) and
    **Archive** (120 Hz) `.tdms` files and explore their TDMS group/channel
    structure.

    Key differences from pupitre `.txt` files:

    | Property | Pupitre (`.txt`) | Overview / Archive (`.tdms`) |
    |----------|-----------------|------------------------------|
    | Sampling rate | ~1 Hz | Overview: 1 Hz / Archive: 120 Hz |
    | Data source | Control system | Acquisition system (pigbrother) |
    | Structure | Flat columns | Groups → Channels |
    | Time column | `t` (elapsed s) | `t` per group (elapsed s) |

    Sample files in `data/` are pre-filled so you can run this notebook
    without a NAS connection.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## File picker

    Enter the path to an Overview `.tdms` file.  The bundled
    `data/M8_Overview_251105-0949.tdms` is pre-filled.
    """)
    return


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import python_magnetrun

    _root = Path(python_magnetrun.__file__).parent.parent / "data"
    default_overview = str(_root / "M8_Overview_251105-0949.tdms")
    return (default_overview,)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Load Overview
    """)
    return


@app.cell
def _(housing_input, overview_input):
    from python_magnetrun.MagnetRun import load_mrun

    overview_mrun = load_mrun(
        overview_input.value,
        housing=housing_input.value,
        auto_resolve=False,
    )
    return (overview_mrun,)


@app.cell
def _(mo, overview_mrun):
    _mdata = overview_mrun.getMData()
    _start = getattr(_mdata, "start_timestamp", None)
    _end = getattr(_mdata, "end_timestamp", None)
    try:
        _dur = _mdata.getDuration()
        _dur_str = f"{_dur:.1f} s"
    except (AttributeError, KeyError, IndexError, TypeError):
        _dur_str = "—"

    mo.md(f"""
    **Overview loaded successfully.**

    | Property | Value |
    |----------|-------|
    | Housing | `{overview_mrun.getHousing()}` |
    | File | `{getattr(_mdata, 'FileName', '—')}` |
    | Start (UTC) | `{_start}` |
    | End (UTC) | `{_end}` |
    | Duration | `{_dur_str}` |
    | Groups | `{len(_mdata.Groups)}` |
    | Keys (Group/Channel) | `{len(overview_mrun.getKeys())}` |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TDMS Groups & Channels

    Overview and Archive files organise their data into **groups**
    (e.g. `Courants_Alimentations`, `Tensions_Aimant`) each containing
    one or more **channels** (e.g. `Courant_GR1`).

    Keys are returned as `"Group/Channel"` strings by `mrun.getKeys()`.
    """)
    return


@app.cell
def _(mo, overview_mrun):
    import pandas as pd

    _rows = []
    for _k in overview_mrun.getKeys():
        if "/" in _k:
            _g, _c = _k.split("/", 1)
        else:
            _g, _c = "—", _k
        _rows.append({"group": _g, "channel": _c, "key": _k})

    mo.ui.table(pd.DataFrame(_rows), label="All Groups & Channels")
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Inspect a group

    Select a TDMS group to preview its DataFrame — columns, shape,
    and first rows.  After loading, each group's DataFrame already
    contains a `t` column (elapsed seconds) and a `timestamp` column
    (naive UTC).
    """)
    return


@app.cell(hide_code=True)
def _(mo, overview_mrun):
    _groups = sorted({
        k.split("/")[0] for k in overview_mrun.getKeys() if "/" in k
    })
    group_select = mo.ui.dropdown(
        options=_groups,
        value=_groups[0] if _groups else None,
        label="Group",
    )
    group_select  # noqa: B018
    return (group_select,)


@app.cell
def _(group_select, mo, overview_mrun):
    _mdata = overview_mrun.getMData()

    mo.stop(
        not group_select.value,
        mo.callout(mo.md("No group selected."), kind="info"),
    )

    _df = _mdata.getTdmsData(group_select.value, channel=None)
    mo.vstack([
        mo.md(f"""
**Group `{group_select.value}`** — shape: **{_df.shape[0]} rows × {_df.shape[1]} columns**

Columns: {", ".join(f"`{c}`" for c in _df.columns)}
"""),
        mo.ui.table(_df.head(10), label="First 10 rows"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Descriptive statistics
    """)
    return


@app.cell
def _(group_select, mo, overview_mrun, pd):
    _mdata2 = overview_mrun.getMData()

    mo.stop(
        not group_select.value,
        mo.callout(mo.md("No group selected."), kind="info"),
    )

    _df2 = _mdata2.getTdmsData(group_select.value, channel=None)
    _num_cols = [c for c in _df2.columns if c not in ("timestamp",)]
    mo.ui.table(
        _df2[_num_cols].describe().reset_index().rename(columns={"index": "stat"}),
        label=f"Statistics — {group_select.value}",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## NAS file browser — pick file for discovery

    Select an Overview file from the NAS to use with `FileDiscovery` below.
    If the NAS is not mounted the manual path entered at the top is used
    as fallback.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import datetime as _dt2

    _today2 = _dt2.date.today()
    disc_nas_start = mo.ui.date(value=_today2 - _dt2.timedelta(days=30), label="From")
    disc_nas_end = mo.ui.date(value=_today2, label="To")
    mo.hstack([disc_nas_start, disc_nas_end], align="start")
    return (disc_nas_end, disc_nas_start)


@app.cell(hide_code=True)
def _(disc_nas_end, disc_nas_start, housing_input, mo):
    import glob as _glob2
    from pathlib import Path as _Path3

    from python_magnetrun.data_dirs import PIGBROTHER_DATA_DIR as _PB_DIR3
    from python_magnetrun.utils.timestamps import parse_filename_timestamp as _parse_ts2

    _housing2 = housing_input.value
    _disc_dir = _Path3(_PB_DIR3) / _housing2 / "Overview"
    _dstart = disc_nas_start.value
    _dend = disc_nas_end.value

    disc_nas_input = None

    if not _disc_dir.exists():
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    f"NAS directory **`{_disc_dir}`** not accessible — "
                    "the manual path above will be used for discovery."
                ),
                kind="info",
            ),
        )

    _disc_files = sorted(_glob2.glob(str(_disc_dir / "*.tdms")))
    _disc_in_range = [
        _f for _f in _disc_files
        if (_f_dt := _parse_ts2(_f)) is not None and _dstart <= _f_dt.date() <= _dend
    ]

    if not _disc_in_range:
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    f"No Overview files found under `{_disc_dir}` between "
                    f"**{_dstart}** and **{_dend}** — using the manual path for discovery."
                ),
                kind="info",
            ),
        )

    disc_nas_input = mo.ui.dropdown(
        options={_Path3(_f).name: _f for _f in _disc_in_range},
        label="Overview file (for discovery)",
    )
    disc_nas_input  # noqa: B018
    return (disc_nas_input,)


@app.cell(hide_code=True)
def _(disc_nas_input, mo, overview_input):
    disc_path = (
        disc_nas_input.value
        if disc_nas_input is not None and getattr(disc_nas_input, "value", None)
        else overview_input.value
    )
    mo.md(f"Discovery will run on: **`{disc_path}`**")
    return (disc_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## File discovery

    `FileDiscovery.discover()` finds all files related to an Overview:
    **Archive** (120 Hz), **Pupitre** control-system `.txt`, and incident
    files (Default, Trigger, Spike).

    The result is a `FileSet` dataclass with one list per file type.

    ```python
    from python_magnetrun.analysis.loaders import FileDiscovery

    discovery = FileDiscovery(
        pupitre_datadir=PUPITRE_DATA_DIR,
        pigbrother_datadir=PIGBROTHER_DATA_DIR,
    )
    file_set = discovery.discover("M9_Overview_241106-1643.tdms", housing="M9")

    print(file_set.archive)   # list of .tdms archive paths
    print(file_set.pupitre)   # list of .txt pupitre paths
    ```
    """)
    return


@app.cell
def _(housing_input, mo, overview_input):
    from python_magnetrun.analysis.loaders import FileDiscovery as _FileDiscovery
    from python_magnetrun.data_dirs import (
        PIGBROTHER_DATA_DIR as _PB_DIR,
    )
    from python_magnetrun.data_dirs import (
        PUPITRE_DATA_DIR as _PUPITRE_DIR,
    )

    _disc_ok = False
    file_set = None
    _disc_error = ""

    try:
        _discovery = _FileDiscovery(
            pupitre_datadir=_PUPITRE_DIR,
            pigbrother_datadir=_PB_DIR,
        )
        file_set = _discovery.discover(
            overview_input.value,
            housing=housing_input.value,
        )
        _disc_ok = True
    except (ValueError, RuntimeError, OSError) as _e:
        _disc_error = str(_e)

    mo.stop(
        not _disc_ok,
        mo.callout(
            mo.md(f"File discovery failed: `{_disc_error}`"),
            kind="warn",
        ),
    )
    return (file_set,)


@app.cell(hide_code=True)
def _(file_set, mo, pd):
    mo.stop(file_set is None, mo.callout(mo.md("No FileSet."), kind="info"))

    _summary = [
        {"type": "Overview", "count": len(file_set.overview),
         "files": "\n".join(file_set.overview) or "—"},
        {"type": "Archive (120 Hz)", "count": len(file_set.archive),
         "files": "\n".join(file_set.archive) or "—"},
        {"type": "Pupitre (.txt)", "count": len(file_set.pupitre),
         "files": "\n".join(file_set.pupitre) or "—"},
        {"type": "Default (incident)", "count": len(file_set.default),
         "files": "\n".join(file_set.default) or "—"},
        {"type": "Trigger (incident)", "count": len(file_set.trigger),
         "files": "\n".join(file_set.trigger) or "—"},
        {"type": "Spike (incident)", "count": len(file_set.spike),
         "files": "\n".join(file_set.spike) or "—"},
    ]
    mo.ui.table(pd.DataFrame(_summary), label="Discovered file set")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Load Archive

    When Archive files are discovered, they can be loaded with
    `load_mrun()` exactly like an Overview.  The difference is only the
    sampling rate (120 Hz → 120× more rows per second).

    Use `load_files_data()` to load and concatenate multiple Archive
    files into a single DataFrame with a continuous `t` axis:

    ```python
    from python_magnetrun.analysis.loaders import load_files_data

    df_archive = load_files_data(
        file_set.archive,
        housing="M9",
        group="Courants_Alimentations",
        keys=["Courant_GR1", "Référence_GR1"],
    )
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Load bundled Archive sample
    """)
    return


@app.cell(hide_code=True)
def _():
    from pathlib import Path as _Path

    import python_magnetrun as _pmr

    default_archive = str(
        _Path(_pmr.__file__).parent.parent / "data" / "M8_Archive_251105-0949.tdms"
    )
    return (default_archive,)


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
def _(archive_input, housing_input):
    from python_magnetrun.MagnetRun import load_mrun as _load_mrun

    archive_mrun = _load_mrun(
        archive_input.value,
        housing=housing_input.value,
        auto_resolve=False,
    )
    return (archive_mrun,)


@app.cell(hide_code=True)
def _(archive_mrun, mo):
    _a_mdata = archive_mrun.getMData()
    _a_start = getattr(_a_mdata, "start_timestamp", None)
    _a_end = getattr(_a_mdata, "end_timestamp", None)
    try:
        _a_dur = _a_mdata.getDuration()
        _a_dur_str = f"{_a_dur:.1f} s"
    except (AttributeError, KeyError, IndexError, TypeError):
        _a_dur_str = "—"

    _a_groups = sorted({
        k.split("/")[0] for k in archive_mrun.getKeys() if "/" in k
    })

    mo.md(f"""
    **Archive loaded successfully.**

    | Property | Value |
    |----------|-------|
    | Housing | `{archive_mrun.getHousing()}` |
    | File | `{getattr(_a_mdata, 'FileName', '—')}` |
    | Start (UTC) | `{_a_start}` |
    | End (UTC) | `{_a_end}` |
    | Duration | `{_a_dur_str}` |
    | Groups | `{len(_a_mdata.Groups)}` |
    | Keys (Group/Channel) | `{len(archive_mrun.getKeys())}` |
    | Available groups | {", ".join(f"`{g}`" for g in _a_groups)} |

    > The Archive file has the same groups and channels as the Overview but
    > with a **120× higher sampling rate** (120 Hz vs 1 Hz).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## NAS File Browser — Overview files

    Discover Overview `.tdms` files from the NAS without entering a path
    manually.  **Housing** is taken from the dropdown above.

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

    from python_magnetrun.data_dirs import PIGBROTHER_DATA_DIR as _PB_DIR2
    from python_magnetrun.utils.timestamps import parse_filename_timestamp as _parse_ts

    _housing = housing_input.value
    _overview_dir = _Path2(_PB_DIR2) / _housing / "Overview"
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

    _no_file = nas_overview_input is None or not getattr(nas_overview_input, "value", None)
    mo.stop(
        _no_file,
        mo.callout(mo.md("Select an Overview file above to load it."), kind="info"),
    )

    nas_mrun = _load_mrun2(
        nas_overview_input.value,
        housing=housing_input.value,
        auto_resolve=False,
    )
    return (nas_mrun,)


@app.cell(hide_code=True)
def _(mo, nas_mrun, nas_overview_input, pd):
    mo.stop(
        nas_mrun is None,
        mo.callout(mo.md("No NAS Overview loaded — select a file above."), kind="info"),
    )

    _nm = nas_mrun.getMData()
    _nas_rows = [
        {
            "group": k.split("/")[0] if "/" in k else "—",
            "channel": k.split("/")[1] if "/" in k else k,
            "key": k,
        }
        for k in nas_mrun.getKeys()
    ]
    try:
        _nas_dur = f"{_nm.getDuration():.1f} s"
    except (AttributeError, KeyError, IndexError, TypeError):
        _nas_dur = "—"

    mo.vstack([
        mo.md(f"""
**NAS Overview loaded.**

| Property | Value |
|----------|-------|
| Housing | `{nas_mrun.getHousing()}` |
| File | `{nas_overview_input.value}` |
| Start (UTC) | `{_nm.start_timestamp}` |
| Duration | `{_nas_dur}` |
| Groups | `{len(_nm.Groups)}` |
"""),
        mo.ui.table(pd.DataFrame(_nas_rows), label="Groups & Channels"),
    ])
    return


if __name__ == "__main__":
    app.run()
