import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # `magnetrun-config` — Configuration Management

    CLI entry point: **`magnetrun-config`**

    This notebook covers the three configuration domains managed by the
    `magnetrun-config` unified CLI:

    | Domain | Sub-command | What it manages |
    |--------|-------------|----------------|
    | `housing` | `magnetrun-config housing` | Per-housing sensor role assignments |
    | `field` | `magnetrun-config field` | Field definitions: symbol, unit, label, cross-format aliases |
    | `plot` | `magnetrun-config plot` | Plot style / colour configuration |

    Configuration files live in two locations:

    - **Bundled** (read-only): installed alongside the package
    - **User** (editable): `~/.config/magnetrun/`
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## `housing` — Per-housing sensor role assignments
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    housing_input = mo.ui.dropdown(
        options=["M8", "M9", "M10"],
        value="M9",
        label="Housing",
    )
    housing_input  # noqa: B018
    return (housing_input,)


@app.cell
def _(housing_input):
    from python_magnetrun.housing_config import get_housing_config, show_housing_config

    hcfg = get_housing_config(housing_input.value)
    hcfg  # noqa: B018
    return (hcfg, get_housing_config, show_housing_config)


@app.cell(hide_code=True)
def _(hcfg, mo):
    import dataclasses

    import pandas as pd

    _rows = []
    for _f in dataclasses.fields(hcfg):
        _v = getattr(hcfg, _f.name)
        if not callable(_v) and not _f.name.startswith("_"):
            _rows.append({"field": _f.name, "value": str(_v)})

    mo.ui.table(pd.DataFrame(_rows), label=f"HousingConfig for {hcfg.housing}")
    return (dataclasses, pd)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Apply a runtime override

    The housing config can be overridden at runtime without modifying any file
    — useful when a run was acquired with swapped supplies.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    override_key = mo.ui.dropdown(
        options=["gr1_current", "gr2_current", "gr1_flow", "gr2_flow"],
        value="gr1_current",
        label="Field to override",
    )
    override_val = mo.ui.text(value="IB", label="New value")
    mo.hstack([override_key, override_val], align="start")
    return (override_key, override_val)


@app.cell
def _(get_housing_config, housing_input, mo, override_key, override_val):
    _overrides = {override_key.value: override_val.value}
    try:
        hcfg_overridden = get_housing_config(housing_input.value, overrides=_overrides)
        _orig = get_housing_config(housing_input.value)
        _orig_val = getattr(_orig, override_key.value, "—")
        _new_val = getattr(hcfg_overridden, override_key.value, "—")
        mo.md(f"""
        Override applied in memory (no file changed):

        | Field | Original | Overridden |
        |-------|----------|-----------|
        | `{override_key.value}` | `{_orig_val}` | `{_new_val}` |
        """)
    except (ValueError, OSError) as _e:
        mo.callout(mo.md(f"Override failed: {_e}"), kind="warn")
    return (hcfg_overridden,)


@app.cell(hide_code=True)
def _(mo, housing_input):
    mo.md(f"""
    ### Equivalent CLI commands

    Show housing config:

    ```bash
    magnetrun-config housing show --housing {housing_input.value}
    ```

    Initialise a user-local copy for editing:

    ```bash
    magnetrun-config housing init --housing {housing_input.value}
    ```

    Edit a field in the user copy:

    ```bash
    magnetrun-config housing set --housing {housing_input.value} --gr1-current IB
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## `field` — Field definitions (`*-defs.json`)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    defs_file_input = mo.ui.dropdown(
        options=["pupitre-defs.json", "pigbrother-defs.json", "hybrid-defs.json"],
        value="pupitre-defs.json",
        label="Defs file",
    )
    defs_file_input  # noqa: B018
    return (defs_file_input,)


@app.cell
def _(defs_file_input, mo):
    import pandas as _pd

    from python_magnetrun.field_defs import list_field_defs, resolve_defs_file

    try:
        _path = resolve_defs_file(defs_file_input.value)
        _entries = list_field_defs(_path)
        _rows_fd = []
        for _entry in _entries:
            _aliases = _entry.get("aliases", {})
            _rows_fd.append({
                "field": _entry["field"],
                "symbol": _entry.get("symbol", "—"),
                "unit": str(_entry.get("unit", "—")),
                "label": _entry.get("label", "—"),
                "aliases": ", ".join(f"{k}={v}" for k, v in _aliases.items()),
            })
        _ok = True
    except (ValueError, OSError) as _e:
        _ok = False
        _err = str(_e)

    mo.stop(
        not _ok,
        mo.callout(mo.md(f"Failed to load `{defs_file_input.value}`: `{_err}`"), kind="danger"),
    )

    mo.ui.table(_pd.DataFrame(_rows_fd), label=f"Field definitions — {defs_file_input.value}")
    return (list_field_defs, resolve_defs_file)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cross-format alias lookup

    Each field can carry aliases that map it to names used in other data formats
    (pigbrother TDMS, hybrid kHz/RMS).  This is how the analysis layer
    correlates the same physical channel across acquisition systems.
    """)
    return


@app.cell(hide_code=True)
def _(defs_file_input, mo, list_field_defs, resolve_defs_file):
    import pandas as _pd2

    _path2 = resolve_defs_file(defs_file_input.value)
    _entries2 = list_field_defs(_path2)
    _with_aliases = [e for e in _entries2 if e.get("aliases")]

    _alias_rows = []
    for _e in _with_aliases:
        for _fmt, _name in _e["aliases"].items():
            _alias_rows.append({
                "pupitre_field": _e["field"],
                "format": _fmt,
                "alias_name": _name,
            })

    if _alias_rows:
        mo.ui.table(_pd2.DataFrame(_alias_rows), label="Cross-format aliases")
    else:
        mo.callout(mo.md("No cross-format aliases defined in this file."), kind="info")
    return


@app.cell(hide_code=True)
def _(mo, defs_file_input):
    mo.md(f"""
    ### Equivalent CLI commands

    List all field definitions:

    ```bash
    magnetrun-config field list --defs-file {defs_file_input.value}
    ```

    Add a new field definition:

    ```bash
    magnetrun-config field add --defs-file {defs_file_input.value} \\
        --key MyField --symbol B --unit tesla --description "My field"
    ```

    Add a cross-format alias:

    ```bash
    magnetrun-config field alias --defs-file {defs_file_input.value} \\
        --key MyField --format pigbrother --alias "Group/Channel"
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## `plot` — Plot style configuration
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The plot domain manages colour / style JSON files used by the plotting
    backend.  Bundled defaults are read-only; user overrides live in
    `~/.config/magnetrun/`.

    ```bash
    # Show current plot configuration
    magnetrun-config plot show

    # Initialise a user-editable copy
    magnetrun-config plot init

    # Validate a custom plot config file
    magnetrun-config plot validate --file ~/.config/magnetrun/plot-config.json
    ```
    """)
    return


@app.cell
def _(mo):
    import dataclasses as _dc

    import pandas as _pd3

    from python_magnetrun.plotting.style import PlotConfig as _PlotCfg

    try:
        _pcfg = _PlotCfg()
        _flat = {}
        for _fname, _fval in _dc.asdict(_pcfg).items():
            if isinstance(_fval, dict):
                for _k, _v in _fval.items():
                    _flat[f"{_fname}.{_k}"] = str(_v)[:80]
            else:
                _flat[_fname] = str(_fval)[:80]
        _cfg_rows = [{"setting": k, "value": v} for k, v in _flat.items()]
        mo.ui.table(_pd3.DataFrame(_cfg_rows), label="Default plot configuration")
    except (TypeError, ValueError) as _e:
        mo.callout(mo.md(f"Could not load plot config: `{_e}`"), kind="warn")
    return


if __name__ == "__main__":
    app.run()
