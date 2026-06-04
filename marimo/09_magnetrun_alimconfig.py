import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # `magnetrun-alimconfig` — Cirrus XML PID Config Parser

    CLI entry point: **`magnetrun-alimconfig`**

    This notebook parses Cirrus XML configuration files to extract the PID
    regulator parameters used by the current control loops.

    XML files are downloaded from the cirrus feed at:
    `https://srv-data-install.lncmi.cnrs.fr/cirrus/<feed>/xml/`

    | Field | Meaning |
    |-------|---------|
    | `@numero` | Regulator type: 1=Current, 2=Voltage, 3–6=Bridge loops |
    | `@numero_jeu` | Magnet site: 17=M7, 18=M8, 19=M9, 20=M10 |
    | `@numero_seuil` | Threshold level: 1–3 (0–60 A, 60–1000 A, above) |
    | `@Ki`, `@Kp`, `@Kd` | PID gains (−1 means 0) |
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Select XML configuration file(s)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    xml_input = mo.ui.text(
        value="",
        placeholder="/path/to/cirrus_config.xml",
        label="Path to Cirrus XML configuration file",
        full_width=True,
    )
    xml_input  # noqa: B018
    return (xml_input,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Tip** — XML files can be downloaded with:
    >
    > ```bash
    > curl -O https://srv-data-install.lncmi.cnrs.fr/cirrus/A1/xml/latest.xml
    > ```
    >
    > or retrieved automatically via `srvdata-to-magnetrun --load-cirrus`.
    """)
    return


@app.cell
def _(mo, xml_input):
    import os

    mo.stop(
        not xml_input.value or not os.path.exists(xml_input.value),
        mo.callout(
            mo.md(
                "Enter the path to a Cirrus XML file above.\n\n"
                "XML files live next to the TDMS archives on the NAS, or can be "
                "downloaded from the cirrus feed on srv-data-install."
            ),
            kind="info",
        ),
    )

    import json

    import xmltodict

    with open(xml_input.value) as _f:
        _xml = _f.read()

    config_dict = xmltodict.parse(_xml)

    _version = config_dict["configuration"].get("@version_config", "—")
    _date = config_dict["configuration"].get("@date", "—")

    mo.md(f"""
    **File parsed successfully.**

    | Property | Value |
    |----------|-------|
    | Config version | `{_version}` |
    | Config date | `{_date}` |
    | File | `{xml_input.value}` |
    """)
    return (config_dict, json, xmltodict, os)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PID regulator parameters
    """)
    return


@app.cell
def _(config_dict, mo):
    import pandas as pd

    _site_dict = {"17": "M7", "18": "M8", "19": "M9", "20": "M10"}
    _regulator_types = {
        "1": "Current",
        "2": "Voltage",
        "3": "Bridge-3",
        "4": "Bridge-4",
        "5": "Bridge-5",
        "6": "Bridge-6",
    }

    _config_regs = (
        config_dict["configuration"]["CIRRUS_CP"]["analogique"]["regs"]
    )
    _regulators = _config_regs.get("regulateur", [])
    if isinstance(_regulators, dict):
        _regulators = [_regulators]

    _rows = []
    for _reg in _regulators:
        _num = _reg.get("@numero", "—")
        _jeu = _reg.get("@numero_jeu", "—")
        _seuil = _reg.get("@numero_seuil", "—")
        _Ki = _reg.get("@Ki", "—")
        _Kp = _reg.get("@Kp", "—")
        _Kd = _reg.get("@Kd", "—")
        _rows.append({
            "regulator": _regulator_types.get(_num, f"type-{_num}"),
            "site": _site_dict.get(_jeu, _jeu),
            "threshold_level": _seuil,
            "Ki": _Ki,
            "Kp": _Kp,
            "Kd": _Kd,
        })

    mo.ui.table(pd.DataFrame(_rows), label="PID parameters — all regulators")
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Current-loop PID parameters only (numero=1)
    """)
    return


@app.cell
def _(config_dict, mo, pd):
    _site_dict = {"17": "M7", "18": "M8", "19": "M9", "20": "M10"}

    _config_regs = config_dict["configuration"]["CIRRUS_CP"]["analogique"]["regs"]
    _regulators = _config_regs.get("regulateur", [])
    if isinstance(_regulators, dict):
        _regulators = [_regulators]

    _current_rows = []
    for _reg in _regulators:
        if _reg.get("@numero") == "1" and _reg.get("@numero_jeu") in _site_dict:
            _current_rows.append({
                "site": _site_dict[_reg["@numero_jeu"]],
                "threshold_level": _reg.get("@numero_seuil", "—"),
                "Ki": _reg.get("@Ki", "—"),
                "Kp": _reg.get("@Kp", "—"),
                "Kd": _reg.get("@Kd", "—"),
                "rapport_entre_boucle": _reg.get("@rapport_entre_boucle", "—"),
            })

    if _current_rows:
        mo.ui.table(pd.DataFrame(_current_rows), label="Current-loop PID parameters")
    else:
        mo.callout(mo.md("No current-loop (numero=1) entries found."), kind="info")
    return


@app.cell(hide_code=True)
def _(mo, xml_input):
    mo.md(f"""
    ## Equivalent CLI command

    ```bash
    magnetrun-alimconfig {xml_input.value or "<config.xml>"} --debug
    ```

    Process multiple files at once:

    ```bash
    magnetrun-alimconfig *.xml --datadir ./configs
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
