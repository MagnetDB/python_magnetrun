import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", title="python_magnetrun — Data Organization")


@app.cell
def __(mo):
    mo.md(r"""
        # python_magnetrun — Data Organization

        This notebook describes how measurement data is organized on the NAS
        and how `python_magnetrun` locates it at runtime.
        """)
    return


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __(mo):
    mo.md(r"""
        ## Data sources

        Three independent acquisition systems produce data during a magnet run:

        | System | File format | Typical sample rate | Module |
        |--------|-------------|---------------------|--------|
        | **Pupitre** (power + cooling installation) | `.txt` (tab-separated) | 1 Hz | `PandasMagnetData` |
        | **PigBrother** (power installation) | `.tdms` | 10–100 Hz | `TdmsMagnetData` |
        | **Hybrid** (FEPC-LNCMI, FEPC-AUX-LNCMI) | `.bin` (kHz groups) + `.rms` (RMS) + `.bin` (trigger) | 10 kHz / 1 kHz | `HybridRun` |
        """)
    return


@app.cell
def __(mo):
    mo.md(r"""
        ## NAS layout

        ```
        /NAS/
        ├── srv-data-install/
        │   └── <HOUSING>/          # e.g. M9, M10, M14, …
        │       └── YYYY/           # year
        │           └── <run>.txt
        ├── pbsurv/
        │   └── <HOUSING>/
        │       ├── Overview/
        │           └── <HOUSING>_Overview_YYMMDD-HHMM.tdms
        │       ├── Fichiers_Archive/
        │           └── <HOUSING>_Archive_YYMMDD-HHMM.tdms
        │       ├── Fichiers_Default/
        │           └── <HOUSING>_Default_YYMMDD-HHMMSS_<DEFAUT>.tdms
        │       ├── Fichiers_Spike/
        │           └── <HOUSING>_Spikes_YYMMDD-HHMMSS.tdms
        │       ├── Fichiers_stats/
        │           └── <HOUSING>_Stats_YYMMDD-HHMMSS.tdms
        └── CEA/
            └── kHz
                └── YYYY/
                    └── FEPC-LNCMI/
                        ├── HOST_1_DATA.CFG
                        .
                        .
                        ├── HHHOST_1_LIST_0.bin
                        .
                        .
                        └── CLN_1.CNV
                    └── FEPC-AUX-LNCMI/
                        ├── HOST_1_DATA.CFG
                        .
                        .
                        ├── HHHOST_1_LIST_0.bin
                        .
                        .
                        └── CLN_1.CNV
            └── rms
                └── YYYY/
                    ├── FEPC-LNCMI/
                        .
                        .
                        └── FEPC-LNCMI__YYYY-MM-DD_HH00--YYYY-MM-DD_HH+100.rms
                    └── FEPC-AUX-LNCMI/
                        .
                        .
                        └── FEPC-LNCMI__YYYY-MM-DD_HH00--YYYY-MM-DD_HH+100.rms
            └── trigger
                    .
                    .
                    └── TRIGGER__YYYY-MM-DD__HH-MM
                        ├── FEPC-LNCMI/
                        └── FEPC-AUX-LNCMI/
        ```

        The NAS is mounted locally via **rclone** (see `00_nas_setup.py`) or
        accessed directly when running on a LNCMI workstation.
        """)
    return


@app.cell
def __(mo):
    mo.md(r"""
        ## File naming conventions

        ### Pupitre `.txt`

        ```
        2016.06.07 - 10:12:08.txt
        │            └─────── HH:MM:SS — run start time
        └──────────────────── YYYYMMDD — date
        ```

        The first line of a pupitre file contains run metadata:

        ```
        <n_points>  <run_id>  <t_start_unix>  <t_end_unix>  <flags...>
        ```

        Line 2 is the column header row.  All subsequent lines are tab-separated
        numerical values with a `Date` and `Time` column.

        ### PigBrother `.tdms`

        TDMS files follow NI's Technical Data Management Streaming format.
        Each file contains multiple *groups*, each group containing one or more
        *channels*.  `TdmsMagnetData` exposes groups via the `Groups` property.

        ### FEPC `.tdms`

        Hybrid data are organized in 3 directories:

        * **kHz** — raw 10 kHz samples per coil channel.
        * **RMS** — 1 kHz -- dowsampled data from kHz acquisition.
        * **trigger** — event timestamps (defauts -- which ones?)

        Each directory one directory per day or incident with subdirectories:

        * **FEPC-LNCMI** - data specific to the hybrid superconducting magnet
        * **FEPC-AUX-LNCMI** - data specific to resistive magnets

        The actual data files are organized as follows:

          * **kHz** — `HOST_1_LIST_0.bin`, `HOST_1_LIST_1.bin`, … (one per recording channel) + calibration files (`HOST_1_DATA.CFG`, `CLN_1.CNV`, …) + metadata (CFG).
          * **RMS** — `FEPC-LNCMI__YYYY-MM-DD_HH00--YYYY-MM-DD_HH+100.rms`
          * **trigger** - same naming convention as kHz.


        `HybridRun` loads all three together.
        """)
    return


@app.cell
def __(mo):
    mo.md(r"""
        ## Housing identifiers

        | ID | Magnet type | GR1 | GR2 |
        |----|-------------|-------|
        | M9 | Resistive magnets | |  |
        | M10 | Resistive magnets | |  |
        | M8 | Hybrid (resistive + superconducting) | | |

        The housing identifier is passed to `load_mrun` so that the correct
        field-definition file (`<housing>-defs.json`) is resolved automatically.
        """)
    return


@app.cell
def __(mo):
    mo.md(r"""
        ## Runtime data-directory resolution

        `load_mrun` resolves files in this order:

        1. Exact path given by the caller.
        2. Auto-resolve via `PUPITRE_DATA_DIR` / `PIGBROTHER_DATA_DIR` environment
           variables (or `~/.config/python_magnetrun/data_dirs.json`).
        3. Subdirectory `<data_dir>/<housing>/`.

        Set up your local environment once (see `00_nas_setup.py`); after that,
        bare filenames like `"M9_20220330.txt"` resolve automatically.
        """)
    return


@app.cell
def __(mo):
    mo.md(r"""
        ## Quick smoke-test

        The cell below loads the bundled sample file from the test suite to
        confirm the package is importable.
        """)
    return


@app.cell
def __():
    from pathlib import Path

    import python_magnetrun

    sample = (
        Path(python_magnetrun.__file__).parent.parent
        / "tests"
        / "data"
        / "sample_pupitre.txt"
    )
    print("Package version :", python_magnetrun.__version__)
    print("Sample file     :", sample)
    print("Exists          :", sample.exists())
    return Path, python_magnetrun, sample


@app.cell
def __(mo, python_magnetrun, sample):
    from python_magnetrun.MagnetRun import load_mrun

    mrun = load_mrun(str(sample), housing="M9", auto_resolve=False)
    keys = mrun.getKeys()
    mo.md(f"""
        **Run loaded.**
        - Housing: `{mrun.getHousing()}`
        - Columns: `{keys}`
        """)
    return load_mrun, keys, mrun


if __name__ == "__main__":
    app.run()
