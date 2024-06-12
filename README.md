# Python3 `MagnetRun`

[![image](https://img.shields.io/pypi/v/python_magnetrun.svg)](https://pypi.python.org/pypi/python_magnetrun)

[![image](https://img.shields.io/travis/Trophime/python_magnetrun.svg)](https://travis-ci.com/Trophime/python_magnetrun)

[![Documentation Status](https://readthedocs.org/projects/python-magnetrun/badge/?version=latest)](https://python-magnetrun.readthedocs.io/en/latest/?badge=latest)

Python `MagnetRun` contains utilities to view and analyze Magnet runs

-   Free software: MIT license
-   Documentation: <https://python-magnetrun.readthedocs.io>.

# Features

-   Extract data from control/monitoring system
-   Inject data from control/monitoring system into `magnetdb`
-   Load `txt`, `cvs` and `tdms` files from control/monitoring system
-   Extract field(s)
-   Plot field(s) vs time
-   Plot field vs field

# Examples

- To retrieve data from control/monitoring system

```bash
python3 -m  python_magnetrun.requests.cli --user email --datadir datadir [--save]
```

_ To list fields recorded during an experiment:

```bash
python3 -m python_magnetrun.utils.txt2csv data/M9_2019.02.14-23_00_38.txt --list
```


_ To view the magnetic field during an experiment:


```bash
python3 -m python_magnetrun.utils.txt2csv data/M9_2019.02.14-23_00_38.txt --plot_vs_time "Field" --show
```

<!--
- To model transient:

```python3 clawtest1.py M9_2019.02.14-23_00_38.txt --npts_per_domain=4437 --duration=3600 --ntimes=360```

- To plot temperature during an experiment and compare with NTU model:

```python3 heatexchanger_primary.py M9_2019.02.14-23_00_38.txt --ohtc=2103.09 --dT=4.93827 [find]```
 -->

- List records that last at least 60 s and with a magnetic filed above 18:

```bash
python -m python_magnetrun.examples.get-record 'python_magnetrun/txt/M8*.txt' select --duration 60 --field 18.
python -m python_magnetrun.examples.get-record 'python_magnetrun/txt/M*.txt' plot --xfield timestamp --fields teb --show
```

- For all pupitre files, perform plateaux detections:

```bash
python -m python_magnetrun.python_magnetrun ~/M9_Overview_240430*.tdms  stats --show --keys Courants_Alimentations/Référence_A1 --threshold 1
```

# INSTALL

To install in a python virtual env on linux

```bash
python -m venv --system-site-packages magnetrun-env
source ./magnetrun-env/bin/activate
pip install -r requirements.txt
```

On Windows:

```cmd
python -m venv --system-site-packages c:\path\to\magnetrun-env
c:\path\to\magnetrun-env\Scripts\activate.bat
pip install -r requirements.txt
```

To quit the virtual env, run `deactivate`.



# TODO

- Rewrite txt2csv to use methods in `utils` and `plots`
- For `tdms` to pandas see
    <https://nptdms.readthedocs.io/en/stable/apireference.html>
- Check `addData` complex formula (involving `freesteam` for ex) with help of python `pyparsing`??
- How to add columns coming from `freesteam`, for instance like rho, cp, \.\.??
- Export Data to `prettytables`, `tabular` or `cvs2md`?
- How to pass option to `matplotlib` within `plotData()`: `*args PARAMETER` TO MAKE OPTIONAL ARGUMENTS?
- Add support for origin files (for B experimental profile) - use `labplot`?? `liborigin`?? Python bindings??
- Get `MagnetRun` files from control/monitoring system??
- For `MagnetRun` add missing fields [U1, Pe1, Tout1, U2 \...\], `--missing, \--nhelices` - see `txt2csv.py`

# Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
