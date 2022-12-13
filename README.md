Python3 MagnetRun
================

[![image](https://img.shields.io/pypi/v/python_magnetrun.svg)](https://pypi.python.org/pypi/python_magnetrun)

[![image](https://img.shields.io/travis/Trophime/python_magnetrun.svg)](https://travis-ci.com/Trophime/python_magnetrun)

[![Documentation Status](https://readthedocs.org/projects/python-magnetrun/badge/?version=latest)](https://python-magnetrun.readthedocs.io/en/latest/?badge=latest)

Python MagnetRun contains utils to view and analyse Magnet runs

-   Free software: MIT license
-   Documentation: <https://python-magnetrun.readthedocs.io>.

Features
--------

-   Extract data from control/monitoring system
-   Inject data from control/monitoring system inot magnetdb
-   Load txt, cvs and tdms files from control/monitoring system
-   Extract field(s)
-   Plot field(s) vs time
-   Plot field vs field

Examples
--------

- To retreive data from control/monitoring system

```python3 -m  python_magnetrun.requests.cli --user email```

_  To list fields recorded during an experiment:

```python3 python_magnetrun.py M9_2019.02.14-23_00_38.txt --list```


_  To view the magnetic field during an experiment:


```python3 python_magnetrun.py M9_2019.02.14-23_00_38.txt --plot_vs_time "Field" --show```

- To model transient:

```python3 clawtest1.py M9_2019.02.14-23_00_38.txt --npts_per_domain=4437 --duration=3600 --ntimes=360```

- To plot temperature during an experiment and compare with NTU model:

```python3 heatexchanger_primary.py M9_2019.02.14-23_00_38.txt --ohtc=2103.09 --dT=4.93827 [find]```


TODO
----

-   tdms to pandas see
    <https://nptdms.readthedocs.io/en/stable/apireference.html>
-   check addData complex formula (involving fresteam for ex) with help of python pyparsing??
-   how to add columns coming from freesteam, for instance like rho,
    cp\...??
-   export Data to prettytables? tabular? cvs2md?
-   how to pass option to matplotlib within plotData()?\*args PARAMETER
    TO MAKE OPTIONAL ARGUMENTS?
-   add support for origin files (for B experimental profil) - use
    labplot?? liborigin?? python bindings??
-   get MagnetRun files from control/monitoring system??
-   for MagnetRun add missing fields [U1, Pe1, Tout1, U2 \...\],
    \--missing, \--nhelices - see txt2csv.py

INSTALL
----

To install in a python virtual env

```
python -m venv --system-site-packages magnetrun-env
source ./magnetrun-env/bin/activate
pip install nptdms
```

Credits
-------

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
