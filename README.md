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

-   Load txt, cvs and tdms files from control/monitoring system
-   Extract field(s)
-   Plot field(s) vs time
-   Plot field vs field

Examples
--------

_  To list fields recorded during an experiment:

```python3 python_magnetrun.py M9_2019.02.14-23_00_38.txt --list```


_  To view the magnetic field during an experiment:


```python3 python_magnetrun.py M9_2019.02.14-23_00_38.txt --plot_vs_time "Field" --show```

- To model transient:

```python3 clawtest1.py M9_2019.02.14-23_00_38.txt --npts_per_domain=4437 --duration=3600 --ntimes=360```

TODO
----

-   tdms to pandas see
    <https://nptdms.readthedocs.io/en/stable/apireference.html>
-   check addData formula with help of python pyparsing??
-   export Data to prettytables? tabular? cvs2md?
-   how to pass option to matplotlib within plotData()?\*args PARAMETER
    TO MAKE OPTIONAL ARGUMENTS?
-   add support for origin files (for B experimental profil) - use
    labplot?? liborigin?? python bindings??
-   get MagnetRun files from control/monitoring system??
-   how to add columns coming from freesteam, for instance like rho,
    cp\...??
-   for MagnetRun add missing fields [U1, Pe1, Tout1, U2 \...\],
    \--missing, \--nhelices - see txt2csv.py

Credits
-------

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
