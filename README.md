# Python3 `MagnetRun`

[![image](https://img.shields.io/pypi/v/python_magnetrun.svg)](https://pypi.python.org/pypi/python_magnetrun)

[![image](https://img.shields.io/travis/Trophime/python_magnetrun.svg)](https://travis-ci.com/Trophime/python_magnetrun)

[![Documentation Status](https://readthedocs.org/projects/python-magnetrun/badge/?version=latest)](https://python-magnetrun.readthedocs.io/en/latest/?badge=latest)

Python `MagnetRun` contains utilities to view and analyze Magnet runs

-   Free software: MIT license
-   Documentation: <https://python-magnetrun.readthedocs.io>.

# Installation

# Devcontainer

# Using Python virtual env

To install in a python virtual env on Linux

For Linux/Mac Os X:

```bash
$ python3 -m venv [--system-site-packages] magnetrun-env
$ source ./magnetrun-env/bin/activate
$ python3 -m pip install -r requirements.txt
```

For windows

```bash
c:\>C:\Python35\python -m venv c:\path\to\magnetrun-env
C:\> C:\path\to\magnetrun-env\Scripts\activate.bat
c:\>C:\Python35\python -m pip install -r requirements.txt
```


To quit the virtual env, run `deactivate`.

# Data

## `Pupitre`

You can:

* get data from `Pupitre`using `python_magnetrun.requests.cli` as described in the next section.
* or, mount `Pupitre`data directory ??

## `PigBrother`

To mount `pigbrother` data, you have to:

* create a `pigbrotherdata` directory
* mount data from `pigbrother` server as `pigbrotherdata`:

```bash
sudo mount -v -t cifs //pigbrother_server_ip/d $pwd/pigbrotherdata -o user=pbsurv,password=passwd
```

[NOTE]
====
Adapt the script with the proper variables
====

# Features

-   Extract data from control/monitoring system
-   Inject data from control/monitoring system into `magnetdb`
-   Load `txt`, `cvs` and `tdms` files from control/monitoring system
-   Extract field(s)
-   Plot field(s) vs time
-   Plot field vs field

# Examples

- To retrieve `pupitre` data from control/monitoring system

```bash
python3 -m  python_magnetrun.requests.cli --user email --datadir datadir [--save]
```

- To list fields recorded during an experiment:

```bash
python3 -m python_magnetrun.python_magnetrun srvdata/M9_2019.02.14---23\:00\:38.txt info --list
```


- To view the magnetic field during an experiment:


```bash
python3 -m python_magnetrun.python_magnetrun srvdata/M9_2019.02.14---23\:00\:38.txt plot --vs_time "Field"
```

<!--
- To model transient:

```python3 clawtest1.py M9_2019.02.14-23_00_38.txt --npts_per_domain=4437 --duration=3600 --ntimes=360```

- To plot temperature during an experiment and compare with NTU model:

```python3 heatexchanger_primary.py M9_2019.02.14-23_00_38.txt --ohtc=2103.09 --dT=4.93827 [find]```
 -->

- List records that last at least 60 s and with a magnetic filed above 18:

```bash
python -m python_magnetrun.examples.get-record srvdata/M8*.txt select --duration 60 --field 18.
```

- View Teb vs timestamp for all M8 records (may crash due to high memory usage):

```bash
python -m python_magnetrun.examples.get-record srvdata/M8*.txt plot --xfield timestamp --fields teb --show
```

- Aggregata Teb data

```bash
python -m python_magnetrun.examples.get-record srvdata/M*---*.txt aggregate --fields teb --show
```

- For stats:

```bash
python -m python_magnetrun.python_magnetrun  srvdata/M8*.txt  stats
```

- For all `pupitre` files, perform plateaux detections:


```bash
python -m python_magnetrun.python_magnetrun  srvdata/M8*.txt  stats --plateau
```


- Plot `pigbrother` and `pupitre` current for Helices insert:

```bash
python -m python_magnetrun.python_magnetrun ~/M9_Overview_240509-1634.tdms ~/M9_2024.05.09---16_34_03.txt \
    plot --vs_time Courants_Alimentations/Courant_GR1 --vs_time IH
```

- Detect Breaking points and anomalies:

Example is functional, but the results are good. The method does not work correctly for other examples

```bash
python -m python_magnetrun.python_magnetrun ~/M9_Overview_240509-1634.tdms  stats --show --keys Courants_Alimentations/Référence_GR1 --detect_bkpts --sav
```
- check field factor (not working properly since Ih and Ib are "piecewise" dependant)

better way to do this - see python_magnetrun/corr_Ih_Ib.py with algo=piecewise-regression or pwlf
once the number of breakpoints are known.


```bash
python -m python_magnetrun.test-fieldfactor /home/LNCMI-G/christophe.trophime/M9_2024.05.13---16_30_51.txt
```

The code returns:

```
                            OLS Regression Results
==============================================================================
Dep. Variable:                      Z   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  1.000
Method:                 Least Squares   F-statistic:                 1.656e+13
Date:                Wed, 19 Jun 2024   Prob (F-statistic):               0.00
Time:                        14:38:41   Log-Likelihood:                 5472.3
No. Observations:                 556   AIC:                        -1.094e+04
Df Residuals:                     553   BIC:                        -1.093e+04
Df Model:                           2
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const        1.62e-06   7.81e-07      2.073      0.039    8.52e-08    3.15e-06
X              0.0009   1.34e-08   6.63e+04      0.000       0.001       0.001
Y              0.0004   4.04e-08   9314.560      0.000       0.000       0.000
==============================================================================
Omnibus:                        3.831   Durbin-Watson:                   0.358
Prob(Omnibus):                  0.147   Jarque-Bera (JB):                3.642
Skew:                          -0.190   Prob(JB):                        0.162
Kurtosis:                       3.110   Cond. No.                     5.84e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 5.84e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
Intercept: 1.6196286010253166e-06, A: 0.0008915003451151302, B: 0.0003765980765178872
```

from [MagnetInfo](https://labs.core-cloud.net/ou/UPR3228/MagnetInfo/SitePages/Field-maps.aspx?web=1), we get the field factors
in the table for M9: fh=8.915 unit? , fB=3.766 unit?

Pb here is that Ih and Ib are actually colinear - at least in piecewise manner

- perform piecewise linear regression

 - piecewise_regression for Ih and Ib
```bash
python -m python_magnetrun.corr_Ih_Ib srvdata/M9_2024.11.06---16\:43\:44.txt --xkey IH --ykey IB --algo piecewise_regression --breakpoints 2
```

 - piecewise linear regression for Ih(t)

```bash
python -m python_magnetrun.corr_Ih_Ib srvdata/M9_2024.11.06---16\:43\:44.txt --xkey t --ykey Field --algo piecewise_regression --breakpoints 8
python -m python_magnetrun.corr_Ih_Ib srvdata/M9_2024.11.06---16\:43\:44.txt --xkey t --ykey Field --algo pwlf --breakpoints 11
```

 - ruptures is not working properly ??
```bash
python -m python_magnetrun.corr_Ih_Ib srvdata/M9_2024.11.06---16\:43\:44.txt --xkey t --ykey Field --algo ruptures --breakpoints 11
```

- parameters identification





# To-do

Refactor:
- [ ] Split argparse options into separate python files
- [ ] add an example / a test for each subcommand in python_magnetrun
- [ ] store stats (which? + plateaus?) data (+ duration) in a dataframe, csv file or a db

Docs:
- [X] docs for aggregate
- [ ] add a note to mount pigbrother data
- [ ] add note to mount pupitre data if applicable

Features:
- [ ] magnetrun actually performs "ETL", can I store processed pupitre data into specific file format??

- [ ] Rewrite txt2csv to use methods in `utils` and `plots` ?done?
- [X] For `tdms` to pandas see
    <https://nptdms.readthedocs.io/en/stable/apireference.html>
- [ ] Check `addData` complex formula (involving `freesteam` or `iapws` for ex) with help of python `pyparsing`??
- [ ] How to add columns coming from `freesteam` or from ?iapws?, for instance like rho, cp, \.\.??
- [ ] Export Data to `prettytables`, `tabular` or `cvs2md`?
- [ ] How to pass option to `matplotlib` within `plotData()`: `*args PARAMETER` TO MAKE OPTIONAL ARGUMENTS?
- [ ] Add support for origin files (for B experimental profile) - use `labplot`?? `liborigin`?? Python bindings??

- [ ] Data from M1, M3, M5 and M7 for complete stats ???
- [ ] Get `MagnetRun` files directly from control/monitoring system??
- [ ] For `MagnetRun` add missing fields [U1, Pe1, Tout1, U2 \...\], `--missing, \--nhelices` - see `txt2csv.py` - link with magnetdb (aka depends on msite configs)

- [ ] for plot with multiple keys, improve legend, save df with only selected fields??
- [ ] for select, add multiple criteria - actually only one field value or threshold

- [ ] Test piecewise linear regression or polynomial
- [ ] Cross lag correlations (see chatgpg discussions)

Usage:
- [ ] systematic check of TinH and TinB?
- [ ] view teb data on daily, monthly, yearly
- [ ] teb forecast from previous data??
- [ ] check independant variables (Ih, Teb, ?Qbrut?) on "plateau" exp - as Ib=f(Ih) with f piece wise 1order polynomial
- [ ] extract data from magnet confile?

- [ ] link with magnet user db - see xdds.csv
- [ ] classification of Field profile
- [ ] data from supra??
- [ ] link with magnettools/hifimagnet for R(i) and L(i)
- [ ] extract R(i), L(i) from U,I timeseries - see chatgpt
- [ ] estimation of heat exhchanger params - see NTU and cooling directory
- [ ] Talim: calorimetric balance to get/estimate disspated power in AC/DC converters



# Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
