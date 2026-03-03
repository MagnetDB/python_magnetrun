Usage
=====

.. _installation:

Installation
------------

Install from source using pip:

.. code-block:: console

   git clone https://github.com/MagnetDB/python_magnetrun
   cd python_magnetrun
   pip install -e .

To include optional signal-processing extras:

.. code-block:: console

   pip install -e ".[signal]"

To include all development tools:

.. code-block:: console

   pip install -e ".[dev]"

Quick Start
-----------

Loading a magnet run from a text file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from python_magnetrun.MagnetRun import MagnetRun

   mrun = MagnetRun.fromtxt(housing="M9", site="mysite", filename="run.txt")

   # List available data keys
   print(mrun.getKeys())

   # Access the underlying pandas DataFrame
   df = mrun.getData()
   print(df.describe())

Loading from a TDMS file (PigBrother)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   mrun = MagnetRun.fromtdms(site="mysite", insert="M9", filename="run.tdms")

Working with MagnetData directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from python_magnetrun.magnetdata import MagnetData

   data = MagnetData.fromtxt("run.txt")

   # Basic statistics for a field
   stats = data.stats("IH")
   print(stats)

   # Add a derived quantity
   data.addData("IH_ref", "IH_ref = Idcct1 + Idcct2")

Filtering spikes
~~~~~~~~~~~~~~~~

.. code-block:: python

   from python_magnetrun.processing.filters import filterpikes

   mrun = filterpikes(
       mrun,
       key="IH",
       inplace=True,
       threshold=5.0,
       twindows=10,
       debug=False,
       show=False,
       input_file="run.txt",
   )

Command-Line Interface
----------------------

The package installs several CLI commands:

``python-magnetrun``
    Main entry point for viewing and analysing runs.

    .. code-block:: console

       python-magnetrun --help

``magnetrun-analysis``
    Advanced signal analysis.

    .. code-block:: console

       magnetrun-analysis --help

``hybrid-magnetrun``
    Process hybrid magnet data (kHz, RMS, trigger).

    .. code-block:: console

       hybrid-magnetrun --help

``srvdata-to-magnetrun``
    Download runs from the control/monitoring server.

    .. code-block:: console

       srvdata-to-magnetrun --help

``magnetrun-pigbrother-logparser``
    Parse PigBrother TDMS log files.

    .. code-block:: console

       magnetrun-pigbrother-logparser --help
