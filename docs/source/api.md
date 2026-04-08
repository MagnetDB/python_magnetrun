# API Reference

## Core Data Model

The core data model is built around two main classes: {class}`~python_magnetrun.MagnetRun.MagnetRun`
as the top-level container and {class}`~python_magnetrun.magnetdata.MagnetData` as the data
back-end. The abstract base class {class}`~python_magnetrun.magnetdata_base.MagnetDataBase`
defines the common interface; concrete implementations cover pandas DataFrames (pupitre `.txt`
and `.csv` files) and TDMS binary files (pigbrother).

```{eval-rst}
.. autosummary::
   :toctree: generated

   python_magnetrun.MagnetRun
   python_magnetrun.magnetdata
   python_magnetrun.magnetdata_base
   python_magnetrun.magnetdata_pandas
   python_magnetrun.magnetdata_tdms
   python_magnetrun.signature
```

---

## Configuration & Field Definitions

Site-specific sensor wiring (which physical channel plays which role on M8, M9, M10) is
managed by {class}`~python_magnetrun.site_config.SiteConfig`. Convenience functions
{func}`~python_magnetrun.site_config.get_site_config`,
{func}`~python_magnetrun.site_config.load_site_config`, and
{func}`~python_magnetrun.site_config.save_site_config` handle JSON persistence.

Field-definition JSON files that map column names to physical metadata (symbol, unit,
cross-format aliases) are handled by {mod}`python_magnetrun.field_defs`.

```{eval-rst}
.. autosummary::
   :toctree: generated

   python_magnetrun.site_config
   python_magnetrun.field_defs
```

---

## Data Preparation (ETL)

{mod}`python_magnetrun.runetl` provides ETL helpers that normalise raw DataFrames on load:
adding computed columns, renaming keys, and dropping noise channels.

```{eval-rst}
.. autosummary::
   :toctree: generated

   python_magnetrun.runetl
```

---

## Analysis

High-level analysis toolkit that combines data loading, time synchronisation,
distance/similarity metrics, and plotting into a single coherent workflow.

```{eval-rst}
.. autosummary::
   :toctree: generated

   python_magnetrun.analysis.config
   python_magnetrun.analysis.loaders
   python_magnetrun.analysis.synchronization
   python_magnetrun.analysis.metrics
   python_magnetrun.analysis.plotting
   python_magnetrun.analysis.processing
```

---

## Signal Processing

Per-channel signal processing utilities. Each sub-module is standalone and can be imported
independently.

| Sub-module | Purpose |
|---|---|
| `stats` | Summary statistics for run data |
| `filters` | Spike / outlier removal |
| `smoothers` | Savitzky-Golay and LOWESS smoothers |
| `peaks` | Peak detection |
| `plateaux` | Plateau detection |
| `breakingpoints` | Breakpoint / changepoint detection |
| `correlations` | Lag and cross-correlation |
| `distance` | Euclidean, MAE, MAPE distance metrics |
| `trends` | Trend analysis |
| `hysteresis` | Hysteresis curve fitting |
| `fit` | Curve and piecewise-linear fitting |

```{eval-rst}
.. autosummary::
   :toctree: generated

   python_magnetrun.processing.stats
   python_magnetrun.processing.filters
   python_magnetrun.processing.smoothers
   python_magnetrun.processing.peaks
   python_magnetrun.processing.plateaux
   python_magnetrun.processing.breakingpoints
   python_magnetrun.processing.correlations
   python_magnetrun.processing.distance
   python_magnetrun.processing.trends
   python_magnetrun.processing.hysteresis
   python_magnetrun.processing.fit
```

---

## Hybrid / FEPC Data

Unified interface for high-frequency data from FEPC acquisition systems.
Three data types are supported: **kHz** (1 kHz analog/digital cards),
**RMS** (10 Hz root-mean-square files), and **Trigger** (event-triggered snapshots).
A fourth type, **VProcess**, provides post-processed virtual-channel files.

{class}`~python_magnetrun.hybrid.HybridRun` is the recommended high-level interface;
it mirrors {class}`~python_magnetrun.MagnetRun.MagnetRun` and allows direct comparison
with pupitre and TDMS data. {class}`~python_magnetrun.hybrid.HybridData` gives lower-level
access to individual binary files.

```{eval-rst}
.. autosummary::
   :toctree: generated

   python_magnetrun.hybrid.hybrid_run
   python_magnetrun.hybrid.hybrid_data
   python_magnetrun.hybrid.outliers
   python_magnetrun.hybrid.data_protocol
   python_magnetrun.hybrid.utils
   python_magnetrun.hybrid.kHz
   python_magnetrun.hybrid.rms
   python_magnetrun.hybrid.trigger
   python_magnetrun.hybrid.vprocess
```

---

## Utilities

File validation, format checking, and glob expansion.

```{eval-rst}
.. autosummary::
   :toctree: generated

   python_magnetrun.utils.validation
   python_magnetrun.utils.files
```

---

## Thermal & Hydraulic Pipelines

End-to-end pipelines that derive thermal quantities (temperatures, heat loads) and hydraulic
parameters (pump curves, flow rates) directly from
{class}`~python_magnetrun.MagnetRun.MagnetRun` DataFrames.

```{eval-rst}
.. autosummary::
   :toctree: generated

   python_magnetrun.thermal_pipeline
   python_magnetrun.waterflow_pipeline
   python_magnetrun.flow_params
```

---

## Data Acquisition

### Server Requests

Utilities for downloading run records from the control/monitoring server and querying
the User DataBase API.

```{eval-rst}
.. autosummary::
   :toctree: generated

   python_magnetrun.requests.connect
   python_magnetrun.requests.deserialize
```

### TDMS Log Parsing

```{eval-rst}
.. autosummary::
   :toctree: generated

   python_magnetrun.tdms.log_parser
```
