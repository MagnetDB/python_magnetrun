# Pupitre Files (.txt)

## Overview

Pupitre files are ASCII CSV-like text files produced by the control desk (pupitre) acquisition system.

- **Format**: ASCII, tab/space-separated columns
- **Acquisition rate**: 1 Hz
- **Max size**: ~10 MB (10 h × 3600 s × (datetime + 88 floats))

## File Structure

The file begins with a metadata header line followed by a column header line, then data rows:

```
<run_id>  <start_t>  <end_t>  ...
Date  Time  Field  Tin1  Tin2  Tout  HP1  BP  Flow1  Icoil1  Ucoil1  DRcoil1  ...
2022.03.30  21:55:17  0.5000  8.4  9.0  ...
```

## Fields

Total: `datetime` + `B` (magnetic field) + 87 other channels.

### Magnetic Field

| Field | Description |
|-------|-------------|
| `Field` | ResistiveMagnetic Field (T) -- computed from coil currents using field factors |
| `SupraField` | Supra Magnetic Field (T) -- optional |

### Power Installation

| Field | Description |
|-------|-------------|
| `Idcct1`..`Idcct4` | DC current from each power supply (A) |
| `Pmagnet` | Total electrical power (MW) |
| `Ptot` | Total electrical power (MW) |

> **Note**: `Pmagnet` is computed from coil currents and voltages, not directly measured.

### Hydraulics

| Field | Description |
|-------|-------------|
| `Tin1` | Water inlet temperature — GR1 circuit (°C) |
| `Tin2` | Water inlet temperature — GR2 circuit (°C) |
| `Tout` | Water outlet temperature (°C) |
| `TAlimout` | Cooling supply outlet temperature (°C) |
| `Flow1`, `Flow2` | Water flow rates (l/s) |
| `rpm1`, `rpm2` | Pump rotation speeds (rpm) |
| `HP1`, `HP2` | High-pressure values (bar) |
| `BP` | Low-pressure value (bar) |
| `teb` | Inlet water temperature from external source (°C) |
| `tsb` | Outlet water temperature to external source (°C) |
| `debitbrut` | Raw flow rate (m³/h) |

> **Note**: `Tin1` is for GR1 circuit, `Tin2` is for GR2 circuit — assignment depends on the magnet experimental site.
> For example, on housing M9, `Tin1` is the inlet for the Bitter circuit and `Tin2` is the inlet for the Helix circuit; on M8 it's the opposite.

### Magnets

| Field | Description |
|-------|-------------|
| `Ucoil1`..`Ucoil16` | Coil voltage (V) |
| `Icoil1`..`Icoil16` | Coil current (A) |
| `DRcoil1`..`DRcoil16` | Coil resistance variation (%) |
| `Tcal1`..`Tcal16` | Calibrated coil temperature (°C) |

- **Helices**: indices 1–14. `Icoil1 = Idcct1 + Idcct2` (site-dependent).
- **Bitters**: indices 15–16. `Icoil15 = Idcct3 + Idcct4` (site-dependent).

> **Note**: These columns are "nulled" when there is no PolyHelices insert in the attached site.
> Depending on the site, some of these columns may be missing or nulled (e.g. `Ucoil15`..`Ucoil16` for sites without Bitters).
> `DRcoil` values are computed from `Ucoil` and `Icoil` and represent the coil resistance variation from reference nominal, which is a proxy for coil temperature.
> `Tcal` values are computed from `DRcoil` using a calibration curve and represent an estimation of the coil temperature in °C.
> Depending on the magnet, `DRcoil` and `Tcal` represent either a couple of Helices or an Helice (for indices 1 to 14). For Bitter magnet, it correspond respectively to the inner and outer Bitter.


## Power and Cooling installation synoptic

![installation](installation-synoptic.png)

## Python Interface

Pupitre `.txt` files are loaded via `MagnetRun.fromtxt()` and exposed as a `MagnetData` object (Type 0).

See [python_magnetrun/pupitre.py](../python_magnetrun/pupitre.py) and [tests/data/sample_pupitre.txt](../tests/data/sample_pupitre.txt) for usage examples.

