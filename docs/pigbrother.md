# PigBrother Files (.tdms)

## Overview

PigBrother files are produced by the PigBrother power monitoring system using National Instruments LabVIEW.

- **Format**: TDMS (proprietary NI LabVIEW binary format, structurally similar to HDF5)
- **Data organisation**: Groups / Channels hierarchy

## Recording Modes

| Mode | Acquisition rate | Max duration | Max size |
|------|-----------------|--------------|----------|
| Overview | 1 Hz | 10 h | ~10 MB |
| Archive | 120 Hz | 5 h | ~500 MB |
| Manual_Trig | 4800 Hz | 60 s | ~50 MB |

**Statistics** (1 Hz): derived from 4800 Hz data — variance, mean, spectral power.

`Manual_Trig` recordings are triggered automatically on current variations during acquisition.

## Groups and Channels

<!-- >
**Note on units**: All TDMS channels are stored as voltage signals (V). Physical quantities
> (currents in A, magnetic field in T) are obtained by applying sensor calibration factors
> (Hall probes, shunts, current transformers). Group names in the Python interface use
> underscores in place of spaces (e.g. `Courants_Alimentations`).
 -->

### Elec Grid (AC) — group `HT default`

| Channel | Description |
|---------|-------------|
| `HT1_R`, `HT1_S`, `HT1_T` | Three-phase voltages of HV line 1 (15 kV, phases R/S/T) |
| `HT2_R`, `HT2_S`, `HT2_T` | Three-phase voltages of HV line 2 (15 kV, phases R/S/T) |

### AC/DC Conversion — groups `Tensions MC1`, `Tensions MC2`

**`Tensions MC1`** — AC voltages on machine converter 1 (transformer TRA1, supplies A1 & A2):

| Channel | Description |
|---------|-------------|
| `Tension_P_A1`, `Tension_P_A2` | Positive AC bridge voltages for supplies A1, A2 |
| `Tension_N_A1`, `Tension_N_A2` | Negative AC bridge voltages for supplies A1, A2 |

**`Tensions MC2`** — AC voltages on machine converter 2 (transformer TRA2, supplies A3 & A4):

| Channel | Description |
|---------|-------------|
| `Tension_P_A3`, `Tension_P_A4` | Positive AC bridge voltages for supplies A3, A4 |
| `Tension_N_A3`, `Tension_N_A4` | Negative AC bridge voltages for supplies A3, A4 |

### DC Bridge Currents — group `Courants ponts`

| Channel | Description |
|---------|-------------|
| `A1_p1`, `A1_p2`, `A1_p3`, `A1_p4` | Per-bridge currents of thyristor rectifier A1 |

### DC Supply — groups `Courants et Ref. Alimentations`, `Tensions Alimentations`

**`Courants et Ref. Alimentations`** (Python: `Courants_Alimentations`) — DC currents and references:

| Channel | Description |
|---------|-------------|
| `Courant_A1`..`Courant_A4` | DC output current of each power supply (A1–A4) |
| `Référence_A1`..`Référence_A4` | Current reference setpoint for each supply |
| `Courant GR1` | Total current delivered to magnet group GR1 (Helix or Bitter, housing-dependent) |
| `Courant GR2` | Total current delivered to magnet group GR2 |
| `Champ magn` | Magnetic field proxy (Hall probe voltage output) |

> [!NOTE]
> * `Courant GR1` and `Courant GR2` are derived from the DC supply currents and depend on the housing configuration of the attached site.
> For example, on housing M9, `Courant GR1` is the current delivered to the Bitter circuit and `Courant GR2` is the current delivered to the Helix circuit; on M8 it's the opposite.
> * `Champ magn` is a magnetic field proxy derived from a Hall probe voltage output. It is not a direct measurement of the magnetic field in Tesla.
> * Reference setpoints are the values used to control the DC supply currents. They are computed by the control system from the requested Magnetic Field by the user.
> * For convenience, `Référence_GR1` and `Référence_GR2` channels are added when loading the tdms file.


**`Tensions Alimentations`** — DC supply voltages:

| Channel | Description |
|---------|-------------|
| `Tension A1`..`Tension A4` | DC output voltage of each power supply (A1–A4) |

### Magnet Voltages — group `Tensions Aimant`

| Channel | Description |
|---------|-------------|
| `Interne1`..`Interne7` | Individual internal Bitter coil voltages |
| `Externe1`, `Externe2` | Individual external Bitter coil voltages |
| `ALL internes` | Sum voltage across all internal Bitter coils |
| `ALL externes` | Sum voltage across all external Bitter coils |

> [!NOTE]
> * In the event of a missing or faulty voltage sensor -- for example `Interne1` -- there may be extra channel name -- like `Interne12`-- that denotes the sum of voltage sensors -- like `Interne1` + `Interne2`.

### Digital Alarms — group `Numérique Alimentation`

> [!WARNING]
> This group is only present in Default files

Per supply (`A1`..`A4`), boolean fault and alarm status bits:

| Channel pattern | Description |
|----------------|-------------|
| `Ax_Def12` | Fault on bridges 1 & 2 |
| `Ax_Def3` | Fault on bridge 3 |
| `Ax_I_MAX` | Over-current alarm |
| `Ax_10_Eps` | 10% epsilon alarm |
| `Ax_Alarm_P1`..`Ax_Alarme_P4` | Per-bridge alarm flags |

## Electrical Chain Summary

![Power installation diagram](pigbrother_power_chain.png)

```
15 kV distribution cells
  → 15 kV/400 V transformers (TRA1, TRA2)
    → Thyristor bridges (AC → DC)
      → Polarity inverters
        → Bus bars / Switches
          → Cables (70 A/cm² → 700 A/cm²)
            → High field magnets (30000 A/cm²)
```

## Python Interface

TDMS files are loaded into a `TdmsMagnetData` object (Type 1). Data is stored as a `dict[str, pd.DataFrame]` keyed by group name. Channels within a group are accessed with `"Group/Channel"` notation.

See [python_magnetrun/magnetdata_tdms.py](../python_magnetrun/magnetdata_tdms.py) for the implementation.
