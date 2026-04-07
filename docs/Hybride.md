# Hybrid Data

> **Field definitions:** The machine-readable channel list with symbols, units,
> and cross-format aliases is in
> [`python_magnetrun/hybrid-defs.json`](../python_magnetrun/hybrid-defs.json).
> The old key listing in [`docs/Hybrid_keys.md`](Hybrid_keys.md) is superseded
> by that file — do not edit it directly.
> See the [main README](../README.md#field-definitions-and-site-configuration)
> for management tools (`magnetrun-field-defs`).
>
> **Site configuration:** Housing-dependent role assignments for hybrid channels
> (e.g. which FEPC channel maps to GR1 for M8) are in
> [`python_magnetrun/M8-site-config.json`](../python_magnetrun/M8-site-config.json).
> Only M8 currently runs hybrid data (resistive + SC combined acquisition).

reference: `ASNet-Structure_données_V2.pdf`

* basedir: `~/LNCMIG-Data/CEA`
* kHz: 1 fichier / par carte / par heure, Archivage journée
* rms: 1 fichier par heure, Archivage journée
* trigger:
* vprocess: 1 Hz (1s),  1 fichier par heure,Archivage journée

kHz
├── 2020
│   ├── 2020-01-30
.
.
└── 2025-12-18
    ├── FEPC-AUX-LNCMI
    └── FEPC-LNCMI

vprocess/
├── 2022
│   ├── 2022-06-17
.
.
├── 2024
│   ├── 2024-05-22
├── 2025-01-01
.
.
.
└── 2025-12-18

# Resistifs data

* stored in kHz/YYYY-MM-DD/FEPC-AUX-LNCMI
* 3 types of files:
  * `.cfg`: fichier de configuration
    * format: p30, p31 
    * header:
  Nom du FEPC
  Nb cartes dans le FEPC : 5 = 3 MIVA + 2 MAD (see p28)
  Infos 1ière carte : Fréquence d’échantillonnage (10000), Buffer de quench (Pre = 20s, Post = 50s), Type de carte, Nombre de voies, frequence, buffer (pre, post)
  ----
  Infos 5ième carte : Fréquence d’échantillonnage (10000), Buffer de quench (Pre = 20s, Post = 50s), Type de carte, Nombre de voies
  Liste des noms de variables 1ière carte, Slot 0
  ...
  Liste des noms de variables 5ième carte, Slot 0
  ACTIVED=True???


  * `.CNV`: Fichiers de calibration .CNV des voies de HOST_2?
    * VAR.CNV: VAR=nom paramtre eneregistre -- see p18 
    * format: p34
    * explication: p32
  * `.bin`: (HOST_2 = FEPC-AUX-LNCMI, 5 cartes MIVA/MAD : 5 fichiers / heure) 
    * HHHOST_2_LISTMAD.bin: HH=heure, MAD=numero carte
    * format depend si DIGITAL (see p27) ou ANALOGIQUE (see p26) - type de carte "MAD" vient de cfg

* rms/YYYY-MM-DD/FEPC-AUX-LNCMI
  * FEPC-AUX-LNCMI_YYYY-MM-DD_0000—YYYY-MM-DD_0100.rms, ..., FEPC-AUX-LNCMI_YYYY-MM-DD_2300—YYYY-MM-DD+1_0000.rms
