import os
from .MagnetRun import MagnetRun

import matplotlib.pyplot as plt

from .cooling import water
from pint import UnitRegistry

# Load pigbrother file
tdms_file_path = (
    "pigbrotherdata/Fichiers_Data/M10/Overview/M10_Overview_241014-0951.tdms"
)

filename = os.path.basename(tdms_file_path)
index = filename.index("_")
site = filename[0:index]
insert = "tututu"

tdms_data = MagnetRun.fromtdms(site, insert, tdms_file_path).getMData()

# Load pupitre file
pupitre_file_path = "srvdata/M10_2024.10.14---09:51:32.txt"
pupitre_data = MagnetRun.fromtxt(site, insert, pupitre_file_path).getMData()

# Add data to pigbrother
tdms_data.addData(
    key="Puissances/HT1",
    formula="Puissances/HT1 = HT_Courant/HT1_RC * Haute_Tension/HT1_R + HT_Courant/HT1_SC * Haute_Tension/HT1_S + HT_Courant/HT1_TC * Haute_Tension/HT1_T",
)
tdms_data.addData(
    key="Puissances/HT2",
    formula="Puissances/HT2 = HT_Courant/HT2_RC * Haute_Tension/HT2_R + HT_Courant/HT2_SC * Haute_Tension/HT2_S + HT_Courant/HT2_TC * Haute_Tension/HT2_T",
)

tdms_data.addData(
    key="Puissances/A1A2",
    formula="Puissances/A1A2 = Puissances/Puissance_A1 + Puissances/Puissance_A1",
)
tdms_data.addData(
    key="Puissances/A3A4",
    formula="Puissances/A3A4 = Puissances/Puissance_A3 + Puissances/Puissance_A4",
)

tdms_data.addData(
    key="Puissances/Helix",
    formula="Puissances/Helix = Tensions_Aimant/ALL_internes * Courants_Alimentations/Courant_GR2",
)
tdms_data.addData(
    key="Puissances/Bitter",
    formula="Puissances/Bitter = Tensions_Aimant/ALL_externes * Courants_Alimentations/Courant_GR1",
)

# Add data to pupitre
pupitre_data.addData(key="PowerH", formula="PowerH = IH * UH")
pupitre_data.addData(key="PowerB", formula="PowerB = IB * UB")

# Plot Power Balance
my_ax = plt.gca()
for key in ["Puissances/HT1", "Puissances/A1A2", "Puissances/Bitter"]:
    tdms_data.plotData(
        x="t",
        y=key,
        ax=my_ax,
    )

pupitre_data.plotData(x="t", y="PowerB", ax=my_ax)
plt.show()
plt.close()

my_ax = plt.gca()
for key in ["Puissances/HT2", "Puissances/A3A4", "Puissances/Helix"]:
    tdms_data.plotData(
        x="t",
        y=key,
        ax=my_ax,
    )

pupitre_data.plotData(x="t", y="PowerH", ax=my_ax)
plt.show()
plt.close()

# Total power
tdms_data.addData(
    key="Puissances/Total",
    formula="Puissances/Total = (Puissances/HT1 + Puissances/HT2) /1.e+6",
)
my_ax = plt.gca()
pupitre_data.plotData(x="t", y="Ptot", ax=my_ax)
tdms_data.plotData(
    x="t",
    y="Puissances/Total",
    ax=my_ax,
)
plt.show()
plt.close()

# Get PAlim
pupitre_data.addData(key="TAlim", formula="TAlim = (TAlimout - ( TinH + TinB)/2)")

ureg = UnitRegistry()

nkey = "rho"
nkey_unit = ("rho", ureg.kilogram / ureg.meter**3)
nkey_params = ["HPH", "TinH"]
nkey_method = water.getRho
pupitre_data.computeData(nkey_method, nkey, nkey_params, nkey_unit)

nkey = "Cp"  # [kJ / kg·K]
nkey_unit = ("Cp", ureg.joule / ureg.kilogram / ureg.kelvin)
nkey_params = ["HPH", "TinH"]
nkey_method = water.getCp
pupitre_data.computeData(nkey_method, nkey, nkey_params, nkey_unit)
