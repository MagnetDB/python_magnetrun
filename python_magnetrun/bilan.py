import os
from .MagnetRun import MagnetRun

import matplotlib.pyplot as plt

from .cooling import water
from pint import UnitRegistry

import argparse

command_line = None
parser = argparse.ArgumentParser("Energy Balance")
parser.add_argument("input_file", help="input pigbrother file")
parser.add_argument(
    "--show",
    help="display graphs (requires X11 server active)",
    action="store_true",
)
parser.add_argument(
    "--pigbrother_datadir",
    help="set srvdata dir (default: )",
    type=str,
    default=None,
)
parser.add_argument(
    "--pupitre_datadir",
    help="set srvdata dir (default: )",
    type=str,
    default="srvdata",
)
parser.add_argument("--debug", help="activate debug mode", action="store_true")
args = parser.parse_args(command_line)

# Load pigbrother file
if args.pigbrother_datadir is None:
    tdms_file_path = (
        args.input_file
    )  # "pigbrotherdata/Fichiers_Data/M10/Overview/M10_Overview_241014-0951.tdms"
else:
    tdms_file_path = os.path.join(args.pigbrother_datadir, args.input_file)

filename = os.path.basename(tdms_file_path)
index = filename.index("_")
site = filename[0:index]

# get insert name from magnetdb
insert = "tututu"
print(f"site={site}, insert={insert}")
tdms_data = MagnetRun.fromtdms(site, insert, tdms_file_path).getMData()

# Load pupitre file ( make it more general)
pupitre = filename.replace(".tdms", "")
print(f"pupitre={pupitre}")
(site, ftype, date) = pupitre.split("_")
print(f"site={site}, ftype={ftype}, date={date}")
time = date.split("-")
print(f"time={time}")
pupitre_datadir = args.pupitre_datadir
pupitre_filter = f"{pupitre_datadir}/{site}_20{time[0][0:2]}.{time[0][2:4]}.{time[0][4:]}---{time[1][0:2]}:{time[1][2:]}:*.txt"
print(f'pupitre_filter="{pupitre_filter}"')
import glob

pupitre_files = glob.glob(pupitre_filter)
# just get the first one for the moment, eg. "srvdata/M10_2024.10.14---09:51:32.txt"
pupitre_file_path = pupitre_files[0]
pupitre_data = MagnetRun.fromtxt(site, insert, pupitre_file_path).getMData()

# Add data for HT tension to pigbrother
# watch out HT tension data are derived from sinusoidal signals
tdms_data.addData(
    key="Puissances/HT1",
    formula="Puissances/HT1 = HT_Courant/HT1_RC * Haute_Tension/HT1_R + HT_Courant/HT1_SC * Haute_Tension/HT1_S + HT_Courant/HT1_TC * Haute_Tension/HT1_T",
)
tdms_data.addData(
    key="Puissances/HT2",
    formula="Puissances/HT2 = HT_Courant/HT2_RC * Haute_Tension/HT2_R + HT_Courant/HT2_SC * Haute_Tension/HT2_S + HT_Courant/HT2_TC * Haute_Tension/HT2_T",
)
tdms_data.addData(
    key="Puissances/HT2_HT1",
    formula="Puissances/HT2_HT1 = Puissances/HT2 - Puissances/HT1",
)
my_ax = plt.gca()
tdms_data.plotData(
    x="t",
    y="Puissances/HT2_HT1",
    ax=my_ax,
)
plt.show()
plt.close()

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

tdms_data.addData(
    key="Puissances/Busbar",
    formula="Puissances/Busbar = (Puissances/A1A2+Puissances/A3A4) - (Puissances/Bitter + Puissances/Helix)",
)
my_ax = plt.gca()
tdms_data.plotData(
    x="t",
    y="Puissances/Busbar",
    ax=my_ax,
)
if args.show:
    plt.show()
else:
    extension = "-BusBar.png"
    imagefile = pupitre_file_path.replace(".txt", "") + extension + ".png"
    print("fsave to {imagefile}")
    plt.savefig(imagefile, dpi=300)
plt.close()

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
    key="Puissances/HT1_MW",
    formula="Puissances/HT1_MW = Puissances/HT1 /1.e+6",
)
tdms_data.addData(
    key="Puissances/HT2_MW",
    formula="Puissances/HT2_MW = Puissances/HT2 /1.e+6",
)
tdms_data.addData(
    key="Puissances/Total",
    formula="Puissances/Total = Puissances/HT1_MW + Puissances/HT2_MW",
)
my_ax = plt.gca()
pupitre_data.plotData(x="t", y="Ptot", ax=my_ax)
tdms_data.plotData(
    x="t",
    y="Puissances/HT1_MW",
    ax=my_ax,
)
tdms_data.plotData(
    x="t",
    y="Puissances/HT2_MW",
    ax=my_ax,
)

if args.show:
    plt.show()
else:
    extension = "-EnergyBalance.png"
    imagefile = pupitre_file_path.replace(".txt", "") + extension + ".png"
    print("fsave to {imagefile}")
    plt.savefig(imagefile, dpi=300)
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
