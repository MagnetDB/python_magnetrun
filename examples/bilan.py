import argparse  # noqa: I001
import os

import matplotlib.pyplot as plt
from pint import UnitRegistry  # noqa: I001

from python_magnetcooling.water_properties import get_cp, get_rho
from python_magnetrun.cli_args import create_base_parser
from python_magnetrun.log_utils import get_logger, setup_logging
from python_magnetrun.MagnetRun import MagnetRun
from python_magnetrun.utils.files import expand_input_files

logger = get_logger(__name__)
command_line = None
base_parser = create_base_parser()
parser = argparse.ArgumentParser("Energy Balance", parents=[base_parser])
parser.add_argument(
    "--show",
    help="display graphs (requires X11 server active)",
    action="store_true",
)
args = parser.parse_args(command_line)
setup_logging(level=args.log_level, log_file=args.log_file)

# Load pigbrother file — expand glob / search in pigbrother_datadir
tdms_files = expand_input_files(args.input_file, {".tdms": args.pigbrother_datadir})
tdms_file_path = tdms_files[0]

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
pupitre_pattern = f"{site}_20{time[0][0:2]}.{time[0][2:4]}.{time[0][4:]}---{time[1][0:2]}:{time[1][2:]}:*.txt"
print(f'pupitre_pattern="{pupitre_pattern}"')

pupitre_files = expand_input_files([pupitre_pattern], {".txt": args.pupitre_datadir})
# just get the first one for the moment, eg. "srvdata/M10_2024.10.14---09:51:32.txt"
pupitre_file_path = pupitre_files[0]
pupitre_data = MagnetRun.fromtxt(site, insert, pupitre_file_path).getMData()

# Add data for HT tension to pigbrother
# watch out HT tension data are derived from sinusoidal signals
status = tdms_data.addData(
    key="Puissances/HT1",
    formula="Puissances/HT1 = HT_Courant/HT1_RC * Haute_Tension/HT1_R + HT_Courant/HT1_SC * Haute_Tension/HT1_S + HT_Courant/HT1_TC * Haute_Tension/HT1_T",
    symbol="P",
    unit="watt",
    label="HT1 Power",
    description="Three-phase HT1 apparent electrical power",
)
if status != 0:
    logger.warning(f"Failed to add Puissances/HT1 (status={status})")
status = tdms_data.addData(
    key="Puissances/HT2",
    formula="Puissances/HT2 = HT_Courant/HT2_RC * Haute_Tension/HT2_R + HT_Courant/HT2_SC * Haute_Tension/HT2_S + HT_Courant/HT2_TC * Haute_Tension/HT2_T",
    symbol="P",
    unit="watt",
    label="HT2 Power",
    description="Three-phase HT2 apparent electrical power",
)
if status != 0:
    logger.warning(f"Failed to add Puissances/HT2 (status={status})")
status = tdms_data.addData(
    key="Puissances/HT2_HT1",
    formula="Puissances/HT2_HT1 = Puissances/HT2 - Puissances/HT1",
    symbol="P",
    unit="watt",
    label="HT2-HT1 Power",
    description="Difference between HT2 and HT1 apparent power",
)
if status != 0:
    logger.warning(f"Failed to add Puissances/HT2_HT1 (status={status})")
my_ax = plt.gca()
tdms_data.plotData(
    x="t",
    y="Puissances/HT2_HT1",
    ax=my_ax,
)
plt.show()
plt.close()

status = tdms_data.addData(
    key="Puissances/A1A2",
    formula="Puissances/A1A2 = Puissances/Puissance_A1 + Puissances/Puissance_A1",
    symbol="P",
    unit="watt",
    label="A1+A2 Power",
    description="Sum of power supply outputs A1 and A2",
)
if status != 0:
    logger.warning(f"Failed to add Puissances/A1A2 (status={status})")
status = tdms_data.addData(
    key="Puissances/A3A4",
    formula="Puissances/A3A4 = Puissances/Puissance_A3 + Puissances/Puissance_A4",
    symbol="P",
    unit="watt",
    label="A3+A4 Power",
    description="Sum of power supply outputs A3 and A4",
)
if status != 0:
    logger.warning(f"Failed to add Puissances/A3A4 (status={status})")

status = tdms_data.addData(
    key="Puissances/Helix",
    formula="Puissances/Helix = Tensions_Aimant/ALL_internes * Courants_Alimentations/Courant_GR2",
    symbol="P",
    unit="watt",
    label="Helix Power",
    description="Helix coil electrical power (voltage × current)",
)
if status != 0:
    logger.warning(f"Failed to add Puissances/Helix (status={status})")
status = tdms_data.addData(
    key="Puissances/Bitter",
    formula="Puissances/Bitter = Tensions_Aimant/ALL_externes * Courants_Alimentations/Courant_GR1",
    symbol="P",
    unit="watt",
    label="Bitter Power",
    description="Bitter coil electrical power (voltage × current)",
)
if status != 0:
    logger.warning(f"Failed to add Puissances/Bitter (status={status})")

status = tdms_data.addData(
    key="Puissances/Busbar",
    formula="Puissances/Busbar = (Puissances/A1A2+Puissances/A3A4) - (Puissances/Bitter + Puissances/Helix)",
    symbol="P",
    unit="watt",
    label="Busbar Power Loss",
    description="Busbar losses: total supply power minus magnet power",
)
if status != 0:
    logger.warning(f"Failed to add Puissances/Busbar (status={status})")
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
status = pupitre_data.addData(
    key="PowerH",
    formula="PowerH = IH * UH",
    symbol="P_H",
    unit="watt",
    label="Upper Magnet Power",
    description="Upper magnet electrical power",
)
if status != 0:
    logger.warning(f"Failed to add PowerH (status={status})")
status = pupitre_data.addData(
    key="PowerB",
    formula="PowerB = IB * UB",
    symbol="P_B",
    unit="watt",
    label="Lower Magnet Power",
    description="Lower magnet electrical power",
)
if status != 0:
    logger.warning(f"Failed to add PowerB (status={status})")

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
status = tdms_data.addData(
    key="Puissances/HT1_MW",
    formula="Puissances/HT1_MW = Puissances/HT1 /1.e+6",
    symbol="P",
    unit="megawatt",
    label="HT1 Power",
    description="HT1 apparent power in MW",
)
if status != 0:
    logger.warning(f"Failed to add Puissances/HT1_MW (status={status})")
status = tdms_data.addData(
    key="Puissances/HT2_MW",
    formula="Puissances/HT2_MW = Puissances/HT2 /1.e+6",
    symbol="P",
    unit="megawatt",
    label="HT2 Power",
    description="HT2 apparent power in MW",
)
if status != 0:
    logger.warning(f"Failed to add Puissances/HT2_MW (status={status})")
status = tdms_data.addData(
    key="Puissances/Total",
    formula="Puissances/Total = Puissances/HT1_MW + Puissances/HT2_MW",
    symbol="P",
    unit="megawatt",
    label="Total Power",
    description="Total electrical power HT1+HT2 in MW",
)
if status != 0:
    logger.warning(f"Failed to add Puissances/Total (status={status})")
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
status = pupitre_data.addData(
    key="TAlim",
    formula="TAlim = (TAlimout - ( TinH + TinB)/2)",
    symbol="ΔT",
    unit="degC",
    label="Cooling ΔT",
    description="Temperature difference between supply and average magnet inlet",
)
if status != 0:
    logger.warning(f"Failed to add TAlim (status={status})")

ureg = UnitRegistry()

nkey_params = ["HPH", "TinH"]
status = pupitre_data.computeData(
    get_rho,
    "rho",
    nkey_params,
    symbol="rho",
    unit=ureg.kilogram / ureg.meter**3,
    label="Water Density",
    description="Cooling water density",
)
if status != 0:
    logger.warning(f"Failed to compute rho (status={status})")

status = pupitre_data.computeData(
    get_cp,
    "Cp",
    nkey_params,
    symbol="Cp",
    unit=ureg.joule / ureg.kilogram / ureg.kelvin,
    label="Specific Heat",
    description="Cooling water specific heat capacity",
)
if status != 0:
    logger.warning(f"Failed to compute Cp (status={status})")
