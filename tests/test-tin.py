"""
view R(I) as function of Tin
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


import re
from natsort import natsorted
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "input_files",
    nargs="+",
    help="enter input file(s) (ex: ~/R_14Helices_Tinit=20degCelsius.csv)",
)
parser.add_argument(
    "--ikey",
    help="specify I key (starting from 1)",
    type=int,
    default=1,
)
parser.add_argument("--list", help="list valid ikeys values", action="store_true")
parser.add_argument("--debug", help="acticate debug", action="store_true")
parser.add_argument("--save", help="save graphs (png format)", action="store_true")
args = parser.parse_args()
print(f"args: {args}", flush=True)

# set width of bar
width = 0.2

Tindex = []
Rvalues = []
RH = {}

for i, file in enumerate(natsorted(args.input_files)):
    print(f"file={file}:", end=" ")
    with open(file, "r") as f:
        # extract Tin from file
        match = re.search(r"Tinit=(\d+\.\d+|\d+)", file)
        if match:
            Tin = float(match.group(1))
        print(f"Tin={Tin}")

        Data = pd.read_csv(f)
        Keys = Data.columns.values.tolist()
        if args.debug:
            print(f"Keys={Keys}")
            print(Data.head())

        if args.list:
            print(f"Ikeys: {Keys[1:]}")

        # first column: R
        print(Data.iloc[:, 0].tolist())

        # get total Resistance
        Rtotal = Data.iloc[-1, args.ikey]

        xkeys = [key.replace("[ohm]", "") for key in Data.iloc[0:-1, 0].tolist()]
        xkeys = [key.replace("R_", "") for key in xkeys]
        x = np.arange(len(xkeys))
        offset = width * i
        plt.bar(
            x + offset,
            Data.iloc[0:-1, args.ikey],
            width,
            label=f"I={Keys[args.ikey]} A,  R={Rtotal:.4f} Ohm, Tin={Tin} C ",
        )

        for j, key in enumerate(xkeys):
            if key not in RH:
                RH[key] = []

            RH[key].append(Data.iloc[j, args.ikey])

        Tindex.append(Tin)
        Rvalues.append(Rtotal)

plt.xticks(x + width / 2, xkeys)
plt.ylabel("Ohm")
plt.grid()
plt.legend()

if not args.save:
    plt.show()
else:
    plt.savefig("R_vs_Tin.png", dpi=300)

plt.close()

# resistance per xkeys
for key in xkeys:
    print(f"RH[{key}]: {pd.DataFrame(RH[key], index=Tindex, columns=['R']).describe()}")

# global resistance
R = pd.DataFrame(Rvalues, index=Tindex, columns=["R"])
print(Rvalues)
print(R.head())
R.plot(marker="o")
plt.grid()
plt.title(f"Rtotal at I={Keys[args.ikey]} A")
plt.ylabel("Ohm")
plt.xlabel("Tin [°C]")
plt.show()
plt.close()
