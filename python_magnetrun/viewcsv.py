import os
import pandas as np
from .magnetdata import MagnetData as mdata

#import matplotlib
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_file", nargs="+", help="enter input file")
parser.add_argument(
    "--xkey",
    type=str,
    help="select xkey",
    default="timestamp",
)
parser.add_argument(
    "--ykey",
    type=str,
    help="select ykey",
    default="IB",
)
parser.add_argument("--info", help="list keys", action="store_true")
args = parser.parse_args()
print(f"args: {args}", flush=True)

xkey = args.xkey
ykey = args.ykey

my_ax = plt.gca()

for file in args.input_file:
    mdata = mdata.fromcsv(file)
    if args.info:
        print(mdata.getKeys())
    mdata.Units()
    
    filename = os.path.basename(file)
    f_extension = os.path.splitext(file)[-1]
    label=filename.replace(f_extension,'')

    mdata.plotData(xkey, ykey, ax=my_ax, label=label)
    plt.show()
    plt.close()