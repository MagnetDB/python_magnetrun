import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np


from python_magnetrun.MagnetRun import MagnetRun
from python_magnetrun.processing.trends import trends
    


parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="enter input file")
parser.add_argument(
    "--site", help="specify a site (ex. M8, M9,...)", default="M9"
)  # use housing instead
parser.add_argument(
    "--key", help="set key to consider", type=str, default="Field"
)
parser.add_argument(
    "--window", help="set a window", type=int, default=10
) 
parser.add_argument(
    "--threshold", help="set a threshold for detection", type=float, default=1.e-3
) 
parser.add_argument("--save", help="activate plot", action="store_true")
args = parser.parse_args()
print(f"args: {args}", flush=True)

supported_formats = [".txt", ".tdms"]

file = args.input_file
filename = os.path.basename(file)
f_extension = os.path.splitext(file)[-1]
print(f"filename: {filename}, extension: {f_extension}")

index = filename.index("_")
site = filename[0:index]
print(f"site detected: {site}")

insert = "tututu"
tkey = "t"

match f_extension:
    case ".txt":
        mrun = MagnetRun.fromtxt(site, insert, file)
    case ".tdms":
        mrun = MagnetRun.fromtdms(site, insert, file)
    case _:
        raise RuntimeError(
            f"so far file with extension in {supported_formats} are implemented"
        )

mdata = mrun.getMData()

# TODO get key symbol and unit from MagnetRun
key = args.key
signature = trends(mdata, tkey, key, window=args.window, threshold=args.threshold, save=args.save, debug=True)
# print(signature)

