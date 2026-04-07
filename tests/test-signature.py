import argparse
import os

import pandas as pd

from python_magnetrun.MagnetRun import MagnetRun

_default_input = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "M9_2019.02.14-23_00_38.txt",
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "input_file", nargs="?", default=_default_input, help="enter input file"
)
parser.add_argument(
    "--housing", help="specify a housing (ex. M8, M9,...)", default="M9"
)
parser.add_argument("--site", help="specify a site ", default="not defined")
parser.add_argument("--key", help="set key to consider", type=str, default="Field")
parser.add_argument("--window", help="set a window", type=int, default=10)
parser.add_argument(
    "--threshold", help="set a threshold for detection", type=float, default=1.0e-3
)
parser.add_argument("--save", help="activate plot", action="store_true")
args, _unknown = parser.parse_known_args()
print(f"args: {args}", flush=True)

supported_formats = [".txt", ".tdms"]

file = args.input_file
filename = os.path.basename(file)
f_extension = os.path.splitext(file)[-1]
print(f"filename: {filename}, extension: {f_extension}")

index = filename.index("_")
housing = filename[0:index] if args.housing is None else args.housing
print(f"housing detected: {housing}")

site = args.site
insert = args.insert
tkey = "t"

match f_extension:
    case ".txt":
        mrun = MagnetRun.fromtxt(housing, site, file)
    case ".tdms":
        mrun = MagnetRun.fromtdms(housing, site, file)
    case _:
        raise RuntimeError(
            f"so far file with extension in {supported_formats} are implemented"
        )

mdata = mrun.getMData()

# TODO get key symbol and unit from MagnetRun
key = args.key
# signature = trends(mdata, tkey, key, window=args.window, threshold=args.threshold, save=args.save, debug=True)
# print(signature)

from python_magnetrun.signature import Signature  # noqa: E402

signature = Signature.from_mdata(mdata, key, "t", args.threshold)
print("regimes:", len(signature.regimes))
print(signature)


# Column names as strings
column_names = ["time", key]

# Method 1: Create DataFrame using a dictionary
df = pd.DataFrame({column_names[0]: signature.times, column_names[1]: signature.values})


basename = os.path.basename(args.input_file)
keyname = key.replace("/", "_")
csv_filename = basename.replace(f_extension, f"-{keyname}.csv")
print(f"save to {csv_filename}")

df.to_csv(csv_filename, index=False)
