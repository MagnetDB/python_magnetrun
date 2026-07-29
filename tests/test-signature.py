import argparse
import logging
import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be set before any plt import

import pandas as pd

from python_magnetrun.cli_args import create_base_parser
from python_magnetrun.log_utils import setup_logging
from python_magnetrun.MagnetRun import MagnetRun, load_mrun

_default_input = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "M9_2019.02.14---23_00_38.txt",
)

logger = logging.getLogger("python_magnetrun")


def test_signature_default():
    from python_magnetrun.signature import Signature

    mrun = MagnetRun.fromtxt("M9", "notdefined", _default_input)
    mdata = mrun.getMData()
    signature = Signature.from_mdata(mdata, "Field", "t", 1.0e-3)
    print(f"regimes: {len(signature.regimes)}")
    print(signature)
    assert len(signature.regimes) > 0


def test_signature_compact():
    from datetime import datetime

    from python_magnetrun.signature import Signature

    signature = Signature(
        name="Field",
        symbol="B",
        unit="T",
        t0=datetime.now(),
        timeshift=0,
        changes=[0, 5, 10, 15, 20],
        regimes=["U", "U", "P", "D", "D"],
        times=[0, 5, 10, 15, 20],
        values=[0, 10, 10, 5, 0],
    )

    signature.compact()

    assert signature.changes == [0, 10, 15]
    assert signature.regimes == ["U", "P", "D"]
    assert signature.times == [0, 10, 15]
    assert signature.values == [0, 10, 5]


def test_signature_compact_no_duplicates():
    from datetime import datetime

    from python_magnetrun.signature import Signature

    signature = Signature(
        name="Field",
        symbol="B",
        unit="T",
        t0=datetime.now(),
        timeshift=0,
        changes=[0, 5, 10],
        regimes=["U", "P", "D"],
        times=[0, 5, 10],
        values=[0, 10, 5],
    )

    signature.compact()

    assert signature.changes == [0, 5, 10]
    assert signature.regimes == ["U", "P", "D"]
    assert signature.times == [0, 5, 10]
    assert signature.values == [0, 10, 5]


def test_signature_compact_all_same():
    from datetime import datetime

    from python_magnetrun.signature import Signature

    signature = Signature(
        name="Field",
        symbol="B",
        unit="T",
        t0=datetime.now(),
        timeshift=0,
        changes=[0, 5, 10],
        regimes=["U", "U", "U"],
        times=[0, 5, 10],
        values=[0, 10, 20],
    )

    signature.compact()

    assert signature.changes == [0]
    assert signature.regimes == ["U"]
    assert signature.times == [0]
    assert signature.values == [0]


def test_signature_plot():
    import matplotlib.axes

    from python_magnetrun.signature import Signature

    mrun = MagnetRun.fromtxt("M9", "notdefined", _default_input)
    mdata = mrun.getMData()
    signature = Signature.from_mdata(mdata, "Field", "t", 1.0e-3)

    ax = signature.plot(show=False, save=False)

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.lines) > 0
    assert len(ax.patches) == max(0, min(len(signature.regimes), len(signature.times) - 1))


if __name__ == "__main__":
    from python_magnetrun.signature import Signature  # noqa: E402

    base_parser = create_base_parser()
    parser = argparse.ArgumentParser("Record signature", parents=[base_parser])
    parser.add_argument("--key", help="set key to consider", type=str, default="Field")
    parser.add_argument("--window", help="set a window", type=int, default=10)
    parser.add_argument(
        "--threshold", help="set a threshold for detection", type=float, default=1.0e-3
    )
    parser.add_argument("--save", help="activate plot", action="store_true")
    args, _unknown = parser.parse_known_args()
    if _unknown:
        parser.error(f"unrecognized arguments: {' '.join(_unknown)}")
    setup_logging(level=args.log_level, log_file=args.log_file)
    logger.debug(f"args: {args}")

    file = args.input_file[0]
    filename = os.path.basename(file)
    f_extension = os.path.splitext(file)[-1]
    print(f"filename: {filename}, extension: {f_extension}")

    try:
        index = filename.index("_")
        housing = filename[0:index] if args.housing is None else args.housing
        print(f"housing detected: {housing}")
    except ValueError:
        housing = args.housing if args.housing is not None else "notdefined"
        print(f"no housing detected - use args.housing {housing} argument instead")

    site = args.site
    insert = args.insert
    tkey = "t"

    # Resolve bare filename against predefined data directories
    if not os.path.exists(file):
        if f_extension == ".tdms":
            datadir = os.path.join(args.pigbrother_datadir, housing, "Overview")
        elif f_extension == ".txt":
            datadir = os.path.join(args.pupitre_datadir, housing)
        else:
            raise ValueError(f"Unsupported file extension: {f_extension}")
        candidate = os.path.join(datadir, filename)
        if os.path.exists(candidate):
            file = candidate
            print(f"resolved file: {file}")
        else:
            raise FileNotFoundError(
                f"File not found: {args.input_file[0]!r}\nAlso searched: {datadir}"
            )

    mrun = load_mrun(file, housing, site)

    mdata = mrun.getMData()

    # TODO get key symbol and unit from MagnetRun
    key = args.key
    # signature = trends(mdata, tkey, key, window=args.window, threshold=args.threshold, save=args.save, debug=True)
    # print(signature)

    signature = Signature.from_mdata(mdata, key, "t", args.threshold)
    print("regimes:", len(signature.regimes))
    print(signature)

    # Column names as strings
    column_names = ["time", key]

    df = pd.DataFrame(
        {column_names[0]: signature.times, column_names[1]: signature.values}
    )

    basename = os.path.basename(args.input_file[0])
    keyname = key.replace("/", "_")
    csv_filename = basename.replace(f_extension, f"-{keyname}.csv")
    print(f"save to {csv_filename}")

    df.to_csv(csv_filename, index=False)
