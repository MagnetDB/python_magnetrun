import argparse
import logging
import os

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


def test_plateau_default():
    from python_magnetrun.processing.plateaux import nplateaus

    mrun = MagnetRun.fromtxt("M9", "notdefined", _default_input)
    mdata = mrun.getMData()
    # see utils/plateaux.py
    pdata = nplateaus(
        mdata,
        xField=("t", *mrun.getUnit("t")),
        yField=("Field", *mrun.getUnit("Field")),
        threshold=1.0e-3,
        num_points_threshold=600,
        show=False,
        save=False,
        verbose=False,
    )
    print(f"plateaux: {len(pdata)}")
    print(pdata)
    assert len(pdata) > 0


if __name__ == "__main__":
    from python_magnetrun.processing.plateaux import nplateaus

    base_parser = create_base_parser()
    parser = argparse.ArgumentParser(
        "Plateau detection for Magnetic Field", parents=[base_parser]
    )
    parser.add_argument(
        "--threshold", help="set a threshold for detection", type=float, default=1.0e-3
    )
    parser.add_argument(
        "--num_points",
        help="set num_points_threshold for detection",
        type=int,
        default=600,
    )
    parser.add_argument("--save", help="activate plot", action="store_true")
    parser.add_argument("--show", help="show plot", action="store_true")
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

    print(
        f"Detection threshold: {args.threshold}, num_points_threshold: {args.num_points}"
    )
    mrun = load_mrun(file, housing, site)

    mdata = mrun.getMData()

    # if tdms, use "" instead of Field

    pdata = nplateaus(
        mdata,
        xField=(tkey, *mrun.getUnit(tkey)),
        yField=("Field", *mrun.getUnit("Field")),
        threshold=args.threshold,
        num_points_threshold=600,
        show=args.show,
        save=args.save,
        verbose=False,
    )
    print(f"plateaux: {len(pdata)}")
    print(pdata)
