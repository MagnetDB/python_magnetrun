import argparse
import os
import re
import sys

import matplotlib
from natsort import natsorted

from python_magnetrun.cli_args import create_base_parser
from python_magnetrun.log_utils import get_logger, setup_logging
from python_magnetrun.MagnetRun import MagnetRun
from python_magnetrun.processing.correlations import lag_correlation
from python_magnetrun.processing.trends import trends

logger = get_logger(__name__)
matplotlib.rcParams["text.usetex"] = True

if __name__ == "__main__":
    # Print all arguments
    print("Script name:", sys.argv[0])
    print("All arguments:", sys.argv[1:])

    base_parser = create_base_parser()
    parser = argparse.ArgumentParser(parents=[base_parser])
    parser.add_argument("--save", help="save graphs (png format)", action="store_true")
    parser.add_argument(
        "--show", help="display graphs (X11 required)", action="store_true"
    )
    parser.add_argument("--window", help="rolling window size", type=int, default=10)
    args = parser.parse_args()
    setup_logging(level=args.log_level, log_file=args.log_file)
    logger.debug("args: %s", args)

    tkeys = ["t", "timestamp"]

    threshold_dict = {
        "TinH": 1,
        "HPH": 1,
        "FlowH": 1,
        "RpmH": 1,
        "IH": 1,
        "IH_ref": 1,
        "UH": 0.1,
        "Ucoil1": 1.0e-2,
        "Ucoil2": 1.0e-2,
        "Ucoil3": 1.0e-2,
        "Ucoil4": 1.0e-2,
        "Ucoil5": 1.0e-2,
        "Ucoil6": 1.0e-2,
        "Ucoil7": 1.0e-2,
        "Ucoil8": 1.0e-2,
        "Ucoil9": 1.0e-2,
        "Ucoil10": 1.0e-2,
        "Ucoil11": 1.0e-2,
        "Ucoil12": 1.0e-2,
        "Ucoil13": 1.0e-2,
        "Ucoil14": 1.0e-2,
        "TinB": 0.1,  #
        "HPB": 0.05,
        "FlowB": 0.5,
        "RpmB": 1,
        "IB": 1,
        "IB_ref": 1,
        "UB": 0.1,
        "Ucoil15": 0.1,
        "Ucoil16": 0.1,
        "Tout": 0.05,
        "TAlimout": 0.02,  #
        "debitbrut": 10,  #
        "teb": 0.1,  #
        "tsb": 0.02,
        "Pmagnet": 0.01,
        "Ptot": 0.05,
        "Q": 0.2,
        "BP": 0.1,  #
        "Field": 1.0e-3,
    }

    input_files = natsorted(args.input_file)
    for _i, file in enumerate(input_files):
        f_extension = os.path.splitext(file)[-1]
        if f_extension != ".txt":
            raise RuntimeError("so far file with tdms extension are implemented")

        dirname = os.path.dirname(file)
        filename = os.path.basename(file).replace(f_extension, "")
        print(f"dirname={dirname}, filename={filename}: ", flush=True)

        result = filename.startswith("M")
        insert = "tututu"
        index = filename.index("_")
        site = filename[0:index]

        mrun = MagnetRun.fromtxt(site, insert, file)
        mdata = mrun.getMData()

        DRkeys = [_key for _key in mdata.Keys if re.match(r"DRcoil\d+", _key)]
        Tcalkeys = [_key for _key in mdata.Keys if re.match(r"Tcal\d+", _key)]
        excluded_keys = DRkeys + Tcalkeys + tkeys
        print("DRkeys: ", DRkeys)
        for key in natsorted(mdata.getKeys()):
            if key not in excluded_keys:
                print(key)

                df = mdata.getData(key)
                print(df.describe())
                # std = mdata.getData(key).std().to_numpy()[0]

                # skip empty keys
                if not (df[key] == 0).all():
                    (change, regime, time, value, component) = trends(
                        mdata,
                        "t",
                        key=f"{key}",
                        window=args.window,
                        threshold=threshold_dict[key],
                        show=args.show,
                        save=args.save,
                        debug=args.debug,
                    )

        ts_pupitre = mdata.getData(["timestamp", "Pmagnet"])
        ts_pupitre["Pmagnet"] = ts_pupitre["Pmagnet"] / ts_pupitre["Pmagnet"].max()
        print(ts_pupitre.head())
        ts_pupitre.set_index("timestamp", inplace=True)
        ts_pupitre_data = {
            "field": "Pmagnet",
            "df": ts_pupitre,
            "range": {"start": 0, "end": None},
        }

        # start_index: ochanges[0][0] + (ochanges[0][1] - ochanges[0][0])/2
        # end:_index ochanges[0][1] + (ochanges[0][2] - ochanges[0][1])/2
        ts_overview = mdata.getData(["timestamp", "TAlimout"])
        ts_overview["TAlimout"] = (
            ts_overview["TAlimout"] / ts_overview["TAlimout"].max()
        )
        ts_overview.set_index("timestamp", inplace=True)
        ts_overview_data = {
            "field": "TAlimout",
            "df": ts_overview,
            "range": {"start": 0, "end": None},
        }
        lag = lag_correlation(
            ts_pupitre_data,
            ts_overview_data,
            show=True,
            save=args.save,
        )
