import os
from datetime import datetime, timedelta
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsorted
from .MagnetRun import MagnetRun

from .utils.convert import convert_to_timestamp

from .processing.correlations import compute_lag
from .signature import Signature

from scipy import stats
from .processing.distance import calc_euclidean, calc_mape, calc_correlation
from sympy import Symbol
from tabulate import tabulate

import json

from .utils.files import extract_data, load_data, merge_data, select_files, find_files


def compute_mode(mdata, group: str, Ikeys_ref: list) -> dict:
    """
    Compute the operation mode (normal or ecomode) based on reference currents.

    Args:
        mdata: MagnetData object containing the data
        group: Group name for data access
        Ikeys_ref: List of reference current keys

    Returns:
        Dictionary with mode information including name, Intercept, Slopes, and Breakpoint
    """
    from .flow_params import pwlf_fit

    mode = {"name": "normal", "Intercept": 0, "Slopes": [1], "Breakpoint": None}

    if len(Ikeys_ref) == 2:
        GR = mdata.getData(
            [
                f"{group}/Référence_GR1",
                f"{group}/Référence_GR2",
            ]
        ).copy()

        GR = GR.query("`Référence_GR1` >= 300 and `Référence_GR2` >= 300")

        if GR.empty:
            print(
                "Warning: No data points found with Référence_GR1 >= 300 and Référence_GR2 >= 300. Using default normal mode."
            )
            return mode

        x = GR["Référence_GR1"].to_numpy()
        y = GR["Référence_GR2"].to_numpy()
        for segment in [1, 2]:
            (mfit, eqns) = pwlf_fit(
                "Référence_GR1",
                x,
                "Référence_GR2",
                y,
                degree=1,
                segment=segment,
            )
            # TODO if error ?my_pwlf.standard_errors()? on brkpoints is big, try with 1 segment
            I0 = eqns[0].evalf(subs={Symbol("x"): 0})
            if abs(I0) <= 10:
                break

        # mode: Intercept, slopes, brkpts
        if segment == 1:
            mode = {
                "name": "normal",
                "Intercept": eqns[0].evalf(subs={Symbol("x"): 0}),
                "Slopes": [float(mfit.beta[1])],
                "Breakpoint": None,
            }
        else:
            mode = {
                "name": "ecomode",
                "Intercept": eqns[0].evalf(subs={Symbol("x"): 0}),
                "Slopes": [
                    float(mfit.beta[1]),
                    float(mfit.beta[1] + mfit.beta[2]),
                ],
                "Breakpoint": float(mfit.fit_breaks[1]),
            }
        print(f"mode:{mode}")
        del GR

    return mode


# new default for pupitre-datadir: srvdata -> /home/LNCMI-G/christophe.trophime/LNCMIG-Data/srv-data-install
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="+", help="enter input file tdms")
    parser.add_argument(
        "--pupitre_datadir",
        help="enter pupitre datadir (default srvdata)",
        type=str,
        default="/home/LNCMI-G/christophe.trophime/LNCMIG-Data/srv-data-install",
    )
    parser.add_argument(
        "--pigbrother_datadir",
        help="enter pigbrother datadir (default pigbrotherdata)",
        type=str,
        default="/home/LNCMI-G/christophe.trophime/github/python_magnetrun/pigbrotherdata/Fichiers_Data",
    )

    parser.add_argument("--logs", nargs="+", help="enter log files from ACQ_ENET")
    parser.add_argument(
        "--log_datadir",
        help="enter log datadir (default srvdata)",
        type=str,
        default="/home/LNCMI-G/christophe.trophime/LNCMIG-Data/srv-data-install",
    )

    parser.add_argument(
        "--tkey",
        help="choose tkey",
        choices=["t", "timestamp"],
        type=str,
        default="t",
    )

    parser.add_argument("--dry_run", help="dry_run mode", action="store_true")
    parser.add_argument("--debug", help="activate debug", action="store_true")
    parser.add_argument("--save", help="save graphs (png format)", action="store_true")
    parser.add_argument(
        "--show", help="display graphs (X11 required)", action="store_true"
    )
    parser.add_argument(
        "--synchronize",
        help="synchronize clock pupitre/pigbrother files",
        action="store_true",
    )
    parser.add_argument(
        "--flow",
        help="compute flow params from pupitre",
        action="store_true",
    )
    parser.add_argument(
        "--lag",
        help="compute lag between pupitre and pigbrother data",
        action="store_true",
    )
    parser.add_argument(
        "--distance", help="compute distance between series", action="store_true"
    )
    parser.add_argument("--bins", help="set bins for histograms", type=int, default=10)
    parser.add_argument(
        "--window", help="set rolling window size", type=int, default=50
    )
    parser.add_argument("--levels", help="set levels", type=int, default=4)
    parser.add_argument(
        "--plot-percent",
        help="percentage of points to plot (0-100, default 10)",
        type=float,
        default=10.0,
    )
    return parser.parse_args()


def setup():
    color_dict = {"U": "red", "P": "green", "D": "blue"}
    channels_dict = {"Référence_GR1": "Courant_GR1", "Référence_GR2": "Courant_GR2"}
    uchannels_dict = {
        "Référence_GR1": [
            "ALL_internes",
            "Interne1",
            "Interne2",
            "Interne3",
            "Interne4",
            "Interne5",
            "Interne6",
            "Interne7",
        ],
        "Référence_GR2": ["ALL_externes", "Externe1", "Externe2"],
    }
    pupitre_dict = {
        "M9": {
            "Référence_GR1": "IH",
            "Référence_GR2": "IB",
            "Référence_GR1_Q": "FlowH",
            "Référence_GR2_Q": "FlowB",
            "Référence_GR1_Rpm": "RpmH",
            "Référence_GR2_Rpm": "RpmB",
            "Référence_GR1_Pin": "HPH",
            "Référence_GR2_Pin": "HPB",
        },
        "M8": {
            "Référence_GR1": "IB",
            "Référence_GR2": "IH",
            "Référence_GR1_Q": "FlowB",
            "Référence_GR2_Q": "FlowH",
            "Référence_GR1_Rpm": "RpmB",
            "Référence_GR2_Rpm": "RpmH",
            "Référence_GR1_Pin": "HPB",
            "Référence_GR2_Pin": "HPH",
        },
        "M10": {
            "Référence_GR1": "IB",
            "Référence_GR2": "IH",
            "Référence_GR1_Q": "FlowB",
            "Référence_GR2_Q": "FlowH",
            "Référence_GR1_Rpm": "RpmB",
            "Référence_GR2_Rpm": "RpmH",
            "Référence_GR1_Pin": "HPB",
            "Référence_GR2_Pin": "HPH",
        },
    }
    upupitre_dict = {
        "M9": {"Référence_GR1": ["UH"], "Référence_GR2": ["UB", "Ucoil15", "Ucoil16"]},
        "M8": {"Référence_GR2": ["UH"], "Référence_GR1": ["UB", "Ucoil15", "Ucoil16"]},
        "M10": {"Référence_GR2": ["UH"], "Référence_GR1": ["UB", "Ucoil15", "Ucoil16"]},
    }
    threshold_dict = {
        "Référence_GR1": 0.5,
        "Courant_GR1": 0.5,
        "ALL_internes": 0.1,
        "Interne1": 1.0e-2,
        "Interne2": 1.0e-2,
        "Interne3": 1.0e-2,
        "Interne4": 1.0e-2,
        "Interne5": 1.0e-2,
        "Interne6": 1.0e-2,
        "Interne7": 1.0e-2,
        "Référence_GR2": 0.5,
        "Courant_GR2": 0.5,
        "ALL_externes": 0.1,
        "Externe1": 0.1,
        "Externe2": 0.1,
        "IH": 1,
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
        "IB": 1,
        "UB": 0.1,
        "Ucoil15": 0.1,
        "Ucoil16": 0.1,
        "debitbrut": 25,
        "Pmagnet": 0.1,
    }
    return (
        color_dict,
        channels_dict,
        uchannels_dict,
        pupitre_dict,
        upupitre_dict,
        threshold_dict,
    )


def synchronize_data(df: pd.DataFrame, t0: datetime) -> tuple:
    t0_ = df["timestamp"].iloc[0]
    timeshift = t0 - t0_
    df["timestamp"] = df["timestamp"] + pd.to_timedelta(timeshift)

    pt0 = df.iloc[0]["timestamp"]
    df.drop(["t"], axis=1, inplace=True)
    df["t"] = df.apply(lambda row: (row.timestamp - pt0).total_seconds(), axis=1)
    return timeshift, df


def downsample_for_plot(x, y, percent=10.0):
    """Downsample data for plotting to reduce memory usage.

    Args:
        x, y: arrays to downsample
        percent: percentage of points to keep (0-100)
    """
    if percent >= 100:
        return x, y
    n = len(x)
    n_keep = max(1, int(n * percent / 100.0))
    step = max(1, n // n_keep)
    return x[::step], y[::step]


def plot_data(
    df_overview: pd.DataFrame,
    df_archive: pd.DataFrame,
    df_pupitre: pd.DataFrame,
    df_incidents: dict | None,
    channels_dict: dict,
    pupitre_dict: dict,
    tlogs: dict,
    site: str,
    tkey: str,
    key: str,
    title: str,
    msg: str,
    args,
):
    # my_ax = plt.gca()
    fig, my_ax = plt.subplots(figsize=(12, 5))
    df_overview.plot(x=tkey, y=key, color="b", ax=my_ax)
    legends = [f"Overview: {key}"]
    df_overview.plot(x=tkey, y=channels_dict[key], marker=".", color="r", ax=my_ax)
    legends.append(f"Overview: {channels_dict[key]}")
    df_archive.plot(x=tkey, y=channels_dict[key], alpha=0.5, color="r", ax=my_ax)
    legends.append(f"Archive: {channels_dict[key]}")
    if not df_pupitre.empty:
        df_pupitre.plot(
            x=tkey,
            y=pupitre_dict[site][key],
            marker=".",
            color="g",
            ax=my_ax,
        )
        legends.append(f"Pupitre: {pupitre_dict[site][key]}")
    plt.legend(labels=legends)

    annotation_dict = {}
    if df_incidents is not None:
        for itype, incident in df_incidents.items():
            # print(f"plot_data: itype={itype}", flush=True)
            for i, idf in enumerate(incident):
                t_mid = idf[tkey].median()
                f_mid = idf[channels_dict[key]].median()
                (point,) = my_ax.plot(t_mid, f_mid, "yo", markersize=8)

                # Add annotation with arrow
                annot = my_ax.annotate(
                    rf"{itype} \#{i+1}",
                    xy=(t_mid, f_mid),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                )

                # Make annotation clickable
                annot.set_picker(True)

                # Store metadata
                annotation_dict[annot] = {
                    "anomaly": rf"{itype} \#{i+1}",
                    "idx": i,
                    "df": idf,
                    "pupitre": (df_pupitre, pupitre_dict[site][key]),
                    "archive": (df_archive, channels_dict[key]),
                }

    for i, (lkey, value) in enumerate(tlogs.items()):
        print(f"logs[{lkey}]={value}", flush=True)
        if value[tkey]:
            y_interp = np.interp(
                value[tkey], df_overview[tkey].values, df_overview[key].values
            )
            print(f"y_interp={y_interp}", flush=True)
            my_ax.plot(value[tkey], y_interp, "bo", markersize=8)

            # Add annotation with arrow
            annot = my_ax.annotate(
                rf"DAQ \#{i+1}",
                xy=(value[tkey][0], y_interp[0]),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="orange", alpha=0.7),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )

    # Track open subplot figures
    open_figures = {}

    def on_pick(event):
        if event.artist in annotation_dict:
            annot = event.artist
            metadata = annotation_dict[annot]
            anomaly = metadata["anomaly"]
            idx = metadata["idx"]
            idf = metadata["df"]
            pupitre, pupitre_key = metadata["pupitre"]
            archive, archive_key = metadata["archive"]

            # Close previous subplot if it exists
            if idx in open_figures:
                plt.close(open_figures[idx])

            # Create subplot
            fig_sub, ax_sub = plt.subplots(figsize=(8, 5))
            # ax_sub.plot(idf["t"], idf[channels_dict[key]], "y-", linewidth=2)
            idf.plot(x="t", y=archive_key, color="y", ax=ax_sub)
            pupitre.plot(x="t", y=pupitre_key, color="g", ax=ax_sub)
            archive.plot(x="t", y=archive_key, color="r", alpha=0.5, ax=ax_sub)
            ax_sub.set_xlabel("t")
            ax_sub.set_xlim(idf.iloc[0]["t"], idf.iloc[-1]["t"])
            ax_sub.set_title(anomaly)
            ax_sub.grid(True, alpha=0.3)

            fig_sub.tight_layout()
            fig_sub.show()

            # Store figure reference
            open_figures[idx] = fig_sub

    # Connect pick event
    fig.canvas.mpl_connect("pick_event", on_pick)
    plt.title(f'{title.replace("_Overview","")}: {key} {msg}')
    plt.grid()
    plt.tight_layout()
    if args.show:
        plt.show()
    if args.save:
        (label, igroup) = key.split("_")
        plt.savefig(f'{title.replace("_Overview","")}-{igroup}.png', dpi=300)
    plt.close()


def compute_flowparams(
    df_pupitre, filename, Ikeys_ref, pupitre_dict, site, args, overview_dict
):
    """
    Compute flow parameters including teb, BP, debitbrut, and flow_params for each current key.

    Args:
        df_pupitre: DataFrame containing pupitre data
        filename: Name of the file being processed
        Ikeys_ref: List of reference current keys
        pupitre_dict: Dictionary mapping site to pupitre field names
        site: Site identifier
        args: Command line arguments containing levels, show flags
        overview_dict: Dictionary to store the computed results

    Returns:
        None (updates overview_dict in place)
    """
    # TODO: save a better approx for teb
    overview_dict[filename]["teb"] = df_pupitre["teb"].mean()
    overview_dict[filename]["BP"] = df_pupitre["BP"].mean()

    from .flow_params import debitbrut

    for nlevel in range(1, args.levels + 1):
        (thresholds, high_values, low_values) = debitbrut(
            df_pupitre, filename, nlevels=nlevel
        )

    overview_dict[filename]["debitbrut"] = {
        "thresholds": thresholds,
        "high": high_values,
        "low": low_values,
    }

    # just a few check on extra_key
    if args.debug:
        for extra_key in ["teb", "debitbrut", "Pmagnet", "BP"]:
            print(f"{extra_key}: {df_pupitre[extra_key].describe()}")
            df_pupitre.plot(x="t", y=extra_key)
            plt.grid()
            if args.show:
                plt.show()
            plt.close()

    # IH/FlowH or IB/FlowB
    print(f"list key df_pupitre: {list(df_pupitre.keys())}")
    for key in Ikeys_ref:
        print(
            f'flow-params for {key}: Ikey={pupitre_dict[site][key]}, RpmKey={pupitre_dict[site][f"{key}_Rpm"]}'
        )
        from .flow_params import compute as flow_params

        flow_params(
            df_pupitre,
            Ikey=pupitre_dict[site][key],
            RpmKey=pupitre_dict[site][f"{key}_Rpm"],
            QKey=pupitre_dict[site][f"{key}_Q"],
            PinKey=pupitre_dict[site][f"{key}_Pin"],
            PoutKey="BP",
            name=f"{filename}",
            show=args.show,
            debug=False,
        )


def compute_lag_correlation(
    key: str,
    filename: str,
    t0: datetime,
    df_overview: pd.DataFrame,
    df_pupitre: pd.DataFrame,
    ot0: datetime,
    channels_dict: dict,
    pupitre_dict: dict,
    site: str,
    threshold_dict: dict,
    symbol: Symbol,
    unit: str,
    args,
) -> tuple:
    """
    Compute lag correlation between pupitre and pigbrother overview data.

    This function creates signatures for both overview and pupitre data,
    finds matching regimes, computes the lag between them, and updates
    the pupitre dataframe timestamps accordingly.

    Args:
        key: Current key being processed (e.g., 'IH_1', 'IB_2')
        filename: Name of the file being processed
        t0: Reference timestamp for signatures
        df_overview: DataFrame containing overview data
        df_pupitre: DataFrame containing pupitre data (will be modified)
        ot0: Overview reference timestamp
        channels_dict: Dictionary mapping keys to channel names
        pupitre_dict: Dictionary mapping site to pupitre field names
        site: Site identifier
        threshold_dict: Dictionary of thresholds for signature detection
        symbol: Symbol for the field being processed
        unit: Unit of measurement
        args: Command line arguments

    Returns:
        Tuple of (lag, updated_df_pupitre)
    """
    print("\nLag correlation: pupitre/pigbrother overview")

    # Create signatures for overview and pupitre data
    osignature = Signature.from_df(
        filename,
        t0,
        df_overview,
        channels_dict[key],
        symbol,
        unit,
        tkey="t",
        threshold=threshold_dict[channels_dict[key]],
        timeshift=0,
        show=False,
        debug=args.debug,
    )
    psignature = Signature.from_df(
        filename,
        t0,
        df_pupitre,
        pupitre_dict[site][key],
        symbol,
        unit,
        tkey="t",
        threshold=threshold_dict[pupitre_dict[site][key]],
        timeshift=0,
        show=False,
        debug=args.debug,
    )

    if args.debug:
        print(
            f"regimes (pupitre[{pupitre_dict[site][key]}]):",
            psignature.regimes,
        )
        print(f"times (overview[{channels_dict[key]}]):", osignature.times)

    # Helper function to compute regime score
    def compute_regime_score(
        regime: str,
        value: tuple,
        time: tuple,
        reference_regime: str,
        reference_value: tuple,
        reference_time: tuple,
    ):
        score = float("inf")
        tscore = float("inf")
        lags = (float("inf"), float("inf"))
        if reference_regime == regime:
            start_diff = abs(value[0] - reference_value[0])
            end_diff = abs(value[1] - reference_value[1])
            score = start_diff + end_diff
            score = abs(
                abs(value[1] - value[0]) - abs(reference_value[1] - reference_value[0])
            )

            tscore = abs(
                abs(time[1] - time[0]) - abs(reference_time[1] - reference_time[0])
            )
            start_lag = time[0] - reference_time[0]
            end_lag = time[1] - reference_time[1]
            lags = (start_lag, end_lag)

        return (score, tscore, lags)

    # Helper function to find best matching regime
    def find_best_matching_regime(signature, reference_signature):
        best_matches = []
        for i, regime in enumerate(signature.regimes):
            best_score = float("inf")
            best_lags = (float("inf"), float("inf"))
            best_index = (0, 0)
            best_match = None

            if regime in ["U", "D"] and i <= len(signature.times) - 2:
                values = (signature.values[i], signature.values[i + 1])
                times = (signature.times[i], signature.times[i + 1])
                for j, ref_regime in enumerate(reference_signature.regimes):
                    # get value and time range
                    if (
                        ref_regime in ["U", "D"]
                        and j <= len(reference_signature.times) - 2
                    ):
                        ref_values = (
                            reference_signature.values[j],
                            reference_signature.values[j + 1],
                        )
                        ref_times = (
                            reference_signature.times[j],
                            reference_signature.times[j + 1],
                        )

                        score, tscore, lags = compute_regime_score(
                            regime,
                            values,
                            times,
                            ref_regime,
                            ref_values,
                            ref_times,
                        )

                        if score < best_score:
                            best_score = score
                            best_match = ref_regime
                            best_lags = lags
                            best_index = (i, j)
                            best_matches.append(
                                (
                                    regime,
                                    best_match,
                                    best_score,
                                    best_lags,
                                    best_index,
                                )
                            )
        return best_matches

    # Find initial best matches
    best_matches = find_best_matching_regime(osignature, psignature)

    duration = df_pupitre.index.values[-1] - df_pupitre.index.values[0]
    for regime, best_match, score, lags, best_index in best_matches:
        rmsg = ""
        if abs(lags[0]) / duration >= 0.2:
            rmsg = "!! unlikely !!"
        print(
            f"Regime {regime} matched with {best_match}, score={score}, lag={lags[0]:.2f}s {rmsg}"
        )

    # Compute and apply lag
    print(
        f"1st lag for {key}: between {channels_dict[key]} - {pupitre_dict[site][key]}"
    )
    df1_data = {
        "df": df_overview.loc[:, ["timestamp", channels_dict[key]]],
        "field": channels_dict[key],
        "range": (osignature.changes[0], osignature.changes[2] + 2),
    }
    df2_data = {
        "df": df_pupitre.loc[:, ["timestamp", pupitre_dict[site][key]]],
        "field": pupitre_dict[site][key],
        "range": (psignature.changes[0], psignature.changes[2] + 2),
    }

    lag = compute_lag(
        "timestamp",
        df1_data,
        df2_data,
        show=args.show,
        save=args.save,
        debug=args.debug,
    )
    print(f"1st lag: {lag.total_seconds()} s")
    df_pupitre["timestamp"] = df_pupitre["timestamp"] - pd.to_timedelta(f"{lag}s")

    # Update timestamp and t
    pt0 = df_pupitre["timestamp"].iloc[0]
    print("new t0 (pupitre):", pt0)
    df_pupitre.drop(["t"], axis=1, inplace=True)
    df_pupitre["t"] = df_pupitre.apply(
        lambda row: (row.timestamp - ot0).total_seconds(), axis=1
    )
    if args.debug:
        print(f"df_pupitre with lag:\n{df_pupitre.head()}")

    # Update times for pupitre signature
    for j in range(len(psignature.times)):
        psignature.times[j] = psignature.times[j] - lag.total_seconds()

    # Find lags at U and D after adjustment
    best_matches = find_best_matching_regime(osignature, psignature)
    for regime, best_match, score, lags, best_index in best_matches:
        rmsg = ""
        if abs(lags[0]) / duration >= 0.2:
            rmsg = "!! unlikely !!"
        # Missing: print statement or logging to report post-adjustment regime matching
        print(
            f"After adjustment - Regime {regime} matched with {best_match}, score={score}, lag={lags[0]:.2f}s {rmsg}"
        )

    return lag, df_pupitre


def process_acqnet_logs(
    log_files: list,
    file: str,
    archive_files: list,
    default_files: list,
    spike_files: list,
    start_ftimestamp: str,
    t0: datetime,
    args,
) -> dict:
    """
    Process ACQ_ENET log files and extract error timestamps.

    Args:
        log_files: List of log file paths to process
        file: Main overview file path
        archive_files: List of archive file paths
        default_files: List of default incident file paths
        spike_files: List of spike incident file paths
        start_ftimestamp: Start timestamp string in format "%Y-%m-%d %H:%M:%S"
        t0: Reference datetime for time calculations
        args: Command line arguments containing debug flag

    Returns:
        Dictionary mapping file basenames to their log data (timestamps and relative times)
    """
    print("\nlogs:")
    print("basename:", [os.path.basename(f) for f in [file] + archive_files])
    t_start = datetime.strptime(start_ftimestamp, "%Y-%m-%d %H:%M:%S")
    print(f"t_start: {t_start} ({type(t_start)})", flush=True)

    tlogs = {}
    for lfile in log_files:
        print("\t", lfile, flush=True)
        logs = json.load(open(lfile))
        for lkey in logs.keys():
            if args.debug:
                print(lkey, logs[lkey])
            if lkey in [os.path.basename(f) for f in [file] + archive_files]:
                tlogs[lkey] = {
                    "timestamp": [],
                    "t": [],
                }
                # if args.debug:
                print(
                    f"processing log file: {lkey}: {logs[lkey].keys()}",
                    flush=True,
                )
                errors = logs[lkey]["errors"]
                for error in errors:
                    t_log_str = error["error_timestamp"]
                    t_log = datetime.strptime(
                        t_log_str, "%Y-%m-%dT%H:%M:%S"
                    )  # "2023-07-18T16:40:36"
                    print(
                        f"log file: {lkey},t_log={t_log} ({type(t_log)}), t_start={t_start} ({type(t0)})",
                        flush=True,
                    )
                    print(
                        lkey,
                        t_log,
                        t0,
                        (t_log - t0).total_seconds(),
                        flush=True,
                    )
                    tlogs[lkey]["timestamp"].append(t_log)
                    tlogs[lkey]["t"].append((t_log - t_start).total_seconds())
                print(f"\t{tlogs[lkey]}", flush=True)

    for lfile in log_files:
        print("\t", lfile, flush=True)
        logs = json.load(open(lfile))
        for lkey in logs.keys():
            print(lkey, logs[lkey])
            if lkey in [os.path.basename(f) for f in default_files + spike_files]:
                tlogs[lkey] = {
                    "t0": [],
                    "t": [],
                }
                print(
                    f"processing log file: {lkey}: {logs[lkey].keys()}",
                    flush=True,
                )

    return tlogs


def compute_distance(
    df_pupitre,
    df_overview,
    key,
    site,
    pupitre_dict,
    channels_dict,
    ot0,
    filename,
    file,
    args,
):
    """
    Compute distance metrics between pupitre and pigbrother data using various methods.

    Args:
        df_pupitre: DataFrame containing pupitre data
        df_overview: DataFrame containing overview/pigbrother data
        key: Reference key for the current data
        site: Site identifier (M9, M10, etc.)
        pupitre_dict: Dictionary mapping keys to pupitre channel names
        channels_dict: Dictionary mapping keys to overview channel names
        ot0: Origin timestamp from overview data
        filename: Base filename for saving plots
        file: Full file path for constructing output filenames
        args: Command line arguments (debug, save, show flags)
    """
    # compute distance between pupitre and pigbrother
    # print("df_pupitre:", df_pupitre)
    xdata = df_pupitre.copy()
    xdata.set_index("timestamp", inplace=True)
    if args.debug:
        print("xdata:\n", xdata.head())
    xdata_index = xdata.index.to_list()
    xdata_resampled = xdata.resample("1s", origin=ot0).asfreq()
    # Interpolate missing values (optional, depending on your use case)
    xdata_resampled = xdata_resampled.interpolate(method="linear")
    # xdata_resampled = xdata_resampled + xdata_resampled.min()

    if args.debug:
        print("xdata_resampled:\n", xdata_resampled.head())
    xdata_resampled.set_index("t", inplace=True)
    # print("after resample xdata_resampled:", xdata_resampled)
    end_index = xdata_resampled.index.values[-1].astype(int)
    print(
        f"\nDistance between pupitre {pupitre_dict[site][key]} and pigbrother {channels_dict[key]} from t=0 to t={end_index} s"
    )

    x = xdata_resampled[pupitre_dict[site][key]].loc[0:end_index].to_numpy().reshape(-1)
    y = df_overview[channels_dict[key]].loc[0:end_index].to_numpy().reshape(-1)

    if args.debug:
        plt.plot(x, label="pupitre", marker=".", color="g")
        plt.plot(y, label="overview", marker="o", color="r", alpha=0.2)
        plt.title(f"distance: {key}")
        plt.xlabel(pupitre_dict[site][key])
        plt.ylabel(channels_dict[key])
        plt.legend()
        plt.grid()
        plt.show()
        plt.close()

    # print('Ib:', x, type(x), x.shape)
    scipy_stats = stats.describe(y - x)

    (label, igroup) = key.split("_")
    tables = []
    headers = [
        "Euclidean",
        "MAE",
        "Pearson",
        "Image",
        "mean",
        "min",
        "max",
        "var",
    ]
    table = [
        calc_euclidean(x, y),
        calc_mape(x, y),
        calc_correlation(x, y),
        f'{filename.replace("_Overview","")}-{igroup}.png',
        scipy_stats.mean,
        scipy_stats.minmax[0],
        scipy_stats.minmax[1],
        scipy_stats.variance,
    ]
    tables.append(table)
    print(tabulate(tables, headers, tablefmt="simple"), "\n")

    # Calculate DTW distance and obtain the warping paths (no need for the C library)
    # see https://medium.com/@markstent/dynamic-time-warping-a8c5027defb6
    from dtaidistance import dtw
    from dtaidistance import dtw_visualisation as dtwvis

    # from scipy.stats import pearsonr

    ts_x = df_overview.loc[:, ["t", channels_dict[key]]]
    ts_x.set_index("t", inplace=True)
    # TODO no longer working since t_offset on overview data
    # ts_x.index = ts_x.index.astype(int)
    if args.debug:
        print("dtw ts_x:", ts_x.head())
    ts_y = xdata_resampled.loc[:, [pupitre_dict[site][key]]]
    # drop negative index in ts_y
    ts_y = ts_y[ts_y.index >= 0]
    ts_y.index = ts_y.index.astype(int)
    if args.debug:
        print("dtw ts_y:", ts_y.head)

    ts_x = ts_x.to_numpy().reshape(-1)
    ts_y = ts_y.to_numpy().reshape(-1)

    distance, paths = dtw.warping_paths(ts_x, ts_y, use_c=False)
    best_path = dtw.best_path(paths)
    similarity_score = distance / len(best_path)
    dtwvis.plot_warpingpaths(ts_x, ts_y, paths, best_path)

    plt.xlabel(channels_dict[key])
    plt.ylabel(pupitre_dict[site][key])
    plt.tight_layout()

    if args.save:
        plt.savefig(
            f"{os.path.basename(file)}-distance-{channels_dict[key]}_{pupitre_dict[site][key]}.png",
            dpi=300,
        )
    if args.show:
        plt.show()
    plt.close()

    # Create a DataFrame to display the similarity score and correlation coefficient
    results_df = pd.DataFrame(
        {
            "Metric": ["DTW Similarity Score"],
            "Value": [similarity_score],
        }
    )

    # Add descriptions for the results
    results_df["Description"] = [
        "Lower scores indicate greater similarity between the time series."
    ]
    print(results_df)

    plt.figure(figsize=(12, 8))

    # Original Time Series Plot
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.plot(ts_x, label=channels_dict[key], color="blue")
    ax1.plot(
        ts_y,
        label=pupitre_dict[site][key],
        linestyle="--",
        color="orange",
    )
    ax1.set_title("Original Time Series")
    ax1.legend()
    ax1.grid(True)

    # Shortest Path Plot (Cost Matrix with the path)
    # In this example, only the path is plotted, not the entire cost matrix.

    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax2.plot(
        np.array(best_path)[:, 0],
        np.array(best_path)[:, 1],
        "green",
        marker="o",
        linestyle="-",
    )
    ax2.set_title("Shortest Path (Best Path)")
    ax2.set_xlabel(channels_dict[key])
    ax2.set_ylabel(pupitre_dict[site][key])
    ax2.grid(True)

    # Point-to-Point Comparison Plot
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2, sharex=ax1)
    ax3.plot(ts_x, label=channels_dict[key], color="blue", marker="o")
    ax3.plot(
        ts_y,
        label=pupitre_dict[site][key],
        color="orange",
        marker="x",
        linestyle="--",
    )
    for a, b in best_path:
        ax3.plot(
            [a, b],
            [ts_x[a], ts_y[b]],
            color="grey",
            linestyle="-",
            linewidth=1,
            alpha=0.5,
        )
    ax3.set_title("Point-to-Point Comparison After DTW Alignment")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    if args.save:
        plt.savefig(
            f"{os.path.basename(file)}-dtw-{key}.png",
            dpi=300,
        )
    if args.show:
        plt.show()
    plt.close()


def expand_input_files(input_patterns: list, datadir: dict) -> list:
    """
    Expand glob patterns in input file arguments.

    Args:
        input_patterns: List of file patterns to expand
        datadir: Dictionary mapping file extensions to their base directories

    Returns:
        List of expanded file paths
    """

    print(f"Expanding input files ({input_patterns})...", flush=True)
    expanded_files = []
    for pattern in input_patterns:
        extension = os.path.splitext(pattern)[-1]
        print(f"pattern: {pattern}, extension: {extension}", flush=True)
        # Check if pattern contains a directory component
        if os.path.dirname(pattern):
            # Pattern has a directory, use it as is
            search_pattern = pattern
        else:
            # No directory in pattern, prepend appropriate datadir based on extension
            base_datadir = datadir[extension]
            if extension == ".tdms":
                (site, mode, timestamp) = pattern.split("_")
                print(
                    f"pattern: site={site}, mode={mode}, timestamp={timestamp}",
                    flush=True,
                )
                search_pattern = os.path.join(base_datadir, site, mode, pattern)
            else:
                search_pattern = os.path.join(base_datadir, pattern)
        print(search_pattern, flush=True)

        matches = glob.glob(search_pattern)
        if matches:
            print(f"matches: {matches}", flush=True)
            expanded_files.extend(matches)
        else:
            # If no matches, keep the original pattern (might be a literal filename)
            print(f"No matches found for pattern: {pattern}", flush=True)
            expanded_files.append(pattern)

    print(f"expanded_files: {expanded_files}", flush=True)
    return expanded_files


def main():
    args = parse_arguments()
    print(args.input_file)

    (
        color_dict,
        channels_dict,
        uchannels_dict,
        pupitre_dict,
        upupitre_dict,
        threshold_dict,
    ) = setup()

    # Create datadir dictionary mapping extensions to data directories
    datadir = {
        ".tdms": args.pigbrother_datadir,
        ".txt": args.pupitre_datadir,
    }

    # Expand glob patterns in input_file arguments
    expanded_files = expand_input_files(args.input_file, datadir)

    input_files = natsorted(expanded_files)
    print(f"input_files: {input_files}", flush=True)

    if args.show and args.save:
        print("error: both --show and --save options activated", flush=True)
        exit(1)

    log_files = natsorted(args.logs) if args.logs else []
    print(f"log_files: {log_files}", flush=True)

    insert = "tututu"

    symbol = str()
    unit = str()
    group = "Courants_Alimentations"
    Ikeys_ref = []
    overview_dict = {}
    for file in input_files:
        print(f"*** processing file: {file} ***", flush=True)
        extension = os.path.splitext(file)[-1]
        dirname = os.path.dirname(file)
        filename = os.path.basename(file).replace(extension, "")
        (site, mode, timestamp) = filename.split("_")
        date, time = timestamp.split("-")
        # Set default dirname if empty
        if not dirname:
            dirname = f"{args.pigbrother_datadir}/{site}/Overview"
            file = os.path.join(dirname, filename + extension)
        # print(f"site={site}, date={date}, time={time}, insert={insert}", flush=True)
        (start_ftimestamp, end_ftimestamp, skip) = extract_data(
            file, site, insert=insert, key=None
        )

        # select files
        pupitre_filter, archive_filter, default_filter, trigger_filter, spike_filter = (
            find_files(args, file, site, date, time)
        )
        print("selecting pupitre files")
        pupitre_files = select_files(
            glob.glob(pupitre_filter), site, start_ftimestamp, end_ftimestamp
        )
        print("selecting archive files done")
        archive_files = select_files(
            glob.glob(archive_filter), site, start_ftimestamp, end_ftimestamp
        )
        default_files = select_files(
            glob.glob(default_filter), site, start_ftimestamp, end_ftimestamp
        )
        trigger_files = select_files(
            glob.glob(trigger_filter), site, start_ftimestamp, end_ftimestamp
        )
        spike_files = select_files(
            glob.glob(spike_filter), site, start_ftimestamp, end_ftimestamp
        )
        print("\nfilters:")
        print(
            f"pfilter: {pupitre_filter},\nafilter: {archive_filter},\ndfilter: {default_filter},\ntfilter: {trigger_filter},\ntfilter: {spike_filter}",
            flush=True,
        )

        print("\nfiles:")
        print(f"pupitre_files={pupitre_files}", flush=True)
        print(f"archive_files={archive_files}", flush=True)
        print(f"default_files={default_files}", flush=True)
        print(f"trigger_files={trigger_files}", flush=True)
        print(f"spike_files={spike_files}", flush=True)
        print("\n")

        if not args.dry_run:
            mrun = MagnetRun.fromtdms(site, insert, file)
            mdata = mrun.getMData()
            Ikeys_ref = [
                key.replace(f"{group}/", "")
                for key in mdata.getKeys()
                if "Référence_GR" in key
            ]
            start, end, skip = extract_data(
                file, site, insert, f"{group}/{Ikeys_ref[0]}"
            )
            # print(f"{filename}: file={file}, start={start}, end={end}", flush=True)

            dict_files = {
                "overview": [file],
                "pupitre": [],
                "archive": [],
                "default": [],
                "trigger": [],
                "spike": [],
            }

            for pfile in pupitre_files:
                pstart, pend, pskip = extract_data(
                    pfile, site, insert, pupitre_dict[site][Ikeys_ref[0]]
                )
                if pstart >= start and pend <= end:
                    dict_files["pupitre"].append(pfile)
            for pfile in archive_files:
                pstart, pend, pskip = extract_data(
                    pfile, site, insert, f"{group}/{channels_dict[Ikeys_ref[0]]}"
                )
                if not pskip:
                    if pstart >= start and pend <= end:
                        dict_files["archive"].append(pfile)
            for pfile in default_files:
                pstart, pend, pskip = extract_data(
                    pfile, site, insert, f"{group}/{channels_dict[Ikeys_ref[0]]}"
                )
                if not pskip:
                    if pstart >= start and pend <= end:
                        dict_files["default"].append(pfile)

            for pfile in trigger_files:
                pstart, pend, pskip = extract_data(
                    pfile, site, insert, f"{group}/{channels_dict[Ikeys_ref[0]]}"
                )
                if not pskip:
                    if pstart >= start and pend <= end:
                        dict_files["trigger"].append(pfile)

            for pfile in spike_files:
                pstart, pend, pskip = extract_data(
                    pfile, site, insert, f"{group}/{channels_dict[Ikeys_ref[0]]}"
                )
                if not pskip:
                    if pstart >= start and pend <= end:
                        dict_files["spike"].append(pfile)

            print(f"{file}: dict_files: {dict_files}", flush=True)
            t0 = mdata.Groups[group][Ikeys_ref[0]]["wf_start_time"]
            print(
                f"{file}: t0 from {group}/{Ikeys_ref[0]}: {t0} ({type(t0)})", flush=True
            )

            symbol, unit = mdata.getUnitKey(f"{group}/{channels_dict[Ikeys_ref[0]]}")

            # Process ACQ_ENET log files
            tlogs = process_acqnet_logs(
                log_files,
                file,
                archive_files,
                default_files,
                spike_files,
                start_ftimestamp,
                t0,
                args,
            )

            # get mode
            print("find mode:")
            bitter_only = True
            mode = compute_mode(mdata, group, Ikeys_ref)

            if bitter_only:
                uprobes = ["ALL_externes", "Externe1", "Externe2"]
                print("\n!!! Selecting only U probes for Bitter !!!\n")

            print(f"dict_files[{filename}]: ", dict_files, flush=True)
            overview_dict[filename] = {
                "mode": mode,
                "signature": {},
                "sources": dict_files,
                "data": {
                    "overview": pd.DataFrame(),
                    "pupitre": pd.DataFrame(),
                    "archive": pd.DataFrame(),
                    "default": [],
                    "trigger": [],
                    "spike": [],
                },
                "t0": t0,
                "BP": {},
                "teb": {},
                "debitbrut": {},
                "flow_params": {},
            }

            for key in Ikeys_ref:
                overview_dict[filename]["signature"][key] = Signature.from_mdata(
                    mdata,
                    f"{group}/{key}",
                    "t",
                    threshold_dict[key],
                )

            # Load Overview data
            Ikeys = Ikeys_ref + [channels_dict[key] for key in Ikeys_ref]
            print("\nLoad Overview data")
            # print(i, filename)
            df_overview_list = load_data(
                overview_dict[filename]["sources"]["overview"],
                site,
                insert,
                group,
                Ikeys,
            )
            # print(f"df_overview_list: {len(df_overview_list)} files")
            overview_dict[filename]["data"]["overview"] = df_overview_list[0]
            # print(
            #     f"{filename}: overview \n {overview_dict[filename]['data']['overview']}"
            # )

            df_overview = overview_dict[filename]["data"][
                "overview"
            ]  # df_overview_list[i]
            ot0 = df_overview["timestamp"].iloc[0]

            print("\nLoad Archive data")
            if len(overview_dict[filename]["sources"]["archive"]) == 0:
                raise RuntimeError(f"no archive file associated with {filename}")

            df_archive_list = load_data(
                overview_dict[filename]["sources"]["archive"],
                site,
                insert,
                group,
                Ikeys,
            )

            df_archive = merge_data(df_archive_list)

            # true if at0 is roughly equal to ot0
            at0 = df_archive.iloc[0]["timestamp"]
            t_offset = (1 / 120.0) / 2.0
            df_archive["t"] = df_archive.apply(
                lambda row: (row.timestamp - at0).total_seconds() + t_offset, axis=1
            )
            overview_dict[filename]["data"]["archive"] = df_archive

            # Load pupitre data
            print("\nLoad Pupitre data")
            Ikeys_p = [pupitre_dict[site][key] for key in Ikeys_ref]
            Flowkeys = []
            for key in Ikeys_ref:
                Flowkeys += [
                    pupitre_dict[site][f"{key}_Q"],
                    pupitre_dict[site][f"{key}_Rpm"],
                    pupitre_dict[site][f"{key}_Pin"],
                ]

            df_pupitre = pd.DataFrame()
            if overview_dict[filename]["sources"]["pupitre"]:
                df_pupitre_list = load_data(
                    overview_dict[filename]["sources"]["pupitre"],
                    site,
                    insert,
                    group,
                    Ikeys_p
                    + Flowkeys
                    + [
                        "BP",
                        "teb",
                        "debitbrut",
                        "Pmagnet",
                    ],
                )
                df_pupitre = merge_data(df_pupitre_list)
                print(f"{filename}: pupitre \n {df_pupitre.keys()}")
                pt0 = df_pupitre.iloc[0]["timestamp"]
                df_pupitre["t"] = df_pupitre.apply(
                    lambda row: (row.timestamp - pt0).total_seconds(), axis=1
                )

                overview_dict[filename]["data"]["pupitre"] = df_pupitre
                # synchronize data ad get timeshit
                timeshift, df_pupitre = synchronize_data(
                    overview_dict[filename]["data"]["pupitre"], ot0
                )

            if args.flow:
                compute_flowparams(
                    df_pupitre,
                    filename,
                    Ikeys_ref,
                    pupitre_dict,
                    site,
                    args,
                    overview_dict,
                )

            # Load incidents data
            print("\nLoad Incidents data")
            at0 = overview_dict[filename]["data"]["archive"].iloc[0]["timestamp"]
            for dtype in ["default", "trigger", "spike"]:
                print(f'{filename}: {overview_dict[filename]["sources"][dtype]}')
                overview_dict[filename]["data"][dtype] = load_data(
                    overview_dict[filename]["sources"][dtype],
                    site,
                    insert,
                    group,
                    Ikeys,
                )

                for i, df in enumerate(overview_dict[filename]["data"][dtype]):
                    it0 = df["timestamp"].iloc[0]
                    t_offset = (1 / 4800.0) / 2.0
                    print(
                        f'{type} file {overview_dict[filename]["sources"][dtype][i]}: it0={it0}, at0={at0}, t0={(it0 - at0).total_seconds() + t_offset}, len={len(df)}'
                    )
                    if args.debug:
                        print(df["timestamp"])
                    df["t"] = df.apply(
                        lambda row: (row.timestamp - at0).total_seconds() + t_offset,
                        axis=1,
                    )

            # save signature per overview file
            print(f"\nProcess Overview Files (signature, lag): {filename} ****")

            # for overview files
            t_offset = 1 / 2.0
            df_overview["t"] = df_overview.apply(
                lambda row: (row.timestamp - ot0).total_seconds() + t_offset, axis=1
            )
            df_archive = overview_dict[filename]["data"]["archive"]
            df_incidents = {
                "default": overview_dict[filename]["data"]["default"],
                "spike": overview_dict[filename]["data"]["spike"],
                "trigger": overview_dict[filename]["data"]["trigger"],
            }

            # get lag correlation
            # lag_correlation(
            print("t0 (overview):", df_overview["timestamp"].iloc[0])
            if not df_pupitre.empty:
                print("t0 (pupitre):", df_pupitre["timestamp"].iloc[0])

            lag_done = False
            msg = "(nosync)"
            for key in Ikeys_ref:
                print(f"**** {key} ****", flush=True)
                tables = []
                headers = [
                    "P",
                    "count",
                    "mean",
                    "std",
                    "min",
                    "25%",
                    "50%",
                    "75%",
                    "max",
                ]
                signature = overview_dict[filename]["signature"][key]
                t0 = overview_dict[filename]["t0"]
                for i, regime in enumerate(signature.regimes):
                    istart = signature.changes[i]
                    iend = df_overview.index.to_list()[-1]
                    if i < len(signature.changes) - 1:
                        iend = signature.changes[i + 1]

                    if regime == "P":
                        table = [i] + df_overview[key].iloc[
                            istart:iend
                        ].describe().to_list()
                        tables.append(table)
                print(tabulate(tables, headers=headers, tablefmt="psql"))
                if args.debug:
                    print(f"regimes (overview[{key}]):", signature.regimes)

                plot_data(
                    df_overview,
                    df_archive,
                    df_pupitre,
                    df_incidents,
                    channels_dict,
                    pupitre_dict,
                    tlogs,
                    site,
                    args.tkey,
                    key,
                    filename,
                    msg,
                    args,
                )

                # Skip lag and distance computation if pupitre data is empty
                if df_pupitre.empty:
                    print(
                        f"Warning: df_pupitre is empty for {key}, skipping lag and distance computation"
                    )
                    continue

                # get lag from 1st U sequence
                if args.lag:
                    if not lag_done:
                        lag, df_pupitre = compute_lag_correlation(
                            key,
                            filename,
                            t0,
                            df_overview,
                            df_pupitre,
                            ot0,
                            channels_dict,
                            pupitre_dict,
                            site,
                            threshold_dict,
                            symbol,
                            unit,
                            args,
                        )
                        lag_done = True

                    # plots
                    msg = f"(sync, 1st lag with pigbrother {lag.total_seconds()} s)"

                    # plot sync data vs t or timestamp
                    plot_data(
                        df_overview,
                        df_archive,
                        df_pupitre,
                        df_incidents,
                        channels_dict,
                        pupitre_dict,
                        tlogs,
                        site,
                        args.tkey,
                        key,
                        filename,
                        msg,
                        args,
                    )

                if args.distance:
                    compute_distance(
                        df_pupitre,
                        df_overview,
                        key,
                        site,
                        pupitre_dict,
                        channels_dict,
                        ot0,
                        filename,
                        file,
                        args,
                    )


if __name__ == "__main__":
    main()
