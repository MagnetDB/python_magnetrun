import os
import argparse
import gc
from functools import partial
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from python_magnetrun.MagnetRun import MagnetRun
from python_magnetrun.processing.hysteresis import remove_low_x_outliers
from python_magnetrun.processing.smoothers import savgol


# Named tuple for detection method configuration
DetectMethod = namedtuple("DetectMethod", ["func", "marker", "color"])


def zscore(df, key: str, debug: bool = False):
    from scipy import stats

    z = np.abs(stats.zscore(df[key]))
    anomalies = df[z > 3]
    if debug:
        print(f"zscore:\nanomalies={len(anomalies)}", flush=True)
    return anomalies


def irq(df, key, debug: bool = False):
    Q1 = df[key].quantile(0.25)
    Q3 = df[key].quantile(0.75)
    IQR = Q3 - Q1
    anomalies = df[(df[key] < Q1 - 1.5 * IQR) | (df[key] > Q3 + 1.5 * IQR)]
    if debug:
        print(f"irq:\nanomalies={len(anomalies)}", flush=True)
    return anomalies


def mad(df, key, debug: bool = False):
    median = df[key].median()
    mad_val = np.median(np.abs(df[key] - median))
    modified_z = 0.6745 * (df[key] - median) / mad_val
    anomalies = df[np.abs(modified_z) > 3.5]
    if debug:
        print(f"mad:\nanomalies={len(anomalies)}", flush=True)
    return anomalies


def movingaverage(df, key, window: int = 7, debug: bool = False):
    ma = df[key].rolling(window=window).mean()
    deviation = np.abs(df[key] - ma)
    threshold = 3 * deviation.std()
    anomalies = df[deviation > threshold]
    if debug:
        print(f"movingaverage:\nanomalies={len(anomalies)}", flush=True)
    return anomalies


def isolationforest(df, key, tkey="t", debug: bool = False):
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(contamination=0.01, random_state=42)
    df_clean = df[[key]].dropna()
    model.fit(df_clean)
    predictions = model.predict(df_clean)
    anomaly_mask = predictions == -1
    anomalies = df_clean[anomaly_mask].copy()
    anomalies[tkey] = df.loc[df_clean.index, tkey]
    del model, predictions, df_clean
    gc.collect()
    if debug:
        print(f"isolationforest:\nanomalies={len(anomalies)}", flush=True)
    return anomalies


def lof(df, key, tkey="t", debug: bool = False):
    from sklearn.neighbors import LocalOutlierFactor

    model = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    df_clean = df[[key]].dropna()
    predictions = model.fit_predict(df_clean)
    anomaly_mask = predictions == -1
    anomalies = df_clean[anomaly_mask].copy()
    anomalies[tkey] = df.loc[df_clean.index, tkey]
    del model, predictions, df_clean
    gc.collect()
    if debug:
        print(f"lof:\nanomalies={len(anomalies)}", flush=True)
    return anomalies


def dbscan(df, key, tkey="t", debug: bool = False):
    from sklearn.cluster import DBSCAN

    model = DBSCAN(eps=0.5, min_samples=5)
    df_clean = df[[key]].dropna()
    clusters = model.fit_predict(df_clean)
    anomaly_mask = clusters == -1
    anomalies = df_clean[anomaly_mask].copy()
    anomalies[tkey] = df.loc[df_clean.index, tkey]
    del model, clusters, df_clean
    gc.collect()
    if debug:
        print(f"dbscan:\nanomalies={len(anomalies)}", flush=True)
    return anomalies


def elliptic_envelope(df, key, tkey="t", debug: bool = False):
    from sklearn.covariance import EllipticEnvelope

    model = EllipticEnvelope(contamination=0.01)
    df_clean = df[[key]].dropna()
    model.fit(df_clean)
    predictions = model.predict(df_clean)
    anomaly_mask = predictions == -1
    anomalies = df_clean[anomaly_mask].copy()
    anomalies[tkey] = df.loc[df_clean.index, tkey]
    del model, predictions, df_clean
    gc.collect()
    if debug:
        print(f"elliptic_envelope:\nanomalies={len(anomalies)}", flush=True)
    return anomalies


def oneclass_svm(df, key, tkey="t", debug: bool = False):
    from sklearn.svm import OneClassSVM

    model = OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")
    df_clean = df[[key]].dropna()
    model.fit(df_clean)
    predictions = model.predict(df_clean)
    anomaly_mask = predictions == -1
    anomalies = df_clean[anomaly_mask].copy()
    anomalies[tkey] = df.loc[df_clean.index, tkey]
    del model, predictions, df_clean
    gc.collect()
    if debug:
        print(f"oneclass_svm:\nanomalies={len(anomalies)}", flush=True)
    return anomalies


def seasonal_decompose(df, key, debug: bool = False):
    from statsmodels.tsa.seasonal import seasonal_decompose as sd

    result = sd(df[key], model="additive", period=12)
    residuals = result.resid
    threshold = residuals.std() * 3
    anomalies = df[np.abs(residuals) > threshold]
    del result, residuals
    gc.collect()
    if debug:
        print(f"seasonal_decompose:\nanomalies={len(anomalies)}", flush=True)
    return anomalies


def stumpy_decompose(df, key, tkey="t", debug: bool = False, m: int = 50):
    import stumpy

    mp = stumpy.stump(df[key].values, m=m)
    matrix_profile = mp[:, 0]
    threshold = np.percentile(matrix_profile, 95)
    # Pad to match df length (matrix profile is shorter by m-1)
    padded_mp = np.full(len(df), np.nan)
    padded_mp[: len(matrix_profile)] = matrix_profile
    anomalies = df[padded_mp > threshold]
    del mp, matrix_profile, padded_mp
    gc.collect()
    if debug:
        print(f"stumpy_decompose:\nanomalies={len(anomalies)}", flush=True)
    return anomalies


# Registry of all available detection methods
DETECT_METHODS = {
    "zscore": DetectMethod(func=zscore, marker=".", color="blue"),
    "irq": DetectMethod(func=irq, marker="o", color="cyan"),
    "mad": DetectMethod(func=mad, marker="v", color="red"),
    "movingaverage": DetectMethod(func=movingaverage, marker="<", color="orange"),
    "isolationforest": DetectMethod(func=isolationforest, marker="1", color="purple"),
    "lof": DetectMethod(func=lof, marker="8", color="brown"),
    "dbscan": DetectMethod(func=dbscan, marker="s", color="green"),
    "elliptic_envelope": DetectMethod(func=elliptic_envelope, marker="p", color="yellow"),
    "oneclass_svm": DetectMethod(func=oneclass_svm, marker="*", color="pink"),
    "seasonal_decompose": DetectMethod(func=seasonal_decompose, marker="d", color="gray"),
    "stumpy_decompose": DetectMethod(func=stumpy_decompose, marker="h", color="magenta"),
}

# Default methods to run
DEFAULT_METHODS = ["mad", "dbscan", "elliptic_envelope", "stumpy_decompose"]


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


skipped_entries = {
    "Courants_Alimentations": [
        "Champ_magn",
        "Référence_GR1",
        "Référence_GR2",
        "Référence_A1",
        "Référence_A2",
        "Référence_A3",
        "Référence_A4",
    ],
    "Tensions_Alimentations": [],
    "Tensions_Aimant": [],
    "Tensions_MC1": [],
    "Tensions_MC2": [],
    "HT_default": [],
    "HTC_default": [],
    "Courants_ponts": [],
    "Numérique_Alimentation": [],
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="enter input file")
    parser.add_argument(
        "--site", help="specify a site (ex. M8, M9,...)", default="M9"
    )
    parser.add_argument(
        "--group",
        help="set group to consider",
        nargs="*",
        type=str,
        default=["Courants_Alimentations"],
    )
    parser.add_argument("--window", help="set a window", type=int, default=10)
    parser.add_argument(
        "--threshold", help="set a threshold for detection", type=float, default=1.0e-3
    )
    parser.add_argument(
        "--plot-percent",
        help="percentage of points to plot (0-100, default 10)",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--use-float32",
        help="use float32 instead of float64 to reduce memory",
        action="store_true",
    )
    parser.add_argument(
        "--methods",
        help=f"detection methods to use (available: {', '.join(DETECT_METHODS.keys())})",
        nargs="*",
        type=str,
        default=DEFAULT_METHODS,
    )
    parser.add_argument("--save", help="activate plot", action="store_true")
    parser.add_argument("--debug", help="activate debug", action="store_true")
    parser.add_argument("--dry_run", help="activate dry_run", action="store_true")
    args = parser.parse_args()
    print(f"args: {args}", flush=True)

    # Validate methods
    invalid_methods = [m for m in args.methods if m not in DETECT_METHODS]
    if invalid_methods:
        print(f"Warning: unknown methods {invalid_methods}, available: {list(DETECT_METHODS.keys())}", flush=True)

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
        case ".tdms":
            mrun = MagnetRun.fromtdms(site, insert, file)
        case _:
            raise RuntimeError(
                f"so far file with extension in {supported_formats} are implemented"
            )

    mdata = mrun.getMData()

    for group in args.group:
        select = mdata.getTdmsData(group, channel=None)
        channels = [
            field
            for field in select.columns.values.tolist()
            if field not in skipped_entries.get(group, [])
        ]
        del select
        gc.collect()
        print(f"channels in group {group}: {channels}", flush=True)

        fig, axs = plt.subplots(
            len(channels), 1, sharex=True, figsize=(10, 5 * len(channels))
        )
        # Handle single channel case
        if len(channels) == 1:
            axs = [axs]

        for i, channel in enumerate(channels):
            print(f"Processing channel {i+1}/{len(channels)}: {channel}", flush=True)

            dt = mdata.Groups[group][channel]["wf_increment"]
            t_offset = mdata.Groups[group][channel]["wf_start_offset"]

            # Get data without copy, create only needed columns
            df_raw = mdata.getData(f"{group}/{channel}")

            # Create a minimal DataFrame with only required columns
            t_values = df_raw.index.values * dt + t_offset
            y_values = df_raw[channel].values

            # Use float32 only if explicitly requested (saves memory but reduces precision)
            dtype = np.float32 if args.use_float32 else np.float64
            df = pd.DataFrame({
                tkey: t_values.astype(dtype),
                channel: y_values.astype(dtype),
            })
            del df_raw, t_values, y_values
            gc.collect()

            if args.debug:
                print(f"data shape: {df.shape}, memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB", flush=True)

            # Downsample for raw data plot
            x_plot, y_plot = downsample_for_plot(
                df[tkey].values, df[channel].values, args.plot_percent
            )
            axs[i].scatter(x_plot, y_plot, alpha=0.1, label="raw", s=1)
            del x_plot, y_plot

            df_clean = remove_low_x_outliers(
                df,
                x_col=tkey,
                y_col=channel,
                x_percentile=25,
                verbose=False,
            )
            del df
            gc.collect()

            # Downsample for clean data plot
            x_plot, y_plot = downsample_for_plot(
                df_clean[tkey].values, df_clean[channel].values, args.plot_percent
            )
            axs[i].plot(x_plot, y_plot, alpha=0.3, label="clean")
            del x_plot, y_plot

            # Plot smoothed data
            x = df_clean[tkey].values
            y = df_clean[channel].values
            smoothed = savgol(y, window=args.window)

            x_plot, smoothed_plot = downsample_for_plot(x, smoothed, args.plot_percent)
            axs[i].plot(x_plot, smoothed_plot, label="smoothed", color="magenta")
            del x, y, smoothed, x_plot, smoothed_plot
            gc.collect()

            if not args.dry_run:
                for method_name in args.methods:
                    if method_name not in DETECT_METHODS:
                        print(f"  Warning: unknown method '{method_name}', skipping", flush=True)
                        continue

                    method = DETECT_METHODS[method_name]
                    detect_func = partial(method.func, debug=args.debug)

                    print(f"  Running {method_name}...", flush=True)
                    try:
                        anomalies_df = detect_func(df_clean, channel)
                        if len(anomalies_df) > 0:
                            # Plot anomalies (usually few, no downsampling needed)
                            axs[i].scatter(
                                anomalies_df[tkey].values,
                                anomalies_df[channel].values,
                                marker=method.marker,
                                color=method.color,
                                label=method_name,
                                s=20,
                            )
                        del anomalies_df
                    except Exception as e:
                        print(f"  Warning: {method_name} failed: {e}", flush=True)
                    gc.collect()

            del df_clean
            gc.collect()

            axs[i].set_ylabel(channel)
            axs[i].legend(loc="upper right", fontsize="small")
            axs[i].grid()

        plt.suptitle(f"{filename}: {group}")
        plt.tight_layout()
        plt.xlabel("time (s)")

        if args.save:
            imagefile = filename.replace(f_extension, "")
            plotfile = f"{imagefile}_{group}_anomalies.png"
            print(f"save to file - {plotfile}")
            plt.savefig(plotfile, dpi=300)
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)

        del fig, axs
        gc.collect()


if __name__ == "__main__":
    main()
