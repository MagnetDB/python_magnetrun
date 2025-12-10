import os
import argparse


from python_magnetrun.MagnetRun import MagnetRun
import matplotlib.pyplot as plt


def remove_low_x_outliers(
    df,
    x_col="x",
    y_col="y",
    x_percentile=25,
    method="iqr",
    threshold=1.5,
    verbose=False,
):
    """
    Detect and remove outliers specifically in the LOW x region.

    Parameters:
    -----------
    df : pandas DataFrame
        Data
    x_col : str
        Name of x column
    y_col : str
        Name of y column
    x_percentile : float
        Consider x values below this percentile as "low" (default 25 = bottom quartile)
    method : str
        'iqr' - Interquartile range on y values in low-x region
        'zscore' - Z-score on y values in low-x region
        'both_dims' - IQR on both x and y in low-x region
    threshold : float
        IQR multiplier or z-score threshold
    verbose : bool
        Print diagnostic info

    Returns:
    --------
    DataFrame with low-x outliers removed
    """
    import numpy as np

    df_clean = df.copy()
    x = df_clean[x_col].values
    y = df_clean[y_col].values

    # Find the threshold x value for "low x" region
    x_cutoff = np.percentile(x, x_percentile)

    # Identify points in low-x region
    low_x_mask = x <= x_cutoff
    n_low_x = np.sum(low_x_mask)

    if verbose:
        print(
            f"Low-x region: x <= {x_cutoff:.4f} ({n_low_x} points, {100*n_low_x/len(df):.1f}%)"
        )

    # Start with all points as inliers
    inlier_mask = np.ones(len(df_clean), dtype=bool)

    if method == "iqr":
        # Apply IQR only to y values in low-x region
        if n_low_x > 0:
            y_low = y[low_x_mask]
            Q1 = np.percentile(y_low, 25)
            Q3 = np.percentile(y_low, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Mark outliers in low-x region
            low_x_outliers = low_x_mask & ((y < lower_bound) | (y > upper_bound))
            inlier_mask = ~low_x_outliers

            if verbose:
                n_removed = np.sum(low_x_outliers)
                print(
                    f"  IQR on low-x y values: y in [{lower_bound:.4f}, {upper_bound:.4f}]"
                )
                print(f"  Removed {n_removed} outliers from low-x region")

    elif method == "zscore":
        # Apply z-score only to y values in low-x region
        if n_low_x > 0:
            y_low = y[low_x_mask]
            mean_y = np.mean(y_low)
            std_y = np.std(y_low)
            if std_y > 0:
                z_scores = np.abs((y[low_x_mask] - mean_y) / std_y)
                low_x_outliers_bool = z_scores > threshold

                # Convert boolean mask back to full-length mask
                low_x_outliers = np.zeros(len(df_clean), dtype=bool)
                low_x_indices = np.where(low_x_mask)[0]
                low_x_outliers[low_x_indices[low_x_outliers_bool]] = True

                inlier_mask = ~low_x_outliers

                if verbose:
                    n_removed = np.sum(low_x_outliers)
                    print(f"  Z-score on low-x y values (threshold={threshold})")
                    print(f"  Removed {n_removed} outliers from low-x region")

    elif method == "both_dims":
        # Apply IQR to both x and y in low-x region
        if n_low_x > 0:
            x_low = x[low_x_mask]
            y_low = y[low_x_mask]

            # IQR on x values in low-x region
            Q1_x = np.percentile(x_low, 25)
            Q3_x = np.percentile(x_low, 75)
            IQR_x = Q3_x - Q1_x
            x_lower = Q1_x - threshold * IQR_x
            x_upper = Q3_x + threshold * IQR_x

            # IQR on y values in low-x region
            Q1_y = np.percentile(y_low, 25)
            Q3_y = np.percentile(y_low, 75)
            IQR_y = Q3_y - Q1_y
            y_lower = Q1_y - threshold * IQR_y
            y_upper = Q3_y + threshold * IQR_y

            # Mark outliers
            low_x_outliers = low_x_mask & (
                (x < x_lower) | (x > x_upper) | (y < y_lower) | (y > y_upper)
            )
            inlier_mask = ~low_x_outliers

            if verbose:
                n_removed = np.sum(low_x_outliers)
                print("  IQR on low-x region (both dims):")
                print(f"    x in [{x_lower:.4f}, {x_upper:.4f}]")
                print(f"    y in [{y_lower:.4f}, {y_upper:.4f}]")
                print(f"  Removed {n_removed} outliers from low-x region")

    else:
        raise ValueError(f"Unknown method: {method}")

    return df_clean[inlier_mask].reset_index(drop=True)


def zscore(df, key, debug: bool = False):
    from scipy import stats
    import numpy as np

    df["zscore"] = np.abs(stats.zscore(df[key]))
    anomalies = df[df["zscore"] > 3]
    if debug:
        print(f"zcore:\nanomalies={anomalies}", flush=True)
    return anomalies


def irq(df, key, debug: bool = False):
    Q1 = df[key].quantile(0.25)
    Q3 = df[key].quantile(0.75)
    IQR = Q3 - Q1
    anomalies = df[(df[key] < Q1 - 1.5 * IQR) | (df[key] > Q3 + 1.5 * IQR)]
    if debug:
        print(f"irq:\nanomalies={anomalies}", flush=True)
    return anomalies


def mad(df, key, debug: bool = False):
    import numpy as np

    median = df[key].median()
    mad = np.median(np.abs(df[key] - median))
    modified_z = 0.6745 * (df[key] - median) / mad
    anomalies = df[np.abs(modified_z) > 3.5]
    if debug:
        print(f"mad:\nanomalies={anomalies}", flush=True)
    return anomalies


def movingaverage(df, key, window: int = 7, debug: bool = False):
    import numpy as np

    # Simple Moving Average

    df["ma"] = df[key].rolling(window=window).mean()
    df["deviation"] = np.abs(df[key] - df["ma"])
    anomalies = df[df["deviation"] > 3 * df["deviation"].std()]
    if debug:
        print(f"movingaverage:\nanomalies={anomalies}", flush=True)
    return anomalies


def isolationforest(df, key, debug: bool = False):
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(contamination=0.01, random_state=42)
    # iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df_clean = df[[key]].dropna()
    model.fit(df_clean)
    df_clean["anomaly"] = model.predict(df_clean)
    df_clean["t"] = df["t"]
    anomalies = df_clean[df_clean["anomaly"] == -1]
    if debug:
        print(f"isolationforest:\nanomalies={anomalies}", flush=True)
    return anomalies


def lof(df, key, debug: bool = False):
    from sklearn.neighbors import LocalOutlierFactor

    model = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    df_clean = df[[key]].dropna()
    df_clean["anomaly"] = model.fit_predict(df_clean)
    df_clean["t"] = df["t"]
    anomalies = df_clean[df_clean["anomaly"] == -1]
    if debug:
        print(f"lof:\nanomalies={anomalies}", flush=True)
    return anomalies


def dbscan(df, key, debug: bool = False):
    from sklearn.cluster import DBSCAN

    model = DBSCAN(eps=0.5, min_samples=5)
    df_clean = df[[key]].dropna()
    df_clean["cluster"] = model.fit_predict(df_clean)
    df_clean["t"] = df["t"]
    anomalies = df_clean[df_clean["cluster"] == -1]
    if debug:
        print(f"dbscan:\nanomalies={anomalies}", flush=True)
    return anomalies


def elliptic_envelope(df, key, debug: bool = False):
    from sklearn.covariance import EllipticEnvelope

    model = EllipticEnvelope(contamination=0.01)
    df_clean = df[[key]].dropna()
    model.fit(df_clean)
    df_clean["anomaly"] = model.predict(df_clean)
    df_clean["t"] = df["t"]
    anomalies = df_clean[df_clean["anomaly"] == -1]
    if debug:
        print(f"elliptic_envelope:\nanomalies={anomalies}", flush=True)
    return anomalies


def oneclass_svm(df, key, debug: bool = False):
    from sklearn.svm import OneClassSVM

    model = OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")
    df_clean = df[[key]].dropna()
    model.fit(df_clean)
    df_clean["anomaly"] = model.predict(df_clean)
    df_clean["t"] = df["t"]
    anomalies = df_clean[df_clean["anomaly"] == -1]
    if debug:
        print(f"oneclass_svm:\nanomalies={anomalies}", flush=True)
    return anomalies


def seasonal_decompose(df, key, debug: bool = False):
    from statsmodels.tsa.seasonal import seasonal_decompose
    import numpy as np

    result = seasonal_decompose(df[key], model="additive", period=12)
    residuals = result.resid
    threshold = residuals.std() * 3
    anomalies = df[np.abs(residuals) > threshold]
    if debug:
        print(f"seasonal_decompose:\nresult={result}", flush=True)
    return anomalies


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
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="enter input file")
parser.add_argument(
    "--site", help="specify a site (ex. M8, M9,...)", default="M9"
)  # use housing instead
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
parser.add_argument("--save", help="activate plot", action="store_true")
parser.add_argument("--debug", help="activate debug", action="store_true")
parser.add_argument("--dry_run", help="activate dry_run", action="store_true")
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
        if field not in skipped_entries[group]
    ]
    print(f"channels in group {group}: {channels}", flush=True)

    fig, axs = plt.subplots(
        len(channels), 1, sharex=True, figsize=(10, 5 * len(channels))
    )
    for i, channel in enumerate(channels):
        dt = mdata.Groups[group][channel]["wf_increment"]
        t_offset = mdata.Groups[group][channel]["wf_start_offset"]

        df = mdata.getData(f"{group}/{channel}").copy()
        df[tkey] = df.index * dt + t_offset
        if args.debug:
            print(f"data:\n{df.head()}", flush=True)

        df.plot(
            x=tkey,
            y=channel,
            kind="scatter",
            alpha=0.1,
            ax=axs[i],
            label="raw",
        )

        df_clean = remove_low_x_outliers(
            df,
            x_col=tkey,
            y_col=channel,
            x_percentile=25,
            verbose=False,
        )
        del df
        df_clean.plot(x=tkey, y=channel, alpha=0.3, ax=axs[i], label="clean")

        # plotted smoothed data
        from python_magnetrun.processing.smoothers import savgol

        x = df_clean["t"].to_numpy()
        y = df_clean[channel].to_numpy()
        smoothed = savgol(y, window=args.window)
        axs[i].plot(
            x,
            smoothed,
            label="smoothed",
            color="magenta",
        )

        if not args.dry_run:
            detect_methods = {
                # "zcore": (zscore, "."),
                # "irq": (irq, "o"),
                "mad": (mad, "v", "red"),
                # "movingaverage": (movingaverage, "<"),
                # "isolationforest": (isolationforest, "1"),
                # "lof": (lof, "8"),
                "dbscan": (dbscan, "s", "green"),
                "elliptic_envelope": (elliptic_envelope, "p", "yellow"),
                # "oneclass_svm": (oneclass_svm, "*"),
                # "seasonal_decompose": (seasonal_decompose, "d"),
            }
            for method_name, (method_func, marker, color) in detect_methods.items():
                anomalies_df = method_func(df_clean, channel, debug=args.debug)
                anomalies_df.plot(
                    x=tkey,
                    y=channel,
                    kind="scatter",
                    marker=marker,
                    color=color,
                    ax=axs[i],
                    label=method_name,
                )

            """
            df["ewma"] = df[channel].ewm(span=7).mean()
            df.plot(
                x=tkey,
                y="ewma",
                kind="scatter",
                marker=".",
                color="yellow",
                ax=my_ax,
                label=r"ewma",
            )
            """
    for ax in axs:
        ax.grid()

    plt.suptitle(f"{filename}: {group}")
    plt.tight_layout()
    plt.xlabel("time (s)")

    if args.save:
        imagefile = filename.replace(f_extension, "")
        plotfile = f"{imagefile}_{group}_anomalies.png"
        print(f"save to file - {plotfile}")
        plt.savefig(plotfile, dpi=300)
    else:
        plt.show()
        plt.close()
