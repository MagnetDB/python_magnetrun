import numpy as np
from scipy import stats
import pandas as pd
from seaborn import reset_defaults
from sklearn.ensemble import IsolationForest
from typing import Union, Tuple, List

from .MagnetRun import MagnetRun


class TimeSeriesAnomalyDetector:
    """A class implementing multiple methods for time series anomaly detection."""

    def __init__(self, series: Union[list, np.ndarray, pd.Series]):
        """
        Initialize the detector with a time series.

        Args:
            series: The time series data
        """
        self.series = np.array(series)
        self.n = len(series)

    def zscore_detection(self, threshold: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Z-score method.

        Args:
            threshold: Number of standard deviations to use as threshold

        Returns:
            Tuple of (anomaly indices, anomaly scores)
        """
        print(f"zcore: threshold={threshold}")

        mean = np.mean(self.series)
        std = np.std(self.series)
        z_scores = np.abs((self.series - mean) / std)
        anomalies = z_scores > threshold
        return np.where(anomalies)[0], z_scores

    def iqr_detection(
        self, iqr_multiplier: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using the Interquartile Range (IQR) method.

        Args:
            iqr_multiplier: Multiplier for IQR to set threshold

        Returns:
            Tuple of (anomaly indices, scores based on distance from IQR bounds)
        """
        print(f"irq: irq={iqr_multiplier}")
        Q1 = np.percentile(self.series, 25)
        Q3 = np.percentile(self.series, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        scores = np.maximum(
            (self.series - upper_bound) / IQR, (lower_bound - self.series) / IQR
        )
        anomalies = (self.series < lower_bound) | (self.series > upper_bound)
        return np.where(anomalies)[0], scores

    def moving_average_detection(
        self, window: int = 20, threshold: float = 3.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using moving average and standard deviation.

        Args:
            window: Size of the moving window
            threshold: Number of standard deviations to use as threshold

        Returns:
            Tuple of (anomaly indices, deviation scores)
        """
        print(f"moving_average: window={window}, threshold={threshold}")

        rolling_mean = pd.Series(self.series).rolling(window=window).mean()
        rolling_std = pd.Series(self.series).rolling(window=window).std()

        deviation = np.abs(self.series - rolling_mean) / rolling_std
        scores = deviation.fillna(0)
        anomalies = scores > threshold
        return np.where(anomalies)[0], scores

    def moving_median_detection(
        self, window: int = 20, threshold: float = 3.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using moving median and standard deviation.

        Args:
            window: Size of the moving window
            threshold: Number of standard deviations to use as threshold

        Returns:
            Tuple of (anomaly indices, deviation scores)
        """
        print(f"moving_median: window={window}, threshold={threshold}")
        rolling_median = pd.Series(self.series).rolling(window=window).median()
        rolling_std = pd.Series(self.series).rolling(window=window).std()

        deviation = np.abs(self.series - rolling_median) / rolling_std
        scores = deviation.fillna(0)
        anomalies = scores > threshold
        return np.where(anomalies)[0], scores

    def moving_zscore_detection(
        self, window: int = 20, threshold: float = 3.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using moving zcore.

        Args:
            window: Size of the moving window
            threshold: Number of standard deviations to use as threshold

        Returns:
            Tuple of (anomaly indices, deviation scores)
        """
        from scipy import stats

        print(f"moving_median: window={window}, threshold={threshold}")
        rolling_median = pd.Series(self.series).rolling(window=window).median()
        mad = stats.median_abs_deviation(self.series)

        deviation = 0.6745 * np.abs(self.series - rolling_median) / mad
        scores = deviation.fillna(0)
        anomalies = scores > threshold
        return np.where(anomalies)[0], scores

    def isolation_forest_detection(
        self, contamination: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Isolation Forest algorithm.

        Args:
            contamination: Expected proportion of anomalies

        Returns:
            Tuple of (anomaly indices, anomaly scores)
        """
        print(f"isolation_forest: contamination={contamination}")
        clf = IsolationForest(contamination=contamination, random_state=42)
        reshaped_data = self.series.reshape(-1, 1)
        scores = -clf.fit_predict(reshaped_data)  # -1 for anomalies, 1 for normal
        return np.where(scores > 0)[0], scores

    def detect_all_methods(self) -> dict:
        """
        Run all anomaly detection methods and return their results.

        Returns:
            Dictionary containing results from all methods
        """

        # TODO:
        # 1. add params as dict for each methods
        # 2. add Zscore with MAD
        results = {
            "zscore": self.zscore_detection(),
            "iqr": self.iqr_detection(),
            "moving_average": self.moving_average_detection(),
            "moving_median": self.moving_median_detection(),
            "moving_zscore": self.moving_zscore_detection(),
            "isolation_forest": self.isolation_forest_detection(),
        }
        return results


def plot_anomalies(
    series: np.ndarray,
    scores,
    anomaly_indices: np.ndarray,
    title: str = "Anomaly Detection",
):
    """
    Plot the time series with highlighted anomalies.
    Requires matplotlib.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle(title)

    ax1.plot(series, label="Original Data")
    ax1.scatter(
        anomaly_indices, series[anomaly_indices], color="red", label="Anomalies"
    )
    ax1.grid()
    ax1.legend()

    ax2.plot(scores, label="Score")
    # ax2.xlabel("t [s]")
    ax2.legend()
    ax2.grid()
    plt.show()
    plt.close()


if __name__ == "__main__":
    import os
    from natsort import natsorted
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="+", help="enter input file")
    parser.add_argument(
        "--site", help="specify a site (ex. M8, M9,...)", default="M9"
    )  # use housing instead
    parser.add_argument("--insert", help="specify an insert", default="notdefined")
    parser.add_argument("--debug", help="acticate debug", action="store_true")
    parser.add_argument("--plot", help="acticate plot", action="store_true")
    parser.add_argument("--save", help="activate plot", action="store_true")
    parser.add_argument("--normalize", help="normalize data", action="store_true")

    parser.add_argument(
        "--threshold",
        help="specify threshold for outliers detection",
        type=float,
        default=2,
    )
    parser.add_argument("--window", help="rolling window size", type=int, default=120)
    args = parser.parse_args()
    print(f"args: {args}", flush=True)

    # load df pandas from input_file
    # check extension
    supported_formats = [".tdms"]

    inputs = {}
    extensions = {}
    for i, file in enumerate(args.input_file):
        f_extension = os.path.splitext(file)[-1]
        if f_extension not in extensions:
            extensions[f_extension] = [i]
        else:
            extensions[f_extension].append(i)
    print(f"extensions: {extensions}")

    input_files = natsorted(args.input_file)
    for file in input_files:
        f_extension = os.path.splitext(file)[-1]
        if f_extension not in supported_formats:
            raise RuntimeError(
                f"so far file with extension in {supported_formats} are implemented"
            )

        filename = os.path.basename(file)
        result = filename.startswith("M")
        insert = "tututu"
        site = "tttt"

        try:
            index = filename.index("_")
            site = filename[0:index]

            match f_extension:
                case ".tdms":
                    mrun = MagnetRun.fromtdms(site, insert, file)
                case _:
                    raise RuntimeError(
                        f"so far file with extension in {supported_formats} are implemented"
                    )
        except Exception as error:
            print(f"{file}: an error occurred when loading:", error)
            continue

        mdata = mrun.getMData()

        threshold = args.threshold
        window = args.window
        anomaly_dicts = {
            "zscore": [],
            "iqr": [],
            "moving_average": [],
            "isolation_forest": [],
        }

        Ikeys = [
            "Courants_Alimentations/Courant_GR1",
            "Courants_Alimentations/Courant_GR2",
        ]
        Ukeys_H = [
            "Tensions_Aimant/Interne1",
            "Tensions_Aimant/Interne2",
            "Tensions_Aimant/Interne3",
            "Tensions_Aimant/Interne4",
            "Tensions_Aimant/Interne5",
            "Tensions_Aimant/Interne6",
            "Tensions_Aimant/Interne7",
        ]
        Ukeys_B = [
            "Tensions_Aimant/Externe1",
            "Tensions_Aimant/Externe2",
        ]
        for key in Ikeys:
            (group, channel) = key.split("/")
            (symbol, unit) = mdata.getUnitKey(key)

            ts = mdata.Data[group][channel]
            if args.normalize:
                ts = ts / ts.max()
            detector = TimeSeriesAnomalyDetector(ts)

            all_results = detector.detect_all_methods()
            for method, result in all_results.items():
                anomaly_indices, scores = result
                plot_anomalies(
                    ts.to_numpy(),
                    scores,
                    anomaly_indices,
                    title=f"{file}: Anomaly Detection for {key} ({method})",
                )
