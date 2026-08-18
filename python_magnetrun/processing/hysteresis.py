import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from python_magnetrun.outliers import detect_outliers

logger = logging.getLogger(__name__)

_VALID_METHODS = {"iqr", "zscore", "mad", "isolation_forest"}


def remove_outliers(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Remove outliers from x,y data using the canonical outlier detector.

    Parameters
    ----------
    df : pd.DataFrame
        Data with x and y columns.
    x_col : str
        Name of x column.
    y_col : str
        Name of y column.
    method : str
        'iqr'    — Interquartile Range (default, robust)
        'zscore' — Z-score
        'mad'    — Median Absolute Deviation (most robust)
    threshold : float
        Multiplier on IQR/MAD, or z-score cutoff (default 1.5).

    Returns
    -------
    pd.DataFrame
        Copy of df with outlier rows removed.

    See Also
    --------
    python_magnetrun.outliers.remove_outliers :
        Array variant operating on ``(data, time)`` numpy arrays; also
        supports interpolation and clipping strategies.
    remove_outliers_by_x_range :
        Domain-knowledge alternative that filters by an explicit x-value
        range rather than a statistical method.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {sorted(_VALID_METHODS)}; got {method!r}")
    mask_x = detect_outliers(df[x_col].values, method=method, threshold=threshold)
    mask_y = detect_outliers(df[y_col].values, method=method, threshold=threshold)
    n_removed = int((mask_x | mask_y).sum())
    logger.debug(f"{method} (threshold={threshold}): removed {n_removed} outliers")
    return df[~(mask_x | mask_y)].reset_index(drop=True)


def remove_outliers_by_x_range(
    df: pd.DataFrame,
    x_col: str = "x",
    x_min: float | None = None,
    x_max: float | None = None,
) -> pd.DataFrame:
    """
    Remove outliers by specifying an acceptable x range.

    Parameters
    ----------
    df : pd.DataFrame
        Data with at least an x column.
    x_col : str
        Name of the x column.
    x_min : float, optional
        Minimum acceptable x value; rows below this are removed.
    x_max : float, optional
        Maximum acceptable x value; rows above this are removed.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with out-of-range x rows removed.

    See Also
    --------
    remove_outliers :
        Statistical variant (IQR / z-score / MAD) that operates on both
        x and y columns.
    """
    df_clean = df.copy()
    mask = np.ones(len(df_clean), dtype=bool)

    if x_min is not None:
        mask = mask & (df_clean[x_col] >= x_min)
    if x_max is not None:
        mask = mask & (df_clean[x_col] <= x_max)

    n_removed = len(df_clean) - np.sum(mask)
    logger.debug(f"X-range method: Removed {n_removed} points")
    logger.debug(f"  Kept x in range: [{x_min}, {x_max}]")

    return df_clean[mask].reset_index(drop=True)


def remove_low_x_outliers(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    x_percentile: float = 25,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
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

    Returns:
    --------
    DataFrame with low-x outliers removed
    """
    df_clean = df.copy()
    x = df_clean[x_col].values
    y = df_clean[y_col].values

    # Find the threshold x value for "low x" region
    x_cutoff = np.percentile(x, x_percentile)

    # Identify points in low-x region
    low_x_mask = x <= x_cutoff
    n_low_x = np.sum(low_x_mask)

    logger.debug(
        f"Low-x region: x <= {x_cutoff:.4f} ({n_low_x} points, {100 * n_low_x / len(df):.1f}%)"
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

            n_removed = np.sum(low_x_outliers)
            logger.debug(f"  IQR on low-x y values: y in [{lower_bound:.4f}, {upper_bound:.4f}]")
            logger.debug(f"  Removed {n_removed} outliers from low-x region")

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

                n_removed = np.sum(low_x_outliers)
                logger.debug(f"  Z-score on low-x y values (threshold={threshold})")
                logger.debug(f"  Removed {n_removed} outliers from low-x region")

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

            n_removed = np.sum(low_x_outliers)
            logger.debug("  IQR on low-x region (both dims):")
            logger.debug(f"    x in [{x_lower:.4f}, {x_upper:.4f}]")
            logger.debug(f"    y in [{y_lower:.4f}, {y_upper:.4f}]")
            logger.debug(f"  Removed {n_removed} outliers from low-x region")

    else:
        raise ValueError(f"Unknown method: {method}")

    return df_clean[inlier_mask].reset_index(drop=True)


def remove_x_region_outliers(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    x_threshold: float | None = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Remove outliers in a specific x region (e.g., x < -0.8).

    Parameters:
    -----------
    df : pandas DataFrame
        Data
    x_col : str
        Name of x column
    y_col : str
        Name of y column
    x_threshold : float
        x value marking boundary (e.g., -0.8 for "remove outliers where x < -0.8")
    method : str
        'iqr' or 'zscore'
    threshold : float
        IQR multiplier or z-score cutoff

    Returns:
    --------
    DataFrame with outliers in x_region removed
    """
    if x_threshold is None:
        raise ValueError("x_threshold must be specified")

    df_clean = df.copy()
    x = df_clean[x_col].values
    y = df_clean[y_col].values

    # Identify points in the specified region
    region_mask = x <= x_threshold
    n_region = np.sum(region_mask)

    logger.debug(f"X-region: x <= {x_threshold:.4f} ({n_region} points)")

    inlier_mask = np.ones(len(df_clean), dtype=bool)

    if method == "iqr":
        if n_region > 0:
            y_region = y[region_mask]
            Q1 = np.percentile(y_region, 25)
            Q3 = np.percentile(y_region, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            region_outliers = region_mask & ((y < lower_bound) | (y > upper_bound))
            inlier_mask = ~region_outliers

            n_removed = np.sum(region_outliers)
            logger.debug(f"  IQR on y in region: y in [{lower_bound:.4f}, {upper_bound:.4f}]")
            logger.debug(f"  Removed {n_removed} outliers")

    elif method == "zscore":
        if n_region > 0:
            y_region = y[region_mask]
            mean_y = np.mean(y_region)
            std_y = np.std(y_region)
            if std_y > 0:
                z_scores = np.abs((y[region_mask] - mean_y) / std_y)
                region_outliers_bool = z_scores > threshold

                region_outliers = np.zeros(len(df_clean), dtype=bool)
                region_indices = np.where(region_mask)[0]
                region_outliers[region_indices[region_outliers_bool]] = True

                inlier_mask = ~region_outliers

                n_removed = np.sum(region_outliers)
                logger.debug(f"  Z-score on y in region (threshold={threshold})")
                logger.debug(f"  Removed {n_removed} outliers")

    else:
        raise ValueError(f"Unknown method: {method}")

    return df_clean[inlier_mask].reset_index(drop=True)


def estimate_hysteresis_parameters(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    n_levels: int | None = None,
) -> dict[str, Any]:
    """
    Estimate multi-level hysteresis parameters from empirical x,y data.

    The function identifies:
    - Output levels (distinct y values or clusters)
    - Ascending thresholds (x values where y transitions low→high while x increases)
    - Descending thresholds (x values where y transitions high→low while x decreases)

    Parameters:
    -----------
    df : pandas DataFrame
        Must have x and y columns
    x_col : str
        Name of the x column
    y_col : str
        Name of the y column
    n_levels : int, optional
        If provided, cluster y values into this many discrete levels.
        If None, auto-detect distinct levels (best for already-discrete data).

    Returns:
    --------
    dict with:
        'thresholds': list of tuples (asc_threshold, desc_threshold) - pure Python floats
        'low_values': list of floats sorted in ascending order
        'high_values': list of floats sorted in ascending order
        'diagnostics': dict with transition information
    """
    x = df[x_col].values
    y = df[y_col].values.copy()

    # Cluster y values into discrete levels if requested
    if n_levels is not None:
        try:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=n_levels, random_state=42)
            y_clusters = km.fit_predict(y.reshape(-1, 1))
            y = km.cluster_centers_[y_clusters].flatten()
            logger.debug(f"Clustered y into {n_levels} levels")
        except ImportError:
            logger.warning("sklearn not available. Using data-driven level detection instead.")

    # Compute derivatives to identify direction of x change
    dx = np.diff(x, prepend=np.nan)
    dy = np.diff(y, prepend=np.nan)

    # Mark direction of x change
    is_increasing = dx[1:] > 1e-10
    is_decreasing = dx[1:] < -1e-10

    # Identify transitions (where y changes significantly)
    transitions = np.where(np.abs(dy[1:]) > 1e-10)[0] + 1

    # Auto-detect distinct output levels
    unique_y = sorted(set(np.round(y, decimals=8)))  # Round to handle floating point errors
    n_levels_found = len(unique_y)

    logger.debug(f"Found {n_levels_found} distinct output levels")
    logger.debug(f"Found {len(transitions)} transitions")

    # Collect threshold observations
    ascending_observations = {}  # level_idx -> [x_values]
    descending_observations = {}  # level_idx -> [x_values]

    for t_idx in transitions:
        if t_idx == 0:
            continue

        y_from = np.round(y[t_idx - 1], decimals=8)
        y_to = np.round(y[t_idx], decimals=8)
        x_at_transition = float(x[t_idx])

        try:
            level_from = unique_y.index(y_from)
            level_to = unique_y.index(y_to)
        except ValueError:
            continue

        # Ascending transition: x increasing and y going from lower to higher level
        if level_to > level_from and is_increasing[t_idx - 1]:
            if level_to not in ascending_observations:
                ascending_observations[level_to] = []
            ascending_observations[level_to].append(x_at_transition)

        # Descending transition: x decreasing and y going from higher to lower level
        if level_to < level_from and is_decreasing[t_idx - 1]:
            if level_from not in descending_observations:
                descending_observations[level_from] = []
            descending_observations[level_from].append(x_at_transition)

    # Compute mean thresholds for each level
    ascending_thresholds = []
    descending_thresholds = []

    for i in range(n_levels_found):
        if i in ascending_observations:
            asc_threshold = float(np.mean(ascending_observations[i]))
        else:
            asc_threshold = None

        if i in descending_observations:
            desc_threshold = float(np.mean(descending_observations[i]))
        else:
            desc_threshold = None

        ascending_thresholds.append(asc_threshold)
        descending_thresholds.append(desc_threshold)

        asc_str = f"{asc_threshold:.4f}" if asc_threshold is not None else "N/A"
        desc_str = f"{desc_threshold:.4f}" if desc_threshold is not None else "N/A"
        logger.debug(
            f"Level {i} (y={unique_y[i]:g}): asc_threshold={asc_str}, desc_threshold={desc_str}"
        )

    # Build thresholds list - pair ascending/descending for each level
    # Filter out levels with missing thresholds
    thresholds = []
    valid_indices = []
    for i, (asc, desc) in enumerate(zip(ascending_thresholds, descending_thresholds, strict=False)):
        if asc is not None and desc is not None:
            thresholds.append((asc, desc))
            valid_indices.append(i)

    # Extract corresponding output values for valid indices (convert to pure Python floats)
    high_values = [float(unique_y[i]) for i in valid_indices]
    low_values = [float(unique_y[max(0, i - 1)]) for i in valid_indices]

    # Sort all lists together by low_values to ensure ascending order
    sorted_items = sorted(zip(low_values, high_values, thresholds, strict=False))
    low_values = [item[0] for item in sorted_items]
    high_values = [item[1] for item in sorted_items]
    thresholds = [item[2] for item in sorted_items]

    # Diagnostics
    diagnostics = {
        "n_transitions": len(transitions),
        "n_levels_found": len(thresholds),
        "n_valid_levels": len(valid_indices),
        "ascending_obs": {k: len(v) for k, v in ascending_observations.items()},
        "descending_obs": {k: len(v) for k, v in descending_observations.items()},
    }

    return {
        "thresholds": thresholds,
        "low_values": low_values,
        "high_values": high_values,
        "diagnostics": diagnostics,
    }


# Example usage
if __name__ == "__main__":
    # Generate synthetic hysteresis data for testing
    np.random.seed(42)

    # Create input signal with multiple cycles
    t = np.linspace(0, 4 * np.pi, 500)
    x_base = np.sin(t)

    # Create multi-level hysteresis output with THREE discrete levels
    # Levels: [0, 1, 2] at output values [0.0, 1.0, 2.0]
    y = np.zeros_like(x_base)
    state = 0

    # Define hysteresis thresholds
    asc_threshold_1 = 0.7  # Transition from level 0->1 when x increases
    desc_threshold_1 = -0.7  # Transition from level 1->0 when x decreases
    asc_threshold_2 = 0.3  # Transition from level 1->2 when x increases
    desc_threshold_2 = -0.3  # Transition from level 2->1 when x decreases

    y_levels = [0.0, 1.0, 2.0]

    for i in range(len(x_base)):
        if state == 0 and x_base[i] > asc_threshold_1:
            state = 1
        elif state == 1:
            if x_base[i] < desc_threshold_1:
                state = 0
            elif x_base[i] > asc_threshold_2:
                state = 2
        elif state == 2 and x_base[i] < desc_threshold_2:
            state = 1

        y[i] = y_levels[state]

    # Add small noise
    y_noisy = y + np.random.normal(0, 0.05, len(y))

    # Create DataFrame
    df = pd.DataFrame({"x": x_base, "y": y_noisy})

    logger.info("=== Original Data ===")
    logger.info(f"Number of points: {len(df)}")
    logger.info(f"x range: [{df['x'].min():.4f}, {df['x'].max():.4f}]")
    logger.info(f"y range: [{df['y'].min():.4f}, {df['y'].max():.4f}]")

    # Option 1: Remove outliers specifically at LOW x values (bottom 25%)
    logger.info("\n=== Removing Low-X Outliers (bottom 25% of x) ===")
    df_clean = remove_low_x_outliers(df, method="iqr", threshold=1.5)
    logger.info(f"Points after cleaning: {len(df_clean)}")

    # Option 2: Remove outliers in a specific x region (e.g., x < -0.8)
    # print("\n=== Removing Outliers in Specific X Region ===")
    # df_clean = remove_x_region_outliers(df, x_threshold=-0.8, method='iqr', threshold=1.5, verbose=True)

    # Option 3: Remove outliers using standard IQR (both dimensions)
    # df_clean = remove_outliers(df, method='iqr', threshold=1.5, verbose=True)

    # Option 4: Remove outliers by x range if you know acceptable bounds
    # df_clean = remove_outliers_by_x_range(df, x_min=-1.0, x_max=1.0, verbose=True)

    # Estimate parameters - use clustering to handle remaining noise
    logger.info("\n=== Parameter Estimation (with clustering) ===")
    result = estimate_hysteresis_parameters(df_clean, n_levels=3)

    logger.info("\n=== Estimated Parameters ===")
    if result["thresholds"]:
        logger.info(f"Thresholds (list of tuples): {result['thresholds']}")
        logger.info(f"Type of first threshold: {type(result['thresholds'][0])}")
        logger.info(f"Type of first element in threshold: {type(result['thresholds'][0][0])}")

        logger.info(f"\nLow values (sorted): {result['low_values']}")
        logger.info(f"Type: {type(result['low_values'][0])}")

        logger.info(f"\nHigh values (sorted): {result['high_values']}")
        logger.info(f"Type: {type(result['high_values'][0])}")
    else:
        logger.info("No valid thresholds found with current data.")

    logger.info(f"\nDiagnostics: {result['diagnostics']}")


def refine_thresholds_with_hysteresis_loop(
    df: pd.DataFrame, x_col: str = "x", y_col: str = "y"
) -> dict[Any, dict[str, Any]]:
    """
    Alternative method: Analyze the hysteresis loop directly.

    For each x value, collect all corresponding y values.
    The ascending path gives one set of values, descending gives another.
    The transition points define the thresholds.
    """
    x = df[x_col].values
    y = df[y_col].values

    # Sort by x to identify ascending portions
    sorted_indices = np.argsort(x)

    # Find turning points (local extrema in x)
    dx = np.diff(x)
    turning_points = np.where(np.diff(np.sign(dx)))[0]

    # Separate into ascending and descending phases
    ascending_mask = np.zeros(len(x), dtype=bool)
    descending_mask = np.zeros(len(x), dtype=bool)

    # Simple heuristic: assume first half is ascending, second is descending
    # In reality, you'd use the turning points more carefully
    if len(turning_points) > 0:
        for i in range(len(turning_points) - 1):
            t1, t2 = turning_points[i], turning_points[i + 1]
            if np.sum(np.diff(x[t1 : t2 + 1]) > 0) > np.sum(np.diff(x[t1 : t2 + 1]) < 0):
                ascending_mask[t1 : t2 + 1] = True
            else:
                descending_mask[t1 : t2 + 1] = True

    # For each unique y level, find min and max x on ascending and descending paths
    unique_y = sorted(set(y))

    thresholds_by_level = {}
    for y_level in unique_y:
        y_indices = np.where(y == y_level)[0]

        if len(y_indices) > 0:
            x_at_level = x[y_indices]
            ascending_x = x_at_level[ascending_mask[y_indices]]
            descending_x = x_at_level[descending_mask[y_indices]]

            asc_threshold = np.min(ascending_x) if len(ascending_x) > 0 else np.inf
            desc_threshold = np.max(descending_x) if len(descending_x) > 0 else -np.inf

            thresholds_by_level[y_level] = {
                "ascending": asc_threshold,
                "descending": desc_threshold,
            }

    return thresholds_by_level


def hysteresis_model(
    x: np.ndarray,
    ascending_threshold: float,
    descending_threshold: float,
    low_value: float = 0,
    high_value: float = 1,
) -> np.ndarray:
    """
    A simple hysteresis function model.

    Parameters:
    -----------
    x : array-like
        Input signal
    ascending_threshold : float
        Threshold for switching from low to high state when input is increasing
    descending_threshold : float
        Threshold for switching from high to low state when input is decreasing
    low_value : float, optional
        Output value in the low state
    high_value : float, optional
        Output value in the high state

    Returns:
    --------
    array-like
        Output signal with hysteresis effect
    """
    if ascending_threshold <= descending_threshold:
        raise ValueError("ascending_threshold must be greater than descending_threshold")

    output = np.zeros_like(x)
    state = False  # Initial state (False = low, True = high)

    for i in range(len(x)):
        if not state and x[i] > ascending_threshold:
            state = True
        elif state and x[i] < descending_threshold:
            state = False

        output[i] = high_value if state else low_value

    return output


def multi_level_hysteresis(
    x: np.ndarray,
    thresholds: list[tuple[float, float]],
    low_values: list[float],
    high_values: list[float],
) -> np.ndarray:
    """
    A hysteresis function model with multiple threshold levels, each with its own low and high output values.

    Parameters:
    -----------
    x : array-like
        Input signal
    thresholds : list of tuples
        List of (ascending_threshold, descending_threshold) pairs for each level
        Must be ordered from lowest to highest threshold level
    low_values : list of float
        Values to output when transitioning from high to low state at each threshold
    high_values : list of float
        Values to output when transitioning from low to high state at each threshold

    Returns:
    --------
    array-like
        Output signal with multi-level hysteresis effect
    """
    if len(thresholds) != len(low_values) or len(thresholds) != len(high_values):
        raise ValueError("thresholds, low_values, and high_values must have the same length")

    # Extract ascending and descending thresholds
    ascending_thresholds = [t[0] for t in thresholds]
    descending_thresholds = [t[1] for t in thresholds]

    # Check that ascending thresholds are in ascending order
    if not all(
        a < b for a, b in zip(ascending_thresholds[:-1], ascending_thresholds[1:], strict=False)
    ):
        raise ValueError("ascending thresholds must be in ascending order")

    # Check that descending thresholds are in ascending order
    if not all(
        a < b for a, b in zip(descending_thresholds[:-1], descending_thresholds[1:], strict=False)
    ):
        raise ValueError("descending thresholds must be in ascending order")

    # Check that each descending threshold is less than its corresponding ascending threshold
    if not all(d < a for d, a in zip(descending_thresholds, ascending_thresholds, strict=False)):
        raise ValueError(
            "Each descending threshold must be less than its corresponding ascending threshold"
        )

    output = np.zeros_like(x)

    # CHANGE: Track active threshold level instead of binary state
    active_level = -1  # Start with no active level

    # For the first point, determine initial level based on value
    for i in range(len(thresholds)):
        if x[0] > ascending_thresholds[i]:
            active_level = i

    # Apply initial output value
    if active_level >= 0:
        output[0] = high_values[active_level]
    else:
        output[0] = low_values[0] if low_values else 0

    # Process the rest of the points
    for i in range(1, len(x)):
        current_value = x[i]
        previous_level = active_level

        # MAJOR CHANGE: Complete rewrite of level transition logic
        # Check if we need to change level
        if active_level >= 0:
            # We're already in a high state at some level
            # Check if we should go up to a higher level
            for j in range(active_level + 1, len(thresholds)):
                if current_value > ascending_thresholds[j]:
                    active_level = j

            # Check if we should go down (potentially multiple levels)
            if current_value < descending_thresholds[active_level]:
                # Find the highest level that's still active
                new_level = -1
                for j in range(active_level):
                    if current_value > descending_thresholds[j]:
                        new_level = j
                active_level = new_level
        else:
            # We're in the lowest state, check if we should go up
            for j in range(len(thresholds)):
                if current_value > ascending_thresholds[j]:
                    active_level = j

        # Set output based on level change
        if active_level != previous_level:
            if active_level >= 0:
                output[i] = high_values[active_level]
            else:
                output[i] = low_values[0] if low_values else 0
        else:
            # No level change, keep previous output
            output[i] = output[i - 1]

    return output


def old_multi_level_hysteresis(
    x: np.ndarray,
    thresholds: list[tuple[float, float]],
    low_values: list[float],
    high_values: list[float],
) -> np.ndarray:
    """
    A hysteresis function model with multiple threshold levels, each with its own low and high output values.

    Parameters:
    -----------
    x : array-like
        Input signal
    thresholds : list of tuples
        List of (ascending_threshold, descending_threshold) pairs for each level
        Must be ordered from lowest to highest threshold level
    low_values : list of float
        Values to output when transitioning from high to low state at each threshold
    high_values : list of float
        Values to output when transitioning from low to high state at each threshold

    Returns:
    --------
    array-like
        Output signal with multi-level hysteresis effect
    """
    if len(thresholds) != len(low_values) or len(thresholds) != len(high_values):
        raise ValueError("thresholds, low_values, and high_values must have the same length")

    # Extract ascending and descending thresholds
    ascending_thresholds = [t[0] for t in thresholds]
    descending_thresholds = [t[1] for t in thresholds]

    # Check that ascending thresholds are in ascending order
    if not all(
        a < b for a, b in zip(ascending_thresholds[:-1], ascending_thresholds[1:], strict=False)
    ):
        raise ValueError("ascending thresholds must be in ascending order")

    # Check that descending thresholds are in ascending order
    if not all(
        a < b for a, b in zip(descending_thresholds[:-1], descending_thresholds[1:], strict=False)
    ):
        raise ValueError("descending thresholds must be in ascending order")

    # Check that each descending threshold is less than its corresponding ascending threshold
    if not all(d < a for d, a in zip(descending_thresholds, ascending_thresholds, strict=False)):
        raise ValueError(
            "Each descending threshold must be less than its corresponding ascending threshold"
        )

    output = np.zeros_like(x)
    state = np.zeros(len(x), dtype=bool)  # Track state for each point (False = low, True = high)

    # For the first point, determine initial state based on value
    # If starting value is already above any ascending threshold, set initial state accordingly
    for i in range(len(thresholds)):
        if x[0] > ascending_thresholds[i]:
            state[0] = True

    # Apply the appropriate initial value
    if state[0]:
        # Find the highest threshold level that the initial value has crossed
        level = max(
            [i for i in range(len(thresholds)) if x[0] > ascending_thresholds[i]],
            default=0,
        )
        output[0] = high_values[level]
    else:
        # Find the highest threshold level where the initial value is above the descending threshold
        level = max(
            [i for i in range(len(thresholds)) if x[0] > descending_thresholds[i]],
            default=-1,
        )
        if level >= 0:
            output[0] = low_values[level]
        else:
            output[0] = low_values[0]  # If below all thresholds, use the lowest value

    logger.debug(f"init: x[0]={x[0]}, state[0]={state[0]} -> {output[0]}")

    # Process the rest of the points
    for i in range(1, len(x)):
        # Start with previous state
        state[i] = state[i - 1]
        logger.debug(f"x[{i}]={x[i]}, init_state[{i}]={state[i]}")

        # Assume No state change, maintain previous output
        output[i] = output[i - 1]

        # Check each threshold level
        for j in range(len(thresholds)):
            if not state[i] and x[i] >= ascending_thresholds[j]:
                # Transition from low to high
                state[i] = True
                output[i] = high_values[j]
                logger.debug(f"(* {j} {ascending_thresholds[j]})")
                break
            elif state[i] and x[i] <= descending_thresholds[j]:
                # Transition from high to low
                state[i] = False
                output[i] = low_values[j]
                logger.debug(f"(x {j} {descending_thresholds[j]})")
                break
        logger.debug(f" new_state[{i}]={state[i]} -> {output[i]}")

    logger.debug(f"multi: , output={type(output)}")
    return output


def relay_hysteresis(
    x: np.ndarray,
    center: float,
    width: float,
    low_value: float = 0,
    high_value: float = 1,
) -> np.ndarray:
    """
    A relay-type hysteresis function centered around a specific value.

    Parameters:
    -----------
    x : array-like
        Input signal
    center : float
        Center point of the hysteresis loop
    width : float
        Width of the hysteresis loop
    low_value : float, optional
        Output value in the low state
    high_value : float, optional
        Output value in the high state

    Returns:
    --------
    array-like
        Output signal with hysteresis effect
    """
    ascending_threshold = center + width / 2
    descending_threshold = center - width / 2

    return hysteresis_model(x, ascending_threshold, descending_threshold, low_value, high_value)


def continuous_hysteresis(
    x: np.ndarray,
    center: float,
    width: float,
    slope: float = 10,
    low_value: float = 0,
    high_value: float = 1,
) -> np.ndarray:
    """
    A continuous hysteresis function with smooth transitions.

    Parameters:
    -----------
    x : array-like
        Input signal
    center : float
        Center point of the hysteresis loop
    width : float
        Width of the hysteresis loop
    slope : float, optional
        Controls the steepness of the transitions
    low_value : float, optional
        Minimum output value
    high_value : float, optional
        Maximum output value

    Returns:
    --------
    array-like
        Output signal with continuous hysteresis effect
    """
    output = np.zeros_like(x)
    state = np.zeros_like(x)

    # Initial state
    state[0] = 0.5

    # Calculate state using a differential equation that exhibits hysteresis
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]

        if dx > 0:  # Ascending
            target = 1 / (1 + np.exp(-slope * (x[i] - (center + width / 2))))
        else:  # Descending or constant
            target = 1 / (1 + np.exp(-slope * (x[i] - (center - width / 2))))

        # Simple relaxation to target
        state[i] = state[i - 1] + 0.1 * (target - state[i - 1])

    # Scale to desired output range
    output = low_value + state * (high_value - low_value)

    return output


# Example usage
if __name__ == "__main__":
    # Generate a sine wave input
    t = np.linspace(0, 4 * np.pi, 1000)
    x = np.sin(t)

    # Model different types of hysteresis
    y1 = hysteresis_model(x, 0.5, -0.5)
    y2 = relay_hysteresis(x, 0, 1)
    y3 = continuous_hysteresis(x, 0, 1, slope=8)

    # Plotting
    plt.figure(figsize=(15, 10))

    # Plot input signal
    plt.subplot(2, 2, 1)
    plt.plot(t, x)
    plt.title("Input Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Plot different hysteresis functions
    plt.subplot(2, 2, 2)
    plt.plot(x, y1)
    plt.title("Discrete Hysteresis")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(x, y2)
    plt.title("Relay Hysteresis")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(x, y3)
    plt.title("Continuous Hysteresis")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot time-domain responses
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(t, x)
    plt.title("Input Signal vs Time")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(t, y1)
    plt.title("Discrete Hysteresis vs Time")
    plt.xlabel("Time")
    plt.ylabel("Output")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(t, y2)
    plt.title("Relay Hysteresis vs Time")
    plt.xlabel("Time")
    plt.ylabel("Output")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(t, y3)
    plt.title("Continuous Hysteresis vs Time")
    plt.xlabel("Time")
    plt.ylabel("Output")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
