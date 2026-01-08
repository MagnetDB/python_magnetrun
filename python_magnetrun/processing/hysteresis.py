import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def remove_outliers(
    df, x_col="x", y_col="y", method="iqr", threshold=1.5, verbose=False
):
    """
    Remove outliers from x,y data using various methods.

    Parameters:
    -----------
    df : pandas DataFrame
        Data with x and y columns
    x_col : str
        Name of x column
    y_col : str
        Name of y column
    method : str
        'iqr' - Interquartile range (default, robust, recommended)
        'zscore' - Z-score method (statistical)
        'isolation_forest' - Isolation Forest (handles multivariate outliers)
        'mad' - Median Absolute Deviation (very robust)
    threshold : float
        For 'iqr'/'mad': multiplier on IQR/MAD (default 1.5, try 1.0-2.0)
        For 'zscore': z-score cutoff (default 3, standard is 2-3)
    verbose : bool
        Print outlier information

    Returns:
    --------
    DataFrame with outliers removed
    """
    df_clean = df.copy()
    x = df_clean[x_col].values
    y = df_clean[y_col].values

    if method == "iqr":
        # Remove outliers in x using IQR method
        Q1_x = np.percentile(x, 25)
        Q3_x = np.percentile(x, 75)
        IQR_x = Q3_x - Q1_x
        lower_bound_x = Q1_x - threshold * IQR_x
        upper_bound_x = Q3_x + threshold * IQR_x

        # Remove outliers in y using IQR method
        Q1_y = np.percentile(y, 25)
        Q3_y = np.percentile(y, 75)
        IQR_y = Q3_y - Q1_y
        lower_bound_y = Q1_y - threshold * IQR_y
        upper_bound_y = Q3_y + threshold * IQR_y

        mask = (
            (x >= lower_bound_x)
            & (x <= upper_bound_x)
            & (y >= lower_bound_y)
            & (y <= upper_bound_y)
        )

        if verbose:
            n_removed = len(df_clean) - np.sum(mask)
            print(f"IQR method (threshold={threshold}): Removed {n_removed} outliers")
            print(f"  x range: [{lower_bound_x:.4f}, {upper_bound_x:.4f}]")
            print(f"  y range: [{lower_bound_y:.4f}, {upper_bound_y:.4f}]")

    elif method == "zscore":
        # Remove outliers using Z-score
        z_x = np.abs((x - np.mean(x)) / np.std(x))
        z_y = np.abs((y - np.mean(y)) / np.std(y))
        mask = (z_x < threshold) & (z_y < threshold)

        if verbose:
            n_removed = len(df_clean) - np.sum(mask)
            print(
                f"Z-score method (threshold={threshold}): Removed {n_removed} outliers"
            )

    elif method == "mad":
        # Median Absolute Deviation (robust to extreme outliers)
        median_x = np.median(x)
        mad_x = np.median(np.abs(x - median_x))

        median_y = np.median(y)
        mad_y = np.median(np.abs(y - median_y))

        lower_bound_x = median_x - threshold * mad_x
        upper_bound_x = median_x + threshold * mad_x
        lower_bound_y = median_y - threshold * mad_y
        upper_bound_y = median_y + threshold * mad_y

        mask = (
            (x >= lower_bound_x)
            & (x <= upper_bound_x)
            & (y >= lower_bound_y)
            & (y <= upper_bound_y)
        )

        if verbose:
            n_removed = len(df_clean) - np.sum(mask)
            print(f"MAD method (threshold={threshold}): Removed {n_removed} outliers")
            print(f"  x range: [{lower_bound_x:.4f}, {upper_bound_x:.4f}]")
            print(f"  y range: [{lower_bound_y:.4f}, {upper_bound_y:.4f}]")

    elif method == "isolation_forest":
        try:
            from sklearn.ensemble import IsolationForest

            iso_forest = IsolationForest(
                contamination=threshold,  # Fraction of outliers expected (0-1)
                random_state=42,
            )
            features = np.column_stack([x, y])
            outlier_labels = iso_forest.fit_predict(features)
            mask = outlier_labels == 1  # 1 = inlier, -1 = outlier

            if verbose:
                n_removed = len(df_clean) - np.sum(mask)
                print(
                    f"Isolation Forest (contamination={threshold}): Removed {n_removed} outliers"
                )
        except ImportError:
            print("Warning: sklearn not available. Falling back to IQR method.")
            return remove_outliers(
                df, x_col, y_col, method="iqr", threshold=1.5, verbose=verbose
            )

    else:
        raise ValueError(f"Unknown method: {method}")

    return df_clean[mask].reset_index(drop=True)


def remove_outliers_by_x_range(df, x_col="x", x_min=None, x_max=None, verbose=False):
    """
    Remove outliers by specifying acceptable x range (simple domain knowledge approach).

    Parameters:
    -----------
    df : pandas DataFrame
        Data
    x_col : str
        Name of x column
    x_min : float, optional
        Minimum acceptable x value
    x_max : float, optional
        Maximum acceptable x value
    verbose : bool
        Print removal info

    Returns:
    --------
    DataFrame with out-of-range x values removed
    """
    df_clean = df.copy()
    mask = np.ones(len(df_clean), dtype=bool)

    if x_min is not None:
        mask = mask & (df_clean[x_col] >= x_min)
    if x_max is not None:
        mask = mask & (df_clean[x_col] <= x_max)

    if verbose:
        n_removed = len(df_clean) - np.sum(mask)
        print(f"X-range method: Removed {n_removed} points")
        print(f"  Kept x in range: [{x_min}, {x_max}]")

    return df_clean[mask].reset_index(drop=True)


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


def remove_x_region_outliers(
    df,
    x_col="x",
    y_col="y",
    x_threshold=None,
    method="iqr",
    threshold=1.5,
    verbose=False,
):
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
    verbose : bool
        Print info

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

    if verbose:
        print(f"X-region: x <= {x_threshold:.4f} ({n_region} points)")

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

            if verbose:
                n_removed = np.sum(region_outliers)
                print(
                    f"  IQR on y in region: y in [{lower_bound:.4f}, {upper_bound:.4f}]"
                )
                print(f"  Removed {n_removed} outliers")

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

                if verbose:
                    n_removed = np.sum(region_outliers)
                    print(f"  Z-score on y in region (threshold={threshold})")
                    print(f"  Removed {n_removed} outliers")

    else:
        raise ValueError(f"Unknown method: {method}")

    return df_clean[inlier_mask].reset_index(drop=True)


def estimate_hysteresis_parameters(
    df, x_col="x", y_col="y", n_levels=None, verbose=False
):
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
    verbose : bool
        If True, print diagnostic information

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
            if verbose:
                print(f"Clustered y into {n_levels} levels")
        except ImportError:
            print(
                "Warning: sklearn not available. Using data-driven level detection instead."
            )

    # Compute derivatives to identify direction of x change
    dx = np.diff(x, prepend=np.nan)
    dy = np.diff(y, prepend=np.nan)

    # Mark direction of x change
    is_increasing = dx[1:] > 1e-10
    is_decreasing = dx[1:] < -1e-10

    # Identify transitions (where y changes significantly)
    transitions = np.where(np.abs(dy[1:]) > 1e-10)[0] + 1

    # Auto-detect distinct output levels
    unique_y = sorted(
        set(np.round(y, decimals=8))
    )  # Round to handle floating point errors
    n_levels_found = len(unique_y)

    if verbose:
        print(f"Found {n_levels_found} distinct output levels")
        print(f"Found {len(transitions)} transitions")

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

        if verbose:
            asc_str = f"{asc_threshold:.4f}" if asc_threshold is not None else "N/A"
            desc_str = f"{desc_threshold:.4f}" if desc_threshold is not None else "N/A"
            print(
                f"Level {i} (y={unique_y[i]:g}): asc_threshold={asc_str}, desc_threshold={desc_str}"
            )

    # Build thresholds list - pair ascending/descending for each level
    # Filter out levels with missing thresholds
    thresholds = []
    valid_indices = []
    for i, (asc, desc) in enumerate(zip(ascending_thresholds, descending_thresholds)):
        if asc is not None and desc is not None:
            thresholds.append((asc, desc))
            valid_indices.append(i)

    # Extract corresponding output values for valid indices (convert to pure Python floats)
    high_values = [float(unique_y[i]) for i in valid_indices]
    low_values = [float(unique_y[max(0, i - 1)]) for i in valid_indices]

    # Sort all lists together by low_values to ensure ascending order
    sorted_items = sorted(zip(low_values, high_values, thresholds))
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
        if state == 0:
            if x_base[i] > asc_threshold_1:
                state = 1
        elif state == 1:
            if x_base[i] < desc_threshold_1:
                state = 0
            elif x_base[i] > asc_threshold_2:
                state = 2
        elif state == 2:
            if x_base[i] < desc_threshold_2:
                state = 1

        y[i] = y_levels[state]

    # Add small noise
    y_noisy = y + np.random.normal(0, 0.05, len(y))

    # Create DataFrame
    df = pd.DataFrame({"x": x_base, "y": y_noisy})

    print("=== Original Data ===")
    print(f"Number of points: {len(df)}")
    print(f"x range: [{df['x'].min():.4f}, {df['x'].max():.4f}]")
    print(f"y range: [{df['y'].min():.4f}, {df['y'].max():.4f}]")

    # Option 1: Remove outliers specifically at LOW x values (bottom 25%)
    print("\n=== Removing Low-X Outliers (bottom 25% of x) ===")
    df_clean = remove_low_x_outliers(df, method="iqr", threshold=1.5, verbose=True)
    print(f"Points after cleaning: {len(df_clean)}")

    # Option 2: Remove outliers in a specific x region (e.g., x < -0.8)
    # print("\n=== Removing Outliers in Specific X Region ===")
    # df_clean = remove_x_region_outliers(df, x_threshold=-0.8, method='iqr', threshold=1.5, verbose=True)

    # Option 3: Remove outliers using standard IQR (both dimensions)
    # df_clean = remove_outliers(df, method='iqr', threshold=1.5, verbose=True)

    # Option 4: Remove outliers by x range if you know acceptable bounds
    # df_clean = remove_outliers_by_x_range(df, x_min=-1.0, x_max=1.0, verbose=True)

    # Estimate parameters - use clustering to handle remaining noise
    print("\n=== Parameter Estimation (with clustering) ===")
    result = estimate_hysteresis_parameters(df_clean, n_levels=3, verbose=True)

    print("\n=== Estimated Parameters ===")
    if result["thresholds"]:
        print(f"Thresholds (list of tuples): {result['thresholds']}")
        print(f"Type of first threshold: {type(result['thresholds'][0])}")
        print(f"Type of first element in threshold: {type(result['thresholds'][0][0])}")

        print(f"\nLow values (sorted): {result['low_values']}")
        print(f"Type: {type(result['low_values'][0])}")

        print(f"\nHigh values (sorted): {result['high_values']}")
        print(f"Type: {type(result['high_values'][0])}")
    else:
        print("No valid thresholds found with current data.")

    print(f"\nDiagnostics: {result['diagnostics']}")


def refine_thresholds_with_hysteresis_loop(df, x_col="x", y_col="y"):
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
            if np.sum(np.diff(x[t1 : t2 + 1]) > 0) > np.sum(
                np.diff(x[t1 : t2 + 1]) < 0
            ):
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
    x, ascending_threshold, descending_threshold, low_value=0, high_value=1
):
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
        raise ValueError(
            "ascending_threshold must be greater than descending_threshold"
        )

    output = np.zeros_like(x)
    state = False  # Initial state (False = low, True = high)

    for i in range(len(x)):
        if not state and x[i] > ascending_threshold:
            state = True
        elif state and x[i] < descending_threshold:
            state = False

        output[i] = high_value if state else low_value

    return output


def multi_level_hysteresis(x, thresholds, low_values, high_values):
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
        raise ValueError(
            "thresholds, low_values, and high_values must have the same length"
        )

    # Extract ascending and descending thresholds
    ascending_thresholds = [t[0] for t in thresholds]
    descending_thresholds = [t[1] for t in thresholds]

    # Check that ascending thresholds are in ascending order
    if not all(
        a < b for a, b in zip(ascending_thresholds[:-1], ascending_thresholds[1:])
    ):
        raise ValueError("ascending thresholds must be in ascending order")

    # Check that descending thresholds are in ascending order
    if not all(
        a < b for a, b in zip(descending_thresholds[:-1], descending_thresholds[1:])
    ):
        raise ValueError("descending thresholds must be in ascending order")

    # Check that each descending threshold is less than its corresponding ascending threshold
    if not all(d < a for d, a in zip(descending_thresholds, ascending_thresholds)):
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


def old_multi_level_hysteresis(x, thresholds, low_values, high_values):
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
        raise ValueError(
            "thresholds, low_values, and high_values must have the same length"
        )

    # Extract ascending and descending thresholds
    ascending_thresholds = [t[0] for t in thresholds]
    descending_thresholds = [t[1] for t in thresholds]

    # Check that ascending thresholds are in ascending order
    if not all(
        a < b for a, b in zip(ascending_thresholds[:-1], ascending_thresholds[1:])
    ):
        raise ValueError("ascending thresholds must be in ascending order")

    # Check that descending thresholds are in ascending order
    if not all(
        a < b for a, b in zip(descending_thresholds[:-1], descending_thresholds[1:])
    ):
        raise ValueError("descending thresholds must be in ascending order")

    # Check that each descending threshold is less than its corresponding ascending threshold
    if not all(d < a for d, a in zip(descending_thresholds, ascending_thresholds)):
        raise ValueError(
            "Each descending threshold must be less than its corresponding ascending threshold"
        )

    output = np.zeros_like(x)
    state = np.zeros(
        len(x), dtype=bool
    )  # Track state for each point (False = low, True = high)

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

    print(f"init: x[0]={x[0]}, state[0]={state[0]} -> {output[0]}")

    # Process the rest of the points
    for i in range(1, len(x)):
        # Start with previous state
        state[i] = state[i - 1]
        print(f"x[{i}]={x[i]}, init_state[{i}]={state[i]}", end="", flush=True)

        # Assume No state change, maintain previous output
        output[i] = output[i - 1]

        # Check each threshold level
        for j in range(len(thresholds)):
            if not state[i] and x[i] >= ascending_thresholds[j]:
                # Transition from low to high
                state[i] = True
                output[i] = high_values[j]
                print("(* ", j, ascending_thresholds[j], end=")", flush=True)
                break
            elif state[i] and x[i] <= descending_thresholds[j]:
                # Transition from high to low
                state[i] = False
                output[i] = low_values[j]
                print("(x ", j, descending_thresholds[j], end=")", flush=True)
                break
        print(f" new_state[{i}]={state[i]} -> {output[i]}", flush=True)

    print(f"multi: , output={type(output)}")
    return output


def relay_hysteresis(x, center, width, low_value=0, high_value=1):
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

    return hysteresis_model(
        x, ascending_threshold, descending_threshold, low_value, high_value
    )


def continuous_hysteresis(x, center, width, slope=10, low_value=0, high_value=1):
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
