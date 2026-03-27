"""
RMS File Reader - Usage Examples
=================================

This file demonstrates how to use the RMS file reader for FEPC-AUX-LNCMI files.
"""

import matplotlib.pyplot as plt
import pandas as pd
from rms_reader import RMSFileReader, read_rms_file


# Example 1: Quick read - simplest usage
# =======================================
def example_quick_read(filepath):
    """Quickly read an RMS file into a DataFrame."""
    df = read_rms_file(filepath)
    print(f"Loaded {len(df)} samples with {len(df.columns)} variables")
    print(df.head())
    return df


# Example 2: Detailed information
# ================================
def example_detailed_info(filepath):
    """Get detailed information about the file structure."""
    reader = RMSFileReader(filepath)
    reader.parse_header()

    # Print summary
    reader.print_summary()

    # Get variable information
    var_info = reader.get_variable_info()
    print("\nVariable Details:")
    print(var_info.to_string())

    # Get metadata
    metadata = reader.get_metadata()
    print("\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    return reader


# Example 3: Selective data reading
# ==================================
def example_selective_reading(filepath, variables_of_interest):
    """Read only specific variables from the file."""
    # Read all data
    df = read_rms_file(filepath)

    # Select only the variables of interest
    available_vars = [v for v in variables_of_interest if v in df.columns]
    df_subset = df[available_vars]

    print(f"Selected {len(available_vars)} variables:")
    print(available_vars)

    return df_subset


# Example 4: Time-based filtering
# ================================
def example_time_filtering(filepath, start_time=None, end_time=None):
    """Read data and filter by time range."""
    df = read_rms_file(filepath)

    if start_time:
        df = df[df.index >= start_time]
    if end_time:
        df = df[df.index <= end_time]

    print(f"Filtered data: {len(df)} samples")
    print(f"Time range: {df.index[0]} to {df.index[-1]}")

    return df


# Example 5: Analyzing digital signals
# =====================================
def example_digital_signals(filepath):
    """Extract and analyze digital signals."""
    reader = RMSFileReader(filepath)
    df = reader.read()

    # Get all digital variables
    var_info = reader.get_variable_info()
    digital_vars = var_info[var_info["type"] == "bit"]["name"].tolist()

    print(f"Found {len(digital_vars)} digital signals:")

    # Analyze each digital signal
    for var in digital_vars:
        if var in df.columns:
            transitions = (df[var].diff() != 0).sum()
            on_time = (df[var] == 1).sum() / len(df) * 100
            print(f"  {var}: {transitions} transitions, ON {on_time:.1f}% of time")

    return df[digital_vars]


# Example 6: Analyzing analog signals
# ====================================
def example_analog_signals(filepath):
    """Extract and analyze analog signals with statistics."""
    reader = RMSFileReader(filepath)
    df = reader.read()

    # Get all analog variables
    var_info = reader.get_variable_info()
    analog_vars = var_info[var_info["type"] == "float32"]["name"].tolist()

    print(f"Found {len(analog_vars)} analog signals")

    # Calculate statistics
    df_analog = df[analog_vars]
    stats = df_analog.describe()

    print("\nStatistics:")
    print(stats)

    return df_analog


# Example 7: Plotting data
# =========================
def example_plotting(filepath, variables_to_plot):
    """Plot selected variables over time."""
    df = read_rms_file(filepath)

    # Filter variables that exist
    vars_to_plot = [v for v in variables_to_plot if v in df.columns]

    if not vars_to_plot:
        print("None of the requested variables found in file")
        return

    # Create subplots
    fig, axes = plt.subplots(len(vars_to_plot), 1, figsize=(12, 3 * len(vars_to_plot)))
    if len(vars_to_plot) == 1:
        axes = [axes]

    for ax, var in zip(axes, vars_to_plot, strict=False):
        df[var].plot(ax=ax)
        ax.set_ylabel(var)
        ax.set_xlabel("Time")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("rms_plot.png", dpi=150)
    print("Plot saved to rms_plot.png")

    return fig


# Example 8: Export to different formats
# =======================================
def example_export(filepath):
    """Export RMS data to various formats."""
    df = read_rms_file(filepath)

    # Export to CSV
    csv_path = filepath.replace(".rms", "_data.csv")
    df.to_csv(csv_path)
    print(f"Exported to CSV: {csv_path}")

    # Export to Excel (requires openpyxl)
    try:
        excel_path = filepath.replace(".rms", "_data.xlsx")
        df.to_excel(excel_path)
        print(f"Exported to Excel: {excel_path}")
    except ImportError:
        print("Excel export requires openpyxl: pip install openpyxl")

    # Export to HDF5 (requires tables)
    try:
        h5_path = filepath.replace(".rms", "_data.h5")
        df.to_hdf(h5_path, key="rms_data", mode="w")
        print(f"Exported to HDF5: {h5_path}")
    except ImportError:
        print("HDF5 export requires tables: pip install tables")

    return df


# Example 9: Detect anomalies in temperature signals
# ===================================================
def example_temperature_analysis(filepath):
    """Analyze temperature signals (TT* variables)."""
    df = read_rms_file(filepath)

    # Find all temperature variables (starting with TT)
    temp_vars = [
        col
        for col in df.columns
        if col.startswith("TT") and not col.endswith("_D1") and not col.endswith("_D2")
    ]

    print(f"Found {len(temp_vars)} temperature sensors")

    # Create analysis
    results = []
    for var in temp_vars:
        if var in df.columns:
            data = df[var]
            results.append(
                {
                    "sensor": var,
                    "mean": data.mean(),
                    "std": data.std(),
                    "min": data.min(),
                    "max": data.max(),
                    "range": data.max() - data.min(),
                }
            )

    results_df = pd.DataFrame(results)
    print("\nTemperature Analysis:")
    print(results_df.to_string(index=False))

    return results_df


# Example 10: Batch processing multiple files
# ============================================
def example_batch_processing(file_list):
    """Process multiple RMS files and combine results."""
    all_data = []

    for filepath in file_list:
        print(f"Processing {filepath}...")
        try:
            df = read_rms_file(filepath)
            # Add filename as a column
            df["source_file"] = filepath
            all_data.append(df)
        except (OSError, ValueError, RuntimeError) as e:
            print(f"  Error: {e}")

    if all_data:
        # Combine all data
        combined_df = pd.concat(all_data, axis=0)
        print(f"\nCombined {len(all_data)} files: {len(combined_df)} total samples")
        return combined_df

    return None


# Main demonstration
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("""
RMS File Reader - Usage Examples

Usage: python rms_examples.py <rms_file_path> [example_number]

Examples:
  1 - Quick read
  2 - Detailed information
  3 - Selective reading
  4 - Time filtering
  5 - Digital signals analysis
  6 - Analog signals analysis
  7 - Plotting
  8 - Export to various formats
  9 - Temperature analysis
  10 - Batch processing
        """)
        sys.exit(1)

    filepath = sys.argv[1]
    example_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    print(f"Running example {example_num} with file: {filepath}\n")

    if example_num == 1:
        example_quick_read(filepath)
    elif example_num == 2:
        example_detailed_info(filepath)
    elif example_num == 3:
        # Example: read only pressure and voltage variables
        vars_of_interest = ["PT205", "PH_V11", "PH_V2", "PH_V6"]
        example_selective_reading(filepath, vars_of_interest)
    elif example_num == 4:
        example_time_filtering(filepath)
    elif example_num == 5:
        example_digital_signals(filepath)
    elif example_num == 6:
        example_analog_signals(filepath)
    elif example_num == 7:
        # Example: plot some key variables
        vars_to_plot = ["PT205", "TT200A", "PH_V11"]
        example_plotting(filepath, vars_to_plot)
    elif example_num == 8:
        example_export(filepath)
    elif example_num == 9:
        example_temperature_analysis(filepath)
    elif example_num == 10:
        # For batch processing, pass multiple files
        file_list = sys.argv[1:]
        example_batch_processing(file_list)
