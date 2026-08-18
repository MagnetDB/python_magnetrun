"""
RMS File Reader - Usage Examples
=================================

This file demonstrates how to use the RMS file reader for FEPC-AUX-LNCMI files.
"""

import logging
import re
from pathlib import Path

import pandas as pd

from python_magnetrun.hybrid.rms.rms_reader import RMSFileReader, read_rms_file
from python_magnetrun.log_utils import get_logger, setup_logging

logger = get_logger(__name__)


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
    _metadata_units = {"frequency": "Hz"}
    print("\nMetadata:")
    for key, value in metadata.items():
        unit = _metadata_units.get(key, "")
        suffix = f" {unit}" if unit else ""
        print(f"  {key}: {value}{suffix}")

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
    from python_magnetrun.plotting import PlotStyle, get_backend, plot_subplots

    df = read_rms_file(filepath)

    # Filter variables that exist
    vars_to_plot = [v for v in variables_to_plot if v in df.columns]

    if not vars_to_plot:
        print("None of the requested variables found in file")
        return

    # Reset DatetimeIndex into a plain column so plot_subplots can use it
    df_plot = df[vars_to_plot].reset_index(names="t")

    b = get_backend("matplotlib")
    fig = plot_subplots(
        df_plot,
        vars_to_plot,
        t_col="t",
        backend=b,
        style=PlotStyle(figsize=(12, 3)),
    )

    out = Path("rms_plot.png")
    b.save(fig, out, dpi=150)
    print(f"Plot saved to {out}")
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


def infer_system_date_from_filename(name: str) -> tuple[str | None, str | None]:
    """Infer (fepc_system, hybrid_date) from an RMS filename stem.

    Supports both separator styles:
    - Double underscore: FEPC-AUX-LNCMI__2025-11-05_1400--2025-11-05_1500
    - Single underscore: FEPC-LNCMI_2025-01-06_0000—2025-01-06_0100
    """
    m = re.match(r"^(.+?)__(\d{4}-\d{2}-\d{2})_\d{4}", name) or re.match(
        r"^(.+?)_(\d{4}-\d{2}-\d{2})_\d{4}", name
    )
    if m:
        return m.group(1), m.group(2)
    return None, None


def resolve_rms_filepaths(
    input_file: str | None,
    hybrid_datadir: str | None,
    hybrid_date: str | None,
    fepc_system: str,
    default_fepc_system: str,
) -> list[str]:
    """Resolve RMS file paths from a direct path or hybrid directory.

    Resolution order:
    1. If *input_file* is given and exists on disk — use it directly.
    2. If *input_file* is given but not found — infer date/system from the
       filename, then search ``hybrid_datadir/rms/<date>/<system>/``.
    3. If no *input_file* — list all ``.rms`` files in
       ``hybrid_datadir/rms/<hybrid_date>/<fepc_system>/``.

    Parameters
    ----------
    input_file:
        Optional path (or bare filename) of an RMS file.
    hybrid_datadir:
        Base directory for hybrid data (contains ``rms/`` subtree).
    hybrid_date:
        Date string ``YYYY-MM-DD`` used when searching the hybrid tree.
    fepc_system:
        FEPC system name (e.g. ``FEPC-LNCMI``).
    default_fepc_system:
        Parser default for fepc_system; used to detect when the user has
        not explicitly overridden it.

    Returns
    -------
    list[str]
        Resolved absolute file paths, never empty.

    Raises
    ------
    FileNotFoundError
        When the file cannot be found or the directory yields no matches.
    ValueError
        When insufficient information is provided to locate the files.
    """
    if input_file:
        candidate = Path(input_file)
        if candidate.exists():
            return [str(candidate)]

        if not (hybrid_date or hybrid_datadir):
            raise FileNotFoundError(f"File not found: {input_file}")

        # Infer date/system from the filename when not explicitly given
        inferred_system, inferred_date = infer_system_date_from_filename(candidate.stem)
        resolved_date = hybrid_date or inferred_date
        resolved_system = (
            fepc_system
            if fepc_system != default_fepc_system
            else (inferred_system or fepc_system)
        )
        logger.info(
            f"Inferred from filename: system='{resolved_system}', date='{resolved_date}'"
        )
        if not (resolved_date and resolved_system):
            raise ValueError(
                f"'{input_file}' not found; could not infer date/system from filename. "
                "Provide --hybrid_date and/or --fepc_system."
            )

        rms_dir = Path(hybrid_datadir) / "rms" / resolved_date / resolved_system
        matches = sorted(rms_dir.glob(f"*{candidate.name}*")) or sorted(
            rms_dir.glob("*.rms")
        )
        if not matches:
            raise FileNotFoundError(
                f"'{input_file}' not found and no .rms files in {rms_dir}"
            )
        resolved = str(matches[0])
        logger.info(f"Resolved '{input_file}' → {resolved}")
        return [resolved]

    # No input_file — collect all files from the hybrid tree
    if not hybrid_date:
        raise ValueError("Provide either input_file or --hybrid_date")

    rms_dir = Path(hybrid_datadir) / "rms" / hybrid_date / fepc_system
    rms_files = sorted(rms_dir.glob("*.rms"))
    if not rms_files:
        raise FileNotFoundError(f"No .rms files found in {rms_dir}")
    logger.info(f"Found {len(rms_files)} RMS file(s) in {rms_dir}")
    return [str(f) for f in rms_files]


# Main demonstration
if __name__ == "__main__":
    import argparse

    from python_magnetrun.cli_args import create_hybrid_parser, validate_file_extension

    hybrid_parser = create_hybrid_parser()
    parser = argparse.ArgumentParser(
        description="RMS File Reader - Usage Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[hybrid_parser],
        epilog="""
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
        """,
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=validate_file_extension([".rms"]),
        help="RMS file to process (optional if --hybrid_date is given)",
    )
    parser.add_argument(
        "--example",
        type=int,
        choices=range(1, 11),
        default=1,
        metavar="N",
        help="example number to run (1-10, default: 1)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="set logging level",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="path to log file (if not specified, logs to console)",
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.WARNING)
    setup_logging(level=log_level, log_file=args.log_file)
    logger.debug(f"Parsed arguments: {args}")

    def _infer_from_filename(name: str) -> tuple[str | None, str | None]:
        """Infer (fepc_system, hybrid_date) from an RMS filename stem.

        Expected format: {system}_YYYY-MM-DD_HHMM—YYYY-MM-DD_HHMM
        Example: FEPC-LNCMI_2025-01-06_0000—2025-01-06_0100
        """
        m = re.match(r"^(.+?)__(\d{4}-\d{2}-\d{2})_\d{4}", name) or re.match(
            r"^(.+?)_(\d{4}-\d{2}-\d{2})_\d{4}", name
        )
        if m:
            return m.group(1), m.group(2)
        return None, None

    filepaths: list[str] = []

    if args.input_file:
        candidate = Path(args.input_file)
        if candidate.exists():
            filepaths = [str(candidate)]
        elif args.hybrid_date or args.hybrid_datadir:
            # File not found at given path — infer date/system from the name and search
            inferred_system, inferred_date = _infer_from_filename(candidate.stem)
            hybrid_date = args.hybrid_date or inferred_date
            fepc_system = (
                args.fepc_system
                if args.fepc_system != parser.get_default("fepc_system")
                else (inferred_system or args.fepc_system)
            )
            logger.info(
                f"Inferred from filename: system='{fepc_system}', date='{hybrid_date}'"
            )
            if hybrid_date and fepc_system:
                rms_dir = Path(args.hybrid_datadir) / "rms" / hybrid_date / fepc_system
                matches = sorted(rms_dir.glob(f"*{candidate.name}*"))
                if not matches:
                    matches = sorted(rms_dir.glob("*.rms"))
                if matches:
                    filepaths = [str(matches[0])]
                    logger.info(f"Resolved '{args.input_file}' → {filepaths[0]}")
                else:
                    parser.error(
                        f"'{args.input_file}' not found and no .rms files in {rms_dir}"
                    )
            else:
                parser.error(
                    f"'{args.input_file}' not found; could not infer date/system from filename. "
                    "Provide --hybrid_date and/or --fepc_system."
                )
        else:
            parser.error(f"File not found: {args.input_file}")

    if not filepaths:
        if args.hybrid_date:
            rms_dir = (
                Path(args.hybrid_datadir) / "rms" / args.hybrid_date / args.fepc_system
            )
            rms_files = sorted(rms_dir.glob("*.rms"))
            if not rms_files:
                parser.error(f"No .rms files found in {rms_dir}")
            filepaths = [str(f) for f in rms_files]
            logger.info(f"Found {len(filepaths)} RMS file(s) in {rms_dir}")
        else:
            parser.error("Provide either input_file or --hybrid_date")

    example_num = args.example

    logger.info(f"Running example {example_num} with {len(filepaths)} file(s)")

    if example_num == 1:
        example_quick_read(filepaths[0])
    elif example_num == 2:
        example_detailed_info(filepaths[0])
    elif example_num == 3:
        # Example: read only pressure and voltage variables
        vars_of_interest = ["PT205", "PH_V11", "PH_V2", "PH_V6"]
        example_selective_reading(filepaths[0], vars_of_interest)
    elif example_num == 4:
        example_time_filtering(filepaths[0])
    elif example_num == 5:
        example_digital_signals(filepaths[0])
    elif example_num == 6:
        example_analog_signals(filepaths[0])
    elif example_num == 7:
        # Example: plot some key variables
        vars_to_plot = ["PT205", "TT200A", "PH_V11"]
        example_plotting(filepaths[0], vars_to_plot)
    elif example_num == 8:
        example_export(filepaths[0])
    elif example_num == 9:
        example_temperature_analysis(filepaths[0])
    elif example_num == 10:
        example_batch_processing(filepaths)
