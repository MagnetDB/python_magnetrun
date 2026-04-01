"""
VProcess Batch Processing Utility
==================================

Process multiple VProcess files in batch mode.

Usage:
    python batch.py --dir /path/to/files --output merged.csv
    python batch.py --dir /path/to/files --vars TT115A TT508A --format hdf5
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from vprocess_reader import VProcessFileReader, read_vprocess_file

# Setup logger
logger = logging.getLogger(__name__)


def find_vprocess_files(
    directory: str, pattern: str = "*.vprocess", recursive: bool = False
) -> list[Path]:
    """
    Find all VProcess files in a directory.

    Parameters
    ----------
    directory : str
        Directory to search
    pattern : str
        File pattern to match (default: *.vprocess)
    recursive : bool
        Whether to search recursively

    Returns
    -------
    list of Path
        List of VProcess file paths, sorted by name
    """
    dir_path = Path(directory)

    files = sorted(dir_path.rglob(pattern)) if recursive else sorted(dir_path.glob(pattern))

    return files


def get_common_variables(file_list: list[Path], verbose: bool = True) -> list[str]:
    """
    Get list of variables common to all files.

    Parameters
    ----------
    file_list : list of Path
        List of VProcess files
    verbose : bool
        Show progress

    Returns
    -------
    list of str
        List of common variable names
    """
    if not file_list:
        return []

    # Get variables from first file
    reader = VProcessFileReader(file_list[0])
    reader.parse_header()
    common_vars = set(var.name for var in reader.variables)

    if verbose:
        logger.info(f"First file has {len(common_vars)} variables")

    # Check remaining files
    for filepath in file_list[1:]:
        try:
            reader = VProcessFileReader(filepath)
            reader.parse_header()
            file_vars = set(var.name for var in reader.variables)
            common_vars &= file_vars
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning(f"Could not read {filepath.name}: {str(e)}")

    if verbose:
        logger.info(f"Found {len(common_vars)} common variables across all files")

    return sorted(list(common_vars))


def process_batch(
    file_list: list[Path],
    selected_vars: list[str] | None = None,
    merge: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Process multiple VProcess files.

    Parameters
    ----------
    file_list : list of Path
        List of VProcess files to process
    selected_vars : list of str, optional
        List of variable names to extract (None = all variables)
    merge : bool
        Whether to merge all files into single DataFrame
    verbose : bool
        Show progress bar

    Returns
    -------
    pd.DataFrame
        Merged DataFrame if merge=True, otherwise last file's data
    """
    dataframes = []

    # Process with progress indication
    total = len(file_list)
    for i, filepath in enumerate(file_list, 1):
        try:
            if verbose:
                logger.info(f"[{i}/{total}] Processing {filepath.name}...")

            df = read_vprocess_file(str(filepath))

            # Filter columns if specified
            if selected_vars:
                available_vars = [v for v in selected_vars if v in df.columns]
                if not available_vars:
                    logger.warning(f"None of the selected variables found in {filepath.name}")
                    continue
                df = df[available_vars]

            if merge:
                dataframes.append(df)
            else:
                # Process individually (could save here)
                pass

        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Error processing {filepath.name}: {str(e)}")

    if merge and dataframes:
        # Concatenate all dataframes
        logger.info("Merging dataframes...")
        merged_df = pd.concat(dataframes, axis=0)
        # Sort by timestamp
        merged_df.sort_index(inplace=True)
        # Remove duplicate timestamps
        merged_df = merged_df[~merged_df.index.duplicated(keep="first")]
        return merged_df
    elif dataframes:
        return dataframes[-1]
    else:
        return pd.DataFrame()


def export_data(df: pd.DataFrame, output_path: str, format: str = "csv") -> None:
    """
    Export DataFrame to various formats.

    Parameters
    ----------
    df : pd.DataFrame
        Data to export
    output_path : str
        Output file path
    format : str
        Output format: 'csv', 'hdf5', 'parquet', 'excel'
    """
    output_file = Path(output_path)

    if format == "csv":
        df.to_csv(output_file)
        logger.info(f"Data exported to CSV: {output_file}")

    elif format in ["hdf5", "h5"]:
        df.to_hdf(output_file, key="vprocess", mode="w")
        logger.info(f"Data exported to HDF5: {output_file}")

    elif format == "parquet":
        df.to_parquet(output_file)
        logger.info(f"Data exported to Parquet: {output_file}")

    elif format in ["excel", "xlsx"]:
        df.to_excel(output_file)
        logger.info(f"Data exported to Excel: {output_file}")

    else:
        raise ValueError(f"Unsupported format: {format}")


def analyze_batch(file_list: list[Path], output_file: str | None = None) -> pd.DataFrame:
    """
    Analyze multiple VProcess files and create summary statistics.

    Parameters
    ----------
    file_list : list of Path
        List of VProcess files
    output_file : str, optional
        Path to save analysis results

    Returns
    -------
    pd.DataFrame
        Summary statistics for each file
    """
    results = []

    for filepath in file_list:
        try:
            reader = VProcessFileReader(filepath)
            reader.parse_header()
            metadata = reader.get_metadata()

            df = reader.read()

            result = {
                "filename": filepath.name,
                "num_samples": len(df),
                "num_variables": len(df.columns),
                "start_time": df.index[0] if len(df) > 0 else None,
                "end_time": df.index[-1] if len(df) > 0 else None,
                "duration_seconds": (
                    (df.index[-1] - df.index[0]).total_seconds() if len(df) > 1 else 0
                ),
                "frequency_hz": metadata.get("frequency", None),
                "file_size_mb": filepath.stat().st_size / (1024 * 1024),
            }

            results.append(result)

        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Error analyzing {filepath.name}: {str(e)}")

    summary_df = pd.DataFrame(results)

    if output_file:
        summary_df.to_csv(output_file, index=False)
        logger.info(f"Analysis saved to: {output_file}")

    return summary_df


def main() -> None:
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Batch process VProcess files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all files in directory and export to CSV
  python batch.py --dir ./data --output merged.csv --merge

  # Export specific variables to HDF5
  python batch.py --dir ./data --vars TT115A TT508A --format hdf5 --merge

  # List common variables across all files
  python batch.py --dir ./data --list-common-vars

  # Analyze files and create summary
  python batch.py --dir ./data --analyze --output summary.csv
        """,
    )

    parser.add_argument("--dir", required=True, help="Directory containing VProcess files")
    parser.add_argument(
        "--pattern",
        default="*.vprocess",
        help="File pattern to match (default: *.vprocess)",
    )
    parser.add_argument("--recursive", "-r", action="store_true", help="Search recursively")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument(
        "--format",
        default="csv",
        choices=["csv", "hdf5", "h5", "parquet", "excel", "xlsx"],
        help="Output format (default: csv)",
    )
    parser.add_argument("--vars", nargs="+", help="List of variables to extract")
    parser.add_argument("--merge", action="store_true", help="Merge all files into single output")
    parser.add_argument(
        "--list-common-vars",
        action="store_true",
        help="List variables common to all files",
    )
    parser.add_argument("--analyze", action="store_true", help="Analyze files and create summary")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )

    # Find files
    file_list = find_vprocess_files(args.dir, args.pattern, args.recursive)

    if not file_list:
        logger.error(f"No VProcess files found in {args.dir} matching pattern {args.pattern}")
        return

    logger.info(f"Found {len(file_list)} VProcess files")

    # List common variables if requested
    if args.list_common_vars:
        common_vars = get_common_variables(file_list, verbose=not args.quiet)
        logger.info(f"Common variables ({len(common_vars)}):")
        for var in common_vars:
            logger.info(f"  - {var}")
        return

    # Analyze files if requested
    if args.analyze:
        summary_df = analyze_batch(file_list, output_file=args.output)
        logger.info("File Analysis Summary:")
        logger.info(f"{summary_df.to_string(index=False)}")
        return

    # Process files
    if args.merge or args.output:
        df = process_batch(file_list, selected_vars=args.vars, merge=True, verbose=not args.quiet)

        logger.info(f"\nMerged data shape: {df.shape}")
        if len(df) > 0:
            logger.info(f"Time range: {df.index[0]} to {df.index[-1]}")
            duration = (df.index[-1] - df.index[0]).total_seconds()
            logger.info(f"Duration: {duration:.1f} seconds ({duration / 3600:.2f} hours)")

        # Export if output path specified
        if args.output:
            export_data(df, args.output, args.format)
    else:
        logger.warning("No action specified. Use --output, --merge, --analyze, or --list-common-vars")
        logger.warning("Use --help for more information")


if __name__ == "__main__":
    main()
