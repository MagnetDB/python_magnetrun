"""Argument parser definitions for the vprocess CLI."""

import argparse


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the vprocess CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all subcommands
    """
    parser = argparse.ArgumentParser(
        description="VProcess Data Tools - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Info command
    info_parser = subparsers.add_parser("info", help="Display file information")
    info_parser.add_argument("file", help="VProcess file")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate file")
    validate_parser.add_argument("file", help="VProcess file")
    validate_parser.add_argument("--check-data", action="store_true", help="Check data integrity")

    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Plot data")
    plot_parser.add_argument("file", help="VProcess file")
    plot_parser.add_argument("--vars", nargs="+", help="Variables to plot")
    plot_parser.add_argument("--overview", action="store_true", help="Plot overview")
    plot_parser.add_argument(
        "--compare", nargs=2, metavar=("VAR1", "VAR2"), help="Compare variables"
    )
    plot_parser.add_argument("--heatmap", action="store_true", help="Correlation heatmap")
    plot_parser.add_argument("--max-vars", type=int, default=10, help="Max variables (default: 10)")
    plot_parser.add_argument(
        "--layout",
        choices=["subplots", "overlay"],
        default="subplots",
        help="Plot layout",
    )
    plot_parser.add_argument("--save", "-s", help="Save to file")
    plot_parser.add_argument("--no-show", action="store_true", help="Don't display plot")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch process files")
    batch_parser.add_argument("--dir", required=True, help="Directory with files")
    batch_parser.add_argument("--pattern", default="*.vprocess", help="File pattern")
    batch_parser.add_argument("-r", "--recursive", action="store_true", help="Search recursively")
    batch_parser.add_argument("--vars", nargs="+", help="Variables to extract")
    batch_parser.add_argument("--merge", action="store_true", help="Merge files")
    batch_parser.add_argument("--output", "-o", help="Output file")
    batch_parser.add_argument(
        "--format",
        default="csv",
        choices=["csv", "hdf5", "parquet", "excel"],
        help="Output format",
    )
    batch_parser.add_argument(
        "--list-common-vars", action="store_true", help="List common variables"
    )
    batch_parser.add_argument("--analyze", action="store_true", help="Analyze files")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--file", help="Test specific file")
    test_parser.add_argument("--create-mock", action="store_true", help="Create mock file")
    test_parser.add_argument("--output", default="mock_data.vprocess", help="Mock file output")
    test_parser.add_argument("--samples", type=int, default=3600, help="Number of samples")
    test_parser.add_argument("--analog", type=int, default=10, help="Analog variables")
    test_parser.add_argument("--digital", type=int, default=2, help="Digital variables")

    return parser
