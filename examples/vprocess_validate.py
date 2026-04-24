"""
VProcess File Validation Utility
=================================

Validate VProcess files and inspect their contents.

Usage:
    python vprocess_validate.py <filepath>
    python vprocess_validate.py <filepath> --check-data
    python vprocess_validate.py <filepath> --quiet
"""

import argparse
import sys

from python_magnetrun.hybrid.vprocess.validate import validate_vprocess_file
from python_magnetrun.log_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and inspect VProcess files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vprocess_validate.py data.vprocess
  python vprocess_validate.py data.vprocess --check-data
  python vprocess_validate.py data.vprocess --check-data --quiet
        """,
    )

    parser.add_argument("filepath", help="Path to VProcess file")
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Perform detailed data validation (slower)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress detailed output"
    )

    args = parser.parse_args()

    if not args.quiet:
        setup_logging()

    results = validate_vprocess_file(
        args.filepath, check_data=args.check_data, verbose=not args.quiet
    )

    sys.exit(0 if results["valid"] else 1)


if __name__ == "__main__":
    main()
