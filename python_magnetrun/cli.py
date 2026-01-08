"""Console script for python_magnetrun."""

import logging
import argparse
import sys

logger = logging.getLogger(__name__)


def main():
    """Console script for python_magnetrun."""
    parser = argparse.ArgumentParser()
    parser.add_argument("_", nargs="*")
    args = parser.parse_args()

    print("Arguments: " + str(args._))
    print("Replace this message by putting your code into " "python_magnetrun.cli.main")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
