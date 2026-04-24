"""Unified CLI for managing magnetrun configuration files.

Entry point: ``magnetrun-config``

Domains
-------
plot
    Manage plot style/color configuration files (``magnetrun-plot-config``).
housing
    Manage ``<Housing>-housing-config.json`` files (``magnetrun-housing-config``).
field
    Manage ``*-defs.json`` field-definition files (``magnetrun-field-defs``).
"""

from __future__ import annotations

import argparse
import sys

from .field_defs import register as _register_field
from .housing_config import register as _register_housing
from .plotting.cli import register as _register_plot


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="magnetrun-config",
        description="Manage magnetrun configuration files.",
    )
    sub = parser.add_subparsers(dest="domain", required=True)
    _register_plot(sub)
    _register_housing(sub)
    _register_field(sub)

    args = parser.parse_args()
    sys.exit(args._domain_handler(args))


if __name__ == "__main__":
    main()
