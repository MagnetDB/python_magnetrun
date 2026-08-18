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

from .configAlims.convertxml import register as _register_alim
from .field_defs import register as _register_field
from .housing_config import register as _register_housing
from .plotting.cli import register as _register_plot


def _run_config(args: argparse.Namespace) -> int:
    """Dispatcher-compatible entry: delegates to the domain handler."""
    return args._domain_handler(args)


def register(sub: argparse._SubParsersAction) -> None:
    """Register the ``config`` subcommand on *sub*."""
    p = sub.add_parser(
        "config",
        help="manage housing and insert configuration",
    )
    domain_sub = p.add_subparsers(dest="domain", required=True)
    _register_plot(domain_sub)
    _register_housing(domain_sub)
    _register_field(domain_sub)
    _register_alim(domain_sub)
    p.set_defaults(_handler=_run_config)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="magnetrun-config",
        description="Manage magnetrun configuration files.",
    )
    sub = parser.add_subparsers(dest="domain", required=True)
    _register_plot(sub)
    _register_housing(sub)
    _register_field(sub)
    _register_alim(sub)

    args = parser.parse_args()
    sys.exit(args._domain_handler(args))


if __name__ == "__main__":
    main()
