"""CLI stub for ``magnetrun compare``.

This module will host cross-domain comparison logic (pupitre vs hybrid vs
simulation).  The subcommand is registered now so ``magnetrun compare`` is
discoverable; the handler will be filled in during future work.
"""

from __future__ import annotations

import argparse
import sys


def _run_compare(args: argparse.Namespace) -> int:
    raise NotImplementedError(
        "'magnetrun compare' is not yet implemented. "
        "See python_magnetrun/comparison/cli.py."
    )


def register(sub: argparse._SubParsersAction) -> None:
    """Register the ``compare`` subcommand on *sub*."""
    p = sub.add_parser(
        "compare",
        help="compare pupitre, hybrid, and simulation run data [not yet implemented]",
    )
    p.set_defaults(_handler=_run_compare)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="magnetrun-compare",
        description="Compare pupitre, hybrid, and simulation run data.",
    )
    args = parser.parse_args()
    sys.exit(_run_compare(args))


if __name__ == "__main__":
    main()
