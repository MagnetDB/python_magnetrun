"""Unified ``magnetrun`` entry point."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="magnetrun",
        description="View, analyse, and process magnet run data.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="subcommand")

    # data commands (operate on run files)
    from .commands.add import register as _r_add
    from .commands.info import register as _r_info
    from .commands.plot import register as _r_plot
    from .commands.select import register as _r_select
    from .commands.signature import register as _r_sig
    from .commands.stats import register as _r_stats

    _r_info(sub)
    _r_add(sub)
    _r_plot(sub)
    _r_select(sub)
    _r_stats(sub)
    _r_sig(sub)

    # workflow commands
    from .analysis.cli import register as _r_ana
    from .comparison.cli import register as _r_cmp
    from .hybrid.cli import register as _r_hyb
    from .processing.cli import register as _r_proc
    from .runlogs.pigbrother import register as _r_log

    _r_ana(sub)
    _r_proc(sub)
    _r_hyb(sub)
    _r_log(sub)
    _r_cmp(sub)

    # infrastructure commands (were separate executables)
    from .config_cli import register as _r_cfg
    from .requests.cli import register as _r_fetch

    _r_fetch(sub)
    _r_cfg(sub)

    args = parser.parse_args()
    sys.exit(args._handler(args))


if __name__ == "__main__":
    main()
