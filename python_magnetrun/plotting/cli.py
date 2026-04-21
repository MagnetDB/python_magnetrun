"""CLI for managing plot style/color configuration files.

Entry point: ``magnetrun-plot-config``

Subcommands
-----------
init
    Write a default :class:`PlotConfig` JSON to a file (default:
    ``plot_config.json`` in the current directory).
show
    Pretty-print the resolved config from a JSON file (or defaults).
validate
    Validate a JSON file and report any unknown/invalid fields.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import fields
from pathlib import Path

from .style import (
    BUNDLED_CONFIG_PATH,
    DEFAULT_COLORS,
    DEFAULT_STYLE,
    PlotConfig,
    load_plot_config,
)

_DEFAULT_OUTPUT = Path("plot_config.json")


def _cmd_init(args: argparse.Namespace) -> int:
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"error: {out} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUNDLED_CONFIG_PATH, out)
    print(f"wrote default plot config to {out}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    if args.file:
        try:
            cfg = load_plot_config(args.file)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        cfg = PlotConfig()
    print(json.dumps(cfg.to_dict(), indent=2))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.file)
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error reading {path}: {exc}", file=sys.stderr)
        return 1

    errors: list[str] = []

    valid_style_keys = {f.name for f in fields(DEFAULT_STYLE)}
    valid_color_keys = {f.name for f in fields(DEFAULT_COLORS)}

    style_data = data.get("style", {})
    colors_data = data.get("colors", {})

    for key in style_data:
        if key not in valid_style_keys:
            errors.append(f"style: unknown field '{key}'")
    for key in colors_data:
        if key not in valid_color_keys:
            errors.append(f"colors: unknown field '{key}'")

    # Try a full round-trip to catch type errors
    try:
        load_plot_config(path)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"parse error: {exc}")

    if errors:
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        print(f"\n{path}: invalid ({len(errors)} error(s))", file=sys.stderr)
        return 1

    print(f"{path}: OK")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="magnetrun-plot-config",
        description="Manage magnetrun plot style/color configuration files.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="write a default config JSON file")
    p_init.add_argument(
        "--output",
        "-o",
        default=str(_DEFAULT_OUTPUT),
        metavar="FILE",
        help=f"output path (default: {_DEFAULT_OUTPUT})",
    )
    p_init.add_argument(
        "--force", "-f", action="store_true", help="overwrite existing file"
    )

    # show
    p_show = sub.add_parser("show", help="print resolved config as JSON")
    p_show.add_argument(
        "file",
        nargs="?",
        default=None,
        metavar="FILE",
        help="config JSON file (omit to show built-in defaults)",
    )

    # validate
    p_val = sub.add_parser("validate", help="validate a config JSON file")
    p_val.add_argument("file", metavar="FILE", help="config JSON file to validate")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "init": _cmd_init,
        "show": _cmd_show,
        "validate": _cmd_validate,
    }
    sys.exit(dispatch[args.command](args))


if __name__ == "__main__":
    main()
