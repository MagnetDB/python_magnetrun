"""etl_cli — manage ``<assembly>.<housing version>.pupitre-etl.json`` spec files.

Registered as the ``etl`` domain of ``magnetrun-config``:

- ``etl FILE show``: print the spec in *FILE*;
- ``etl FILE create --from-housing M9 [--assembly NAME] [--at TIME]``: write
  the housing-only spec of a housing, in the wiring version in force at
  *TIME* (default: now).

Magnet parts are added from DuckDB (``to_duckdb``), not here.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .housing_config import get_housing_config
from .runetl import (
    PupitreEtlSpec,
    load_pupitre_etl_spec,
    save_pupitre_etl_spec,
)


def _parse_at(text: str | None) -> datetime:
    """Parse ``--at`` (ISO 8601; no offset = Europe/Paris time); ``None`` = now."""
    if text is None:
        return datetime.now(UTC)
    when = datetime.fromisoformat(text)
    if when.utcoffset() is None:
        when = when.replace(tzinfo=ZoneInfo("Europe/Paris"))
    return when


def show_pupitre_etl_spec(spec: PupitreEtlSpec) -> None:
    """Print a summary of *spec* to stdout.

    Parameters
    ----------
    spec : PupitreEtlSpec
        Spec to summarise.
    """
    housing = spec.housing
    print(f"Assembly : {spec.assembly or '-'}")
    print(f"Housing  : {housing['name']}  version {housing.get('version') or '(base)'}")
    print(f"  remove : {spec.remove}")
    print(f"  rename : {spec.rename}")
    for fd in spec.add.values():
        print(f"  add    : {fd['formula']}")
    print(f"  companions: {housing.get('companions', {})}")
    for magnet, wiring in spec.magnets.items():
        print(f"Magnet {magnet}: {wiring}")
    for rule in spec.rules:
        unit = f" {rule['unit']}" if rule.get("unit") else ""
        tag = f"  -> {rule['tag']}" if rule.get("tag") else ""
        print(
            f"  rule   : {rule['name']}: {rule['stat']}({rule['expr']}) <= {rule['tol']}{unit}"
            f"  [{rule.get('scope') or '-'}]{tag}"
        )


def _populate_etl_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "json_file",
        help="Path to the <assembly>.<housing version>.pupitre-etl.json file",
    )
    sub = parser.add_subparsers(dest="etl_command", required=True)

    sub.add_parser("show", help="Print the spec in the file")

    p_create = sub.add_parser("create", help="Create a housing-only spec file")
    p_create.add_argument(
        "--from-housing", required=True, metavar="HOUSING", help="Housing name (e.g. M9)"
    )
    p_create.add_argument("--assembly", default=None, help="Assembly name to record")
    p_create.add_argument(
        "--at",
        default=None,
        metavar="TIME",
        help="ISO 8601 time selecting the housing version (no offset = "
        "Europe/Paris time; default: now)",
    )


def _dispatch_etl(args: argparse.Namespace) -> int:
    if args.etl_command == "show":
        show_pupitre_etl_spec(load_pupitre_etl_spec(args.json_file))

    elif args.etl_command == "create":
        cfg = get_housing_config(args.from_housing, at=_parse_at(args.at))
        spec = PupitreEtlSpec.from_housing(cfg, assembly=args.assembly)
        dest = Path(args.json_file)
        save_pupitre_etl_spec(dest, spec)
        print(f"Created {dest}")
        show_pupitre_etl_spec(spec)

    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'etl' domain into a parent subparsers group."""
    p = sub.add_parser("etl", help="Manage <assembly>.<version>.pupitre-etl.json files")
    _populate_etl_parser(p)
    p.set_defaults(_domain_handler=_dispatch_etl)
