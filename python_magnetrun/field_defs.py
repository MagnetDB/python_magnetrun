"""field_defs — utilities for managing ``*-defs.json`` field-definition files.

These files map column/channel names to physical metadata::

    {
        "Field":  {"symbol": "B",  "unit": "tesla",   "description": "Magnetic field"},
        "Icoil1": {"symbol": "I",  "unit": "ampere",  "description": ""},
        "timestamp": {"symbol": "time", "unit": null,  "description": ""}
    }

Cross-format aliases (housing-independent name correspondences) are stored
under the ``"aliases"`` key of each entry::

    "Idcct1": {
        "symbol": "I", "unit": "ampere", "description": "",
        "aliases": {
            "pigbrother": "Courants_Alimentations/Courant_A1",
            "hybrid":     "FEPC-AUX-LNCMI/ALIM1_J1"
        }
    }

Housing-dependent role assignments (e.g. GR1 current = IH for M9 but IB for M8)
are NOT stored here — they belong in ``housing_config.py``.

All public functions are pure file-IO helpers; they have no dependency on any
MagnetData class so they can be used from CLIs, notebooks, or any data class.
"""

from __future__ import annotations

import argparse
import functools
import importlib.resources
import json
import logging
from pathlib import Path

from .magnetdata_base import DataType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Package-data helpers
# ---------------------------------------------------------------------------

_USER_CONFIG_DIR = Path.home() / ".config" / "magnetrun"


def _bundled_defs_path(filename: str) -> Path:
    """Return the path to a ``*-defs.json`` file bundled with the package.

    Uses :mod:`importlib.resources` so it works correctly after installation
    (including inside zip/wheel distributions).
    """
    ref = importlib.resources.files("python_magnetrun") / filename
    return Path(str(ref))


def resolve_defs_file(
    filename: str | Path,
    user_dir: Path | None = None,
) -> Path:
    """Resolve a ``*-defs.json`` filename to a concrete :class:`~pathlib.Path`.

    Search order
    ------------
    1. If *filename* is an absolute path, return it as-is.
    2. If *filename* exists relative to the current directory, return it.
    3. User config dir: *user_dir* if given, else ``~/.config/magnetrun/``.
    4. Bundled package default (installed alongside the package).

    Parameters
    ----------
    filename:
        A bare name like ``"pupitre-defs.json"`` or any concrete path.
    user_dir:
        Override for the user config directory lookup (step 3).

    Returns
    -------
    Path
        Resolved path; always points to an existing file.

    Raises
    ------
    FileNotFoundError
        If the file cannot be found in any of the search locations.
    """
    p = Path(filename)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    udir = user_dir or _USER_CONFIG_DIR
    candidate = udir / p.name
    if candidate.exists():
        logger.debug(f"resolve_defs_file: using user override {candidate}")
        return candidate
    bundled = _bundled_defs_path(p.name)
    if bundled.exists():
        logger.debug(f"resolve_defs_file: using bundled default {bundled}")
        return bundled
    raise FileNotFoundError(
        f"resolve_defs_file: {p.name!r} not found in {udir}, "
        f"or the package bundle. Searched: {udir}, {bundled.parent}"
    )


# Sentinel — distinct from None so update_field_def can tell "not supplied"
# from "explicitly set to None" (= dimensionless).
_UNSET = object()


# ---------------------------------------------------------------------------
# Low-level I/O
# ---------------------------------------------------------------------------


def load_defs(json_file: str | Path) -> dict:
    """Return the raw dict loaded from *json_file*.

    Bare filenames (e.g. ``"pupitre-defs.json"``) are resolved via
    :func:`resolve_defs_file`; absolute or existing relative paths are used
    directly.
    """
    path = resolve_defs_file(json_file)
    with open(path) as fh:
        return json.load(fh)


def save_defs(json_file: str | Path, defs: dict) -> None:
    """Write *defs* back to *json_file* (pretty-printed, trailing newline)."""
    with open(json_file, "w") as fh:
        json.dump(defs, fh, indent=2)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Field management
# ---------------------------------------------------------------------------


def add_field_def(
    json_file: str | Path,
    key: str,
    symbol: str,
    unit: str | None,
    description: str = "",
    label: str = "",
    overwrite: bool = False,
) -> None:
    """Add a new field entry to *json_file*.

    Parameters
    ----------
    json_file:
        Path to the ``*-defs.json`` file.
    key:
        Column/channel name to add.
    symbol:
        Physical symbol (e.g. ``"I"``, ``"T"``).
    unit:
        Pint-parseable unit string, or ``None`` for dimensionless/timestamp.
    description:
        Optional human-readable description.
    label:
        Optional human-readable plot label (e.g. ``"Magnetic Field"``).
        Stored only when non-empty to keep files compact.
    overwrite:
        If True, silently replace an existing entry; otherwise raise ``KeyError``.
    """
    defs = load_defs(json_file)
    if key in defs and not key.startswith("_"):
        if not overwrite:
            raise KeyError(f"{key!r} already exists in {json_file}")
        logger.info(f"add_field_def: overwriting existing entry {key!r}")
    entry: dict = {"description": description, "symbol": symbol, "unit": unit}
    if label:
        entry["label"] = label
    defs[key] = entry
    save_defs(json_file, defs)
    logger.info(f"add_field_def: added {key!r} to {json_file}")


def update_field_def(
    json_file: str | Path,
    key: str,
    symbol: str | None = None,
    unit: str | None | object = _UNSET,
    description: str | None = None,
    label: str | None = None,
) -> None:
    """Update one or more attributes of an existing field entry in *json_file*.

    Only keyword arguments you explicitly supply are modified; the rest are
    left intact.  Pass ``unit=None`` to mark a field as dimensionless/timestamp.

    Parameters
    ----------
    json_file:
        Path to the ``*-defs.json`` file.
    key:
        Field name to update (must already exist).
    symbol:
        New symbol, or ``None`` to leave unchanged.
    unit:
        New unit string, ``None`` to set dimensionless, or omit to leave unchanged.
    description:
        New description, or ``None`` to leave unchanged.
    label:
        New human-readable plot label, or ``None`` to leave unchanged.
        Pass ``""`` to remove the label from the entry.
    """
    defs = load_defs(json_file)
    if key not in defs or key.startswith("_"):
        raise KeyError(f"update_field_def: {key!r} not found in {json_file!r}")
    entry: dict = defs[key]
    if symbol is not None:
        entry["symbol"] = symbol
    if unit is not _UNSET:
        entry["unit"] = unit
    if description is not None:
        entry["description"] = description
    if label is not None:
        if label == "":
            entry.pop("label", None)  # remove when explicitly cleared
        else:
            entry["label"] = label
    defs[key] = entry
    save_defs(json_file, defs)
    logger.info(f"update_field_def: updated {key!r} in {json_file}")


def delete_field_def(json_file: str | Path, key: str) -> None:
    """Remove a field entry from *json_file*.

    Parameters
    ----------
    json_file:
        Path to the ``*-defs.json`` file.
    key:
        Column/channel name to remove (must already exist).
    """
    defs = load_defs(json_file)
    if key not in defs or key.startswith("_"):
        raise KeyError(f"delete_field_def: {key!r} not found in {json_file!r}")
    del defs[key]
    save_defs(json_file, defs)
    logger.info(f"delete_field_def: removed {key!r} from {json_file}")


def list_field_defs(json_file: str | Path) -> list[dict]:
    """Return all non-comment entries in *json_file* as a list of dicts.

    Each dict has keys ``"field"``, ``"symbol"``, ``"unit"``, ``"label"``,
    ``"description"``, ``"aliases"``.  A formatted table is also printed to
    stdout.
    """
    defs = load_defs(json_file)
    rows = [
        {
            "field": key,
            "symbol": entry.get("symbol", ""),
            "unit": entry.get("unit", ""),
            "label": entry.get("label", ""),
            "description": entry.get("description", ""),
            "aliases": entry.get("aliases", {}),
        }
        for key, entry in defs.items()
        if not key.startswith("_")
    ]
    if rows:
        w = max(len(r["field"]) for r in rows)
        header = f"{'Field':<{w}}  {'Symbol':<8}  {'Unit':<20}  {'Label':<20}  Description"
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r['field']:<{w}}  {r['symbol']:<8}  {str(r['unit']):<20}  {r['label']:<20}  {r['description']}"
            )
            for fmt, target in r["aliases"].items():
                print(f"  {'':>{w}}  alias[{fmt}] = {target}")
    return rows


# ---------------------------------------------------------------------------
# Alias management
# ---------------------------------------------------------------------------


def add_alias(
    json_file: str | Path,
    key: str,
    format_name: str,
    target_field: str,
) -> None:
    """Add or update a cross-format alias for *key* in *json_file*.

    Aliases express housing-independent name correspondences between formats.
    Do NOT use aliases for housing-dependent mappings (e.g. GR1=IH vs IB);
    those belong in ``housing_config.py``.

    Parameters
    ----------
    json_file:
        Path to the ``*-defs.json`` file.
    key:
        Field name to annotate (must already exist).
    format_name:
        Name of the target format (e.g. ``"pigbrother"``, ``"hybrid"``).
    target_field:
        Name of the corresponding field in the target format
        (use the full key, e.g. ``"Courants_Alimentations/Courant_A1"``).
    """
    defs = load_defs(json_file)
    if key not in defs or key.startswith("_"):
        raise KeyError(f"add_alias: {key!r} not found in {json_file!r}")
    entry = defs[key]
    aliases: dict = entry.get("aliases", {})
    if format_name in aliases:
        logger.warning(
            f"add_alias: {key!r} already has alias {format_name!r} = {aliases[format_name]!r} in {json_file!r} — skipping"
        )
        print(
            f"Warning: {key!r} already has alias [{format_name!r}] = {aliases[format_name]!r} — skipping"
        )
        return
    aliases[format_name] = target_field
    entry["aliases"] = aliases
    defs[key] = entry
    save_defs(json_file, defs)
    logger.info(f"add_alias: {key!r}[{format_name!r}] = {target_field!r} in {json_file}")


def remove_alias(
    json_file: str | Path,
    key: str,
    format_name: str,
) -> None:
    """Remove a cross-format alias from *key* in *json_file*.

    Parameters
    ----------
    json_file:
        Path to the ``*-defs.json`` file.
    key:
        Field name to modify (must already exist).
    format_name:
        Alias format to remove (must exist in the field's ``aliases``).
    """
    defs = load_defs(json_file)
    if key not in defs or key.startswith("_"):
        raise KeyError(f"remove_alias: {key!r} not found in {json_file!r}")
    aliases: dict = defs[key].get("aliases", {})
    if format_name not in aliases:
        raise KeyError(
            f"remove_alias: format {format_name!r} not in aliases for {key!r}"
        )
    del aliases[format_name]
    defs[key]["aliases"] = aliases
    save_defs(json_file, defs)
    logger.info(f"remove_alias: removed {format_name!r} alias from {key!r} in {json_file}")


def get_aliases(json_file: str | Path, key: str) -> dict[str, str]:
    """Return the ``aliases`` dict for *key* in *json_file* (may be empty).

    Parameters
    ----------
    json_file:
        Path to the ``*-defs.json`` file.
    key:
        Field name to query (must already exist).

    Returns
    -------
    dict
        ``{format_name: target_field}`` mapping.
    """
    defs = load_defs(json_file)
    if key not in defs or key.startswith("_"):
        raise KeyError(f"get_aliases: {key!r} not found in {json_file!r}")
    return dict(defs[key].get("aliases", {}))


def build_crossref(
    defs_files: dict[str, str | Path],
) -> dict[str, dict[str, dict[str, str]]]:
    """Build a unified cross-reference index from multiple defs files.

    Collects every field that has at least one alias and organises the result
    by format so that lookups in either direction are O(1).

    Parameters
    ----------
    defs_files:
        ``{format_name: path_to_defs_json}`` — one entry per format.

    Returns
    -------
    dict
        ``index[format_name][field_name]`` = ``{other_format: other_field}``.

    Example
    -------
    ::

        # Bare names are resolved via resolve_defs_file() — bundled defaults
        # are used automatically; drop a file into ~/.config/magnetrun/ to
        # override for a specific format.
        index = build_crossref({
            "pupitre":    "pupitre-defs.json",
            "pigbrother": "pigbrother-defs.json",
            "hybrid":     "hybrid-defs.json",
        })
        index["pupitre"]["Idcct1"]
        # -> {"pigbrother": "Courants_Alimentations/Courant_A1",
        #     "hybrid":     "FEPC-AUX-LNCMI/ALIM1_J1"}
    """
    index: dict[str, dict[str, dict[str, str]]] = {fmt: {} for fmt in defs_files}
    for fmt, path in defs_files.items():
        defs = load_defs(path)
        for key, entry in defs.items():
            if key.startswith("_"):
                continue
            aliases = entry.get("aliases", {})
            if aliases:
                index[fmt][key] = dict(aliases)
    return index


# ---------------------------------------------------------------------------
# Cross-format channel matching
# ---------------------------------------------------------------------------

_DEFS_FILE_BY_DATATYPE = {
    DataType.PUPITRE: "pupitre",
    DataType.ENSIGHT: "pupitre",  # Ensight reuses pupitre-defs.json (bare channel names)
    DataType.TDMS: "pigbrother",
}


@functools.lru_cache(maxsize=8)
def _cached_defs(fmt: str) -> dict:
    """Load and cache "<fmt>-defs.json" via resolve_defs_file's normal search order."""
    return load_defs(f"{fmt}-defs.json")


def _match_key(fmt: str, group_name: str, channel: str) -> str:
    """Return the *-defs.json lookup key for *channel* in *group_name*.

    PigBrother entries are keyed by the full "Group/Channel" path; every
    other format is keyed by the bare channel name.
    """
    return f"{group_name}/{channel}" if fmt == "pigbrother" else channel


def _resolve_entry_meta(group_name: str, matched: dict[str, str]) -> dict:
    """Build a UI-ready entry for one physical channel matched across formats.

    *matched* is ``{format_name: channel_name}`` for one or more formats.
    Label/unit/description are taken from the first format (in *matched*
    iteration order) whose field definition provides them.
    """
    infos = {
        fmt: _cached_defs(fmt).get(_match_key(fmt, group_name, ch), {})
        for fmt, ch in matched.items()
    }

    def _first(key):
        for info in infos.values():
            value = info.get(key)
            if value:
                return value
        return None

    unit = _first("unit") or ""
    unit_str = f" [{unit}]" if unit else ""
    label = _first("label") or _first("symbol") or next(iter(matched.values()))

    return {
        "id": group_name + "__" + "_".join(f"{f}-{c}" for f, c in sorted(matched.items())),
        "label": f"{label}{unit_str}",
        "description": _first("description") or "",
        "channels": matched,
    }


def match_channels_across_formats(
    channels_by_type: dict[DataType, set[str]],
    group_name: str,
) -> list[dict]:
    """Match channels across formats for one group, via cross-format aliases.

    *channels_by_type* is ``{DataType: {channel_names...}}`` — channels
    actually present in *group_name*, gathered by the caller from
    already-loaded MagnetData objects (``list_groups()``/``get_group_data()``).
    DataTypes with no defs/aliases support (e.g. HYBRID, HTS) are silently
    skipped. Only channels present in more than one format are returned — a
    channel with no counterpart elsewhere isn't a comparison.

    Parameters
    ----------
    channels_by_type:
        ``{DataType: {channel_names...}}`` for the selected files.
    group_name:
        Group name the channels belong to (used for PigBrother's
        ``"Group/Channel"`` defs-file keying).

    Returns
    -------
    list of dict
        Each entry has ``"id"``, ``"label"``, ``"description"``, and
        ``"channels"`` (``{format_name: channel_name}``) — ready for direct
        display.
    """
    channels_by_format: dict[str, set[str]] = {}
    for dtype, channels in channels_by_type.items():
        fmt = _DEFS_FILE_BY_DATATYPE.get(dtype)
        if fmt is None:
            continue
        channels_by_format.setdefault(fmt, set()).update(channels)

    if not channels_by_format:
        return []

    # PigBrother channels carry the group in their key ("Group/Channel"), so
    # they are the most reliable side to resolve aliases from when present.
    reference_fmt = (
        "pigbrother" if "pigbrother" in channels_by_format else next(iter(channels_by_format))
    )

    entries = []
    for channel in sorted(channels_by_format[reference_fmt]):
        aliases = _cached_defs(reference_fmt).get(
            _match_key(reference_fmt, group_name, channel), {}
        ).get("aliases", {})

        matched = {reference_fmt: channel}
        for other_fmt, other_channels in channels_by_format.items():
            if other_fmt == reference_fmt:
                continue
            target = aliases.get(other_fmt)
            if target and target in other_channels:
                matched[other_fmt] = target

        # Only keep channels matched across more than one format — a
        # standalone channel with no counterpart isn't a comparison.
        if len(matched) > 1:
            entries.append(_resolve_entry_meta(group_name, matched))

    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _populate_field_parser(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("json_file", help="Path to the *-defs.json file")
    sub = parser.add_subparsers(dest="field_command", required=True)

    sub.add_parser("list", help="Print all field definitions (including aliases)")

    p_add = sub.add_parser("add", help="Add a new field entry")
    p_add.add_argument("key", help="Column/channel name")
    p_add.add_argument("symbol", help="Physical symbol (e.g. I, T, B)")
    p_add.add_argument(
        "unit", help='Pint-parseable unit string, or "null" for dimensionless'
    )
    p_add.add_argument("--description", default="", help="Human-readable description")
    p_add.add_argument("--label", default="", help="Human-readable plot label (e.g. 'Magnetic Field')")
    p_add.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the entry if it already exists",
    )

    p_upd = sub.add_parser(
        "update", help="Change symbol, unit, or description of a field"
    )
    p_upd.add_argument("key", help="Field name to update (must already exist)")
    p_upd.add_argument("--symbol", default=None, help="New symbol")
    p_upd.add_argument(
        "--unit",
        default=_UNSET,
        help='New unit string, or "null" to set dimensionless',
    )
    p_upd.add_argument("--description", default=None, help="New description")
    p_upd.add_argument("--label", default=None, help="New plot label (pass empty string to remove)")

    for _del_cmd in ("delete", "remove"):
        p_del = sub.add_parser(_del_cmd, help="Remove a field entry")
        p_del.add_argument("key", help="Field name to remove (must already exist)")

    p_aa = sub.add_parser("alias-add", help="Add or update a cross-format alias")
    p_aa.add_argument("key", help="Field name to annotate")
    p_aa.add_argument("format", help="Target format name (e.g. pigbrother, hybrid)")
    p_aa.add_argument("target", help="Field name in the target format (use full key)")

    for _alias_rm_cmd in ("alias-remove", "alias-delete"):
        p_ar = sub.add_parser(_alias_rm_cmd, help="Remove a cross-format alias")
        p_ar.add_argument("key", help="Field name to modify")
        p_ar.add_argument("format", help="Alias format to remove")

    p_as = sub.add_parser("alias-show", help="Show aliases for a field")
    p_as.add_argument("key", help="Field name to query")

    p_cr = sub.add_parser(
        "crossref",
        help="Build a cross-reference index from multiple defs files",
    )
    p_cr.add_argument(
        "--format",
        metavar="NAME=FILE",
        action="append",
        dest="formats",
        required=True,
        help='Format spec "name=path/to/defs.json" (repeat for each format)',
    )


def _dispatch_field(args: argparse.Namespace) -> int:
    import sys

    if args.field_command == "list":
        list_field_defs(args.json_file)

    elif args.field_command == "add":
        unit_val: str | None = None if args.unit.lower() == "null" else args.unit
        add_field_def(
            args.json_file,
            args.key,
            args.symbol,
            unit_val,
            description=args.description,
            label=args.label,
            overwrite=args.overwrite,
        )
        print(f"Added {args.key!r} to {args.json_file}")

    elif args.field_command == "update":
        upd_unit: str | None | object = _UNSET
        if args.unit is not _UNSET:
            upd_unit = None if str(args.unit).lower() == "null" else args.unit
        update_field_def(
            args.json_file,
            args.key,
            symbol=args.symbol,
            unit=upd_unit,
            description=args.description,
            label=args.label,
        )
        print(f"Updated {args.key!r} in {args.json_file}")

    elif args.field_command in ("delete", "remove"):
        delete_field_def(args.json_file, args.key)
        print(f"Deleted {args.key!r} from {args.json_file}")

    elif args.field_command == "alias-add":
        add_alias(args.json_file, args.key, args.format, args.target)
        print(f"Alias added: {args.key!r}[{args.format!r}] = {args.target!r}")

    elif args.field_command in ("alias-remove", "alias-delete"):
        remove_alias(args.json_file, args.key, args.format)
        print(f"Alias removed: {args.key!r}[{args.format!r}]")

    elif args.field_command == "alias-show":
        aliases = get_aliases(args.json_file, args.key)
        if not aliases:
            print(f"{args.key!r} has no aliases")
        else:
            for fmt, target in aliases.items():
                print(f"  {fmt}: {target}")

    elif args.field_command == "crossref":
        defs_files: dict[str, str | Path] = {}
        for spec in args.formats:
            if "=" not in spec:
                print(
                    f"error: --format must be NAME=FILE, got: {spec!r}",
                    file=sys.stderr,
                )
                return 2
            name, path = spec.split("=", 1)
            defs_files[name] = path
        index = build_crossref(defs_files)
        for fmt, fields in index.items():
            if not fields:
                continue
            print(f"\n[{fmt}]")
            w = max(len(f) for f in fields)
            for field, aliases in sorted(fields.items()):
                alias_str = ", ".join(f"{k}={v}" for k, v in aliases.items())
                print(f"  {field:<{w}}  →  {alias_str}")

    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'field' domain into a parent subparsers group."""

    p = sub.add_parser("field", help="Manage *-defs.json field-definition files")
    _populate_field_parser(p)
    p.set_defaults(_domain_handler=_dispatch_field)


