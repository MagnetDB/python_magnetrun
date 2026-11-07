"""ETL functions for preparing MagnetRun data."""

import ast
import copy
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from natsort import natsorted

from .magnetdata_base import DataType, MagnetDataBase
from .utils.narwhals_compat import to_pandas

if TYPE_CHECKING:
    from .housing_config import HousingConfig

logger = logging.getLogger(__name__)

_FKEYS_PATTERN = re.compile(
    r"Flow\w+|Rpm\w+|HP\w+|Pmagnet|Ptot|Idcct\d|(Supra)?Field|TotalField|TAlimout|teb|tsb"
)


def _cleanup_pupitre_icoil(data: MagnetDataBase, cfg) -> None:
    """Remove zero/duplicate Icoil columns and rename surviving ones to GR role names.

    Uses *cfg* (a :class:`~python_magnetrun.housing_config.HousingConfig`) to
    determine which Icoil indices belong to GR1 vs GR2 (via
    ``voltage_channels_gr1/gr2``) and what the role names are
    (``reference_gr1_current`` / ``reference_gr2_current``).

    If the role name is already present in the DataFrame (added via
    ``pupitre_formula_map``, as in M8), the rename is skipped for that GR.
    All remaining ``Icoil\\d+`` columns are dropped at the end.
    """
    logger.info(f"{data.__class__.__name__}: {data.FileName}")

    assert data.Type == DataType.PUPITRE

    df = to_pandas(data.getData())

    # Never drop the GR1/GR2 reference current columns (e.g. IH/IB), even if
    # all-zero: a magnet running on a single supply reports zeros for the
    # unused GR, but the column is still expected downstream.
    protected_cols = {cfg.reference_gr1_current, cfg.reference_gr2_current} - {""}

    # Drop all-zero columns (skip columns matched by Fkeys patterns)
    zero_cols = [
        col
        for col in df.columns
        if (df[col] == 0).all()
        and not _FKEYS_PATTERN.match(col)
        and col not in protected_cols
    ]
    if zero_cols:
        logger.warning(
            f"{data.__class__.__name__}: dropping zero columns {zero_cols} from {data.FileName!r}"
        )
        data.removeData(zero_cols)

    # Resolve duplicate Icoil columns
    Ikeys: list = natsorted([k for k in data.getKeys() if re.match(r"Icoil\d+", k)])
    if len(Ikeys) > 2:
        ikeys_df = to_pandas(data.getData(Ikeys))
        remove = []
        for i in range(len(Ikeys)):
            for j in range(i + 1, len(Ikeys)):
                diff = ikeys_df[Ikeys[i]] - ikeys_df[Ikeys[j]]
                if abs(diff.mean()) <= 1e-2:
                    remove.append(Ikeys[j])
        remove = list(set(remove))
        if remove:
            logger.warning(
                f"{data.__class__.__name__}: dropping duplicate Icoil columns {remove} from {data.FileName!r}"
            )
            data.removeData(remove)
        Ikeys = natsorted([k for k in data.getKeys() if re.match(r"Icoil\d+", k)])

    # Build GR1/GR2 Icoil index sets from voltage channel config
    def _icoil_indices(channels) -> set[int]:
        return {int(c.removeprefix("Ucoil")) for c in channels if c.startswith("Ucoil")}

    gr1_indices = _icoil_indices(cfg.voltage_channels_gr1)
    gr2_indices = _icoil_indices(cfg.voltage_channels_gr2)
    existing = set(data.getKeys())

    # Rename first surviving Icoil from each GR to its role name,
    # but only when the role name is not already in the DataFrame.
    rename_map: dict[str, str] = {}
    for role, indices in (
        (cfg.reference_gr1_current, gr1_indices),
        (cfg.reference_gr2_current, gr2_indices),
    ):
        if not role or role in existing:
            continue
        for k in Ikeys:
            m = re.match(r"Icoil(\d+)$", k)
            if m and int(m.group(1)) in indices and k not in rename_map:
                rename_map[k] = role
                break

    if rename_map:
        logger.info(
            f"{data.__class__.__name__}: renaming {rename_map} in {data.FileName!r}"
        )
        for old, new in rename_map.items():
            data.renameData(columns={old: new})

    # Drop all remaining Icoil columns
    leftover = [k for k in data.getKeys() if re.match(r"Icoil\d+", k)]
    if leftover:
        logger.info(
            f"{data.__class__.__name__}: dropping Icoil columns {leftover} from {data.FileName!r}"
        )
        data.removeData(leftover)

    logger.info(
        f"{data.__class__.__name__}: keys after Icoil cleanup: {natsorted(data.getKeys())}"
    )


def _data_start(data: MagnetDataBase) -> datetime | None:
    """Return the start of *data* as an aware datetime (pupitre: local time, else UTC)."""
    start = data.getStartDate()
    if not start:
        return None
    try:
        naive = datetime.strptime(f"{start[0]} {start[1]}", "%Y.%m.%d %H:%M:%S")
    except ValueError:
        logger.warning(
            f"prepareData: unparsable start {start[:2]} in {data.FileName!r}, "
            "using the base housing wiring"
        )
        return None
    tz = ZoneInfo("Europe/Paris") if data.Type == DataType.PUPITRE else timezone.utc
    return naive.replace(tzinfo=tz)


def prepareData(
    data: MagnetDataBase,
    housing: str,
    keys_to_remove: list[str] | None = None,
    keys_to_rename: dict[str, str] | None = None,
    keys_to_add: dict[str, dict[str, Any]] | None = None,
    debug: bool = False,
) -> None:
    """Prepare magnet run data by adding computed fields and renaming columns.

    When *keys_to_add* and *keys_to_rename* are both ``None`` the ETL maps are
    derived automatically from the :class:`~python_magnetrun.housing_config.HousingConfig`
    for the given *housing*, in the wiring version in force at the start of
    *data* (pupitre ``Date``/``Time`` read as Europe/Paris local time):

    - **PUPITRE**: adds ``pupitre_formula_map`` entries plus the UH/UB voltage
      sum formulas (filtered to columns actually present), and renames
      Flow/Rpm/Tin/HP index columns to role-based names.
    - **TDMS** (pigbrother): adds ``pigbrother_formula_map`` entries.
    - **HYBRID**: adds hybrid voltage sum formulas.

    :param data: MagnetDataBase object to prepare in-place
    :param housing: Housing name (e.g. "M8", "M9", "M10")
    :param keys_to_remove: list of column names to remove, defaults to None
    :param keys_to_rename: dict mapping old column names to new names, defaults to None
    :param keys_to_add: dict mapping new column names to their formulas, defaults to None
    :param debug: Enable debug output, defaults to False
    """
    from .housing_config import get_housing_config

    cfg = get_housing_config(housing, at=_data_start(data))

    # Auto-build ETL maps from HousingConfig when caller passes None
    if keys_to_add is None and keys_to_rename is None:
        if data.Type == DataType.PUPITRE:
            available = data.getKeys()
            keys_to_add = {
                **cfg.pupitre_formula_map,
                **cfg.get_pupitre_voltage_formulas(available),
            }
            keys_to_rename = cfg.get_pupitre_rename_map()
        elif data.Type == DataType.TDMS:
            keys_to_add = cfg.pigbrother_formula_map
        elif data.Type == DataType.HYBRID:
            keys_to_add = {
                **cfg.hybrid_formula_map,
                **cfg.get_hybrid_voltage_formulas(data.getKeys()),
            }

    data.addTime()
    data.getDuration()

    data.cleanupData(
        keys_to_remove=keys_to_remove,
        keys_to_rename=keys_to_rename,
        keys_to_add=keys_to_add,
        debug=debug,
    )

    if data.Type == DataType.PUPITRE:
        _cleanup_pupitre_icoil(data, cfg)

    logger.debug(f"MagnetRun.prepareData: data.keys={data.getKeys()}")


# ---------------------------------------------------------------------------
# Spec-driven Pupitre ETL (PLAN_pupitre_etl_spec.md)
# ---------------------------------------------------------------------------

SPEC_KIND = "pupitre-etl-spec"
SPEC_FORMAT_VERSION = 2

_FIELD_DEF_KEYS = {"formula", "symbol", "unit", "label", "description"}
_SPEC_HOUSING_KEYS = {"name", "version", "remove", "rename", "add", "defaults", "companions"}
_SPEC_MAGNET_KEYS = {"channels", "remove", "add"}
_RULE_REQUIRED_KEYS = {"name", "expr", "stat", "tol"}
_RULE_KEYS = _RULE_REQUIRED_KEYS | {"unit", "tag", "scope"}
_RULE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
RULE_STATS = {
    "max_abs": lambda values: values.abs().max(),
    "p99_abs": lambda values: values.abs().quantile(0.99),
    "mean_abs": lambda values: values.abs().mean(),
}


def rule_columns(expr: str) -> list[str]:
    """Return the column names used by a rule expression, in order of appearance.

    Parameters
    ----------
    expr : str
        Rule expression, e.g. ``"IH - Icoil2"`` or ``"abs(Idcct1) - Idcct2"``.

    Returns
    -------
    list[str]
        Names other than called functions (``abs``), without duplicates.

    Raises
    ------
    ValueError
        If *expr* is not a single expression (syntax error, assignment), or
        uses ``@`` variables or backtick-quoted names.
    """
    if "@" in expr:
        raise ValueError(f"rule expr {expr!r}: '@' variables are not allowed")
    if "`" in expr:
        raise ValueError(
            f"rule expr {expr!r}: backtick names are reserved for multi-source rules "
            "(not supported yet)"
        )
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"rule expr {expr!r} is not an expression: {exc.msg}") from None
    called = {
        id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    names = [
        node for node in ast.walk(tree) if isinstance(node, ast.Name) and id(node) not in called
    ]
    names.sort(key=lambda node: node.col_offset)
    return list(dict.fromkeys(node.id for node in names))


def check_rule(rule: dict) -> None:
    """Validate a check rule.

    Parameters
    ----------
    rule : dict
        ``name``, ``expr``, ``stat`` (one of :data:`RULE_STATS`), ``tol``
        (≥ 0, in the rule's unit), and optionally ``unit``, ``tag`` and
        ``scope``.

    Raises
    ------
    ValueError
        On missing or unknown keys, a bad name, statistic or tolerance, or an
        expression that does not parse or uses no column.
    """
    missing = _RULE_REQUIRED_KEYS - set(rule)
    unknown = set(rule) - _RULE_KEYS
    if missing or unknown:
        raise ValueError(
            f"rule {rule.get('name')!r}: missing {sorted(missing)}, unknown {sorted(unknown)} "
            f"(allowed {sorted(_RULE_KEYS)})"
        )
    if not isinstance(rule["name"], str) or not _RULE_NAME.fullmatch(rule["name"]):
        raise ValueError(f"rule name {rule['name']!r}: letters, digits and '_' only")
    if rule["stat"] not in RULE_STATS:
        raise ValueError(f"rule {rule['name']!r}: stat {rule['stat']!r} not in {list(RULE_STATS)}")
    tol = rule["tol"]
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or tol < 0:
        raise ValueError(f"rule {rule['name']!r}: tol {tol!r} must be a number ≥ 0")
    if not rule_columns(rule["expr"]):
        raise ValueError(f"rule {rule['name']!r}: expr {rule['expr']!r} uses no column")


def _check_field_defs(where: str, defs: dict) -> None:
    """Raise ``ValueError`` unless each field definition has exactly the field-def keys."""
    for key, fd in defs.items():
        if not isinstance(fd, dict) or set(fd) != _FIELD_DEF_KEYS:
            got = sorted(fd) if isinstance(fd, dict) else fd
            raise ValueError(
                f"PupitreEtlSpec: {where} add {key!r} needs keys "
                f"{sorted(_FIELD_DEF_KEYS)}, got {got}"
            )


@dataclass(frozen=True)
class PupitreEtlSpec:
    """Explicit ETL spec for the pupitre files of one assembly and housing version.

    Attributes
    ----------
    assembly : str or None
        Assembly name; ``None`` for a housing-only spec.
    housing : dict
        Housing part: ``name``, ``version`` (housing-config version it was
        built from), ``remove`` (full-match regex patterns), ``rename``,
        ``add`` and ``defaults`` (field definitions), ``companions``.
    magnets : dict
        Magnet name → magnet wiring with ``channels``, ``remove`` and
        ``add``; never ``rename``.
    rules : list[dict]
        Check rules, already merged across housing, assembly and magnet
        levels (see :func:`check_rule`); ``scope`` records the level.

    Raises
    ------
    ValueError
        On unknown keys, incomplete field definitions, an ``add`` key given
        twice (housing ``add`` or two magnet wirings), a channel claimed by
        two magnet wirings, an invalid rule or two rules with one name.
    """

    assembly: str | None
    housing: dict
    magnets: dict = field(default_factory=dict)
    rules: list = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate keys, field definitions, rules and merge conflicts."""
        unknown = set(self.housing) - _SPEC_HOUSING_KEYS
        if unknown or "name" not in self.housing:
            raise ValueError(
                f"PupitreEtlSpec: housing part needs 'name' and only "
                f"{sorted(_SPEC_HOUSING_KEYS)}; unknown {sorted(unknown)}"
            )
        _check_field_defs("housing", self.housing.get("add", {}))
        _check_field_defs("housing defaults", self.housing.get("defaults", {}))
        added = {key: "the housing" for key in self.housing.get("add", {})}
        claimed: dict[str, str] = {}
        for magnet, part in self.magnets.items():
            unknown = set(part) - _SPEC_MAGNET_KEYS
            if unknown:
                raise ValueError(
                    f"PupitreEtlSpec: magnet {magnet!r} has unknown key(s) "
                    f"{sorted(unknown)} (allowed {sorted(_SPEC_MAGNET_KEYS)})"
                )
            _check_field_defs(f"magnet {magnet!r}", part.get("add", {}))
            for key in part.get("add", {}):
                if key in added:
                    raise ValueError(
                        f"PupitreEtlSpec: {key!r} added by both {added[key]} "
                        f"and magnet {magnet!r}"
                    )
                added[key] = f"magnet {magnet!r}"
            for channel in part.get("channels", []):
                if channel in claimed:
                    raise ValueError(
                        f"PupitreEtlSpec: channel {channel!r} claimed by magnets "
                        f"{claimed[channel]!r} and {magnet!r}"
                    )
                claimed[channel] = magnet
        names: set[str] = set()
        for rule in self.rules:
            check_rule(rule)
            if rule["name"] in names:
                raise ValueError(f"PupitreEtlSpec: two rules named {rule['name']!r}")
            names.add(rule["name"])

    @property
    def remove(self) -> list[str]:
        """Remove patterns: the housing's, then each magnet part's."""
        patterns = list(self.housing.get("remove", []))
        for part in self.magnets.values():
            patterns += part.get("remove", [])
        return patterns

    @property
    def rename(self) -> dict[str, str]:
        """Rename map (housing only)."""
        return dict(self.housing.get("rename", {}))

    @property
    def add(self) -> dict[str, dict]:
        """Field definitions to add: housing ``add``, magnet ``add``, then the unreplaced ``defaults``."""
        defs = dict(self.housing.get("add", {}))
        for part in self.magnets.values():
            defs.update(part.get("add", {}))
        for key, fd in self.housing.get("defaults", {}).items():
            defs.setdefault(key, fd)
        return defs

    def to_dict(self) -> dict:
        """Return the JSON form of the spec (structure kept, not flattened).

        Returns
        -------
        dict
            ``{"kind", "version", "assembly", "housing", "magnets", "rules"}``.
        """
        return {
            "kind": SPEC_KIND,
            "version": SPEC_FORMAT_VERSION,
            "assembly": self.assembly,
            "housing": copy.deepcopy(self.housing),
            "magnets": copy.deepcopy(self.magnets),
            "rules": copy.deepcopy(self.rules),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PupitreEtlSpec":
        """Build a spec from its JSON form.

        Parameters
        ----------
        d : dict
            Output of :meth:`to_dict` (or a JSON file of the same shape).

        Returns
        -------
        PupitreEtlSpec
            The validated spec.

        Raises
        ------
        ValueError
            If ``kind`` or the format ``version`` is not supported, or the
            content is invalid (see the class).
        """
        if d.get("kind") != SPEC_KIND:
            raise ValueError(f"PupitreEtlSpec: kind {d.get('kind')!r} is not {SPEC_KIND!r}")
        if d.get("version") != SPEC_FORMAT_VERSION:
            raise ValueError(
                f"PupitreEtlSpec: format version {d.get('version')!r} is not "
                f"{SPEC_FORMAT_VERSION}"
            )
        return cls(
            assembly=d.get("assembly"),
            housing=d["housing"],
            magnets=d.get("magnets") or {},
            rules=d.get("rules") or [],
        )

    @classmethod
    def from_housing(
        cls, cfg: "HousingConfig", assembly: str | None = None
    ) -> "PupitreEtlSpec":
        """Build the housing-only spec of a housing config.

        Parameters
        ----------
        cfg : HousingConfig
            Housing config already resolved at the file start
            (``get_housing_config(housing, at=t0)``); its ``version`` is
            recorded in ``housing.version``.
        assembly : str, optional
            Assembly name to record; no magnet wiring and no rule is added.

        Returns
        -------
        PupitreEtlSpec
            Spec adding ``IH``/``IB`` (``pupitre_formula_map``), the voltage
            sums as ``defaults``, renaming Flow/Rpm/Tin/HP to role names,
            removing every ``Icoil`` and the companions of removed ``Ucoil``.
        """
        housing = {
            "name": cfg.name,
            "version": cfg.version,
            "remove": [r"Icoil\d+"],
            "rename": cfg.get_pupitre_rename_map(),
            "add": copy.deepcopy(cfg.pupitre_formula_map),
            "defaults": cfg.get_pupitre_voltage_formulas(),
            "companions": {"Ucoil": ["Icoil", "DRcoil", "Tcal"]},
        }
        return cls(assembly=assembly, housing=housing, magnets={})


def load_pupitre_etl_spec(path: str | Path) -> PupitreEtlSpec:
    """Load a ``*.pupitre-etl.json`` file.

    Parameters
    ----------
    path : str or :class:`~pathlib.Path`
        JSON file written by :func:`save_pupitre_etl_spec`.

    Returns
    -------
    PupitreEtlSpec
        The validated spec.
    """
    return PupitreEtlSpec.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_pupitre_etl_spec(path: str | Path, spec: PupitreEtlSpec) -> None:
    """Write *spec* to *path* as pretty-printed JSON.

    Parameters
    ----------
    path : str or :class:`~pathlib.Path`
        Destination, by convention ``<assembly>.<housing version>.pupitre-etl.json``.
    spec : PupitreEtlSpec
        Spec to write.
    """
    Path(path).write_text(
        json.dumps(spec.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def pupitre_file_start(path: str | Path) -> datetime:
    """Return the start of a pupitre file: its first ``Date``/``Time`` row, in Europe/Paris time.

    Parameters
    ----------
    path : str or :class:`~pathlib.Path`
        Pupitre ``.txt`` file (line 1 = parameter line) or a ``.csv`` export.

    Returns
    -------
    datetime
        Timezone-aware start time.
    """
    import pandas as pd

    is_txt = str(path).endswith(".txt")
    row = pd.read_csv(
        path, sep=r"\s+" if is_txt else ",", engine="python", skiprows=1 if is_txt else 0, nrows=1
    )
    naive = datetime.strptime(
        f"{row['Date'].iloc[0]} {row['Time'].iloc[0]}", "%Y.%m.%d %H:%M:%S"
    )
    return naive.replace(tzinfo=ZoneInfo("Europe/Paris"))


def find_pupitre_etl_spec(
    etl_dir: str | Path, assembly: str, housing: str, at: datetime
) -> Path:
    """Return ``<etl_dir>/<assembly>.<housing version>.pupitre-etl.json`` for the version at *at*.

    Parameters
    ----------
    etl_dir : str or :class:`~pathlib.Path`
        Directory of exported spec files.
    assembly : str
        Assembly name.
    housing : str
        Housing name, used to resolve the housing-config version.
    at : datetime
        Timezone-aware time, normally the file start (:func:`pupitre_file_start`).

    Returns
    -------
    :class:`~pathlib.Path`
        Existing spec file.

    Raises
    ------
    FileNotFoundError
        If no spec file exists for that assembly and version.
    """
    from .housing_config import get_housing_config

    version = get_housing_config(housing, at=at).version
    path = Path(etl_dir) / f"{assembly}.{version}.pupitre-etl.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No ETL spec {path} (assembly {assembly!r}, housing version {version!r} at {at})"
        )
    return path


def _housing_version_check(data: MagnetDataBase, spec: PupitreEtlSpec) -> dict:
    """Compare the spec's housing version with the one in force at the data start."""
    from .housing_config import get_housing_config

    start = _data_start(data)
    data_version = (
        get_housing_config(spec.housing["name"], at=start).version if start else None
    )
    spec_version = spec.housing.get("version", "")
    return {
        "kind": "housing_version",
        "spec": spec_version,
        "data": data_version,
        "ok": None if data_version is None else data_version == spec_version,
    }


def _rule_checks(data: MagnetDataBase, spec: PupitreEtlSpec) -> list[dict]:
    """Evaluate each rule: its statistic over all rows (NaN ignored) must be ≤ ``tol``."""
    keys = data.getKeys()
    results = []
    for rule in spec.rules:
        columns = rule_columns(rule["expr"])
        result = {
            "kind": "rule",
            "name": rule["name"],
            "scope": rule.get("scope"),
            "expr": rule["expr"],
            "stat": rule["stat"],
            "value": None,
            "tol": rule["tol"],
            "unit": rule.get("unit"),
            "tag": rule.get("tag"),
            "missing": [c for c in columns if c not in keys],
            "ok": None,
        }
        if not result["missing"]:
            df = to_pandas(data.getData(columns))
            value = RULE_STATS[rule["stat"]](df.eval(rule["expr"]))
            if value == value:  # not NaN
                result["value"] = float(value)
                result["ok"] = bool(value <= rule["tol"])
        results.append(result)
    return results


def _channel_checks(
    data: MagnetDataBase, spec: PupitreEtlSpec, removed: list[str]
) -> list[dict]:
    """Check that each removed channel (``Ucoil n`` and its companions) is exactly 0."""
    keys = data.getKeys()
    results = []
    for base, companions in spec.housing.get("companions", {}).items():
        for key in natsorted(removed):
            match = re.fullmatch(rf"{re.escape(base)}(\d+)", key)
            if not match:
                continue
            n = match.group(1)
            cols = [f"{f}{n}" for f in [base, *companions] if f"{f}{n}" in keys]
            df = to_pandas(data.getData(cols))
            fields = {}
            for col in cols:
                peak = df[col].abs().max()
                if peak == peak and peak != 0:  # not NaN, not 0
                    fields[col] = float(peak)
            results.append(
                {"kind": "channel", "channel": int(n), "fields": fields, "ok": not fields}
            )
    return results


def prepare_pupitre(data: MagnetDataBase, spec: PupitreEtlSpec) -> list[dict]:
    """Apply an explicit ETL spec to pupitre data, in place.

    Order: housing-version check → ``addTime`` → rename → add → rule and
    channel checks → remove (full-match patterns) → companions → ``t``
    dedupe. No content-based dropping (all-zero, duplicate columns) and no
    ``Icoil`` guessing. Failed checks are logged as warnings; they never keep
    or drop a column. Rules therefore see raw, added and renamed names, not
    the original name of a renamed column.

    Parameters
    ----------
    data : MagnetDataBase
        Pupitre data (pandas or polars backend), not yet prepared.
    spec : PupitreEtlSpec
        Spec of the file's assembly and housing version.

    Returns
    -------
    list[dict]
        Check results, each with a ``kind`` (``housing_version``, ``rule``,
        ``channel``) and an ``ok`` (``True``, ``False`` or ``None`` when it
        could not be evaluated).
    """
    assert data.Type == DataType.PUPITRE

    results = [_housing_version_check(data, spec)]  # needs Date/Time, before addTime
    data.addTime()
    data.getDuration()

    keys = data.getKeys()
    missing = [old for old in spec.rename if old not in keys]
    if missing:
        logger.warning(f"prepare_pupitre: {data.FileName}: no column to rename for {missing}")
    rename = {old: new for old, new in spec.rename.items() if old in keys}
    if rename:
        data.renameData(rename)

    keys = data.getKeys()
    existing = [key for key in spec.add if key in keys]
    if existing:
        logger.warning(f"prepare_pupitre: {data.FileName}: {existing} already present, not added")
    add = {key: fd for key, fd in spec.add.items() if key not in keys}
    data.cleanupData(keys_to_add=add, drop_redundant_columns=False)

    keys = data.getKeys()
    removed = [k for k in keys if any(re.fullmatch(p, k) for p in spec.remove)]
    for pattern in spec.remove:
        if not any(re.fullmatch(pattern, k) for k in keys):
            logger.debug(f"prepare_pupitre: {data.FileName}: {pattern!r} matches no column")

    results += _rule_checks(data, spec)
    results += _channel_checks(data, spec, removed)
    for result in results:
        if result["ok"] is False:
            logger.warning(f"prepare_pupitre: {data.FileName}: failed check {result}")

    if removed:
        data.removeData(removed)

    keys = data.getKeys()
    orphans = []
    for base, companions in spec.housing.get("companions", {}).items():
        for key in keys:
            for comp in companions:
                match = re.fullmatch(rf"{re.escape(comp)}(\d+)", key)
                if match and f"{base}{match.group(1)}" not in keys:
                    orphans.append(key)
    if orphans:
        data.removeData(orphans)

    logger.debug(f"prepare_pupitre: data.keys={data.getKeys()}")
    return results
