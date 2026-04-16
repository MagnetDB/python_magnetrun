"""housing_config — Housing-dependent sensor role assignments.

Each housing (M8, M9, M10, …) wires the same physical sensors differently.
This module captures those default role assignments and provides tools to
load, save, and manage per-housing JSON config files, plus a clean way to
produce runtime-overridden copies for the rare cases where the wiring
differs from the default (e.g. GR1 current = IB instead of IH for M9).

**What belongs here**

- Which pupitre field plays each GR role (current, flow, rpm, pressure,
  voltage) for a given housing — i.e. housing-dependent role assignments.
- Which data formats are available for each housing
  (``"pupitre"``, ``"pigbrother"``, ``"hybrid"``).

**What does NOT belong here**

- Fixed, housing-independent name correspondences between formats (those
  live as ``"aliases"`` in the ``*-defs.json`` files).
- Analysis constants, sampling rates, thresholds, colours (those live in
  ``analysis/config.py``).

**Per-housing JSON files** (``<Housing>-housing-config.json``) are the
canonical editable source.  The in-module ``HOUSING_CONFIGS`` dict is built
from those files at import time and used as a fast in-memory cache.

Example
-------
::

    # Load directly from a JSON file
    from python_magnetrun.housing_config import load_housing_config
    cfg = load_housing_config("M9-housing-config.json")

    # Get the built-in default (no file needed)
    from python_magnetrun.housing_config import get_housing_config
    cfg = get_housing_config("M9")
    cfg.reference_gr1_current   # "IH"

    # Runtime override for an atypical M9 run (GR1/GR2 supplies swapped)
    cfg_swapped = get_housing_config("M9", overrides={"gr1_current": "IB",
                                                      "gr2_current": "IH"})

    # Persist a modified config back to its JSON file
    from python_magnetrun.housing_config import update_housing_config
    update_housing_config("M9-housing-config.json", {"gr1_current": "IB"})
"""

from __future__ import annotations

import dataclasses
import importlib.resources
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_USER_CONFIG_DIR = Path.home() / ".config" / "magnetrun"


def get_bundled_housing_config_path(housing: str) -> Path:
    """Return the path to the bundled ``<Housing>-housing-config.json`` template.

    Uses :mod:`importlib.resources` so it works correctly after installation.
    The bundled file is read-only — use :func:`get_user_housing_config_path` to
    obtain a writable user-local copy.
    """
    filename = f"{housing}-housing-config.json"
    ref = importlib.resources.files("python_magnetrun") / filename
    return Path(str(ref))


def get_user_housing_config_path(housing: str) -> Path:
    """Return ``~/.config/magnetrun/<Housing>-housing-config.json``.

    The directory is created if it does not yet exist.  This is the
    recommended location for persistent per-housing customisations.
    """
    _USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return _USER_CONFIG_DIR / f"{housing}-housing-config.json"


# ---------------------------------------------------------------------------
# Role → HousingConfig field-name mapping
# ---------------------------------------------------------------------------

#: Maps semantic role names (used in ``overrides`` dicts) to the corresponding
#: ``HousingConfig`` dataclass field names.
ROLE_TO_FIELD: dict[str, str] = {
    "gr1_current": "reference_gr1_current",
    "gr2_current": "reference_gr2_current",
    "gr1_flow": "reference_gr1_flow",
    "gr2_flow": "reference_gr2_flow",
    "gr1_rpm": "reference_gr1_rpm",
    "gr2_rpm": "reference_gr2_rpm",
    "gr1_pin": "reference_gr1_pin",
    "gr2_pin": "reference_gr2_pin",
    "voltage_channels_gr1": "voltage_channels_gr1",
    "voltage_channels_gr2": "voltage_channels_gr2",
    "pupitre_formula_map": "pupitre_formula_map",
    "pigbrother_formula_map": "pigbrother_formula_map",
    "hybrid_formula_map": "hybrid_formula_map",
    "reference_gr1_voltage": "reference_gr1_voltage",
    "reference_gr2_voltage": "reference_gr2_voltage",
}

# Fields that are stored as lists in JSON but tuples in the dataclass
_TUPLE_FIELDS = {"formats", "voltage_channels_gr1", "voltage_channels_gr2"}

# Flow/Rpm/Tin/HP base names used to build the pupitre rename map
_PUPITRE_FLOW_FIELDS = ("Flow", "Rpm", "Tin", "HP")


# ---------------------------------------------------------------------------
# HousingConfig dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HousingConfig:
    """Housing-specific sensor role assignment.

    Captures the default wiring for one housing.  Use :func:`get_housing_config`
    with ``overrides`` to obtain a modified copy for atypical runs.

    The canonical source for each housing is a ``<Housing>-housing-config.json``
    file.  Use :func:`load_housing_config` / :func:`save_housing_config` to read
    and write those files, and :func:`update_housing_config` to modify individual
    fields in place.

    Attributes
    ----------
    name : str
        Housing identifier (e.g. ``"M9"``).
    formats : tuple[str, ...]
        Data formats available for this housing, in order of preference.
        Known values: ``"pupitre"``, ``"pigbrother"``, ``"hybrid"``.
    reference_gr1_current : str
        Pupitre field for GR1 supply current (e.g. ``"IH"``).
    reference_gr2_current : str
        Pupitre field for GR2 supply current (e.g. ``"IB"``).
    reference_gr1_flow : str
        Pupitre field for GR1 coolant flow.
    reference_gr2_flow : str
        Pupitre field for GR2 coolant flow.
    reference_gr1_rpm : str
        Pupitre field for GR1 pump RPM.
    reference_gr2_rpm : str
        Pupitre field for GR2 pump RPM.
    reference_gr1_pin : str
        Pupitre field for GR1 inlet pressure.
    reference_gr2_pin : str
        Pupitre field for GR2 inlet pressure.
    voltage_channels_gr1 : tuple[str, ...]
        Pupitre voltage-probe fields associated with GR1.
    voltage_channels_gr2 : tuple[str, ...]
        Pupitre voltage-probe fields associated with GR2.

    Notes
    -----
    M9 default: GR1 → H supply (IH, FlowH, RpmH, HPH), GR2 → B supply.
    M8/M10 default: GR1 → B supply (IB, FlowB, RpmB, HPB), GR2 → H supply.
    This swap is a physical wiring difference, not a naming convention.
    """

    name: str
    formats: tuple = field(default_factory=tuple)

    reference_gr1_current: str = ""
    reference_gr2_current: str = ""
    reference_gr1_flow: str = ""
    reference_gr2_flow: str = ""
    reference_gr1_rpm: str = ""
    reference_gr2_rpm: str = ""
    reference_gr1_pin: str = ""
    reference_gr2_pin: str = ""
    voltage_channels_gr1: tuple = field(default_factory=tuple)
    voltage_channels_gr2: tuple = field(default_factory=tuple)

    # ETL formula maps — keys_to_add passed to cleanupData per format
    pupitre_formula_map: dict = field(default_factory=dict)
    pigbrother_formula_map: dict = field(default_factory=dict)

    # Hybrid (M8-only) derived current formulas; empty string = not applicable
    hybrid_formula_map: dict = field(default_factory=dict)

    # Target aggregate voltage field names for GR1/GR2
    reference_gr1_voltage: str = "UH"
    reference_gr2_voltage: str = "UB"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain JSON-serialisable dict (tuples → lists)."""
        d = dataclasses.asdict(self)
        for key in _TUPLE_FIELDS:
            if key in d:
                d[key] = list(d[key])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> HousingConfig:
        """Construct a :class:`HousingConfig` from a plain dict (lists → tuples).

        Comment keys (starting with ``"_"``) are silently ignored.
        """
        clean = {k: v for k, v in d.items() if not k.startswith("_")}
        for key in _TUPLE_FIELDS:
            if key in clean and isinstance(clean[key], list):
                clean[key] = tuple(clean[key])
        return cls(**clean)

    # ------------------------------------------------------------------
    # Role-based lookups (keyed by TDMS reference channel name)
    # ------------------------------------------------------------------

    def get_pupitre_channel(self, reference_key: str) -> str:
        """Return the pupitre current field for a TDMS reference channel."""
        return {
            "Référence_GR1": self.reference_gr1_current,
            "Référence_GR2": self.reference_gr2_current,
        }.get(reference_key, "")

    def get_flow_channel(self, reference_key: str) -> str:
        """Return the pupitre flow field for a TDMS reference channel."""
        return {
            "Référence_GR1": self.reference_gr1_flow,
            "Référence_GR2": self.reference_gr2_flow,
        }.get(reference_key, "")

    def get_rpm_channel(self, reference_key: str) -> str:
        """Return the pupitre RPM field for a TDMS reference channel."""
        return {
            "Référence_GR1": self.reference_gr1_rpm,
            "Référence_GR2": self.reference_gr2_rpm,
        }.get(reference_key, "")

    def get_pin_channel(self, reference_key: str) -> str:
        """Return the pupitre inlet-pressure field for a TDMS reference channel."""
        return {
            "Référence_GR1": self.reference_gr1_pin,
            "Référence_GR2": self.reference_gr2_pin,
        }.get(reference_key, "")

    def get_voltage_channels(self, reference_key: str) -> tuple:
        """Return the pupitre voltage-probe fields for a TDMS reference channel."""
        return {
            "Référence_GR1": self.voltage_channels_gr1,
            "Référence_GR2": self.voltage_channels_gr2,
        }.get(reference_key, ())

    def supports_format(self, fmt: str) -> bool:
        """Return True if *fmt* is available for this housing."""
        return fmt in self.formats

    # ------------------------------------------------------------------
    # Computed ETL helpers
    # ------------------------------------------------------------------

    def get_pupitre_rename_map(self) -> dict[str, str]:
        """Derive the Flow/Rpm/Tin/HP index→role rename map from the GR1 flow role.

        M9 has ``reference_gr1_flow = "FlowH"`` → GR1 index 1 → rename
        ``Flow1→FlowH``, ``Flow2→FlowB``.  M8/M10 has
        ``reference_gr1_flow = "FlowB"`` → rename ``Flow1→FlowB``,
        ``Flow2→FlowH``.  Returns ``{}`` when ``reference_gr1_flow`` is unset.
        """
        if not self.reference_gr1_flow:
            return {}
        if self.reference_gr1_flow.endswith("H"):
            h_idx, b_idx = "1", "2"
        else:
            h_idx, b_idx = "2", "1"
        result: dict[str, str] = {}
        for base in _PUPITRE_FLOW_FIELDS:
            result[f"{base}{h_idx}"] = f"{base}H"
            result[f"{base}{b_idx}"] = f"{base}B"
        return result

    def get_pupitre_voltage_formulas(
        self, available_keys: list[str] | None = None
    ) -> dict[str, str]:
        """Return ``{reference_gr1_voltage: formula, reference_gr2_voltage: formula}``.

        Only Ucoil channels present in *available_keys* are summed.  If
        *available_keys* is ``None``, all channels in
        ``voltage_channels_gr1``/``voltage_channels_gr2`` are used.
        """

        def _sum_formula(target: str, channels: tuple, keys: list[str] | None) -> str:
            srcs = [c for c in channels if keys is None or c in keys]
            return f"{target} = {' + '.join(srcs)}" if srcs else ""

        result: dict[str, str] = {}
        f1 = _sum_formula(
            self.reference_gr1_voltage, self.voltage_channels_gr1, available_keys
        )
        f2 = _sum_formula(
            self.reference_gr2_voltage, self.voltage_channels_gr2, available_keys
        )
        if f1:
            result[self.reference_gr1_voltage] = f1
        if f2:
            result[self.reference_gr2_voltage] = f2
        return result

    def get_hybrid_voltage_formulas(
        self, available_keys: list[str] | None = None
    ) -> dict[str, str]:
        """Return voltage sum formulas for hybrid (kHz) data (M8-only).

        Returns ``{}`` for housings that do not support the ``"hybrid"`` format.
        Currently delegates to :meth:`get_pupitre_voltage_formulas` since hybrid
        voltage channels use the same Ucoil lists; callers can override by
        passing their own ``available_keys``.
        """
        if not self.supports_format("hybrid"):
            return {}
        return self.get_pupitre_voltage_formulas(available_keys)

    # ------------------------------------------------------------------
    # Backward-compatibility helper
    # ------------------------------------------------------------------

    def to_pupitre_dict(self) -> dict[str, str]:
        """Return the legacy ``pupitre_dict`` format used by the original code."""
        return {
            "Référence_GR1": self.reference_gr1_current,
            "Référence_GR2": self.reference_gr2_current,
            "Référence_GR1_Q": self.reference_gr1_flow,
            "Référence_GR2_Q": self.reference_gr2_flow,
            "Référence_GR1_Rpm": self.reference_gr1_rpm,
            "Référence_GR2_Rpm": self.reference_gr2_rpm,
            "Référence_GR1_Pin": self.reference_gr1_pin,
            "Référence_GR2_Pin": self.reference_gr2_pin,
        }


# ---------------------------------------------------------------------------
# JSON file I/O
# ---------------------------------------------------------------------------


def load_housing_config(json_file: str | Path) -> HousingConfig:
    """Load a :class:`HousingConfig` from a ``<Housing>-housing-config.json`` file.

    Parameters
    ----------
    json_file:
        Path to the JSON file (e.g. ``"M9-housing-config.json"``).

    Returns
    -------
    HousingConfig
        The loaded configuration.
    """
    with open(json_file) as fh:
        d = json.load(fh)
    cfg = HousingConfig.from_dict(d)
    logger.debug(f"load_housing_config: loaded {cfg.name!r} from {json_file}")
    return cfg


def save_housing_config(json_file: str | Path, config: HousingConfig) -> None:
    """Write *config* to *json_file* (pretty-printed, trailing newline).

    Parameters
    ----------
    json_file:
        Destination path.
    config:
        The :class:`HousingConfig` to persist.
    """
    with open(json_file, "w") as fh:
        json.dump(config.to_dict(), fh, indent=2)
        fh.write("\n")
    logger.info(f"save_housing_config: wrote {config.name!r} to {json_file}")


def update_housing_config(
    json_file: str | Path,
    overrides: dict,
) -> HousingConfig:
    """Apply *overrides* to the config stored in *json_file* and save it.

    *overrides* is keyed by **role names** (see :data:`ROLE_TO_FIELD`), not
    by raw dataclass field names.  The file is updated in place and the new
    :class:`HousingConfig` is returned.

    Parameters
    ----------
    json_file:
        Path to the JSON file to modify.
    overrides:
        ``{role: new_value}`` dict.  For tuple fields
        (``"voltage_channels_gr1"``, ``"voltage_channels_gr2"``) pass a list.

    Returns
    -------
    HousingConfig
        The updated configuration.

    Raises
    ------
    ValueError
        If any key in *overrides* is not a valid role name.
    """
    invalid = set(overrides) - set(ROLE_TO_FIELD)
    if invalid:
        raise ValueError(
            f"update_housing_config: unknown role(s) {sorted(invalid)}. "
            f"Valid: {sorted(ROLE_TO_FIELD)}"
        )
    base = load_housing_config(json_file)
    field_overrides = {ROLE_TO_FIELD[role]: val for role, val in overrides.items()}
    # Convert lists → tuples for tuple fields
    for fname in _TUPLE_FIELDS:
        if fname in field_overrides and isinstance(field_overrides[fname], list):
            field_overrides[fname] = tuple(field_overrides[fname])
    new_cfg = dataclasses.replace(base, **field_overrides)
    save_housing_config(json_file, new_cfg)
    logger.info(f"update_housing_config: updated {new_cfg.name!r} in {json_file}")
    return new_cfg


def show_housing_config(config: HousingConfig) -> None:
    """Print a formatted summary of *config* to stdout."""
    print(f"Housing : {config.name}")
    print(f"Formats : {', '.join(config.formats)}")
    print(f"  GR1 current : {config.reference_gr1_current}")
    print(f"  GR2 current : {config.reference_gr2_current}")
    print(f"  GR1 flow    : {config.reference_gr1_flow}")
    print(f"  GR2 flow    : {config.reference_gr2_flow}")
    print(f"  GR1 rpm     : {config.reference_gr1_rpm}")
    print(f"  GR2 rpm     : {config.reference_gr2_rpm}")
    print(f"  GR1 pin     : {config.reference_gr1_pin}")
    print(f"  GR2 pin     : {config.reference_gr2_pin}")
    print(f"  GR1 voltages: {', '.join(config.voltage_channels_gr1)}")
    print(f"  GR2 voltages: {', '.join(config.voltage_channels_gr2)}")
    print(f"  GR1 voltage target : {config.reference_gr1_voltage}")
    print(f"  GR2 voltage target : {config.reference_gr2_voltage}")
    if config.pupitre_formula_map:
        print(f"  Pupitre formulas   : {list(config.pupitre_formula_map)}")
    if config.pigbrother_formula_map:
        print(f"  Pigbrother formulas: {list(config.pigbrother_formula_map)}")
    if config.hybrid_formula_map:
        print(f"  Hybrid formulas: {list(config.hybrid_formula_map)}")


# ---------------------------------------------------------------------------
# Pre-defined housing configurations (built-in defaults)
# ---------------------------------------------------------------------------

_PIGBROTHER_FORMULA_MAP = {
    "Courants_Alimentations/Référence_GR1": (
        "Courants_Alimentations/Référence_GR1 = Référence_A1 + Référence_A2"
    ),
    "Courants_Alimentations/Référence_GR2": (
        "Courants_Alimentations/Référence_GR2 = Référence_A3 + Référence_A4"
    ),
}
_HYBRID_FORMULA_MAP = {
    "FEPC-AUX-LNCMI/ALIM1": ("FEPC-AUX-LNCMI/ALIM1_J1 + FEPC-AUX-LNCMI/ALIM1_J2"),
    "FEPC-AUX-LNCMI/ALIM2": ("FEPC-AUX-LNCMI/ALIM2_J1 + FEPC-AUX-LNCMI/ALIM2_J2"),
}

HOUSING_CONFIGS: dict[str, HousingConfig] = {
    "M9": HousingConfig(
        name="M9",
        formats=("pupitre", "pigbrother"),
        reference_gr1_current="IH",
        reference_gr2_current="IB",
        reference_gr1_flow="FlowH",
        reference_gr2_flow="FlowB",
        reference_gr1_rpm="RpmH",
        reference_gr2_rpm="RpmB",
        reference_gr1_pin="HPH",
        reference_gr2_pin="HPB",
        voltage_channels_gr1=(
            "Ucoil1",
            "Ucoil2",
            "Ucoil3",
            "Ucoil4",
            "Ucoil5",
            "Ucoil6",
            "Ucoil7",
            "Ucoil8",
            "Ucoil9",
            "Ucoil10",
            "Ucoil11",
            "Ucoil12",
            "Ucoil13",
            "Ucoil14",
        ),
        voltage_channels_gr2=("Ucoil15", "Ucoil16"),
        reference_gr1_voltage="UH",
        reference_gr2_voltage="UB",
        pupitre_formula_map={
            "IH_ref": "IH_ref = Idcct1 + Idcct2",
            "IB_ref": "IB_ref = Idcct3 + Idcct4",
        },
        pigbrother_formula_map=_PIGBROTHER_FORMULA_MAP,
    ),
    "M8": HousingConfig(
        name="M8",
        formats=("pupitre", "pigbrother", "hybrid"),
        reference_gr1_current="IB",
        reference_gr2_current="IH",
        reference_gr1_flow="FlowB",
        reference_gr2_flow="FlowH",
        reference_gr1_rpm="RpmB",
        reference_gr2_rpm="RpmH",
        reference_gr1_pin="HPB",
        reference_gr2_pin="HPH",
        voltage_channels_gr1=("Ucoil15", "Ucoil16"),
        voltage_channels_gr2=(
            "Ucoil1",
            "Ucoil2",
            "Ucoil3",
            "Ucoil4",
            "Ucoil5",
            "Ucoil6",
            "Ucoil7",
            "Ucoil8",
            "Ucoil9",
            "Ucoil10",
            "Ucoil11",
            "Ucoil12",
            "Ucoil13",
            "Ucoil14",
        ),
        reference_gr1_voltage="UB",
        reference_gr2_voltage="UH",
        pupitre_formula_map={
            "IH_ref": "IH_ref = Idcct3 + Idcct4",
            "IB_ref": "IB_ref = Idcct1 + Idcct2",
        },
        pigbrother_formula_map=_PIGBROTHER_FORMULA_MAP,
        hybrid_formula_map=_HYBRID_FORMULA_MAP,
    ),
    "M10": HousingConfig(
        name="M10",
        formats=("pupitre", "pigbrother"),
        reference_gr1_current="IB",
        reference_gr2_current="IH",
        reference_gr1_flow="FlowB",
        reference_gr2_flow="FlowH",
        reference_gr1_rpm="RpmB",
        reference_gr2_rpm="RpmH",
        reference_gr1_pin="HPB",
        reference_gr2_pin="HPH",
        voltage_channels_gr1=("Ucoil15", "Ucoil16"),
        voltage_channels_gr2=(
            "Ucoil1",
            "Ucoil2",
            "Ucoil3",
            "Ucoil4",
            "Ucoil5",
            "Ucoil6",
            "Ucoil7",
            "Ucoil8",
            "Ucoil9",
            "Ucoil10",
            "Ucoil11",
            "Ucoil12",
            "Ucoil13",
            "Ucoil14",
        ),
        reference_gr1_voltage="UB",
        reference_gr2_voltage="UH",
        pupitre_formula_map={
            "IH_ref": "IH_ref = Idcct3 + Idcct4",
            "IB_ref": "IB_ref = Idcct1 + Idcct2",
        },
        pigbrother_formula_map=_PIGBROTHER_FORMULA_MAP,
    ),
}
"""Built-in default sensor role assignments per housing."""


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def get_housing_config(
    housing: str,
    json_file: str | Path | None = None,
    overrides: dict | None = None,
) -> HousingConfig:
    """Return the :class:`HousingConfig` for *housing*.

    Resolution order:

    1. If *json_file* is given, load the base config from that file.
    2. If ``~/.config/magnetrun/<Housing>-housing-config.json`` exists, load it.
    3. Otherwise use the built-in default from :data:`HOUSING_CONFIGS`.
    4. Apply *overrides* on top (if any).

    Parameters
    ----------
    housing:
        Housing identifier, e.g. ``"M8"``, ``"M9"``, ``"M10"``.
        Used only for validation when *json_file* is not given.
    json_file:
        Optional path to a ``<Housing>-housing-config.json`` file.
        When provided, *housing* is verified against the ``"name"`` field
        in the file.
    overrides:
        Optional ``{role: value}`` dict for runtime adjustments.
        Keys are role names from :data:`ROLE_TO_FIELD`.

    Returns
    -------
    HousingConfig
        Immutable config, possibly loaded from file and/or overridden.

    Raises
    ------
    ValueError
        If *housing* is unknown (no file given), the file's ``name`` field
        does not match *housing*, or an override key is not a valid role.

    Examples
    --------
    ::

        # Built-in default
        cfg = get_housing_config("M9")

        # From a custom file
        cfg = get_housing_config("M9", json_file="my-M9-config.json")

        # Atypical run: GR1/GR2 swapped
        cfg = get_housing_config("M9", overrides={"gr1_current": "IB",
                                                   "gr2_current": "IH"})
    """
    if json_file is not None:
        base = load_housing_config(json_file)
        if base.name != housing:
            raise ValueError(
                f"get_housing_config: file {json_file!r} has name={base.name!r}, "
                f"expected {housing!r}"
            )
    else:
        # Check user config dir before falling back to hardcoded defaults.
        user_file = _USER_CONFIG_DIR / f"{housing}-housing-config.json"
        if user_file.exists():
            logger.debug(f"get_housing_config: loading user config {user_file}")
            base = load_housing_config(user_file)
        elif housing not in HOUSING_CONFIGS:
            raise ValueError(
                f"Unknown housing: {housing!r}. Available: {sorted(HOUSING_CONFIGS)}"
            )
        else:
            base = HOUSING_CONFIGS[housing]

    if not overrides:
        return base

    invalid = set(overrides) - set(ROLE_TO_FIELD)
    if invalid:
        raise ValueError(
            f"get_housing_config: unknown override role(s): {sorted(invalid)}. "
            f"Valid roles: {sorted(ROLE_TO_FIELD)}"
        )
    field_overrides = {ROLE_TO_FIELD[role]: val for role, val in overrides.items()}
    for fname in _TUPLE_FIELDS:
        if fname in field_overrides and isinstance(field_overrides[fname], list):
            field_overrides[fname] = tuple(field_overrides[fname])
    logger.info(f"get_housing_config: {housing!r} with overrides {overrides}")
    return dataclasses.replace(base, **field_overrides)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the ``magnetrun-housing-config`` command."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="magnetrun-housing-config",
        description="Manage <Housing>-housing-config.json housing configuration files.",
    )
    parser.add_argument(
        "json_file",
        help="Path to the <Housing>-housing-config.json file",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # show
    sub.add_parser("show", help="Print the configuration in the file")

    # create
    p_create = sub.add_parser("create", help="Create a new housing config file")
    p_create.add_argument("housing", help="Housing name (e.g. M9)")
    p_create.add_argument(
        "--from-builtin",
        metavar="HOUSING",
        default=None,
        help="Initialise from a built-in config (e.g. --from-builtin M9)",
    )

    # update
    p_upd = sub.add_parser("update", help="Update one or more role fields")
    for role in sorted(ROLE_TO_FIELD):
        p_upd.add_argument(
            f"--{role.replace('_', '-')}",
            dest=role,
            default=None,
            metavar="VALUE",
            help=f"New value for role '{role}'",
        )

    args = parser.parse_args()

    if args.command == "show":
        cfg = load_housing_config(args.json_file)
        show_housing_config(cfg)

    elif args.command == "create":
        if args.from_builtin:
            if args.from_builtin not in HOUSING_CONFIGS:
                parser.error(
                    f"Unknown built-in housing: {args.from_builtin!r}. "
                    f"Available: {sorted(HOUSING_CONFIGS)}"
                )
            base = HOUSING_CONFIGS[args.from_builtin]
            cfg = dataclasses.replace(base, name=args.housing)
        else:
            cfg = HousingConfig(name=args.housing)
        dest = Path(args.json_file)
        save_housing_config(dest, cfg)
        print(f"Created {dest}")
        user_path = get_user_housing_config_path(args.housing)
        if not user_path.exists() and dest != user_path:
            print(
                f"Tip: copy to {user_path} so 'get_housing_config({args.housing!r})' "
                "picks it up automatically without specifying a path."
            )

    elif args.command == "update":
        role_overrides = {
            role: getattr(args, role)
            for role in ROLE_TO_FIELD
            if getattr(args, role) is not None
        }
        if not role_overrides:
            parser.error("No fields to update — specify at least one --<role> option")
        cfg = update_housing_config(args.json_file, role_overrides)
        print(f"Updated {args.json_file}")
        show_housing_config(cfg)


if __name__ == "__main__":
    main()
