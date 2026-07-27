#!/usr/bin/env python3
"""Example: show pupitre <-> pigbrother field correspondence for a housing.

Combines the two layers the package uses to relate field names across
acquisition systems:

1. Housing-independent aliases (the ``"aliases"`` key in ``*-defs.json``) —
   fixed name correspondences such as ``Idcct1`` <-> ``Courants_Alimentations/Courant_A1``.
2. Housing-dependent GR role assignments (:mod:`python_magnetrun.housing_config`) —
   which pupitre field plays each GR role for a given housing, e.g. GR1
   current is ``IH`` on M9 but ``IB`` on M8/M10.

No data files are required — the correspondence is a property of the
configuration, not of any particular run.

Usage
-----
::

    python field_correspondence_example.py --housing M9
    python field_correspondence_example.py --housing M8
    python field_correspondence_example.py --housing M10
"""

from __future__ import annotations

import argparse
import sys

from tabulate import tabulate

from python_magnetrun.field_defs import build_crossref, get_aliases
from python_magnetrun.housing_config import HousingConfig, get_housing_config

_NOT_RECORDED = "— (not recorded by pigbrother)"
_NO_ALIAS = "—"

# TDMS reference-channel keys used throughout housing_config.py's role lookups.
_GR_REFERENCE_KEYS = (("GR1", "Référence_GR1"), ("GR2", "Référence_GR2"))


def print_fixed_aliases() -> None:
    """Print every pupitre field with a fixed (housing-independent) pigbrother alias.

    Uses :func:`~python_magnetrun.field_defs.build_crossref` over the bundled
    ``pupitre-defs.json`` and ``pigbrother-defs.json`` files.
    """
    index = build_crossref(
        {"pupitre": "pupitre-defs.json", "pigbrother": "pigbrother-defs.json"}
    )
    rows = [
        (field, aliases["pigbrother"])
        for field, aliases in sorted(index["pupitre"].items())
        if "pigbrother" in aliases
    ]
    print("\n=== Fixed field aliases (housing-independent) ===")
    print(tabulate(rows, headers=["Pupitre field", "Pigbrother key"], tablefmt="simple"))


def print_role_table(cfg: HousingConfig) -> None:
    """Print GR-role correspondence (current, voltage, flow, rpm, pin) for *cfg*.

    Parameters
    ----------
    cfg : HousingConfig
        Housing configuration to resolve role fields from.
    """
    rows = []
    for gr, ref_key in _GR_REFERENCE_KEYS:
        current_field = cfg.get_pupitre_channel(ref_key)
        rows.append(
            (gr, "current", current_field, f"Courants_Alimentations/{ref_key}")
        )

        voltage_field = (
            cfg.reference_gr1_voltage
            if ref_key == "Référence_GR1"
            else cfg.reference_gr2_voltage
        )
        pigbrother_voltage = get_aliases("pupitre-defs.json", voltage_field).get(
            "pigbrother", _NO_ALIAS
        )
        rows.append((gr, "voltage (sum)", voltage_field, pigbrother_voltage))

        rows.append((gr, "flow", cfg.get_flow_channel(ref_key), _NOT_RECORDED))
        rows.append((gr, "rpm", cfg.get_rpm_channel(ref_key), _NOT_RECORDED))
        rows.append((gr, "inlet pressure", cfg.get_pin_channel(ref_key), _NOT_RECORDED))

    print(f"\n=== Housing-dependent role assignments ({cfg.name}) ===")
    print(
        tabulate(
            rows,
            headers=["GR", "Role", "Pupitre field", "Pigbrother key"],
            tablefmt="simple",
        )
    )


def print_voltage_probe_table(cfg: HousingConfig) -> None:
    """Print per-probe Ucoil <-> Tensions_Aimant correspondence grouped by GR.

    Parameters
    ----------
    cfg : HousingConfig
        Housing configuration providing ``voltage_channels_gr1``/``gr2``.
    """
    rows = []
    for gr, channels in (("GR1", cfg.voltage_channels_gr1), ("GR2", cfg.voltage_channels_gr2)):
        for channel in channels:
            pigbrother_channel = get_aliases("pupitre-defs.json", channel).get(
                "pigbrother", _NO_ALIAS
            )
            rows.append((gr, channel, pigbrother_channel))

    print(f"\n=== Voltage probe channels by GR ({cfg.name}) ===")
    print(tabulate(rows, headers=["GR", "Pupitre field", "Pigbrother key"], tablefmt="simple"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--housing", default="M9", help="housing name, e.g. M8, M9, M10 (default: M9)"
    )
    args = parser.parse_args()

    cfg = get_housing_config(args.housing)

    print(f"Housing : {cfg.name}")
    print(f"Formats : {', '.join(cfg.formats)}")
    if "pigbrother" not in cfg.formats:
        print(f"Warning: {cfg.name!r} does not list 'pigbrother' among its formats")

    print_fixed_aliases()
    print_role_table(cfg)
    print_voltage_probe_table(cfg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
