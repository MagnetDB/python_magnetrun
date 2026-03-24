"""
Field name correspondence between pupitre (.txt), pigbrother (.tdms) and hybrid data.

The mapping between IH/IB (pupitre) and GR1/GR2 (TDMS/hybrid) depends on the
**housing** (M8, M9, M10), not the site in the magnetdb sense:

  M9  : GR1 = IH,  GR2 = IB
  M8  : GR1 = IB,  GR2 = IH   (swapped)
  M10 : GR1 = IB,  GR2 = IH   (same as M8)

All housing-dependent dicts can be overridden at runtime by passing an `overrides`
dict to the helper functions.

Hybrid key format: "type/FEPC_system/variable"
  e.g. "kHz/FEPC-LNCMI/I_H1", "kHz/FEPC-AUX-LNCMI/ALIM1_J1"

TDMS key format: "Group/Channel"
  e.g. "Courants_Alimentations/Référence_GR1"
"""

from typing import Dict, Optional

# ---------------------------------------------------------------------------
# pupitre field name -> TDMS field key, keyed by housing
# Extend or override per housing as needed.
# ---------------------------------------------------------------------------

PUPITRE_TO_TDMS_BY_HOUSING: Dict[str, Dict[str, str]] = {
    "M9": {
        # currents
        "IH":  "Courants_Alimentations/Référence_GR1",
        "IB":  "Courants_Alimentations/Référence_GR2",
        # voltages — TODO: verify exact TDMS channel names against real M9 TDMS files
        "UH":  "Tensions_Alimentations/Tension_GR1",
        "UB":  "Tensions_Alimentations/Tension_GR2",
        # internal coil voltages — TODO: verify Group/Channel names for Ucoil15/16 in M9
        "Ucoil15": "Tensions_Aimant/ALL_internes",
        "Ucoil16": "Tensions_Aimant/ALL_internes",
    },
    "M8": {
        # GR1/GR2 are swapped relative to M9
        "IH":  "Courants_Alimentations/Référence_GR2",
        "IB":  "Courants_Alimentations/Référence_GR1",
        # voltages — TODO: verify exact TDMS channel names against real M8 TDMS files
        "UH":  "Tensions_Alimentations/Tension_GR2",
        "UB":  "Tensions_Alimentations/Tension_GR1",
        # TODO: verify Group/Channel names for Ucoil15/16 in M8
        "Ucoil15": "Tensions_Aimant/ALL_externes",
        "Ucoil16": "Tensions_Aimant/ALL_externes",
    },
    "M10": {
        # Same convention as M8
        "IH":  "Courants_Alimentations/Référence_GR2",
        "IB":  "Courants_Alimentations/Référence_GR1",
        # voltages — TODO: verify exact TDMS channel names against real M10 TDMS files
        "UH":  "Tensions_Alimentations/Tension_GR2",
        "UB":  "Tensions_Alimentations/Tension_GR1",
        # TODO: verify Group/Channel names for Ucoil15/16 in M10
        "Ucoil15": "Tensions_Aimant/ALL_externes",
        "Ucoil16": "Tensions_Aimant/ALL_externes",
    },
}

# ---------------------------------------------------------------------------
# pupitre GR role per housing: which pupitre field plays GR1 / GR2
# ---------------------------------------------------------------------------

PUPITRE_FIELD_BY_HOUSING: Dict[str, Dict[str, str]] = {
    "M9":  {"GR1": "IH", "GR2": "IB", "VoltGR1": "UH", "VoltGR2": "UB"},
    "M8":  {"GR1": "IB", "GR2": "IH", "VoltGR1": "UB", "VoltGR2": "UH"},
    "M10": {"GR1": "IB", "GR2": "IH", "VoltGR1": "UB", "VoltGR2": "UH"},
}

# ---------------------------------------------------------------------------
# hybrid key -> pupitre field, per FEPC system.
# GR1 is always the "H" supply in FEPC naming; adapt if wiring differs.
# Override with get_hybrid_to_pupitre() when housing convention differs.
# ---------------------------------------------------------------------------

HYBRID_TO_PUPITRE_BY_HOUSING: Dict[str, Dict[str, str]] = {
    "M9": {
        "kHz/FEPC-LNCMI/I_H1":          "IH",
        "kHz/FEPC-LNCMI/I_B1":          "IB",
        "kHz/FEPC-AUX-LNCMI/ALIM1_J1":  "IH",
        "kHz/FEPC-AUX-LNCMI/ALIM1_J2":  "IB",
        "rms/FEPC-LNCMI/I_H1":           "IH",
        "rms/FEPC-LNCMI/I_B1":           "IB",
        "rms/FEPC-AUX-LNCMI/ALIM1_J1":  "IH",
        "rms/FEPC-AUX-LNCMI/ALIM1_J2":  "IB",
    },
    "M8": {
        "kHz/FEPC-LNCMI/I_H1":          "IB",   # swapped
        "kHz/FEPC-LNCMI/I_B1":          "IH",
        "kHz/FEPC-AUX-LNCMI/ALIM1_J1":  "IB",
        "kHz/FEPC-AUX-LNCMI/ALIM1_J2":  "IH",
        "rms/FEPC-LNCMI/I_H1":           "IB",
        "rms/FEPC-LNCMI/I_B1":           "IH",
        "rms/FEPC-AUX-LNCMI/ALIM1_J1":  "IB",
        "rms/FEPC-AUX-LNCMI/ALIM1_J2":  "IH",
    },
    "M10": {
        "kHz/FEPC-LNCMI/I_H1":          "IB",
        "kHz/FEPC-LNCMI/I_B1":          "IH",
        "kHz/FEPC-AUX-LNCMI/ALIM1_J1":  "IB",
        "kHz/FEPC-AUX-LNCMI/ALIM1_J2":  "IH",
        "rms/FEPC-LNCMI/I_H1":           "IB",
        "rms/FEPC-LNCMI/I_B1":           "IH",
        "rms/FEPC-AUX-LNCMI/ALIM1_J1":  "IB",
        "rms/FEPC-AUX-LNCMI/ALIM1_J2":  "IH",
    },
}

# ---------------------------------------------------------------------------
# hybrid key -> TDMS key (Group/Channel).
# Independent of housing: GR1/GR2 naming is consistent in TDMS.
# TODO: verify channel names once real TDMS + hybrid files are available together.
# ---------------------------------------------------------------------------

HYBRID_TO_TDMS: Dict[str, str] = {
    "kHz/FEPC-LNCMI/I_H1":          "Courants_Alimentations/Référence_GR1",
    "kHz/FEPC-LNCMI/I_B1":          "Courants_Alimentations/Référence_GR2",
    "kHz/FEPC-AUX-LNCMI/ALIM1_J1":  "Courants_Alimentations/Référence_GR1",
    "kHz/FEPC-AUX-LNCMI/ALIM1_J2":  "Courants_Alimentations/Référence_GR2",
    "rms/FEPC-LNCMI/I_H1":           "Courants_Alimentations/Référence_GR1",
    "rms/FEPC-LNCMI/I_B1":           "Courants_Alimentations/Référence_GR2",
    "rms/FEPC-AUX-LNCMI/ALIM1_J1":  "Courants_Alimentations/Référence_GR1",
    "rms/FEPC-AUX-LNCMI/ALIM1_J2":  "Courants_Alimentations/Référence_GR2",
}

# ---------------------------------------------------------------------------
# Helper functions (support runtime overrides)
# ---------------------------------------------------------------------------


def get_pupitre_to_tdms(
    housing: str,
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Return the pupitre->TDMS field mapping for a given housing.

    :param housing: Housing name (e.g. "M8", "M9", "M10")
    :param overrides: Optional dict merged on top of the base mapping,
                      useful for non-standard channel wiring.
    :return: Dict mapping pupitre field names to TDMS Group/Channel keys.
    :raises KeyError: If housing is not in PUPITRE_TO_TDMS_BY_HOUSING.
    """
    if housing not in PUPITRE_TO_TDMS_BY_HOUSING:
        raise KeyError(
            f"Unknown housing '{housing}'. "
            f"Known housings: {list(PUPITRE_TO_TDMS_BY_HOUSING)}"
        )
    base = dict(PUPITRE_TO_TDMS_BY_HOUSING[housing])
    if overrides:
        base.update(overrides)
    return base


def get_pupitre_fields(
    housing: str,
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Return the GR-role -> pupitre field mapping for a given housing.

    :param housing: Housing name (e.g. "M8", "M9", "M10")
    :param overrides: Optional dict merged on top of the base mapping.
    :return: Dict with keys "GR1", "GR2", "VoltGR1", "VoltGR2".
    :raises KeyError: If housing is not in PUPITRE_FIELD_BY_HOUSING.
    """
    if housing not in PUPITRE_FIELD_BY_HOUSING:
        raise KeyError(
            f"Unknown housing '{housing}'. "
            f"Known housings: {list(PUPITRE_FIELD_BY_HOUSING)}"
        )
    base = dict(PUPITRE_FIELD_BY_HOUSING[housing])
    if overrides:
        base.update(overrides)
    return base


def get_hybrid_to_pupitre(
    housing: str,
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Return the hybrid->pupitre field mapping for a given housing.

    :param housing: Housing name (e.g. "M8", "M9", "M10")
    :param overrides: Optional dict merged on top of the base mapping.
    :return: Dict mapping hybrid key paths to pupitre field names.
    :raises KeyError: If housing is not in HYBRID_TO_PUPITRE_BY_HOUSING.
    """
    if housing not in HYBRID_TO_PUPITRE_BY_HOUSING:
        raise KeyError(
            f"Unknown housing '{housing}'. "
            f"Known housings: {list(HYBRID_TO_PUPITRE_BY_HOUSING)}"
        )
    base = dict(HYBRID_TO_PUPITRE_BY_HOUSING[housing])
    if overrides:
        base.update(overrides)
    return base


def get_hybrid_to_tdms(
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Return the hybrid->TDMS field mapping (housing-independent).

    :param overrides: Optional dict merged on top of the base mapping.
    :return: Dict mapping hybrid key paths to TDMS Group/Channel keys.
    """
    base = dict(HYBRID_TO_TDMS)
    if overrides:
        base.update(overrides)
    return base
