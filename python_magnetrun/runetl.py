"""ETL functions for preparing MagnetRun data."""

import logging
import re

from natsort import natsorted

from .magnetdata_base import DataType, MagnetDataBase

logger = logging.getLogger(__name__)


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

    df = data.getData()

    # Drop all-zero columns (skip Flow* and Field*)
    zero_cols = [
        col
        for col in df.columns
        if (df[col] == 0).all()
        and not col.startswith("Flow")
        and not col.startswith("Field")
        and not col.startswith("Idcct")
    ]
    if zero_cols:
        logger.warning(
            f"{data.__class__.__name__}: dropping zero columns {zero_cols} from {data.FileName!r}"
        )
        data.removeData(zero_cols)

    # Resolve duplicate Icoil columns
    Ikeys: list = natsorted([k for k in data.getKeys() if re.match(r"Icoil\d+", k)])
    if len(Ikeys) > 2:
        ikeys_df = data.getData(Ikeys)
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


def prepareData(
    data: MagnetDataBase,
    housing: str,
    keys_to_remove: list[str] | None = None,
    keys_to_rename: dict[str, str] | None = None,
    keys_to_add: dict[str, str] | None = None,
    debug: bool = False,
) -> None:
    """Prepare magnet run data by adding computed fields and renaming columns.

    When *keys_to_add* and *keys_to_rename* are both ``None`` the ETL maps are
    derived automatically from the :class:`~python_magnetrun.housing_config.HousingConfig`
    for the given *housing*:

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

    cfg = get_housing_config(housing)

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
    _duration = data.getDuration()

    data.cleanupData(
        keys_to_remove=keys_to_remove,
        keys_to_rename=keys_to_rename,
        keys_to_add=keys_to_add,
        debug=debug,
    )

    if data.Type == DataType.PUPITRE:
        _cleanup_pupitre_icoil(data, cfg)

    logger.debug(f"MagnetRun.prepareData: data.keys={data.getKeys()}")
