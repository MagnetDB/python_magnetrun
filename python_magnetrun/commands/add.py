"""Add command: add computed or formula-based fields to MagnetRun data."""

import argparse
import logging

from ..commands.plot import _resolve_plot_config, _save_or_show_figure
from ..magnetdata_base import DataType
from ..plotting.backend import get_backend

logger = logging.getLogger(__name__)


def _plot_fields(mdata, fields: list[str], title: str, args, cfg) -> None:
    """Plot *fields* from *mdata* using the configured backend."""
    backend_name = getattr(args, "backend", "matplotlib")
    b = get_backend(backend_name)
    normalize = getattr(args, "normalize", False)

    fig = b.subplots(1, share_x=False, style=cfg.style)
    logger.debug(f"mdata: filename={mdata.FileName!r}, type={mdata.Type}")

    for field in fields:
        try:
            tkey = "t"
            logger.debug(f"initial tkey={tkey!r} for field {field!r}")
            if mdata.Type == DataType.TDMS:
                logger.debug(f"tkey={tkey!r} for field {field!r}")
                tkey = f"{field.split('/')[0]}/t"

            logger.debug(
                f"try to load field {field!r} with tkey={tkey!r} for mdata type {mdata.Type}"
            )
            df = mdata.getData([tkey, field])
            logger.debug(
                f"loaded field {field!r} with tkey={tkey!r}, df columns={df.columns}"
            )
            logger.debug(f"tkey={tkey!r}, key={field!r}")
        except (KeyError, RuntimeError) as e:
            logger.error(f"could not load field {field!r}: {e}")
            continue
        t = df[tkey.split("/")[-1]].to_numpy(dtype=float)
        y = df[field.split("/")[-1]].to_numpy(dtype=float)
        try:
            symbol, unit = mdata.getUnitKey(field)
            unit_str = f"{unit:~P}" if unit is not None else "?"
            ylabel = f"{symbol} [{unit_str}]"
        except (KeyError, RuntimeError):
            ylabel = field
        b.add_series(fig, 0, t, y, label=field, normalize=normalize, ylabel=ylabel)

    if title and hasattr(fig, "update_layout"):
        fig.update_layout(title_text=title)
    elif title and hasattr(fig, "suptitle"):
        fig.suptitle(title)

    _save_or_show_figure(fig, args, b, [], fields, backend_name, dpi=cfg.style.dpi)


def add_field(mrun, args):
    """Add computed or formula-based fields to MagnetRun data.

    :param mrun: MagnetRun instance
    :type mrun: MagnetRun
    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    """
    cfg = _resolve_plot_config(args)
    mdata = mrun.getMData()
    logger.debug(mdata.getKeys())

    if args.compute:
        from pint import UnitRegistry  # noqa: I001

        from python_magnetcooling.water_properties import get_rho

        ureg = UnitRegistry()
        nkey_params = ["HPH", "TinH"]

        status = mdata.computeData(
            get_rho,
            "rho",
            nkey_params,
            symbol="rho",
            unit=ureg.kilogram / ureg.meter**3,
            label="Water Density",
            description="Cooling water density",
        )
        if status != 0:
            logger.warning(f"Failed to compute rho (status={status})")
        nkey = "rho"
        logger.debug(mdata.getKeys())
        logger.debug(mdata.getData("rho").describe())

        if args.plot:
            _plot_fields(mdata, [nkey] + list(nkey_params), nkey, args, cfg)

    if args.formula:
        logger.debug(f"add {args.formula}, plot={args.plot}")

        nkey = args.formula.split(" = ")[0]
        nsymbol = getattr(args, "symbol", "") or nkey
        nunit = getattr(args, "unit", "")
        nlabel = getattr(args, "label", "")
        ndescription = getattr(args, "description", "")

        logger.debug(
            f"try to add nkey={nkey}: "
            f"formula={args.formula[1:]},"
            f"symbol={nsymbol!r}, unit={nunit!r}, "
            f"label={nlabel!r}, description={ndescription!r}"
        )
        status = mdata.addData(
            key=nkey,
            formula=args.formula,
            symbol=nsymbol,
            unit=nunit,
            label=nlabel,
            description=ndescription,
        )
        if status != 0:
            logger.warning(f"Failed to add {nkey} (status={status})")
        logger.debug(mdata.getKeys())

        if args.plot:
            extra_fields = []
            if args.vs_time:
                for key in args.vs_time[0]:
                    extra_fields.append(key)
            _plot_fields(mdata, [nkey] + extra_fields, nkey, args, cfg)


def _run(args: "argparse.Namespace") -> int:

    from ..log_utils import setup_logging
    from ._shared import load_inputs

    log_level = getattr(__import__("logging"), getattr(args, "log_level", "WARNING").upper(), 30)
    setup_logging(level=log_level, log_file=getattr(args, "log_file", None))

    input_files, inputs, _extensions = load_inputs(args)
    if not inputs:
        logger.error("No files loaded.")
        return 1

    for file in input_files:
        if file not in inputs:
            continue
        mrun = inputs[file]["data"]
        add_field(mrun, args)

    return 0


def register(sub: "argparse._SubParsersAction") -> None:

    from ..cli_args import (
        create_base_parser,
        create_common_plot_parser,
        create_managed_plots_parser,
    )

    base = create_base_parser(add_input_file=False)
    plot_parser = create_common_plot_parser()
    managed_plots_parser = create_managed_plots_parser()

    p = sub.add_parser(
        "add",
        parents=[base, plot_parser, managed_plots_parser],
        help="add computed or formula-based fields",
    )
    p.add_argument("input_file", nargs="+", help="input file(s)")
    p.add_argument("--formula", type=str, default="", help="formula for the new column")
    p.add_argument("--symbol", type=str, default="", help="physical symbol (e.g. 'B')")
    p.add_argument("--unit", type=str, default="", help="unit string (e.g. 'tesla')")
    p.add_argument("--label", type=str, default="", help="human-readable label")
    p.add_argument("--description", type=str, default="", help="description of the field")
    p.add_argument("--compute", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.set_defaults(_handler=_run)
