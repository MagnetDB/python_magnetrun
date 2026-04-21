"""Add command: add computed or formula-based fields to MagnetRun data."""

import logging

from ..commands.plot import _handle_output, _resolve_plot_config
from ..plotting.backend import get_backend

logger = logging.getLogger(__name__)


def _plot_fields(mdata, fields: list[str], title: str, args, cfg) -> None:
    """Plot *fields* from *mdata* using the configured backend."""
    backend_name = getattr(args, "backend", "matplotlib")
    b = get_backend(backend_name)
    normalize = getattr(args, "normalize", False)

    fig = b.subplots(1, share_x=False, style=cfg.style)

    for field in fields:
        try:
            df = mdata.getData(["t", field])
        except (KeyError, RuntimeError) as e:
            logger.error(f"could not load field {field!r}: {e}")
            continue
        t = df["t"].to_numpy(dtype=float)
        y = df[field].to_numpy(dtype=float)
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

    _handle_output(fig, args, b, [], fields, backend_name, dpi=cfg.style.dpi)


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
        nkey = "rho"
        nkey_unit = ("rho", ureg.kilogram / ureg.meter**3)
        nkey_params = ["HPH", "TinH"]
        nkey_method = get_rho

        mdata.computeData(nkey_method, nkey, nkey_params, nkey_unit)
        logger.debug(mdata.getKeys())
        logger.debug(mdata.getData("rho").describe())

        if args.plot:
            _plot_fields(mdata, [nkey] + list(nkey_params), nkey, args, cfg)

    if args.formula:
        logger.debug(f"add {args.formula}, plot={args.plot}")

        nkey = args.formula.split(" = ")[0]
        nunit = ""

        logger.debug(f"try to add nkey={nkey} (formula={args.formula[1:]})")
        mdata.addData(key=nkey, formula=args.formula, unit=nunit)
        logger.debug(mdata.getKeys())

        if args.plot:
            extra_fields = []
            if args.vs_time:
                for key in args.vs_time[0]:
                    extra_fields.append(key)
            _plot_fields(mdata, [nkey] + extra_fields, nkey, args, cfg)
