"""Add command: add computed or formula-based fields to MagnetRun data."""

import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def add_field(mrun, args):
    """Add computed or formula-based fields to MagnetRun data.

    :param mrun: MagnetRun instance
    :type mrun: MagnetRun
    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    """
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
            my_ax = plt.gca()
            mdata.plotData(x="t", y=nkey, ax=my_ax, normalize=args.normalize)
            for param in nkey_params:
                mdata.plotData(x="t", y=param, ax=my_ax, normalize=args.normalize)

            if not args.save:
                plt.show()
            else:
                imagefile = nkey
                logger.info(f"saveto: {imagefile}_vs_time.png")
                plt.savefig(f"{imagefile}_vs_time.png", dpi=300)
            plt.close()

    if args.formula:
        logger.debug(f"add {args.formula}, plot={args.plot}")

        nkey = args.formula.split(" = ")[0]
        nunit = ""

        # self.units[key] = ("U", ureg.volt)
        logger.debug(f"try to add nkey={nkey} (formula={args.formula[1:]})")
        mdata.addData(key=nkey, formula=args.formula, unit=nunit)
        logger.debug(mdata.getKeys())
        if args.plot:
            my_ax = plt.gca()
            mdata.plotData(x="t", y=nkey, ax=my_ax, normalize=args.normalize)

            logger.debug(f"args.vs_time: {args.vs_time}")
            if args.vs_time:
                for key in args.vs_time[0]:
                    logger.debug(key)
                    mdata.plotData(x="t", y=key, ax=my_ax, normalize=args.normalize)

            if not args.save:
                plt.show()
            else:
                imagefile = nkey
                logger.info(f"saveto: {imagefile}_vs_time.png")
                plt.savefig(f"{imagefile}_vs_time.png", dpi=300)
            plt.close()
