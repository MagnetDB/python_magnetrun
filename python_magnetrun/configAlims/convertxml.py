import argparse
import logging

logger = logging.getLogger(__name__)

site_dict = {
    "17": "M7",
    "18": "M8",
    "19": "M9",
    "20": "M10",
}


def _populate_alim_parser(p: argparse.ArgumentParser) -> None:
    """Add alim arguments to *p*."""
    p.add_argument("input_file", nargs="+", help="XML config file(s) from CIRRUS")
    p.add_argument(
        "--datadir",
        help="working directory (default: current)",
        type=str,
        default=".",
    )
    p.add_argument("--debug", help="activate debug mode", action="store_true")


def _run_alim(args: argparse.Namespace) -> int:
    """Execute the alim subcommand."""
    import json
    import os

    import xmltodict

    pwd = os.getcwd()
    if args.datadir != ".":
        os.chdir(args.datadir)

    for file in args.input_file:
        logger.info(f"{file}")
        with open(file) as f:
            xml = f.read()
            config_dict = xmltodict.parse(xml)
            if args.debug:
                logger.debug(json.dumps(config_dict, indent=4))

        version = config_dict["configuration"]["@version_config"]
        date = config_dict["configuration"]["@date"]
        logger.info(f"Version: {version}, Date: {date}")

        config_regs = config_dict["configuration"]["CIRRUS_CP"]["analogique"]["regs"]
        for _i, reg_value in enumerate(config_regs["regulateur"]):
            numero = reg_value.get("@numero", "unknown")
            numero_jeu = reg_value.get("@numero_jeu", "unknown")
            numero_seuil = reg_value.get("@numero_seuil", "unknown")
            Ki = reg_value.get("@Ki", -1)
            Kp = reg_value.get("@Kp", -1)
            Kd = reg_value.get("@Kd", -1)
            rapport_entre_boucle = reg_value.get("@rapport_entre_boucle", "unknown")

            if numero == str(1) and numero_jeu in site_dict:
                logger.info(
                    f"Current PID {site_dict[numero_jeu]}: Numero Seuil: {numero_seuil}, Ki: {Ki}, Kp: {Kp}, Kd: {Kd}, Rapport Entre Boucle: {rapport_entre_boucle}"
                )

        # configuration.CIRRUS_CP.analogique.rampes.rampe ?? control power shutdown in case of alarm
    os.chdir(pwd)
    return 0


def register(sub: "argparse._SubParsersAction") -> None:
    """Register the ``alim`` domain into a parent subparsers group."""
    p = sub.add_parser("alim", help="inspect CIRRUS XML alim config files")
    _populate_alim_parser(p)
    p.set_defaults(_domain_handler=_run_alim)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _populate_alim_parser(parser)
    return parser.parse_args()


def main() -> None:
    import sys

    args = parse_arguments()
    sys.exit(_run_alim(args))


if __name__ == "__main__":
    main()
