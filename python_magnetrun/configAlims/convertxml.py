site_dict = {
    "17": "M7",
    "18": "M8",
    "19": "M9",
    "20": "M10",
}


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="+", help="enter input file xml")
    parser.add_argument(
        "--datadir",
        help="enter datadir (ex srvdata)",
        type=str,
        default=".",
    )
    parser.add_argument("--debug", help="activate debug mode", action="store_true")
    return parser.parse_args()


def main():
    import os
    import json
    import xmltodict

    args = parse_arguments()

    input_files = args.input_file
    datadir = args.datadir

    pwd = os.getcwd()
    if args.datadir != ".":
        os.chdir(datadir)

    # get config from: https://srv-data-install.lncmi.cnrs.fr/cirrus/A1/xml/
    for file in input_files:
        print(file)
        with open(file, "r") as f:
            xml = f.read()
            config_dict = xmltodict.parse(xml)
            if args.debug:
                print(json.dumps(config_dict, indent=4))

        # @version_config
        # @date
        # configuration.CIRRUS_CP.analogique.regs.regulateur
        # @numero: de 1 a 6 (Current, Voltage, Pont1 to Pont4)
        # @numero_jeu: 1 a 20 (eg: M9=19, M10=20, M8=18)
        # @numero_seuil: 1 a 3 (from 0 to 60A, from 60 to 1000A, above see R14L09 p13)
        # get values for @Ki, @Kp, @Kd (any vamues == -1 means 0)
        # what about rapport_entre_boucle (lien entre A1 et A2 - aka GR1 - , lien entre A3 et A4 - aka GR2)?
        version = config_dict["configuration"]["@version_config"]
        date = config_dict["configuration"]["@date"]
        print(f"Version: {version}, Date: {date}")

        config_regs = config_dict["configuration"]["CIRRUS_CP"]["analogique"]["regs"]
        for i, reg_value in enumerate(config_regs["regulateur"]):
            # print(f'value[{i}]={reg_value}, type={type(reg_value)}')
            # for key, values in reg_value.items():
            #    print(key, values)

            numero = reg_value.get("@numero", "unknown")
            numero_jeu = reg_value.get("@numero_jeu", "unknown")
            numero_seuil = reg_value.get("@numero_seuil", "unknown")
            Ki = reg_value.get("@Ki", -1)
            Kp = reg_value.get("@Kp", -1)
            Kd = reg_value.get("@Kd", -1)
            rapport_entre_boucle = reg_value.get("@rapport_entre_boucle", "unknown")
            # print(f'looking for numero={numero} (type={type(numero)})')

            if numero == str(1):
                # print(f'looking for numero_jeu={numero_jeu}, site_dict={site_dict}')
                if numero_jeu in site_dict:
                    print(
                        f"Current PID {site_dict[numero_jeu]}: Numero Seuil: {numero_seuil}, Ki: {Ki}, Kp: {Kp}, Kd: {Kd}, Rapport Entre Boucle: {rapport_entre_boucle}"
                    )

        # configuration.CIRRUS_CP.analogique.rampes.rampe ?? control power shutdown in case of alarm
        # numero
    os.chdir(pwd)


if __name__ == "__main__":
    main()
