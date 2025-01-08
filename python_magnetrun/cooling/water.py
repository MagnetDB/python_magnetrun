r"""
Provides methods to compute water properties
and Dimensionless numbers for cooling flow

see report from E3 students on gDrive
https://docs.google.com/document/d/1B1nQD_1XNmJza03_Z_OsNWEXZztIjap9FPjAimKZWpA/edit
"""

use_freesteam = True
try:
    import freesteam as st
except:
    from iapws import IAPWS97

    """
    P: Pressure, [MPa]
    T: Temperature, [K]
    cp: Specific isobaric heat capacity, [kJ/kg·K]
    rho: Density, [kg/m³]
    mu: Dynamic viscosity, [Pa·s]
    k: Thermal conductivity, [W/m·K]
    water=IAPWS97(T=15.2+273.15, P=0.101325)
    print(water.cp, water.rho, water.mu, water.k)
    """
    print("!!! Using IAWPS instead of freesteam")
    use_freesteam = False


def getRho(pbar, celsius) -> float:
    """
    compute water volumic mass as a function
    of pressure and temperature
    """

    rho = None
    kelvin = celsius + 273.0

    if use_freesteam:
        pascal = pbar * 1e5
        rho = st.steam_pT(pascal, kelvin).rho
    else:
        mpa = pbar * 1e-1
        rho = IAPWS97(T=kelvin, P=mpa).rho
        # rho = state.rho
    # print("rho(%g,%g)=%g" % (pbar,celsius,rho))
    return rho


def getCp(pbar, celsius) -> float:
    """
    compute water volumic specific heat as a function
    of pressure and temperature
    """

    cp = None
    kelvin = celsius + 273.0

    if use_freesteam:
        pascal = pbar * 1e5
        cp = st.steam_pT(pascal, kelvin).cp
    else:
        mpa = pbar * 1e-1
        state = IAPWS97(T=kelvin, P=mpa)
        cp = state.cp * 1.0e3
    # print("cp(%g,%g)=%g" % (pbar,celsius,cp))
    return cp


def getK(pbar, celsius) -> float:
    """
    compute water thermal conductivity as a function
    of pressure and temperature
    """

    k = None
    kelvin = celsius + 273.0

    if use_freesteam:
        pascal = pbar * 1e5
        k = st.steam_pT(pascal, kelvin).k
    else:
        mpa = pbar * 1e-1
        state = IAPWS97(T=kelvin, P=mpa)
        k = state.k
    # print("cp(%g,%g)=%g" % (pbar,celsius,cp))
    return k


def getMu(pbar, celsius) -> float:
    """
    compute water dynamic viscosity as a function
    of pressure and temperature
    """

    mu = None
    kelvin = celsius + 273.0
    if use_freesteam:
        pascal = pbar * 1e5
        mu = st.steam_pT(pascal, kelvin).mu
    else:
        mpa = pbar * 1e-1
        mu = IAPWS97(T=kelvin, P=mpa).mu
    # print("cp(%g,%g)=%g" % (pbar,celsius,cp))
    return mu


def getNusselt(u: float, d: float, pbar: float, celsius: float, params: list) -> float:
    """
    compute Nusselt

    params [a, b, c]: coefficent for actual correlation
    """

    # HX specific data
    a = params[0]  # 0.207979
    b = params[1]  # 0.640259
    c = params[2]  # 0.397994

    Nusselt = (
        a * pow(getReynolds(u, d, pbar, celsius), b) * pow(getPrandtl(pbar, celsius), c)
    )
    return Nusselt


def getReynolds(u: float, d: float, pbar: float, celsius: float) -> float:
    """
    compute Reynolds

    u: water velocity
    d: hydraulic diameter
    """

    Reynolds = getRho(pbar, celsius) * u * d / getMu(pbar, celsius)
    return Reynolds


def getPrandtl(pbar: float, celsius: float) -> float:
    """
    compute Prandtl
    """

    Prandtl = getMu(pbar, celsius) * getCp(pbar, celsius) / getK(pbar, celsius)
    return Prandtl


def getHeatExchangeCoeff(
    u: float, d: float, pbar: float, celsius: float, params: list
) -> float:
    """
    computes HeatExchangeCoeff
    """

    h = getNusselt(u, d, pbar, celsius, params) * getK(pbar, celsius) / d
    return h


def getOHTC(
    u_h: float,
    u_c: float,
    d: float,
    pbar_h: float,
    celsius_h: float,
    pbar_c: float,
    celsius_c: float,
    params: list,
) -> float:
    """
    computes heat exchange coefficient
    """

    Ohtc = 1.0 / getHeatExchangeCoeff(u_h, d, pbar_h, celsius_h, params)
    Ohtc += 1.0 / getHeatExchangeCoeff(u_c, d, pbar_c, celsius_c, params)

    return 1.0 / Ohtc


if __name__ == "__main__":
    import os
    import sys

    import matplotlib

    # print("matplotlib=", matplotlib.rcParams.keys())
    matplotlib.rcParams["text.usetex"] = True
    # matplotlib.rcParams['text.latex.unicode'] = True key not available
    import matplotlib.pyplot as plt

    from ..MagnetRun import MagnetRun
    from ..processing import smoothers as datatools

    command_line = None

    import argparse

    parser = argparse.ArgumentParser("Cooling loop Heat Exchanger")
    parser.add_argument(
        "input_file", help="input txt file (ex. M10_2020.10.04_20-2009_43_31.txt)"
    )
    parser.add_argument(
        "--debit_alim",
        help="specify flowrate for power cooling (default: 30 m3/h)",
        type=float,
        default=30,
    )
    parser.add_argument(
        "--show",
        help="display graphs (requires X11 server active)",
        action="store_true",
    )
    parser.add_argument("--debug", help="activate debug mode", action="store_true")
    args = parser.parse_args(command_line)
    # check extension
    f_extension = os.path.splitext(args.input_file)[-1]
    if f_extension != ".txt":
        print("so far only txt file support is implemented")
        sys.exit(0)

    filename = os.path.basename(args.input_file)
    result = filename.startswith("M")
    if result:
        try:
            index = filename.index("_")
            args.site = filename[0:index]
            print("site detected: %s" % args.site)
        except:
            print("no site detected - use args.site argument instead")
            # pass

    mrun = MagnetRun.fromtxt(args.site, args.input_file)
    if not args.site:
        args.site = mrun.getSite()

    # Adapt filtering and smoothing params to run duration
    tau = 400
    duration = mrun.getDuration()
    if duration <= 10 * tau:
        tau = min(duration // 10, 10)
        print(f"Modified smoothing param: {tau} over {duration} s run")
        # args.markevery = 2 * tau

    # print("type(mrun):", type(mrun))
    mrun.getMData().addTime()
    start_timestamp = mrun.getMData().getStartDate()

    keys = mrun.getKeys()
    if "HP" not in keys:
        mrun.getMData().addData("HP", "HP = HP1 + HP2")
    if "Flow" not in keys:
        mrun.getMData().addData("Flow", "Flow = Flow1 + Flow2")
    if "Tin" not in keys:
        mrun.getMData().addData("Tin", "Tin = (Tin1 + Tin2)/2.")

    # smooth data if needed
    for key in ["debitbrut", "Flow", "teb", "Tout", "HP", "BP"]:
        mrun = datatools.smooth(
            mrun,
            key,
            inplace=True,
            tau=tau,
            debug=args.debug,
            show=args.show,
            input_file=args.input_file,
        )

    # Geom specs from HX Datasheet
    Nc = int((553 - 1) / 2.0)  # (Number of plates -1)/2
    Ac = 3.0e-3 * 1.174  # Plate spacing * Plate width [m^2]
    de = 2 * 3.0e-3  # 2*Plate spacing [m]
    # coolingparams = [0.207979, 0.640259, 0.397994]
    coolingparams = [0.07, 0.8, 0.4]

    # Compute OHTC
    df = mrun.getData()
    df["MeanU_h"] = df.apply(
        lambda row: ((row.Flow) * 1.0e-3 + args.debit_alim / 3600.0) / (Ac * Nc), axis=1
    )
    df["MeanU_c"] = df.apply(lambda row: (row.debitbrut / 3600.0) / (Ac * Nc), axis=1)
    df["Ohtc"] = df.apply(
        lambda row: getOHTC(
            row.MeanU_h,
            row.MeanU_c,
            de,
            row.BP,
            row.Tout,
            row.BP,
            row.teb,
            coolingparams,
        ),
        axis=1,
    )

    ax = plt.gca()
    df.plot(x="t", y="Ohtc", ax=ax, color="red")
    plt.ylabel(r"Q[$W/m^2/K$]")
    plt.xlabel(r"t [s]")
    plt.grid(b=True)
    plt.title(mrun.getInsert().replace(r"_", r"\_"))
    if args.show:
        plt.show()
    else:
        imagefile = args.input_file.replace(f_extension, ".png")
        print("save to %s" % imagefile)
        plt.savefig(imagefile, dpi=300)
        plt.close()
