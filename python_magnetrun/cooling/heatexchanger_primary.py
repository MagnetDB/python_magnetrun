#! /usr/bin/python3

from __future__ import unicode_literals
from typing import Tuple

import math
import os
import sys
from ..MagnetRun import MagnetRun

import matplotlib

# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available
import matplotlib.pyplot as plt

import pandas as pd
from . import water as w

import ht
from ..processing import smoothers as smoothtools
from ..processing import filters as filtertools


def mixingTemp(Flow1, P1, T1, Flow2, P2, T2):
    """
    computes the mixing temperature
    """
    Tmix = w.getRho(P1, T1) * w.getCp(P1, T1) * Flow1 * T1
    Tmix += w.getRho(P2, T2) * w.getCp(P2, T2) * Flow2 * T2
    Tmix /= (
        w.getRho(P1, T1) * w.getCp(P1, T1) * Flow1
        + w.getRho(P2, T2) * w.getCp(P2, T2) * Flow2
    )
    return Tmix


def display_Q(
    inputfile: str,
    f_extension: str,
    mrun,
    debit_alim,
    ohtc,
    dT,
    show: bool = False,
    extension: str = "-Q.png",
):
    """
    Plot Heat profiles.
    """
    df = mrun.getMData().getPandasData(key=None)

    def apply_heatexchange(row, ohtc_value):
        return (
            heatexchange(
                ohtc_value,
                row.teb,
                row.Thi,
                row.debitbrut / 3600.0,
                row.Flowhot,
                10,
                row.BP,
            )[2]
            / 1.0e6
        )

    if ohtc != "None":
        df["QNTU"] = df.apply(lambda row: apply_heatexchange(row, ohtc), axis=1)
    else:
        df["QNTU"] = df.apply(lambda row: apply_heatexchange(row, row.Ohtc), axis=1)

    df["Qhot"] = df.apply(
        lambda row: ((row.Flow) * 1.0e-3 + 0 / 3600.0)
        * (
            w.getRho(row.BP, row.Tout) * w.getCp(row.BP, row.Tout) * (row.Tout)
            - w.getRho(row.HP, row.Tin) * w.getCp(row.HP, row.Tin) * row.Tin
        )
        / 1.0e6,
        axis=1,
    )

    df["QhotHx"] = df.apply(
        lambda row: (row.Flowhot)
        * (
            w.getRho(row.BP, row.Thi) * w.getCp(row.BP, row.Thi) * row.Thi
            - w.getRho(row.HP, row.Tin) * w.getCp(row.HP, row.Tin) * row.Tin
        )
        / 1.0e6,
        axis=1,
    )

    df["QcoldHx"] = df.apply(
        lambda row: row.debitbrut
        / 3600.0
        * (
            w.getRho(10, row.tsb) * w.getCp(10, row.tsb) * row.tsb
            - w.getRho(10, row.teb) * w.getCp(10, row.teb) * row.teb
        )
        / 1.0e6,
        axis=1,
    )

    ax = plt.gca()
    df["Pinstall"] = df.apply(
        lambda row: debit_alim
        / 3600.0
        * (
            w.getRho(row.BP, row.TAlimout)
            * w.getCp(row.BP, row.TAlimout)
            * row.TAlimout
            - w.getRho(row.HP, row.Tin) * w.getCp(row.HP, row.Tin) * row.Tin
        )
        / 1.0e6,
        axis=1,
    )
    df.plot(x="t", y="Pinstall", ax=ax, color="yellow")
    plt.ylabel(r"Q[MW]")
    plt.xlabel(r"t [s]")
    plt.grid(True)
    plt.show()
    plt.close()

    ax = plt.gca()
    df.plot(x="t", y="Pmagnet", ax=ax)
    df.plot(x="t", y="Ptot", ax=ax)
    plt.ylabel(r"Q[MW]")
    plt.xlabel(r"t [s]")
    plt.grid(True)
    plt.show()
    plt.close()

    ax = plt.gca()
    df["Ppumps_Pinstall"] = df["Ptot"] - df["Pmagnet"]
    df["Ppumps"] = df["Ppumps_Pinstall"] - df["Pinstall"]
    df.plot(x="t", y="Ppumps", ax=ax)
    df.plot(x="t", y="Pinstall", ax=ax)
    plt.ylabel(r"Q[MW]")
    plt.xlabel(r"t [s]")
    plt.grid(True)
    plt.show()
    plt.close()

    ax = plt.gca()
    df.plot(x="t", y="Qhot", ax=ax)
    df.plot(x="t", y="Pmagnet", ax=ax)
    df.plot(x="t", y="Ptot", ax=ax)
    plt.ylabel(r"Q[MW]")
    plt.xlabel(r"t [s]")
    plt.grid(True)

    experiment = mrun.getInsert().replace(r"_", r"\_")
    if ohtc != "None":
        if isinstance(ohtc, (float, int, str)):
            plt.title(
                f"HeatBalance Magnet side:{experiment}: h={ohtc} $W/m^2/K$, dT={dT}"
            )
    else:
        plt.title(
            f"HeatBalance Magnet side: {experiment}: h=formula $W/m^2/K$, dT={dT}"
        )

    if show:
        plt.show()
    else:
        extension = "-Q_magnetside.png"
        imagefile = inputfile.replace(f_extension, extension)
        print(f"save to {imagefile}")
        plt.savefig(imagefile, dpi=300)
    plt.close()

    ax = plt.gca()
    df.plot(
        x="t",
        y="QhotHx",
        ax=ax,
        marker="o",
        markevery=800,
        alpha=0.5,
    )
    df.plot(x="t", y="QcoldHx", ax=ax)
    plt.ylabel(r"Q[MW]")
    plt.xlabel(r"t [s]")
    plt.grid(True)

    if ohtc != "None":
        if isinstance(ohtc, (float, int, str)):
            plt.title(f"HeatBalance HX side: {experiment}: h={ohtc} $W/m^2/K$, dT={dT}")
    else:
        plt.title(f"HeatBalance HX side: {experiment}: h=formula $W/m^2/K$, dT={dT}")

    if show:
        plt.show()
    else:
        extension = "-Q_hxside.png"
        imagefile = inputfile.replace(f_extension, extension)
        print(f"save to {imagefile}")
        plt.savefig(imagefile, dpi=300)
    plt.close()


def display_T(
    inputfile: str,
    f_extension: str,
    mrun,
    tsb_key: str,
    tin_key: str,
    debit_alim: float,
    ohtc,
    dT: float,
    show: bool = False,
    extension: str = "-coolingloop.png",
    debug: bool = False,
):
    """
    Plot Temperature profiles.
    """
    print("othc=", ohtc)
    print("debit_alim=", debit_alim)
    df = mrun.getMData().getPandasData(key=None)
    print(df.head())

    def apply_tsb(row, ohtc_value):
        return heatexchange(
            ohtc_value,
            row.teb,
            row.Thi,
            row.debitbrut / 3600.0,
            row.Flowhot,
            10,
            row.BP,
        )

    if ohtc != "None":
        df[[tsb_key, tin_key]] = df.apply(
            lambda row: pd.Series(apply_tsb(row, ohtc)[:2]), axis=1
        )
    else:
        df[[tsb_key, tin_key]] = df.apply(
            lambda row: pd.Series(apply_tsb(row, row.Ohtc)[:2]), axis=1
        )

    print(df[[tin_key, tsb_key]].head())
    ax = plt.gca()
    df.plot(
        x="t",
        y=tsb_key,
        ax=ax,
        color="blue",
        marker="o",
        markevery=800,
        alpha=0.5,
    )
    df.plot(x="t", y="tsb", ax=ax, color="blue")
    df.plot(x="t", y="teb", ax=ax, color="orange", linestyle="--")
    df.plot(
        x="t",
        y=tin_key,
        ax=ax,
        color="red",
        marker="o",
        markevery=800,
        alpha=0.5,
    )
    df.plot(x="t", y="Tin", ax=ax, color="red")
    df.plot(x="t", y="Tout", ax=ax, color="yellow", linestyle="--")
    df.plot(
        x="t",
        y="cTout",
        ax=ax,
        color="yellow",
        marker="o",
        markevery=800,
        alpha=0.5,
    )
    df.plot(
        x="t",
        y="Thi",
        ax=ax,
        color="yellow",
        marker="x",
        markevery=800,
        alpha=0.5,
    )
    plt.ylabel(r"T[C]")
    plt.xlabel(r"t [s]")
    plt.grid(True)

    experiment = mrun.getInsert().replace(r"_", r"\_")

    if ohtc != "None":
        if isinstance(ohtc, (float, int, str)):
            plt.title(f"{experiment}: h={ohtc} $W/m^2/K$, dT={dT}")
    else:
        plt.title(f"{experiment}: h=computed $W/m^2/K$, dT={dT}")

    if show:
        plt.show()
    else:
        imagefile = inputfile.replace(f_extension, extension)
        print(f"save to {imagefile}")
        plt.savefig(imagefile, dpi=300)
    plt.close()


def estimate_temperature_elevation(
    power, flow_rate, inlet_temp, outlet_pressure, inlet_pressure, iterations=10
):
    """
    Estimate the temperature elevation of water in a pipe section where power is dissipated,
    accounting for temperature-dependent properties using the IAPWS97 package.

    Parameters:
    - power: Dissipated power in watts (W).
    - flow_rate: Volumetric flow rate in m^3/s.
    - inlet_temp: Inlet temperature in degrees Celsius.
    - outlet_temp: Outlet pressure in bar.
    - inlet_temp: Inlet pressure in bar.
    - iterations: Number of iterations for convergence.

    Returns:
    - outlet_temp: Outlet temperature in degrees Celsius.
    """
    eps = 1.0e-3
    inlet_temp_k = inlet_temp
    outlet_temp_k = inlet_temp_k

    for i in range(iterations):
        # Calculate properties at the current average temperature
        avg_temp_k = (inlet_temp_k + outlet_temp_k) / 2
        avg_pressure_k = (inlet_pressure + outlet_pressure) / 2

        rho = w.getRho(celsius=avg_temp_k, pbar=avg_pressure_k)
        cp = w.getCp(celsius=avg_temp_k, pbar=avg_pressure_k)

        # Mass flow rate (kg/s) from volumetric flow rate
        mass_flow_rate = rho * flow_rate

        # Recalculate temperature elevation
        delta_t = power / (mass_flow_rate * cp)

        # Update outlet temperature
        error = (outlet_temp_k - (inlet_temp_k + delta_t)) / outlet_temp_k
        # print("error=", error)
        outlet_temp_k = inlet_temp_k + delta_t
        if abs(error) <= eps:
            break

    # Convert outlet temperature back to Celsius
    outlet_temp = outlet_temp_k
    # print(
    #     f"Estimated outlet temperature:  {outlet_temp:.2f} °C after {i} iterations (error={error}, power={power:.2f} W, Tin={inlet_temp:.2f} C, Flow={flow_rate:.2f}) m/s."
    # )

    return outlet_temp


# def heatBalance(Tin, Pin, Debit, Power, debug=False):
#    """
#    Computes Tout from heatBalance
#
#    inputs:
#    Tin: input Temp in K
#    Pin: input Pressure (Bar)
#    Debit: Flow rate in kg/s
#    """
#
#    dT = Power / (w.getRho(Tin, Pin) * Debit * w.getCp(Tin, Pin))
#    Tout = Tin + dT
#    return Tout


def calculate_heat_capacity_and_density(
    Pci: float, Tci: float, Phi: float, Thi: float
) -> Tuple[float, float, float, float]:
    Cp_cold = w.getCp(Pci, Tci)  # J/kg/K
    Cp_hot = w.getCp(Phi, Thi)  # J/kg/K
    rho_hot = w.getRho(Phi, Thi)
    rho_cold = w.getRho(Pci, Tci)
    return Cp_cold, Cp_hot, rho_hot, rho_cold


def calculate_mass_flow_rates(
    Debitc: float, Debith: float, rho_cold: float, rho_hot: float
) -> Tuple[float, float]:
    m_hot = rho_hot * Debith  # kg/s
    m_cold = rho_cold * Debitc  # kg/s
    return m_hot, m_cold


def validate_results(
    NTU: float,
    Q: float,
    Tco: float,
    Tho: float,
    h: float,
    Tci: float,
    Thi: float,
    Pci: float,
    Phi: float,
    Debitc: float,
    Debith: float,
):
    if NTU == float("inf") or math.isnan(NTU):
        print("Tci=", Tci, "Thi=", Thi)
        print("Pci=", Pci, "Phi=", Phi)
        print("Debitc=", Debitc, "Debith=", Debith)
        raise Exception("NTU not valid")

    if Q == float("inf") or math.isnan(Q):
        print("Tci=", Tci, "Thi=", Thi)
        print("Pci=", Pci, "Phi=", Phi)
        print("Debitc=", Debitc, "Debith=", Debith)
        raise Exception("Q not valid")

    if Tco is None:
        print("h=", h)
        print("Tci=", Tci, "Thi=", Thi)
        print("Pci=", Pci, "Phi=", Phi)
        print("Debitc=", Debitc, "Debith=", Debith)
        raise Exception("Tco not valid")

    if Tho is None:
        print("h=", h)
        print("Tci=", Tci, "Thi=", Thi)
        print("Pci=", Pci, "Phi=", Phi)
        print("Debitc=", Debitc, "Debith=", Debith)
        raise Exception("Tho not valid")


def heatexchange(
    h: float,
    Tci: float,
    Thi: float,
    Debitc: float,
    Debith: float,
    Pci: float,
    Phi: float,
    debug: bool = False,
) -> Tuple[float, float, float]:
    """
    NTU Model for heat Exchanger

    compute the output temperature for the heat exchanger
    as a function of input temperatures and flow rates

    Tci: input Temp on cold side
    Thi: input Temp on hot side
    TA: output from cooling alim (on hot side)

    Debitc: m^3/h
    Debith: l/s
    """

    A = 1063.4  # m^2
    Cp_cold, Cp_hot, rho_hot, rho_cold = calculate_heat_capacity_and_density(
        Pci, Tci, Phi, Thi
    )
    m_hot, m_cold = calculate_mass_flow_rates(Debitc, Debith, rho_cold, rho_hot)

    # For plate exchanger
    result = ht.hx.P_NTU_method(
        m_hot, m_cold, Cp_hot, Cp_cold, UA=h * A, T1i=Thi, T2i=Tci, subtype="1/1"
    )

    NTU = result["NTU1"]
    Q = result["Q"]
    Tco = result["T2o"]
    Tho = result["T1o"]

    validate_results(NTU, Q, Tco, Tho, h, Tci, Thi, Pci, Phi, Debitc, Debith)

    return Tco, Tho, Q


if __name__ == "__main__":
    command_line = None

    import argparse

    parser = argparse.ArgumentParser("Cooling loop Heat Exchanger")
    parser.add_argument(
        "input_file", help="input txt file (ex. M10_2020.10.04_20-2009_43_31.txt)"
    )
    parser.add_argument(
        "--nhelices", help="specify number of helices", type=int, default=14
    )
    parser.add_argument(
        "--ohtc",
        help="specify heat exchange coefficient (ex. 4000 W/K/m^2 or None)",
        type=str,
        default="None",
    )
    parser.add_argument(
        "--dT",
        help="specify dT for Tout (aka accounting for alim cooling, ex. 0)",
        type=float,
        default=0,
    )
    parser.add_argument("--site", help="specify a site (ex. M8, M9,...)", type=str)
    parser.add_argument(
        "--debit_alim",
        help="specify flowrate for power cooling - one half only (default: 60 m3/h)",
        type=float,
        default=60,
    )
    parser.add_argument(
        "--show",
        help="display graphs (requires X11 server active)",
        action="store_true",
    )
    parser.add_argument("--debug", help="activate debug mode", action="store_true")
    # parser.add_argument("--save", help="save graphs to png", action='store_true')

    # raw|filter|smooth post-traitement of data
    parser.add_argument(
        "--pre",
        help="select a pre-traitment for data",
        type=str,
        choices=["raw", "filtered", "smoothed"],
        default="smoothed",
    )
    # define params for post traitment of data
    parser.add_argument(
        "--pre_params",
        help="pass param for pre-traitment method",
        type=str,
        default="400",
    )

    parser.add_argument(
        "--Q",
        help="specify Q factor for Flow (aka cooling magnets, ex. 1)",
        type=float,
        default=1,
    )
    args = parser.parse_args(command_line)

    tau = 400
    if args.pre == "smoothed":
        print("smoothed options")
        tau = float(args.pre_params)

    threshold = 0.5
    twindows = 10
    if args.pre == "filtered":
        print("filtered options")
        params = args.pre_params.split(";")
        threshold = float(params[0])
        twindows = int(params[1])

    print("args: ", args)

    # check extension
    f_extension = os.path.splitext(args.input_file)[-1]
    if f_extension != ".txt":
        print("so far only txt file support is implemented")
        sys.exit(0)

    housing = args.site
    filename = os.path.basename(args.input_file)
    result = filename.startswith("M")
    if result:
        try:
            index = filename.index("_")
            args.site = filename[0:index]
            housing = args.site
            print(f"site detected: {args.site}")
        except:
            print("no site detected - use args.site argument instead")
            pass

    mrun = MagnetRun.fromtxt(housing, args.site, args.input_file)
    if not args.site:
        args.site = mrun.getSite()

    experiment = mrun.getInsert().replace(r"_", r"\_")

    # Adapt filtering and smoothing params to run duration
    duration = mrun.getMData().getDuration()
    if duration <= 10 * tau:
        tau = min(duration // 10, 10)
        print(f"Modified smoothing param: {tau} over {duration} s run")
        # args.markevery = 8 * tau

    # print("type(mrun):", type(mrun))
    start_timestamp = mrun.getMData().getStartDate()

    if "Flow" not in mrun.getKeys():
        mrun.getMData().addData("Flow", "Flow = FlowH + FlowB")
    if "Tin" not in mrun.getKeys():
        mrun.getMData().addData("Tin", "Tin = (TinH + TinB)/2.")
    if "HP" not in mrun.getKeys():
        mrun.getMData().addData("HP", "HP = (HPH + HPB)/2.")
    if "TAlimout" not in mrun.getKeys():
        # Talim not defined, try to estimate it
        print("TAlimout key not present - set TAlimout=0")
        mrun.getMData().addData("Talim", "TAlimout = 0")

    # extract data
    keys = [
        "t",
        "teb",
        "tsb",
        "debitbrut",
        "Tout",
        "Tin",
        "Flow",
        "BP",
        "HP",
        "Pmagnet",
    ]
    units = ["s", "C", "C", "m\u00b3/h", "C", "C", "l/s", "bar", "MW"]
    # df = mrun.getMData().extractData(keys)

    if args.debug:
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)

    if "PH" not in mrun.getKeys():
        mrun.getMData().addData("PH", "PH = UH * IH")
    if "PB" not in mrun.getKeys():
        mrun.getMData().addData("PB", "PB = UB * IB")
    if "Pt" not in mrun.getKeys():
        mrun.getMData().addData("Pt", "Pt = (PH + PB)/1.e+6")
    df = mrun.getMData().getPandasData(key=None)

    # estimate dTH: PH / (rho * Cp * FlowH)
    def apply_ToutH(row):
        return estimate_temperature_elevation(
            row.PH, row.FlowH * 1.0e-3, row.TinH, row.BP, row.HPH
        )

    df["ToutH"] = df.apply(lambda row: pd.Series(apply_ToutH(row)), axis=1)

    # estimate dTH: PH / (rho * Cp * FlowH)
    def apply_ToutB(row):
        return estimate_temperature_elevation(
            row.PB, row.FlowB * 1.0e-3, row.TinB, row.BP, row.HPB
        )

    df["ToutB"] = df.apply(lambda row: pd.Series(apply_ToutB(row)), axis=1)

    # estimate Tout: mixingTemp(ToutH, ToutB)
    def apply_mixingTemp(row):
        return mixingTemp(
            row.FlowH,
            row.BP,
            row.ToutH,
            row.FlowB,
            row.BP,
            row.ToutB,
        )

    df["cTout"] = df.apply(lambda row: pd.Series(apply_mixingTemp(row)), axis=1)

    ax = plt.gca()
    df.plot(x="t", y="Tout", ax=ax, color="blue", marker="o", alpha=0.5, markevery=800)
    df.plot(x="t", y="cTout", ax=ax, color="blue", linestyle="--")
    plt.xlabel(r"t [s]")
    plt.ylabel(r"T [C]")
    plt.title(f"{experiment}: Tout")
    plt.grid(True)
    plt.show()
    plt.close()

    pretreatment_keys = ["debitbrut", "Flow", "teb", "Tout", "PH", "PB", "Pt"]
    if "TAlimout" in mrun.getKeys():
        pretreatment_keys.append("TAlimout")
    else:
        mrun.getMData().addData("TAlimout", "TAlimout = 0")

    # filter spikes
    # see: https://ocefpaf.github.io/python4oceanographers/blog/2015/03/16/outlier_detection/
    if args.pre == "filtered":
        for key in pretreatment_keys:
            mrun = filtertools.filterpikes(
                mrun,
                key,
                inplace=True,
                threshold=threshold,
                twindows=twindows,
                debug=args.debug,
                show=args.show,
                input_file=args.input_file,
            )
        print("Filtered pikes done")

    # smooth data Locally Weighted Linear Regression (Loess)
    # see: https://xavierbourretsicotte.github.io/loess.html(
    if args.pre == "smoothed":
        for key in pretreatment_keys:
            mrun = smoothtools.smooth(
                mrun,
                key,
                inplace=True,
                tau=tau,
                debug=args.debug,
                show=args.show,
                input_file=args.input_file,
            )
        print("smooth data done")
    print(mrun.getKeys())

    # Geom specs from HX Datasheet
    Nc = int((553 - 1) / 2.0)  # (Number of plates -1)/2
    Ac = 3.0e-3 * 1.174  # Plate spacing * Plate width [m^2]
    de = 2 * 3.0e-3  # 2*Plate spacing [m]
    # coolingparams = [0.207979, 0.640259, 0.397994]  # from nominal values
    coolingparams = [0.1249, 0.65453, 0.40152]  # from student param fits
    # coolingparams = [0.07, 0.8, 0.4]

    # Compute OHTC
    df["Flowhot"] = df.apply(
        lambda row: ((row.Flow) * 1.0e-3 + args.debit_alim / 3600.0), axis=1
    )
    df["MeanU_h"] = df.apply(
        lambda row: (row.Flowhot) / (Ac * Nc),
        axis=1,
    )
    df["MeanU_c"] = df.apply(lambda row: (row.debitbrut / 3600.0) / (Ac * Nc), axis=1)

    def apply_mixingThi(row):
        return mixingTemp(
            row.Flow,
            row.BP,
            row.Tout,
            args.debit_alim / 3600.0,
            row.BP,
            row.TAlimout,
        )

    df["Thi"] = df.apply(lambda row: pd.Series(apply_mixingThi(row)), axis=1)
    ax = plt.gca()
    df.plot(x="t", y="Tout", ax=ax)
    df.plot(x="t", y="cTout", ax=ax)
    df.plot(x="t", y="Thi", ax=ax)
    plt.ylabel(r"T[C]")
    plt.xlabel(r"t [s]")
    plt.grid(True)
    plt.show()
    plt.close()

    df["Ohtc"] = df.apply(
        lambda row: w.getOHTC(
            row.MeanU_h,
            row.MeanU_c,
            de,
            row.BP,
            row.Thi,
            row.BP,
            row.teb,
            coolingparams,
        ),
        axis=1,
    )
    ax = plt.gca()
    df.plot(
        x="t",
        y="Ohtc",
        ax=ax,
        color="red",
        marker="o",
        markevery=800,
        alpha=0.5,
    )
    plt.grid(True)
    plt.xlabel(r"t [s]")
    plt.ylabel(r"$W/m^2/K$")
    plt.title(f"{experiment}: Heat Exchange Coefficient")
    if args.show:
        plt.show()
    else:
        imagefile = args.input_file.replace(".txt", "-ohtc.png")
        plt.savefig(imagefile, dpi=300)
        print(f"save to {imagefile}")
    plt.close()

    display_T(
        args.input_file,
        f_extension,
        mrun,
        "itsb",
        "iTin",
        args.debit_alim,
        args.ohtc,
        args.dT,
        args.show,
        "-coolingloop.png",
        args.debug,
    )
    display_Q(
        args.input_file,
        f_extension,
        mrun,
        args.debit_alim,
        args.ohtc,
        args.dT,
        args.show,
        "-Q.png",
    )
