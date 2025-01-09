#! /usr/bin/python3

from __future__ import unicode_literals

# from _typeshed import NoneType

import math
import os
import sys
from ..MagnetRun import MagnetRun

import numpy as np
import matplotlib

# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available
import matplotlib.pyplot as plt

import pandas as pd
from . import water as w

import ht
import tabulate
from ..processing import smoothers as smoothtools
from ..processing import filters as filtertools

tables = []


def mixingTemp(Flow1, P1, T1, Flow2, P2, T2):
    """
    computes the mixing temperature
    """
    Flow = Flow1 + Flow2
    Tmix = w.getRho(P1, T1) * w.getCp(P1, T1) * T1 * Flow1
    Tmix += w.getRho(P2, T2) * w.getCp(P2, T2) * T2 * Flow2
    Tmix /= w.getRho((P1 + P2) / 2.0, T2) * w.getCp((P1 + P2) / 2.0, T2) * Flow
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
    plot Heat profiles
    """

    df = mrun.getMData().getPandasData(key=None)
    # print("type(mrun.getData()):", type(mrun.getData()))
    # print("type(df):", type(df), type(df['Tin']))

    if "Flowhot" not in df:
        df["Flowhot"] = df.apply(
            lambda row: ((row.Flow) * 1.0e-3 + (2 * debit_alim) / 3600.0), axis=1
        )
    if "Thi" not in df:
        df["Thi"] = df.apply(
            lambda row: mixingTemp(
                row.Flow * 1.0e-3,
                row.BP,
                row.Tout + dT,
                2 * debit_alim / 3600.0,
                row.BP,
                row.TAlimout,
            ),
            axis=1,
        )

    if ohtc != "None":
        df["QNTU"] = df.apply(
            lambda row: heatexchange(
                ohtc, row.teb, row.Thi, row.debitbrut / 3600.0, row.Flowhot, 10, row.BP
            )[2]
            / 1.0e6,
            axis=1,
        )
    else:
        df["QNTU"] = df.apply(
            lambda row: heatexchange(
                row.Ohtc,
                row.teb,
                row.Thi,
                row.debitbrut / 3600.0,
                row.Flowhot,
                10,
                row.BP,
            )[2]
            / 1.0e6,
            axis=1,
        )

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
            w.getRho(row.BP, row.Thi) * w.getCp(row.BP, row.Thi) * (row.Thi)
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
    # print("df.keys:", df.columns.values.tolist(), "mrun.keys=", mrun.getKeys())

    # estimate Pinstall
    ax = plt.gca()
    df["Pinstall"] = df.apply(
        lambda row: (2 * debit_alim)
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

    # heat Balance on Magnet side
    ax = plt.gca()
    df.plot(x="t", y="Qhot", ax=ax)
    df.plot(
        x="t",
        y="Pt",
        ax=ax,
        marker="o",
        alpha=0.5,
        markevery=args.markevery,
    )
    df.plot(x="t", y="Pmagnet", ax=ax, color="yellow")
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
        # if isinstance(ohtc, type(df['Tin'])):
        plt.title(
            f"HeatBalance Magnet side: {experiment}: h=formula $W/m^2/K$, dT={dT}"
        )

    if show:
        plt.show()
    else:
        extension = "-Q_magnetside.png"
        imagefile = inputfile.replace(f_extension, extension)
        print("fsave to {imagefile}")
        plt.savefig(imagefile, dpi=300)
    plt.close()

    # heat Balance on HX side
    ax = plt.gca()
    df.plot(
        x="t",
        y="QhotHx",
        ax=ax,
        marker="o",
        alpha=0.5,
        markevery=args.markevery,
    )
    df.plot(x="t", y="QcoldHx", ax=ax)
    plt.ylabel(r"Q[MW]")
    plt.xlabel(r"t [s]")
    plt.grid(True)

    if ohtc != "None":
        if isinstance(ohtc, (float, int, str)):
            plt.title(f"HeatBalance HX side: {experiment}: h={ohtc} $W/m^2/K$, dT={dT}")
    else:
        # if isinstance(ohtc, type(df['Tin'])):
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
    inputfile,
    f_extension,
    mrun,
    tsb_key,
    tin_key,
    debit_alim,
    ohtc,
    dT,
    show=False,
    extension="-coolingloop.png",
    debug=False,
):
    """
    plot Temperature profiles
    """

    print("othc=", ohtc)
    print("debit_alim=", debit_alim)
    df = mrun.getMData().getPandasData(key=None)
    print(df.head())

    df["Flowhot"] = df.apply(
        lambda row: ((row.Flow) * 1.0e-3 + (2 * debit_alim) / 3600.0), axis=1
    )
    df["Thi"] = df.apply(
        lambda row: mixingTemp(
            row.Flow * 1.0e-3,
            row.BP,
            row.Tout + dT,
            2 * debit_alim / 3600.0,
            row.BP,
            row.TAlimout,
        ),
        axis=1,
    )

    if ohtc != "None":
        df[tin_key] = df.apply(
            lambda row: heatexchange(
                ohtc,
                row.teb,
                row.Thi,
                row.debitbrut / 3600.0,
                row.Flowhot,
                10,
                row.BP,
            )[1],
            axis=1,
        )
        df[tsb_key] = df.apply(
            lambda row: heatexchange(
                ohtc,
                row.teb,
                row.Thi,
                row.debitbrut / 3600.0,
                row.Flowhot,
                10,
                row.BP,
            )[0],
            axis=1,
        )
    else:
        df[tin_key] = df.apply(
            lambda row: heatexchange(
                row.Ohtc,
                row.teb,
                row.Thi,
                row.debitbrut / 3600.0,
                row.Flowhot,
                10,
                row.BP,
            )[1],
            axis=1,
        )
        df[tsb_key] = df.apply(
            lambda row: heatexchange(
                row.Ohtc,
                row.teb,
                row.Thi,
                row.debitbrut / 3600.0,
                row.Flowhot,
                10,
                row.BP,
            )[0],
            axis=1,
        )

    ax = plt.gca()
    df.plot(
        x="t",
        y=tsb_key,
        ax=ax,
        color="blue",
        marker="o",
        alpha=0.5,
        markevery=args.markevery,
    )
    df.plot(x="t", y="tsb", ax=ax, color="blue")
    df.plot(x="t", y="teb", ax=ax, color="orange", linestyle="--")
    df.plot(
        x="t",
        y=tin_key,
        ax=ax,
        color="red",
        marker="o",
        alpha=0.5,
        markevery=args.markevery,
    )
    df.plot(x="t", y="Tin", ax=ax, color="red")
    df.plot(x="t", y="Tout", ax=ax, color="yellow", linestyle="--")
    df.plot(
        x="t",
        y="Thi",
        ax=ax,
        color="yellow",
        marker="o",
        alpha=0.5,
        markevery=args.markevery,
        grid=True,
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


def heatBalance(Tin, Pin, Debit, Power, debug=False):
    """
    Computes Tout from heatBalance

    inputs:
    Tin: input Temp in K
    Pin: input Pressure (Bar)
    Debit: Flow rate in kg/s
    """

    dT = Power / (w.getRho(Tin, Pin) * Debit * w.getCp(Tin, Pin))
    Tout = Tin + dT
    return Tout


def heatexchange(h, Tci, Thi, Debitc, Debith, Pci, Phi, debug: bool = False):
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

    # if debug:
    #     print("heatexchange:",
    #           "h=", U,
    #           "Tci=", Tci, "Thi=", Thi,
    #           "Pci=", Pci, "Phi=", Phi,
    #           "Debitc=", Debitc, "Debith=", Debith, "DebitA=", DebitA)

    A = 1063.4  # m^2
    Cp_cold = w.getCp(Pci, Tci)  # J/kg/K
    Cp_hot = w.getCp(Phi, Thi)  # J/kg/K
    m_hot = w.getRho(Phi, Thi) * Debith  # kg/s
    m_cold = w.getRho(Pci, Tci) * Debitc  # kg/s

    # For plate exchanger
    result = ht.hx.P_NTU_method(
        m_hot, m_cold, Cp_hot, Cp_cold, UA=h * A, T1i=Thi, T2i=Tci, subtype="1/1"
    )

    # returns a dictionnary:
    # Q : Heat exchanged in the heat exchanger, [W]
    # UA : Combined area-heat transfer coefficient term, [W/K]
    # T1i : Inlet temperature of stream 1, [K]
    # T1o : Outlet temperature of stream 1, [K]
    # T2i : Inlet temperature of stream 2, [K]
    # T2o : Outlet temperature of stream 2, [K]
    # P1 : Thermal effectiveness with respect to stream 1, [-]
    # P2 : Thermal effectiveness with respect to stream 2, [-]
    # R1 : Heat capacity ratio with respect to stream 1, [-]
    # R2 : Heat capacity ratio with respect to stream 2, [-]
    # C1 : The heat capacity rate of fluid 1, [W/K]
    # C2 : The heat capacity rate of fluid 2, [W/K]
    # NTU1 : Thermal Number of Transfer Units with respect to stream 1 [-]
    # NTU2 : Thermal Number of Transfer Units with respect to stream 2 [-]

    NTU = result["NTU1"]
    if NTU == float("inf") or math.isnan(NTU):
        print("Tci=", Tci, "Thi=", Thi)
        print("Pci=", Pci, "Phi=", Phi)
        print("Debitc=", Debitc, "Debith=", Debith)
        raise Exception("NTU not valid")

    Q = result["Q"]
    if Q == float("inf") or math.isnan(Q):
        print("Tci=", Tci, "Thi=", Thi)
        print("Pci=", Pci, "Phi=", Phi)
        print("Debitc=", Debitc, "Debith=", Debith)
        raise Exception("Q not valid")

    Tco = result["T2o"]
    if Tco == None:
        print("h=", h)
        print("Tci=", Tci, "Thi=", Thi)
        print("Pci=", Pci, "Phi=", Phi)
        print("Debitc=", Debitc, "Debith=", Debith)
        raise Exception("Tco not valid")
    Tho = result["T1o"]
    if Tho == None:
        print("h=", h)
        print("Tci=", Tci, "Thi=", Thi)
        print("Pci=", Pci, "Phi=", Phi)
        print("Debitc=", Debitc, "Debith=", Debith)
        raise Exception("Tho not valid")

    """
    if dT != 0 and m_alim_A1A2*m_alim_A3A4 != 0:
        dT -= Thi * ( m_hot/(m_hot + m_alim_A1A2 + m_alim_A3A4) -1)
        dT_alim = ( dT/(m_alim_A1A2/(m_hot + m_alim_A1A2 + m_alim_A3A4)) ) / 2. - Tho
        P_A1A2 = dT_alim*m_alim_A1A2*Cp_hot
        P_A3A4 = dT_alim*m_alim_A3A4*Cp_hot
        if debug:
            print("heatexchange: ", NTU, Tco, Tho, Q)
            print("m_alim: ", m_alim_A1A2 + m_alim_A1A2, "m_hot:", m_hot, "%.2f" % ((m_alim_A1A2 + m_alim_A1A2)/m_hot*100), "%")
            # TODO check with site
            print("dT_alim:", dT_alim,
                  "P_A1A2[MW]:", P_A1A2/1.e+6, "%.2f" % (P_A1A2/abs(PowerH)*100), "%",
                  "P_A3A4[MW]:", P_A3A4/1.e+6, "%.2f" % (P_A3A4/abs(PowerB)*100), "%",
                  "PH[MW]", abs(PowerH/1.e+6),
                  "PB[MW]", abs(PowerB/1.e+6))
    """
    return (Tco, Tho, Q)


def find(
    df,
    unknows: list,
    dTini: float,
    hini: float,
    hmin: float,
    hmax: float,
    algo: str,
    lalgo: str,
    maxeval: float,
    stopval: float,
    select: int = 0,
    site: str = "M9",
    debit_alim: float = 30,
    debug: bool = False,
):
    """
    Use nlopt to find h, dT that give the best approximation for Hx output temperature

    unknows = list of optim var (eg ["dT"] or ["h", "dT"])
    returns a dict
    """

    tables = []
    headers = [
        "dT[C]",
        "h[W/m\u00b2/K]",
        "e_Tin[]",
        "e_tsb[]",
        "e_T[]",
        "Heat Balance[MW]",
    ]

    import nlopt

    print(f"find {len(unknows)} params: {unknows}")

    opt = nlopt.opt()
    if algo == "Direct":
        opt = nlopt.opt(nlopt.GN_DIRECT, len(unknows))
    elif algo == "Direct_L":
        opt = nlopt.opt(nlopt.GN_DIRECT_L, len(unknows))
    elif algo == "CRS2":
        opt = nlopt.opt(nlopt.GN_CRS2_LM, len(unknows))
    elif algo == "MLSL":
        opt = nlopt.opt(nlopt.G_MLSL, len(unknows))

    # if lalgo == "Nelder-Mead":
    #     local_opt = nlopt.opt(nlopt.LN_NELDER_MEAD, len(unknows))
    # elif lalgo == "Cobyla":
    #     local_opt = nlopt.opt(nlopt.LN_LN_COBYLA, len(unknows))
    # local_opt.set_maxeval(maxeval)
    # local_opt.set_ftol_rel(stopval)
    # if lalgo != "None":
    #     opt.set_local_optimizer(local_opt)

    opt.set_maxeval(maxeval)
    opt.set_ftol_rel(stopval)
    opt.set_ftol_abs(1.0e-5)
    # opt.set_xtol_rel([tol, tol]) if 2 params? or float?
    # opt.set_xtol_abs([1.e-5, 1.e-5]) if 2 opt params
    if args.debug:
        print(
            "nlopt [ftol fabs xtol xabs]: ",
            opt.get_ftol_rel(),
            opt.get_ftol_abs(),
            opt.get_xtol_rel(),
            opt.get_xtol_abs(),
        )
    print(
        "nlopt [ftol fabs xtol xabs]: ",
        opt.get_ftol_rel(),
        opt.get_ftol_abs(),
        opt.get_xtol_rel(),
        opt.get_xtol_abs(),
    )

    # bounds
    lbounds = []
    ubounds = []
    for unknow in unknows:
        if unknow == "dT":
            lbounds.append(-10)
            ubounds.append(10)
        elif unknow == "h":
            lbounds.append(hmin)
            ubounds.append(hmax)

    opt.set_lower_bounds(lbounds)
    opt.set_upper_bounds(ubounds)
    print("bound:", lbounds, ubounds)

    # init_vals
    init_vals = []
    for unknow in unknows:
        if unknow == "dT":
            init_vals.append(dTini)
        elif unknow == "h":
            init_vals.append(hini)
    print("init_vals:", init_vals)

    # use *f_data to pass extra args: df_, subtype
    # ex:
    # fdata = (df_, sbutype)
    # error_Tin(x, **fdata)
    # df_ = fdata[0], subtype = fdata[1], debug = fdata[2]
    # eventually check type with isinstanceof() / type()
    # question: how to take into account error_tsb also??

    if len(unknows) == 2:
        select = 2
    print("select: ", select)

    def error_Tin(
        x,
        df_=df,
        unknows: list = unknows,
        hini: float = hini,
        dTini: float = dTini,
        select: int = select,
        debug: bool = debug,
    ):
        """compute error between measures and computed data"""

        ohtc = hini
        dT = dTini

        if len(unknows) == 1:
            if unknows[0] == "dT":
                dT = x[0]
            elif unknows[0] == "h":
                ohtc = x[0]
        else:
            ohtc = x[1]
            dT = x[0]

        df["cThi"] = df.apply(
            lambda row: mixingTemp(
                row.Flow * 1.0e-3,
                row.BP,
                row.Tout + dT,
                2 * debit_alim / 3600.0,
                row.BP,
                row.TAlimout,
            ),
            axis=1,
        )

        df_["cTin"] = df_.apply(
            lambda row: heatexchange(
                ohtc, row.teb, row.cThi, row.debitbrut / 3600.0, row.FlowH, 10, row.BP
            )[1],
            axis=1,
        )
        diff = np.abs(df_["Tin"] - df_["cTin"])
        L2_Tin = math.sqrt(np.dot(df_["Tin"], df_["Tin"]))
        error_Tin = math.sqrt(np.dot(diff, diff)) / L2_Tin  # diff.size

        df_["ctsb"] = df_.apply(
            lambda row: heatexchange(
                ohtc, row.teb, row.cThi, row.debitbrut / 3600.0, row.FlowH, 10, row.BP
            )[0],
            axis=1,
        )
        diff = np.abs(df_["tsb"] - df_["ctsb"])
        L2_tsb = math.sqrt(np.dot(df_["tsb"], df_["tsb"]))
        error_tsb = math.sqrt(np.dot(diff, diff)) / L2_tsb  # diff.size

        df["cQhot"] = df.apply(
            lambda row: (row.FlowH)
            * (
                w.getRho(row.BP, row.cThi) * w.getCp(row.BP, row.cThi) * (row.cThi)
                - w.getRho(row.HP, row.cTin) * w.getCp(row.HP, row.cTin) * row.cTin
            )
            / 1.0e6,
            axis=1,
        )
        df["cQcold"] = df.apply(
            lambda row: row.debitbrut
            / 3600.0
            * (
                w.getRho(10, row.ctsb) * w.getCp(10, row.ctsb) * row.ctsb
                - w.getRho(10, row.teb) * w.getCp(10, row.teb) * row.teb
            )
            / 1.0e6,
            axis=1,
        )
        df["cdQ"] = df.apply(lambda row: row.cQhot - row.cQcold, axis=1)

        df["Qhot"] = df.apply(
            lambda row: (row.FlowH)
            * (
                w.getRho(row.BP, row.Thi) * w.getCp(row.BP, row.Thi) * (row.Thi)
                - w.getRho(row.HP, row.Tin) * w.getCp(row.HP, row.Tin) * row.cTin
            )
            / 1.0e6,
            axis=1,
        )
        df["Qcold"] = df.apply(
            lambda row: row.debitbrut
            / 3600.0
            * (
                w.getRho(10, row.tsb) * w.getCp(10, row.tsb) * row.tsb
                - w.getRho(10, row.teb) * w.getCp(10, row.teb) * row.teb
            )
            / 1.0e6,
            axis=1,
        )
        df["dQ"] = df.apply(lambda row: row.Qhot - row.Qcold, axis=1)

        diff = np.abs(df_["Qhot"] - df_["cQhot"])
        L2_Qhot = math.sqrt(np.dot(df_["Qhot"], df_["Qhot"]))
        error_qhot = math.sqrt(np.dot(diff, diff)) / L2_Qhot

        diff = np.abs(df_["Qcold"] - df_["cQcold"])
        L2_Qcold = math.sqrt(np.dot(df_["Qcold"], df_["Qcold"]))
        error_qcold = math.sqrt(np.dot(diff, diff)) / L2_Qcold

        error_T = 0
        if select == 0:
            error_T = math.sqrt(error_Tin * error_Tin)
        if select == 1:
            error_T = math.sqrt(error_tsb * error_tsb)
        if select == 2:
            error_T = math.sqrt(error_Tin * error_Tin + error_tsb * error_tsb)
        if select == 3:
            error_T = df["cdQ"].mean()

        if debug:
            print(
                f"error_Tin({x})",
                error_Tin,
                error_tsb,
                error_T,
                df["cdQ"].mean(),
                select,
                ohtc,
                dT,
            )

        tables.append([dT, ohtc, error_Tin, error_tsb, error_T, df["cdQ"].mean()])

        del df_["ctsb"]
        del df_["cTin"]

        return error_T

    def myfunc(x, grad):
        if grad.size > 0:
            grad[0] = 0.0
            grad[1] = 0.0
        return error_Tin(x)

    opt.set_min_objective(myfunc)
    x = opt.optimize(init_vals)
    minf = opt.last_optimum_value()
    status = opt.last_optimize_result()
    print("optimum: x=", x, "obj=", minf, "(code = ", status, ")")

    # how to mark line with optimum value in red??
    # loop over tables, if line correspond to x[0] then change line to red: a = "\033[1;32m%s\033[0m" %a
    # #Color
    # R = "\033[0;31;40m" #RED
    # G = "\033[0;32;40m" # GREEN
    # Y = "\033[0;33;40m" # Yellow
    # B = "\033[0;34;40m" # Blue
    # N = "\033[0m" # Reset
    if status >= 0:
        for line in tables:
            tmp = 0
            for i, unknow in enumerate(unknows):
                tmp += int(line[i] == x[i])
            if tmp == len(unknows):
                for i, item in enumerate(line):
                    line[i] = f"\033[1;32m{item}\033[0m"
        print("\n", tabulate.tabulate(tables, headers, tablefmt="simple"), "\n")

    optval = {}
    for i, unknow in enumerate(unknows):
        optval[unknow] = x[i]
    return (optval, status)


def findQ(
    df,
    unknows: list,
    Qini: float,
    qmin: float,
    qmax: float,
    algo: str,
    lalgo: str,
    maxeval: float,
    stopval: float,
    select: int = 0,
    site: str = "M9",
    debit_alim: float = 30,
    debug: bool = False,
):
    """
    Use nlopt to find Q that give the best approximation for Hx output temperature

    unknows = list of optim var (eg ["dT"] or ["h", "dT"])
    returns a dict
    """

    tables = []
    headers = ["Q[%]", "e_Tin[]", "e_tsb[]", "e_T[]", "Heat Balance[MW]"]

    import nlopt

    print(f"findQ {len(unknows)} params: {unknows}")

    opt = nlopt.opt()
    if algo == "Direct":
        opt = nlopt.opt(nlopt.GN_DIRECT, len(unknows))
    elif algo == "Direct_L":
        opt = nlopt.opt(nlopt.GN_DIRECT_L, len(unknows))
    elif algo == "CRS2":
        opt = nlopt.opt(nlopt.GN_CRS2_LM, len(unknows))
    elif algo == "MLSL":
        opt = nlopt.opt(nlopt.G_MLSL, len(unknows))

    # if lalgo == "Nelder-Mead":
    #     local_opt = nlopt.opt(nlopt.LN_NELDER_MEAD, len(unknows))
    # elif lalgo == "Cobyla":
    #     local_opt = nlopt.opt(nlopt.LN_LN_COBYLA, len(unknows))
    # local_opt.set_maxeval(maxeval)
    # local_opt.set_ftol_rel(stopval)
    # if lalgo != "None":
    #     opt.set_local_optimizer(local_opt)

    opt.set_maxeval(maxeval)
    opt.set_ftol_rel(stopval)
    opt.set_ftol_abs(1.0e-5)
    # opt.set_xtol_rel([tol, tol]) if 2 params? or float?
    # opt.set_xtol_abs([1.e-5, 1.e-5]) if 2 opt params
    if args.debug:
        print(
            "nlopt [ftol fabs xtol xabs]: ",
            opt.get_ftol_rel(),
            opt.get_ftol_abs(),
            opt.get_xtol_rel(),
            opt.get_xtol_abs(),
        )
    print(
        "nlopt [ftol fabs xtol xabs]: ",
        opt.get_ftol_rel(),
        opt.get_ftol_abs(),
        opt.get_xtol_rel(),
        opt.get_xtol_abs(),
    )

    # bounds
    lbounds = []
    ubounds = []
    for unknow in unknows:
        if unknow == "Q":
            lbounds.append(0.5)
            ubounds.append(2)

    opt.set_lower_bounds(lbounds)
    opt.set_upper_bounds(ubounds)
    print("bound:", lbounds, ubounds)

    # init_vals
    init_vals = []
    for unknow in unknows:
        if unknow == "Q":
            init_vals.append(Qini)
    print("init_vals:", init_vals)

    # use *f_data to pass extra args: df_, subtype
    # ex:
    # fdata = (df_, sbutype)
    # error_Tin(x, **fdata)
    # df_ = fdata[0], subtype = fdata[1], debug = fdata[2]
    # eventually check type with isinstanceof() / type()
    # question: how to take into account error_tsb also??

    def error_Tin(
        x,
        df_=df,
        unknows: list = unknows,
        Qini: float = Qini,
        select: int = select,
        debug: bool = debug,
    ):
        """compute error between measures and computed data"""

        Q = Qini

        if len(unknows) == 1:
            if unknows[0] == "Q":
                Q = x[0]

        df["cThi"] = df.apply(
            lambda row: mixingTemp(
                Q * row.Flow * 1.0e-3,
                row.BP,
                row.Tout,
                2 * debit_alim / 3600.0,
                row.BP,
                row.TAlimout,
            ),
            axis=1,
        )

        # recompute FlowH
        df["cFlowH"] = df.apply(
            lambda row: ((Q * row.Flow) * 1.0e-3 + (2 * args.debit_alim) / 3600.0),
            axis=1,
        )

        df_["cTin"] = df_.apply(
            lambda row: heatexchange(
                row.Ohtc,
                row.teb,
                row.cThi,
                row.debitbrut / 3600.0,
                row.cFlowH,
                10,
                row.BP,
            )[1],
            axis=1,
        )
        diff = np.abs(df_["Tin"] - df_["cTin"])
        L2_Tin = math.sqrt(np.dot(df_["Tin"], df_["Tin"]))
        error_Tin = math.sqrt(np.dot(diff, diff)) / L2_Tin  # diff.size

        df_["ctsb"] = df_.apply(
            lambda row: heatexchange(
                row.Ohtc,
                row.teb,
                row.cThi,
                row.debitbrut / 3600.0,
                row.cFlowH,
                10,
                row.BP,
            )[0],
            axis=1,
        )
        diff = np.abs(df_["tsb"] - df_["ctsb"])
        L2_tsb = math.sqrt(np.dot(df_["tsb"], df_["tsb"]))
        error_tsb = math.sqrt(np.dot(diff, diff)) / L2_tsb  # diff.size

        df["cQhot"] = df.apply(
            lambda row: (row.cFlowH)
            * (
                w.getRho(row.BP, row.cThi) * w.getCp(row.BP, row.cThi) * (row.cThi)
                - w.getRho(row.HP, row.cTin) * w.getCp(row.HP, row.cTin) * row.cTin
            )
            / 1.0e6,
            axis=1,
        )
        df["cQcold"] = df.apply(
            lambda row: row.debitbrut
            / 3600.0
            * (
                w.getRho(10, row.ctsb) * w.getCp(10, row.ctsb) * row.ctsb
                - w.getRho(10, row.teb) * w.getCp(10, row.teb) * row.teb
            )
            / 1.0e6,
            axis=1,
        )
        df["cdQ"] = df.apply(lambda row: row.cQhot - row.cQcold, axis=1)

        df["Qhot"] = df.apply(
            lambda row: (row.FlowH)
            * (
                w.getRho(row.BP, row.Thi) * w.getCp(row.BP, row.Thi) * (row.Thi)
                - w.getRho(row.HP, row.Tin) * w.getCp(row.HP, row.Tin) * row.cTin
            )
            / 1.0e6,
            axis=1,
        )
        df["Qcold"] = df.apply(
            lambda row: row.debitbrut
            / 3600.0
            * (
                w.getRho(10, row.tsb) * w.getCp(10, row.tsb) * row.tsb
                - w.getRho(10, row.teb) * w.getCp(10, row.teb) * row.teb
            )
            / 1.0e6,
            axis=1,
        )
        df["dQ"] = df.apply(lambda row: row.Qhot - row.Qcold, axis=1)

        diff = np.abs(df_["Qhot"] - df_["cQhot"])
        L2_Qhot = math.sqrt(np.dot(df_["Qhot"], df_["Qhot"]))
        error_qhot = math.sqrt(np.dot(diff, diff)) / L2_Qhot

        diff = np.abs(df_["Qcold"] - df_["cQcold"])
        L2_Qcold = math.sqrt(np.dot(df_["Qcold"], df_["Qcold"]))
        error_qcold = math.sqrt(np.dot(diff, diff)) / L2_Qcold

        error_T = 0
        if select == 0:
            error_T = math.sqrt(error_Tin * error_Tin)
        if select == 1:
            error_T = math.sqrt(error_tsb * error_tsb)
        if select == 2:
            error_T = math.sqrt(error_Tin * error_Tin + error_tsb * error_tsb)
        if select == 3:
            error_T = df["cdQ"].mean()

        if debug:
            print(
                f"error_Tin({x})",
                error_Tin,
                error_tsb,
                error_T,
                df["cdQ"].mean(),
                select,
                Q,
            )

        tables.append([Q, error_Tin, error_tsb, error_T, df["cdQ"].mean()])

        del df_["ctsb"]
        del df_["cTin"]

        return error_T

    def myfunc(x, grad):
        if grad.size > 0:
            grad[0] = 0.0
        return error_Tin(x)

    opt.set_min_objective(myfunc)
    x = opt.optimize(init_vals)
    minf = opt.last_optimum_value()
    status = opt.last_optimize_result()
    print("optimum: x=", x, "obj=", minf, "(code = ", status, ")")

    # how to mark line with optimum value in red??
    # loop over tables, if line correspond to x[0] then change line to red: a = "\033[1;32m%s\033[0m" %a
    # #Color
    # R = "\033[0;31;40m" #RED
    # G = "\033[0;32;40m" # GREEN
    # Y = "\033[0;33;40m" # Yellow
    # B = "\033[0;34;40m" # Blue
    # N = "\033[0m" # Reset
    if status >= 0:
        for line in tables:
            tmp = 0
            for i, unknow in enumerate(unknows):
                tmp += int(line[i] == x[i])
            if tmp == len(unknows):
                for i, item in enumerate(line):
                    line[i] = "\033[1;32m{item}\033[0m"
        print("\n", tabulate.tabulate(tables, headers, tablefmt="simple"), "\n")

    optval = {}
    for i, unknow in enumerate(unknows):
        optval[unknow] = x[i]
    return (optval, status)


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
        help="specify flowrate for power cooling - one half only (default: 30 m3/h)",
        type=float,
        default=30,
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
        "--markevery",
        help="set marker every ... display method",
        type=int,
        default="800",
    )

    # define subparser: find
    subparsers = parser.add_subparsers(
        title="commands", dest="command", help="sub-command help"
    )

    # make the following options dependent to find + nlopt
    parser_nlopt = subparsers.add_parser(
        "find", help="findhT help"
    )  # , parents=[parser])
    parser_nlopt.add_argument(
        "--error",
        help="specify error (0 for hot, 1 for cold, 2 for a mix)",
        type=int,
        choices=range(0, 2),
        default=0,
    )
    parser_nlopt.add_argument(
        "--unknows",
        help="specifiy optim keys (eg h or dTh or dT;h",
        type=str,
        default="dT;h",
    )
    parser_nlopt.add_argument(
        "--tol",
        help="specifiy relative tolerances (eg h or dTh or dT;h",
        type=str,
        default="1.e-5;1.e-5",
    )
    parser_nlopt.add_argument(
        "--abstol",
        help="specifiy absolute tolerances (eg h or dTh or dT;h",
        type=str,
        default="1.e-5;1.e-5",
    )
    parser_nlopt.add_argument(
        "--algo",
        help="specifiy optim algo",
        type=str,
        choices=["Direct_L", "Direct", "CRS2", "MLSL"],
        default="Direct_L",
    )
    parser_nlopt.add_argument(
        "--local",
        help="specifiy optim algo",
        type=str,
        choices=["None", "Nelder-Mead", "Cobyla"],
        default="None",
    )
    parser_nlopt.add_argument(
        "--stopval", help="stopping criteria for nlopt", type=float, default=1.0e-2
    )
    parser_nlopt.add_argument(
        "--maxeval", help="stopping max eval for nlopt", type=int, default=1000
    )
    # parser_nlopt.set_defaults(func=optim)

    parser_nlopt = subparsers.add_parser(
        "findQ", help="findQ help"
    )  # , parents=[parser])
    parser.add_argument(
        "--Q",
        help="specify Q factor for Flow (aka cooling magnets, ex. 1)",
        type=float,
        default=1,
    )
    parser_nlopt.add_argument(
        "--error",
        help="specify error (0 for hot, 1 for cold, 2 for a mix)",
        type=int,
        choices=range(0, 2),
        default=0,
    )
    parser_nlopt.add_argument(
        "--unknows", help="specifiy optim keys (eg Q", type=str, default="Q"
    )
    parser_nlopt.add_argument(
        "--tol",
        help="specifiy relative tolerances (eg h or dTh or dT;h",
        type=str,
        default="1.e-5;1.e-5",
    )
    parser_nlopt.add_argument(
        "--abstol",
        help="specifiy absolute tolerances (eg h or dTh or dT;h",
        type=str,
        default="1.e-5;1.e-5",
    )
    parser_nlopt.add_argument(
        "--algo",
        help="specifiy optim algo",
        type=str,
        choices=["Direct_L", "Direct", "CRS2", "MLSL"],
        default="Direct_L",
    )
    parser_nlopt.add_argument(
        "--local",
        help="specifiy optim algo",
        type=str,
        choices=["None", "Nelder-Mead", "Cobyla"],
        default="None",
    )
    parser_nlopt.add_argument(
        "--stopval", help="stopping criteria for nlopt", type=float, default=1.0e-2
    )
    parser_nlopt.add_argument(
        "--maxeval", help="stopping max eval for nlopt", type=int, default=1000
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

    optkeys = []
    if args.command == "find" or args.command == "findQ":
        print("find options")
        optkeys = args.unknows.split(";")  # returns a list
        # check valid keys

    # nlopt_args = parser_nlopt.parse_args()
    # smoothed_args = parser_smoothed.parse_args()
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

    # Adapt filtering and smoothing params to run duration
    duration = mrun.getMData().getDuration()
    if duration <= 10 * tau:
        tau = min(duration // 10, 10)
        print(f"Modified smoothing param: {tau} over {duration} s run")
        args.markevery = 8 * tau

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

    # TODO: move to magnetdata
    max_tap = 0
    for i in range(1, args.nhelices + 1):
        ukey = f"Ucoil{i}"
        # print ("Ukey=%s" % ukey, (ukey in keys) )
        if ukey in mrun.getKeys():
            max_tap = i

    if max_tap != args.nhelices and max_tap != args.nhelices // 2:
        print("Check data: inconsistant U probes and helices")
        sys.exit(1)

    missing_probes = []
    for i in range(1, max_tap + 1):
        ukey = f"Ucoil{i}"
        if ukey not in mrun.getKeys():
            # Add an empty column
            # print ("Ukey=%s" % ukey, (ukey in keys) )
            mrun.getMData().addData(ukey, f"{ukey} = 0")
            missing_probes.append(i)

    if missing_probes:
        print("Missing U probes:", missing_probes)

    # TODO verify if Ucoil starts at 1 if nhelices < 14
    formula = "UH = "
    for i in range(args.nhelices + 1):
        ukey = f"Ucoil{i}"
        if ukey in mrun.getKeys():
            if i != 1:
                formula += " + "
            formula += ukey
    # print("UH", formula)
    if "UH" not in mrun.getKeys():
        mrun.getMData().addData("UH", formula)

    formula = "UB = Ucoil15 + Ucoil16"
    # print("UB", formula)
    if "UB" not in mrun.getKeys():
        mrun.getMData().addData("UB", formula)

    if "PH" not in mrun.getKeys():
        mrun.getMData().addData("PH", "PH = UH * IH")
    if "PB" not in mrun.getKeys():
        mrun.getMData().addData("PB", "PB = UB * IB")
    if "Pt" not in mrun.getKeys():
        mrun.getMData().addData("Pt", "Pt = (PH + PB)/1.e+6")

    # estimate dTH: PH / (rho * Cp * FlowH)
    mrun.getMData().addData("dTh", "dTh = PH / (1000 * 4180 * FlowH*1.e-3)")
    # estimate dTB: PB / (rho * Cp * Flow2)
    mrun.getMData().addData("dTb", "dTb = PB / (1000 * 4180 * FlowB*1.e-3)")
    # estimate Tout: ( (Tin1+dTh)*Flow1 + (Tin2+dTb)*Flow2 ) / (Flow1+Flow2)
    mrun.getMData().addData(
        "cTout", "cTout = ( (TinH+dTh)*FlowH + (TinB+dTb)*FlowB ) / (FlowH+FlowB)"
    )
    df = mrun.getMData().getPandasData(key=None)
    experiment = mrun.getInsert().replace(r"_", r"\_")

    ax = plt.gca()
    df.plot(x="t", y="Tout", ax=ax, color="blue", marker="o", alpha=0.5, markevery=800)
    df.plot(x="t", y="cTout", ax=ax, color="blue", linestyle="--")
    plt.xlabel(r"t [s]")
    plt.ylabel(r"T [C]")
    plt.title(f"{experiment}: Tout")
    plt.grid(True)
    plt.show()
    plt.close()

    # Geom specs from HX Datasheet
    Nc = int((553 - 1) / 2.0)  # (Number of plates -1)/2
    Ac = 3.0e-3 * 1.174  # Plate spacing * Plate width [m^2]
    de = 2 * 3.0e-3  # 2*Plate spacing [m]
    # coolingparams = [0.207979, 0.640259, 0.397994]  # from nominal values
    coolingparams = [0.1249, 0.65453, 0.40152]  # from student param fits
    # coolingparams = [0.07, 0.8, 0.4]

    # Compute OHTC
    df["MeanU_h"] = df.apply(
        lambda row: ((row.Flow) * 1.0e-3 + 2 * args.debit_alim / 3600.0) / (Ac * Nc),
        axis=1,
    )
    df["MeanU_c"] = df.apply(lambda row: (row.debitbrut / 3600.0) / (Ac * Nc), axis=1)
    df["Ohtc"] = df.apply(
        lambda row: w.getOHTC(
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
    df.plot(
        x="t",
        y="Ohtc",
        ax=ax,
        color="red",
        marker="o",
        alpha=0.5,
        markevery=args.markevery,
        grid=True,
    )
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

    if args.command == "find":
        # Compute Tin, Tsb
        df = mrun.getMData().getPandasData(key=None)

        if "FlowH" not in df:
            df["FlowH"] = df.apply(
                lambda row: ((row.Flow) * 1.0e-3 + (2 * args.debit_alim) / 3600.0),
                axis=1,
            )
        if "Thi" not in df:
            df["Thi"] = df.apply(
                lambda row: mixingTemp(
                    row.Flow * 1.0e-3,
                    row.BP,
                    row.Tout,
                    2 * args.debit_alim / 3600.0,
                    row.BP,
                    row.TAlimout,
                ),
                axis=1,
            )

        (opt, status) = find(
            df,
            optkeys,
            args.dT,
            args.ohtc,
            100,
            6000,
            args.algo,
            args.local,
            args.maxeval,
            args.stopval,
            select=args.error,
            site=args.site,
            debit_alim=args.debit_alim,
            debug=args.debug,
        )

        if status < 0:
            print(f"Optimization {args.algo} failed with error={status}")
            sys.exit(1)

        dT = args.dT
        h = args.ohtc
        for key in optkeys:
            if key == "dT":
                dT = opt["dT"]
            elif key == "h":
                h = opt["h"]

        # Get solution for optimum
        display_T(
            args.input_file,
            f_extension,
            mrun,
            "ctsb",
            "cTin",
            args.debit_alim,
            h,
            dT,
            args.show,
            "-T-find.png",
        )
        display_Q(
            args.input_file,
            f_extension,
            mrun,
            args.debit_alim,
            h,
            dT,
            args.show,
            "-Q-find.png",
        )

    if args.command == "findQ":
        # Compute Q
        df = mrun.getData()

        if "FlowH" not in df:
            df["FlowH"] = df.apply(
                lambda row: ((row.Flow) * 1.0e-3 + (2 * args.debit_alim) / 3600.0),
                axis=1,
            )
        if "Thi" not in df:
            df["Thi"] = df.apply(
                lambda row: mixingTemp(
                    row.Flow * 1.0e-3,
                    row.BP,
                    row.Tout,
                    2 * args.debit_alim / 3600.0,
                    row.BP,
                    row.TAlimout,
                ),
                axis=1,
            )

        (opt, status) = findQ(
            df,
            optkeys,
            args.Q,
            0.5,
            2,
            args.algo,
            args.local,
            args.maxeval,
            args.stopval,
            select=args.error,
            site=args.site,
            debit_alim=args.debit_alim,
            debug=args.debug,
        )

        if status < 0:
            print(f"Optimization {args.algo} failed with error={status}")
            sys.exit(1)

        Q = args.Q
        for key in optkeys:
            if key == "Q":
                Q = opt["Q"]

        # Get solution for optimum
        display_T(
            args.input_file,
            f_extension,
            mrun,
            "ctsb",
            "cTin",
            args.debit_alim,
            args.ohtc,
            args.dT,
            args.show,
            "-T-findQ.png",
        )
        display_Q(
            args.input_file,
            f_extension,
            mrun,
            args.debit_alim,
            args.ohtc,
            args.dT,
            args.show,
            "-Q-findQ.png",
        )
