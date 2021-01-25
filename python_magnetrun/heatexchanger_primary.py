#! /usr/bin/python3

from __future__ import unicode_literals

import math
import os
import sys
import datetime
import python_magnetrun

import numpy as np
import matplotlib
# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available
import matplotlib.pyplot as plt

import pandas as pd
import freesteam as st

import ht
import tabulate
import datatools

tables = []

def display_Q(inputfile, f_extension, mrun, debit_alim, ohtc, dT, show=False, extension="-Q.png"):
    """
    plot Heat profiles
    """

    df = mrun.getData()

    df['QNTU'] = df.apply(lambda row: heatexchange(ohtc, dT, row.teb, row.Tout, row.debitbrut, row.Flow, 10, row.BP, row.PH, row.PB, args.site, args.debit_alim)[2]/1.e+6, axis=1)

    df["Qhot"] = df.apply(lambda row: ((row.Flow)*1.e-3+0/3600.)*(rho(row.BP, row.Tout)*cp(row.BP, row.Tout)*(row.Tout+0)-rho(row.HP, row.Tin)*cp(row.HP, row.Tin)*row.Tin)/1.e+6, axis=1)
    df["Qhot1"] = df.apply(lambda row: ((row.Flow)*1.e-3+(2*debit_alim)/3600.)*(rho(row.BP, row.Tout)*cp(row.BP, row.Tout)*(row.Tout+dT)-rho(row.HP, row.Tin)*cp(row.HP, row.Tin)*row.Tin)/1.e+6, axis=1)
    df["Qcold"] = df.apply(lambda row: row.debitbrut/3600.*(rho(10, row.tsb)*cp(10, row.tsb)*row.tsb-rho(10, row.teb)*cp(10, row.teb)*row.teb)/1.e+6, axis=1)

    ax = plt.gca()
    df.plot(x='t', y='Qhot', ax=ax, color='red')
    df.plot(x='t', y='Qhot1', ax=ax, color='blue', marker='o', alpha = .5, markevery=args.markevery)
    df.plot(x='t', y='Qcold', ax=ax, color='green')
    df.plot(x='t', y='Pt', ax=ax, color='yellow')
    plt.ylabel(r'Q[MW]')
    plt.xlabel(r't [s]')
    plt.grid(b=True)
    plt.title(mrun.getInsert().replace(r"_",r"\_") + ": h=%g $W/m^2/K$, dT=%g" % (ohtc,dT))
    if show:
        plt.show()
    else:
        imagefile = inputfile.replace(f_extension, extension)
        print("save to %s" % imagefile)
        plt.savefig(imagefile, dpi=300)
        plt.close()

def display_T(inputfile, f_extension, mrun, tsb_key, tin_key, debit_alim, ohtc, dT, show=False, extension="-coolingloop.png", debug=False):
    """
    plot Temperature profiles
    """

    df = mrun.getData()
    df[tin_key] = df.apply(lambda row: heatexchange(ohtc, dT, row.teb, row.Tout, row.debitbrut, row.Flow, 10, row.BP, row.PH, row.PB, args.site, debit_alim, debug)[1], axis=1)
    df[tsb_key] = df.apply(lambda row: heatexchange(ohtc, dT, row.teb, row.Tout, row.debitbrut, row.Flow, 10, row.BP, row.PH, row.PB, args.site, debit_alim)[0], axis=1)

    ax = plt.gca()
    df.plot(x='t', y=tsb_key, ax=ax, color='blue', marker='o', alpha = .5, markevery=args.markevery)
    df.plot(x='t', y='tsb', ax=ax, color='blue')
    df.plot(x='t', y='teb', ax=ax, color='blue', linestyle='--')
    df.plot(x='t', y=tin_key, ax=ax, color='red', marker='o', alpha = .5, markevery=args.markevery)
    df.plot(x='t', y='Tin', ax=ax, color='red')
    df.plot(x='t', y='Tout', ax=ax, color='red', linestyle='--')
    plt.xlabel(r't [s]')
    plt.grid(b=True)
    plt.title(mrun.getInsert().replace(r"_",r"\_") + ": h=%g $W/m^2/K$, dT=%g" % (ohtc,dT))

    if show:
        plt.show()
    else:
        imagefile = inputfile.replace(f_extension, extension)
        print("save to %s" % imagefile)
        plt.savefig(imagefile, dpi=300)
    plt.close()

def rho(pbar, celsius):
    """
    compute water volumic mass as a function
    of pressure and temperature
    """

    pascal = pbar * 1e+5
    kelvin = celsius+273.
    rho = st.steam_pT(pascal, kelvin).rho
    # print("rho(%g,%g)=%g" % (pbar,celsius,rho))
    return rho

def cp(pbar, celsius):
    """
    compute water volumic specific heat as a function
    of pressure and temperature
    """

    pascal = pbar * 1e+5
    kelvin = celsius+273.
    cp = st.steam_pT(pascal, kelvin).cp
    # print("cp(%g,%g)=%g" % (pbar,celsius,cp))
    return cp

def heatexchange(h, dT, Tci, Thi, Debitc, Debith, Pci, Phi, PowerH, PowerB, site, DebitA, debug=False):
    """
    NTU Model for heat Exchanger

    compute the output temperature for the heat exchanger
    as a function of input temperatures and flow rates

    Debitc: m3/h
    Debith: l/s
    DebitA: m3/h
    """

    if h is None:
        U = 4041 # 4485 # W/m^2/K
    else:
        U = float(h)

    # if debug:
    #     print("heatexchange:",
    #           "h=", U,
    #           "Tci=", Tci, "Thi=", Thi,
    #           "Pci=", Pci, "Phi=", Phi,
    #           "Debitc=", Debitc, "Debith=", Debith, "DebitA=", DebitA)

    A = 1063.4 # m^2
    Cp_cold = cp(Pci, Tci) # J/kg/K
    Cp_hot = cp(Phi, Thi) # J/kg/K
    m_hot = rho(Phi, Thi) * Debith * 1.e-3 # kg/s
    m_cold = rho(Pci, Tci) * Debitc / 3600. # kg/s

    # Alim
    DebitA1A2 = DebitA # m³/h  # plutot 40 m³/h
    DebitA3A4 = DebitA # m³/h
    m_alim_A1A2 = rho(Phi, Thi) * DebitA1A2 / 3600. # Tho plutot?
    m_alim_A3A4 = rho(Phi, Thi) * DebitA3A4 / 3600. # ..........?
    # if site == "M9":
    #     # P_A1A2 = ? xx % of PowerH or PowerB depending on site
    #     # P_A3A3 = ? yy % of PowerH or PowerB depending on site
    # elif site in ["M8", "M10"]:
    #     # P_A1A2 = ? xx % of PowerH or PowerB depending on site
    #     # P_A3A3 = ? yy % of PowerH or PowerB depending on site
    # dT_A1A2 = P_A1A2/m_alim_A1A2
    # dT_A3A4 = P_A1A2/m_alim_A1A2

    m_hot += m_alim_A1A2
    m_hot += m_alim_A3A4
    # dThi = Thi * m_hot/(m_hot + m_alim_A1A2 + m_alim_A3A4)
    # dThi += (Tho+dT_A1A2) * m_alim_A1A2/(m_hot + m_alim_A1A2 + m_alim_A3A4)
    # dThi += (Tho+dT_A3A4) * m_alim_A3A4/(m_hot + m_alim_A1A2 + m_alim_A3A4)
    # dThi -= Thi
    Thi += dT # aka dThi

    # For plate exchanger
    result = ht.hx.P_NTU_method(m_hot, m_cold, Cp_hot, Cp_cold, UA=U*A, T1i=Thi, T2i=Tci, subtype='1/1')

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
    if NTU == float('inf') or math.isnan(NTU):
        print("Tci=", Tci, "Thi=", Thi)
        print("Pci=", Tci, "Phi=", Thi)
        print("Debitc=", Debitc, "Debith=", Debith, "DebitA=", DebitA)
        raise  Exception("NTU not valid")

    Q = result["Q"]
    if Q  == float('inf') or math.isnan(Q):
        print("Tci=", Tci, "Thi=", Thi)
        print("Pci=", Tci, "Phi=", Thi)
        print("Debitc=", Debitc, "Debith=", Debith, "DebitA=", DebitA)
        raise  Exception("Q not valid")

    Tco = result["T2o"]
    if Tco  == None:
        print("h=", U)
        print("Tci=", Tci, "Thi=", Thi)
        print("Pci=", Tci, "Phi=", Thi)
        print("Debitc=", Debitc, "Debith=", Debith, "DebitA=", DebitA)
        raise  Exception("Tco not valid")
    Tho = result["T1o"]
    if Tho  == None:
        print("h=", U)
        print("Tci=", Tci, "Thi=", Thi)
        print("Pci=", Tci, "Phi=", Thi)
        print("Debitc=", Debitc, "Debith=", Debith, "DebitA=", DebitA)
        raise  Exception("Tho not valid")

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
    return (Tco, Tho, Q)

def find(df, 
         unknows: list, 
         dTini: float, hini: float, hmin: float, hmax: float, 
         algo: str, lalgo: str, maxeval: float, stopval: float, select=0, 
         site="M9", debit_alim="30", debug=False):
    """
    Use nlopt to find h, dT that give the best approximation for Hx output temperature

    unknows = list of optim var (eg ["dT"] or ["h", "dT"])
    returns a dict
    """

    tables = []
    headers = ["dT[C]", "h[W/m\u00b2/K]", "e_Tin[]", "e_tsb[]", "e_T[]", "Heat Balance[MW]"]

    import nlopt
    print("find %d params:" %  len(unknows), unknows)

    opt = None
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
    opt.set_ftol_abs(1.e-5)
    # opt.set_xtol_rel([tol, tol]) if 2 params? or float?
    # opt.set_xtol_abs([1.e-5, 1.e-5]) if 2 opt params
    if args.debug:
        print("nlopt [ftol fabs xtol xabs]: ", opt.get_ftol_rel(), opt.get_ftol_abs() , opt.get_xtol_rel(), opt.get_xtol_abs() )
    print("nlopt [ftol fabs xtol xabs]: ", opt.get_ftol_rel(), opt.get_ftol_abs() , opt.get_xtol_rel(), opt.get_xtol_abs() )

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

    def error_Tin(x, df_=df, unknows: list=unknows, hini: float=hini, dTini: float=dTini, select: int=select, debug: bool=debug):
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

        df_['cTin'] = df_.apply(lambda row: heatexchange(ohtc, dT, row.teb, row.Tout, row.debitbrut, row.Flow, 10, row.BP, row.PH, row.PB, site, debit_alim)[1], axis=1)
        diff =  np.abs(df_["Tin"] - df_['cTin'])
        L2_Tin = math.sqrt(np.dot( df_['Tin'], df_['Tin'] ))
        error_Tin = math.sqrt(np.dot( diff, diff )) /L2_Tin # diff.size

        df_['ctsb'] = df_.apply(lambda row: heatexchange(ohtc, dT, row.teb, row.Tout, row.debitbrut, row.Flow, 10, row.BP, row.PH, row.PB, site, debit_alim)[0], axis=1)
        diff =  np.abs(df_["tsb"] - df_['ctsb'])
        L2_tsb = math.sqrt(np.dot( df_['tsb'], df_['tsb'] ))
        error_tsb = math.sqrt(np.dot( diff, diff )) / L2_tsb #diff.size

        df["cQhot"] = df.apply(lambda row: ((row.Flow)*1.e-3+(2*debit_alim)/3600.)*(rho(row.BP, row.Tout)*cp(row.BP, row.Tout)*(row.Tout+dT)-rho(row.HP, row.cTin)*cp(row.HP, row.cTin)*row.cTin)/1.e+6, axis=1)
        df["cQcold"] = df.apply(lambda row: row.debitbrut/3600.*(rho(10, row.ctsb)*cp(10, row.ctsb)*row.ctsb-rho(10, row.teb)*cp(10, row.teb)*row.teb)/1.e+6, axis=1)
        df["cdQ"] = df.apply(lambda row: row.cQhot - row.cQcold, axis=1)
        
        df["Qhot"] = df.apply(lambda row: ((row.Flow)*1.e-3+(2*debit_alim)/3600.)*(rho(row.BP, row.Tout)*cp(row.BP, row.Tout)*(row.Tout+dT)-rho(row.HP, row.Tin)*cp(row.HP, row.Tin)*row.cTin)/1.e+6, axis=1)
        df["Qcold"] = df.apply(lambda row: row.debitbrut/3600.*(rho(10, row.ctsb)*cp(10, row.tsb)*row.tsb-rho(10, row.teb)*cp(10, row.teb)*row.teb)/1.e+6, axis=1)
        df["dQ"] = df.apply(lambda row: row.Qhot - row.Qcold, axis=1)

        diff =  np.abs(df_["Qhot"] - df_['cQhot'])
        L2_Qhot = math.sqrt(np.dot( df_['Qhot'], df_['Qhot'] ))
        error_qhot = math.sqrt(np.dot( diff, diff )) / L2_Qhot

        diff =  np.abs(df_["Qcold"] - df_['cQcold'])
        L2_Qcold = math.sqrt(np.dot( df_['Qcold'], df_['Qcold'] ))
        error_qcold = math.sqrt(np.dot( diff, diff )) / L2_Qcold

        error_T = 0
        if select == 0:
            error_T = math.sqrt(error_Tin*error_Tin)
        if select == 1:
            error_T = math.sqrt(error_tsb*error_tsb)
        if select == 2:
            error_T = math.sqrt(error_Tin*error_Tin + error_tsb*error_tsb)
        if select == 3:
            error_T = df["cdQ"].mean()

        if debug:
            print("error_Tin(%s)" % x, error_Tin, error_tsb, error_T, df["cdQ"].mean(), select, ohtc, dT)

        tables.append([dT, ohtc, error_Tin, error_tsb, error_T, df["cdQ"].mean()])

        del df_['ctsb']
        del df_['cTin']

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
                for i,item in enumerate(line):
                    line[i] = "\033[1;32m%s\033[0m" % item
        print( "\n", tabulate.tabulate(tables, headers, tablefmt="simple"), "\n")

    optval = {}
    for i,unknow in enumerate(unknows):
        optval[unknow] = x[i]
    return (optval, status)

if __name__ == "__main__":

    command_line = None

    import argparse
    parser = argparse.ArgumentParser("Cooling loop Heat Exchanger")
    parser.add_argument("input_file", help="input txt file (ex. M10_2020.10.04_20-2009_43_31.txt)")
    parser.add_argument("--nhelices", help="specify number of helices", type=int, default=14)
    parser.add_argument("--ohtc", help="specify heat exchange coefficient (ex. 4000)", type=float, default=4000)
    parser.add_argument("--dT", help="specify dT for Tout (aka accounting for alim cooling, ex. 0)", type=float, default=0)
    parser.add_argument("--site", help="specify a site (ex. M8, M9,...)", type=str)
    parser.add_argument("--debit_alim", help="specify flowrate for power cooling (default: 30 m3/h)", type=float, default=30)
    parser.add_argument("--show", help="display graphs (requires X11 server active)", action='store_true')
    parser.add_argument("--debug", help="activate debug mode", action='store_true')
    # parser.add_argument("--save", help="save graphs to png", action='store_true')

    # raw|filter|smooth post-traitement of data
    parser.add_argument("--pre", help="select a pre-traitment for data", type=str, choices=['raw','filtered','smoothed'], default='smoothed')
    # define params for post traitment of data
    parser.add_argument("--pre_params", help="pass param for pre-traitment method", type=str, default='400')
    parser.add_argument("--markevery", help="set marker every ... display method", type=int, default='800')

    # define subparser: find
    subparsers = parser.add_subparsers(title="commands", dest="command", help='sub-command help')

    # make the following options dependent to find + nlopt
    parser_nlopt = subparsers.add_parser('find', help='findh help') #, parents=[parser])
    parser_nlopt.add_argument("--error", help="specify error (0 for hot, 1 for cold, 2 for a mix)", type=int, choices=range(0, 2), default=0)
    parser_nlopt.add_argument("--unknows", help="specifiy optim keys (eg h or dTh or dT;h", type=str, default="dT;h")
    parser_nlopt.add_argument("--tol", help="specifiy relative tolerances (eg h or dTh or dT;h", type=str, default="1.e-5;1.e-5")
    parser_nlopt.add_argument("--abstol", help="specifiy absolute tolerances (eg h or dTh or dT;h", type=str, default="1.e-5;1.e-5")
    parser_nlopt.add_argument("--algo", help="specifiy optim algo", type=str, choices=["Direct_L", "Direct", "CRS2", "MLSL"], default="Direct_L")
    parser_nlopt.add_argument("--local", help="specifiy optim algo", type=str, choices=["None", "Nelder-Mead", "Cobyla"], default="None")
    parser_nlopt.add_argument("--stopval", help="stopping criteria for nlopt", type=float, default=1.e-2)
    parser_nlopt.add_argument("--maxeval", help="stopping max eval for nlopt", type=int, default=1000)
    #parser_nlopt.set_defaults(func=optim)

    args = parser.parse_args(command_line)

    tau = 400
    if args.pre == 'smoothed':
        print("smoothed options")
        tau = float(args.pre_params)

    threshold = 0.5
    twindows = 10
    if args.pre == 'filtered':
        print("filtered options")
        params = args.pre_params.split(';')
        threshold = float(params[0])
        twindows = int(params[1])

    optkeys = []
    if args.command == 'find':
        print("find options")
        optkeys = args.unknows.split(";") # returns a list
        # check valid keys

    # nlopt_args = parser_nlopt.parse_args()
    # smoothed_args = parser_smoothed.parse_args()
    print("args: ", args)

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
            pass

    mrun = python_magnetrun.MagnetRun.fromtxt(args.site, args.input_file)
    if not args.site:
        args.site = mrun.getSite()

    # Adapt filtering and smoothing params to run duration
    duration = mrun.getDuration()
    if duration <= 10*tau:
        tau = min(duration // 10, 10)
        print("Modified smoothing param: %g over %g s run", tau, duration)
        args.markevery = 2 * tau

    # print("type(mrun):", type(mrun))
    mrun.getMData().addTime()
    start_timestamp = mrun.getMData().getStartDate()

    mrun.getMData().addData("Flow", "Flow = Flow1 + Flow2")
    mrun.getMData().addData("Tin", "Tin = (Tin1 + Tin2)/2.")
    mrun.getMData().addData("HP", "HP = (HP1 + HP2)/2.")

    # extract data
    keys = ["t", "teb", "tsb", "debitbrut", "Tout", "Tin", "Flow", "BP", "HP", "Pmagnet"]
    units = ["s","C","C","m\u00B3/h","C","C","l/s","bar", "MW"]
    # df = mrun.getMData().extractData(keys)

    if args.debug:
        pd.set_option("display.max_rows", None, "display.max_columns", None)

    # filter spikes
    # see: https://ocefpaf.github.io/python4oceanographers/blog/2015/03/16/outlier_detection/
    if args.pre == 'filtered':
        for key in ["debitbrut", "Flow"]:
            mrun = datatools.filterpikes(mrun, key, inplace=True, threshold=threshold, twindows=twindows, debug=args.debug, show=args.show, input_file=args.input_file)
        print("Filtered pikes done")

    # TODO: move to magnetdata
    max_tap = 0
    for i in range(1,args.nhelices+1):
        ukey = "Ucoil%d" % i
        # print ("Ukey=%s" % ukey, (ukey in keys) )
        if ukey in mrun.getKeys():
            max_tap=i

    if max_tap != args.nhelices and max_tap != args.nhelices//2:
        print("Check data: inconsistant U probes and helices")
        sys.exit(1)

    missing_probes=[]
    for i in range(1,max_tap+1):
        ukey = "Ucoil%d" % i
        if not ukey in mrun.getKeys():
            # Add an empty column
            # print ("Ukey=%s" % ukey, (ukey in keys) )
            mrun.getMData().addData(ukey, "%s = 0" % ukey)
            missing_probes.append(i)

    if missing_probes:
        print("Missing U probes:", missing_probes)

    # TODO verify if Ucoil starts at 1 if nhelices < 14
    formula = "UH = "
    for i in range(args.nhelices+1):
        ukey = "Ucoil%d" % i
        if ukey in mrun.getKeys():
            if i != 1:
                formula += " + "
            formula += ukey
    # print("UH", formula)
    mrun.getMData().addData("UH", formula)

    formula = "UB = Ucoil15 + Ucoil16"
    # print("UB", formula)
    mrun.getMData().addData("UB", formula)

    mrun.getMData().addData("PH", "PH = UH * IH")
    mrun.getMData().addData("PB", "PB = UB * IB")
    mrun.getMData().addData("Pt", "Pt = (PH + PB)/1.e+6")

    # smooth data Locally Weighted Linear Regression (Loess)
    # see: https://xavierbourretsicotte.github.io/loess.html(
    if args.pre == 'smoothed':
        for key in ["debitbrut", "Flow", "teb", "Tout", "PH", "PB", "Pt"]:
            mrun = datatools.smooth(mrun, key, inplace=True, tau=tau, debug=args.debug, show=args.show, input_file=args.input_file)
        print("smooth data done")

    display_T(args.input_file, f_extension, mrun, 'itsb', 'iTin', args.debit_alim, args.ohtc, args.dT, args.show, "-coolingloop.png", args.debug)
    display_Q(args.input_file, f_extension, mrun, args.debit_alim, args.ohtc, args.dT, args.show, "-Q.png")
    
    if args.command == 'find':
        # Compute Tin, Tsb
        df = mrun.getData()
        (opt, status) = find(df, optkeys, args.dT, args.ohtc, 100, 6000, args.algo, args.local, args.maxeval, args.stopval, select=args.error, site=args.site, debit_alim=args.debit_alim, debug=args.debug)

        if status < 0:
            print("Optimization %s failed with %d error: ", (args.algo, status) )
            sys.exit(1)
            
        dT = args.dT
        h = args.ohtc
        for key in optkeys:
            if key == "dT":
                dT = opt["dT"]
            elif key == "h":
                h = opt["h"]

        # Get solution for optimum
        display_T(args.input_file, f_extension, mrun, 'ctsb', 'cTin', args.debit_alim, h, dT, args.show, "-T-find.png")
        display_Q(args.input_file, f_extension, mrun, args.debit_alim, h, dT, args.show, "-Q-find.png")

        
