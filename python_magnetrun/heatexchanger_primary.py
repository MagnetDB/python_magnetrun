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

tables = []

def rho(pbar, celsius):
    """
    compute water volumic mass as a function 
    of pressure and temperature
    """

    pascal = pbar * 1e+5
    kelvin = celsius+273.
    rho = st.steam_pT(pascal, kelvin).rho
    return rho

def cp(pbar, celsius):
    """
    compute water volumic specific heat as a function 
    of pressure and temperature
    """

    pascal = pbar * 1e+5
    kelvin = celsius+273.
    cp = st.steam_pT(pascal, kelvin).cp
    return cp

# steam/oil crossflow
# internal (magnet) / external (drac) counterflow

#subtype='crossflow, mixed Cmax'
#subtype='counterflow'
def heatexchange(h, Tci, Thi, Debitc, Debith, Pci, Phi, subtype='counterflow'):
    """
    NTU Model for heat Exchanger

    compute the output temperature for the heat exchanger
    as a function of input temperatures and flow rates
    """

    if h is None:
        U = 4041 # 4485 # W/m^2/K
    else:
        U = float(h)

    A = 1063.4 # m^2
    Cp_cold = cp(Pci, Tci) # J/kg/K
    Cp_hot = cp(Phi,Thi) # J/kg/K
    m_hot = rho(Phi,Thi)*Debith*1.e-3 #5.2 # kg/s rho(Tout)*(Flow1+Flow2)??
    m_cold = rho(Pci, Tci)*Debitc/3600. #0.725 # kg/s rho*(teb)*debitbrut??

    Cmin = ht.calc_Cmin(mh=m_hot, mc=m_cold, Cph=Cp_hot, Cpc=Cp_cold)
    Cmax = ht.calc_Cmax(mh=m_hot, mc=m_cold, Cph=Cp_hot, Cpc=Cp_cold) # ?? 36000 kW??
    Cr = ht.calc_Cr(mh=m_hot, mc=m_cold, Cph=Cp_hot, Cpc=Cp_cold)
    NTU = ht.NTU_from_UA(UA=U*A, Cmin=Cmin)
    #eff = effectiveness_from_NTU(NTU=NTU, Cr=Cr, subtype='crossflow, mixed Cmax')
    eff = ht.effectiveness_from_NTU(NTU=NTU, Cr=Cr, subtype=subtype)
    Q = eff*Cmin*(Thi - Tci)
    Tco = Tci + Q/(m_cold*Cp_cold)
    Tho = Thi - Q/(m_hot*Cp_hot)

    #print("NTU=", NTU, "eff=", eff, "Q=", Q, "Cmax=", Cmax, "Cr=", Cr, "Cp_cold=", Cp_cold, "Cp_hot=", Cp_hot, "m_cold=", m_cold, "m_hot=", m_hot)

    return (Tco, Tho, Q)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="input txt file (ex. M10_2020.10.04_20-2009_43_31.txt)")
    parser.add_argument("--site", help="specify a site (ex. M8, M9,...)", default="M9")
    parser.add_argument("--show", help="display graphs (requires X11 server active)", action='store_true')
    parser.add_argument("--save", help="save graphs to png", action='store_true')
    parser.add_argument("--stopval", help="stopping criteria for nlopt", type=float, default=1.e-2)
    parser.add_argument("--subtype", help="specify type of heat exchanger", type=str, default='counterflow')
    args = parser.parse_args()

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
    dkeys = mrun.getKeys()

    ax = plt.gca()
    # if args.show:
    #     mrun.getMData().plotData(x='Time', y='Field', ax=ax)
    #     plt.xlabel(r't [s]')
    #     plt.ylabel(r'B [T]')
    #     plt.grid(b=True)
    #     plt.title(mrun.getInsert().replace("_","\_")) # replace _ by \_ before adding title
    #     plt.show()

    # extract data
    keys = ["Date", "Time", "teb", "tsb", "debitbrut", "Tout", "Tin1", "Flow1", "Tin2", "Flow2", "BP", "HP1", "HP2"]
    units = ["C","C","m\u00B3/h","C","C","l/s","C","l/s","bar","bar","bar"]
    df = mrun.getMData().extractData(keys)

    # Add a time column
    tformat="%Y.%m.%d %H:%M:%S"
    start_date=df["Date"].iloc[0]
    start_time=df["Time"].iloc[0]
    end_time=df["Time"].iloc[-1]
    print ("start_time=", start_time, "start_date=", start_date)

    t0 = datetime.datetime.strptime(df['Date'].iloc[0]+" "+df['Time'].iloc[0], tformat)

    df["t"] = df.apply(lambda row: (datetime.datetime.strptime(row.Date+" "+row.Time, tformat)-t0).total_seconds(), axis=1)
    units.append("s")

    del df['Date']
    del df['Time']

    # Compute Tin, Tsb
    h = 4000
    tables = []
    headers = ["h[W/m\u00b2/K]", "e_Tin[]", "e_tsb[]", "e_T[]"]

    import nlopt
    opt = nlopt.opt(nlopt.GN_DIRECT_L, 1)

    # use *f_data to pass extra args: df_, subtype
    # ex:
    # fdata = (df_, sbutype)
    # error_Tin(x, **fdata)
    # df_ = fdata[0], subtype = fdata[1], debug = fdata[2]
    # eventually check type with isinstanceof() / type()
    # question: how to take into account error_tsb also??
    
    def error_Tin(x, df_=df, subtype=args.subtype):
        # Now loop over h to find best h
        
        df_['cTin'] = df_.apply(lambda row: heatexchange(x[0], row.teb, row.Tout, row.debitbrut, row.Flow1+row.Flow2, 10, row.BP, subtype)[1], axis=1)
        df_['ctsb'] = df_.apply(lambda row: heatexchange(x[0], row.teb, row.Tout, row.debitbrut, row.Flow1+row.Flow2, 10, row.BP, subtype)[0], axis=1)

        diff =  np.abs((df_["Tin1"] + df_['Tin2'])/2. - df_['cTin'])
        error_Tin = math.sqrt(np.dot( diff, diff )) / diff.size

        diff =  np.abs(df_["tsb"] - df_['ctsb'])
        error_tsb = math.sqrt(np.dot( diff, diff )) / diff.size

        error_T = math.sqrt(error_Tin*error_Tin + error_tsb*error_tsb)
        
        # print("error_Tin(%g)" % x, error_Tin, error_tsb, error_T)

        tables.append([x[0], error_Tin, error_tsb, error_T])

        del df_['ctsb']
        del df_['cTin']
        
        return error_Tin #error_T

    def myfunc(x, grad):
        if grad.size > 0:
            grad[0] = 0.0
        return error_Tin(x)

    opt.set_min_objective(myfunc)
    opt.set_ftol_rel(args.stopval)
    # opt.set_ftol_rel(args.stopval)
    print("nlopt [ftol fabs xtol xabs]: ", opt.get_ftol_rel(), opt.get_ftol_abs() , opt.get_xtol_rel(), opt.get_xtol_abs() )
    opt.set_lower_bounds(100)
    opt.set_upper_bounds(4500)
    x = opt.optimize([4000.])
    minf = opt.last_optimum_value()
    print("optimum: x=", x[0], "obj=", minf, "(code = ", opt.last_optimize_result(), ")")

    # how to mark line with optimum value in red??
    # loop over tables, if line correspond to x[0] then change line to red: a = "\033[1;32m%s\033[0m" %a
    # #Color
    # R = "\033[0;31;40m" #RED
    # G = "\033[0;32;40m" # GREEN
    # Y = "\033[0;33;40m" # Yellow
    # B = "\033[0;34;40m" # Blue
    # N = "\033[0m" # Reset
    for line in tables:
        if line[0] == x[0]:
            for i,item in enumerate(line):
                line[i] = "\033[1;32m%s\033[0m" % item
    print( "\n", tabulate.tabulate(tables, headers, tablefmt="simple"), "\n")

    # Get solution for optimum
    df['cTin'] = df.apply(lambda row: heatexchange(x[0], row.teb, row.Tout, row.debitbrut, row.Flow1+row.Flow2, 10, row.BP, args.subtype)[1], axis=1)
    df['ctsb'] = df.apply(lambda row: heatexchange(x[0], row.teb, row.Tout, row.debitbrut, row.Flow1+row.Flow2, 10, row.BP, args.subtype)[0], axis=1)
    ax = plt.gca()
    df.plot(x='t', y='ctsb', ax=ax) #, color='blue')
    df.plot(x='t', y='tsb', ax=ax) #, color='blue', linestyle='-')
    df.plot(x='t', y='cTin', ax=ax) #, color='red')
    df.plot(x='t', y='Tin1', ax=ax) #, color='red', linestyle='--')
    df.plot(x='t', y='Tin2', ax=ax) #, color='red', linestyle='-.')
    plt.xlabel(r't [s]')
    plt.grid(b=True)
    plt.title(mrun.getInsert().replace("_","\_") + " (" + args.subtype + "): h=%g $W/m^2/K$" % x[0])

    if args.show:
        plt.show()
    if args.save:
        imagefile = args.input_file.replace(f_extension, "-h.png")
        print("save to %s" % imagefile)
        plt.savefig('%s.png' % imagefile, dpi=300)
        plt.close()

    df['QNTU'] = df.apply(lambda row: heatexchange(x[0], row.teb, row.Tout, row.debitbrut, row.Flow1+row.Flow2, 10, row.BP, args.subtype)[2], axis=1)
    df["Qhprimaire"] = df.apply(lambda row: (row.Flow1+row.Flow2)*1.e-3*(rho(row.BP, row.Tout)*cp(row.BP, row.Tout)*row.Tout-rho(row.HP1, row.Tin1)*cp(row.HP1, row.Tin1)*row.Tin1), axis=1)
    df["Qcprimaire"] = df.apply(lambda row: row.debitbrut/3600.*(rho(10, row.Tout)*cp(10, row.tsb)*row.tsb-rho(10, row.teb)*cp(10, row.teb)*row.Tin1), axis=1)
        
    ax = plt.gca()
    df.plot(x='t', y='QNTU', ax=ax, color='red')
    df.plot(x='t', y='Qhprimaire', ax=ax, color='blue')
    df.plot(x='t', y='Qcprimaire', ax=ax, color='green')
    plt.xlabel(r't [s]')
    plt.ylabel(r'Q[W]')
    plt.grid(b=True)
    plt.title(mrun.getInsert().replace("_","\_") + "(" + args.subtype + ")")
    if args.show:
        plt.show()
    if args.save:
        imagefile = args.input_file.replace(f_extension, "-Q.png")
        print("save to %s" % imagefile)
        plt.savefig('%s.png' % imagefile, dpi=300)
        plt.close()
