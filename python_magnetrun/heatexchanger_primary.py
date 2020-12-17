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

# steam/oil crossflow
# internal (magnet) / external (drac) counterflow

#subtype='crossflow, mixed Cmax'
#subtype='counterflow'
def heatexchange(h, Tci, Thi, Debitc, Debith, Pci, Phi):
    """
    NTU Model for heat Exchanger

    compute the output temperature for the heat exchanger
    as a function of input temperatures and flow rates
    """

    subtype='1/1'
    
    if h is None:
        U = 4041 # 4485 # W/m^2/K
    else:
        U = float(h)

    # print("heatexchange:")
    # print("h=", U)
    # print("Tci=", Tci, "Thi=", Thi)
    # print("Pci=", Tci, "Phi=", Thi)
    # print("Debitc=", Debitc, "Debith=", Debith)

    A = 1063.4 # m^2
    Cp_cold = cp(Pci, Tci) # J/kg/K
    Cp_hot = cp(Phi, Thi) # J/kg/K
    m_hot = rho(Phi, Thi) * Debith * 1.e-3 # kg/s 
    m_cold = rho(Pci, Tci) * Debitc / 3600. # kg/s

    # For plate exchanger
    result = ht.hx.P_NTU_method(m_hot, m_cold, Cp_hot, Cp_cold, UA=U*A, T1i=Thi, T2i=Tci, subtype='1/1')
    # print("result=", result)
    # sys.exit(1)

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
        print("Debitc=", Debitc, "Debith=", Debith)
        raise  Exception("NTU not valid")

    Q = result["Q"]
    if Q  == float('inf') or math.isnan(Q):
        print("Tci=", Tci, "Thi=", Thi)
        print("Pci=", Tci, "Phi=", Thi)
        print("Debitc=", Debitc, "Debith=", Debith)
        raise  Exception("Q not valid")

    Tco = result["T2o"]
    if Tco  == None:
        print("h=", U)
        print("Tci=", Tci, "Thi=", Thi)
        print("Pci=", Tci, "Phi=", Thi)
        print("Debitc=", Debitc, "Debith=", Debith)
        raise  Exception("Tco not valid")
    Tho = result["T1o"]
    if Tho  == None:
        print("h=", U)
        print("Tci=", Tci, "Thi=", Thi)
        print("Pci=", Tci, "Phi=", Thi)
        print("Debitc=", Debitc, "Debith=", Debith)
        raise  Exception("Tho not valid")
    # print("heatexchange: ", NTU, Tco, Tho, Q)
    return (Tco, Tho, Q)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="input txt file (ex. M10_2020.10.04_20-2009_43_31.txt)")
    parser.add_argument("--site", help="specify a site (ex. M8, M9,...)", default="M9")
    parser.add_argument("--show", help="display graphs (requires X11 server active)", action='store_true')
    parser.add_argument("--save", help="save graphs to png", action='store_true')
    parser.add_argument("--stopval", help="stopping criteria for nlopt", type=float, default=1.e-2)
    parser.add_argument("--threshold", help="stopping criteria for nlopt", type=float, default=0.5)
    parser.add_argument("--window", help="stopping criteria for nlopt", type=int, default=10)
    parser.add_argument("--debug", help="activate debug mode", action='store_true')
        
    args = parser.parse_args()

    threshold = args.threshold
    twindows = args.window
    
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
    mrun.getMData().addTime()
    start_timestamp = mrun.getMData().getStartDate()

    mrun.getMData().addData("Flow", "Flow = Flow1 + Flow2")
    mrun.getMData().addData("Tin", "Tin = (Tin1 + Tin2)/2.")
    mrun.getMData().addData("HP", "HP = (HP1 + HP2)/2.")
    
    dkeys = mrun.getKeys()
    # print("dkeys=", dkeys)
    # sys.exit(1)
    
    if args.debug:
        ax = plt.gca()
        mrun.getMData().plotData(x='t', y='teb', ax=ax)
        mrun.getMData().plotData(x='t', y='Tout', ax=ax)
        plt.xlabel(r't [s]')
        plt.ylabel(r'[C]')
        plt.grid(b=True)
        plt.title(mrun.getInsert().replace("_","\_") + ":" + start_timestamp[0] + " " + start_timestamp[1] )
        if args.show:
            plt.show()
        if args.save:
            imagefile = args.input_file.replace(f_extension, "-teb.png")
            print("save to %s" % imagefile)
            plt.savefig(imagefile, dpi=300)
        plt.close()

    # extract data
    keys = ["t", "teb", "tsb", "debitbrut", "Tout", "Tin", "Flow", "BP", "HP"]
    units = ["s","C","C","m\u00B3/h","C","C","l/s","bar"]
    df = mrun.getMData().extractData(keys)

    if args.debug:
        pd.set_option("display.max_rows", None, "display.max_columns", None)
    
    # filter spikes 
    # see: https://ocefpaf.github.io/python4oceanographers/blog/2015/03/16/outlier_detection/
    kw = dict(marker='o', linestyle='none', color='r', alpha=0.3)
    for key in ["debitbrut", "Flow"]:
        mean = df[key].mean()
        # print("%s=" % key, type(df[key]), df[key])
        if mean != 0:
            var = df[key].var()
            # print("mean(%s)=%g" % (key,mean), "std=%g" % math.sqrt(var) )
            filtered = df[key].rolling(window=twindows, center=True).median().fillna(method='bfill').fillna(method='ffill')
            filteredkey = "filtered%s" % key
            # print("** ", filteredkey, type(filtered))
            
            df = df.assign(**{filteredkey: filtered.values})
            # print("*** ", filteredkey, df[filteredkey])
            
            difference = np.abs((df[key] - filtered)/mean*100) 
            outlier_idx = difference > threshold
        
            if args.debug:
                ax = plt.gca()
                df[key].plot()
                # filtered.plot()
                df[filteredkey].plot()
                df[key][outlier_idx].plot(**kw)
        
                ax.legend()
                plt.grid(b=True)
                plt.title(mrun.getInsert().replace("_","\_") + ": Filtered %s" % key)
                if args.show:
                    plt.show()
                if args.save:
                    imagefile = args.input_file.replace(f_extension, "-%s.png" % key)
                    print("save to %s" % imagefile)
                    plt.savefig(imagefile, dpi=300)
                plt.close()

            # replace key by filtered ones
            del df[key]
            df.rename(columns={filteredkey: key}, inplace=True)
    print("Filtered pikes done")
    
            
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

    subtype = '1/1' # counterflow

    def error_Tin(x, df_=df, debug=args.debug):
        # Now loop over h to find best h
        
        df_['cTin'] = df_.apply(lambda row: heatexchange(x[0], row.teb, row.Tout, row.debitbrut, row.Flow, 10, row.BP)[1], axis=1)
        df_['ctsb'] = df_.apply(lambda row: heatexchange(x[0], row.teb, row.Tout, row.debitbrut, row.Flow, 10, row.BP)[0], axis=1)

        diff =  np.abs(df_["Tin"] - df_['cTin'])
        L2_Tin = math.sqrt(np.dot( df_['Tin'], df_['Tin'] ))
        error_Tin = math.sqrt(np.dot( diff, diff )) /L2_Tin # diff.size

        diff =  np.abs(df_["tsb"] - df_['ctsb'])
        L2_tsb = math.sqrt(np.dot( df_['tsb'], df_['tsb'] ))
        error_tsb = math.sqrt(np.dot( diff, diff )) / L2_tsb #diff.size

        error_T = math.sqrt(error_Tin*error_Tin + error_tsb*error_tsb)
        
        if debug:
            print("error_Tin(%g)" % x, error_Tin, error_tsb, error_T)

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
    if args.debug:
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
    df['cTin'] = df.apply(lambda row: heatexchange(x[0], row.teb, row.Tout, row.debitbrut, row.Flow, 10, row.BP)[1], axis=1)
    df['ctsb'] = df.apply(lambda row: heatexchange(x[0], row.teb, row.Tout, row.debitbrut, row.Flow, 10, row.BP)[0], axis=1)
    ax = plt.gca()
    df.plot(x='t', y='ctsb', ax=ax) #, color='blue')
    df.plot(x='t', y='tsb', ax=ax) #, color='blue', linestyle='-')
    df.plot(x='t', y='cTin', ax=ax) #, color='red')
    df.plot(x='t', y='Tin', ax=ax) #, color='red', linestyle='--')
    plt.xlabel(r't [s]')
    plt.grid(b=True)
    plt.title(mrun.getInsert().replace("_","\_") + ": h=%g $W/m^2/K$" % x[0])

    if args.show:
        plt.show()
    if args.save:
        imagefile = args.input_file.replace(f_extension, "-h.png")
        print("save to %s" % imagefile)
        plt.savefig(imagefile, dpi=300)
        plt.close()

    df['QNTU'] = df.apply(lambda row: heatexchange(x[0], row.teb, row.Tout, row.debitbrut, row.Flow, 10, row.BP)[2], axis=1)
    df["Qhot"] = df.apply(lambda row: (row.Flow)*1.e-3*(rho(row.BP, row.Tout)*cp(row.BP, row.Tout)*row.Tout-rho(row.HP, row.Tin)*cp(row.HP, row.Tin)*row.Tin), axis=1)
    df["Qcold"] = df.apply(lambda row: row.debitbrut/3600.*(rho(10, row.tsb)*cp(10, row.tsb)*row.tsb-rho(10, row.teb)*cp(10, row.teb)*row.teb), axis=1)
        
    ax = plt.gca()
    df.plot(x='t', y='QNTU', ax=ax, color='red')
    df.plot(x='t', y='Qhot', ax=ax, color='blue')
    df.plot(x='t', y='Qcold', ax=ax, color='green')
    plt.xlabel(r't [s]')
    plt.ylabel(r'Q[W]')
    plt.grid(b=True)
    plt.title(mrun.getInsert().replace("_","\_"))
    if args.show:
        plt.show()
    if args.save:
        imagefile = args.input_file.replace(f_extension, "-Q.png")
        print("save to %s" % imagefile)
        plt.savefig(imagefile, dpi=300)
        plt.close()
