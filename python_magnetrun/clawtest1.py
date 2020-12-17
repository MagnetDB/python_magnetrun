
#!/usr/bin/env python
# encoding: utf-8

r"""
One-dimensional advection
=========================

Solve the linear advection equation:

.. math::
    q_t + u q_x = 0.

Here q is the density of some conserved quantity and u is the velocity.

The initial condition is a Gaussian and the boundary conditions are periodic.
The final solution is identical to the initial data because the wave has
crossed the domain exactly once.

L=2*h with h=1.577944e-01 (from HL-31.d par ex - lattest helix)
rho=st.steam_pT(pascal, kelvin).rho
cp = st.steam_pT(pascal, kelvin).cp

Flow1: l/s
Section= (from HL-31.d par ex)
Tin1
U1: sum(Icoil1,...,Icoil7)
Power1: U1*(Idcct1+Idcct2)

psi: Power1*(x+h)/(2*h)

x0 = -1.5*h
x1 = 2*h
x = pyclaw.Dimension(x0,x1,nx,name='x')
psi[0,:] = [ Power1*(xi+h)/(2*h) if abs(xi) <= h else 0 for xi in xc]

"Sh0":"262.292e-6",
"Sh1":"139.392e-6",
"Sh2":"176.149e-6",
"Sh3":"217.995e-6",
"Sh4":"264.365e-6",
"Sh5":"315.259e-6",
"Sh6":"373.504e-6",
"Sh7":"439.1e-6",
"Sh8":"511.483e-6",
"Sh9":"590.085e-6",
"Sh10":"674.908e-6",
"Sh11":"765.952e-6",
"Sh12":"863.215e-6",
"Sh13":"961.045e-6",
"Sh14":"292.364e-6"

"""

from __future__ import absolute_import
import sys
import os
import math
import numpy as np
from clawpack import riemann

import pandas as pd
import freesteam as st
import ht
import python_magnetrun
import heatexchanger_primary
import tabulate

df = None
duration = math.nan
npts_per_domain = math.nan
ntimes = math.nan

x0 = math.nan
x1 = math.nan
L = math.nan
Hx0 = math.nan
Hx1 = math.nan
Section = math.nan
SectionB = math.nan

tables = []
headers = ["t", "Tin", "Tout", "teb", "tsb"]


def mapping(Xorig):
    """create mesh per section"""

    print("mapping: x0=" , x0, "L=", L, "Hx0=", Hx0, "Hx1=", Hx1, "x1=" , x1, "npt=" , npts_per_domain) 
    ni = npts_per_domain -1
    physCoords = []
    
    for coord in Xorig:
        i = int(coord)
        remainder = int(divmod(i,ni)[1])
        if i < ni:
            x = x0 + i * abs(-L-x0)/float(ni)
            print("[1]:", i, "->", x)
        elif i >= ni and i < 2*ni:
            x = -L + remainder * 2*L/float(ni)
            print("[2]:", i, "->", x)
        elif i >= 2*ni and i < 3*ni:
            x = L + remainder * (Hx0-L)/float(ni)
            print("[3]:", i, "->", x)
        elif i >= 3*ni and i < 4*ni:
            x = Hx0 + remainder * (Hx1-Hx0)/float(ni)
            print("[4]:", i, "->", x)
        elif i >= 4*ni and i < 5*ni:
            x = Hx1 + remainder * (x1-Hx1)/float(ni)
            print("[5]:", i, "->", x)
        else:
            x = x1
            print(i, "->", x, "X")

        physCoords.append(x)
        # print("i=", i, "x=", x)
        
    return physCoords


def compute_u(Tin, Pin, Debit, Section):
    """compute mean velocity"""
    
    u = Debit * 1.e-3 /Section
    rho = get_rho( Pin, Tin)
    cp = get_cp( Pin, Tin)

    return u

# see http://www.clawpack.org/pyclaw/problem.html#adding-source-terms
def step_Euler_radial(solver, state, dt):
    """define step for classic"""

    xc = state.grid.p_centers
    # print("xc=", xc, "type(xc)=", type(xc))
    # #
    # sys.exit(1)
    
    q   = state.q
    # print("q=", q, "type(q)=", type(q))
    aux = state.aux

    i0 = math.floor(state.t)

    nx = state.problem_data['nx']
    L = state.problem_data['L']
    # Section = state.problem_data['Section']
    # SectionB = state.problem_data['SectionB']

    # Tin = (df['Tin1'][i0] + df['Tin2'][i0])/2. # q at x=x0
    # Pin = (df['HP1'][i0] + df['HP2'][i0])/2.
    # Debith = df['Flow1'][i0] + df['Flow2'][i0]
    # u =compute_u(Tin, Pin, Debith, Section)

    # # Helices
    # Power1 = abs(df['U1'][i0] * df['I1'][i0])
    # Q = Power1 / ( Section * (2*L) ) / (get_rho(Tin,Pin)*get_cp(Tin,Pin))

    # # Bitter
    # Power2 = abs(df['U2'][i0] * df['I2'][i0])
    # Q += Power2 / ( SectionB * (2*L) ) / (get_rho(Tin,Pin)*get_cp(Tin,Pin))

    # Tci = df['teb'][i0]
    # Thi = q[0,3*nx]
    # Debitc = df['debitbrut'][i0]
    # Debith = df['Flow1'][i0] + df['Flow2'][i0]
    # Pci = 10
    # Phi = 10
    # Volc = 1986.04 * 1e-3 #Volh = 1986.04 * 1e+3
    # Qth = heatexchange(4041, Tci, Thi, Debitc, Debith, Pci, Phi)[2] / Volc / (get_rho(Tin,Pin)*get_cp(Tin,Pin))

    u = state.problem_data['u']
    Q = state.problem_data['Q']
    Qth = state.problem_data['Q']
    Power1 = state.problem_data['Power1']
    Power2 = state.problem_data['Power2']

    tables.Append([state.t, q[0,nx], q[0,2*nx], q[0,3*nx], q[0,4*nx]])

    aux = state.aux
    aux[0,:] = u
    
    # Add Magnet Q
    psi = np.empty(q.shape)
    psi[0,:] = [ Q if abs(xi) <= L else 0 for xi in xc]

    # Add HeatExchanger Q
    HX0 = state.problem_data['HX0']
    HX1 = state.problem_data['HX1']
    psi_hx = np.empty(q.shape)
    psi_hx[0,:] = 0 # [ -Qth if xi <= HX1 and xi >= HX0 else 0 for xi in xc]

    q[0,:] = q[0,:] + dt * (psi[0,:] + psi_hx[0,:])

def dq_Euler_radial(solver, state, dt):
    """define step for sharpclaw"""

    xc = state.grid.p_centers
    q   = state.q
    aux = state.aux

    i0 = math.floor(state.t)

    nx = state.problem_data['nx']
    L = state.problem_data['L']
    Section = state.problem_data['Section']
    SectionB = state.problem_data['SectionB']

    Tin = q[0,nx] # (df['Tin1'][i0] + df['Tin2'][i0])/2. # q at x=x0
    Pin = (df['HP1'][i0] + df['HP2'][i0])/2.
    Debith = df['Flow1'][i0] + df['Flow2'][i0]
    u = compute_u(Tin, Pin, Debith, Section)

    # Helices
    Power1 = abs(df['PH'][i0])
    Q = Power1 / ( Section * (2*L) ) / (rho*cp)

    # Bitter
    Power2 = abs(df['PB'][i0])
    # Q += Power2 / ( SectionB * (2*L) ) / (rho*cp)

    Tci = df['teb'][i0]
    Thi = q[0,3*nx]
    Debitc = df['debitbrut'][i0]
    Debith = df['Flow1'][i0] + df['Flow2'][i0]
    Pci = 10
    Phi = 10
    Volc = 1986.04 * 1e-3 # Volh = 1986.04 * 1e+3
    Qth = heatexchange(4041, Tci, Thi, Debitc, Debith, Pci, Phi)[2] / Volc / (rho*cp)

    tables.Append([state.t, q[0,nx], q[0,2*nx], q[0,3*nx], q[0,4*nx]])

    aux = state.aux
    aux[0,:] = u

    # Add Magnet
    psi = np.empty(q.shape)
    psi[0,:] = [ Q if abs(xi) <= L else 0 for xi in xc]

    # Add HeatExchanger Q
    psi_hx = np.empty(q.shape)
    psi_hx[0,:] = 0 # [ Qth if xi <= HX1 and xi >= HX0 else 0 for xi in xc]

    dq = np.empty(q.shape)
    dq[0,:] = + dt * (psi[0,:] + psi_hx[0,:])

    return dq

def auxinit(state):
    """
    Define advectionfield
    """
    # Initilize petsc Structures for aux
    xc=state.grid.loop.centers
    u = state.problem_data['u']

    aux = state.aux
    aux[0,:] = u

# def setup(num_output_times=10, tfinal=3, nx=10,
#           kernel_language='Python',
#           use_petsc=False, solver_type='classic',
#           weno_order=5,
#           time_integrator='SSP104',
#           outdir='./_output',
#           claw_pkg='amrclaw'):
def setup(kernel_language='Python',
          use_petsc=False,
          solver_type='classic',
          weno_order=5,
          time_integrator='SSP104',
          outdir='./_output',
          claw_pkg='amrclaw'):
    """
    Setp Clawpack simu
    """

    global npt
    
    nx = npts_per_domain
    num_output_times = ntimes
    tfinal = duration

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if kernel_language == 'Fortran':
        riemann_solver = riemann.advection_1D
    elif kernel_language == 'Python':
        riemann_solver = riemann.advection_1D_py.advection_1D

    if solver_type=='classic':
        solver = pyclaw.ClawSolver1D(riemann_solver)
        solver.step_source=step_Euler_radial
    elif solver_type=='sharpclaw':
        solver = pyclaw.SharpClawSolver1D(riemann_solver)
        solver.dq_src=dq_Euler_radial
        solver.weno_order = weno_order
        solver.time_integrator = time_integrator
        if time_integrator == 'SSPLMMk3':
            solver.lmm_steps = 5
            solver.check_lmm_cond = True
    else: raise Exception('Unrecognized value of solver_type.')


    Tin = df['Tin'][0]
    Pin = df['HP'][0]
    rho = heatexchanger_primary.rho(Pin, Tin)
    cp = heatexchanger_primary.cp(Pin, Tin)
    print ("rho=", rho, "cp=", cp, "rhocp=", rho*cp)

    Flow = 140.e-3

    # Bitter
    Power2 = 12.5e+6
    Qth2 = Power2 / (SectionB*(2*L))
    uB = Flow/SectionB

    print ("Power2=", Power2,
           "delta T = Pe/(rho Cp Debit)=", Power2/(rho*cp*Flow),
           "Qth = Power/(Section*(2*L) )=", Qth2,
           "Qth/(rhocp)=", Qth2/(rho*cp),
           "Section=", SectionB, "u=Flow/Section=", uB)

    # Helices
    Power1 = 12.5e+6
    Qth1 = Power1 / (Section*(2*L))
    u = Flow/Section
    print ("Power1=", Power1,
           "delta T = Pe/(rho Cp Debit)=", Power1/(rho*cp*Flow),
           "Qth = Power/(Section*(2*L))=", Qth1,
           "Qth/(rhocp)=", Qth1/(rho*cp),
           "Section=", Section, "u=Flow/Section=", u)

    Qth = Qth1+Qth2
    
    # Add path to primary heat exchanget
    # x = pyclaw.Dimension(-1,200,5*nx,name='loop')
    x = pyclaw.Dimension(0,float(5*nx+1),5*nx,name='loop')
    domain = pyclaw.Domain(x)

    solver.kernel_language = kernel_language
    verbosity = 1
    total_steps = 30

    solver.bc_lower[0] = pyclaw.BC.extrap # periodic if HX considered otherwise extrap
    solver.bc_upper[0] = pyclaw.BC.extrap # periodic if HX considered otherwise extrap

    # Define BC for aux
    num_aux = 1
    solver.aux_bc_lower[0] = pyclaw.BC.extrap #u
    solver.aux_bc_upper[0] = pyclaw.BC.extrap #u

    state = pyclaw.State(domain,solver.num_eqn, num_aux)
    state.problem_data['nx'] = nx  # Number of interval per section
    state.problem_data['rhocp'] = rho*cp  # Specific Heat
    state.problem_data['u'] = u  # Advection velocity
    state.problem_data['L'] = L  # Electric Length of Magnet
    state.problem_data['Power1'] = Power1  # Magnet Power
    state.problem_data['Power2'] = Power2  # Magnet Power
    state.problem_data['Q'] = Qth/(rho*cp)   # Magnet equivalent
    state.problem_data['Section'] = Section   # Colling Section
    state.problem_data['SectionB'] = SectionB   # Colling Section

    state.problem_data['HX0'] = Hx0   # Entry of Heat Exchanger
    state.problem_data['HX1'] = Hx1   # Output of Heat Exchanger
    state.problem_data['HXQth'] = -Qth/(rho*cp)    # Qth of Heat Exchanger (from NTU model)

    # Create mapping
    state.grid.mapc2p = mapping
    # print("grid: ", state.grid, "type(state.grid)=", type(state.grid))
    # print("grid nodes: ", state.grid.c_nodes)
    # print("grid p_nodes: ", state.grid.p_nodes)

    state.q[0,:] = df['Tin'][0] # Tin

    auxinit(state)
    # print("t Power1 Power2 u Q Qth Tin Tout, Tho")
    
    claw = pyclaw.Controller()
    claw.keep_copy = True
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.dt_variable=True
    claw.cfl_desired = 0.9
    claw.verbosity = 4

    if outdir is not None:
        claw.outdir = outdir
    else:
        claw.output_format = None

    claw.tfinal = tfinal
    claw.setplot = setplot
    claw.num_output_times = num_output_times

    return claw

def add_source(current_data):
    """
    Plot Qth using VisClaw.
    """
    from pylab import plot
    x = current_data.x
    t = current_data.t
    print("xlower =", current_data.xlower)
    print("xupper =", current_data.xupper)
    print("t =", current_data.t)
    print("current_data =", type(current_data))
    
    # Geometry
    print("add_source: npt=" , npts_per_domain, "L=", L, "Hx0=" , x0, "Hx1=" , x1, "type(current_data)=", type(current_data))

    qsource = [ 20 if abs(xi) <= L else 15 for xi in x]
    qcooler = [ 10 if (xi-Hx0) >=0 and (xi-Hx1) <= 0 else 15 for xi in x]
    
    qth = [ (a+b)-30 for a,b in zip(qsource,qcooler)]
    plot(x, qth, 'r', label="source")

def setplot(plotdata):
    """
    Plot solution using VisClaw.
    """
    plotdata.clearfigures()  # clear any old figures,axes,items data

    plotfigure = plotdata.new_plotfigure(name='T', figno=1)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    # plotaxes.ylimits = [10,45]
    plotaxes.title = 'Temperature Profile'
    
    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = 0
    plotitem.plotstyle = '-o'
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':2,'markersize':5}

    # plotaxes.afteraxes = add_source

    return plotdata


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="input txt file (ex. M9_2019.02.14-23_00_38.txt)")
    parser.add_argument("--npts_per_domain", help="number of points per domain", type=int, default=10)
    parser.add_argument("--ntimes", help="number of time steps", type=int, default=20)
    parser.add_argument("--duration", help="specify simu duration", type=float, default=30)
    parser.add_argument("--iplot", help="activate plot", action='store_true')
    parser.add_argument("--debug", help="activate debug mode", action='store_true')
    parser.add_argument("--nhelices", help="specify number of helices", type=int, default=14)
    parser.add_argument("--site", help="specify a site (ex. M8, M9,...)", default="M9")
    args = parser.parse_args()

    npts_per_domain = args.npts_per_domain
    ntimes = args.ntimes
    duration = args.duration
    
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
    insert = mrun.getInsert()
    
    # TODO # Load geom cfg from json
    # load helices insert data from .json file
    # then get d file to get Sections

    # Load data for Bitter from site config
    # import json
    # if args.config:
    #     geomdata = json.load(args.site + "-cfg.json")
    
    # load ini file (geometry)
    L = 1.578e-01

    Spipe = 2*math.pi*pow(130.e-3,2)
    Pipe = 70
    Hx0 = Pipe+L
    Hx1 = Hx0 + 4.9

    x0 = -1.5*L
    x1 = Hx1+Pipe+L

    print("x0=%g" % x0, "-L=%g" % -L, "L=%g" % L, "Hx0=%g" % Hx0, "Hx1=%g" % Hx1, "x1=%g" % x1)

    Section =  262.292e-6 \
	+ 139.392e-6 \
        + 176.149e-6 \
	+ 217.995e-6 \
	+ 264.365e-6 \
        + 315.259e-6 \
        + 373.504e-6 \
        + 439.1e-6 \
        + 511.483e-6 \
        + 590.085e-6 \
        + 674.908e-6 \
        + 765.952e-6 \
        + 863.215e-6 \
        + 961.045e-6 \
        + 292.364e-6

    # Bitter M9/M10
    SectionB = (3606.68+7010.63)*1.e-6
    
    
    mrun.getMData().addTime()
    start_timestamp = mrun.getMData().getStartDate()

    mrun.getMData().addData("Flow", "Flow = Flow1 + Flow2")
    mrun.getMData().addData("Tin", "Tin = (Tin1 + Tin2)/2.")
    mrun.getMData().addData("HP", "HP = (HP1 + HP2)/2.")

    # TODO: move to magnetdata
    max_tap=0
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
        if not ukey in keys:
            # Add an empty column
            # print ("Ukey=%s" % ukey, (ukey in keys) )
            mrun.getMData().addData(ukey, "%s = 0" % ukey)
            missing_probes.append(i)

    if len(missing_probles):
        print("Missing U probes:", missing_probes)

    formula = "UH = "
    for i in range(args.nhelices+1):
        ukey = "Ucoil%d" % i
        if ukey in keys:
            formula += ukey
    print("UH", formula)
    mrun.getMData().addData("UH", formula)
    
    formula = "UB = Ucoil15 + Ucoil16"
    print("UB", formula)
    mrun.getMData().addData("UB", formula)

    mrun.getMData().addData("PH", "PH = UH * IH")
    mrun.getMData().addData("PB", "PB = UB * IB")
    
    dkeys = mrun.getKeys()

    # extract data
    # keys = ["t", "teb", "tsb", "debitbrut", "Tout", "Tin", "Flow", "BP", "HP"]
    # units = ["s","C","C","m\u00B3/h","C","C","l/s","bar"]
    df = mrun.getMData().extractData() # keys

    # Create
    # output = run_app_from_main(setup,setplot)
    claw = setup()
    claw.run()

    print( "\n", tabulate.tabulate(tables, headers, tablefmt="simple"), "\n")
    # TODO plot(key vs tables values) for key in Tin, Tout, tsb, ctsb
    # Turn tables in a panda dataframe
    
    if args.iplot:
        # using matplotlib:
        # read data from files in outdir
        # https://matplotlib.org/3.3.3/api/animation_api.html
        # remember to purge outdir after plot/anim
        # or better rename outdir
        from clawpack.pyclaw import plot
        plot.interactive_plot()
