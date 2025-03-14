#!/usr/bin/env python
# encoding: utf-8

r"""
Cooling loop model (One-dimensional advection)
=========================

Solve the linear advection equation:

.. math::
    q_t + u q_x = 0.

Here q is the water temperature and u is the velocity.

The initial condition is given from an actual experiment.
The boundary conditions are computed at the entry and exit of the
heat exchanger.

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

import tabulate
from . import heatexchanger_primary
from . import water as w

df = None
duration = math.nan
ntimes = math.nan

x0 = math.nan
x1 = math.nan
L = math.nan
Section = math.nan
SectionB = math.nan
Q = math.nan

tables = []
headers = [
    "t[s]",
    "Tin1[C]",
    "Tin2[C]",
    "Tout[C]",
    "Thi[C]",
    "Tho[C]",
    "Power1[MW]",
    "Power2[MW]",
]


def interpolate(t: float, a: float, b: float, fa: float, fb: float) -> float:
    """interpolate between a and b"""

    return (t - a) * (fa - fb) / (b - a) + fa


def compute_u(Q: float, S: float) -> float:
    """compute mean velocity"""

    return Q * 1.0e-3 / S


def Joules(x: float, Q: float, L: float) -> float:
    """compute Joules Losses"""
    val = 0
    if abs(x) <= L:
        val = Q
    return val


# see http://www.clawpack.org/pyclaw/problem.html#adding-source-terms
def step_Euler_radial(solver, state, dt):
    """define step for classic"""

    xc = state.grid.p_centers
    # print("xc=", xc, type(xc))

    q = state.q
    aux = state.aux
    # print("u[%g]: " % state.t, aux[0,:])

    i0 = math.floor(state.t)
    # print("i0=", i0)
    i1 = i0 + 1

    nx = state.problem_data["nx"]
    L = state.problem_data["L"]
    Section = state.problem_data["Section"]
    SectionB = state.problem_data["SectionB"]
    SectionP = state.problem_data["SectionP"]
    df = state.problem_data["df"]

    Tin = interpolate(state.t, i0, i1, df["Tin"][i0], df["Tin"][i1])
    Pin = interpolate(state.t, i0, i1, df["HP"][i0], df["HP"][i1])
    Debit_hot = interpolate(state.t, i0, i1, df["Flow"][i0], df["Flow"][i1])
    u = compute_u(Debit_hot, SectionP)

    # Helices
    Tin1 = interpolate(state.t, i0, i1, df["TinH"][i0], df["TinH"][i1])
    Pin1 = interpolate(state.t, i0, i1, df["HPH"][i0], df["HPH"][i1])
    rho = w.getRho(Pin1, Tin1)
    cp = w.getCp(Pin1, Tin1)
    Power1 = abs(interpolate(state.t, i0, i1, df["PH"][i0], df["PH"][i1]))
    QH = Power1 / (Section * (2 * L)) / (rho * cp)
    DebitH = interpolate(state.t, i0, i1, df["FlowH"][i0], df["FlowH"][i1])
    uH = compute_u(DebitH, Section)

    # Bitter
    Tin2 = interpolate(state.t, i0, i1, df["TinB"][i0], df["TinB"][i1])
    Pin2 = interpolate(state.t, i0, i1, df["HPB"][i0], df["HPB"][i1])
    rho = w.getRho(Pin2, Tin2)
    cp = w.getCp(Pin2, Tin2)
    Power2 = abs(interpolate(state.t, i0, i1, df["PB"][i0], df["PB"][i1]))
    QB = Power2 / (SectionB * (2 * L)) / (rho * cp)
    DebitB = interpolate(state.t, i0, i1, df["FlowB"][i0], df["FlowB"][i1])
    uB = compute_u(DebitB, SectionB)

    Q = QH + QB

    # Tout = q[0,2213]
    Tout = q[0, 2223]  # depends on npt

    # Pipe
    DebitP = interpolate(state.t, i0, i1, df["Flow"][i0], df["Flow"][i1])
    uP = compute_u(DebitP, SectionP)

    # update velocity
    state.aux[0, :] = uP
    state.aux[0, :] = [uH if abs(xi) <= L else 0 for xi in xc]

    # Add Magnet Q
    psi = np.empty(q.shape)
    for i, x in enumerate(xc[0]):
        psi[0, i] = Joules(x, Q, L)
    # print("psi:", psi[0,:])

    # Add HeatExchanger Q: give input and outpout BC
    Tci = interpolate(state.t, i0, i1, df["teb"][i0], df["teb"][i1])
    Thi = q[0, -1]
    Debitc = interpolate(state.t, i0, i1, df["debitbrut"][i0], df["debitbrut"][i1])
    Debith = interpolate(state.t, i0, i1, df["Flow"][i0], df["Flow"][i1])
    Pci = 10
    Phi = 10

    Q = heatexchanger_primary.heatexchange(4041, Tci, Thi, Debitc, Debith, Pci, Phi)
    Tco = Q[0]
    Tho = Q[1]
    if state.t == i0:
        tables.append([state.t, Tin1, Tin2, Tout, Thi, Tho, Power1, Power2])
    # QHx = res[2] / Volc / (rho*cp)

    # # custom Bcs
    # solver.bc_lower[0] = Tho
    # solver.bc_upper[0] = Thi

    # Howto get index for x=L ??
    # tables.append([state.t, q[0,0], q[0,x=L], q[0,-1], Tco])

    q[0, :] = q[0, :] + dt * psi[0, :]


def dq_Euler_radial(solver, state, dt):
    """define step for sharpclaw"""

    xc = state.grid.p_centers
    q = state.q
    aux = state.aux

    i0 = math.floor(state.t)
    # interpolate df values from state.t betwenn i0 and i0+1

    nx = state.problem_data["nx"]
    L = state.problem_data["L"]
    Section = state.problem_data["Section"]
    SectionB = state.problem_data["SectionB"]
    SectionP = state.problem_data["SectionP"]
    df = state.problem_data["df"]

    Tin = df["Tin"][i0]
    Pin = df["HP"][i0]
    Debith = df["Flow"][i0]
    u = compute_u(Debith, SectionP)

    # Helices
    rho = w.getRho(df["HP"][i0], df["Tin"][i0])
    cp = w.getCp(df["HP"][i0], df["Tin"][i0])
    Power1 = abs(df["PH"][i0])
    Q = Power1 / (Section * (2 * L)) / (rho * cp)
    DebitH = df["FlowH"][i0]
    u = compute_u(DebitH, Section)

    # Bitter
    Power2 = abs(df["PB"][i0])
    QB = Power2 / (SectionB * (2 * L)) / (rho * cp)
    DebitB = df["Flow2"][i0]
    u = compute_u(DebitB, SectionB)

    Tci = df["teb"][i0]
    Thi = q[0, -1]
    Debitc = df["debitbrut"][i0]
    Debith = df["Flow"][i0]
    Pci = 10
    Phi = 10
    Hx = heatexchanger_primary.heatexchange(4041, Tci, Thi, Debitc, Debith, Pci, Phi)
    Tco = Hx[0]
    Tho = Hx[1]
    # Volc = 1986.04 * 1e-3 # Volh = 1986.04 * 1e+3
    # QHx = Hx[2] / Volc / (rho*cp)

    # tables.Append([state.t, q[0,0], q[0,nx], q[0,2*nx], q[0,3*nx], q[0,4*nx]])

    aux = state.aux
    aux[0, :] = u

    # Add Magnet
    psi = np.empty(q.shape)
    psi[0, :] = [Q if abs(xi) <= L else 0 for xi in xc]

    dq = np.empty(q.shape)
    dq[0, :] = +dt * psi[0, :]

    return dq


def auxinit(state):
    """
    Define advectionfield
    """

    u = state.problem_data["u"]
    df = state.problem_data["df"]

    # xc is a tuple when only 1 domain
    # a np array otherwise
    xc = state.grid.p_centers
    print("xc:", type(xc), len(xc))

    L = state.problem_data["L"]
    print(f"L: {L}", type(L))

    Section = state.problem_data["Section"]
    SectionB = state.problem_data["SectionB"]
    SectionP = state.problem_data["SectionP"]

    # Helices
    DebitH = df["FlowH"][0]
    uH = compute_u(DebitH, Section)
    print(f"uH: {uH}", type(uH))

    # Bitters
    DebitB = df["FlowB"][0]
    uB = compute_u(DebitB, SectionB)
    print(f"uB: {uB}")

    # Pipe
    DebitP = df["Flow"][0]
    uP = compute_u(DebitP, SectionP)
    print(f"uP: {uP}")

    state.aux[0, :] = uP
    print("state_aux: ", type(state.aux), state.aux.shape)
    for i, xi in enumerate(xc):
        if abs(xi) <= L:
            state.aux[i] = uH
    # state.aux[0, :] = [uH if abs(xi) <= L else uP for xi in xc]
    print(f"u[{state.t}]: {state.aux[0, :]}")


def setup(
    df,
    nx,
    num_output_times,
    tfinal,
    kernel_language="Python",
    use_petsc=False,
    solver_type="classic",
    weno_order=5,
    time_integrator="SSP104",
    outdir="./_output",
    claw_pkg="amrclaw",
):
    """
    Setup Clawpack simu
    """

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if kernel_language == "Fortran":
        riemann_solver = riemann.advection_1D
    elif kernel_language == "Python":
        riemann_solver = riemann.advection_1D_py.advection_1D

    if solver_type == "classic":
        solver = pyclaw.ClawSolver1D(riemann_solver)
        solver.step_source = step_Euler_radial
    elif solver_type == "sharpclaw":
        solver = pyclaw.SharpClawSolver1D(riemann_solver)
        solver.dq_src = dq_Euler_radial
        solver.weno_order = weno_order
        solver.time_integrator = time_integrator
        if time_integrator == "SSPLMMk3":
            solver.lmm_steps = 5
            solver.check_lmm_cond = True
    else:
        raise Exception("Unrecognized value of solver_type.")

    Tin = df["Tin"][0]
    Pin = df["HP"][0]
    rho = w.getRho(Pin, Tin)
    cp = w.getCp(Pin, Tin)
    # print ("rho=", rho, "cp=", cp, "rhocp=", rho*cp)

    Flow = 140.0e-3

    # Bitter
    Power2 = 12.5e6
    Qth2 = Power2 / (SectionB * (2 * L))
    uB = Flow / SectionB

    print(
        "Power2=",
        Power2,
        "delta T = Pe/(rho Cp Debit)=",
        Power2 / (rho * cp * Flow),
        "Qth = Power/(Section*(2*L) )=",
        Qth2,
        "Qth/(rhocp)=",
        Qth2 / (rho * cp),
        "Section=",
        SectionB,
        "u=Flow/Section=",
        uB,
    )

    # Helices
    Power1 = 12.5e6
    Qth1 = Power1 / (Section * (2 * L))
    u = Flow / Section
    print(
        "Power1=",
        Power1,
        "delta T = Pe/(rho Cp Debit)=",
        Power1 / (rho * cp * Flow),
        "Qth = Power/(Section*(2*L))=",
        Qth1,
        "Qth/(rhocp)=",
        Qth1 / (rho * cp),
        "Section=",
        Section,
        "u=Flow/Section=",
        u,
    )

    Qth = Qth1 + Qth2

    # Domain (grid.p_centers is a tuple)
    x = pyclaw.Dimension(x0, x1, nx, name="loop")
    #
    # For multiple domains:
    # x = pyclaw.Dimension(0,float(5*nx+1),5*nx,name='loop')
    # use mapping see ~/Clawpack/test1.py
    # grid.p_centers is a numpy array
    domain = pyclaw.Domain(x)
    xc = domain.grid.p_centers

    # Defines Gauges
    ids = []
    print("Finding gauge index")
    for i, x in enumerate(xc[0]):
        if abs(abs(x) / L - 1) < 1.0e-2:
            ids.append(i)
            print("x[%d] = %g" % (i, x))

    # Howto get index for x=L ??

    solver.kernel_language = kernel_language
    verbosity = 1
    total_steps = 30

    # ????Customn BC???
    solver.bc_lower[0] = pyclaw.BC.extrap  # periodic if HX considered otherwise extrap
    solver.bc_upper[0] = pyclaw.BC.extrap  # periodic if HX considered otherwise extrap

    # Define BC for aux
    num_aux = 1

    # ????Customn BC??? ???Custom u???
    solver.aux_bc_lower[0] = pyclaw.BC.extrap  # u
    solver.aux_bc_upper[0] = pyclaw.BC.extrap  # u

    state = pyclaw.State(domain, solver.num_eqn, num_aux)
    state.problem_data["nx"] = nx  # Number of interval per section
    state.problem_data["u"] = u  # Advection velocity
    state.problem_data["L"] = L  # Electric Length of Magnet
    state.problem_data["Section"] = Section  # Section Helices
    state.problem_data["SectionB"] = SectionB  # Section Bitters

    # TODO load from cfg.json
    SectionP = 2 * math.pi * pow(130.0e-3, 2)
    state.problem_data["SectionP"] = SectionP  # Section Pipes

    state.problem_data["x0"] = x0  # Output of Heat Exchanger
    state.problem_data["x1"] = x1  # Entry of Heat Exchanger

    state.problem_data["df"] = df  # experimental data - needed to be smoothed here

    # # Create mapping
    # state.grid.mapc2p = mapping

    # print("initial condition")
    state.q[0, :] = df["Tin"][0]  # Tin
    auxinit(state)

    claw = pyclaw.Controller()
    claw.keep_copy = True
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver
    claw.dt_variable = True
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
    print(
        "add_source: L",
        L,
        "x0=",
        x0,
        "x1=",
        x1,
        "type(current_data)=",
        type(current_data),
    )

    qth = [20 if abs(xi) <= L else 15 for xi in x]

    plot(x, qth, "r", label="source")


def setplot(plotdata):
    """
    Plot solution using VisClaw.
    """
    plotdata.clearfigures()  # clear any old figures,axes,items data

    plotfigure = plotdata.new_plotfigure(name="T", figno=1)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    # howto force yrange in plot??
    # plotaxes.ylimits = [5,45]
    plotaxes.title = "Temperature Profile"

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type="1d_plot")
    plotitem.plot_var = 0
    plotitem.plotstyle = "-o"
    plotitem.color = "b"
    plotitem.kwargs = {"linewidth": 2, "markersize": 5}

    # how to add additionnal curves ??
    # check ~/Clawpack
    # plotaxes.afteraxes = add_source

    return plotdata


if __name__ == "__main__":
    from clawpack.pyclaw.util import run_app_from_main

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", help="input txt file (ex. M9_2019.02.14-23_00_38.txt)"
    )
    parser.add_argument(
        "--npts_per_domain", help="number of points per domain", type=int, default=10
    )
    parser.add_argument("--ntimes", help="number of time steps", type=int, default=20)
    parser.add_argument(
        "--duration", help="specify simu duration", type=float, default=30
    )
    parser.add_argument("--iplot", help="activate plot", action="store_true")
    parser.add_argument("--debug", help="activate debug mode", action="store_true")
    parser.add_argument(
        "--nhelices", help="specify number of helices", type=int, default=14
    )
    parser.add_argument("--site", help="specify a site (ex. M8, M9,...)", default="M9")
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
    args = parser.parse_args()

    tau = 400
    if args.pre == "smoothed":
        print("smoothed options")
        tau = float(args.pre_params)

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

    from ..MagnetRun import MagnetRun

    housing = args.site
    mrun = MagnetRun.fromtxt(housing, args.site, args.input_file)
    insert = "tutut"  # mrun.getInsert()

    # TODO # Load geom cfg from json
    # load helices insert data from .json file
    # then get d file to get Sections

    # Load data for Helices / Bitter from site config
    # import json
    # if args.config:
    #     geomdata = json.load(args.insert + "-cfg.json")

    # load ini file (geometry)
    L = 1.578e-01
    print("recommended nx=", 70 / (2 * L / 10.0) * 2 + 1)

    Spipe = 2 * math.pi * pow(130.0e-3, 2)
    Pipe = 70

    x0 = -70  # output from Hx
    x1 = 70  # input from Hx

    print(f"x0={x0}, -L={-L}, Pipe={Pipe}, x1={x1}")

    Section = (
        262.292e-6
        + 139.392e-6
        + 176.149e-6
        + 217.995e-6
        + 264.365e-6
        + 315.259e-6
        + 373.504e-6
        + 439.1e-6
        + 511.483e-6
        + 590.085e-6
        + 674.908e-6
        + 765.952e-6
        + 863.215e-6
        + 961.045e-6
        + 292.364e-6
    )

    # Bitter M9/M10
    SectionB = (3606.68 + 7010.63) * 1.0e-6

    start_timestamp = mrun.getMData().getStartDate()

    # Add Power Data
    mrun.getMData().addData("PH", "PH = UH * IH")
    mrun.getMData().addData("PB", "PB = UB * IB")

    # use smoothed data instead of raw data
    smoothedkeys = [
        "TinH",
        "HPH",
        "FlowH",
        "TinB",
        "HPB",
        "FlowB",
        "tsb",
        "teb",
        "debitbrut",
        "PH",
        "UH",
        "IH",
        "PB",
        "UB",
        "IB",
    ]
    from ..processing import smoothers as smoothtools

    for key in smoothedkeys:
        mrun = smoothtools.smooth(
            mrun,
            key,
            inplace=True,
            tau=tau,
            debug=args.debug,
            show=False,
            input_file=args.input_file,
        )

    dkeys = mrun.getKeys()

    mrun.getMData().addData("Flow", "Flow = FlowH + FlowH")
    mrun.getMData().addData("Tin", "Tin = (TinH + TinH)/2.")
    mrun.getMData().addData("HP", "HP = (HPH + HPH)/2.")

    # extract data
    keys = [
        "t",
        "teb",
        "tsb",
        "debitbrut",
        "Tout",
        "Tin",
        "TinH",
        "TinB",
        "FlowH",
        "FlowB",
        "Flow",
        "BP",
        "HP",
        "HPH",
        "HPB",
        "PH",
        "PB",
    ]
    units = [
        "s",
        "C",
        "C",
        "m\u00b3/h",
        "C",
        "C",
        "C",
        "C",
        "l/s",
        "l/s",
        "l/s",
        "bar",
        "bar",
        "bar",
    ]
    df = mrun.getMData().extractData(keys)

    # coolingparams = [0.207979, 0.640259, 0.397994]  # from nominal values
    coolingparams = [0.1249, 0.65453, 0.40152]  # from student param fits

    # Create
    # output = run_app_from_main(setup,setplot)
    claw = setup(df, args.npts_per_domain, args.ntimes, args.duration)
    claw.run()

    print("\n", tabulate.tabulate(tables, headers, tablefmt="simple"), "\n")
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
