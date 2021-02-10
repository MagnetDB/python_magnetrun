import argparse
import water
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

functions = {
    'rho': water.getRho,
    'cp': water.getCp,
    'k': water.getK,
    'mu': water.getMu
    }
descriptions = {
    'rho': ("Density", "kg/m\u00B3"),
    'cp': ("Specific Heat", "kJ/kg/K"),
    'k': ("Thermal Conductivity", "W/m/K"),
    'mu': ("Dynamic Viscosity", "Pa.s")
}

parser = argparse.ArgumentParser("Water Properties")
parser.add_argument(
    "--property", help="select a property (default: rho m3/h)", type=str, choices=['rho', 'cp', 'k', 'mu'], default='rho')
parser.add_argument(
    "--Trange", help="set Temperature range in Celsius divided into n points (eg 'T0;T1,n')", type=str, default='10;50;50')
parser.add_argument(
    "--Prange", help="set Pressure range in Bar divided into n points (eg 'P0;P1;n')", type=str, default='0.1;40;50')

args = parser.parse_args()

wprop = args.property
(T0, T1, nT) = args.Trange.split(';')
(P0, P1, nP) = args.Prange.split(';')


def myfunc(T, P):
    return functions[wprop](P, T)

Trange = np.linspace(float(T0), float(T1), int(nT))
Prange = np.linspace(float(P0), float(P1), int(nP))
(x, y) = np.meshgrid(Trange, Prange)

myfunc_vec = np.vectorize(myfunc)
p = lambda x, y: myfunc_vec(x, y) 

fig = plt.figure()
ax = plt.gca()

CS = plt.contour(x, y, p(x, y), 20)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Water %s' % descriptions[wprop][0])
ax.set(xlabel='T [C]', ylabel='P [Bar]',  title='%s [%s]' % (wprop, descriptions[wprop][1]) )

plt.show()
