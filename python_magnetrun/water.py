r"""
Provides methods to compute water properties
and Dimensionless numbers for cooling flow

see report from E3 students on gDrive
https://docs.google.com/document/d/1B1nQD_1XNmJza03_Z_OsNWEXZztIjap9FPjAimKZWpA/edit
"""

import freesteam as st

def getRho(pbar, celsius):
    """
    compute water volumic mass as a function
    of pressure and temperature
    """

    pascal = pbar * 1e+5
    kelvin = celsius+273.
    rho = st.steam_pT(pascal, kelvin).rho
    # print("rho(%g,%g)=%g" % (pbar,celsius,rho))
    return rho

def getCp(pbar, celsius):
    """
    compute water volumic specific heat as a function
    of pressure and temperature
    """

    pascal = pbar * 1e+5
    kelvin = celsius+273.
    cp = st.steam_pT(pascal, kelvin).cp
    # print("cp(%g,%g)=%g" % (pbar,celsius,cp))
    return cp

def getNusselt():
    """
    compute Nusselt
    """

    Nusselt = None
    return Nusselt

def getReynolds():
    """
    compute Reynolds
    """

    Reynolds= None
    return Reynolds

def getEnthalpy():
    """
    computes Enthalpy
    """

    Enthalpy = None
    return Enthalpy

def getOTHC():
    """
    computes heat exchange coefficient
    """

    Ohtc = None
    return Ohtc