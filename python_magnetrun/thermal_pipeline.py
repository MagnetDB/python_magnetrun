"""
Thermal pipeline: compute derived thermal quantities for water-cooled magnet circuits
from prepared MagnetRun DataFrames using python_magnetcooling.

This module is the thermal analogue of waterflow_pipeline.py.
It operates on DataFrames already prepared by prepareData_legacy() and adds
per-circuit and global thermal columns using an iterative calorimetric scheme.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CoolingCircuitColumns:
    """Column name mapping for a single cooling circuit in a prepared DataFrame.

    After prepareData_legacy() for M9:
      - Helix: voltage="UH", current="IH", flow="FlowH",
               temp_in="TinH", pressure_in="HPH", back_pressure="BP"
      - Bitter: voltage="UB", current="IB", flow="FlowB",
                temp_in="TinB", pressure_in="HPB", back_pressure="BP"

    For M8/M10 the H/B assignment is swapped; pass the correct mapping.
    """

    voltage: str       # sum of Ucoil voltages for this circuit [V], e.g. "UH"
    current: str       # circuit current [A], e.g. "IH"
    flow: str          # volumetric flow rate [l/s], e.g. "FlowH"
    temp_in: str       # inlet water temperature [°C], e.g. "TinH"
    pressure_in: str   # inlet pressure [bar], e.g. "HPH"
    back_pressure: str # back (outlet) pressure [bar], e.g. "BP"

    @property
    def suffix(self) -> str:
        """Suffix derived from flow column name, e.g. 'H' from 'FlowH'."""
        return self.flow.replace("Flow", "")


def compute_circuit_thermal(
    df: pd.DataFrame,
    circuit: CoolingCircuitColumns,
    n_iter: int = 5,
) -> pd.DataFrame:
    """Compute thermal quantities for one cooling circuit and add them to df.

    Added columns (suffix = circuit.suffix, e.g. "H" or "B"):
      - ``Pe{s}``   : electrical power [MW]
      - ``DP{s}``   : pressure drop HP - BP [bar]
      - ``DT{s}``   : calorimetric temperature rise [°C] (iterative)
      - ``rho{s}``  : water density at mean conditions [kg/m³]
      - ``cp{s}``   : specific heat at mean conditions [J/kg/K]
      - ``Tout{s}`` : outlet temperature Tin + DT [°C]
      - ``Tw{s}``   : mean water temperature Tin + DT/2 [°C]
      - ``P{s}``    : mean circuit pressure (HP + BP)/2 [bar]

    The iterative scheme:
      1. Initialise water properties at inlet (BP, Tin).
      2. Compute DT = Pe·1e6 / (rho·cp·Flow·1e-3).
      3. Update water properties at mean conditions (BP + DP/2, Tin + DT/2).
      4. Repeat *n_iter* times.

    :param df: DataFrame prepared with prepareData_legacy()
    :param circuit: Column name mapping for this circuit
    :param n_iter: Number of iterations for property update, defaults to 5
    :return: df with added thermal columns (in-place modification + return)
    :raises ImportError: If python_magnetcooling is not installed
    :raises KeyError: If any required column is missing from df
    """
    try:
        from python_magnetcooling.water_properties import get_rho, get_cp
    except ImportError as exc:
        raise ImportError(
            "python_magnetcooling is required for thermal computations."
        ) from exc

    required = [
        circuit.voltage, circuit.current, circuit.flow,
        circuit.temp_in, circuit.pressure_in, circuit.back_pressure,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"compute_circuit_thermal: missing columns {missing}")

    s = circuit.suffix
    Pe_col   = f"Pe{s}"
    DP_col   = f"DP{s}"
    DT_col   = f"DT{s}"
    rho_col  = f"rho{s}"
    cp_col   = f"cp{s}"
    Tout_col = f"Tout{s}"
    Tw_col   = f"Tw{s}"
    P_col    = f"P{s}"

    bp  = circuit.back_pressure
    tin = circuit.temp_in
    hp  = circuit.pressure_in
    fl  = circuit.flow

    # Electrical power [MW]
    df[Pe_col] = df[circuit.voltage] * df[circuit.current] / 1.0e6

    # Pressure drop [bar]
    df[DP_col] = df[hp] - df[bp]

    # Initial water properties at inlet conditions (BP, Tin [°C])
    df[rho_col] = df.apply(lambda row: get_rho(row[bp], row[tin]), axis=1)
    df[cp_col]  = df.apply(lambda row: get_cp(row[bp], row[tin]), axis=1)

    # Iterative DT refinement with updated mean-condition properties
    for _ in range(n_iter):
        df[DT_col] = df.apply(
            lambda row: (
                row[Pe_col] * 1.0e6
                / (row[rho_col] * row[cp_col] * row[fl] * 1.0e-3)
                if row[fl] != 0
                else row[tin]
            ),
            axis=1,
        )
        df[rho_col] = df.apply(
            lambda row: get_rho(
                row[bp] + row[DP_col] / 2.0,
                row[tin] + row[DT_col] / 2.0,
            ),
            axis=1,
        )
        df[cp_col] = df.apply(
            lambda row: get_cp(
                row[bp] + row[DP_col] / 2.0,
                row[tin] + row[DT_col] / 2.0,
            ),
            axis=1,
        )

    df[Tout_col] = df[tin] + df[DT_col]
    df[Tw_col]   = df[tin] + df[DT_col] / 2.0
    df[P_col]    = (df[hp] + df[bp]) / 2.0

    logger.debug(
        "compute_circuit_thermal: circuit=%s, added columns %s",
        s,
        [Pe_col, DP_col, DT_col, rho_col, cp_col, Tout_col, Tw_col, P_col],
    )
    return df


def compute_global_thermal(
    df: pd.DataFrame,
    circuit1: CoolingCircuitColumns,
    circuit2: CoolingCircuitColumns,
) -> pd.DataFrame:
    """Compute global thermal quantities combining two cooling circuits.

    Requires that compute_circuit_thermal() has already been called for both
    circuits so that their per-circuit columns (Pe, rho, cp, Tout, Flow) exist.

    Added columns:
      - ``Power``  : total electrical power Pe1 + Pe2 [MW]
      - ``Toutg``  : energy-weighted mixed outlet temperature [°C]

    The mixed outlet temperature formula is:
      Toutg = Σ(ρᵢ·cpᵢ·Flowᵢ·Toutᵢ) / Σ(ρᵢ·cpᵢ·Flowᵢ)

    :param df: DataFrame with per-circuit thermal columns already computed
    :param circuit1: First circuit mapping (typically helices)
    :param circuit2: Second circuit mapping (typically bitter)
    :return: df with added global columns (in-place modification + return)
    """
    s1 = circuit1.suffix
    s2 = circuit2.suffix

    df["Power"] = df[f"Pe{s1}"] + df[f"Pe{s2}"]

    w1 = df[f"rho{s1}"] * df[f"cp{s1}"] * df[circuit1.flow]
    w2 = df[f"rho{s2}"] * df[f"cp{s2}"] * df[circuit2.flow]
    df["Toutg"] = (w1 * df[f"Tout{s1}"] + w2 * df[f"Tout{s2}"]) / (w1 + w2)

    logger.debug("compute_global_thermal: added Power, Toutg")
    return df
