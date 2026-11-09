"""Estimate the clock lag between Hybrid (CEA FEPC) and pupitre data of an M8 run.

The Hybrid supply currents (kHz data, averaged over 1 s) are compared with
the pupitre circuit currents of the same run while scanning a clock lag:

* pupitre ``IB`` vs ``FEPC-AUX-LNCMI`` ``ALIM1_J1 + ALIM1_J2`` (GR1),
* pupitre ``IH`` vs ``FEPC-AUX-LNCMI`` ``ALIM2_J1 + ALIM2_J2`` (GR2),
* pupitre ``SupraField`` vs ``FEPC-LNCMI`` ``I_BOB`` (the fitted ratio gives
  the supra field factor in use).

Convention: ``pupitre time + lag = hybrid time``.  A figure shows IB / IH
against their Hybrid equivalents without and with the lag correction.

Usage::

    python hybrid_pupitre_lag.py "<M8 pupitre .txt>" 2026-10-30 --hours 7,8,9,10 --save
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from python_magnetrun.cli_args import create_managed_plots_parser
from python_magnetrun.data_dirs import HYBRID_DATA_DIR
from python_magnetrun.hybrid.hybrid_run import HybridRun, LoadOptions
from python_magnetrun.log_utils import get_logger, setup_logging
from python_magnetrun.MagnetRun import load_mrun
from python_magnetrun.utils.narwhals_compat import to_pandas

logger = get_logger(__name__)

SYSTEMS = ("FEPC-AUX-LNCMI", "FEPC-LNCMI")
PUPITRE_COLOR = "#2a78d6"
HYBRID_COLOR = "#eb6834"


def load_pupitre(path: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Load the IB, IH and SupraField series of an M8 pupitre file.

    Parameters
    ----------
    path : str
        Path to the M8 pupitre ``.txt`` file.

    Returns
    -------
    tuple of pandas.Series
        ``(IB, IH, SupraField)`` series [A, A, T] indexed by naive-UTC
        :class:`~pandas.DatetimeIndex`.
    """
    mdata = load_mrun(path, housing="M8", auto_resolve=False).getMData()
    series = []
    for key in ("IB", "IH", "SupraField"):
        df = to_pandas(mdata.getData(["timestamp", key])).dropna().sort_values("timestamp")
        series.append(pd.Series(df[key].to_numpy(), index=pd.DatetimeIndex(df.timestamp), name=key))
    return tuple(series)


def hybrid_1s(
    base_dir: str,
    date_str: str,
    system: str,
    variables: list[str],
    hours: list[int],
    sessions_log: str | None = None,
) -> pd.Series:
    """Sum Hybrid kHz variables of one system, averaged over 1 s.

    Parameters
    ----------
    base_dir : str
        Hybrid data base directory (``.../CEA``).
    date_str : str
        Day of the run, ``YYYY-MM-DD``.
    system : str
        FEPC system, ``"FEPC-AUX-LNCMI"`` or ``"FEPC-LNCMI"``.
    variables : list of str
        kHz variable names to sum (e.g. ``["ALIM1_J1", "ALIM1_J2"]``).
    hours : list of int
        UTC hours to load; elapsed times count from ``hours[0]:00:00``.
    sessions_log : str, optional
        EXPERIENCES_LOG CSV: resistive currents are kept only inside M8
        sessions (Otsu voltage mask otherwise, see
        :class:`~python_magnetrun.hybrid.hybrid_run.LoadOptions`).

    Returns
    -------
    pandas.Series
        Summed signal [A] on a 1 s naive-UTC :class:`~pandas.DatetimeIndex`.
    """
    hrun = HybridRun.fromdir(base_dir=base_dir, date_str=date_str, fepc_system=system, housing="M8")
    anchor = pd.Timestamp(date_str) + pd.Timedelta(hours=hours[0])
    total = None
    for var in variables:
        data, elapsed = hrun.getData(
            f"kHz/{system}/{var}", hours=hours, options=LoadOptions(sessions_log=sessions_log)
        )
        s = pd.Series(data, index=anchor + pd.to_timedelta(elapsed, unit="s")).sort_index()
        s = s.resample("1s").mean()
        total = s if total is None else total.add(s, fill_value=np.nan)
    return total


def scan_lag(
    pu: pd.Series, hy: pd.Series, min_level: float, lags: range
) -> tuple[int, float, float, int] | None:
    """Find the clock lag minimising the rms residual ``hy - ratio * pu``.

    Parameters
    ----------
    pu : pandas.Series
        Pupitre signal, naive-UTC index.
    hy : pandas.Series
        Hybrid signal, naive-UTC index (1 s grid).
    min_level : float
        Only samples with ``|pu| > min_level`` (or ``|hy| > min_level``)
        enter the residual, to ignore the idle parts of the run.
    lags : range
        Candidate lags [s]; convention ``pupitre time + lag = hybrid time``.

    Returns
    -------
    tuple or None
        ``(lag [s], rms residual, fitted ratio hy/pu, matched samples)``,
        or ``None`` if no lag gives at least 100 matched samples.
    """
    best = None
    for lag in lags:
        shifted = pd.Series(pu.to_numpy(), index=pu.index + pd.Timedelta(seconds=lag))
        j = pd.concat([shifted.rename("pu"), hy.rename("hy")], axis=1, join="inner").dropna()
        j = j[(j.pu.abs() > min_level) | (j.hy.abs() > min_level)]
        if len(j) < 100:
            continue
        ratio = float(np.dot(j.hy, j.pu) / np.dot(j.pu, j.pu))
        rms = float(np.sqrt(np.mean((j.hy - ratio * j.pu) ** 2)))
        if best is None or rms < best[1]:
            best = (lag, rms, ratio, len(j))
    return best


def estimate_lag(pu: pd.Series, hy: pd.Series, min_level: float, max_lag: int):
    """Estimate the lag with a 10 s coarse scan refined at 1 s over +/- 15 s.

    Parameters
    ----------
    pu, hy : pandas.Series
        Pupitre and Hybrid signals, see :func:`scan_lag`.
    min_level : float
        Activity threshold, see :func:`scan_lag`.
    max_lag : int
        Half width of the coarse scan [s].

    Returns
    -------
    tuple or None
        Same as :func:`scan_lag`.
    """
    coarse = scan_lag(pu, hy, min_level, range(-max_lag, max_lag + 1, 10))
    if coarse is None:
        return None
    return scan_lag(pu, hy, min_level, range(coarse[0] - 15, coarse[0] + 16))


def plot_alignment(pairs: list[tuple], date_str: str, title: str | None):
    """Plot pupitre currents against their Hybrid equivalents, without and with lag.

    Parameters
    ----------
    pairs : list of tuple
        ``(label, pupitre series, hybrid series, hybrid label, rms at lag 0,
        (lag, rms, ratio, n))`` per row.
    date_str : str
        Day of the run, used in the default title.
    title : str, optional
        Figure title override.

    Returns
    -------
    matplotlib.figure.Figure
        A ``len(pairs) x 2`` grid: left without lag, right pupitre shifted by lag.
    """
    fig, axes = plt.subplots(len(pairs), 2, figsize=(14, 4.5 * len(pairs)), sharex=True, squeeze=False)
    for row, (label, pu, hy, hy_label, rms0, (lag, rms, _, _)) in enumerate(pairs):
        for col, shift in enumerate((0, lag)):
            ax = axes[row, col]
            ax.plot(hy.index, hy.to_numpy(), color=HYBRID_COLOR, lw=1.2, label=f"Hybrid {hy_label}")
            ax.plot(pu.index + pd.Timedelta(seconds=shift), pu.to_numpy(), color=PUPITRE_COLOR,
                    lw=1.2, label=f"pupitre {label}")
            what = "no lag" if col == 0 else f"pupitre shifted by {lag:+d} s"
            ax.set_title(f"{label} vs {hy_label}: {what} (rms {rms0 if col == 0 else rms:.0f} A)")
            ax.set_ylabel(f"{label} [A]")
            ax.grid(color="#e4e3df", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(loc="center", frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("time [UTC]")
    fig.suptitle(title or f"M8 {date_str}: pupitre vs Hybrid currents (pupitre time + lag = Hybrid time)")
    fig.tight_layout()
    return fig


def main() -> None:
    """Parse arguments, estimate the lags, print them and plot IB / IH alignment."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0], parents=[create_managed_plots_parser()])
    parser.add_argument("pupitre_file", help="M8 pupitre .txt file")
    parser.add_argument("date", help="day of the run, YYYY-MM-DD (Hybrid data directory)")
    parser.add_argument("--hours", help="UTC hours to load, comma separated (default: pupitre time span)")
    parser.add_argument("--list", action="store_true", help="list the Hybrid kHz variables and exit")
    parser.add_argument("--max-lag", type=int, default=7200, help="half width of the coarse lag scan [s]")
    parser.add_argument("--hybrid_datadir", default=HYBRID_DATA_DIR or None,
                        help="base directory for Hybrid data (default: HYBRID_DATA_DIR)")
    parser.add_argument("--sessions-log", default=None,
                        help="EXPERIENCES_LOG CSV: keep Hybrid resistive currents only inside M8 sessions "
                             "(default: Otsu voltage mask)")
    parser.add_argument("--log-level", default="ERROR",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()
    setup_logging(level=args.log_level)

    IB, IH, SF = load_pupitre(args.pupitre_file)
    start, end = IB.index[0], IB.index[-1]
    print(f"pupitre {len(IB)} rows, {start} -> {end} (UTC), IB max {IB.max():.0f} A, IH max {IH.max():.0f} A")

    if args.list:
        for system in SYSTEMS:
            hrun = HybridRun.fromdir(base_dir=args.hybrid_datadir, date_str=args.date,
                                     fepc_system=system, housing="M8")
            print(f"{system}: kHz variables {hrun.HybridData.get_khz_variables(system)}")
        return

    hours = [int(h) for h in args.hours.split(",")] if args.hours else list(range(start.hour, end.hour + 1))
    alim1 = hybrid_1s(args.hybrid_datadir, args.date, "FEPC-AUX-LNCMI", ["ALIM1_J1", "ALIM1_J2"], hours,
                      args.sessions_log)
    alim2 = hybrid_1s(args.hybrid_datadir, args.date, "FEPC-AUX-LNCMI", ["ALIM2_J1", "ALIM2_J2"], hours,
                      args.sessions_log)
    ibob = hybrid_1s(args.hybrid_datadir, args.date, "FEPC-LNCMI", ["I_BOB"], hours)
    print(f"hybrid hours {hours} (UTC): ALIM1 max {alim1.max():.0f} A, ALIM2 max {alim2.max():.0f} A, "
          f"I_BOB max {ibob.max():.1f} A")

    pairs = []
    for label, pu, hy, hy_label in (("IB", IB, alim1, "ALIM1 (GR1)"), ("IH", IH, alim2, "ALIM2 (GR2)")):
        best = estimate_lag(pu, hy, 100.0, args.max_lag)
        if best is None:
            print(f"{label} vs {hy_label}: no overlap with Hybrid data")
            continue
        rms0 = scan_lag(pu, hy, 100.0, range(0, 1))
        lag, rms, ratio, n = best
        print(f"pupitre {label} vs Hybrid {hy_label}: lag {lag:+d} s, rms {rms:.1f} A, ratio {ratio:.4f}, n {n}")
        pairs.append((label, pu, hy, hy_label, rms0[1] if rms0 else float("nan"), best))

    best = estimate_lag(SF, ibob, 0.05, args.max_lag)
    if best is None:
        print("SupraField vs I_BOB: no overlap with Hybrid data")
    else:
        lag, rms, ratio, n = best
        print(f"pupitre SupraField vs Hybrid I_BOB: lag {lag:+d} s, rms {rms:.2f} A, "
              f"ffS in use = 1/ratio = {1e4 / ratio:.3f} G/A, n {n}")

    if not pairs:
        return
    fig = plot_alignment(pairs, args.date, args.title)
    if args.save is not None:
        output = args.save or f"hybrid_pupitre_lag_{args.date}.png"
        fig.savefig(output, dpi=150)
        print(f"saved {output}")
    if args.show or args.save is None:
        plt.show()


if __name__ == "__main__":
    main()
