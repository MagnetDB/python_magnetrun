"""Raw Hybrid ALIM currents and M8 voltages during the M8 and M10 sessions of 2026-10-14 (EXPERIENCES_LOG)."""
import logging
import sys

import numpy as np
import pandas as pd

from python_magnetrun.data_dirs import HYBRID_DATA_DIR
from python_magnetrun.hybrid.hybrid_run import HybridRun

logging.disable(logging.WARNING)
date = sys.argv[1]
hours = [int(h) for h in sys.argv[2].split(",")]
hd = HybridRun.fromdir(base_dir=HYBRID_DATA_DIR, date_str=date, fepc_system="FEPC-AUX-LNCMI", housing="M8").HybridData
anchor = pd.Timestamp(date) + pd.Timedelta(hours=hours[0])
cols = {}
for v in ("ALIM1_J1", "ALIM1_J2", "ALIM2_J1", "ALIM2_J2", "BITTER_V1", "BITTER_V2", "PH_V14"):
    d, t = hd.read_khz_variable("FEPC-AUX-LNCMI", v, hours=hours, apply_calib=True)
    t = np.asarray(t, dtype=float)
    cols[v] = pd.Series(d, index=anchor + pd.to_timedelta(t - t[0], unit="s")).sort_index().resample("60s").mean()
df = pd.DataFrame(cols)
df["ALIM1"] = df.ALIM1_J1 + df.ALIM1_J2
df["ALIM2"] = df.ALIM2_J1 + df.ALIM2_J2
df["R_B [mOhm]"] = 1000 * df.BITTER_V2 / df.ALIM1
df["R_PH14 [mOhm]"] = 1000 * df.PH_V14 / df.ALIM2
out = df[["ALIM1", "BITTER_V1", "BITTER_V2", "R_B [mOhm]", "ALIM2", "PH_V14", "R_PH14 [mOhm]"]]
pd.set_option("display.width", 200)
print(out.iloc[::10].round(3).to_string())
