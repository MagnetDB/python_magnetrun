# README Examples Checklist

Track which README examples have been verified to work correctly.

---

## Basic Usage

### List available fields

| # | Command | Status |
|---|---------|--------|
| 1 | `python3 -m python_magnetrun.cli srvdata/M9_2019.02.14---23:00:38.txt info --list` | [x] |

### Select records by criteria

| # | Command | Status |
|---|---------|--------|
| 2 | `python3 examples/get-record.py srvdata/M8*.txt select --duration 60 --field 18.` | [x] |

### Plotting

| # | Command | Status |
|---|---------|--------|
| 3 | `python3 -m python_magnetrun.cli srvdata/M9_2019.02.14---23:00:38.txt plot --vs_time "Field"` | [x] |
| 4 | `python3 -m python_magnetrun.cli srvdata/M10_2025.01.27---*.txt pigbrotherdata/…/M10_Overview_250127-1605.tdms plot --key_vs_key timestamp-IH --key_vs_key timestamp-Courants_Alimentations/Référence_GR1` | [ ] |
| 5 | `python3 -m python_magnetrun.cli ~/M9_Overview_240509-1634.tdms ~/M9_2024.05.09---16_34_03.txt --hybrid_datadir /path/to/hybrid --hybrid_date 2024-05-09 --fepc_system FEPC-LNCMI plot --vs_time Courants_Alimentations/Courant_GR1 --vs_time IH --vs_time_hybrid "kHz/FEPC-LNCMI/I_H1"` | [ ] |

### Statistics and plateau detection

| # | Command | Status |
|---|---------|--------|
| 6 | `python3 -m python_magnetrun.cli srvdata/M8*.txt stats` | [ ] |
| 7 | `python3 -m python_magnetrun.cli srvdata/M8*.txt stats --plateau` | [ ] |
| 8 | `python3 examples/get-record.py srvdata/M*---*.txt aggregate --fields teb --show` | [ ] |

### Derived quantities

| # | Command | Status |
|---|---------|--------|
| 9  | `python3 -m python_magnetrun.cli srvdata/M10_2020.10.03---09:56:20.txt add --formula "PowerH = IH * UH / 1.e+6" --plot` | [x] |
| 10 | `python3 -m python_magnetrun.cli pigbrotherdata/…/M10_Overview_201003-0956.tdms add --formula "Tensions_Aimant/Power_internes = Tensions_Aimant/ALL_internes * Courants_Alimentations/Courant_GR2 / 1.e+6" --plot` | [ ] |

---

## Field Definitions CLI (`magnetrun-field-defs`)

| # | Command | Status |
|---|---------|--------|
| 11 | `magnetrun-field-defs pupitre-defs.json list` | [ ] |
| 12 | `magnetrun-field-defs pupitre-defs.json add NewSensor I ampere --description "New coil current"` | [ ] |
| 13 | `magnetrun-field-defs pupitre-defs.json update Field --symbol Bz --description "Axial field"` | [ ] |
| 14 | `magnetrun-field-defs pupitre-defs.json alias-add Idcct1 hybrid "FEPC-AUX-LNCMI/ALIM1_J1"` | [ ] |
| 15 | `magnetrun-field-defs pupitre-defs.json alias-show Idcct1` | [ ] |
| 16 | `magnetrun-field-defs pupitre-defs.json crossref --format pupitre=pupitre-defs.json --format pigbrother=pigbrother-defs.json --format hybrid=hybrid-defs.json` | [ ] |

---

## Site Config CLI (`magnetrun-site-config`)

| # | Command | Status |
|---|---------|--------|
| 17 | `magnetrun-site-config M9-site-config.json show` | [X] |
| 18 | `magnetrun-site-config M11-site-config.json create M11 --from-builtin M9` | [ ] |
| 19 | `magnetrun-site-config M11-site-config.json update --gr1-current IB --gr2-current IH` | [ ] |

---

## Analysis CLI

| # | Command | Status |
|---|---------|--------|
| 20 | `python3 -m python_magnetrun.analysis.cli M9_Overview_*.tdms --show` | [x] |
| 21 | `python3 -m python_magnetrun.analysis.cli input.tdms --synchronize --lag --distance --downsample 10 --show --save --debug --log-file analysis.log` | [ ] |
| 22 | `python3 -m python_magnetrun.analysis.cli input.tdms --json-log analysis.json --quiet` | [ ] |

---

## Advanced Usage

### Breakpoint detection and run signature

| # | Command | Status |
|---|---------|--------|
| 23 | `python3 tests/test-signature.py srvdata/M10_2025.01.27---15:39:29.txt --window=10 --threshold 1.e-2` | [ ] |
| 24 | `python3 -m python_magnetrun.analysis pigbrotherdata/…/M10_Overview_250211-*.tdms --key Référence_GR1 --show --synchronize` | [ ] |

### Anomaly detection

| # | Command | Status |
|---|---------|--------|
| 25 | `python3 tests/test-anomalies.py <file>.tdms --group Courants_Alimentations --methods dbscan mad --method-params dbscan.eps=0.3 mad.threshold=4.0` | [ ] |
| 26 | `python3 tests/test-anomalies.py data.tdms --methods dbscan mad --method-params-json '{"dbscan": {"eps": 0.3}, "mad": {"threshold": 4.0}}'` | [ ] |
| 27 | `python3 tests/test-anomalies.py data.tdms --config params.yaml --method-params dbscan.eps=0.1` | [ ] |

### Piecewise linear regression

| # | Command | Status |
|---|---------|--------|
| 28 | `python3 examples/corr_Ih_Ib.py srvdata/M9_2024.11.06---16:43:44.txt --xkey IH --ykey IB --algo piecewise_regression --breakpoints 2` | [ ] |
| 29 | `python3 examples/corr_Ih_Ib.py srvdata/M9_2024.11.06---16:43:44.txt --xkey t --ykey Field --algo pwlf --breakpoints 11` | [ ] |

### Field factor identification

| # | Command | Status |
|---|---------|--------|
| 30 | `python3 tests/test-fieldfactor.py ~/M9_2024.05.13---16_30_51.txt` | [ ] |

---

## Python API

### Data loading

| # | Snippet | Status |
|---|---------|--------|
| 31 | `MagnetData.fromtxt(...)` — `.Keys`, `.Data` (DataFrame with `t`, `Field`, `IH`, `IB`) | [ ] |
| 32 | `MagnetData.fromtdms(...)` — `.Keys`, `.Data["Courants_Alimentations"]` | [ ] |
| 33 | `FileDiscovery` + `load_data(file_set)` — `data["overview"]`, `data["pupitre"]` | [ ] |
| 34 | `read_rms_file(...)` — columns, DataFrame head | [ ] |
| 35 | `parse_cfg_file(...)` + `read_hour_file(...)` — `.get_analog_slots()`, array shape | [ ] |
| 36 | `HybridData(base_dir, date_str, fepc_system)` — `.Keys`, `.load_rms_data()`, `.load_khz_config()`, `.get_khz_variables()` | [ ] |

### Configuration

| # | Snippet | Status |
|---|---------|--------|
| 37 | `AnalysisConfig.for_housing("M9")` — `.housing.reference_gr1_current`, `.thresholds.get(...)` | [ ] |
| 38 | `get_housing_config("M9")` — plain, with `json_file=`, with `overrides=` | [ ] |
| 39 | `update_housing_config("M9-housing-config.json", {...})` | [ ] |

### Field definitions API

| # | Snippet | Status |
|---|---------|--------|
| 40 | `get_aliases("pupitre-defs.json", "Idcct1")` | [ ] |
| 41 | `build_crossref({"pupitre": ..., "pigbrother": ..., "hybrid": ...})` | [ ] |

### Synchronization and metrics

| # | Snippet | Status |
|---|---------|--------|
| 42 | `compute_lag(series1, series2)` — `.lag_seconds`, `.is_reliable` | [ ] |
| 43 | `synchronize_data(df_overview, df_pupitre, key=...)` | [ ] |
| 44 | `calc_euclidean`, `calc_mae`, `calc_mape`, `calc_correlation` — `.value` | [ ] |
| 45 | `compute_dtw_distance(series1, series2)` — `.similarity_score` | [ ] |
| 46 | `compute_tlcc(series1, series2, max_lag=50)` | [ ] |
