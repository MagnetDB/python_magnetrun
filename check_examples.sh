#!/usr/bin/env bash
# Verify examples from README.md / docs/examples_checklist.md
# Skips Python API section and commands that require external data files.

set -uo pipefail

# Use non-interactive matplotlib backend to avoid display windows
export MPLBACKEND=Agg

PASS=0
FAIL=0
SKIP=0
ERRORS=()

# Change to repo root
cd "$(dirname "$0")" || exit 1

TESTS="tests"
EXAMPLES="examples"
PKGDIR="python_magnetrun"

run_cmd() {
    local id="$1"
    shift
    printf "[%2s] " "$id"
    if "$@" > /dev/null 2>&1; then
        echo "PASS: $*"
        (( PASS++ )) || true
    else
        local code=$?
        echo "FAIL (exit $code): $*"
        ERRORS+=("[$id] $*")
        (( FAIL++ )) || true
    fi
}

skip_cmd() {
    local id="$1"
    local reason="$2"
    shift 2
    printf "[%2s] " "$id"
    echo "SKIP ($reason): $*"
    (( SKIP++ )) || true
}

echo "========================================"
echo " Checking examples (non-API)"
echo "========================================"
echo ""

# ---------------------------------------------------------------------------
echo "--- Basic Usage ---"

# 1: List available fields
run_cmd 1 magnetrun info \
    --housing M9 "2019.02.14 - 23:00:38.txt" --list

# 2: Select records by criteria (example script — not the main CLI)
run_cmd 2 python3 "$EXAMPLES/get-record.py" --housing M8 '2025.*.txt' \
    select --duration 60 --field 18.

# 3: Plot vs time (single field)
run_cmd 3 magnetrun plot \
    "data/M9_2019.02.14---23_00_38.txt" --vs_time "Field"

# 4: Plot key vs key with mixed file types (TDMS resolved via env)
run_cmd 4 magnetrun plot \
    --housing M10 '2025.01.27 - *.txt' M10_Overview_250127-1605.tdms \
    --key_vs_key timestamp-IH \
    --key_vs_key "Courants_Alimentations/timestamp-Courants_Alimentations/Référence_GR2"

# 5: Hybrid plot — requires external files not in repo
skip_cmd 5 "requires external hybrid data" \
    "magnetrun plot --housing M8 M8_Overview_250522-0802.tdms '2025.05.22 - 08:02:56.txt' --hybrid_date 2025-05-22"

echo ""
echo "--- Statistics and plateau detection ---"

# 6: Stats (all txt files in data/)
run_cmd 6 magnetrun stats 'data/*.txt'

# 7: Stats with plateau detection
run_cmd 7 magnetrun stats \
    --housing M8 '2025.*.txt' --plateau

# 8: Aggregate (example script — not the main CLI)
run_cmd 8 python3 "$EXAMPLES/get-record.py" --housing M9 '2025.*.txt' \
    aggregate --fields teb --show

echo ""
echo "--- Derived quantities ---"

# 9: Add formula with symbol/unit/label/description
run_cmd 9 magnetrun add \
    "data/M10_2020.10.23---20_10_41.txt" \
    --formula "PowerH = IH * UH / 1.e+6" \
    --symbol P_H --unit megawatt \
    --label "Insert Power" \
    --description "Insert electrical power in MW" \
    --plot --save "$PWD/power.png"

# 10: Add formula on a PigBrother TDMS file (resolved via env)
run_cmd 10 magnetrun add \
    --housing M10 M10_Overview_201003-0956.tdms \
    --formula "Tensions_Aimant/Power_internes = Tensions_Aimant/ALL_internes * Courants_Alimentations/Courant_GR2 / 1.e+6" \
    --symbol P --unit megawatt \
    --label "Bitter Power" \
    --description "Bitter electrical power in MW" \
    --plot

echo ""
echo "--- Field Definitions CLI (magnetrun config field) ---"

run_cmd 11 magnetrun config field "$PKGDIR/pupitre-defs.json" list

run_cmd 12 magnetrun config field "$PKGDIR/pupitre-defs.json" \
    add NewSensor I ampere --description "New coil current"

run_cmd 13 magnetrun config field "$PKGDIR/pupitre-defs.json" \
    delete NewSensor

run_cmd 14 magnetrun config field "$PKGDIR/pupitre-defs.json" \
    update Field --symbol Bz --description "Resistive Magnets Axial field"

run_cmd 15 magnetrun config field "$PKGDIR/pupitre-defs.json" \
    alias-add Idcct1 hybrid "FEPC-AUX-LNCMI/ALIM1_J1"

run_cmd 16 magnetrun config field "$PKGDIR/pupitre-defs.json" \
    alias-show Idcct1

run_cmd 17 magnetrun config field "$PKGDIR/pupitre-defs.json" crossref \
    --format "pupitre=$PKGDIR/pupitre-defs.json" \
    --format "pigbrother=$PKGDIR/pigbrother-defs.json" \
    --format "hybrid=$PKGDIR/hybrid-defs.json"

echo ""
echo "--- Housing Config CLI (magnetrun config housing) ---"

run_cmd 18 magnetrun config housing "$PKGDIR/M9-housing-config.json" show

run_cmd 19 magnetrun config housing M11-housing-config.json create M11 --from-builtin M9

run_cmd 20 magnetrun config housing M11-housing-config.json update \
    --gr1-current IB --gr2-current IH

echo ""
echo "--- Analysis CLI ---"

run_cmd 21 magnetrun analysis \
    data/M9_Default_200921-123303_Courants50Hz.tdms --show

# 22-23: require generic input.tdms
skip_cmd 22 "requires generic input.tdms" \
    "magnetrun analysis input.tdms --synchronize --lag ..."
skip_cmd 23 "requires generic input.tdms" \
    "magnetrun analysis input.tdms --json-log analysis.json ..."

echo ""
echo "--- Advanced Usage: Breakpoint detection and run signature ---"

# 24: magnetrun signature (promoted from test-signature.py)
run_cmd 24 magnetrun signature \
    --housing M9 "2025.01.27 - 15:39:29.txt" --threshold 1e-2

# 25: analysis with synchronize on PigBrother overview files (TDMS resolved via env)
run_cmd 25 magnetrun analysis \
    --housing M10 'M10_Overview_250211-*.tdms' \
    --synchronize --show

echo ""
echo "--- Advanced Usage: Anomaly detection ---"

# 26-28: require TDMS data files
skip_cmd 26 "requires TDMS data file" \
    "python3 $TESTS/test-anomalies.py <file>.tdms --group Courants_Alimentations ..."
skip_cmd 27 "requires TDMS data file" \
    "python3 $TESTS/test-anomalies.py data.tdms --methods dbscan mad ..."
skip_cmd 28 "requires TDMS data file + params.yaml" \
    "python3 $TESTS/test-anomalies.py data.tdms --config params.yaml ..."

echo ""
echo "--- Advanced Usage: Piecewise linear regression ---"

run_cmd 29 python3 "$EXAMPLES/corr_Ih_Ib.py" \
    --housing M9 "2024.11.06 - 16:43:44.txt" \
    --xkey IH --ykey IB --algo piecewise_regression --breakpoints 2

run_cmd 30 python3 "$EXAMPLES/corr_Ih_Ib.py" \
    --housing M9 "2024.11.06 - 16:43:44.txt" \
    --xkey t --ykey Field --algo pwlf --breakpoints 11

echo ""
echo "--- Advanced Usage: Field factor identification ---"

run_cmd 31 python3 "$TESTS/test-fieldfactor.py" \
    --housing M10 "data/M10_2020.10.23---20_10_41.txt"

# ---------------------------------------------------------------------------
# Clean up generated files
rm -f M11-housing-config.json M11-site-config.json analysis.json analysis.log power.png

echo ""
echo "========================================"
printf " Results: %d passed, %d failed, %d skipped\n" $PASS $FAIL $SKIP
echo "========================================"

if [[ ${#ERRORS[@]} -gt 0 ]]; then
    echo ""
    echo "Failed commands:"
    printf '  %s\n' "${ERRORS[@]}"
fi

[[ $FAIL -eq 0 ]]
