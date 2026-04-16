#!/usr/bin/env bash
# run_hybrid_plotting_examples.sh — Run the examples from README_hybrid_plotting.md
#
# Data directories resolved from env vars (same priority as data_dirs.py):
#   hybrid:     MAGNETRUN_HYBRID_DATA_DIR > HYBRID_DATADIR
#   pupitre:    MAGNETRUN_PUPITRE_DATA_DIR > PUPITRE_DATADIR > MAGNETRUN_DATA_DIR
#   pigbrother: MAGNETRUN_PIGBROTHER_DATA_DIR > PIGBROTHER_DATADIR > PIGBROTHER
#
# Interactive --show is replaced by --save to avoid blocking.
# Output images are written to a temporary directory and removed on exit.
#
# Usage:
#   cd /path/to/python_magnetrun
#   bash examples/run_hybrid_plotting_examples.sh

set -uo pipefail

export MPLBACKEND=Agg

PASS=0
FAIL=0
SKIP=0
ERRORS=()

# Change to repo root
cd "$(dirname "$0")/.." || exit 1

EXAMPLES="examples"

# ---------------------------------------------------------------------------
# Resolve data directories
# ---------------------------------------------------------------------------
_first_env() {
    local default="${@: -1}"
    for v in "${@:1:$#-1}"; do
        local val="${!v:-}"
        [[ -n "$val" ]] && echo "$val" && return
    done
    echo "$default"
}

HYBRID_DIR="$(_first_env MAGNETRUN_HYBRID_DATA_DIR HYBRID_DATADIR "")"
PUPITRE_DIR="$(_first_env MAGNETRUN_PUPITRE_DATA_DIR PUPITRE_DATADIR MAGNETRUN_DATA_DIR "")"
PIGBROTHER_DIR="$(_first_env MAGNETRUN_PIGBROTHER_DATA_DIR PIGBROTHER_DATADIR PIGBROTHER "")"

# ---------------------------------------------------------------------------
# Temporary output directory (cleaned up on exit)
# ---------------------------------------------------------------------------
OUTDIR="$(mktemp -d /tmp/magnetrun_hybrid_examples_XXXXXX)"
trap 'rm -rf "$OUTDIR"' EXIT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
run_cmd() {
    local id="$1"; shift
    printf "[%2s] " "$id"
    if eval "$@" > /dev/null 2>&1; then
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
    local id="$1" reason="$2"; shift 2
    printf "[%2s] " "$id"
    echo "SKIP ($reason): $*"
    (( SKIP++ )) || true
}

# ---------------------------------------------------------------------------
echo "========================================================"
echo "  plot_hybrid_with_pupitre_tdms — README examples"
echo "========================================================"
echo "  HYBRID_DIR     : ${HYBRID_DIR:-(not set)}"
echo "  PUPITRE_DIR    : ${PUPITRE_DIR:-(not set)}"
echo "  PIGBROTHER_DIR : ${PIGBROTHER_DIR:-(not set)}"
echo "  OUTPUT_DIR     : $OUTDIR"
echo "========================================================"
echo ""

SCRIPT="python $EXAMPLES/plot_hybrid_with_pupitre_tdms.py"
_NO_HYBRID="hybrid dir not set or not found — set MAGNETRUN_HYBRID_DATA_DIR"

# ---------------------------------------------------------------------------
echo "--- Argument-parser smoke test ---"

run_cmd 1 $SCRIPT --help

echo ""
echo "--- Example 1: basic usage (M8, FEPC-AUX-LNCMI, ALIM1_J1) ---"

if [[ -n "$HYBRID_DIR" && -d "$HYBRID_DIR" ]]; then
    run_cmd 2 $SCRIPT \
        -d 2025-11-02 \
        -s FEPC-AUX-LNCMI \
        -k ALIM1_J1 \
        --site M8 \
        --hybrid-dir "$HYBRID_DIR" \
        --save "$OUTDIR/example1_M8_ALIM1_J1.png"
else
    skip_cmd 2 "$_NO_HYBRID" \
        "$SCRIPT -d 2025-11-02 -s FEPC-AUX-LNCMI -k ALIM1_J1 --site M8 ..."
fi

echo ""
echo "--- Example 2: custom directories (M9, FEPC-LNCMI, I_H1) ---"

_skip2=""
if [[ -z "$HYBRID_DIR" || ! -d "$HYBRID_DIR" ]]; then
    _skip2="$_NO_HYBRID"
elif [[ -z "$PUPITRE_DIR" ]]; then
    _skip2="PUPITRE_DIR not set — set MAGNETRUN_PUPITRE_DATA_DIR"
elif [[ -z "$PIGBROTHER_DIR" ]]; then
    _skip2="PIGBROTHER_DIR not set — set MAGNETRUN_PIGBROTHER_DATA_DIR"
fi

if [[ -z "$_skip2" ]]; then
    run_cmd 3 $SCRIPT \
        -d 2025-01-27 \
        -s FEPC-LNCMI \
        -k I_H1 \
        --site M9 \
        --hybrid-dir "$HYBRID_DIR" \
        --pupitre-dir "$PUPITRE_DIR" \
        --pigbrother-dir "$PIGBROTHER_DIR" \
        --save "$OUTDIR/example2_M9_I_H1.png"
else
    skip_cmd 3 "$_skip2" \
        "$SCRIPT -d 2025-01-27 -s FEPC-LNCMI -k I_H1 --site M9 ..."
fi

echo ""
echo "--- Example 3: specific hours (M10, FEPC-AUX-LNCMI, hours 10,11,12) ---"

if [[ -n "$HYBRID_DIR" && -d "$HYBRID_DIR" ]]; then
    run_cmd 4 $SCRIPT \
        -d 2025-01-27 \
        -s FEPC-AUX-LNCMI \
        -k ALIM1_J1 \
        --site M10 \
        --hours 10,11,12 \
        --hybrid-dir "$HYBRID_DIR" \
        --save "$OUTDIR/example3_M10_hours.png"
else
    skip_cmd 4 "$_NO_HYBRID" \
        "$SCRIPT -d 2025-01-27 -s FEPC-AUX-LNCMI -k ALIM1_J1 --site M10 --hours 10,11,12 ..."
fi

echo ""
echo "--- Example 4: save to file (M10, FEPC-AUX-LNCMI, ALIM1_J1) ---"

if [[ -n "$HYBRID_DIR" && -d "$HYBRID_DIR" ]]; then
    run_cmd 5 $SCRIPT \
        -d 2025-01-27 \
        -s FEPC-AUX-LNCMI \
        -k ALIM1_J1 \
        --site M10 \
        --hybrid-dir "$HYBRID_DIR" \
        --save "$OUTDIR/example4_comparison_plot.png"
else
    skip_cmd 5 "$_NO_HYBRID" \
        "$SCRIPT -d 2025-01-27 -s FEPC-AUX-LNCMI -k ALIM1_J1 --site M10 --save ..."
fi

echo ""
echo "--- Example 5: save + show (M10, FEPC-AUX-LNCMI, ALIM1_J1) ---"

if [[ -n "$HYBRID_DIR" && -d "$HYBRID_DIR" ]]; then
    run_cmd 6 $SCRIPT \
        -d 2025-01-27 \
        -s FEPC-AUX-LNCMI \
        -k ALIM1_J1 \
        --site M10 \
        --hybrid-dir "$HYBRID_DIR" \
        --save "$OUTDIR/example5_comparison_plot.png"
else
    skip_cmd 6 "$_NO_HYBRID" \
        "$SCRIPT -d 2025-01-27 -s FEPC-AUX-LNCMI -k ALIM1_J1 --site M10 --save ... --show"
fi

# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
printf "  Results: %d passed, %d failed, %d skipped\n" \
    "$PASS" "$FAIL" "$SKIP"
if [[ ${#ERRORS[@]} -gt 0 ]]; then
    echo ""
    echo "  Failed commands:"
    for e in "${ERRORS[@]}"; do
        echo "    $e"
    done
fi
echo "========================================================"

[[ $FAIL -eq 0 ]]
