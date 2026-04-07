#!/usr/bin/env bash
# run_hybrid_plotting_examples.sh — Run the examples from README_hybrid_plotting.md
#
# All examples use plot_hybrid_with_pupitre_tdms.py.
# Interactive --show is replaced by --save to avoid blocking.
# Output images are written to a temporary directory and removed on exit.
#
# Data directories are resolved from env vars (same priority chain as
# python_magnetrun.data_dirs):
#   hybrid:     MAGNETRUN_HYBRID_DATA_DIR > HYBRID_DATADIR
#   pupitre:    MAGNETRUN_PUPITRE_DATA_DIR > PUPITRE_DATADIR > MAGNETRUN_DATA_DIR
#   pigbrother: MAGNETRUN_PIGBROTHER_DATA_DIR > PIGBROTHER_DATADIR > PIGBROTHER
#
# Usage:
#   cd /path/to/python_magnetrun
#   bash examples/run_hybrid_plotting_examples.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXAMPLES_DIR="$ROOT_DIR/examples"

# ---------------------------------------------------------------------------
# Resolve data directories
# ---------------------------------------------------------------------------
_first_env() {
    local default="${@: -1}"
    local vars=("${@:1:$#-1}")
    for v in "${vars[@]}"; do
        local val="${!v:-}"
        if [[ -n "$val" ]]; then
            echo "$val"
            return
        fi
    done
    echo "$default"
}

HYBRID_DIR="$(_first_env \
    MAGNETRUN_HYBRID_DATA_DIR HYBRID_DATADIR \
    "")"

PUPITRE_DIR="$(_first_env \
    MAGNETRUN_PUPITRE_DATA_DIR PUPITRE_DATADIR MAGNETRUN_DATA_DIR \
    "")"

PIGBROTHER_DIR="$(_first_env \
    MAGNETRUN_PIGBROTHER_DATA_DIR PIGBROTHER_DATADIR PIGBROTHER \
    "")"

# ---------------------------------------------------------------------------
# Temporary output directory (cleaned up on exit)
# ---------------------------------------------------------------------------
OUTDIR="$(mktemp -d /tmp/magnetrun_hybrid_examples_XXXXXX)"
trap 'rm -rf "$OUTDIR"' EXIT

# ---------------------------------------------------------------------------
# Test runner (mirrors verify_examples.sh)
# ---------------------------------------------------------------------------
PASS=0
FAIL=0
SKIP=0
LOG_FILE="$(mktemp /tmp/magnetrun_hybrid_log_XXXXXX.log)"
trap 'rm -rf "$OUTDIR" "$LOG_FILE"' EXIT

run_test() {
    local label="$1"
    local cmd="$2"
    local skip_reason="${3:-}"

    if [[ -n "$skip_reason" ]]; then
        printf "  SKIP  %-60s  (%s)\n" "$label" "$skip_reason"
        SKIP=$((SKIP + 1))
        return
    fi

    if (cd "$ROOT_DIR" && eval "$cmd" >"$LOG_FILE" 2>&1); then
        printf "  PASS  %s\n" "$label"
        PASS=$((PASS + 1))
    else
        printf "  FAIL  %s\n" "$label"
        tail -5 "$LOG_FILE" | sed 's/^/        /'
        FAIL=$((FAIL + 1))
    fi
}

# ---------------------------------------------------------------------------
# Header
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

SCRIPT="python examples/plot_hybrid_with_pupitre_tdms.py"
_NO_HYBRID="hybrid dir not set or not found — set MAGNETRUN_HYBRID_DATA_DIR"

# ---------------------------------------------------------------------------
# Smoke test: --help must always work (no data required)
# ---------------------------------------------------------------------------
echo "--- Argument-parser smoke test ---"
run_test "plot_hybrid_with_pupitre_tdms.py --help" \
    "$SCRIPT --help"
echo ""

# ---------------------------------------------------------------------------
# Check whether hybrid data is present; skip all data-dependent tests if not.
# ---------------------------------------------------------------------------
_hybrid_ok=""
if [[ -n "$HYBRID_DIR" && -d "$HYBRID_DIR" ]]; then
    _hybrid_ok="yes"
fi

# ---------------------------------------------------------------------------
# Example 1 — Basic usage (M8, FEPC-AUX-LNCMI, ALIM1_J1)
# README: python plot_hybrid_with_pupitre_tdms.py \
#           -d 2025-11-02 -s FEPC-AUX-LNCMI -k ALIM1_J1 --site M8 --show
# ---------------------------------------------------------------------------
echo "--- Example 1: basic usage (M8, FEPC-AUX-LNCMI, ALIM1_J1) ---"
if [[ -n "$_hybrid_ok" ]]; then
    run_test "basic: M8 FEPC-AUX-LNCMI ALIM1_J1 2025-11-02" \
        "$SCRIPT \
            -d 2025-11-02 \
            -s FEPC-AUX-LNCMI \
            -k ALIM1_J1 \
            --site M8 \
            --hybrid-dir \"$HYBRID_DIR\" \
            --save \"$OUTDIR/example1_M8_ALIM1_J1.png\""
else
    run_test "basic: M8 FEPC-AUX-LNCMI ALIM1_J1 2025-11-02" "" "$_NO_HYBRID"
fi
echo ""

# ---------------------------------------------------------------------------
# Example 2 — Custom data directories (M9, FEPC-LNCMI, I_H1)
# README: python plot_hybrid_with_pupitre_tdms.py \
#           -d 2025-01-27 -s FEPC-LNCMI -k I_H1 --site M9 \
#           --hybrid-dir ... --pupitre-dir ... --pigbrother-dir ... --show
# ---------------------------------------------------------------------------
echo "--- Example 2: custom directories (M9, FEPC-LNCMI, I_H1) ---"
_skip2=""
if [[ -z "$_hybrid_ok" ]]; then
    _skip2="$_NO_HYBRID"
elif [[ -z "$PUPITRE_DIR" ]]; then
    _skip2="PUPITRE_DIR not set — set MAGNETRUN_PUPITRE_DATA_DIR"
elif [[ -z "$PIGBROTHER_DIR" ]]; then
    _skip2="PIGBROTHER_DIR not set — set MAGNETRUN_PIGBROTHER_DATA_DIR"
fi

if [[ -z "$_skip2" ]]; then
    run_test "custom dirs: M9 FEPC-LNCMI I_H1 2025-01-27" \
        "$SCRIPT \
            -d 2025-01-27 \
            -s FEPC-LNCMI \
            -k I_H1 \
            --site M9 \
            --hybrid-dir \"$HYBRID_DIR\" \
            --pupitre-dir \"$PUPITRE_DIR\" \
            --pigbrother-dir \"$PIGBROTHER_DIR\" \
            --save \"$OUTDIR/example2_M9_I_H1.png\""
else
    run_test "custom dirs: M9 FEPC-LNCMI I_H1 2025-01-27" "" "$_skip2"
fi
echo ""

# ---------------------------------------------------------------------------
# Example 3 — Plot specific hours (M10, FEPC-AUX-LNCMI, ALIM1_J1, hours 10-12)
# README: python plot_hybrid_with_pupitre_tdms.py \
#           -d 2025-01-27 -s FEPC-AUX-LNCMI -k ALIM1_J1 --site M10 \
#           --hours 10,11,12 --show
# ---------------------------------------------------------------------------
echo "--- Example 3: specific hours (M10, FEPC-AUX-LNCMI, hours 10,11,12) ---"
if [[ -n "$_hybrid_ok" ]]; then
    run_test "hours: M10 FEPC-AUX-LNCMI ALIM1_J1 2025-01-27 --hours 10,11,12" \
        "$SCRIPT \
            -d 2025-01-27 \
            -s FEPC-AUX-LNCMI \
            -k ALIM1_J1 \
            --site M10 \
            --hours 10,11,12 \
            --hybrid-dir \"$HYBRID_DIR\" \
            --save \"$OUTDIR/example3_M10_hours.png\""
else
    run_test "hours: M10 FEPC-AUX-LNCMI ALIM1_J1 2025-01-27 --hours 10,11,12" \
        "" "$_NO_HYBRID"
fi
echo ""

# ---------------------------------------------------------------------------
# Example 4 — Save plot to file (no --show)
# README: python plot_hybrid_with_pupitre_tdms.py \
#           -d 2025-01-27 -s FEPC-AUX-LNCMI -k ALIM1_J1 --site M10 \
#           --save comparison_plot.png
# ---------------------------------------------------------------------------
echo "--- Example 4: save to file (M10, FEPC-AUX-LNCMI, ALIM1_J1) ---"
if [[ -n "$_hybrid_ok" ]]; then
    run_test "save: M10 FEPC-AUX-LNCMI ALIM1_J1 2025-01-27" \
        "$SCRIPT \
            -d 2025-01-27 \
            -s FEPC-AUX-LNCMI \
            -k ALIM1_J1 \
            --site M10 \
            --hybrid-dir \"$HYBRID_DIR\" \
            --save \"$OUTDIR/example4_comparison_plot.png\""
else
    run_test "save: M10 FEPC-AUX-LNCMI ALIM1_J1 2025-01-27" "" "$_NO_HYBRID"
fi
echo ""

# ---------------------------------------------------------------------------
# Example 5 — Save and show (--show omitted to avoid blocking; --save kept)
# README: python plot_hybrid_with_pupitre_tdms.py \
#           -d 2025-01-27 -s FEPC-AUX-LNCMI -k ALIM1_J1 --site M10 \
#           --save comparison_plot.png --show
# ---------------------------------------------------------------------------
echo "--- Example 5: save + show (M10, FEPC-AUX-LNCMI, ALIM1_J1) ---"
if [[ -n "$_hybrid_ok" ]]; then
    run_test "save+show: M10 FEPC-AUX-LNCMI ALIM1_J1 2025-01-27" \
        "$SCRIPT \
            -d 2025-01-27 \
            -s FEPC-AUX-LNCMI \
            -k ALIM1_J1 \
            --site M10 \
            --hybrid-dir \"$HYBRID_DIR\" \
            --save \"$OUTDIR/example5_comparison_plot.png\""
else
    run_test "save+show: M10 FEPC-AUX-LNCMI ALIM1_J1 2025-01-27" "" "$_NO_HYBRID"
fi
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "========================================================"
printf "  Results: %d passed, %d failed, %d skipped\n" \
    "$PASS" "$FAIL" "$SKIP"
echo "========================================================"

[[ $FAIL -eq 0 ]]
