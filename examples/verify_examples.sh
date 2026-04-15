#!/usr/bin/env bash
# verify_examples.sh — Run example scripts and report pass / fail / skip.
#
# Data directories resolved from env vars (same priority as data_dirs.py):
#   pupitre:    MAGNETRUN_PUPITRE_DATA_DIR > PUPITRE_DATADIR
#   pigbrother: MAGNETRUN_PIGBROTHER_DATA_DIR > PIGBROTHER_DATADIR
#   hybrid:     MAGNETRUN_HYBRID_DATA_DIR > HYBRID_DATADIR
#
# Optional:
#   PROPOSALS_CSV   path to a proposals CSV file required by proposal.py

set -uo pipefail

export MPLBACKEND=Agg

PASS=0
FAIL=0
SKIP=0
ERRORS=()

# Change to repo root
cd "$(dirname "$0")/.." || exit 1

EXAMPLES="examples"
LOCAL_DATA="data"

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

PUPITRE_DIR="$(_first_env MAGNETRUN_PUPITRE_DATA_DIR PUPITRE_DATADIR "$LOCAL_DATA")"
PIGBROTHER_DIR="$(_first_env MAGNETRUN_PIGBROTHER_DATA_DIR PIGBROTHER_DATADIR "$LOCAL_DATA")"
HYBRID_DIR="$(_first_env MAGNETRUN_HYBRID_DATA_DIR HYBRID_DATADIR "")"

PUPITRE_FILE="$(find "$PUPITRE_DIR" -maxdepth 2 -name "*.txt" ! -name "*profiles*" 2>/dev/null | sort | head -1 || true)"
TDMS_FILE="$(find "$PIGBROTHER_DIR" -maxdepth 2 -name "*.tdms" 2>/dev/null | sort | head -1 || true)"

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

_have() { python -c "$1" >/dev/null 2>&1; }

HAS_MAGNETCOOLING=0;          _have "from python_magnetcooling import WaterFlow"            && HAS_MAGNETCOOLING=1
HAS_MAGNETCOOLING_WF=0;       _have "from python_magnetcooling.waterflow import WaterFlow"  && HAS_MAGNETCOOLING_WF=1
HAS_PWLF=0;                   _have "import pwlf"                                           && HAS_PWLF=1
HAS_SKLEARN=0;                 _have "import sklearn"                                        && HAS_SKLEARN=1

# ---------------------------------------------------------------------------
echo "========================================"
echo " python_magnetrun — examples verification"
echo "========================================"
echo "  PUPITRE_DIR   : ${PUPITRE_DIR:-(not set)}"
echo "  PIGBROTHER_DIR: ${PIGBROTHER_DIR:-(not set)}"
echo "  HYBRID_DIR    : ${HYBRID_DIR:-(not set)}"
echo "  PUPITRE_FILE  : ${PUPITRE_FILE:-(none found)}"
echo "  TDMS_FILE     : ${TDMS_FILE:-(none found)}"
echo "  PROPOSALS_CSV : ${PROPOSALS_CSV:-  (not set)}"
echo "========================================"
echo ""

# ---------------------------------------------------------------------------
echo "--- Self-contained (synthetic data) ---"

if [[ $HAS_MAGNETCOOLING_WF -eq 1 ]]; then
    run_cmd  1 python "$EXAMPLES/flow_params_pipeline.py"
else
    skip_cmd 1 "python_magnetcooling top-level WaterFlow not importable" \
        "python $EXAMPLES/flow_params_pipeline.py"
fi

if [[ $HAS_MAGNETCOOLING_WF -eq 1 && $HAS_PWLF -eq 1 ]]; then
    run_cmd  2 python "$EXAMPLES/flow_params_magnetrun_pipeline.py"
else
    skip_cmd 2 "python_magnetcooling or pwlf not available" \
        "python $EXAMPLES/flow_params_magnetrun_pipeline.py"
fi

if [[ $HAS_MAGNETCOOLING_WF -eq 1 ]]; then
    run_cmd  3 python "$EXAMPLES/waterflow_debitbrut_example.py"
else
    skip_cmd 3 "python_magnetcooling.waterflow not importable" \
        "python $EXAMPLES/waterflow_debitbrut_example.py"
fi

run_cmd  4 python "$EXAMPLES/example_fepc_usage.py"
skip_cmd  5 python "$EXAMPLES/example_trigger_usage.py"

echo ""
echo "--- Argument-parser smoke test (--help) ---"

run_cmd  6 python "$EXAMPLES/collect-data.py" --help
run_cmd  7 python "$EXAMPLES/userdb.py" --help
run_cmd  8 python "$EXAMPLES/plot_fepc_data.py" --help
run_cmd  9 python "$EXAMPLES/plot_trigger_data.py" --help
run_cmd 10 python "$EXAMPLES/plot_rms.py" --help
run_cmd 11 python "$EXAMPLES/plot_vprocess.py" --help
run_cmd 12 python "$EXAMPLES/plot_hybrid_with_pupitre_tdms.py" --help
run_cmd 13 python "$EXAMPLES/bilan.py" --help
run_cmd 14 python "$EXAMPLES/get-record.py" --help

echo ""
echo "--- file_stats.py ---"

run_cmd 15 python "$EXAMPLES/file_stats.py" "$LOCAL_DATA" txt

echo ""
echo "--- Pupitre (.txt) dependent ---"

if [[ -n "$PUPITRE_FILE" ]]; then
    run_cmd 16 python "$EXAMPLES/outliers.py" '"$PUPITRE_FILE"' --housing M8

    if [[ $HAS_PWLF -eq 1 ]]; then
        run_cmd 17 python "$EXAMPLES/corr_Ih_Ib.py" '"$PUPITRE_FILE"' --housing M8
    else
        skip_cmd 17 "pwlf not available" "python $EXAMPLES/corr_Ih_Ib.py '$PUPITRE_FILE'" --housing M8
    fi

    run_cmd 18 python "$EXAMPLES/cmp_fields.py" '"$PUPITRE_FILE"' --xkey Field --ykey IH --housing M8
    run_cmd 19 python "$EXAMPLES/pupitre.py" '"$PUPITRE_FILE"'

    if [[ $HAS_SKLEARN -eq 1 ]]; then
        run_cmd 20 python "$EXAMPLES/timeseries-anomaly-detection.py" '"$PUPITRE_FILE"' --housing M8
    else
        skip_cmd 20 "scikit-learn not available" \
            "python $EXAMPLES/timeseries-anomaly-detection.py '$PUPITRE_FILE' --housing M8"
    fi

    run_cmd 21 python "$EXAMPLES/get-record.py" '"$PUPITRE_FILE"' \
        --pupitre_datadir "$PUPITRE_DIR" --pigbrother_datadir "$PIGBROTHER_DIR" select

    run_cmd 22 python "$EXAMPLES/get-record.py" '"$PUPITRE_FILE"' \
        --pupitre_datadir "$PUPITRE_DIR" --pigbrother_datadir "$PIGBROTHER_DIR" \
        stats --fields Field
else
    _skip="no .txt in $PUPITRE_DIR — set MAGNETRUN_PUPITRE_DATA_DIR"
    for i in 16 17 18 19 20 21 22; do
        skip_cmd $i "$_skip" "(pupitre-dependent example)"
    done
fi

echo ""
echo "--- Pigbrother (.tdms) dependent ---"

if [[ -n "$TDMS_FILE" ]]; then
    run_cmd 23 python "$EXAMPLES/bilan.py" '"$TDMS_FILE"' \
        --pupitre_datadir '"$PUPITRE_DIR"' --pigbrother_datadir '"$PIGBROTHER_DIR"'
else
    skip_cmd 23 "no .tdms in $PIGBROTHER_DIR — set MAGNETRUN_PIGBROTHER_DATA_DIR" \
        "python $EXAMPLES/bilan.py ..."
fi

echo ""
echo "--- Hybrid data dependent ---"

if [[ -n "$HYBRID_DIR" && -d "$HYBRID_DIR" ]]; then
    run_cmd 24 python "$EXAMPLES/plot_hybrid_minimal.py"
else
    skip_cmd 24 "hybrid dir not found — set MAGNETRUN_HYBRID_DATA_DIR or HYBRID_DATADIR" \
        "python $EXAMPLES/plot_hybrid_minimal.py"
fi

echo ""
echo "--- Proposals CSV dependent ---"

if [[ -n "${PROPOSALS_CSV:-}" && -f "$PROPOSALS_CSV" ]]; then
    run_cmd 25 python "$EXAMPLES/proposal.py" "$PROPOSALS_CSV" --mdatadir "$PUPITRE_DIR"
else
    skip_cmd 25 "PROPOSALS_CSV not set or file not found" \
        "python $EXAMPLES/proposal.py \$PROPOSALS_CSV --mdatadir ..."
fi

# ---------------------------------------------------------------------------
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
