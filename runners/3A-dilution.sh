#!/usr/bin/env bash
# runners/3A-dilution.sh -- Phase 3A dilution diagnostic over the champTa/champVe
# winner SAEs. Self-contained, committed runner (see CLAUDE.md "runners/").
# Runs on Deep Brain, where the _h.pt code caches and champion-suffixed BSP
# labels live. It: (1) verifies deps, (2) checks each run's inputs are present
# before invoking (clear SKIP otherwise), (3) writes one JSON per run to
# saes/quarto/analysis/<run_id>_dilution-<bsps>.json + a combined
# 3A_gate_summary.json, and (4) emits stage_3A-dilution.md at the repo root.
#
# Method + thresholds: docs/diary/2026-07-21_3A-dilution-diagnostic.md
# Usage:  bash runners/3A-dilution.sh            # run all present configs
#         DRY_RUN=1 bash runners/3A-dilution.sh  # only report what would run
#
# NOTE: ASCII only (host prints under cp1252).

set -uo pipefail
cd "$(dirname "$0")/.." || { echo "FATAL: cannot cd to repo root"; exit 1; }
export PYTHONUTF8=1

SLUG="3A-dilution"
GAME=quarto
CACHE_DIR="saes/${GAME}/cache"
DATA_DIR="data/${GAME}"
ANALYSIS_DIR="saes/${GAME}/analysis"
DRY_RUN="${DRY_RUN:-0}"
TOP_K="${TOP_K:-64}"

# --- run table: "run_id|bsps|random_run_id" (random_run_id may be 'none') ------
RUNS=(
  "F04-champTa-s42-jumprelu-t64-exp8-s4.fc1|tigerTa|none"
  "F04-champTa-s42-jumprelu-t64-exp8-s4.fc1|gorillaTa|none"
  "E05-champTa-s42-batchtopk-k32-exp8-s4.conv2|tigerTa|none"
  "I04-champTa-lh100-s43-anchored-jumprelu-t64-exp8-s4.fc1|tigerTa|none"
  "E01-champVe-s42-topk-k32-exp8-s4.conv2|gorillaVe|none"
  "E05-champVe-s42-batchtopk-k32-exp8-s4.conv2|tigerVe|none"
  "I04-champVe-lh100-s42-anchored-jumprelu-t64-exp8-s4.fc1|tigerVe|none"
  "I03-champVe-lh030-s43-anchored-jumprelu-t64-exp8-s4.fc1|gorillaVe|none"
)

echo "=================================================================="
echo "Phase 3A dilution diagnostic runner"
echo "=================================================================="

# --- 1. environment ------------------------------------------------------------
command -v python >/dev/null 2>&1 || { echo "FATAL: python not on PATH"; exit 1; }
python - <<'PY'
import sys
missing = []
for mod in ("torch", "numpy", "sklearn", "scipy", "networkx", "docopt"):
    try:
        __import__(mod)
    except Exception as e:
        missing.append(f"{mod} ({e})")
if missing:
    print("FATAL: missing dependencies: " + ", ".join(missing)); sys.exit(1)
print("deps OK: torch numpy sklearn scipy networkx docopt")
PY
[ $? -ne 0 ] && exit 1

[ -f "lib/sae/dilution.py" ] || { echo "FATAL: lib/sae/dilution.py missing"; exit 1; }
[ -f "scripts/dilution_diagnostic.py" ] || { echo "FATAL: scripts/dilution_diagnostic.py missing"; exit 1; }
mkdir -p "$ANALYSIS_DIR"

# --- 2. pre-flight: report per-run input presence ------------------------------
label_exists() { ls "${DATA_DIR}/bsp_labels-${1}_"[0-9]*.pt >/dev/null 2>&1; }
echo
echo "Pre-flight (input presence):"
runnable=()
for entry in "${RUNS[@]}"; do
    IFS='|' read -r rid bsps rand <<< "$entry"
    ok=1; why=""
    [ -f "${CACHE_DIR}/${rid}_h.pt" ] || { ok=0; why="${why} no _h.pt;"; }
    label_exists "$bsps" || { ok=0; why="${why} no bsp_labels-${bsps};"; }
    if [ "$rand" != "none" ] && [ ! -f "${CACHE_DIR}/${rand}_h.pt" ]; then
        why="${why} (random control ${rand} absent -> permutation null);"
    fi
    if [ $ok -eq 1 ]; then
        echo "  [ OK ] ${rid}  bsps=${bsps}${why:+  NOTE:${why}}"
        runnable+=("$entry")
    else
        echo "  [SKIP] ${rid}  bsps=${bsps} --${why}"
    fi
done

if [ ${#runnable[@]} -eq 0 ]; then
    echo; echo "Nothing runnable here (expected on Speed Mind; caches live on Deep Brain)."
    echo "Build a cache with: python sae_eval.py evaluate saes/${GAME}/<run_id>.pt --bsps=<set> --force"
    exit 0
fi

if [ "$DRY_RUN" = "1" ]; then
    echo; echo "DRY_RUN=1 -> not executing. ${#runnable[@]} run(s) would execute."
    exit 0
fi

# --- 3. execute ----------------------------------------------------------------
echo; echo "Executing ${#runnable[@]} run(s)..."
for entry in "${runnable[@]}"; do
    IFS='|' read -r rid bsps rand <<< "$entry"
    echo; echo "------------------------------------------------------------------"
    echo ">> ${rid}  bsps=${bsps}  random=${rand}"
    python scripts/dilution_diagnostic.py --run-id="$rid" --bsps="$bsps" \
        --random-run-id="$rand" --top-k="$TOP_K"
done

# --- 4. combined gate summary --------------------------------------------------
echo; echo "Writing combined gate summary..."
python - "$ANALYSIS_DIR/3A_gate_summary.json" "$ANALYSIS_DIR"/*_dilution-*.json <<'PY'
import json, sys, glob
out_path = sys.argv[1]
rows = []
for p in sys.argv[2:]:
    with open(p) as f:
        d = json.load(f)
    s, g = d["summary"], d["gate_g3a"]
    rows.append({"run_id": s["run_id"], "bsp_set": s["bsp_set"],
                 "n_threat_bsps": g["n_threat_bsps"], "n_diluted": g["n_diluted"],
                 "n_tiled": g["n_tiled"], "n_captured": g["n_captured"],
                 "n_absent": g["n_absent"], "geometric_frac": g["geometric_frac"],
                 "verdict": g["verdict"]})
with open(out_path, "w") as f:
    json.dump({"runs": rows}, f, indent=2, sort_keys=True)
print(f"{'run_id':<52}{'bsps':<11}{'geo':>6}  verdict")
print("-" * 82)
for r in rows:
    print(f"{r['run_id']:<52}{r['bsp_set']:<11}{r['geometric_frac']:>6.2f}  {r['verdict']}")
print(f"\nSaved: {out_path}")
PY

# --- 5. stage plan -------------------------------------------------------------
echo; echo "Emitting stage plan..."
python scripts/emit_stage.py --slug="$SLUG" \
    "$ANALYSIS_DIR"/*_dilution-*.json \
    "$ANALYSIS_DIR/3A_gate_summary.json"

echo
echo "Done. Per-run JSON in ${ANALYSIS_DIR}/; commit plan in stage_${SLUG}.md"
