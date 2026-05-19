#!/usr/bin/env bash
# ============================================================================
# Linear-probe baseline sweep for the champS4 follow-up (Phase 2B diagnostic).
#
# Trains per-BSP L2 logistic regression on raw activations and reports F1, MCC,
# F1-lift — directly comparable to sae_eval coverage metrics. Establishes the
# upper bound any SAE can recover from these activations.
#
# Runs 8 configurations: {s4.fc1, s4.conv2} × {trained, random} × {gorillaS4,
# hawkS4}. Outputs go to data/quarto/linear_probe_<bsp_set>_<act_stem>_results.json.
#
# Usage:  bash scripts/lp_baseline_champS4.sh
# Expects:
#   - CWD = games-interp project root
#   - data/quarto/{s4.fc1,s4.conv2}_amalgam_s4{,_random}_activations.pt
#   - data/quarto/bsp_labels-{gorillaS4_164,hawkS4_173}.pt
#   - data/quarto/bsp_schema-{gorillaS4_164,hawkS4_173}.json
# Speed Mind does not have the .pt activation files locally; run this on the
# Deep Brain server where the champS4 amalgam_s4 data lives.
# ============================================================================
set -euo pipefail

DATA=data/quarto
SEED=42

HOOKS=(s4.fc1 s4.conv2)
SETS=("gorillaS4 164" "hawkS4 173")

for hook in "${HOOKS[@]}"; do
    for net_tag in "" "_random"; do
        ACT="$DATA/${hook}_amalgam_s4${net_tag}_activations.pt"
        if [ ! -f "$ACT" ]; then
            echo "WARN: missing $ACT — skipping" >&2
            continue
        fi
        for entry in "${SETS[@]}"; do
            read -r animal count <<< "$entry"
            LBL="$DATA/bsp_labels-${animal}_${count}.pt"
            SCH="$DATA/bsp_schema-${animal}_${count}.json"
            tag_label="${net_tag:+random }${net_tag:-trained}"
            echo "=== LP: $hook ($tag_label)  vs  $animal ==="
            python scripts/linear_probe_baseline.py \
                "$ACT" "$LBL" "$SCH" --seed="$SEED"
        done
    done
done

echo "=== Done. Results: $DATA/linear_probe_*S4*.json ==="
