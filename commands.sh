#!/usr/bin/env bash
# ============================================================================
# Phase 2B-sweep recipe — 2026-05-19
#
# Two-champion sweep at fixed architecture (both QuartoCNNAutoregUnifiedS4):
#   - champS4 (Sa_archScan, uniform-sampled training) — 12 new SAE configs
#   - champTa (Ta_minimaxSelect, depth-2 minimax training) — 13 SAE configs
#                                                            + LP baseline pass
#
# Runs on Deep Brain (3x A6000, all equal). Speed Mind lacks the .pt files.
# Actual A6000 timing: ~5-15 min/config (25k batches, early-stop). Full sweep
# (~25 configs across 3 GPUs) completes in ~3h — one session, not overnight.
#
# Prerequisites (one-time, see configs/champTa/README.md):
#   - models/quarto/20260516_1452-Ta_minimaxSelect(1)0514_DEPTH_2_E_4350.pt   (trained)
#   - models/quarto/20260516_1452-Ta_minimaxSelect(1)0514_DEPTH_2_E_0000.pt   (random)
#   - models/quarto/<champS4 trained + random>                                (already in repo)
# ============================================================================
set -euo pipefail

DATA=data/quarto
SEED=42
NUM_GAMES=10000
DEVICE=cuda

CHAMP_TA="models/quarto/20260516_1452-Ta_minimaxSelect(1)0514_DEPTH_2_E_4350.pt"
RAND_TA="models/quarto/20260515_1110-Ta_minimaxSelect(1)0514_DEPTH_2_E_0000.pt"

POSITIONS_TA="$DATA/positions-amalgam_ta_unique.pt"

# ============================================================================
# PART 1 — champTa data generation (positions, BSP labels, activations)
# ----------------------------------------------------------------------------
# Skip these steps if data/quarto/positions-amalgam_ta_unique.pt already
# exists. Expected wall time: ~1h on a single GPU.
# ============================================================================

echo "=== [1/4] Competence audit: champTa ==="
python scripts/model_competence_audit.py \
    --model-config=configs/models/champTa.yaml \
    --positions="$DATA/positions-amalgam_unique.pt" \
    --num-positions=5000 --device="$DEVICE" --seed="$SEED" \
    --output="$DATA/model_competence_audit-champTa.json"

echo "=== [2/4] champTa self-play positions ($NUM_GAMES games x 4 modes) ==="
for mode in random_v_random model_v_random random_v_model model_v_model; do
    python scripts/generate_positions.py \
        --game quarto_s4 --opponents "$mode" \
        --model "$CHAMP_TA" --num-games "$NUM_GAMES" --seed "$SEED"
done

echo "=== [2b/4] Aggregate + dedup champTa positions ==="
shopt -s nullglob
RAW_FILES=(
    "$DATA/positions-random_v_random_raw.pt"
    "$DATA"/positions-model_v_random-*Ta_minimaxSelect*_raw.pt
    "$DATA"/positions-random_v_model-*Ta_minimaxSelect*_raw.pt
    "$DATA"/positions-model_v_model-*Ta_minimaxSelect*_raw.pt
)
python scripts/deduplicate_positions.py "${RAW_FILES[@]}" --output "$POSITIONS_TA"

echo "=== [3/4] BSP labels: gorillaTa / hawkTa ==="
python scripts/compute_bsp_labels.py "$POSITIONS_TA" \
    --game quarto --name gorilla \
    --output "$DATA/bsp_labels-gorillaTa_164.pt" \
    --schema-out "$DATA/bsp_schema-gorillaTa_164.json"
python scripts/compute_bsp_labels.py "$POSITIONS_TA" \
    --game quarto --name hawk \
    --output "$DATA/bsp_labels-hawkTa_173.pt" \
    --schema-out "$DATA/bsp_schema-hawkTa_173.json"

echo "=== [4/4] Activations: champTa trained + random, s4.fc1 + s4.conv2 ==="
python scripts/collect_activations.py "$CHAMP_TA" \
    --hook s4.fc1 --game quarto_s4 \
    --positions-file "$POSITIONS_TA" \
    --output "$DATA/s4.fc1_amalgam_ta_activations.pt" --device "$DEVICE"
python scripts/collect_activations.py "$RAND_TA" \
    --hook s4.fc1 --game quarto_s4 \
    --positions-file "$POSITIONS_TA" \
    --output "$DATA/s4.fc1_amalgam_ta_random_activations.pt" --device "$DEVICE"
python scripts/collect_activations.py "$CHAMP_TA" \
    --hook s4.conv2 --game quarto_s4 \
    --positions-file "$POSITIONS_TA" \
    --output "$DATA/s4.conv2_amalgam_ta_activations.pt" \
    --device "$DEVICE" --flatten-position
python scripts/collect_activations.py "$RAND_TA" \
    --hook s4.conv2 --game quarto_s4 \
    --positions-file "$POSITIONS_TA" \
    --output "$DATA/s4.conv2_amalgam_ta_random_activations.pt" \
    --device "$DEVICE" --flatten-position

# ============================================================================
# PART 2 — LP baseline on champTa (8 configs)
# ----------------------------------------------------------------------------
# Establishes the upper bound for any SAE on champTa activations. Mirror of
# the 2026-05-19 champS4 LP block; outputs land at
#   data/quarto/linear_probe_<bsp_set>_<act_stem>_results.json
# Wall time: ~2-3h sequentially; ~1h split across 3 GPUs (run by hand).
# ============================================================================
echo "=== LP baseline: champTa ==="
HOOKS=(s4.fc1 s4.conv2)
SETS=("gorillaTa 164" "hawkTa 173")
for hook in "${HOOKS[@]}"; do
    for net_tag in "" "_random"; do
        ACT="$DATA/${hook}_amalgam_ta${net_tag}_activations.pt"
        [ -f "$ACT" ] || { echo "WARN: missing $ACT — skipping" >&2; continue; }
        for entry in "${SETS[@]}"; do
            read -r animal count <<< "$entry"
            LBL="$DATA/bsp_labels-${animal}_${count}.pt"
            SCH="$DATA/bsp_schema-${animal}_${count}.json"
            echo "--- LP: $hook (${net_tag:+random}${net_tag:-trained})  vs  $animal ---"
            python scripts/linear_probe_baseline.py \
                "$ACT" "$LBL" "$SCH" --seed="$SEED"
        done
    done
done

# ============================================================================
# PART 3 — SAE sweep launchers (3 GPUs, run each in its own terminal)
# ----------------------------------------------------------------------------
# All three A6000s are equal — use even round-robin splits.
# Observed training time: ~5-15 min/config (25k batches, early-stop on A6000).
# Estimated wall time: ~25 configs / 3 GPUs × ~12 min avg = ~1.7h training
#                    + ~5 min/config × 2 BSP sets × 9 configs = ~1.5h eval
#                    → full sweep + eval in ~3h (single session).
# --eval --bsps=<primary> runs inline eval after each training job so results
# appear as training progresses. Hawk-set eval is in the follow-up block.
# Comment out / uncomment the desired block before running.
# ============================================================================

# # --- champS4 (12 new) then champTa (13 new) — both sweeps, one command ---
# python validate_sweep.py --configs=configs/champS4 --smoke-test && python validate_sweep.py --configs=configs/champTa --smoke-test
#
# python run_sweep.py --configs=configs/champS4 --gpu=0 --split=1/3 --eval --bsps=gorillaS4 --skip-existing & python run_sweep.py --configs=configs/champS4 --gpu=1 --split=2/3 --eval --bsps=gorillaS4 --skip-existing & python run_sweep.py --configs=configs/champS4 --gpu=2 --split=3/3 --eval --bsps=gorillaS4 --skip-existing & wait && python run_sweep.py --configs=configs/champTa --gpu=0 --split=1/3 --eval --bsps=gorillaTa --skip-existing & python run_sweep.py --configs=configs/champTa --gpu=1 --split=2/3 --eval --bsps=gorillaTa --skip-existing & python run_sweep.py --configs=configs/champTa --gpu=2 --split=3/3 --eval --bsps=gorillaTa --skip-existing & wait

# # --- Hawk-set eval + final report (run AFTER both sweeps complete) ---
# # gorillaS4/gorillaTa are already cached from inline --eval above; only hawk needed.
# echo "=== Evaluating champS4 sweep against hawkS4 ==="
# for cfg in configs/champS4/{A01v2,E*,F*}-champS4-*.yaml; do
#     ckpt=$(python scripts/run_id.py "$cfg" --checkpoint)
#     [ -f "$ckpt" ] || { echo "WARN: $ckpt not found" >&2; continue; }
#     python sae_eval.py evaluate "$ckpt" --bsps=hawkS4
# done
#
# echo "=== Evaluating champTa sweep against hawkTa ==="
# for cfg in configs/champTa/*-champTa-*.yaml; do
#     ckpt=$(python scripts/run_id.py "$cfg" --checkpoint)
#     [ -f "$ckpt" ] || { echo "WARN: $ckpt not found" >&2; continue; }
#     python sae_eval.py evaluate "$ckpt" --bsps=hawkTa
# done
#
# # --- Final report: top runs + cross-champion A/B ---
# python scripts/registry_query.py top --bsps=gorillaS4 --limit=20
# python scripts/registry_query.py top --bsps=hawkS4 --limit=20
# python scripts/registry_query.py top --bsps=gorillaTa --limit=20
# python scripts/registry_query.py top --bsps=hawkTa --limit=20
# for id in E01 E02 E03 E04 E05 E06 E07 F01 F02 F03 F04; do
#     echo "--- $id: champS4 vs champTa ---"
#     S4=$(python scripts/run_id.py configs/champS4/$id-champS4-*.yaml)
#     TA=$(python scripts/run_id.py configs/champTa/$id-champTa-*.yaml)
#     python scripts/registry_query.py compare "$S4" "$TA" \
#         --bsps=gorillaS4 --bsps-b=gorillaTa || true
# done

echo "=== Phase 2B data + LP done. Uncomment Part 3 blocks to launch sweeps (~3h on 3x A6000). ==="
