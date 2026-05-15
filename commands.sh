#!/usr/bin/env bash
# ============================================================================
# Full SAE re-evaluation pipeline for the champS4 Quarto model.
#
# Mirrors the original Aa_replay (champAa) pipeline against the new unified-aux
# autoregressive model. See configs/champS4/README.md for the rationale and
# Quarto-specifications.md for naming conventions.
#
# Usage:  bash commands.sh
# Expects: CWD = games-interp project root, CUDA available.
# ============================================================================
set -euo pipefail

CHAMP="models/quarto/20260514_0815-Sa_archScan(3)0512_ARCH_S4_uniform512_E_5000.pt"
RAND_S4="models/quarto/20260512_1845-Sa_archScan(3)0512_ARCH_S4_uniform512_E_0000.pt"
SEED=42
NUM_GAMES=10000
DEVICE=cuda

DATA=data/quarto
POSITIONS="$DATA/positions-amalgam_s4_unique.pt"

# ----------------------------------------------------------------------------
# Step 1 — Model competence audit (gates interpretability claims)
# ----------------------------------------------------------------------------
echo "=== [1/6] Competence audit: champS4 ==="
python scripts/model_competence_audit.py \
    --model-config=configs/models/champS4.yaml \
    --positions="$DATA/positions-amalgam_unique.pt" \
    --num-positions=5000 --device="$DEVICE" --seed="$SEED" \
    --output="$DATA/model_competence_audit-champS4.json"

# Optional re-baseline of champAa via the new YAML-driven entry point.
# Comment out if you already have the v1-schema result on disk.
echo "=== [1b/6] Competence audit: champAa (re-baseline) ==="
python scripts/model_competence_audit.py \
    --model-config=configs/models/champAa.yaml \
    --positions="$DATA/positions-amalgam_unique.pt" \
    --num-positions=5000 --device="$DEVICE" --seed="$SEED" \
    --output="$DATA/model_competence_audit-champAa.json"

# ----------------------------------------------------------------------------
# Step 2 — Generate S4 self-play positions (4 opponent modes)
# ----------------------------------------------------------------------------
echo "=== [2/6] Generating S4 self-play positions ($NUM_GAMES games × 4 modes) ==="
for mode in random_v_random model_v_random random_v_model model_v_model; do
    python scripts/generate_positions.py \
        --game quarto_s4 --opponents "$mode" \
        --model "$CHAMP" --num-games "$NUM_GAMES" --seed "$SEED"
done

# ----------------------------------------------------------------------------
# Step 3 — Aggregate + deduplicate across the 4 raw files
# ----------------------------------------------------------------------------
echo "=== [3/6] Deduplicating across opponent modes ==="
# Glob picks up the 4 raw files just produced. The S4-specific files have the
# Sa_archScan model tag injected by extract_model_name(); random_v_random is
# model-agnostic.
shopt -s nullglob
RAW_FILES=(
    "$DATA/positions-random_v_random_raw.pt"
    "$DATA"/positions-model_v_random-*_raw.pt
    "$DATA"/positions-random_v_model-*_raw.pt
    "$DATA"/positions-model_v_model-*_raw.pt
)
python scripts/deduplicate_positions.py "${RAW_FILES[@]}" --output "$POSITIONS"

# ----------------------------------------------------------------------------
# Step 4 — BSP labels for both sets against the new S4 positions
# ----------------------------------------------------------------------------
# We use distinct animal names (gorillaS4 / hawkS4) so that label files live
# alongside the champAa ones in data/quarto/ without glob collisions, and so
# that the eval registry key (run_id:bsp_set) carries the distribution tag
# alongside the concept menu. The champion + position distribution are
# different here even though the concept categories are identical.
echo "=== [4/6] Computing BSP labels for S4 positions ==="
python scripts/compute_bsp_labels.py "$POSITIONS" \
    --game quarto --name gorilla \
    --output "$DATA/bsp_labels-gorillaS4_164.pt" \
    --schema-out "$DATA/bsp_schema-gorillaS4_164.json"
python scripts/compute_bsp_labels.py "$POSITIONS" \
    --game quarto --name hawk \
    --output "$DATA/bsp_labels-hawkS4_173.pt" \
    --schema-out "$DATA/bsp_schema-hawkS4_173.json"

# ----------------------------------------------------------------------------
# Step 5 — Collect activations (champion + random-init control, 2 hooks)
# ----------------------------------------------------------------------------
echo "=== [5/6] Collecting activations ==="
# s4.fc1 (d_act=512) — no flatten
python scripts/collect_activations.py "$CHAMP" \
    --hook s4.fc1 --game quarto_s4 \
    --positions-file "$POSITIONS" \
    --output "$DATA/s4.fc1_amalgam_s4_activations.pt" --device "$DEVICE"

python scripts/collect_activations.py "$RAND_S4" \
    --hook s4.fc1 --game quarto_s4 \
    --positions-file "$POSITIONS" \
    --output "$DATA/s4.fc1_amalgam_s4_random_activations.pt" --device "$DEVICE"

# s4.conv2 (32 channels × 4×4 → 512 flat) — flatten for position-level SAEs
python scripts/collect_activations.py "$CHAMP" \
    --hook s4.conv2 --game quarto_s4 \
    --positions-file "$POSITIONS" \
    --output "$DATA/s4.conv2_amalgam_s4_activations.pt" \
    --device "$DEVICE" --flatten-position

python scripts/collect_activations.py "$RAND_S4" \
    --hook s4.conv2 --game quarto_s4 \
    --positions-file "$POSITIONS" \
    --output "$DATA/s4.conv2_amalgam_s4_random_activations.pt" \
    --device "$DEVICE" --flatten-position

# ----------------------------------------------------------------------------
# Step 6 — Train + evaluate the 4 champS4 SAEs (top runs from the Aa sweep)
# ----------------------------------------------------------------------------
echo "=== [6/6] Sweep: train the champS4 configs ==="
python validate_sweep.py --configs=configs/champS4 --smoke-test
# NOTE: --eval is deliberately omitted; the sweep's default eval uses
# --bsps=gorilla, which would pick the wrong (champAa) BSP labels for S4 runs.
# We run two explicit eval loops below against the S4-specific label sets.
python run_sweep.py --configs=configs/champS4 --gpu=1 --skip-existing

echo "=== Evaluating champS4 SAEs against gorillaS4 + hawkS4 ==="
for cfg in configs/champS4/*.yaml; do
    ckpt=$(python scripts/run_id.py "$cfg" --checkpoint)
    if [ ! -f "$ckpt" ]; then
        echo "WARN: expected checkpoint not found: $ckpt" >&2
        continue
    fi
    python sae_eval.py evaluate "$ckpt" --bsps=gorillaS4
    python sae_eval.py evaluate "$ckpt" --bsps=hawkS4
done

# Run-ids derived from the YAMLs (deterministic; see scripts/run_id.py).
# This avoids hard-coding the trainer's filename rule in shell strings.
declare -A S4_RUN_ID
declare -A AA_RUN_ID
S4_RUN_ID[C01]=$(python scripts/run_id.py configs/champS4/C01-champS4-conv2-topk-k16-exp8-s42.yaml)
S4_RUN_ID[C07]=$(python scripts/run_id.py configs/champS4/C07-champS4-conv2-jumprelu-t32-exp8-s42.yaml)
S4_RUN_ID[D02]=$(python scripts/run_id.py configs/champS4/D02-champS4-conv2-topk-k64-exp16-s42.yaml)
S4_RUN_ID[A01]=$(python scripts/run_id.py configs/champS4/A01-champS4-fc1-batchtopk-k16-exp2-s42.yaml)
AA_RUN_ID[C01]=$(python scripts/run_id.py configs/followup/C01-conv2-topk-k16-exp8-arch-s42.yaml)
AA_RUN_ID[C07]=$(python scripts/run_id.py configs/followup/C07-conv2-jumprelu-t32-exp8-arch-s42.yaml)
AA_RUN_ID[D02]=$(python scripts/run_id.py configs/followup/D02-conv2-topk-k64-exp16-s42.yaml)
AA_RUN_ID[A01]=$(python scripts/run_id.py configs/anakin/fc1-batchtopk-k16-exp8.yaml)

# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
echo "=== Top runs in unified registry (gorilla — champAa) ==="
python scripts/registry_query.py top --bsps=gorilla --limit=20
echo "=== Top runs in unified registry (gorillaS4 — champS4) ==="
python scripts/registry_query.py top --bsps=gorillaS4 --limit=20

echo "=== Pairwise comparisons (champS4 vs champAa, cross-BSP-set) ==="
for tag in C01 C07 D02 A01; do
    echo "--- $tag: ${S4_RUN_ID[$tag]} (gorillaS4) vs ${AA_RUN_ID[$tag]} (gorilla) ---"
    python scripts/registry_query.py compare \
        "${S4_RUN_ID[$tag]}" "${AA_RUN_ID[$tag]}" \
        --bsps=gorillaS4 --bsps-b=gorilla || true
done

echo "=== Done ==="
