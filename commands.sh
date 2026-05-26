#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# champVe onboarding pipeline  (2026-05-25)
# =============================================================================
# champVe = Ve_oracleAblation(4) [DISABLE_NEVER, E=10000].
# Same arch as champS4 / champTa (QuartoCNNAutoregUnifiedS4); the variable is
# the training procedure -- minimax-oracle SELECT distillation, oracle never
# disabled, 10k epochs (vs Ta's 4350 with re-enable-only schedule).
# Head-to-head benchmark: champVe beats champTa 59.4% (500 games each side).
#
# Pipeline order: audit -> positions -> dedup -> BSP labels -> activations
# -> linear-probe baseline -> SAE sweep + multi-BSP eval -> read-out.
#
# The script is fully sequential (set -e: stops on first error).
# GPU-parallel steps use & + wait but still gate subsequent steps.
# =============================================================================

CHAMP_VE='models/quarto/20260524_1904-Ve_oracleAblation(4)0522_DISABLE_NEVER_10k_E_10000.pt'
CHAMP_VE_RANDOM='models/quarto/20260522_1832-Ve_oracleAblation(4)0522_DISABLE_NEVER_10k_E_0000.pt'
VE_MODEL_STEM='Ve_oracleAblation'

# --- 0. Competence audit (sanity: model plays Quarto, random net doesn't) ---
if [ -f data/quarto/audit-champVe.json ]; then
    echo "=== Step 0: audit-champVe.json already exists, skipping ==="
else
    echo "=== Step 0: Competence audit ==="
    python scripts/model_competence_audit.py \
        --model-config=configs/models/champVe.yaml \
        --num-positions=5000 --device=cuda \
        --output=data/quarto/audit-champVe.json
fi

# --- 1. Generate raw positions (4 opponent modes, seed=42) ---
echo "=== Step 1: Generate raw positions (4 modes x 10k games) ==="
for mode in random_v_random model_v_random random_v_model model_v_model; do
    echo "  -> $mode"
    python scripts/generate_positions.py --game quarto_s4 \
        --opponents "$mode" --model "$CHAMP_VE" \
        --num-games 10000 --seed 42 --device cuda \
        --output-dir data/quarto_Ve4
done

# --- 2. Aggregate + deduplicate across all four files (NOT per-file dedup!) ---
echo "=== Step 2: Deduplicate positions ==="
python scripts/deduplicate_positions.py \
    data/quarto_Ve4/positions-random_v_random_raw.pt \
    "data/quarto_Ve4/positions-model_v_random-${VE_MODEL_STEM}_raw.pt" \
    "data/quarto_Ve4/positions-random_v_model-${VE_MODEL_STEM}_raw.pt" \
    "data/quarto_Ve4/positions-model_v_model-${VE_MODEL_STEM}_raw.pt" \
    --output data/quarto/positions-amalgam_ve_unique.pt

# --- 3. BSP labels: gorillaVe (164) + hawkVe (173) + tigerVe (36) ---
echo "=== Step 3: Compute BSP labels (gorilla, hawk, tiger) ==="
python scripts/compute_bsp_labels.py data/quarto/positions-amalgam_ve_unique.pt \
    --game quarto_s4 --name gorilla \
    --output data/quarto/bsp_labels-gorillaVe_164.pt \
    --schema-out data/quarto/bsp_schema-gorillaVe_164.json

python scripts/compute_bsp_labels.py data/quarto/positions-amalgam_ve_unique.pt \
    --game quarto_s4 --name hawk \
    --output data/quarto/bsp_labels-hawkVe_173.pt \
    --schema-out data/quarto/bsp_schema-hawkVe_173.json

python scripts/compute_bsp_labels.py data/quarto/positions-amalgam_ve_unique.pt \
    --game quarto_s4 --name tiger \
    --output data/quarto/bsp_labels-tigerVe_36.pt \
    --schema-out data/quarto/bsp_schema-tiger_36.json

# --- 4. Collect activations: trained + random, s4.fc1 + s4.conv2 ---
echo "=== Step 4: Collect activations (4 combos: 2 models x 2 hooks) ==="
echo "  -> trained s4.fc1"
python scripts/collect_activations.py "$CHAMP_VE" --hook s4.fc1 --game quarto_s4 \
    --positions-file data/quarto/positions-amalgam_ve_unique.pt \
    --output data/quarto/s4.fc1_amalgam_ve_activations.pt --device cuda

echo "  -> random s4.fc1"
python scripts/collect_activations.py "$CHAMP_VE_RANDOM" --hook s4.fc1 --game quarto_s4 \
    --positions-file data/quarto/positions-amalgam_ve_unique.pt \
    --output data/quarto/s4.fc1_amalgam_ve_random_activations.pt --device cuda

echo "  -> trained s4.conv2 (flatten)"
python scripts/collect_activations.py "$CHAMP_VE" --hook s4.conv2 --game quarto_s4 \
    --positions-file data/quarto/positions-amalgam_ve_unique.pt \
    --output data/quarto/s4.conv2_amalgam_ve_activations.pt --device cuda --flatten-position

echo "  -> random s4.conv2 (flatten)"
python scripts/collect_activations.py "$CHAMP_VE_RANDOM" --hook s4.conv2 --game quarto_s4 \
    --positions-file data/quarto/positions-amalgam_ve_unique.pt \
    --output data/quarto/s4.conv2_amalgam_ve_random_activations.pt --device cuda --flatten-position

# --- 5. Linear-probe baseline (LP upper bound on the new activations) ---
echo "=== Step 5: Linear-probe baselines (2 hooks x 3 BSP sets) ==="
for hook in s4.fc1 s4.conv2; do
    echo "  -> $hook / gorilla"
    python scripts/linear_probe_baseline.py \
        "data/quarto/${hook}_amalgam_ve_activations.pt" \
        data/quarto/bsp_labels-gorillaVe_164.pt \
        data/quarto/bsp_schema-gorillaVe_164.json
    echo "  -> $hook / hawk"
    python scripts/linear_probe_baseline.py \
        "data/quarto/${hook}_amalgam_ve_activations.pt" \
        data/quarto/bsp_labels-hawkVe_173.pt \
        data/quarto/bsp_schema-hawkVe_173.json
    echo "  -> $hook / tiger"
    python scripts/linear_probe_baseline.py \
        "data/quarto/${hook}_amalgam_ve_activations.pt" \
        data/quarto/bsp_labels-tigerVe_36.pt \
        data/quarto/bsp_schema-tiger_36.json
done

# --- 6. SAE sweep: C/E/F/H configs, s42 only, 3-way GPU split ---
echo "=== Step 6: SAE sweep + gorillaVe eval (3 GPUs) ==="
python run_sweep.py --configs=configs/champVe --gpu=0 --split=1/3 --eval --bsps=gorillaVe --skip-existing &
pid0=$!
python run_sweep.py --configs=configs/champVe --gpu=1 --split=2/3 --eval --bsps=gorillaVe --skip-existing &
pid1=$!
python run_sweep.py --configs=configs/champVe --gpu=2 --split=3/3 --eval --bsps=gorillaVe --skip-existing &
pid2=$!
wait $pid0 $pid1 $pid2

# --- 7. Second eval pass: hawkVe (training already done) ---
echo "=== Step 7: hawkVe eval pass ==="
python run_sweep.py --configs=configs/champVe --gpu=0 --eval --bsps=hawkVe --skip-existing &
pid0=$!
python run_sweep.py --configs=configs/champVe --gpu=1 --eval --bsps=hawkVe --skip-existing &
pid1=$!
wait $pid0 $pid1

# --- 7b. Anchored I/J series: primary eval against tigerVe ---
echo "=== Step 7b: tigerVe eval (I/J tier, 3 GPUs) ==="
python run_sweep.py --configs=configs/champVe --gpu=0 --split=1/3 --eval --bsps=tigerVe --skip-existing --tier=I,J &
pid0=$!
python run_sweep.py --configs=configs/champVe --gpu=1 --split=2/3 --eval --bsps=tigerVe --skip-existing --tier=I,J &
pid1=$!
python run_sweep.py --configs=configs/champVe --gpu=2 --split=3/3 --eval --bsps=tigerVe --skip-existing --tier=I,J &
pid2=$!
wait $pid0 $pid1 $pid2

# --- 7c. Regression eval for I/J: gorillaVe + hawkVe ---
echo "=== Step 7c: I/J regression eval (gorillaVe + hawkVe) ==="
python run_sweep.py --configs=configs/champVe --gpu=0 --eval --bsps=gorillaVe --skip-existing --tier=I,J &
pid0=$!
python run_sweep.py --configs=configs/champVe --gpu=1 --eval --bsps=hawkVe    --skip-existing --tier=I,J &
pid1=$!
wait $pid0 $pid1

# --- 8. Read-out ---
echo "=== Step 8: Results read-out ==="
python scripts/registry_query.py top --bsps=gorillaVe --limit=20
python scripts/registry_query.py top --bsps=hawkVe    --limit=20
python scripts/registry_query.py top --bsps=tigerVe   --limit=20

python scripts/registry_query.py compare \
    F00-champVe-s42-batchtopk-k16-exp8-s4.fc1 \
    F00-champTa-s42-batchtopk-k16-exp8-s4.fc1 \
    --bsps=gorillaVe --bsps-b=gorillaTa

python scripts/registry_query.py compare \
    E05-champVe-s42-batchtopk-k32-exp8-s4.conv2 \
    E05-champTa-s42-batchtopk-k32-exp8-s4.conv2 \
    --bsps=gorillaVe --bsps-b=gorillaTa

echo "=== Pipeline complete ==="
