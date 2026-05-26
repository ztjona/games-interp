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
# =============================================================================

CHAMP_VE='models/quarto/20260524_1904-Ve_oracleAblation(4)0522_DISABLE_NEVER_10k_E_10000.pt'
CHAMP_VE_RANDOM='models/quarto/20260522_1832-Ve_oracleAblation(4)0522_DISABLE_NEVER_10k_E_0000.pt'

# --- 0. Competence audit (sanity: model plays Quarto, random net doesn't) ---
python scripts/model_competence_audit.py \
    --model-config=configs/models/champVe.yaml \
    --num-positions=5000 --device=cuda \
    --output=data/quarto/audit-champVe.json

# --- 1. Generate raw positions (4 opponent modes, seed=42) ---
for mode in random_v_random model_v_random random_v_model model_v_model; do
    python scripts/generate_positions.py --game quarto_s4 \
        --opponents "$mode" --model "$CHAMP_VE" \
        --num-games 10000 --seed 42 --device cuda
done

# --- 2. Aggregate + deduplicate across all four files (NOT per-file dedup!) ---
# Output uses the champVe tag 've' (lowercase, metals-theme slot).
# NOTE: re-list data/quarto/positions-*_raw.pt after step 1 to confirm the
# exact model-stem filenames before launching this command.
python scripts/deduplicate_positions.py \
    data/quarto/positions-random_v_random_raw.pt \
    data/quarto/positions-model_v_random-*Ve_oracleAblation*_raw.pt \
    data/quarto/positions-random_v_model-*Ve_oracleAblation*_raw.pt \
    data/quarto/positions-model_v_model-*Ve_oracleAblation*_raw.pt \
    --output data/quarto/positions-amalgam_ve_unique.pt

# --- 3. BSP labels: gorillaVe (164) + hawkVe (173) ---
# --name picks the category filter from quarto_s4.BSP_SETS; --output/--schema-out
# carry the champion tag because compute_bsp_labels does not auto-tag yet.
python scripts/compute_bsp_labels.py data/quarto/positions-amalgam_ve_unique.pt \
    --game quarto_s4 --name gorilla \
    --output data/quarto/bsp_labels-gorillaVe_164.pt \
    --schema-out data/quarto/bsp_schema-gorillaVe_164.json

python scripts/compute_bsp_labels.py data/quarto/positions-amalgam_ve_unique.pt \
    --game quarto_s4 --name hawk \
    --output data/quarto/bsp_labels-hawkVe_173.pt \
    --schema-out data/quarto/bsp_schema-hawkVe_173.json

# tigerVe (36) -- agent-relative threat set, needed for anchored I/J series.
# Schema is basis-only (bsp_schema-tiger_36.json), shared across champions.
python scripts/compute_bsp_labels.py data/quarto/positions-amalgam_ve_unique.pt \
    --game quarto_s4 --name tiger \
    --output data/quarto/bsp_labels-tigerVe_36.pt \
    --schema-out data/quarto/bsp_schema-tiger_36.json

# --- 4. Collect activations: trained + random, s4.fc1 + s4.conv2 ---
# s4.conv2 needs --flatten-position (4x4 spatial flattened into batch dim).
python scripts/collect_activations.py "$CHAMP_VE" --hook s4.fc1 --game quarto_s4 \
    --positions-file data/quarto/positions-amalgam_ve_unique.pt \
    --output data/quarto/s4.fc1_amalgam_ve_activations.pt --device cuda

python scripts/collect_activations.py "$CHAMP_VE_RANDOM" --hook s4.fc1 --game quarto_s4 \
    --positions-file data/quarto/positions-amalgam_ve_unique.pt \
    --output data/quarto/s4.fc1_amalgam_ve_random_activations.pt --device cuda

python scripts/collect_activations.py "$CHAMP_VE" --hook s4.conv2 --game quarto_s4 \
    --positions-file data/quarto/positions-amalgam_ve_unique.pt \
    --output data/quarto/s4.conv2_amalgam_ve_activations.pt --device cuda --flatten-position

python scripts/collect_activations.py "$CHAMP_VE_RANDOM" --hook s4.conv2 --game quarto_s4 \
    --positions-file data/quarto/positions-amalgam_ve_unique.pt \
    --output data/quarto/s4.conv2_amalgam_ve_random_activations.pt --device cuda --flatten-position

# --- 5. Linear-probe baseline (LP upper bound on the new activations) ---
# tigerVe schema is basis-only (bsp_schema-tiger_36.json) per the existing
# convention; gorilla/hawk schemas are champion-tagged.
for hook in s4.fc1 s4.conv2; do
    python scripts/linear_probe_baseline.py \
        data/quarto/${hook}_amalgam_ve_activations.pt \
        data/quarto/bsp_labels-gorillaVe_164.pt \
        data/quarto/bsp_schema-gorillaVe_164.json
    python scripts/linear_probe_baseline.py \
        data/quarto/${hook}_amalgam_ve_activations.pt \
        data/quarto/bsp_labels-hawkVe_173.pt \
        data/quarto/bsp_schema-hawkVe_173.json
    python scripts/linear_probe_baseline.py \
        data/quarto/${hook}_amalgam_ve_activations.pt \
        data/quarto/bsp_labels-tigerVe_36.pt \
        data/quarto/bsp_schema-tiger_36.json
done

# --- 6. SAE sweep: 19 configs (C/E/F/H), s42 only, 3-way GPU split ---
# Pre-flight (optional, takes ~1 min):
# python validate_sweep.py --configs=configs/champVe --smoke-test
python run_sweep.py --configs=configs/champVe --gpu=0 --split=1/3 --eval --bsps=gorillaVe --skip-existing &
python run_sweep.py --configs=configs/champVe --gpu=1 --split=2/3 --eval --bsps=gorillaVe --skip-existing &
python run_sweep.py --configs=configs/champVe --gpu=2 --split=3/3 --eval --bsps=gorillaVe --skip-existing
wait

# --- 7. Second eval pass: hawkVe (training already done -> --skip-existing only evals) ---
python run_sweep.py --configs=configs/champVe --gpu=0 --eval --bsps=hawkVe --skip-existing &
python run_sweep.py --configs=configs/champVe --gpu=1 --eval --bsps=hawkVe --skip-existing
wait

# --- 7b. Anchored I/J series: primary eval against tigerVe (24 configs, 3 GPUs) ---
python run_sweep.py --configs=configs/champVe --gpu=0 --split=1/3 --eval --bsps=tigerVe --skip-existing --tier=I,J &
python run_sweep.py --configs=configs/champVe --gpu=1 --split=2/3 --eval --bsps=tigerVe --skip-existing --tier=I,J &
python run_sweep.py --configs=configs/champVe --gpu=2 --split=3/3 --eval --bsps=tigerVe --skip-existing --tier=I,J
wait

# --- 7c. Regression eval for I/J: gorillaVe + hawkVe (training already done) ---
python run_sweep.py --configs=configs/champVe --gpu=0 --eval --bsps=gorillaVe --skip-existing --tier=I,J &
python run_sweep.py --configs=configs/champVe --gpu=1 --eval --bsps=hawkVe    --skip-existing --tier=I,J
wait

# --- 8. Read-out: per-BSP-set tops + champVe-vs-champTa twin compares ---
python scripts/registry_query.py top --bsps=gorillaVe --limit=20
python scripts/registry_query.py top --bsps=hawkVe    --limit=20
python scripts/registry_query.py top --bsps=tigerVe   --limit=20

# Twin compares (Ve vs Ta) -- same SAE config, different champion.
python scripts/registry_query.py compare \
    F00-champVe-s42-batchtopk-k16-exp8-s4.fc1 \
    F00-champTa-s42-batchtopk-k16-exp8-s4.fc1 \
    --bsps=gorillaVe

python scripts/registry_query.py compare \
    E05-champVe-s42-batchtopk-k32-exp8-s4.conv2 \
    E05-champTa-s42-batchtopk-k32-exp8-s4.conv2 \
    --bsps=gorillaVe
