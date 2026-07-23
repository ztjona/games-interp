#!/usr/bin/env bash
# runners/champYb.sh -- champYb end-to-end on Deep Brain: data pipeline-to-parity
# -> LP baselines -> SAE sweep (FULL configs/champYb) -> eval -> Atlas export ->
# stage plan. Self-contained, committed (see CLAUDE.md "runners/").
#
# champYb = Yb_hotChamp(3) [hot lambda=1.0, seedB, E=10000]; QuartoCNNAutoregUnifiedS4Hot
# (train-only fc_hot head; hookable trunk identical to champS4/Ta/Ve). Game
# module quarto_s4_hot. Tags: champion Yb / position suffix yb / BSP suffix Yb.
# Detail: docs/diary/2026-06-18_champYb-onboarding.md.
#
# No tunable flags: it always uses ALL visible GPUs (auto-detected) for every
# parallelizable stage, and always trains the full configs/champYb sweep.
# Re-runnable: expensive steps skip when their output already exists.
#
# ASCII only in the source; PYTHONUTF8=1 is exported so any unicode the Python
# tools print does not crash under the host's cp1252 console.

set -uo pipefail
cd "$(dirname "$0")/.." || { echo "FATAL: cannot cd to repo root"; exit 1; }
export PYTHONUTF8=1

SLUG="champYb"
CHAMP_YB='models/quarto/20260614_2031-Yb_hotChamp(3)0612_HOT_1.0_seedB_E_10000.pt'
CHAMP_YB_RANDOM='models/quarto/20260612_1414-Yb_hotChamp(3)0612_HOT_1.0_seedB_E_0000.pt'
YB_MODEL_STEM='Yb_hotChamp'
POS_UNIQUE='data/quarto/positions-amalgam_yb_unique.pt'

# --- GPUs: detect and use all of them -----------------------------------------
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${NGPU:-0}" -lt 1 ]; then
    echo "FATAL: no GPU detected (nvidia-smi -L). This runner is for Deep Brain."
    exit 1
fi
echo "Using ${NGPU} GPU(s)."

# wait for a list of PIDs; fail if any did
wait_all() { local rc=0 p; for p in "$@"; do wait "$p" || rc=1; done; return $rc; }

# --- 0. Competence audit ------------------------------------------------------
if [ -f data/quarto/audit-champYb.json ]; then
    echo "=== Step 0: audit exists, skipping ==="
else
    echo "=== Step 0: Competence audit ==="
    python scripts/model_competence_audit.py \
        --model-config=configs/models/champYb.yaml \
        --num-positions=5000 --device=cuda \
        --output=data/quarto/audit-champYb.json
fi

# --- 1-2. Positions (4 modes across GPUs) + dedup -----------------------------
if [ -f "$POS_UNIQUE" ]; then
    echo "=== Step 1-2: $POS_UNIQUE exists, skipping position gen + dedup ==="
else
    echo "=== Step 1: Generate raw positions (4 modes x 10k, fanned across GPUs) ==="
    pids=(); i=0
    for mode in random_v_random model_v_random random_v_model model_v_model; do
        gpu=$(( i % NGPU ))
        echo "  -> $mode on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu python scripts/generate_positions.py \
            --game quarto_s4_hot --opponents "$mode" --model "$CHAMP_YB" \
            --num-games 10000 --seed 42 --device cuda --output-dir data/quarto_Yb &
        pids+=($!); i=$(( i + 1 ))
    done
    wait_all "${pids[@]}" || { echo "FATAL: a position-gen job failed"; exit 1; }

    echo "=== Step 2: Deduplicate positions ==="
    python scripts/deduplicate_positions.py \
        data/quarto_Yb/positions-random_v_random_raw.pt \
        "data/quarto_Yb/positions-model_v_random-${YB_MODEL_STEM}_raw.pt" \
        "data/quarto_Yb/positions-random_v_model-${YB_MODEL_STEM}_raw.pt" \
        "data/quarto_Yb/positions-model_v_model-${YB_MODEL_STEM}_raw.pt" \
        --output "$POS_UNIQUE"
fi

# --- 3. BSP labels: gorillaYb (164) + hawkYb (173) + tigerYb (36) -------------
echo "=== Step 3: Compute BSP labels ==="
python scripts/compute_bsp_labels.py "$POS_UNIQUE" --game quarto_s4_hot --name gorilla \
    --output data/quarto/bsp_labels-gorillaYb_164.pt \
    --schema-out data/quarto/bsp_schema-gorillaYb_164.json
python scripts/compute_bsp_labels.py "$POS_UNIQUE" --game quarto_s4_hot --name hawk \
    --output data/quarto/bsp_labels-hawkYb_173.pt \
    --schema-out data/quarto/bsp_schema-hawkYb_173.json
python scripts/compute_bsp_labels.py "$POS_UNIQUE" --game quarto_s4_hot --name tiger \
    --output data/quarto/bsp_labels-tigerYb_36.pt \
    --schema-out data/quarto/bsp_schema-tiger_36.json

# --- 4. Activations: 4 jobs (2 models x 2 hooks) fanned across GPUs ------------
echo "=== Step 4: Collect activations (fanned across GPUs) ==="
# job spec: "model|hook|outfile|extra"
act_jobs=(
    "${CHAMP_YB}|s4.fc1|data/quarto/s4.fc1_amalgam_yb_activations.pt|"
    "${CHAMP_YB_RANDOM}|s4.fc1|data/quarto/s4.fc1_amalgam_yb_random_activations.pt|"
    "${CHAMP_YB}|s4.conv2|data/quarto/s4.conv2_amalgam_yb_activations.pt|--flatten-position"
    "${CHAMP_YB_RANDOM}|s4.conv2|data/quarto/s4.conv2_amalgam_yb_random_activations.pt|--flatten-position"
)
pids=(); i=0
for spec in "${act_jobs[@]}"; do
    IFS='|' read -r mdl hook outf extra <<< "$spec"
    if [ -f "$outf" ]; then echo "  -> $outf exists, skipping"; continue; fi
    gpu=$(( i % NGPU ))
    echo "  -> $hook $(basename "$mdl") on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/collect_activations.py "$mdl" \
        --hook "$hook" --game quarto_s4_hot --positions-file "$POS_UNIQUE" \
        --output "$outf" --device cuda $extra &
    pids+=($!); i=$(( i + 1 ))
done
[ ${#pids[@]} -gt 0 ] && { wait_all "${pids[@]}" || { echo "FATAL: an activation job failed"; exit 1; }; }

# --- 5. Linear-probe baselines (2 hooks x 3 BSP sets) -------------------------
echo "=== Step 5: Linear-probe baselines ==="
for hook in s4.fc1 s4.conv2; do
    python scripts/linear_probe_baseline.py "data/quarto/${hook}_amalgam_yb_activations.pt" \
        data/quarto/bsp_labels-gorillaYb_164.pt data/quarto/bsp_schema-gorillaYb_164.json
    python scripts/linear_probe_baseline.py "data/quarto/${hook}_amalgam_yb_activations.pt" \
        data/quarto/bsp_labels-hawkYb_173.pt data/quarto/bsp_schema-hawkYb_173.json
    python scripts/linear_probe_baseline.py "data/quarto/${hook}_amalgam_yb_activations.pt" \
        data/quarto/bsp_labels-tigerYb_36.pt data/quarto/bsp_schema-tiger_36.json
done

# --- 6. SAE sweep: FULL configs/champYb, sharded across all GPUs ---------------
echo "=== Step 6: SAE sweep (configs/champYb) across ${NGPU} GPU(s) ==="
pids=()
for (( g=0; g<NGPU; g++ )); do
    python run_sweep.py --configs=configs/champYb --gpu="$g" \
        --split="$(( g + 1 ))/${NGPU}" --eval --bsps=gorillaYb --skip-existing &
    pids+=($!)
done
wait_all "${pids[@]}" || { echo "FATAL: a sweep shard failed"; exit 1; }

# --- 7. Cross-BSP eval (hawkYb, tigerYb), fanned across GPUs, throttled --------
echo "=== Step 7: Cross-BSP eval (hawkYb, tigerYb) ==="
i=0
for ck in saes/quarto/*champYb*.pt; do
    [ -e "$ck" ] || continue
    for bsps in hawkYb tigerYb; do
        gpu=$(( i % NGPU ))
        CUDA_VISIBLE_DEVICES=$gpu python sae_eval.py evaluate "$ck" --bsps="$bsps" || true &
        i=$(( i + 1 ))
        (( i % NGPU == 0 )) && wait
    done
done
wait
python scripts/registry_query.py top --bsps=gorillaYb --limit=20 || true

# --- 8. Atlas export (updates the tracked shipped_saes.jsonl manifest) ---------
echo "=== Step 8: Atlas viz export ==="
python scripts/export_viz_data.py --game quarto --bsps gorillaYb

# --- 9. Stage plan (stage_champYb.md at repo root) ----------------------------
echo "=== Step 9: Stage plan ==="
python scripts/emit_stage.py --slug="$SLUG" \
    --shipped-jsonl saes/quarto/shipped_saes.jsonl \
    data/quarto/audit-champYb.json \
    data/quarto/bsp_schema-gorillaYb_164.json \
    data/quarto/bsp_schema-hawkYb_173.json \
    data/quarto/bsp_schema-tiger_36.json \
    data/quarto/linear_probe_*_yb_activations_results.json \
    saes/quarto/training_registry.json \
    saes/quarto/eval_registry.json \
    saes/quarto/shipped_saes.jsonl \
    data/quarto/bsp_labels-gorillaYb_164.pt \
    data/quarto/bsp_labels-hawkYb_173.pt \
    data/quarto/bsp_labels-tigerYb_36.pt \
    data/quarto/s4.fc1_amalgam_yb_activations.pt \
    data/quarto/s4.conv2_amalgam_yb_activations.pt

echo
echo "=== champYb runner complete; commit plan in stage_${SLUG}.md ==="
