# Arnold Experiment Sweep

**Purpose:** Systematic architecture × hyperparameter sweep for Quarto fc1 SAEs.  
**Delegate to:** Autonomous coding agent (e.g., Claude agent mode).  
**Prerequisites:** All bug fixes from Session 2 merged (TopK aux loss, BatchTopK/JumpReLU kwargs, new metrics).

---

## Context

Two pilot SAEs were trained on Quarto fc1 (128-dim bottleneck, expansion ×8 → 1024 features):

| Pilot | Issue |
|-------|-------|
| `pilot-vanilla-exp8-fc1` | L0 = 574 / 1024, near-identity collapse — l1_weight=0.001 far too weak |
| `pilot-topk-k16-exp8-fc1` | **INVALID** — aux loss had zero-gradient bug, 88.67% dead features |

Both pilots used 25K steps (overkill — convergence was at ~5K steps for 128-dim input).

**Goal:** Find the best (architecture, sparsity) combination for fc1 before scaling expansion factor.

---

## Experiment Matrix

10 runs, organized in 3 priority tiers. Run Tier 1 first. Compare results. Then decide whether Tier 2 and 3 are necessary.

### Tier 1 — Core Sweep (run first)

| # | Config File | Architecture | Key Hyperparams | Rationale |
|---|------------|-------------|-----------------|-----------|
| 1 | `arnold-vanilla-l1_005.yaml` | vanilla | l1=0.005 | 5× stronger than failed pilot |
| 2 | `arnold-vanilla-l1_01.yaml` | vanilla | l1=0.01 | 10× stronger than failed pilot |
| 3 | `arnold-topk-k32.yaml` | topk | k=32 | Fixed aux loss; k=32 balances sparsity/utility |
| 4 | `arnold-topk-k64.yaml` | topk | k=64 | Higher capacity; still sparse (6.25% active) |

### Tier 2 — Architecture Comparison (run after Tier 1 analysis)

| # | Config File | Architecture | Key Hyperparams | Rationale |
|---|------------|-------------|-----------------|-----------|
| 5 | `arnold-gated-l1_005.yaml` | gated | l1=0.005 | Anti-shrinkage; matches vanilla l1 for comparison |
| 6 | `arnold-gated-l1_01.yaml` | gated | l1=0.01 | Anti-shrinkage; stronger sparsity |
| 7 | `arnold-batchtopk-k32.yaml` | batchtopk | k=32 | Variable sparsity per position (board complexity varies) |

### Tier 3 — Adaptive Sparsity (run if Tier 1-2 show promise)

| # | Config File | Architecture | Key Hyperparams | Rationale |
|---|------------|-------------|-----------------|-----------|
| 8 | `arnold-jumprelu-t32.yaml` | jumprelu | l0_target=32 | Learned threshold; targets L0≈32 |
| 9 | `arnold-jumprelu-t64.yaml` | jumprelu | l0_target=64 | Learned threshold; targets L0≈64 |
| 10 | `arnold-vanilla-l1_05.yaml` | vanilla | l1=0.05 | Strong sparsity regime |

---

## Shared Hyperparameters

All experiments use:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `hook` | fc1 | Shared bottleneck, 128-dim, best SAE target |
| `expansion` | 8 | 1024 features; baseline before scaling |
| `batch_size` | 4096 | Covers ~1.7% of dataset per step |
| `num_batches` | 5000 | Sufficient for convergence (pilot converged at ~5K) |
| `lr` | 3e-4 | Standard for SAEs (Bricken et al. 2023) |
| `seed` | 42 | Reproducibility |
| `log_every` | 100 | 50 log points (enough for training curves) |
| `data` | `data/quarto/fc1_amalgam_activations.pt` | 240,845 positions × 128-dim |

---

## How to Run

### Single Experiment

```bash
cd /path/to/games-interp
python sae_train.py --config=configs/arnold/arnold-vanilla-l1_005.yaml
```

### Full Tier (sequential)

```bash
# Tier 1
for cfg in configs/arnold/arnold-vanilla-l1_005.yaml \
           configs/arnold/arnold-vanilla-l1_01.yaml \
           configs/arnold/arnold-topk-k32.yaml \
           configs/arnold/arnold-topk-k64.yaml; do
    python sae_train.py --config="$cfg"
done
```

### Parallel (separate terminals)

```bash
# Terminal 1
python sae_train.py --config=configs/arnold/arnold-vanilla-l1_005.yaml
# Terminal 2
python sae_train.py --config=configs/arnold/arnold-topk-k32.yaml
```

**Expected runtime:** ~2-5 min per experiment on CPU (5K steps, 128-dim, batch 4096).

---

## Output Files

Each experiment produces:
- **Checkpoint:** `saes/quarto/arnold-{arch}-{suffix}-fc1.pt`
- **Metrics log:** `saes/quarto/arnold-{arch}-{suffix}-fc1_metrics.jsonl`
- **Registry entry:** auto-appended to `saes/quarto/training_registry.json`

Naming follows the existing convention: `{experiment}-{arch}[-k{K}]-exp{E}-{hook}`

---

## Success Criteria

After training, compare all experiments on these metrics (from final training step):

| Metric | Good Range | Red Flag |
|--------|-----------|----------|
| `fvu` | < 0.05 | > 0.15 (not reconstructing well) |
| `l0` | 20–80 | > 200 (not sparse) or < 5 (too sparse, losing info) |
| `l0_std` | > 0 for non-TopK | = 0 for vanilla/gated (implies all positions identical) |
| `dead_features_pct` | < 5% | > 20% (features wasted) |
| `median_feat_freq` | > 0.001 | = 0 (most features unused) |

### Decision Tree After Tier 1

```
IF vanilla l1=0.01 has (FVU < 0.05) AND (20 < L0 < 80) AND (dead < 5%):
    → Vanilla is a strong baseline. Proceed to Tier 2 to test gated/batchtopk.

IF topk k=32 has (FVU < 0.05) AND (dead < 5%):
    → TopK aux loss fix works. Compare vs vanilla at similar L0.

IF topk k=32 still has (dead > 20%):
    → Aux loss fix may be insufficient. Try larger aux_loss_weight (0.1).
    → Flag for manual investigation.

IF all Tier 1 results have (FVU > 0.10):
    → Expansion 8 may be too small. Consider running exp=16 variants.
    → Check activation data quality (distribution, normalization).
```

---

## Evaluation (After Training)

### Layer 1 Evaluation (BSP Coverage)

For each trained checkpoint, run the board-bench evaluation:

```bash
# Using the skill evaluation script
python ~/.agents/skills/sae-board-bench/scripts/sae_eval.py evaluate \
    saes/quarto/arnold-vanilla-exp8-fc1.pt \
    --model models/quarto/20260227_1103-Aa_replay\(2\)0226_NUM_EPOCHs_BUFFER_8_E_5000.pt \
    --game quarto \
    --hook fc1 \
    --data data/quarto/fc1_amalgam_activations.pt \
    --tag "arnold-vanilla-l1_005"
```

**BSP label files:**
- Full set: `data/quarto/bsp_labels-gorilla_164.pt` (164 BSPs)
- Cell-only: `data/quarto/bsp_labels-fox_87.pt` (87 BSPs)

**BSP schema files:**
- `data/quarto/bsp_schema-gorilla_164.json`
- `data/quarto/bsp_schema-fox_87.json`

### Comparison

After evaluating all Tier 1 checkpoints:

```bash
python ~/.agents/skills/sae-board-bench/scripts/sae_eval.py history --game quarto
```

Then compare best two:

```bash
python ~/.agents/skills/sae-board-bench/scripts/sae_eval.py compare <exp_id_1> <exp_id_2>
```

---

## Follow-Up: Expansion Factor Sweep

After identifying the best 2 architectures from the arnold sweep, scale expansion factor:

| Expansion | Dict Size | Purpose |
|-----------|-----------|---------|
| 4 | 512 | Undercomplete — tests if fewer features suffice |
| 8 | 1024 | Baseline (this sweep) |
| 16 | 2048 | Overcomplete — more capacity for fine-grained features |
| 32 | 4096 | Large overcomplete — diminishing returns test |

Create configs like `arnold-{best_arch}-exp16.yaml` modifying only `expansion`.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'docopt'` | `pip install docopt` |
| CUDA OOM | Reduce `batch_size` to 2048 |
| Training stuck (loss not decreasing) | Check learning rate; verify activation data is normalized |
| All features dead in TopK | Increase `aux_loss_weight` from 0.01 to 0.1 |
| Vanilla L0 > 500 | l1_weight too low; use 0.01 or higher |
| Terminal output missing | Use background mode + poll with `get_terminal_output` |
| `KeyError` in constructor | Check `sae_train.py:get_arch_kwargs` matches architecture `__init__` |

---

## Notes for Autonomous Agent

1. **Run experiments sequentially within a tier** — avoid parallel on single GPU to prevent OOM.
2. **Check registry after each run** — `cat saes/quarto/training_registry.json | python -m json.tool | tail -30` to verify the entry was saved.
3. **Save a summary** after completing each tier — write metrics table to `configs/arnold/results.md`.
4. **Don't proceed to Tier 2 without analyzing Tier 1** — use the Decision Tree above.
5. **Mark any failed runs** with `"invalid": true` in the registry (same pattern as pilot-topk).
6. **Report final metrics as JSON** in the terminal output for easy parsing.
