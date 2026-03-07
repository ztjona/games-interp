# Arnold Beta Experiment Sweep

**Purpose:** Re-run architectures that were invalidated by implementation bugs discovered during the arnold sweep analysis (2026-03-06). Also includes one new TopK geometry test.

**Prerequisite:** Bug fixes merged to `lib/sae/architectures.py`:
- **GatedSAE** — added via-gate auxiliary loss (routes reconstruction gradients to W_gate)
- **JumpReLUSAE** — replaced `(h>0).float()` L0 penalty with sigmoid kernel estimator (differentiable w.r.t. theta)

---

## What Changed vs Arnold

| Run | Arnold Result | Root Cause | Fix |
|-----|--------------|------------|-----|
| Gated (all variants) | FVU=1.0, L0=0, dead=99% | W_gate got zero recon gradient (step-function gate) | Via-gate aux loss: `ReLU(gate_pre) @ W_dec.detach()` |
| JumpReLU t32/t64 | L0≈600 regardless of target | `(h>0).float()` has ∂/∂θ=0 | Sigmoid STE: `σ((z−θ)/ε)` for L0 penalty |

Plus one new geometry test:
- **TopK k=128** — tests the hypothesis that high dead-feature% is a geometry mismatch (k=32,64 active out of 1024 is too sparse; k=128 = 12.5% matches real L0 ratio better)

---

## Experiment Matrix

| # | Config | Architecture | Key Hyperparams | vs. Arnold |
|---|--------|-------------|-----------------|------------|
| 1 | `arnold_beta-gated-l1_005.yaml` | gated | l1=0.005 | Same HP, fixed implementation |
| 2 | `arnold_beta-gated-l1_01.yaml` | gated | l1=0.01 | Same HP, fixed implementation |
| 3 | `arnold_beta-jumprelu-t32.yaml` | jumprelu | l0_target=32 | Same HP, fixed implementation |
| 4 | `arnold_beta-jumprelu-t64.yaml` | jumprelu | l0_target=64 | Same HP, fixed implementation |
| 5 | `arnold_beta-topk-k128.yaml` | topk | k=128 | New: geometry test (12.5% active) |

---

## Success Criteria

| Architecture | Expected if fix works |
|---|---|
| Gated | FVU < 0.01, L0 > 0, dead < 20% |
| JumpReLU t32 | L0 converges to ≈ 32 (±15) |
| JumpReLU t64 | L0 converges to ≈ 64 (±20) |
| TopK k=128 | dead < 40% (vs 61% at k=32, 72% at k=64) |

---

## How to Run

```bash
# Full sweep
for cfg in configs/arnold_beta/arnold_beta-gated-l1_005.yaml \
           configs/arnold_beta/arnold_beta-gated-l1_01.yaml \
           configs/arnold_beta/arnold_beta-jumprelu-t32.yaml \
           configs/arnold_beta/arnold_beta-jumprelu-t64.yaml \
           configs/arnold_beta/arnold_beta-topk-k128.yaml; do
    python sae_train.py --config="$cfg"
done
```
