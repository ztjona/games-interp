# Anakin Sweep — Comprehensive SAE Architecture Comparison

**Created:** 2026-04-01  
**Design:** 33 configs across 2 hooks, 6 architectures, multiple sparsity levels and expansion factors  
**Target:** 4-day unattended sweep on 3 GPUs (RTX 4000 + 2× P4000)

## Sweep Structure

### fc1 (d_input=128) — 23 configs

| Tier | Focus | Configs | Key Variables |
|------|-------|---------|---------------|
| **A** | Architecture comparison | 6 | All 6 archs at exp=8 |
| **B** | TopK k-sweep | 3 | k ∈ {16, 64, 128} at exp=8 |
| **C** | BatchTopK k-sweep | 2 | k ∈ {16, 64} at exp=8 |
| **D** | JumpReLU l0-sweep | 2 | l0 ∈ {32, 128} at exp=8 |
| **E** | Expansion sweep (topk-k32) | 2 | exp ∈ {4, 16} |
| **F** | Best archs at exp=16 | 4 | batchtopk, gated, jumprelu, p-anneal |
| **G** | Low-penalty variants | 2 | vanilla l1=1e-4, gated l1=5e-4 |
| **H** | Seed stability | 2 | seeds 43, 44 for topk-k32-exp8 |

### conv2 (d_input=512) — 10 configs

| Tier | Focus | Configs | Key Variables |
|------|-------|---------|---------------|
| **I** | Architecture comparison | 6 | All 6 archs at exp=4 (d_dict=2048) |
| **J** | TopK k-sweep | 2 | k ∈ {32, 128} at exp=4 |
| **K** | Expansion sweep (topk-k64) | 2 | exp ∈ {2, 8} |

## Design Rationale

**fc1 as primary hook:** 128-dim shared bottleneck before dual heads. Linear probes show cell concepts are accessible (F1≈0.6–1.0). SAE baseline coverage = 0.30.

**conv2 as comparison:** 512-dim spatial features. Existing baseline shows coverage=0.34 with 90% dead features — undertrained at 5000 steps. The sweep uses 25000 steps for fair comparison.

**Sparsity matching across hooks:** conv2 configs use k=64 at exp=4 (3.12% sparsity), matching fc1's k=32 at exp=8 (3.12%).

**conv2 data note:** All conv2 configs use `conv2_512_amalgam_activations.pt` (275K × 512, full-spatial), NOT the per-cell `conv2_amalgam_activations.pt` (4.4M × 32).

## Execution Commands

### Pre-flight validation
```bash
python validate_sweep.py --smoke-test
```

### Launch sweep (3 separate terminals)
```bash
# Terminal 1: GPU 0 (RTX 4000)
python run_sweep.py --configs=configs/anakin --gpu=0 --split=1/3 --eval --skip-existing

# Terminal 2: GPU 1 (P4000)
python run_sweep.py --configs=configs/anakin --gpu=1 --split=2/3 --eval --skip-existing

# Terminal 3: GPU 2 (P4000)
python run_sweep.py --configs=configs/anakin --gpu=2 --split=3/3 --eval --skip-existing
```

### Subset execution (optional)
```bash
# Only architecture comparison tiers
python run_sweep.py --configs=configs/anakin --gpu=0 --tier=A,I --eval

# Only fc1 configs
python run_sweep.py --configs=configs/anakin --gpu=0 --tier=A,B,C,D,E,F,G,H --eval
```

### Monitor progress
```bash
# Check GPU utilization
nvidia-smi -l 5

# Tail the sweep log
tail -f logs/sweep_gpu0_*.log

# Check eval registry for completed results
python sae_eval.py history
```

## Expected Outputs

Each config produces:
- `saes/quarto/anakin-{suffix}-{hook}.pt` — SAE checkpoint
- `saes/quarto/anakin-{suffix}-{hook}_metrics.jsonl` — training metrics
- Entry in `saes/quarto/eval_registry.json` (if --eval)
- Entry in `saes/quarto/training_registry.json`

## Config Regeneration

If parameters need adjustment:
```bash
# Edit scripts/generate_anakin_configs.py, then:
rm configs/anakin/*.yaml
python scripts/generate_anakin_configs.py
```
