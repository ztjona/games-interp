# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PhD research on mechanistic interpretability of board-game neural networks. The current focus is **Quarto** (a CNN DQN); Othello and Tic-tac-toe are planned but not implemented. The core technique is training Sparse Autoencoders (SAEs) on hooked activations and evaluating them against hand-defined ground-truth concepts called **BSPs** (Board State Properties).

Source-of-truth research docs (read these before changing experimental scope):
- `RESEARCH-STATUS.md` — current phase, hypotheses, winners, open problems, active plan
- `Quarto-specifications.md` — game, model architecture, hookable layers, dataset catalog, naming conventions
- `BSP-schema-summary.md` — full BSP schema for gorilla and hawk sets

## Common commands

Setup:
```bash
pip install -r requirements.txt   # torch, numpy, tqdm, quartopy, pyyaml, docopt, plotly, dash, pytest
```

Tests (CPU-only, full suite is fast):
```bash
pytest                                              # all tests
pytest tests/test_sae.py                            # SAE architecture/training tests
pytest tests/test_bsp_logic.py                      # BSP definitions
pytest tests/test_sae_eval.py                       # eval pipeline
pytest tests/test_sae.py::test_topk_sparsity_exact  # single test by node-id
pytest -k "topk and not jumprelu"                   # filter by name
```

Train one SAE (preferred path is YAML configs under `configs/`):
```bash
python sae_train.py --config=configs/anakin/fc1-topk-k32-exp8.yaml
python sae_train.py <experiment> <architecture> [options]   # CLI form, see docstring
```

Evaluate one SAE (data, BSP labels, and BSP schema auto-resolve from checkpoint metadata):
```bash
python sae_eval.py evaluate saes/quarto/<run_id>.pt          # uses gorilla BSPs by default
python sae_eval.py evaluate <ckpt> --bsps=hawk --force       # re-eval against a different BSP set
python sae_eval.py history --arch jumprelu                   # browse past evals (wide table)
python sae_eval.py compare <run_id_1> <run_id_2>
```

Query the evaluation registry from the command line (preferred over grepping
`saes/quarto/eval_registry.json` directly — LLM/agent-friendly, deterministic):
```bash
python scripts/registry_query.py top --bsps=gorilla --limit=10           # top by F1-lift
python scripts/registry_query.py top --bsps=hawk --hook=conv2 --exclude=random
python scripts/registry_query.py category <run_id> --bsps=gorilla        # per-category
python scripts/registry_query.py compare <run_a> <run_b> --bsps=gorilla  # side-by-side
python scripts/registry_query.py bsps                                    # which BSP sets exist
python scripts/registry_query.py top --bsps=gorilla --json | jq ...      # composable
```

Audit model competence (game-behavior tests, independent of any SAE — gates
interpretability claims):
```bash
python scripts/model_competence_audit.py --num-positions=5000 --device=cuda
python scripts/model_competence_audit.py --model=<newer.pt> --random-model=none --output=data/quarto/audit-<tag>.json
```

Multi-config / multi-GPU sweep:
```bash
python validate_sweep.py --smoke-test                                          # pre-flight check
python run_sweep.py --configs=configs/anakin --gpu=0 --split=1/3 --eval --skip-existing
```

Live training monitor (Dash, port 8050):
```bash
python scripts/plot_training.py 'saes/quarto/*_metrics.jsonl' --live
```

Visualization data export (for the sibling `boardSAE-atlas/` static site):
```bash
python scripts/export_viz_data.py --game quarto                  # JSON bundles → public/data/quarto/
python scripts/export_onnx.py --game quarto                      # ONNX encoders → public/models/quarto/
```

Linear-probe baseline (upper bound for any SAE on the same activations):
```bash
python scripts/linear_probe_baseline.py \
    data/quarto/fc1_amalgam_activations.pt \
    data/quarto/bsp_labels-gorilla_164.pt \
    data/quarto/bsp_schema-gorilla_164.json
```

Data pipeline (rarely needed — datasets already exist on disk).
Models: trained=`20260227_1103-Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8_E_5000.pt`, random=`20260226_1420-…_E_0000.pt` (both in `models/quarto/`).
All four raw files were generated with `--seed 42` on 2026-03-27 — the exact seed is stored
in the provenance dict inside each `.pt` file so it can always be verified.
```bash
# Step 1 — Generate raw positions (once; reused for all activation hooks)
MODEL=models/quarto/20260227_1103-Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8_E_5000.pt
for mode in random_v_random model_v_random random_v_model model_v_model; do
    python scripts/generate_positions.py --game quarto --opponents $mode --model $MODEL --num-games 10000 --seed 42
done

# Step 2 — Aggregate all four raw files, then deduplicate across them (NOT per-file!)
# This is both the aggregation and deduplication step in one command.
python scripts/deduplicate_positions.py \
    data/quarto/positions-random_v_random_raw.pt \
    data/quarto/positions-model_v_random-Aa_replay_raw.pt \
    data/quarto/positions-random_v_model-Aa_replay_raw.pt \
    data/quarto/positions-model_v_model-Aa_replay_raw.pt \
    --output data/quarto/positions-amalgam_unique.pt

# Step 3 — Collect activations for primary hook (trained model)
# Use --flatten-position for conv layers (B,C,H,W) -> (B, C*H*W); not needed for fc1
python scripts/collect_activations.py $MODEL --hook conv2 --game quarto \
    --positions-file data/quarto/positions-amalgam_unique.pt \
    --output data/quarto/conv2_512_amalgam_activations.pt --device cuda --flatten-position

# Step 3b — Collect activations for random-model controls (G-series)
RANDOM_MODEL=models/quarto/20260226_1420-Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8_E_0000.pt
python scripts/collect_activations.py $RANDOM_MODEL --hook conv2 --game quarto \
    --positions-file data/quarto/positions-amalgam_unique.pt \
    --output data/quarto/conv2_512_amalgam_random_activations.pt --device cuda --flatten-position

# Step 4 — Compute BSP labels (position-level; same file serves all hooks)
python scripts/compute_bsp_labels.py data/quarto/positions-amalgam_unique.pt \
    --game quarto --name gorilla
python scripts/compute_bsp_labels.py data/quarto/positions-amalgam_unique.pt \
    --game quarto --name hawk
```

## Architecture

The codebase has three layers: a **library** (`lib/sae/`), a small set of **top-level CLIs** that compose it, and **per-game scripts** (`scripts/games/<game>.py`) that supply the game-specific glue.

### `lib/sae/` — game-agnostic SAE library

- `architectures.py` — `BaseSAE` plus six variants (`VanillaSAE`, `TopKSAE`, `BatchTopKSAE`, `GatedSAE`, `JumpReLUSAE`, `PAnnealingSAE`), all registered in `ARCHITECTURES`. They share weights `W_enc, b_enc, W_dec, b_dec` and only differ in `encode()` and `compute_loss()`. Decoder columns are unit-normalized after every optimizer step.
- `train.py` — `train_sae`, `iter_batches`, `load_activation_data`, checkpoint I/O, metrics (`fvu`, `l0`, `l0_std`, `median_feat_freq`, `dead_features_pct`).
- `eval.py` — feature ↔ BSP matching (precision/recall/F1 over `(d_dict, num_bsps)`), per-category coverage, board reconstruction. **All eval functions operate on pre-computed tensors and are game-agnostic** — game knowledge enters only via `bsp_labels` and `bsp_schema`.
- `hooks.py` — `load_game_model`, `ActivationStore`, hook utilities; works on any nn.Module via named modules.
- `registry.py` — appends rows to `saes/<game>/training_registry.json`.

### Top-level CLIs

- `sae_train.py` — accepts a YAML config (preferred) **or** positional CLI args; auto-resolves `data/<game>/<hook>_amalgam_activations.pt` if `data:` is not given. Outputs go to `saes/<game>/{experiment}-{arch}-{hook}-*.pt` plus a sibling `*_metrics.jsonl`. The trainer **appends architecture + hook to the checkpoint stem itself**, so the `experiment:` field in YAML must stay short — do not duplicate `arch` or `hook` into it.
- `sae_eval.py` — three subcommands (`evaluate`, `compare`, `history`). Auto-resolves data, BSP labels, and BSP schema from checkpoint metadata. Caches `{run_id}_h.pt` (full `(N × d_dict)` activations) and `{run_id}_matching-{animal}.pt` (per-feature P/R/F1 vs every BSP) under `saes/<game>/`. Re-eval requires `--force`.
- `run_sweep.py` — orchestrator that calls `sae_train.py` (then optionally `sae_eval.py`) per YAML in a directory. Uses `--split=N/M` for round-robin partitioning across multiple GPUs run in separate terminals; sets `CUDA_VISIBLE_DEVICES` per process.
- `validate_sweep.py` — pre-flight checks (deps, GPU memory, fixture data, optional 10-step smoke test per architecture).

### Per-game module: `scripts/games/quarto.py`

Single source of truth for: model loading (`QuartoCNN.from_file`), the four opponent modes (`random_v_random`, `model_v_random`, `random_v_model`, `model_v_model`), `get_all_bsp_definitions()` (337 BSPs total), and `compute_bsp_vector()`. New games would add a sibling `scripts/games/<game>.py` and register in `scripts/games/__init__.py`.

### Data layout

- `data/<game>/positions-<metal>_unique.pt` — deduplicated position tensors (`{boards, pieces, metadata, provenance}`)
- `data/<game>/<hook>_<metal>_activations.pt` — `(N, d_act)` activation tensor matched to a positions file
- `data/<game>/bsp_labels-<animal>_<count>.pt` — `(N, num_bsps)` binary tensor
- `data/<game>/bsp_schema-<animal>_<count>.json` — BSP definitions
- `saes/<game>/{run_id}.pt` + `{run_id}_metrics.jsonl` + `training_registry.json` + `eval_registry.json`
- `eval_registry.json` keys use the format `run_id:bsp_set` (e.g. `A01-random-control-s42-batchtopk-k16-exp8-fc1:gorilla`) so the same checkpoint can be evaluated against multiple BSP sets without collision. Legacy entries with plain `run_id` keys are still read correctly.
- `saes/*.pt` checkpoints are gitignored by default; force-add key ones with `git add -f`. `*.jsonl` metrics and `*_registry.json` files are tracked.

### Quarto model — hookable layers

`models/quarto/CNN_uncoupled.py` defines `QuartoCNN`. Layer name → activation shape:
- `conv1` → (B, 16, 4, 4)
- `conv2` → (B, 32, 4, 4) — **preferred hook for threat-related work** (Phase 1F result: threat info is linearly accessible here, mostly lost by fc1)
- `fc1` → (B, 128) — bottleneck before dual heads, still the default hook in legacy configs
- `fc2_board` (B, 16), `fc2_piece` (B, 16) — Q-value heads

When working with conv hooks, activations may be flattened (`B, C, H, W` → `B*H*W, C`) via `--flatten` in `collect_activations.py`.

## Domain conventions

These naming rules are project-specific and are required for files to flow through the auto-resolution logic in `sae_eval.py`:

- **Position datasets** use the **metals** theme (`copper, bronze, iron, steel, amalgam`), one metal per opponent-mode mixture. `amalgam` = combined+deduped across all four modes.
- **BSP sets** use the **animals** theme (`gorilla` = full 164, `hawk_173` = reframed Nanda-style threat set, `fox` = cells/phase only, etc.).
- Filename patterns: `positions-<metal>_unique.pt`, `bsp_labels-<animal>_<count>.pt`, `bsp_schema-<animal>_<count>.json`.
- **Follow-up SAE run IDs** (from 2026-04-24): `{Major}{Minor}-{tag}-s{seed}`, e.g. `A01-random-control-s42`. The trainer appends `-{arch}-{hook}` to that stem, so do **not** include arch/hook/expansion in `experiment:`. New `{Major}{Minor}` whenever the *condition* (hook, arch family, data source, purpose) changes; only `s{seed}` varies for replicates. See `configs/followup/README.md` and `Quarto-specifications.md` for the policy.
- Binary attribute suffixes follow quartopy enum naming: `_tall`, `_black`, `_square`, `_with_hole` (positives only — negative complements are not currently probed; this is a known gap).

## Things that have bitten this project before

- **BSP set filtering — `*_337.pt` is a bug, not a set** (noted 2026-05-07). `get_all_bsp_definitions()` returns the union of gorilla (164) + hawk (173) = 337 BSPs as a *menu*. Gorilla and hawk are **alternative bases** for the same threat concepts (see `BSP-schema-summary.md` "Gorilla ↔ Hawk Correspondence"); a 337-eval pools redundant signals and produces a meaningless coverage number. Always evaluate gorilla and hawk separately. `compute_bsp_labels.py --name gorilla|hawk` now auto-resolves to the right `--only-categories` via `quarto.BSP_SETS`; calling it without `--name` matching a known set or `--only-categories` falls back to all 337 BSPs and prints a warning. If you encounter a `bsp_labels-*_337.pt` on disk, it was generated before this auto-resolution existed (pre-2026-05-07) — re-run with `--name gorilla` and `--name hawk` to produce the canonical 164/173 files. Columns of an existing `_337` file are gorilla[0:164] then hawk[164:337] in stable order if you need to salvage rather than recompute.
- **BatchTopK eval mode**: must run in **train mode** (batch-level sparsity). Inference mode uses calibrated thresholds and collapses L0 at eval time. `sae_eval.py` handles this — preserve that behavior if you touch evaluation.
- **BatchTopK `_thresholds_calibrated` does not survive save/load**. The flag is a plain Python attribute, not a registered buffer, so it is *not* in `state_dict()` and resets to `False` on every `load_checkpoint()`. To detect whether a loaded BatchTopK was calibrated, check the buffer instead: `torch.any(sae._threshold_estimate != 0)`. The buffer *is* persisted, and `train_sae()` calibrates it as the last step of training, so any SAE produced via the normal pipeline is fine. `export_onnx.py` uses the buffer-based check.
- **Decoder normalization**: `normalize_decoder()` must run after every optimizer step; without it, the L1 penalty trivially shrinks `h` instead of producing sparsity.
- **`offered_piece` BSPs** sit at the trivial-baseline F1 ≈ 0.667 (P=0.5, R=1.0). Any run reporting that exact number found *no* discriminative features, not real signal. Keep it in per-category breakdowns for transparency, but **exclude it from headline / threat-focused rankings** (deprioritized 2026-04-27). The MCC and F1-lift columns added 2026-05-11 collapse to 0 for these trivial features — use them in headline tables and the artifact disappears.
- **Three coverage metrics, always together** (rule from 2026-05-11): every eval writes `coverage` (F1, literature standard), `coverage_mcc` (Matthews correlation; base-rate-invariant), and `coverage_f1_lift` (F1 minus the 2p/(1+p) trivial-baseline, clipped at 0). All three are derivable from the same TP/FP/FN/TN matrices, so the extra cost is negligible. `scripts/backfill_eval_metrics.py --game=quarto` retro-fills older registry entries from existing `_matching-*.pt` and `_h.pt` caches under `saes/<game>/cache/`. F1-lift is the recommended *headline* metric for trained-vs-random comparisons (the gap is ~3.5× more discriminating than raw F1).
- **No new broad unsupervised arch sweeps on fc1** (deprioritized 2026-04-27; scope narrowed 2026-04-29). Anakin's σ=0.004 settles this *for fc1*: within the well-tuned fc1 middle ground, additional variants yield <0.01 coverage gains. This does NOT apply to conv2: batchtopk, vanilla, and p-annealing have never been run on conv2; SAE/LP efficiency on conv2 is only 42% vs 84% on fc1; and the fc1 winner (BatchTopK-k16) has never been tested on conv2. A full architecture sweep on conv2 is justified (Campaigns C–G, launched 2026-04-29). New fc1 sweeps should focus on Guided/anchored/E2E variants.
- **fc1 is for bottleneck comparison only** (deprioritized 2026-04-27). Phase 1F redirected threat work to conv2; do not start new fc1-only threat investigations.
- **Per-cell conv2 SAEs require cell-relative BSPs** (noted 2026-04-29). Training SAEs on per-cell activations (conv2_amalgam_activations.pt, shape [N×16, 32]) and evaluating against position-level BSPs is semantically invalid: a perfect cell-specific feature has 1/16 recall (it fires for 1 of 16 cells per position, but the BSP label is 1 for all 16), capping F1 at 0.118. Per-cell SAEs need a new BSP set where labels are computed relative to the cell being processed (e.g., "is THIS cell occupied by a tall piece?"), not the position.
- **Deduplicate AFTER aggregating** raw position files, not before — early-game positions repeat heavily across opponent modes, and per-file dedup leaves cross-file duplicates.
- **`mode_2x2=True`** is required when generating positions for current work. Pre-2026-03-27 data lived under `legacy_mode2x2_false/` and is provenance-only; do not mix.
- **`sae_eval.py` re-eval skipping**: the registry check uses `run_id:bsp_set` as the key. If you evaluate the same checkpoint against a *different* BSP set without `--force`, the gorilla (or prior) result is in the registry under a different key, so the new eval will run correctly. If you somehow have the old plain-`run_id` key style already cached for a run, the new bsp_set eval will proceed (no collision). Always pass `--force` if you need to recompute an existing `run_id:bsp_set` pair.
- **docopt quirks**: avoid `--` prefix collisions and line continuations inside docstrings, they break parsing silently.
- **Multi-GPU sweep**: GPU 0 (RTX 4000) is significantly slower than the P4000s on this machine — `run_sweep.py --split` should keep heavy conv2 configs off GPU 0.

## Style

The CLI scripts in this repo use **docopt-style module docstrings** as the source of truth for arguments — keep the docstring and any `argparse`-style help in sync if you add options. Top-level scripts (`sae_train.py`, `sae_eval.py`, `run_sweep.py`, `validate_sweep.py`) and most `scripts/*.py` follow this pattern.
