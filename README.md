# games-interp

PhD research on **mechanistic interpretability of board-game neural networks**.
The active focus is **Quarto** (a CNN DQN); Othello and Tic-tac-toe are scaffolded
but not implemented. The core technique is training Sparse Autoencoders (SAEs) on
hooked activations and evaluating them against hand-defined ground-truth concepts
called **BSPs** (Board State Properties).

## Where to look first

| File | What it's for |
|---|---|
| [`RESEARCH-STATUS.md`](RESEARCH-STATUS.md) | Current phase, headline metrics, hypothesis status, active plan, deprioritized list. Read this first. |
| [`CLAUDE.md`](CLAUDE.md) | Agent operating manual: documentation contract, common commands, architecture, conventions, recurring failure modes. |
| [`docs/diary/`](docs/diary/) | Dated entries with full per-sweep tables, AI-reasoned interpretation, design notes. [diary README](docs/diary/README.md) is the index. |
| [`Quarto-specifications.md`](Quarto-specifications.md) | Stable Quarto reference: game rules, model architecture, hookable layers, naming conventions, run-id rule. |
| [`docs/BSP-schema-summary.md`](docs/BSP-schema-summary.md) | BSP schema for gorilla / hawk / tiger bases. |
| [`configs/models/`](configs/models/) | Source of truth for each champion model (paths, benchmarks, hooks). |
| [`commands.sh`](commands.sh) | **Transient.** The user's current pipeline-execution recipe; rewritten per phase. Not authoritative infrastructure. |

## Quickstart

```bash
pip install -r requirements.txt
pytest                              # CPU-only test suite (fast)
python sae_train.py --config=configs/champTa/F04-champTa-s42.yaml
python sae_eval.py evaluate saes/quarto/<run_id>.pt --bsps=gorillaTa
python scripts/registry_query.py top --bsps=gorillaTa --limit=10
```

See [`CLAUDE.md` § Common commands](CLAUDE.md) for the full command surface
(training, eval, sweeps, anchor analysis, viz export, linear-probe baselines).

## Repo layout (essentials)

```
lib/sae/             # Game-agnostic SAE library (6 architectures, train, eval, hooks)
sae_train.py         # Train one SAE (YAML or CLI)
sae_eval.py          # Evaluate / compare / history subcommands
run_sweep.py         # Multi-config / multi-GPU sweep orchestrator
scripts/games/       # Per-game glue (quarto.py, quarto_s4.py)
scripts/             # CLIs (positions, activations, BSPs, audit, viz, registry)
configs/             # YAML training configs, organized by sweep / champion
configs/models/      # Per-champion model registry
models/quarto/       # Game-model checkpoints (force-added)
data/quarto/         # Positions, activations, BSP labels (gitignored)
saes/quarto/         # SAE checkpoints + training/eval registries
docs/diary/          # Long-form research log
tests/               # pytest suite
```

## Documentation policy

The repo has a strict layered doc structure to prevent drift. The full contract
(what goes where, what gets pruned) lives at the top of [`CLAUDE.md`](CLAUDE.md).
The short version: **`RESEARCH-STATUS.md` holds the one-line ledger; long
results and rationales go in `docs/diary/`. Never duplicate — link.**

## License

See [`LICENSE`](LICENSE).
