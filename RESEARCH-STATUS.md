# Quarto SAE Research — Quick Reference

> **Doc convention:** this file is the *high-level* ledger — current phase,
> headline numbers, hypothesis status, policy decisions. Sweep-level
> specifics (full per-category tables, per-config breakdowns, AI-assisted
> interpretation, design rationale) live in
> [`docs/diary/`](docs/diary/README.md). Supervisor meeting snapshots —
> one file per cycle, self-contained, frozen after the meeting — live in
> [`docs/advances-supervisor/`](docs/advances-supervisor/).
> When in doubt, add a one-line bullet here and a dated diary entry there.

## Project State

### 2026-05-19 (post-sweep) — current

**Active phase:** Phase 2B-sweep complete. 25 SAE configs trained on Deep
Brain across champS4 (12) and champTa (13) at the shared
`QuartoCNNAutoregUnifiedS4` architecture; 8-run LP baseline on champTa
landed alongside.

- **champTa is strictly better than champS4** on every behavioral test
  (winning placement +8 pp, loss avoidance +24 pp) and at every LP cell
  (F1-lift Δ +0.030 to +0.112; biggest gain on fc1 / hawk).
- **H9 (concept distillation) ✅ CONFIRMED.** Depth-2 minimax distillation
  amplifies threat structure rather than producing a shortcut: every
  diagnostic hawk category rises on champTa relative to champS4.
- **Architecture winner transfers:** `E05 BatchTopK k=32 exp=8 s4.conv2`
  is the gorilla winner on both champions; both hawk winners are at fc1.
- **Sweep success criterion (SAE ≥ 50 % of LP F1-lift) FAILS** at every
  cell; conv2 / hawk sits at 11 %. Next direction is concept-targeted /
  wider-expansion SAEs at a new SAE objective, not more breadth at the
  current budget. Design: [`docs/diary/2026-05-19_concept-targeted-saes.md`](docs/diary/2026-05-19_concept-targeted-saes.md).

**Open verification work:** F04-Ta jumprelu hawk winner has ~7 metrics
rows (likely early-stop), so the +0.028 lift over F01-S4 needs a clean
rerun with patience disabled before being promoted as a final claim.
Protocol in [`docs/diary/phase-2B.md`](docs/diary/phase-2B.md) §
"Verification protocols".

Full details: [`docs/diary/phase-2B.md`](docs/diary/phase-2B.md).

### Earlier states (compact)

- **2026-05-19 (pre-sweep)** — champS4 LP baseline diagnostic done; champTa champion just landed (+16.6 pp WR vs champS4); rescoped to 25-config two-champion sweep. → see [`phase-2B.md`](docs/diary/phase-2B.md) ch. 2.
- **2026-05-18** — champS4 mini-sweep (4 configs) underperformed champAa twins; diagnostic LP-first plan written. → see [`phase-2B.md`](docs/diary/phase-2B.md) ch. 1.
- **2026-05-11** — Phase 2A complete on Deep Brain (33 conv2 arch configs, C01 winner at 0.353). Reporting standard adopted (F1 + MCC + F1-lift). Phase 1G competence audit revealed champAa's "attack-recognition / defence-blindness" asymmetry. → see [`phase-2A.md`](docs/diary/phase-2A.md) and [`phase-1.md`](docs/diary/phase-1.md) § Phase 1G.
- **2026-05-04** — Phases 0–1F complete on champAa. Data pipeline, Anakin SAE-architecture sweep, conv2 LP. → see [`phase-1.md`](docs/diary/phase-1.md).

## Key Metrics Summary

| Probe / SAE | Champion | Hook | BSP set | F1 | F1-lift |
|---|---|---|---|:---:|:---:|
| Linear probe (trained) | Aa | fc1 | gorilla_164 | 0.402 | — |
| Linear probe (random)  | Aa | fc1 | gorilla_164 | 0.194 | — |
| Linear probe (trained) | Aa | conv2 | gorilla_164 | 0.789 | — |
| Linear probe (trained) | S4 | fc1 | gorillaS4 | 0.719 | 0.520 |
| Linear probe (trained) | S4 | conv2 | gorillaS4 | 0.877 | 0.678 |
| Linear probe (trained) | S4 | fc1 | hawkS4 | 0.584 | 0.548 |
| Linear probe (trained) | S4 | conv2 | hawkS4 | 0.750 | 0.715 |
| Linear probe (trained) | **Ta** | fc1 | gorillaTa | 0.810 | 0.595 |
| Linear probe (trained) | **Ta** | conv2 | gorillaTa | 0.923 | **0.708** |
| Linear probe (trained) | **Ta** | fc1 | hawkTa | 0.702 | 0.660 |
| Linear probe (trained) | **Ta** | conv2 | hawkTa | 0.826 | **0.785** |
| Best SAE (anakin batchtopk-k16 fc1) | Aa | fc1 | gorilla | 0.338 | 0.141 |
| Best SAE (C01 topk-k16-exp8 conv2) | Aa | conv2 | gorilla | **0.353** | — |
| Best SAE (E05 batchtopk-k32 conv2) | **Ta** | conv2 | gorillaTa | 0.428 | **0.214** |
| Best SAE (F04 jumprelu-t64 fc1) | **Ta** | fc1 | hawkTa | 0.214 | **0.172** |

## Hypotheses Status

| # | Hypothesis | Status |
|---|---|---|
| H1 | fc1 encodes cell-level BSPs linearly but not threats *(on champAa)* | ✅ CONFIRMED across 28 SAE configs |
| H1b | Threats accessible in reframed basis | ⚠️ PARTIAL — count encoding 14× better but still F1 = 0.315 on Aa; reframed_count F1-lift 0.18–0.25 on Ta |
| H2 | Feature absorption | OPEN |
| H3 | Concept heterogeneity → architecture-dependent failures | ❌ WEAKENED — most architectures perform similarly |
| H5 | Feature shrinkage (L1 / ReLU) | ⚠️ PARTIAL — Gated / Vanilla in Tier 3, P-Annealing not Tier 1 |
| H6 | Wrong hook point — conv2 may be better | ✅ SUPPORTED on Aa; partially **inverted on Ta for hawk** (fc1 SAEs > conv2 SAEs on rare threats; see phase-2B.md Claim 4) |
| H7 | Offered piece is not learned | ✅ CONFIRMED — F1 = 0.667 is trivial baseline |
| H8 | Threat info is spatially encoded, lost at fc1 bottleneck | ⚠️ CHAMPION-SPECIFIC — confirmed for champAa, overturned for champS4 / champTa. Unified-aux family preserves threats through fc1; minimax distillation amplifies them. |
| H9 | Oracle distillation distils *concepts* (vs non-decomposable shortcut) | ✅ CONFIRMED (2026-05-19). All four hawk diagnostic cells rise on Ta; biggest at fc1 / hawk (+0.112 lift). See [phase-2B.md](docs/diary/phase-2B.md) ch. 3. |

## BSP Sets

- **gorilla_164** — 7 categories: `cell_occupancy`, `cell_attribute`, `threat_line`, `threat_square_2x2`, `offered_piece`, `global`, `game_phase`. Position-level labels.
- **hawk_173** — 7 reframed categories (Nanda-style threat reframing): `reframed_count` × 40, `reframed_completable` × 40, `reframed_any_threat` × 10, `reframed_sq_count` × 36, `reframed_sq_completable` × 36, `reframed_sq_any_threat` × 9, `reframed_global` × 2. Position-level labels.
- Per-champion suffixes (`gorillaS4`, `gorillaTa`, etc.) recompute labels against that champion's self-play position distribution so base rates match the activations.
- See [`BSP-schema-summary.md`](BSP-schema-summary.md) for full schema and gorilla ↔ hawk correspondence.

## Reporting Standard (REQUIRED for every winner claim, adopted 2026-05-11)

1. **Always report three coverage metrics side-by-side** — F1 (literature standard), MCC (base-rate-invariant), F1-lift (F1 minus trivial `2p/(1+p)` baseline, clipped at 0). Eval pipeline writes all three to `eval_registry.json`; `scripts/backfill_eval_metrics.py` retro-fills.
2. **Always compare against LP ceiling AND random-network control.** The learned-gap fraction `(cov_SAE − cov_rand_SAE) / (LP_trained − LP_random)` is the cleanest way to claim an SAE captures *learned* structure rather than the architectural prior.
3. **Always present the per-category breakdown count-weighted by N.** Headline mean drags down through high-N low-F1 categories (76 of 164 gorilla BSPs are threats at ~0.08).

Full table template and rationale: [`phase-2A.md`](docs/diary/phase-2A.md) § "Reporting standard".

## Active plan / next steps

1. **F04-Ta verification rerun** (≤ 30 min on Deep Brain) — patience-disabled rerun of jumprelu-t64-fc1 to confirm the +0.028 hawk lift over F01-S4 isn't a truncation artifact. Protocol: [`phase-2B.md`](docs/diary/phase-2B.md) § "Verification protocols".
2. **Concept-targeted SAE experiments** — five candidate approaches sketched in [`2026-05-19_concept-targeted-saes.md`](docs/diary/2026-05-19_concept-targeted-saes.md). Recommended ordering: wider exp grid → higher k → anchored hawk SAEs → E2E → matryoshka. Decision gates pre-registered. Implementation in a new session.
3. **Cell-relative BSP set ("new animal")** — only relevant if per-cell conv2 SAEs (`--flatten-per-cell`) are reactivated. Not on the current path with `--flatten-position`.

## Deprioritized

- **Broad unsupervised arch sweeps on fc1** — Anakin (28 configs, σ = 0.004) showed this is second-order for fc1. Scope clarified 2026-04-29: this deprioritization is fc1-specific. Conv2 had its full arch sweep in Phase 2A (Campaigns C–G). **Caveat (2026-05-19):** fc1 deprioritization is `CNN_uncoupled`-specific. On champS4 / champTa the fc1 bottleneck preserves threat information; fc1 SAE work on threats is reactivated for the unified-aux family.
- **fc1-only threat investigation on champAa** — Phase 1F redirected this work to conv2. fc1 retained only as bottleneck-comparison reference for the Aa champion.
- **`offered_piece` as a coverage signal** — F1 ≈ 0.667 is the trivial all-positive baseline. Keep it in per-category breakdowns for transparency; exclude from headline / threat-focused rankings.
- **Phase 4 (auxiliary threat-prediction head during DQN training)** — retired 2026-05-19. Motivated by champAa's collapsed fc1 threat signal; the unified-aux objective and minimax distillation solved the bottleneck problem at training time instead.
- **Anchored / guided SAEs on champAa** — would have failed by construction (champAa doesn't compute completability). **Re-enabled for champTa** where completability is decodable; see [`2026-05-19_concept-targeted-saes.md`](docs/diary/2026-05-19_concept-targeted-saes.md).
- **Matryoshka on champAa** — deprioritised 2026-05-11 (missing concepts aren't being absorbed, they're never represented). Re-evaluate on champTa once anchored / E2E results land.

## Follow-up Run Naming (from 2026-04-24 onward)

- Experiment IDs of the form `{Major}{Minor}-{tag}-s{seed}` so checkpoints, eval caches, and registry keys stay unique.
- Current follow-up panels use `A01`/`A02` for random-model controls, `B01`–`B04` for the conv2 completion panel, `C/D/F/G` for Phase 2A campaigns, `E/F` for Phase 2B-sweep.
- Keep the seed in the experiment ID itself, because the trainer names checkpoints from `experiment + architecture suffix + hook`.

## Known Issues

- **docopt:** avoid `--` prefix collisions and line continuations in docstrings (silent parser failure).
- **GPU 0 is significantly slower than GPUs 1 / 2 locally** — use `run_sweep.py --split` to avoid assigning heavy conv2 configs to GPU 0. (Deep Brain's three A6000s are equal.)
- **`offered_piece` F1 = 0.667** is a metrics artefact (trivial baseline), not real coverage.
- **conv2 overall coverage** mixes strongly-learned categories with architecture-easy categories — always compare against the random conv2 control.
- **BatchTopK `_thresholds_calibrated` does not survive save/load.** Detect via the buffer: `torch.any(sae._threshold_estimate != 0)`. `export_onnx.py` already uses this check.
- **`_337.pt` BSP files** are bugs, not a BSP set — gorilla and hawk are alternative bases for the same threat concepts and must be evaluated separately. See `CLAUDE.md` § "Things that have bitten this project before".
