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

### 2026-05-26 (champVe full sweep complete) -- current

**110 evals across gorillaVe, hawkVe, tigerVe.** Anchored I04 lh=1.0 recipe transfers perfectly from Ta to Ve: tigerVe F1-lift **0.255** (vs tigerTa 0.257); SAE/LP efficiency 85.6% on both champions. Lambda ordering and J-series failure mode reproduce. Ve LP improves on state concepts (gorilla conv2 lift 0.762 vs Ta 0.708, +0.054) but NOT on strategic tiger concepts (fc1 flat, conv2 -0.039). Non-anchored conv2 SAEs harder to train on Ve. Full results: [`docs/diary/2026-05-26_champVe-results.md`](docs/diary/2026-05-26_champVe-results.md).

### 2026-05-26 (unified cross-champion position dataset)

Per-champion position distributions bias evaluations toward each champion's visited states. New tool `scripts/unify_positions.py` merges all champion amalgams into a single deduplicated pool; BSP set suffix encodes position count (e.g. `gorilla156k`) so the evaluation basis is always clear. Day-to-day sweeps keep champion-specific data; unified pool used only for cross-champion reporting. Design: [`docs/diary/2026-05-26_unified-position-dataset.md`](docs/diary/2026-05-26_unified-position-dataset.md).

### Earlier states (compact)

- **2026-05-25 (Anchored sweep I/J on champTa)** -- I04 anchored-jumprelu lh=1.0 wins tigerTa (+62% F1-lift vs baseline), recipe confirmed on champVe (see above). --> see [`docs/diary/2026-05-25_anchored-sweep-ij-results.md`](docs/diary/2026-05-25_anchored-sweep-ij-results.md).

- **2026-05-22 (Tiger reframing audit closed)** — pre-registered decision rule fires: tiger SAE/LP efficiency exceeds hawk by +25 pp (conv2/Ta) to +51 pp (conv2/S4). Supervision pivot now targets tiger, not hawk. Secondary: fc1 > conv2 for tiger; Sweep H null reproduces on tiger; real feature absorption on champTa (max 6–10 BSPs/feature). → see [`docs/diary/2026-05-22_reframings-audit-tiger.md`](docs/diary/2026-05-22_reframings-audit-tiger.md).
- **2026-05-22 (Sweep H closed)** — capacity scan (12 configs) closes 0% of the conv2/hawk gap; concept-targeting promoted to primary direction. → see [`docs/diary/phase-2B.md`](docs/diary/phase-2B.md) ch. 4.
- **2026-05-19 (post-sweep)** — Phase 2B-sweep complete (25 configs across champS4/champTa). H9 (concept distillation) **CONFIRMED**; champTa strictly better than champS4 at every cell. Best architecture transfers: E05 BatchTopK k=32 exp=8 conv2 is the gorilla winner on both. Sweep success criterion (SAE ≥ 50% of LP F1-lift) **FAILS** at every cell. F04 patience-fix verification (2026-05-20): fc1-on-hawk claim STANDS on both champions. → see [`docs/diary/phase-2B.md`](docs/diary/phase-2B.md) and [`docs/diary/2026-05-19_concept-targeted-saes.md`](docs/diary/2026-05-19_concept-targeted-saes.md).
- **2026-05-18** — champS4 mini-sweep (4 configs) underperformed champAa twins; diagnostic LP-first plan. → see [`docs/diary/phase-2B.md`](docs/diary/phase-2B.md) ch. 1.
- **2026-05-11** — Phase 2A complete on Deep Brain (33 conv2 arch configs, C01 winner at 0.353). Reporting standard adopted (F1 + MCC + F1-lift). Phase 1G competence audit revealed champAa's "attack-recognition / defence-blindness" asymmetry. → see [`docs/diary/phase-2A.md`](docs/diary/phase-2A.md) and [`docs/diary/phase-1.md`](docs/diary/phase-1.md) § Phase 1G.
- **2026-05-04** — Phases 0–1F complete on champAa. Data pipeline, Anakin SAE-architecture sweep, conv2 LP. → see [`docs/diary/phase-1.md`](docs/diary/phase-1.md).

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
| Linear probe (trained) | **Ta** | fc1 | tigerTa | **0.591** | **0.301** |
| Linear probe (trained) | **Ta** | conv2 | tigerTa | 0.463 | 0.175 |
| Linear probe (trained) | S4 | fc1 | tigerS4 | 0.440 | 0.179 |
| Linear probe (trained) | S4 | conv2 | tigerS4 | 0.335 | 0.098 |
| Best SAE (F04 jumprelu-t64 fc1) | **Ta** | fc1 | tigerTa | 0.449 | **0.158** |
| Best SAE (E05 batchtopk-k32 conv2) | **Ta** | conv2 | tigerTa | 0.353 | 0.063 |
| Linear probe (trained) | **Ve** | fc1 | gorillaVe | 0.802 | 0.605 |
| Linear probe (trained) | **Ve** | conv2 | gorillaVe | **0.959** | **0.762** |
| Linear probe (trained) | **Ve** | fc1 | hawkVe | 0.692 | 0.662 |
| Linear probe (trained) | **Ve** | conv2 | hawkVe | 0.851 | **0.822** |
| Linear probe (trained) | **Ve** | fc1 | tigerVe | 0.549 | 0.298 |
| Linear probe (trained) | **Ve** | conv2 | tigerVe | 0.385 | 0.136 |
| Best SAE (I03 anch-jrelu lh030 fc1) | **Ve** | fc1 | gorillaVe | 0.391 | **0.195** |
| Best SAE (E01 topk-k32 conv2) | **Ve** | conv2 | gorillaVe | 0.389 | 0.192 |
| Best SAE (I04 anch-jrelu lh100 fc1) | **Ve** | fc1 | tigerVe | 0.506 | **0.255** |

† S4-fc1 LP rerun at `max_iter=5000` (closed 2026-05-22 PM) confirmed the headline numbers stand: F1=0.4402, MCC=0.3797, F1-lift=0.1789 vs the `max_iter=1000` original of 0.440/0.380/0.179. The 5 unconverged-warning BSPs were cosmetically not-converged but functionally at their optimum.

## Hypotheses Status

| # | Hypothesis | Status |
|---|---|---|
| H1 | fc1 encodes cell-level BSPs linearly but not threats *(on champAa)* | ✅ CONFIRMED across 28 SAE configs |
| H1b | Threats accessible in reframed basis | ⚠️ PARTIAL — count encoding 14× better but still F1 = 0.315 on Aa; reframed_count F1-lift 0.18–0.25 on Ta |
| H1c | Target-set mismatch contributes to the SAE/LP wall | ✅ CONFIRMED (2026-05-22). Tiger SAE/LP efficiency exceeds hawk by +25 pp (conv2/Ta) to +51 pp (conv2/S4). Agent-relative framing recovers a meaningful chunk of the wall; supervision pivot now targets tiger. |
| H2 | Feature absorption | ⚠️ PARTIAL (2026-05-22) — F04-Ta fc1 max 6 BSPs/feature on tiger; E05-Ta conv2 max ≈ 10. Champion-era SAEs do show absorption (contradicting the older champAa-era "never represented" claim). Matryoshka re-activated on champTa. |
| H3 | Concept heterogeneity → architecture-dependent failures | ❌ WEAKENED — most architectures perform similarly |
| H5 | Feature shrinkage (L1 / ReLU) | ⚠️ PARTIAL — Gated / Vanilla in Tier 3, P-Annealing not Tier 1 |
| H6 | Wrong hook point — conv2 may be better | ✅ SUPPORTED on Aa; partially **inverted on Ta for hawk** (fc1 SAEs > conv2 SAEs on rare threats; see phase-2B.md Claim 4) |
| H7 | Offered piece is not learned | ✅ CONFIRMED — F1 = 0.667 is trivial baseline |
| H8 | Threat info is spatially encoded, lost at fc1 bottleneck | ⚠️ CHAMPION-SPECIFIC — confirmed for champAa, overturned for champS4 / champTa. Unified-aux family preserves threats through fc1; minimax distillation amplifies them. |
| H9 | Oracle distillation distils *concepts* (vs non-decomposable shortcut) | ✅ CONFIRMED (2026-05-19). Extended on Ve (2026-05-26): DISABLE_NEVER deepens state encoding (gorilla conv2 LP +0.054 vs Ta) but does NOT boost strategic/tiger concepts (fc1 flat, conv2 -0.039). Concept distillation has a ceiling on agent-relative BSPs. See [phase-2B.md](docs/diary/phase-2B.md) ch. 3 and [champVe-results.md](docs/diary/2026-05-26_champVe-results.md). |

## BSP Sets

- **gorilla_164** — 7 categories: `cell_occupancy`, `cell_attribute`, `threat_line`, `threat_square_2x2`, `offered_piece`, `global`, `game_phase`. State-only. Position-level labels.
- **hawk_173** — 7 reframed categories (Nanda-style threat reframing): `reframed_count` × 40, `reframed_completable` × 40, `reframed_any_threat` × 10, `reframed_sq_count` × 36, `reframed_sq_completable` × 36, `reframed_sq_any_threat` × 9, `reframed_global` × 2. State-only. Position-level labels.
- **tiger_36** — 6 agent-relative categories (added 2026-05-22, attacks the "state-only" gap shared by gorilla/hawk): `tiger_decision_global` × 5, `tiger_offered_completing_attr` × 4, `tiger_line_winnable` × 10, `tiger_square_winnable` × 9, `tiger_pool_winning_count` × 4, `tiger_pool_safe_count` × 4. Includes pool-reasoning concepts (count of "poison" / "safe" pieces the player could offer). See [`docs/diary/2026-05-22_reframings-audit-tiger.md`](docs/diary/2026-05-22_reframings-audit-tiger.md).
- Per-champion suffixes (`gorillaS4`, `gorillaTa`, `tigerS4`, `tigerTa`, etc.) recompute labels against that champion's self-play position distribution so base rates match the activations. The **schema** file is keyed by basis only (`bsp_schema-gorilla_164.json`, `bsp_schema-tiger_36.json`) because the concept menu is distribution-independent; only the label tensor varies per champion.
- See [`docs/BSP-schema-summary.md`](docs/BSP-schema-summary.md) for full schema and gorilla ↔ hawk correspondence.

## Reporting Standard (REQUIRED for every winner claim, adopted 2026-05-11)

1. **Always report three coverage metrics side-by-side** — F1 (literature standard), MCC (base-rate-invariant), F1-lift (F1 minus trivial `2p/(1+p)` baseline, clipped at 0). Eval pipeline writes all three to `eval_registry.json`; `scripts/backfill_eval_metrics.py` retro-fills.
2. **Always compare against LP ceiling AND random-network control.** The learned-gap fraction `(cov_SAE − cov_rand_SAE) / (LP_trained − LP_random)` is the cleanest way to claim an SAE captures *learned* structure rather than the architectural prior.
3. **Always present the per-category breakdown count-weighted by N.** Headline mean drags down through high-N low-F1 categories (76 of 164 gorilla BSPs are threats at ~0.08).

Full table template and rationale: [`phase-2A.md`](docs/diary/phase-2A.md) § "Reporting standard".

## Active plan / next steps

1. ~~**F04-Ta verification rerun**~~ ✅ done 2026-05-20.
2. ~~**Sweep H — capacity scan**~~ ✅ done 2026-05-22. Gate FAILS; concept-targeting promoted. See [`docs/diary/phase-2B.md`](docs/diary/phase-2B.md) ch. 4.
3. ~~**Reframing audit — `tiger` BSPs**~~ ✅ done 2026-05-22. **Target-set mismatch contributing**; tiger SAE/LP efficiency exceeds hawk by +25 to +51 pp depending on cell. Anchored / matryoshka pivot now targets *tiger*. Results: [`docs/diary/2026-05-22_reframings-audit-tiger.md`](docs/diary/2026-05-22_reframings-audit-tiger.md) §"Results".
4. **Literature review refresh** (PI-owned, in progress; finishes before next session). Last broad scan predates matryoshka / E2E / BatchTopK family follow-ups; needs fresh pass on (a) SAE variants 2025–2026 — especially anchored / supervised forms, (b) board-game interp work since Karvonen, (c) low-base-rate / rare-concept SAE methods.
5. **Anchored SAE on champTa fc1 → tigerTa** (next implementation, after step 4). Largest LP headroom of any cell (F1-lift 0.301 vs SAE 0.158), tiger validated as the supervision target. Two architectures (anchored-jumprelu, anchored-batchtopk) × four λ values × three seeds = 24 runs. Detailed plan: [`docs/diary/2026-05-22_anchored-sae-champta-tiger.md`](docs/diary/2026-05-22_anchored-sae-champta-tiger.md).
6. **Matryoshka SAE** (parallel engineering track, starts when step 5 has results to compare against). Re-activated on champTa after the 2026-05-22 feature-sharing evidence (max 6–10 BSPs/feature) showed real absorption — contradicting the champAa-era deprioritization. Targets the overall SAE/LP efficiency gap on dense categories; should be evaluated against tiger primarily, gorilla/hawk secondarily.
7. **E2E SAEs** — start when at least one of {anchored, matryoshka} has landed. E2E is most diagnostic *against a strong baseline*, not as a first move; per the 2026-05-19 design note the contrast is what makes the result informative.
8. ~~**S4-fc1 LP `max_iter=5000` rerun**~~ ✅ done 2026-05-22 PM. Headline 0.179 lift stands (rerun: 0.1789); 5 unconverged-warning BSPs were at their functional optimum.
9. **Novel SAE variant placeholder** — base-rate-weighted reconstruction loss, deferred until after anchored / matryoshka / E2E results land.
10. **Cell-relative BSP set ("new animal")** — only relevant if per-cell conv2 SAEs (`--flatten-per-cell`) are reactivated. Not on the current path.

Design rationale: [`docs/diary/2026-05-19_concept-targeted-saes.md`](docs/diary/2026-05-19_concept-targeted-saes.md). Next-experiment design: [`docs/diary/2026-05-22_anchored-sae-champta-tiger.md`](docs/diary/2026-05-22_anchored-sae-champta-tiger.md).

## Deprioritized

- **Broad unsupervised arch sweeps on fc1** — Anakin (28 configs, σ = 0.004) showed this is second-order for fc1. Scope clarified 2026-04-29: this deprioritization is fc1-specific. Conv2 had its full arch sweep in Phase 2A (Campaigns C–G). **Caveat (2026-05-19):** fc1 deprioritization is `CNN_uncoupled`-specific. On champS4 / champTa the fc1 bottleneck preserves threat information; fc1 SAE work on threats is reactivated for the unified-aux family.
- **fc1-only threat investigation on champAa** — Phase 1F redirected this work to conv2. fc1 retained only as bottleneck-comparison reference for the Aa champion.
- **`offered_piece` as a coverage signal** — F1 ≈ 0.667 is the trivial all-positive baseline. Keep it in per-category breakdowns for transparency; exclude from headline / threat-focused rankings.
- **Phase 4 (auxiliary threat-prediction head during DQN training)** — retired 2026-05-19. Motivated by champAa's collapsed fc1 threat signal; the unified-aux objective and minimax distillation solved the bottleneck problem at training time instead.
- **Anchored / guided SAEs on champAa** — would have failed by construction (champAa doesn't compute completability). **Re-enabled for champTa** with `tigerTa` as the supervision target (2026-05-22 decision-rule outcome — *not* hawk); see [`2026-05-22_anchored-sae-champta-tiger.md`](docs/diary/2026-05-22_anchored-sae-champta-tiger.md).
- **Matryoshka on champAa** — deprioritised 2026-05-11 (missing concepts aren't being absorbed, they're never represented). **Re-activated on champTa** (2026-05-22) after tiger evals showed real absorption (max 6–10 BSPs/feature).

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
