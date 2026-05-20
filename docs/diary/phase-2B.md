# Phase 2B — Two champions, matched architecture (2026-05-18 → 2026-05-19)

A new model family (`QuartoCNNAutoregUnifiedS4`) landed two champions in
quick succession:

- **champS4** — Sa(3) unified-aux trunk, uniform-sampled training (2026-05-18).
- **champTa** — same trunk + depth-2 minimax oracle distillation on the
  SELECT head (2026-05-19, +16.6 pp head-to-head WR vs champS4).

Phase 2B works the shared architecture as a controlled experiment: same
hooks, same SAE recipes, same BSP schema — only the training procedure
varies. Predecessor: [`phase-2A.md`](phase-2A.md). Parent:
[`../../RESEARCH-STATUS.md`](../../RESEARCH-STATUS.md).

Phase 2B unfolded in three chapters:

1. **2026-05-18 — champS4 mini-sweep** (4 configs, mirror of champAa top runs). Surprise result: champS4 SAEs *underperform* champAa twins. Diagnostic plan written.
2. **2026-05-19 morning — LP baseline on champS4** + rescoped 25-config sweep plan. champS4 activations are *more* separable than champAa, but SAE/LP efficiency collapses from ~70 % to ~20 %.
3. **2026-05-19 afternoon — champTa LP + full 25-config sweep** completes on Deep Brain. champTa wins matched-config every time; oracle-distillation hypothesis confirmed.

---

## Chapter 1 — champS4 mini-sweep (2026-05-18) [DIRECT]

`commands.sh` re-ran the full pipeline on the new model: competence audit, S4-self-play position regeneration, BSP labels against the new distribution (`gorillaS4` / `hawkS4`), and the four top-Aa configs as a mini-sweep (`A01`, `C01`, `C07`, `D02`).

### Model competence audit (champS4 vs champAa)

| Test | champS4 | champAa | Δ |
|---|---:|---:|---:|
| A. Winning-placement acc | **0.787** | 0.700 | +0.087 |
| B. Losing-piece avoidance | **0.620** | 0.405 | **+0.215** |
| C. Offered-piece sensitivity | **0.228** | 0.144 | +0.084 |
| D. Q(empty) − Q(occupied), frac positive | 0.119 | 0.253 | **−0.134** |
| E. Late-game (11–15) mean entropy | 1.249 | 1.312 | small decrease |

The unified-aux model finally reasons about *what piece it hands the opponent* (test B doubles from 0.40 → 0.62), the first time any Quarto agent in this project rises above random on the defense axis. Test D regresses: the unmasked board head is even less legality-aware than champAa.

### SAE results [DIRECT, gorillaS4]

| run | F1-lift | cov | MCC | FVU | L0 | dead % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| D02 topk-k64-exp16 conv2 | 0.128 | 0.327 | 0.282 | 0.012 | 64 | 93.1 |
| A01 batchtopk-k16-exp2 fc1 | 0.125 | 0.324 | 0.292 | 0.058 | 16 | **96.6** |
| C07 jumprelu-t32-exp8 conv2 | 0.106 | 0.305 | 0.247 | 0.075 | 45 | 95.3 |
| C01 topk-k16-exp8 conv2 | 0.039 | 0.225 | 0.226 | 0.061 | 16 | 58.0 |

Pairwise vs the corresponding champAa twin:

| pair | Δ F1-lift | Δ FVU | comment |
|---|:---:|:---:|---|
| A01 fc1 batchtopk | −0.016 | +0.049 | small coverage drop, much higher FVU |
| C01 conv2 topk k16 | **−0.116** | +0.036 | severe collapse (cov > 0.5 falls 0.40 → 0.04) |
| C07 conv2 jumprelu | −0.033 | +0.049 | L0 drifted up; metrics jsonl has only 7 rows (early-stop) |
| D02 conv2 topk k64 | −0.006 | +0.008 | essentially matches Aa; best reconstruction (91 % reconstructable) |

On **hawkS4** all four cluster at lift 0.058–0.074 vs the champAa anakin-batchtopk peak of 0.105 — threats uniformly harder to recover on the new champion.

### Diagnostic plan written 2026-05-18 [INFERENTIAL]

Headline puzzle: **champS4 plays better but its activations are harder to interpret** with the recipes that won on champAa. FVU 3–6× higher across the board, one config (C01 k = 16) collapsed entirely. Candidate causes, decreasing likelihood:

1. **The S4 activations are less linearly separable.** Four-run spread consistent with a representation-level shift, not an SAE-recipe issue. Need LP ceiling to disambiguate.
2. **A01 expansion mismatch.** A01-champS4 uses `expansion=2` to match the *dictionary size* of the Aa twin (d_dict = 1024 from d_act = 128 × 8 = d_act = 512 × 2), but S4 fc1 has 4× more activation dimensions to compress — 96.6 % dead consistent.
3. **C07 truncated by patience.** `*_metrics.jsonl` has 7 rows vs 51 — likely `patience=20` + `min_improvement=0.02` triggered on jumprelu's θ warm-up plateau.
4. **C01 k = 16 too tight for S4 conv2.** Same hyperparams as Aa winner but cov > 50 collapsed to 4 %. Dead-features fell 81 % → 58 %, so features stayed alive but carry less concept information.

Plan, ordered by cost: (1) LP baseline on champS4; (2) fix C07 patience; (3) rerun A01 at exp = 8; (4) only then launch scoped conv2-S4 sweep around C01.

---

## Chapter 2 — LP baseline on champS4 + rescoped sweep plan (2026-05-19 morning) [DIRECT, 8 LP runs]

### Headline LP table

| Hook × BSP | LP F1 | LP MCC | LP F1-lift | random-net F1-lift | champAa LP F1 (was) |
|---|---:|---:|---:|---:|---:|
| fc1 × gorillaS4 | 0.719 | 0.715 | 0.520 | 0.105 | 0.402 |
| fc1 × hawkS4 | **0.584** | 0.588 | **0.548** | 0.001 | 0.200 |
| conv2 × gorillaS4 | **0.877** | 0.875 | **0.678** | 0.206 | 0.789 |
| conv2 × hawkS4 | **0.750** | 0.751 | **0.715** | 0.053 | — |

champS4 has *more* learned-specific structure than champAa at every hook (trained − random gaps are 2–6× larger). Random-net controls drop near zero on hawk — unambiguously learned signal.

### SAE / LP efficiency [INFERENTIAL]

| run | SAE F1-lift | LP F1-lift | efficiency |
|---|---:|---:|---:|
| A01 fc1 batchtopk-k16 (exp = 2) | 0.125 | 0.520 | 24 % |
| D02 conv2 topk-k64-exp16 | 0.128 | 0.678 | 19 % |
| C07 conv2 jumprelu-t32-exp8 | 0.106 | 0.678 | 16 % |
| C01 conv2 topk-k16-exp8 (collapsed) | 0.039 | 0.678 | 6 % |
| *champAa anakin-batchtopk-k16-fc1 (ref)* | 0.141 | ~0.19 | ~74 % |

The same SAE budgets recover ~3× less of the available signal on S4. Headroom is large — multiplying current S4 F1-lift by 2.5× would still only reach ~50 % efficiency.

### Hypothesis update

- **H8** (threat info is spatial; lost at fc1) was confirmed for champAa. **Overturned for champS4**: hawkS4 fc1 LP F1-lift = 0.548 (vs hawk-on-Aa fc1 = ≈ 0.005). The unified-aux training preserves threat information through the bottleneck. H8 is now a *champion-specific* property of `CNN_uncoupled`, not a Quarto-DQN property.
- **Completability** was near-zero on champAa LP at both hooks (≈ 0.003 fc1, 0.027 conv2). On champS4 it is decodable: line completability lift 0.356 (fc1), 0.634 (conv2); square completability 0.523 / 0.732. This matches the competence audit's huge jump on test B (40 % → 62 %) — the model needed completability to defend, so it learned it.

### Policy decisions revisited

- **fc1 deprioritization (2026-04-27)** is *champAa-specific*. On champS4 fc1 retains threats and is the cheapest hook with full concept coverage (LP gorilla 0.72 / hawk 0.58). fc1 SAE work on threats is reactivated.
- **Phase 4 architectural fix** (auxiliary threat-prediction head on fc1 during DQN training) was motivated by champAa's collapsed fc1 threat signal. champS4 already preserves threats through fc1 implicitly. **Phase 4 is no longer needed** — the unified-aux objective solved the bottleneck problem at training time.

### Rescoped sweep (Phase 2B-sweep)

With the LP ceiling known *and* a second champion (`champTa`) available at the same architecture, the real sweep is a **25-config two-champion plan** on 3 × A6000. Success criterion: SAE F1-lift ≥ 50 % of LP F1-lift on the same hook × BSP. All configs use seed = 42 (single-seed justification: Anakin σ = 0.004, Phase 2A σ ≈ 0.005 — established seed stability for TopK / BatchTopK training dynamics on `CNN_uncoupled`; expected to transfer to S4).

**Shared E/F IDs across champions** — each E0X / F0X is the same recipe on both champions, so `registry_query.py compare A B --bsps=gorillaS4 --bsps-b=gorillaTa` works for every pair, isolating training-procedure effect at fixed architecture.

| ID | hook | arch | k / θ | exp | champS4 | champTa | notes |
|---|---|---|---:|---:|:---:|:---:|---|
| A01v2 | s4.fc1 | batchtopk | k = 16 | **8** | ✓ | (= F00) | champS4 fc1 exp-fix |
| C01 | s4.conv2 | topk | k = 16 | 8 | (orig kept) | ✓ | collapse-zone probe |
| E01 | s4.conv2 | topk | k = 32 | 8 | ✓ | ✓ | mid-budget; just above C01 collapse |
| E02 | s4.conv2 | topk | k = 48 | 8 | ✓ | ✓ | curve filler |
| E03 | s4.conv2 | topk | k = 64 | 8 | ✓ | ✓ | direct A/B against D02 |
| E04 | s4.conv2 | topk | k = 96 | 8 | ✓ | ✓ | upper budget |
| E05 | s4.conv2 | batchtopk | k = 32 | 8 | ✓ | ✓ | arch cross-check |
| E06 | s4.conv2 | batchtopk | k = 64 | 8 | ✓ | ✓ | arch cross-check |
| E07 | s4.conv2 | jumprelu | t = 32 | 8 | ✓ | ✓ | min_improvement = 0.005 (was 0.02; fixes C07) |
| F00 | s4.fc1 | batchtopk | k = 16 | 8 | (= A01v2) | ✓ | champTa starts with correct exp = 8 |
| F01 | s4.fc1 | topk | k = 32 | 8 | ✓ | ✓ | LP fc1 gorilla = 0.72 — large untapped target |
| F02 | s4.fc1 | topk | k = 64 | 8 | ✓ | ✓ | mid-budget fc1 |
| F03 | s4.fc1 | batchtopk | k = 32 | 8 | ✓ | ✓ | direct A/B vs A01v2 / F00 |
| F04 | s4.fc1 | jumprelu | t = 64 | 8 | ✓ | ✓ | fc1 arch breadth; patience fix baked in |

Total: 12 champS4 + 13 champTa = 25 SAE configs + 8 LP runs.

### Pre-registered: oracle-distillation concern for champTa

champTa is trained by distilling a depth-2 minimax oracle during the self-play select step. Two plausible interpretability outcomes — pre-registered so the LP result reads as a decision, not a post-hoc rationalisation:

1. **Best case — concept distillation.** Cheapest way to match the oracle's policy across many positions is to compute the same intermediate concepts (threat positions, completability). Distillation regularises feature emergence — would mirror what champS4's unified-aux did to threats at fc1 (0.005 → 0.548). Expected signature: LP-Ta F1-lift ≥ LP-S4 on every category, biggest gains on `reframed_completable` and `threat_*`.
2. **Worst case — non-decomposable shortcut.** Network pattern-matches board configs directly to the oracle's chosen move without computing intermediate concepts. Good play, worse interpretability. Expected signature: LP-Ta competence ≥ champS4 *but* LP F1-lift on completability *drops*; threats may stay flat.

`reframed_completable` and `threat_*` are the diagnostic categories because they are exactly what depth-2 minimax computes internally. Decide which path to pursue *after* the LP lands; do not prejudge from the gameplay benchmark alone.

---

## Chapter 3 — champTa LP + 25-config sweep results (2026-05-19 afternoon)

### What ran

`commands.sh` had three parts. Outputs on disk confirm all three completed:

| Part | Block | Evidence |
|---|---|---|
| 1 | champTa data gen — competence audit, positions, gorillaTa/hawkTa labels, four `s4.{fc1,conv2}` activations | `model_competence_audit-champTa.json`, `bsp_labels-{gorillaTa,hawkTa}_*.pt` present |
| 2 | LP baseline (champTa, 8 runs) | 8 `linear_probe_*_s4.{fc1,conv2}_amalgam_ta[_random]_*.json` present |
| 3 | SAE sweep launchers (commented in script; uncommented at run time on Deep Brain) | `eval_registry.json` shows 16 champS4 × {gorillaS4, hawkS4} + 13 champTa × {gorillaTa, hawkTa} |

### Numbers [DIRECT — from competence audits, LP json, eval registry]

#### Competence audit — three champions side-by-side

| Test | champAa | champS4 | champTa |
|---|---:|---:|---:|
| A. Winning-placement acc | 0.700 | 0.787 | **0.868** |
| B. Losing-piece avoidance | 0.405 | 0.620 | **0.859** |
| C. Offered-piece sensitivity | 0.144 | 0.228 | 0.218 |
| D. Q(empty) − Q(occupied), frac positive | 0.253 | 0.119 | 0.160 |
| E. Late-game (11–15) mean entropy | 1.312 | 1.249 | **1.208** |

#### Linear-probe ceiling — `coverage_f1_lift`

| Hook | BSP set | champS4 trained | champTa trained | Δ Ta–S4 | champS4 random | champTa random |
|---|---|---:|---:|---:|---:|---:|
| s4.conv2 | gorilla | 0.678 | **0.708** | +0.030 | 0.206 | 0.168 |
| s4.conv2 | hawk    | 0.715 | **0.785** | +0.070 | 0.053 | 0.012 |
| s4.fc1   | gorilla | 0.520 | **0.595** | +0.075 | 0.105 | 0.046 |
| s4.fc1   | hawk    | 0.548 | **0.660** | +0.112 | 0.001 | 0.001 |

#### SAE sweep — top run per (champion × BSP set)

| Champion | BSP | Best run | F1-lift | coverage | MCC | FVU | L0 |
|---|---|---|---:|---:|---:|---:|---:|
| champS4 | gorillaS4 | `E05-champS4-s42-batchtopk-k32-exp8-s4.conv2` | 0.2124 | 0.411 | 0.402 | 0.047 | 32 |
| champTa | gorillaTa | `E05-champTa-s42-batchtopk-k32-exp8-s4.conv2` | **0.2135** | 0.428 | 0.402 | 0.037 | 32 |
| champS4 | hawkS4    | `F01-champS4-s42-topk-k32-exp8-s4.fc1`        | 0.1445 | 0.180 | 0.206 | 0.016 | 32 |
| champTa | hawkTa    | `F04-champTa-s42-jumprelu-t64-exp8-s4.fc1`    | **0.1724** | 0.214 | 0.246 | 0.012 | 66 |

#### Per-category breakdown — gorilla winners (both `E05 BatchTopK k=32 exp=8 s4.conv2`)

| Category | n | champS4 F1 | champS4 MCC | champTa F1 | champTa MCC | Δ F1 | Δ MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| cell_attribute | 64 | 0.694 | 0.653 | 0.690 | 0.627 | −0.004 | −0.026 |
| cell_occupancy | 16 | 0.812 | 0.723 | 0.775 | 0.681 | −0.037 | −0.042 |
| game_phase | 3 | 0.484 | 0.165 | 0.496 | 0.152 | +0.012 | −0.013 |
| global | 1 | 0.518 | 0.256 | 0.580 | 0.266 | +0.062 | +0.010 |
| offered_piece | 4 | 0.672 | 0.383 | 0.673 | 0.467 | +0.001 | +0.084 |
| threat_line | 40 | 0.071 | 0.133 | **0.113** | **0.155** | **+0.042** | **+0.022** |
| threat_square_2x2 | 36 | 0.073 | 0.140 | **0.123** | **0.171** | **+0.050** | **+0.031** |

#### Per-category breakdown — hawk winners (different per champion)

S4: `F01 topk-k32 fc1`; Ta: `F04 jumprelu-t64 fc1`.

| Category | n | champS4 F1 | champS4 MCC | champTa F1 | champTa MCC | Δ F1 | Δ MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| reframed_global | 2 | 0.530 | 0.399 | 0.668 | 0.480 | +0.138 | +0.081 |
| reframed_any_threat | 10 | 0.210 | 0.196 | 0.280 | 0.258 | +0.070 | +0.062 |
| reframed_sq_any_threat | 9 | 0.221 | 0.218 | 0.347 | 0.321 | +0.126 | +0.103 |
| reframed_count | 40 | 0.227 | 0.230 | 0.220 | 0.250 | −0.007 | +0.020 |
| reframed_sq_count | 36 | 0.258 | 0.264 | 0.297 | 0.322 | +0.039 | +0.058 |
| reframed_completable | 40 | 0.087 | 0.145 | 0.109 | 0.161 | +0.022 | +0.016 |
| reframed_sq_completable | 36 | 0.115 | 0.178 | 0.165 | 0.228 | +0.050 | +0.050 |

#### Matched-config A/B (champTa minus champS4 F1-lift)

| Config | Δ gorilla | Δ hawk |
|---|---:|---:|
| E01 topk k32 conv2 | +0.009 | +0.004 |
| E05 batchtopk k32 conv2 | +0.001 | +0.020 |
| F01 topk k32 fc1 | +0.015 | +0.022 |
| F02 topk k64 fc1 | +0.024 | +0.030 |
| F03 batchtopk k32 fc1 | +0.057 | +0.072 |
| F04 jumprelu t64 fc1 | +0.039 | +0.036 |

#### SAE / LP efficiency for champTa

| Hook | gorilla | hawk |
|---|---:|---:|
| s4.conv2 | 0.214 / 0.708 = **30 %** | 0.090 / 0.785 = **11 %** |
| s4.fc1   | 0.163 / 0.595 = **27 %** | 0.172 / 0.660 = **26 %** |

The conv2 / hawk cell is the worst by a large margin.

### Provisional interpretation [AI-REASONED PROVISIONAL ANALYSIS — read sceptically]

> A previous version of this analysis (in the same conversation) **incorrectly**
> attributed the conv2/hawk efficiency gap to a per-cell-vs-position BSP
> mismatch — that argument applies to the legacy Aa pipeline
> (`--flatten-per-cell`), not to this sweep, which uses `--flatten-position`
> throughout. The mistake is documented here as a warning to future readers:
> verify the activation collection flag before invoking that ceiling argument.

**Claim 1. champTa is a strictly better champion than champS4.**
Behavior: +8.1 pp winning placement, +23.9 pp loss avoidance, more decisive late-game entropy. Concept content: champTa wins at every hook × BSP cell, biggest deltas on hawk / fc1. SAE recovery: champTa wins every matched-config head-to-head. The pre-registered "oracle shortcut" alternative is rejected.

**Claim 2. The training signal lands hardest on threat structure at the fc1 bottleneck.**
LP gain Ta − S4 ordered by size: fc1/hawk +0.112 > fc1/gorilla +0.075 > conv2/hawk +0.070 > conv2/gorilla +0.030. fc1 was previously deprioritized (2026-04-27) assuming threats were lost there; that rule was revised to "champAa-specific" after the champS4 LP and is now strengthened: **fc1 on the unified-aux family carries threat information, and minimax distillation amplifies it.**

**Claim 3. The architecture winner transfers across training procedures.**
`E05 BatchTopK k=32 exp=8 s4.conv2` is the gorilla winner on both champions. On hawk, both champion winners are at fc1 — the *hook* is shared even though the SAE arch differs.

**Claim 4. fc1 SAEs beat conv2 SAEs on hawk, even though conv2 LP beats fc1 LP.**
The most surprising finding of the sweep. Two candidate mechanisms:

- **Sparsity allocation under base-rate skew.** Hawk threat BSPs have base rates ~1–2 %. A k = 32 SAE feature must align with rare positives. At conv2 the 512-dim activation space contains many high-base-rate cell-attribute signals competing for k slots; at fc1 the representation is denser and more pre-mixed, so a sparse feature there can land on a higher-level "threat exists" axis more easily. (Compatible with per-category numbers — fc1 wins biggest on `reframed_*_any_threat` and `reframed_global`, both position-summary axes.)
- **Optimization rather than representation.** SAE training loss is dominated by dense cell-attribute activations; rare threat signals contribute negligibly to MSE, so SAE features starve. LP doesn't suffer this because each BSP gets its own classifier.

Both predict the same fix: **concept-targeted or anchored SAEs that route capacity to rare hawk BSPs**, or significantly wider expansion at fixed k. See [`2026-05-19_concept-targeted-saes.md`](2026-05-19_concept-targeted-saes.md).

**Claim 5. The 11 % conv2/hawk efficiency is the binding SAE/LP wall now.**
It is **not** the per-cell BSP artifact. It is the ordinary SAE-budget vs LP-degrees-of-freedom gap, magnified by hawk's low base rates. Headroom is large — 2.5× current would only reach 50 % efficiency.

**Claim 6. Phase 4 (architectural fix on champAa) remains unneeded.**
champS4 already preserves threats through fc1 implicitly via the unified-aux objective; champTa does so more strongly via minimax distillation. Both are training-time interventions that obviated the proposed inference-time auxiliary head.

### Hypothesis status updates

| # | Hypothesis | Status before | Status after | Evidence |
|---|---|---|---|---|
| H8 | Threat info is spatially encoded; lost at fc1 bottleneck | CHAMPION-SPECIFIC (Aa only) | CHAMPION-SPECIFIC, stronger Ta evidence | hawkTa fc1 LP F1-lift = 0.660 (vs hawkAa fc1 = 0.005) |
| H9 | Oracle distillation distils concepts (vs non-decomposable shortcut) | OPEN — awaits LP | ✅ CONFIRMED — concept distillation | All four hawk diagnostic categories rise; F1-lift Ta > S4 at every cell |

### Pre-registered sweep gate

**Sweep success criterion: SAE F1-lift ≥ 50 % of LP F1-lift.** **FAILS** at every hook × BSP cell. Best is fc1 gorilla at ~27 %; conv2 / hawk sits at 11 %. The two-champion sweep moved the SAE numbers (champTa modestly beats champS4 everywhere) but did not narrow the LP gap. Decision: shift away from breadth at the current SAE budget toward concept-targeted / wider-expansion approaches. See [`2026-05-19_concept-targeted-saes.md`](2026-05-19_concept-targeted-saes.md).

---

## Verification protocols (open work)

### F04-Ta hawk win — patience-fix rerun

The hawk winner on champTa is `F04 jumprelu-t64-exp8 s4.fc1` at 0.1724 lift. The `*_metrics.jsonl` for this run has only ~7 rows (vs ~51 for top-k / batch-topk siblings), indicating early-stop fired ~3 k batches in. The sweep already lowered `min_improvement` from 0.02 to 0.005 to avoid the C07 truncation pattern, but jumprelu warm-up is plateau-prone and may have early-stopped before convergence even with the relaxed gate.

**Why this matters:** the F04-Ta hawk lead over the F01-S4 hawk winner (+0.028 lift) is the single piece of evidence that **fc1 SAEs can beat conv2 SAEs on hawk** under sparsity constraints. If F04-Ta's training was truncated, the true converged lift might be higher (strengthening the claim) or lower (weakening it). Either way, we need a clean number before promoting the conclusion to RESEARCH-STATUS as decided.

**Protocol:**

1. **Rerun F04-Ta with patience disabled (or `min_improvement=0.001`)** on Deep Brain. Same seed, same config, same activations. Run the full 25 k batches without early-stop.
2. **Compare metrics jsonl row count** — expected ~51 rows if it trains to completion. If the new run also hits early-stop at < 20 k batches, the threshold needs to be even smaller (or jumprelu's loss plateau is real and the original 0.1724 is the converged number).
3. **Re-evaluate against hawkTa** via `python sae_eval.py evaluate <new_ckpt> --bsps=hawkTa --force` (the `--force` flag is required to overwrite the existing registry entry).
4. **Decision rule:**
   - New lift ≥ 0.16 → fc1-on-hawk claim stands; promote to RESEARCH-STATUS as established.
   - New lift drops below conv2 winner (0.090) → fc1 advantage was a truncation artifact; reweight conv2 hawk runs and re-examine F01-S4 too (it may share the issue).
   - 0.090 < new lift < 0.16 → ambiguous; would need a seed replicate (s43) before promoting.
5. **Sanity check on champS4 side:** rerun F04-champS4 with the same fix. If S4's F04 also moves substantially, the patience knob was the dominant variable, not the training procedure — important for any future jumprelu config.

Total cost: ≤ 30 min on 1 × A6000 for both reruns (jumprelu is fast even at full 25 k). Worth doing before claiming the "fc1 > conv2 on hawk" surprise in any external writeup.

### Concept-targeted SAE direction

Design note: [`2026-05-19_concept-targeted-saes.md`](2026-05-19_concept-targeted-saes.md). Implementation deferred to a new session.

## Pointers

- Run IDs are in `eval_registry.json` keyed as `<run_id>:<bsp_set>`.
- Per-run per-feature caches: `saes/quarto/cache/{run_id}_h.pt` and `{run_id}_matching-{bsp_set}.pt` on Deep Brain (gitignored locally).
- Cross-check any claim with `python scripts/registry_query.py top --bsps=<set> --limit=20` or `... category <run_id> --bsps=<set>`.
- Competence audits use 5000 sampled positions for A/B/C and the full 275,916 amalgam positions for D/E; same seed (42) across all champions.
