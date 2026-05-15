# Research Directions

**Audience:** subagents planning next experiments on the Quarto interpretability project.
**Status:** written 2026-05-11; refreshed 2026-05-15 after the champS4 model and unified-registry infrastructure landed.
**Source of truth for project state:** `RESEARCH-STATUS.md`, `Quarto-specifications.md`.

## Where we are

Settled empirical findings (do not re-litigate):

1. **fc1 is well-saturated for unsupervised SAEs** (Anakin sweep σ ≈ 0.004 across architectures). BatchTopK k=16 exp=8 is the fc1 reference winner.
2. **conv2 ranks similarly** (C01 topk-k16 exp8, F1-lift 0.155 on gorilla). The trained-vs-random gap is ≈ 0.10 F1-lift on gorilla; ≈ 0.06 on hawk.
3. **Threat-count BSPs** (`reframed_count`, `reframed_sq_count`) get partial coverage (~F1-lift 0.12). **Threat-completability BSPs** (`reframed_completable`, `reframed_sq_completable`) sit at noise floor (~0.04).
4. **The behavioural audit (Phase 1G) explains #3:** the trained agent recognises immediate wins (Test A: 0.70 vs random 0.22) but does NOT reason about what piece it gives the opponent (Test B: 0.41 vs random 0.43). The completability concept is not represented in the model.
5. **Hawk reframing helps SAEs ~2×, not 14× like linear probes.** The reframed BSPs are still better targets than gorilla `threat_line` for any future SAE work on attack-side threats.
6. **Random-network SAEs are non-trivial baselines.** A random conv stack passes most BSP information through (random-features regime). The right learning metric is the *learned-gap fraction*
   `(SAE_trained − SAE_random) / (LP_trained − LP_random)`, not raw coverage.
7. **A second champion exists (2026-05-14).** `champS4` (unified-aux autoreg CNN, fc1=512) wins ~60% head-to-head vs `champAa` and is now the second target substrate. conv2 still has `d_act=32` in both, so per-cell conv2 SAEs are directly comparable; fc1 dimensions differ (128 vs 512) and only same-champion comparisons are meaningful there. See `Quarto-specifications.md` for the champion tags and artifact-naming policy.
8. **One eval registry, two champions (2026-05-14).** All champion SAEs write into `saes/quarto/eval_registry.json` with keys `{run_id}:{bsp_set}`. Distribution-specific BSPs use distinct animal names (`gorilla`/`hawk` for champAa, `gorillaS4`/`hawkS4` for champS4). Cross-champion comparisons use `registry_query compare <runA> <runB> --bsps=<setA> --bsps-b=<setB>`. This makes "same SAE recipe on two models" a single-command experiment.

What this means: **the most productive directions are no longer "more unsupervised SAE variants on the current model."** Phase 2A was the right place to stop on that branch. The new question is whether the Phase 1G asymmetry (attack recognition without defence) is a property of champAa specifically or of self-play DQN training in general — champS4 is the first cheap test of that.

## Recommended next directions (prioritised)

### D1 — Architectural intervention on the model itself (now a 2-step path)

**D1a (immediate, ~3 days): characterise champS4 before training anything new.**
champS4 was trained for unrelated reasons but it is a stronger player than champAa. Before committing a week to T1 (aux threat head), run the existing pipeline on champS4 and ask the *same* questions:

- Does champS4 pass Phase 1G Test B (defence recognition)? Re-run `scripts/model_competence_audit.py` against the new checkpoint.
- Linear-probe baseline on `s4.fc1` and `s4.conv2` for `gorillaS4` and `hawkS4`. If LP-on-fc1 for `reframed_completable` is materially above champAa's noise floor, the bottleneck problem partially solved itself.
- One unsupervised SAE on each S4 hook (mirror C01 + A01 recipes via `configs/champS4/`). Compare `learned-gap fraction` against champAa using `registry_query compare … --bsps-b=…`.

**D1b (only if D1a is inconclusive): T1 from `training-recommendations.md`.** Aux threat head on whichever champion has the cleaner architecture for it. The original D1 cost/risk analysis still applies.

**Cost.** D1a ~3 days (data is already partially in flight via `commands.sh`). D1b ~1 week if launched.

**Risk.** Low. D1a is pure measurement; it cannot regress the project. The worst case is "champS4 looks like champAa" which is itself a finding (architecture / aux-loss alone don't fix the credit-assignment problem) and motivates D1b.

**Success criterion.** D1a: any of (a) Test B above random by >0.10, (b) LP `reframed_completable` on `s4.fc1` > 0.30, or (c) S4 SAE F1-lift on completability BSPs > 0.10. Hitting any one re-routes the dissertation narrative away from "the model can't represent defence" toward "the *original* model couldn't."

### D2 — End-to-end SAE on the *current* model (diagnostic, parallel to D1)

**Why.** Originally proposed (Braun et al. 2024) as an architecture comparison. After Phase 1G, the meaningful question is repositioned: *"once we project conv2 onto the directions that actually drive the Q-heads, what fraction of variance is the immediate-win check vs everything else?"* E2E SAEs train against downstream-task loss rather than activation MSE, so they explicitly weight features by Q-head importance.

**Predicted outcome.** Win-detection features will dominate; defensive features will get even lower coverage than under vanilla SAEs. This *strengthens* the attack-recognition / defence-blindness asymmetry as a measurable phenomenon, not a model-bug anecdote.

**Implementation cost.** Moderate. Add a forward hook on `fc2_board` / `fc2_piece`; modify the SAE loss to include `MSE(model_output_with_reconstruction, model_output_clean)`. ~3 days.

**Run on both champions.** Same loss, two models. The unified registry makes the comparison a single `registry_query compare` call. If champS4 already routes more threat information to the Q-heads (D1a), E2E will surface it more sharply there.

**Compatible with D1.** Both can run; results are most informative side-by-side.

### D3 — Anchored / guided SAEs against the surviving threat targets

**Why.** Unsupervised SAEs recovered `reframed_count` at F1-lift 0.12. That's not nothing — it's evidence the model encodes threat counts but unsupervised dictionaries waste capacity on cell-attribute features. A guided SAE (Marks et al., dictionary-learning with a small set of anchor BSPs in the dictionary basis) should be able to push count coverage to ≥0.30 lift.

**Do NOT pursue against `reframed_completable`.** The concept is not represented (D2 will confirm). Anchoring against an absent concept gives garbage features.

**Cost.** Low–moderate. ~3 days for the architecture, ~2 days for the sweep. Mirror the sweep on champS4 against `gorillaS4`/`hawkS4` (incremental cost: hours, not days, given `commands.sh`).

**Dependence.** Independent of D1b. Best read after D1a: anchor against whichever champion's LP shows the count concept most cleanly.

### D4 — Cell-relative BSPs for per-cell conv2 SAEs

**Why.** Phase-2A noted that training SAEs on per-cell conv2 activations (shape `[N*16, 32]`) against position-level BSPs caps F1 at ≈0.118 by construction — a perfect cell-specific feature only fires for 1/16 of the matching positions. The fix is a new BSP set where labels are *relative to the processed cell* ("is THIS cell occupied by a tall piece," "is THIS cell on a 3-in-a-row line").

**Cost.** Moderate. Requires a new `scripts/compute_bsp_labels.py` mode plus a new animal-themed BSP set (e.g. `lemur` for cell-relative). ~4 days of work + one conv2-per-cell sweep.

**Value.** Conv2 has 16 spatial positions; per-cell SAEs may give 16× sample efficiency on cell-specific features. This is the cleanest way to test "is conv2 building per-cell features the SAE is averaging away."

**Risk.** Medium. The reframing requires care to keep BSP definitions consistent with `quartopy` cell indexing.

### D5 — Causal validation of the cell-occupancy and attribute features

**Why.** Headline coverage is high for `cell_occupancy` (F1=0.80, MCC=0.71) and `cell_attribute` (F1=0.54, lift=0.25). Phase 2A's reporting standard requires causal verification before *any* interpretability claim. We've never done it on Quarto.

**Method.** Activation patching: zero or swap a candidate "cell (r,c) is occupied" SAE feature in conv2; measure the change in `qav_board[r*4+c]` and the per-cell BSP label flip rate.

**Cost.** Low. ~2 days. Reuses C01 as the test SAE on champAa; mirror on the champS4 conv2 winner once D1a's S4 SAEs exist (negligible extra cost — same script, different checkpoint).

**Why it matters.** This is the smallest experiment that converts the existing high-coverage numbers from correlational to causal. Without it the Quarto chapter has *no* causal claim at all. Replicating the patch result on champS4 also tests whether the cell-occupancy circuit is *architecture-invariant* (cheap robustness check).

### D6 — Cross-game replication (Othello, Tic-tac-toe)

**Status (2026-05-15).** Still not implemented at the game-module level: `scripts/games/` contains only `quarto.py` and `quarto_s4.py`. Othello and tic-tac-toe data/model directories exist as placeholders.

**Why.** The attack/defence asymmetry hypothesis (Phase 1G) is much stronger if it replicates: any self-play DQN trained on a two-player perfect-information game will exhibit it, given similar credit-assignment dynamics. The champAa↔champS4 comparison (D1a) is a weaker, *intra-game* version of this test; a positive D1a divergence would *raise* the value of D6 (we'd want to know how broadly the asymmetry generalises).

**Cost.** High. Each new game needs: model architecture, training run, position generation, BSP definitions, and the audit adapted. ~3 weeks per game.

**When.** Only after D1 has resolved (both D1a and, if launched, D1b) — we want one clean Quarto narrative across two champions before generalising to new games.

## Deprioritised / do NOT pursue

- **More unsupervised SAE architectures on the current fc1.** Anakin σ = 0.004 settled this. (already documented as deprioritised, restated for emphasis)
- **Larger expansion factors on conv2.** Dead-features pct at exp=8 is already 60–87%; exp=16 didn't help meaningfully.
- **Matryoshka SAEs on the current model.** Solves feature absorption; the missing concepts here aren't absorbed, they're absent. Reconsider after D1 succeeds.
- **Tuning seeds for the F-series.** Phase 2A F0x results showed seed σ < 0.01 on F1-lift. We have enough seeds.
- **Direct anchored SAEs against `reframed_completable`.** Anchoring an absent concept (see D3).

## Sequencing plan

```
Week 1:    D1a (champS4 audit + LP + 1 SAE per hook)     ── gating decision
Week 1-2:  D5  (causal patching on C01)                  ┐ parallel, both champions
Week 1-2:  D2  (E2E SAE)                                 ┘ once D1a S4 data exists
Week 2-3:  D1b (T1 aux-threat training) ONLY IF D1a is inconclusive
Week 3:    Evaluate whichever D1b model trained; cross-champion compare
Week 4-5:  D3  (guided SAEs on count BSPs) on best champion
Week 6-7:  D4  (cell-relative BSPs + per-cell conv2 SAE) on both champions
Week 8+:   D6  (cross-game replication) — only after a clean two-champion story
```

This plan answers, in order: *does the second model already partially fix the bottleneck? are the existing features causal? does the bottleneck show in E2E? if not fixed for free, can we fix it by training? can guided SAEs reach the LP ceiling? do per-cell SAEs surface new structure? does it all replicate to other games?*

## Decision criteria for course-correction

After Week 1 (D1a evaluation), branch:

- **champS4 passes D1a success criterion (any of Test B, LP completability, S4 SAE lift) → skip D1b.** Re-route narrative to "architecture/aux-loss choice already exposes defence concepts; here is the SAE-level evidence." Continue D2/D3/D4/D5 on champS4 (with champAa as the contrast baseline).
- **champS4 fails D1a but is materially better than champAa on something measurable** → narrative shifts to "partial recovery via architecture; full recovery needs explicit threat supervision." Launch D1b (T1) on champS4.
- **champS4 looks like champAa on every D1a metric** → strong evidence the failure is credit-assignment, not architecture. Launch D1b on champAa (cheaper, more established pipeline) and consider T2 (reward shaping) in parallel.

After D1b (if launched), the original Week-3 branch applies to whichever model was trained:

- **Trained model passes thresholds → continue all of D3/D4/D5/D6 on it.**
- **Trained model fails Test B but improves fc1 LP** → "preservation through bottleneck possible but not sufficient." Continue D3 on new model; add T2/T3 from `training-recommendations.md`.
- **Trained model fails everything** → credit-assignment problem requires reward shaping (T2). Pause D3/D4. Spend one week on T2 before deciding whether to write up the negative result.

## What success looks like (for the dissertation)

A minimum publishable Quarto chapter contains:

1. The Phase 1F finding: threat info is present at conv2, lost at fc1.
2. The Phase 2A finding: 33 unsupervised SAEs plateau across architectures.
3. The Phase 1G finding: behavioural audit shows attack-recognition without defence on champAa.
4. A two-champion comparison (D1a): does the asymmetry replicate on champS4, partially recover, or vanish?
5. EITHER (D1a/D1b success) a model that exposes threats to generic SAEs — *interpretability via architectural choice or auxiliary supervision* — OR (sustained failure) a clean characterisation of why self-play DQN agents systematically fail to be interpretable on asymmetric concepts across two architectures.
6. (D5) at least one causally-verified non-threat feature on at least one champion, so the chapter has a positive interpretability claim regardless of #5's outcome.

Any of the #5 endings paired with #4 and #6 is a complete contribution. The current results without #4–#6 are *only* a literature review plus a negative result; with them, the chapter has methods development, a robustness comparison across architectures, and a substantive empirical claim.
