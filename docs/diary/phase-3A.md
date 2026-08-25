# Phase 3A — Dilution diagnostic (2026-07-21 -> 2026-08-25)

Status: **CLOSED 2026-08-25. Gate G-3A PASSES; 3C proceeds.**
Final report (glossary, validity audit, every table):
[`2026-08-25_3A-final-report.md`](2026-08-25_3A-final-report.md).

Parent: [`phase-3.md`](phase-3.md). Split out of it on 2026-08-25, when the
combined ledger passed the ~400-line cap and 3A closed -- so the parent stays
the *live* ledger for 3B / 3B-causal / 3C / 3D and this file freezes.
Method + verdict rule: [`../methods-reference.md`](../methods-reference.md) S3.

## What 3A asked

For one concept and one trained SAE: **how is this concept carried in the
dictionary?** Not "how well can one feature detect it" (that is the coverage
metric) -- but whether the information is present at all, and if so whether it
is concentrated in one atom or spread across many.

**Gate G-3A**: `geometric_frac >= 0.50` over a run's threat BSPs -> 3C proceeds;
else 3C is deprioritised in favour of hooks / E2E. 3B runs regardless.

## Answer, in one paragraph

G-3A passes. Over 114 cells and **12,151 concept-verdicts, 74.6% are `spread`**
-- the information is in the code, the flat dictionary just is not presenting it
in one atom. The residual wall is **not** agent-relativity, as this ledger
believed until 2026-08-25: it is a **disjunction over attribute poles**. Within
one basis, attribute-pinned concepts are 80.1% `captured` (`knee_k` 1) while the
OR-over-four-poles category is 0.0% `captured` at *higher* recoverable
information, and `knee_k` lands on the disjunct count known by construction
(mode 4 for a 4-way OR, mode 9 for tiger's 8-way). 3C's brief is therefore an
**aggregating readout over existing clean sibling atoms**, not a manifold method.

## Chapters

*(chronological; each is date-stamped)*

### 2026-07-21 — litv2 reassessment + 3A implementation

Reassessment of the Phase 3 plan against the 2026-07-14 paper-DB synthesis v2
(~278 summaries, superseding the 2026-06-08 synthesis that founded the phase).
The 3A–3D skeleton survives; the changes are additive and are now reflected in
the working-steps table above. Full method spec for 3A:
[`2026-07-21_3A-dilution-diagnostic.md`](2026-07-21_3A-dilution-diagnostic.md).

What the v2 literature changes [AI-REASONED PROVISIONAL ANALYSIS]:

- **G11 — causal validation promoted earlier.** `adversarial-world-model-
  verification` (Balogh & Jelasity, ICLR 2026): across 24 chess LMs, high-
  accuracy board-state probes are causally epiphenomenal (probe vs next-move
  gradients near-orthogonal, cosine dist ~0.99). Our Phase 2 "SAE/LP wall"
  framing assumes LP directions reflect representations the model *uses*. New
  **3B-causal** step (gradient-alignment screen + clamp/steer spot-check) runs
  before any 3C compute. Gradient alignment alone cannot decide causality (it
  is a local, gauge-dependent, first-order measure); the **intervention is the
  discriminator** — orthogonal-gradient + intervention-effect = grad false
  negative, orthogonal + no-effect = epiphenomenal. Fold in amnesic-probing /
  INLP (Elazar, Ravfogel) as the LP-side ablation. Quarto DQNs demonstrably act
  on threats, so full epiphenomenality is unlikely; a CNN/RL result is novel
  either way (the chess negative is next-token-LM only).
- **3A reinterpreted, not changed.** `optimality-structures-sparse-dictionaries`
  (Dorrell) proves splitting/absorption are *optimum* properties on hierarchical
  data, not training artifacts (L1-vanilla derivation; TopK/JumpReLU inherit the
  qualitative prediction, not exact conditions). Prior on G-3A moves to
  diluted/tiled; an `absent` verdict becomes the surprising, paper-worthy result.
  3A validates the prediction empirically with hierarchy depth known by
  construction — no LLM setting can. **Caveat:** we have not *directly* observed
  splitting/absorption in our SAEs yet (only symptoms: threat F1 collapse, 98–99%
  dead features in Sweep H); 3A is the diagnostic that measures the mechanism.
  Community scoring also neutralises feature *duplication* (score the community,
  not "the" latent) and surfaces polysemanticity — both were undercounted by the
  single-best-F1 matching used through Phase 2.
- **3B grows two cheap measurements.** κ_ms multi-scale curvature per concept
  family (`geometric-wall-sae-scaling`, links α/β to a second scaling theory);
  and **H11** (see RESEARCH-STATUS Hypotheses). Jiang linearity is LP-only and
  part of 3B. Important caveat: Jiang's guarantee is **softmax-CE-specific**;
  our champions are value-based RL (TD/MSE), so H11 is an *extension test into
  RL* — decodability != causality, and the Aa→S4→Ta→Ve axis measures **policy
  competence**, not a CE loss.
- **3C roster refresh.** Head-to-head becomes Matryoshka-vs-H-SAE-vs-**MP-SAE**
  (`matching-pursuit-hierarchical-sae`); add a **sign-aware/bipolar arm**
  (Bi-JumpReLU) for tiger's anticorrelated me-vs-opponent pairs; add a one-line
  **γ=‖μ‖/‖σ‖ dead-feature pre-check** (`activation-outliers-feature-death`);
  gate augmented with **centered cross-seed feature overlap** (uncentered cosine
  inflates stability); never select on the sparsity–reconstruction frontier.
- **3D** pre-registers the maze-transformer add(~80–100%)-vs-remove(~0%)
  asymmetry (`transformers-use-causal-world-models`).
- **Thesis framing upgrade.** The safe path (3A+3B) reframes from "a geometry-
  aware eval" to "the validated ground-truth SAE benchmark the field lacks"
  (SAEBench headline metrics rank a perfect oracle below trained SAEs; board
  games are the one domain where the trained-vs-random control works). The
  proposed-SAE deliverable is unchanged: I04 anchored-JumpReLU is the banked
  working method (supervised, honest limits), 3C attempts an unsupervised
  improvement.

Anchored-JumpReLU I04 — the banked numbers [DIRECT, registry, tigerTa, best
seed s43]: coverage **F1 = 0.547, MCC = 0.496**, F1-lift 0.257 (3 seeds
0.252–0.257) vs best unsupervised F04 (F1 0.449, MCC 0.333, lift 0.158) — i.e.
+0.10 F1 / +0.16 MCC / +62% on the lift metric. Uneven by category: strong on
pool counts / decision_global (F1 0.83–0.88) but still weak on the target
conjunctions (`line_winnable` 0.263, `square_winnable` 0.405) even with direct
supervision; 12/36 anchored slots polysemantic. Confirms "better, not good
enough, and supervised" — a fallback, not the scalable headline.

3A is implemented and validated (synthetic planted-concept tests + a champAa
smoke run); it awaits the champTa/Ve caches on Deep Brain. Run with
`bash runner3A.sh`.


### 2026-07-27 (PM) -- 3A first run, calibration failure, rule 3A.2

3A executed on Deep Brain (12 run x BSP-set combos, champTa/Ve/Yb). Full record,
glossary, and before/after tables:
[`2026-07-27_3A-dilution-results.md`](2026-07-27_3A-dilution-results.md).

- **Gate G-3A passes, and 3C proceeds** -- every *unsupervised* run on tiger is
  0.96-1.00 geometric (diluted). Unchanged by the recalibration below.
- **But the pre-registered calibration check failed.** The method spec required
  the supervised anchored positive control to land at `captured`; `I04-champYb`
  came out 22/23 `diluted` despite community_size 2 / intrinsic_dim 1.01 /
  top_phi 0.77. Across all 12 reports `captured` fired **1 time in 488**
  concept-verdicts -- the non-geometric arm of the gate was effectively
  unreachable, so the "pass" carried no information.
- **Cause:** the `captured` branch ANDed `knee_k` (measured over the candidate
  list, inflated by ~2% noise creep in the R2 tail) with `community_size`
  (measured over the community, counting redundancy rather than necessity).
- **Fix -- rule 3A.2** (`lib/sae/dilution.py:classify`): adds a sufficient
  condition `solo_frac >= 0.70 AND (intrinsic_dim <= 2 OR knee_k <= 2)`, where
  `solo_frac` = share of all recoverable R2 carried by the single best latent
  (new first-class metric). The old condition is kept, so nothing that was
  `captured` stops being `captured`. `rule_version` is stamped into every
  report; verdicts remain a pure function of stored metrics, so
  `dilution_diagnostic.py reclassify <report>...` re-verdicts without the
  multi-GB `_h` caches. 20 tests pin the four observed shapes.
- **New finding the old rule hid:** champYb's *unsupervised* fc1 SAE has
  essentially captured the gorilla state-threats (geometric 1.00 -> **0.11**,
  68/76 captured) while champTa's stays at 1.00 diluted. The residual wall is
  therefore specific to **agent-relative (tiger) conjunctions on the strongest
  champion, unsupervised** -- a much sharper 3C target than "beat 0.255".
- **Strongest H10 evidence is continuous, not categorical:** Ta->Yb on tiger
  conjunctions, mean asymptote_r2 0.211->0.379 (line) / 0.294->0.534 (square)
  while single-feature F1 fell 0.234->0.180 / 0.300->0.268, on a target whose
  base rate halved. Information up, extraction down, same unsupervised SAE.
  Headline `solo_frac` / `intrinsic_dim` / `top_phi`, not the 4-way label.
- **Panel revised to 17 runs**: adds hawk (dilution may be basis-dependent --
  the 2026-05-22 audit deprioritised hawk as a *supervision* target, which says
  nothing about this), and adds the missing `F04-champVe` fc1 unsupervised row
  (never evaluated on tigerVe, which is why the first read could only contrast
  Ta vs Yb).
- **Open:** no random-model SAE control exists for Ta/Ve/Yb (activations do,
  SAEs do not -- ~6 short training runs); the unified 156k pool has not been
  built on Deep Brain, so cross-champion base rates are still unmatched.


### 2026-08-21 - 3A residuals closed; `hen` built; rule 3A.4

Two entries, one day. Morning: the three residuals from 2026-08-17 measured -
[`2026-08-21_3A-residuals-and-handoff.md`](2026-08-21_3A-residuals-and-handoff.md).
Afternoon: its handoff executed -
[`2026-08-21_hen-basis-and-rule-3A4.md`](2026-08-21_hen-basis-and-rule-3A4.md).

- **`solo_frac` is NOT bimodal - RETRACTED.** 69 category medians form a
  continuum (13 above 0.8, 36 below 0.4, **20 in between**). The 0.70 threshold
  discretises rather than discovers, which makes an uncertainty band required.
- **Per-concept verdicts flip 12-17% on seed alone**, against coverage-metric
  seed sd of 0.001-0.006 - thresholding amplifies seed noise ~10x. The
  aggregate G-3A tally is robust; single-seed per-concept verdicts are not
  quotable.
- **`dead_features_pct` overstates TopK deadness by 31 pp** (85.2% reported vs
  54.5% never-firing) and is exact for JumpReLU/BatchTopK. TopK dictionaries
  ARE overcomplete (3.64x); the others are not. The metric is not comparable
  across architectures.
- **`hen` built and verified**: 173 BSPs on the four NEGATIVE attribute poles,
  mirroring hawk category-for-category. `tiger == OR(hawk UNION hen)` holds with
  **zero violations** on all three champions (~290k positions each), and
  **47.8-48.7% of tiger's positives are wins hawk cannot express**. So the
  gorilla/hawk threat menu was only ever half the menu.
- **Rule 3A.4** adds a stability band over EVERY threshold `classify` crosses,
  not just the `captured` one - because all four of K04's measured seed flips
  went through the learned-signal floor with `solo_frac` never moving. It marks
  the anchored champYb positive control `[0.17, 0.96]`, i.e. undetermined.
- **Two handoff claims corrected**: the 15 `_h` encodes were NOT done (nor were
  the 4 anchored controls - the prune's rationale string described an intention,
  not its plan), and `check_sae_usable.py` exits 2 on a missing cache while the
  runner gated before regenerating. The panel runner now encodes first.

Operational: `runners/3A-dilution.ps1 -Panel [-WithHen]` runs the 76-entry panel
(54 panel + 18 hen + 4 anchored controls over 37 checkpoints) from
`scripts/build_3a_panel_runlist.py`. Not yet launched.


### 2026-08-25 - 3A CLOSED. G-3A passes; the wall is a DISJUNCTION wall

Panel (2026-08-23, 15h50m) + seed grid (2026-08-25) executed, then audited for
validity before closing. Full record, glossary, validity audit and every table:
[`2026-08-25_3A-final-report.md`](2026-08-25_3A-final-report.md).

- **G-3A PASSES.** 114 cells (54 panel + 18 hen + 38 seed-grid + 4 anchored
  controls), **12,151 concept-verdicts: 74.6% `spread` / 18.6% `captured` /
  6.8% `absent`**; 88 cells `3C-proceeds`, 0 provisional, 100% random-control
  coverage. On the 3C target cell (champYb / `s4.fc1` / tiger) **nine**
  unsupervised dictionaries - 3 architectures, 4 expansions, 3 seeds - agree at
  `geometric_frac` 0.96-1.00 with **zero undecided concepts**, on a target whose
  information content is **14.6x** the learned-signal floor.
- **THE FINDING: the wall is a DISJUNCTION wall, not an agent-relativity wall.**
  Within the hawk basis alone (never agent-relative), on champYb fc1:
  attribute-**pinned** concepts are **80.1% captured** (`solo_frac` 0.928,
  `knee_k` 1) while hawk's own `*_any_threat` - an OR over the four attribute
  poles - is **0.0% captured** (`solo_frac` 0.311, `knee_k` 4) **at higher
  recoverable information** (R2 0.741 vs 0.530). Decomposed: agent-relativity
  costs ~9 pp of capture, the disjunction costs ~70 pp. This **corrects** the
  standing "the wall is specific to agent-relative (tiger) conjunctions" - tiger
  is spread *because* every tiger threat BSP is a disjunction.
- **The mechanism is visible, not inferred.** `knee_k` lands on the disjunct
  count known by construction: pinned -> mode **1**; `*_any_threat` (4 poles) ->
  mode **4**, 57.5% at <=4; tiger `*_winnable` (8 poles, since
  `tiger == OR(hawk U hen)`) -> mode **9**, median 11. `hen`'s contribution to
  the phase is exactly this: it makes tiger's disjunct count 8 rather than 4,
  turning the distribution into a quantitative prediction. Precondition: the
  signature appears only where atoms exist at all (champYb/fc1).
- **Validity audit, five checks.** (1) A **selection-aware null** (permute
  labels BEFORE candidate selection, then run the whole pipeline) returns
  **R2 = -0.001**, so the top-64-of-alive selection buys **zero** held-out R2
  from noise - the headline number is honest and the random-model floor's
  0.03-0.05 is real architectural prior, not double-dipping. (2) The controls
  have *smaller* alive pools (120/190) than the runs they gate (173-690), so the
  floor is not over-strong. (3) **45.9%** of concept-instances sit below
  `min_positives`; measured direction pushes them toward `absent`, i.e.
  conservative for the gate - but `methods-reference` claimed a provisional flag
  that was never implemented. (4) Panel selection on threat-family MCC biases
  `geometric_frac` **down**, so the conclusion survives a panel of the best
  dictionaries. (5) **Defect:** all 114 reports were stamped `rule_version 3A.5`
  while banded at 3A.4's width, because the CLI hard-coded the old constants.
  Point verdicts unchanged; `undecided` **23.3% -> 11.5%** (measured flip rate
  **11.4%**), straddling cells **15 -> 6**. Fixed, reclassified, 3 tests added.
- **Stability.** Measured cross-seed flip rate 11.4% over 16 cells; **23 of 24
  panel cells agree** across their three conditions (median within-cell
  `geometric_frac` range **0.02**). The single disagreement, all 6 straddling
  cells and the worst seed flip (30.4%) are **the same cell**: champYb / conv2 /
  tiger, where real R2 0.05-0.07 sits on a random floor of 0.031-0.046 with
  `random_margin = 0.02`. Its gate verdict is **not quotable**; the measurement
  is (1.6x the floor - conv2 does not usefully carry these concepts).
- **Handoff sharpened.** 3C's brief is an **aggregating readout over existing
  sibling atoms** targeted at disjunctive concepts on champYb/`s4.fc1` - the
  atoms are already clean (`knee_k` 1, `solo_frac` ~0.93); what is missing is an
  OR over them. Favours the hierarchical arm (Matryoshka / H-SAE / MP-SAE) over
  the manifold/bilinear arm. 3B-causal still runs first (G11).


## Pointers

- 3A code: `lib/sae/dilution.py` (game-agnostic core), CLI
  `scripts/dilution_diagnostic.py`, panel selection `scripts/select_3a_panel.py`
  + `scripts/build_3a_panel_runlist.py`, stability `scripts/verdict_stability.py`,
  summary `scripts/summarize_3a_gate.py`, runner `runners/3A-dilution.ps1`
  (launch detached via `runners/launch.ps1 3A-dilution -Panel -WithHen`),
  tests `tests/test_dilution.py`.
- Artefacts: `saes/quarto/analysis/{run_id}_dilution-{bsp_set}.json` (114
  live, 16 in `superseded/`), `3A_gate_summary.json`, `3A_panel.json`,
  `3A_panel_runlist.json`, `3A_verdict_stability.json`.
- Eval caches: `saes/quarto/cache/{run_id}_h.pt`,
  `{run_id}_matching-{bsp_set}.pt` (Deep Brain; gitignored locally).
- Every rule constant lives in `lib.sae.dilution.DilutionConfig`; verdicts are a
  pure function of the stored metrics, so a threshold change is a `reclassify`,
  never a re-run.
- Cross-check claims: `python scripts/registry_query.py top|category|compare`.
