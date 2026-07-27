# Phase 3 — Geometric concept structure (2026-06-09 → )

Status: OPEN. Founding design note (rationale, evidence, H10, gates):
[`2026-06-09_geometric-pivot.md`](2026-06-09_geometric-pivot.md).

Parent: [`../../RESEARCH-STATUS.md`](../../RESEARCH-STATUS.md).
Predecessor: [`phase-2B.md`](phase-2B.md) (closed 2026-06-09, ch. 5 —
includes the executed-in-place "Phase 2C" concept-targeting arc).

## Scope

Phase 2 established *that* unsupervised SAEs hit a wall on relational
concepts and that neither capacity (Sweep H) nor target reframing alone
(tiger) removes it; anchoring buys back targeted slots only. Phase 3
asks **why**, geometrically, and **what architecture follows**:

- **H10**: the wall is geometric — conjunction/threat concepts occupy
  multi-dimensional structure that flat dictionary atoms dilute or tile;
  hierarchy co-linearity drives the measured absorption.

Working steps (gates and full method in the founding note, as revised by the
2026-07-21 litv2 reassessment chapter below):

| Step | What | Gate | Status |
|---|---|---|---|
| 3A | Dilution diagnostic (co-firing communities, restricted-R², intrinsic dim) on champVe/champTa/**champYb** SAE caches | G-3A: diluted/tiled vs absent | code ready + runs on Deep Brain (regens missing `_h` via `sae_eval --force`) |
| 3B | Ground-truth geometry: α/β allocation regime; κ_ms curvature; polytope + hierarchy-orthogonality; **H11 linearity-vs-decision-relevance** (LP-only) | — (always runs) | pending |
| 3B-causal | Gradient-alignment screen + clamp/steer on the **top-K candidate features per BSP** (rank by causal effect, not F1-argmax); LP vs anchored-SAE vs unsup-SAE causal effect (G11 insurance) | — (informs 3C budget) | pending |
| 3C | Geometry-aware SAE variants: γ pre-check, hierarchical anchoring, **Matryoshka-vs-H-SAE-vs-MP-SAE**, bilinear slots, **sign-aware arm (tiger)** | G-3C: beat I04 0.255 or ≥50% threat-gap closure, 3 seeds, +centered cross-seed stability | gated on 3A |
| 3D | Causal subspace patching → move-change rate; pre-register add-vs-remove asymmetry | — | after 3C |

## Chapters

*(appended as results land; date-stamped)*

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

### 2026-07-27 — champYb results + method refinements

champYb fully evaluated (43/43 x gorilla/hawk/tiger). Full record + tables:
[`2026-07-27_champYb-results.md`](2026-07-27_champYb-results.md). Headline
[AI-REASONED PROVISIONAL]:

- **Unsupervised SAEs still hit the conjunction wall on Yb** (line/square
  winnable ~0.18–0.27, same as Ta/Ve). No unsupervised breakthrough.
- **Anchored extracts conjunctions far better on Yb** (I04 line 0.60 / square
  0.81 vs Ta 0.26 / 0.41) — the largest anchored-vs-unsupervised gap of any
  champion. So the hot-piece (completion-threat) training makes the info **more
  present**; flat unsupervised SAEs still dilute it. "SAE saturation" is an
  *unsupervised-method* ceiling (H10/Dorrell), not a model-representation
  ceiling. gorilla-unsup rose on Yb (0.52) and tiger-anchored ~doubled (0.77),
  so much of the play gain *is* visible. → Yb is the cleanest 3C target.

Method refinements adopted (fold into 3B-causal / 3C / 3D):

- **Alignment is greedy argmax on decodability, not causality.** `eval.py`
  matches each BSP to the max-F1 feature; that feature may be epiphenomenal
  while the causal one ranks #2+. Adopt: prefer **MCC** for rare threats and
  treat F1-vs-MCC disagreement as a robustness flag; **export top-K candidate
  features per BSP** (not just argmax); rank the top-K by **causal effect**
  (ablation/steering, gradient-alignment pre-screen) in 3B-causal/3D. 3A already
  scores **communities** (top-K by signed phi), not a single feature.
- **Anchored-SAE-vs-LP "true performance" is causal**, not decodability (the
  decodability gap is just the cost of the SAE's sparsity constraint). Compare
  the causal effect of LP dir vs anchored-SAE feature vs unsupervised-SAE
  feature for the same concept.
- **Karvonen chess/othello** (~48–50% board-coverage): pin their metric vs our
  F1-lift first; a comparable cross-check strengthens the "validated benchmark"
  claim. Inspect their eval/SAE impl; do not reproduce their training.

Operational: `runners/3A-dilution.ps1` now covers champYb (unsup F04/E05 +
anchored I04 control) and regenerates a missing `_h` via `sae_eval --force`
(plain eval skips already-registered runs and never writes `_h`). Export
`_parse_run_id` fixed for namespaced hooks (`s4.fc1`/`s4.conv2`; were
`hook=null` in `shipped_saes.jsonl`).

## Pointers

- Literature base: 5 papers ingested 2026-06-09 (tags in founding note);
  synthesis at papers-DB `synthesis/2026-06-08_sae-landscape.md`.
- Eval caches for 3A: `saes/quarto/cache/{run_id}_h.pt`,
  `{run_id}_matching-{bsp_set}.pt` (Deep Brain; gitignored locally).
- 3A code: `lib/sae/dilution.py` (game-agnostic core), CLI
  `scripts/dilution_diagnostic.py`, runner `runners/3A-dilution.ps1` (launch
  detached via `runners/launch.ps1 3A-dilution`), tests `tests/test_dilution.py`.
  Output: `saes/quarto/analysis/{run_id}_dilution-{bsp_set}.json` +
  `3A_gate_summary.json`.
- Cross-check claims: `python scripts/registry_query.py top|category|compare`.
