# Phase 3 — Geometric concept structure (2026-06-09 → )

Status: OPEN. Founding design note (rationale, evidence, H10, gates):
[`2026-06-09_geometric-pivot.md`](2026-06-09_geometric-pivot.md).

Parent: [`../../RESEARCH-STATUS.md`](../../RESEARCH-STATUS.md).
Predecessor: [`phase-2B.md`](phase-2B.md) (closed 2026-06-09, ch. 5 —
includes the executed-in-place "Phase 2C" concept-targeting arc).
Sub-ledger: [`phase-3A.md`](phase-3A.md) — the dilution diagnostic, **CLOSED
2026-08-25**. Split out when this file passed the ~400-line cap; every 3A
chapter moved there, so this file is the live ledger for 3B / 3B-causal / 3C /
3D. Chapters below are the Phase-3-wide ones (champion results, metric policy).


## Scope

Phase 2 established *that* unsupervised SAEs hit a wall on relational
concepts and that neither capacity (Sweep H) nor target reframing alone
(tiger) removes it; anchoring buys back targeted slots only. Phase 3
asks **why**, geometrically, and **what architecture follows**:

- **H10**: the wall is geometric — conjunction/threat concepts occupy
  multi-dimensional structure that flat dictionary atoms dilute or tile;
  hierarchy co-linearity drives the measured absorption.

Working steps (gates and full method in the founding note, as revised by the
2026-07-21 litv2 reassessment chapter, now in [`phase-3A.md`](phase-3A.md)):

**Order changed 2026-09-04**: the causal step runs BEFORE ground-truth
geometry. The IDs are unchanged -- `3B.1` already means the three-term
decomposition and eight frozen diary entries reference `3B-causal` as written,
so the reorder is encoded as *order*, not as a rename. `3B-geom` is a display
alias for the geometry track, adopted because a bare "3B" is ambiguous once a
sibling runs first; entries dated before 2026-09-04 that say "3B" mean
`3B-geom`. Rationale:
[`2026-09-04_3A-close-and-3B-reorder.md`](2026-09-04_3A-close-and-3B-reorder.md) S3.

| Runs | Step | What | Gate | Status |
|:---:|---|---|---|---|
| — | 3A | Dilution diagnostic (co-firing communities, restricted-R², intrinsic dim) on champVe/champTa/**champYb** SAE caches | G-3A: diluted/tiled vs absent | ✅ **CLOSED 2026-08-25 — G-3A PASSES.** 114 cells, 12,151 concept-verdicts, 74.6% `spread`; the wall is a **disjunction** wall. Ledger: [`phase-3A.md`](phase-3A.md); report: [`2026-08-25_3A-final-report.md`](2026-08-25_3A-final-report.md) |
| **1** | **3B-causal** | Gradient-alignment screen + clamp/steer on the **top-K candidate features per BSP** (rank by causal effect, not F1-argmax); LP vs anchored-SAE vs unsup-SAE causal effect (G11 insurance) | — (informs 3C budget) | **PRE-REGISTERED 2026-09-12** — full interchange; Wave 1 = offered-piece swap (pinned vs disjunctive on the same move). Gate: DAS-1. [`2026-09-12_3B-causal-preregistration.md`](2026-09-12_3B-causal-preregistration.md). Not yet built |
| **2** | **3B** (alias `3B-geom`) | Ground-truth geometry: α/β allocation regime; κ_ms curvature; polytope + hierarchy-orthogonality; **H11 linearity-vs-decision-relevance** (LP-only) | — (always runs) | pending -- needs no new compute (alpha by construction, beta from the width sweep on disk) |
| **3** | 3C | Geometry-aware SAE variants (brief sharpened by 3A — an **aggregating readout over existing sibling atoms**): γ pre-check, hierarchical anchoring, **SASA block decoders (LEAD ARM, adopted 2026-09-04)** vs Matryoshka / H-SAE / MP-SAE vs bilinear slots, **sign-aware arm (tiger)** | G-3C: beat I04 0.255 or ≥50% threat-gap closure, 3 seeds, +centered cross-seed stability | **UNBLOCKED** (G-3A passed) |
| **4** | 3D | Causal subspace patching → move-change rate; pre-register add-vs-remove asymmetry | — | after 3C |

## Chapters

*(appended as results land; date-stamped. 3A's chapters live in
[`phase-3A.md`](phase-3A.md).)*


### 2026-09-12 -- 3B-causal pre-registered; champYb's legality filter

Full design (frozen on commit):
[`2026-09-12_3B-causal-preregistration.md`](2026-09-12_3B-causal-preregistration.md).

- **Interchange, not steering.** A dose α is a confound three ways (scale
  across latents, off-distribution drift, and negative coordinates at a
  post-ReLU hook). Interchange copies the value a representation takes on a
  real input where the concept holds, so the dose comes from the data.
- **Wave 1 = offered-piece swap on a fixed board.** It flips the 8 pinned
  `*_completable` concepts and their OR (tiger `*_winnable`) together, with the
  same target cell, so the pinned-vs-disjunctive contrast of 3A §5.1 is tested
  causally with every nuisance fixed. H-C3 predicts the top-1 latent works
  only for its own pole and the top-`knee_k` set recovers the variable.
- **champYb has not learned legality**: raw place argmax on an occupied cell
  67.2% (chance 34.2%, untrained 47.4%); the engine retries invalid moves
  rather than penalising them. Decision: proceed. Counterfactuals keep the
  legal set identical, and illegal logits are never scored.
- Disclosed before any data: the `‖W d‖` screen and its r = −0.19 with MCC,
  hence H-C4 is two-sided.


### 2026-09-04 -- 3A bookkeeping closed; plan reordered; SASA adopted

Full record:
[`2026-09-04_3A-close-and-3B-reorder.md`](2026-09-04_3A-close-and-3B-reorder.md).

- **Paper branch folded in** (fast-forward, zero divergence). Its two non-paper
  edits are corrections with one root cause: champYb's conv2 numbers were
  computed on `E05`, the degenerate dictionary 3A had *already* replaced with
  `K04` on 2026-08-17. The efficiency numbers (2026-08-14) predate that repair
  and were never propagated. Corrected: champYb `square_threat` conv2 best
  latent **0.115 -> 0.178**; fc1-over-conv2 isolation **"3-7x" -> 1.8-4.8x**
  (recomputed independently over all 12 matched state-threat cells: 1.79x to
  4.81x). One internal inconsistency inside the paper branch itself (`1.4-4.8x`
  vs `1.8-4.8x` in the same file) fixed to `1.8`.
- **The substitution is now an ARTEFACT, not a build-time patch.** The paper
  swapped the column inside `make_tables.py`; the project instead generated
  `K04-champYb-...conv2_sae-lp-efficiency.json` and moved the `E05` report to
  `superseded/`. Rebuilt paper tables are byte-identical in every reported
  field. General rule earned: *a repaired dictionary is substituted in the
  artefact, not in the script that reads it.*
- **Plan reordered: 3B-causal runs first.** Already the standing instruction in
  three places while the numbered plan said otherwise. No rename (`3B.1` is
  taken by the three-term decomposition; eight frozen entries reference
  `3B-causal`); the order is encoded as a `Runs` column plus the `3B-geom`
  display alias.
- **SASA adopted as 3C's lead arm.** `subspace-aware-sparse-autoencoders-
  effective` proves splitting is objective-driven for features of intrinsic
  dimension >= 2, and its converse (block size r >= d_i => one group is the
  global minimiser) is testable HERE and nowhere else, because `knee_k` gives
  d_i by construction. Caveat recorded: disjunction-spans-a-subspace is an
  analogy needing a test, and the paper's empirics are single-run.
- **Top-K candidate export BUILT** (plan item 7, 3B-causal's prerequisite).
  Derived from the `_matching-*.pt` caches -- no `_h`, no GPU, no cache-format
  change. Measured what reporting standard 4 had only warned about: F1-argmax
  and MCC-argmax **disagree on ~a third of concepts**, and on up to 10% the F1
  pick is outside the MCC top-16. Tests 961 -> 969.
- **k-dose ladder deferred**, not dropped: it would reopen a closed phase whose
  reports back a submitted paper, and its prevalence confound (base rate rises
  with the number of OR'd poles) is not yet designed away. It returns as the
  test of SASA's `r = d_i` prediction in 3C.
- **No `_h` pruning** until reviewer comments land and 3B-causal starts.


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


### 2026-08-11 -- per-mode coverage (a finding made and RETRACTED), minimax probe, Tier decision

Full record: [`2026-08-11_per-mode-coverage-and-minimax-probe.md`](2026-08-11_per-mode-coverage-and-minimax-probe.md).
Data-integrity audit that preceded it:
[`2026-08-11_position-dataset-integrity-audit.md`](2026-08-11_position-dataset-integrity-audit.md).

- **RETRACTED: "coverage is ~35% worse on the positions champions actually
  play".** The raw per-mode numbers looked decisive -- tiger coverage 0.167 on
  self-play vs 0.270 on random play, monotone across two hooks, two
  architectures, two champions, and both supervised and unsupervised training.
  It was a **prevalence artefact**: the base rate falls in lockstep
  (0.050 -> 0.013), and **MCC is not prevalence-invariant** (a fixed-quality
  detector loses 25% MCC over that range).
- **Corrected result:** with rows AND positives matched exactly per concept
  (5 resamples), the gap vanishes and slightly reverses --
  self-play is **+7.0%** (F04-Yb), **+8.5%** (E05-Ve), **+0.8%** (I04-Yb anchored)
  *better* than random play. Population overlap (3.5-4.4%) and game phase
  (mean 5.87 vs 5.19-5.67 pieces) were checked and ruled out. **There is no
  on-policy coverage deficit.**
- **The durable lesson is about the metric.** MCC replaces F1 but does NOT
  remove the need to match prevalence when comparing populations, champions or
  datasets. This bites the cross-champion story directly: champTa's tiger base
  rate (0.045) is ~2x champVe/champYb's (0.023). New Reporting Standard clause 5.
- **Method note:** consistency of an effect across many conditions is not
  evidence against a confound when the confound is present in every condition.
  The control that caught it (`scripts/investigate_mode_gap.py`) is cheap and
  should be standard for any two-population comparison.
- **Tier decision.** Tier 1 (per-champion, roster-independent) stays self-play +
  vs-random + random_v_random. Pairwise champion matchups AND minimax go to
  Tier 2 (the shared unified pool): a Tier-1 set that depends on the roster must
  be regenerated every time a champion is added. champTa-rebuild is therefore a
  straight correction, unblocked.
- **MinimaxBot is affordable.** `quartopy.bot.minimax_bot.MinimaxBot`
  (alpha-beta, depth 2) subclasses the pipeline's `BotAI`. Probe:
  `minimax_v_random` 0.089 s/game (0.25 h per 10k games), `minimax_v_model`
  0.528 s/game (1.47 h). Modes are file-disjoint and can run concurrently.
- **10,000 games is a budget choice, not a coverage one.** Marginal unique-
  position yield falls only ~9% over a full 10k run (9.27 -> 8.42 new/game in
  self-play) -- nowhere near saturation against a legal space of 2.07e16
  (6.7e12 after the 3,072-element symmetry group). Keep 10k for every new mode
  for UNIFORMITY; the champTa defect was a composition asymmetry.
- Tools added: `scripts/per_mode_coverage.py`, `scripts/investigate_mode_gap.py`,
  `scripts/probe_minimax_speed.py`, `scripts/validate_datasets.py` (provenance
  gate, wired into the 3A and unified-pool runners), `scripts/compute_orbit_ids.py`.


### 2026-08-11 (PM) -- metric set finalised; BSP prevalence audit

Full records: [`2026-08-11_per-mode-coverage-and-minimax-probe.md`](2026-08-11_per-mode-coverage-and-minimax-probe.md)
(retraction + prevalence), [`2026-08-11_bsp-prevalence-audit.md`](2026-08-11_bsp-prevalence-audit.md).
Definitions: [`../methods-reference.md`](../methods-reference.md) S1.

- **The metric set is frozen at three numbers**, emitted by every eval:
  `coverage_mcc` (headline), `coverage_youden_j` (Youden's J = TPR-FPR,
  prevalence-INVARIANT), `coverage_mcc_at_pref` (MCC standardised to
  **p_ref = 0.025**, stored per row). MCC-vs-J divergence reads off how much of a
  gap is base rate rather than quality; `mcc_at_pref` is the number that is
  comparable across populations. Verified: standardisation reproduces subsample
  matching to 4 decimal places without the sampling noise.
- **F1 family DEMOTED.** `registry_query.py top` now leads with MCC and J and
  shows a parenthesised (F1); F1-lift is out of the default view. Both columns
  are still computed and stored (F1 for the literature comparison at write-up,
  F1-lift so older rows stay readable). F1-lift was never a fix: it corrects the
  trivial-F1 FLOOR, not the prevalence SCALING (44% swing).
- **Threshold policy settled.** `h > 0` for our sparse dictionaries (structural
  zeros -- no choice to make); **maximise J** for any deliberate sweep or a dense
  dictionary, because maximising F1/MCC picks a DIFFERENT threshold at different
  prevalences and moves the confound upstream of the metric; **precision bar**
  for reconstruction, where a false positive corrupts the output.
- **BSP prevalence audit.** The menu spans 0.0023-0.8714 (380x). hawk's
  `completable` families sit at **0.003** (76/173 BSPs under 0.01);
  `tiger_pool_safe_count` sits at **0.748**, i.e. trivial F1 = 0.856. Two
  consequences: the banked "anchored is strong on pool counts, F1 0.83-0.88" is
  barely above its trivial baseline (its MCC 0.702 is what makes it real), and
  **the 2026-05-22 tiger-over-hawk decision compared bases whose prevalence
  differs ~7x**, so its numerical margin needs re-checking with `mcc_at_pref`
  before it appears in a chapter. The agent-relative argument for tiger is
  independent and unaffected.
- **Karvonen attribution corrected.** The three thresholds in
  `circuits/analysis.py::get_above_below_counts` (precision >= 0.95, off-rate
  <= 0.1, support >= 10) gate WHICH FEATURES become board-state classifiers; the
  reported reconstruction number is then selected by **argmax F1** over an
  activation-threshold sweep (`eval_board_reconstruction.py::print_out_results`).
  So the cross-validation should be re-pointed at our `board_reconstruction`
  (precision bar 0.9), NOT at F1-lift. Paper is in the DB as
  `sae-evaluation-metrics` (topic `sparse-autoencoders`).
- Tools added: `scripts/bsp_prevalence.py`, `scripts/choose_p_ref.py`.


### 2026-08-11 (evening) -- basis re-check, prevalence in the registry

Full record: [`2026-08-11_basis-recheck-prevalence.md`](2026-08-11_basis-recheck-prevalence.md).

- **The 2026-05-22 tiger-over-hawk margin does not reproduce.** One SAE
  (F04-champYb), one set of codes, only the framing changing. At matched
  prevalence (p_ref = 0.025) the LINE triad is gorilla 0.550 / hawk 0.552 /
  tiger 0.173, and SQUARE is gorilla 0.851 / hawk 0.794 / tiger 0.291. Raw MCC
  had gorilla ahead of hawk by +66%/+72%; **90-95% of that was prevalence**, and
  the LINE comparison inverts. tiger scores 3-4x WORSE than both -- the opposite
  direction to the 2026-05-22 conclusion.
- **Not a refutation, but the numerical justification is void.** The original
  statistic was SAE/LP *efficiency* (not reproduced here, needs LP numbers), on
  champS4 and champTa -- and champTa's dataset is quarantined. The *conceptual*
  argument for tiger (the only agent-relative, decision-upstream basis) is
  independent and stands; it is what 3B-causal needs. Re-run the efficiency
  statistic properly before citing any number.
- **Absolute quality is poor everywhere**: precision 0.07-0.52 across all three
  framings. The triads rank framings; they do not show any framing is well
  extracted.
- **Methodological catch:** a first pass selected features by J and reported
  MCC@pref. Incoherent -- at base 0.003 the J-optimal latent fires on 10% of
  positions (J = 0.85, precision = 0.07), flattering exactly the rarest families
  (hawk's). **The selection criterion must match the reported metric.**
- **A basis is a packaging convention; a CATEGORY is the analysis unit.** Within
  tiger, `line_winnable` (0.023) and `pool_safe_count` (0.748) share only a
  filename. Across bases the TRIADS (`threat_line` / `reframed_completable` /
  `tiger_line_winnable`) describe one game fact in three framings and are the
  comparison worth making.
- **Prevalence now written to the registry**: `mean/median/min/max_base_rate`,
  `frac_bsps_very_rare`, `frac_bsps_trivial_f1` -- so a reader can always tell a
  weak SAE from a rare concept. Needs a re-eval to populate older rows.

## Pointers

- Literature base: 5 papers ingested 2026-06-09 (tags in founding note);
  synthesis at papers-DB `synthesis/2026-06-08_sae-landscape.md`.
- Eval caches: `saes/quarto/cache/{run_id}_h.pt`,
  `{run_id}_matching-{bsp_set}.pt` (Deep Brain; gitignored locally).
- 3A code and artefacts: see [`phase-3A.md`](phase-3A.md) "Pointers".
- Cross-check claims: `python scripts/registry_query.py top|category|compare`.
