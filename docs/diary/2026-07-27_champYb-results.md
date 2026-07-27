# champYb SAE results + method refinements (2026-07-27)

Status: frozen self-contained record. Parent ledger: [`phase-3.md`](phase-3.md).
Numbers from the eval registry are `[DIRECT]`; interpretation is
`[AI-REASONED PROVISIONAL ANALYSIS]`. Cross-champion comparisons are on each
champion's OWN self-play distribution (not the unified pool), so they are
suggestive, not matched — the unified pool ([`unify_positions.py`]) is the
rigorous follow-up.

champYb = `Yb_hotChamp(3)` [hot lambda=1.0, seedB, E=10000]; S4 trunk +
train-only `fc_hot` depth-1 hot-piece (completion-threat) head. Full sweep
(43 configs) trained + evaluated on gorillaYb / hawkYb / tigerYb (43/43 each).

## 1. Cross-champion coverage [DIRECT]

Best **unsupervised** SAE (cov / MCC / F1-lift):

| basis | champS4 | champTa | champVe | champYb |
|---|---|---|---|---|
| gorilla (state) | 0.421 / 0.41 / 0.222 | 0.428 / 0.40 / 0.213 | 0.389 / 0.38 / 0.192 | **0.523 / 0.50 / 0.323** |
| tiger (agent-rel) | 0.376 / 0.20 / 0.112 | 0.449 / 0.33 / 0.158 | (see registry) | 0.414 / 0.28 / 0.165 |

Best **anchored** (supervised) SAE on tiger (cov / MCC / F1-lift):

| | champTa | champVe | champYb |
|---|---|---|---|
| I04 anchored-jumprelu | 0.547 / 0.50 / 0.257 | 0.507 / 0.45 / 0.255 | **0.774 / 0.73 / 0.525** |

Per-category on the **conjunctive threat** concepts (F1), champTa vs champYb:

| tiger category | Ta unsup F04 | Yb unsup F04 | Ta anchored I04 | Yb anchored I04 |
|---|---:|---:|---:|---:|
| line_winnable | 0.234 | 0.180 | 0.263 | **0.602** |
| square_winnable | 0.300 | 0.268 | 0.405 | **0.805** |
| offered_completing_attr | 0.343 | 0.242 | 0.590 | **0.860** |

## 2. The finding [AI-REASONED PROVISIONAL]

- **Unsupervised SAEs do NOT beat the conjunction wall on Yb.** line/square
  winnable stay ~0.18–0.27 — the same wall as Ta/Ve (even a touch lower). There
  is **no unsupervised breakthrough**.
- **Anchored extracts conjunctions far better on Yb (0.60/0.81) than Ta
  (0.26/0.41)** — the largest anchored-vs-unsupervised gap of any champion. So
  the hot-piece (completion-threat) training makes the conjunction info **more
  present / linearly available** in Yb's activations (supervision can pull it
  out), while **flat unsupervised SAEs still dilute it**.
- **Reconciling "big play gain" with "SAE saturation":** it is not uniform
  saturation. gorilla-unsupervised rose on Yb (0.39–0.43 → 0.52) and
  tiger-anchored ~doubled (0.51 → 0.77) — much of Yb's improvement *is* visible.
  What "saturates" is the **unsupervised-SAE-method ceiling** on multi-dim
  conjunctions (H10 / Dorrell: an objective-level optimum property, insensitive
  to model quality). The anchored 0.77 proves the info got *more* accessible in
  Yb; unsupervised just cannot surface it. Also, play strength != BSP
  decodability — the win-rate gains may live in value/policy computations the
  BSP menu does not probe.
- **Consequence:** Yb has the most "present-but-hidden" conjunction structure →
  the single cleanest **3C target** (close the anchored-vs-unsupervised gap with
  an *unsupervised* geometry-aware method). Do **not** headline the anchored
  numbers — the thesis pursues a generalizable unsupervised method.

## 3. Method refinements (adopt going forward)

- **Feature-BSP alignment is greedy argmax on decodability, not causality.**
  `eval.py` matches each BSP to the max-F1 feature (max-MCC also stored but not
  headlined). The matched feature can be a *spectator* while the causally-used
  feature ranks #2+ (the G11 epiphenomenality trap at the matching level).
  Fixes: (a) treat **F1-vs-MCC disagreement** as a robustness signal and prefer
  **MCC** for rare threats (base rate ~0.02); (b) export **top-K candidate
  features per BSP** (not just argmax) so the causal track can rank them;
  (c) 3A already sidesteps this by scoring **communities** (top-K by signed phi),
  not a single feature.
- **Causal ranking, not decodability ranking, is the arbiter.** In 3B-causal /
  3D, take the top-K per BSP and rank them by **intervention effect** (ablation +
  steering → targeted-move-change rate); gradient-alignment
  `cos(feature dir, decision-gradient)` is the cheap pre-screen. The causal
  order can differ from the F1 order.
- **Anchored-SAE vs LP "true performance" is causal, not decodability.** The
  LP is the unconstrained linear ceiling; the anchored SAE is slightly below it
  (cost of sparsity+reconstruction) — an uninteresting gap. The real comparison
  is the **causal effect** of the LP direction vs the anchored-SAE feature vs the
  unsupervised-SAE feature for the same concept (does clamping change play
  appropriately, or is it epiphenomenal?).
- **3A / 3C are not causal.** 3A is structural (communities / restricted-R² /
  intrinsic dim); 3C is coverage-based. Causality lives only in 3B-causal + 3D
  (ablation + steering, add-vs-remove asymmetry pre-registered).
- **Karvonen chess/othello cross-validation (benchmark alignment).** Their
  ~48–50% board-coverage on harder games is likely a *different metric* than our
  F1-lift-vs-BSPs; pin the definition first, then a comparable cross-check would
  strengthen the "validated benchmark" claim (esp. given SAEBench ranking a
  perfect oracle below trained SAEs). Read/clone to inspect the eval + SAE
  implementation; do NOT reproduce chess/othello training (off-thesis).

## 4. Operational notes

- 3A needs the `_h` code cache; a prior disk cleanup deleted the champTa/Ve/Yb
  caches, so `runners/3A-dilution.ps1` now regenerates a missing `_h` via
  `sae_eval ... --force` (plain eval **skips** already-registered runs and never
  writes `_h`). `_h` caches are 4.6–38 GB — see the disk / delete-after-eval /
  PowerShell-runner notes in `CLAUDE.md`.
- Export `_parse_run_id` fixed to recognize namespaced hooks (`s4.fc1`,
  `s4.conv2`); previously all S4-family run-ids showed `hook=null` in
  `shipped_saes.jsonl`. Re-run the export to refresh the manifest.
