# Design note — Concept-targeted SAEs to close the conv2/hawk wall

Status: design only, no implementation. Target: a new session in Phase 2C
or 2D. Triggered by [`phase-2B.md`](phase-2B.md) sweep result — 25 unsupervised
configs hit a hard SAE-budget wall on rare hawk threat concepts.

Parent: [`../../RESEARCH-STATUS.md`](../../RESEARCH-STATUS.md).

## Why [INFERENTIAL — Phase 2B sweep evidence]

The Phase 2B-sweep produced two observations the design must respect:

1. **The LP ceiling is high.** champTa LP F1-lift on conv2/hawk = 0.785, fc1/hawk = 0.660. The threat concepts *are* linearly readable from the activations; the question is whether an SAE can route its sparse capacity onto those directions.
2. **Unsupervised SAEs recover only 11–30 % of LP lift** despite spanning {top-k, batch-top-k, jumprelu} × {fc1, conv2} × {k ∈ 16..96} × {exp = 8}. Increasing k by 6× (16 → 96) on conv2 did not close the gap on hawk. Pure architectural breadth is exhausted.

Two compatible mechanisms predict the same failure mode:

- **Reconstruction-loss dominance.** SAE training minimizes MSE on the activation tensor. Dense cell-attribute and cell-occupancy signals account for ~95 % of activation variance; rare hawk threats (base rate 1–2 %) contribute essentially nothing to MSE. The SAE has no gradient reason to allocate features to threats.
- **Sparsity-budget allocation under base-rate skew.** A k = 32 feature slot is "worth" more if it lights up on 35 % of positions (cell occupancy) than on 1.5 % of positions (a specific threat). The L0 optimum gives most slots to dense concepts.

Both mechanisms predict that **biasing the SAE objective toward rare, named concepts** should close the gap. That is the unifying theme of the candidates below.

## Candidate approaches

Sorted by implementation cost (cheapest first); each is a separable experiment.

### 1. Wider expansion at fixed k

**Idea.** Hold k = 32–64 (the current winners) but increase `expansion` from 8 to {16, 32, 64}. More dictionary slots means a feature can specialise on a rare concept without sacrificing a dense one.

**What it tests.** "Not enough features" vs "wrong features". If exp = 32 closes a meaningful fraction of the gap, the SAE was just under-resourced; if it doesn't move, the objective itself is the problem and we need (3)–(5).

**Predicted outcome.** Modest improvement on conv2 hawk (best guess: 0.090 → 0.12, ~13 % LP efficiency). Phase 2A showed exp16 didn't help on conv2 at the champAa LP ceiling, but the *Ta* ceiling is much higher, so the same null may not recur.

**Cost.** Trivial. Three configs per champion, ~3 h on Deep Brain.

**Why it's the right first step.** It establishes a baseline for "how much can pure scale buy us" before introducing supervision. Without that baseline, any anchored/E2E win is ambiguous between "supervision works" and "more capacity works".

### 2. Higher k specifically on hawk-targeted runs

**Idea.** Push k to {128, 192, 256} on conv2 fc1, holding exp = 8. Direct test of the sparsity-allocation explanation: if more active features per position unlocks threat detection, the budget was binding.

**What it tests.** Same null/alt as (1) but on the active-set side rather than the dictionary side.

**Predicted outcome.** L0 = 256 will be near-dense and FVU near zero; if F1-lift on hawk doesn't move, sparsity per se was not the binding constraint — confirming that the objective needs supervision, not just bigger budgets.

**Cost.** Same as (1). Could be combined into a single 3 × 3 grid of (k, exp) on the F04 jumprelu / E05 batchtopk recipes.

### 3. Anchored / guided SAEs (concept-supervised loss term)

**Idea.** Add a small auxiliary loss that encourages designated features to fire on (and only on) specific hawk BSPs. Loose specification:

```
L_total = L_MSE + λ_sparsity · L_L0 + λ_anchor · L_anchor
L_anchor = Σ_k BCE(h_anchor[k] activations, BSP_label_k)
```

where `h_anchor[k]` is a designated subset of the SAE feature dimension (e.g., the first 76 features, one per gorilla threat BSP, with the rest left for unsupervised learning).

**What it tests.** Whether the SAE *can* represent the threat axis cleanly given gradient pressure to do so. If anchored features hit F1 close to the LP ceiling, the gap was purely allocation. If they plateau well below LP, the activation space genuinely lacks a sparse linear direction for that concept (and we'd revisit whether the LP is overfitting via its 512-dim affine freedom).

**Predicted outcome.** This should close most of the gap on conv2 hawk. The risk is that anchored features *replace* the unsupervised dictionary's natural concepts rather than augmenting them — early experiments should ablate λ_anchor across {0.01, 0.1, 1.0} to find the regime where both anchored and unsupervised features survive.

**Cost.** Moderate. Implementation: ~2 days of code (new loss term, anchor-index plumbing, eval that separates anchored vs unsupervised features). Training cost same as unsupervised SAE.

**Note from Phase 1G (2026-05-11):** the original conclusion was "anchored SAEs against `reframed_completable` would fail by construction" because champAa didn't compute completability. **That logic does not apply to champTa** — the model now uses completability (LP F1-lift = 0.523 fc1 / 0.732 conv2). Anchored SAEs on champTa hawk are well-motivated.

### 4. End-to-end (E2E) SAEs

**Idea.** Train the SAE to preserve downstream Q-head loss (Braun et al., 2024) rather than activation MSE. The objective becomes "reconstruct in a way that doesn't change the model's behavior," which weights features by how load-bearing they are for the Q heads.

**What it tests.** Whether threat features are recovered *because* the model uses them. E2E acts as a diagnostic in the opposite direction of (3): instead of forcing the SAE to find threats, E2E lets the model "vote" on which features matter.

**Predicted outcome.** On champAa (Phase 1G), E2E would have *deprioritised* threat features (champAa barely uses defensive reasoning). On champTa the prediction flips: E2E should *amplify* threat / completability features because the model demonstrably uses them (test B 86 % loss avoidance). Direct diagnostic for whether champTa's strong play routes through interpretable concepts.

**Cost.** Higher. Requires hook into `fc2_board` / `fc2_piece`, gradient-through-model on the reconstructed activations, and careful loss weighting. ~1 week of code + tuning.

### 5. Matryoshka SAEs

**Idea.** Nested dictionaries `d ⊂ 2d ⊂ 4d ⊂ 8d`, each sparse and each forced to reconstruct independently (Bussmann et al., 2024). Attacks feature absorption / splitting — the suspected cause of the BatchTopK / JumpReLU 99 %-dead pattern at conv2.

**What it tests.** Whether non-threat coverage can be improved by giving the SAE multiple resolution scales. Less directly aimed at the threat wall, more at the overall SAE/LP efficiency on dense categories.

**Predicted outcome.** Probably helps non-threat coverage (closes the 30 % efficiency on conv2 gorilla) without strongly moving threats. Useful complement to (3) / (4), not a substitute.

**Cost.** Moderate-to-high. ~1 week including the eval path that exposes per-scale coverage.

## Recommended ordering (when this work resumes)

| Step | Approach | Decision | Continue / pivot |
|---|---|---|---|
| 1 | Wider exp grid on F04 / E05 | Does pure capacity move conv2 hawk lift > 0.13? | If yes: tune exp until saturation, then stop. If no: go to step 2. |
| 2 | Higher k on conv2 / fc1 hawk | Does k = 256 move hawk lift? | If no: sparsity is not the binding constraint — the objective needs supervision. Go to step 3. |
| 3 | Anchored hawk SAEs on champTa fc1 (LP target 0.66) | Do anchored features approach LP F1 on rare threat BSPs? | If yes: validate the recipe on conv2 (LP target 0.79). If no: revisit LP overfitting hypothesis. |
| 4 | E2E SAEs on champTa fc1 | Does E2E find threat features unsupervised when the model uses them? | Diagnostic; result either way is publishable. |
| 5 | Matryoshka on top of the winner from (3) / (4) | Marginal coverage gain on non-threat categories. | Optional polish. |

## Decision gates (pre-registered for next session)

- **Promote concept-targeting to primary direction** if step 2 shows no k-driven improvement on conv2/hawk and step 3 anchored runs reach > 60 % LP efficiency on at least one hawk subcategory.
- **Abandon concept-targeting, revisit data** if step 1 *already* closes the conv2/hawk gap to > 50 % LP efficiency. Means we were under-resourced, not mis-objectived.
- **Open a new sub-investigation** if step 4 (E2E) shows that the model uses concepts the LP misses or vice versa — that would be a new mechanistic finding worth its own diary entry.

## Open questions

- *How to evaluate anchored features fairly.* If features are forced to track BSPs, "F1 on BSP" is no longer an independent measurement. Need a held-out metric — e.g., anchored-feature behavior on transformed boards (rotation / reflection invariance), or transfer to a held-out BSP not used during training.
- *Whether to use gorilla or hawk as the anchor target.* Gorilla has more concepts (164) but redundant with cell_attribute; hawk has fewer but maps directly to threats. Tentatively: hawk for the threat anchor, leave 75 % of features unsupervised for dense concepts.
- *Cross-champion comparability.* If anchored SAEs are trained on champTa, the resulting concept directory is champTa-specific. Re-running on champS4 with the same recipe gives a clean A/B that may strengthen H9 evidence or expose the limits of distillation.

## Not on this list (and why)

- **Crosscoders (conv2 ↔ fc1 transfer).** Worth doing eventually, but not until we have a working concept-targeted SAE on one hook — otherwise we'd be transferring features that don't recover the rare categories on either side.
- **Transcoders (replace fc with sparse map).** Architectural surgery; out of scope for an interpretability-side intervention.
- **Phase 4 (auxiliary training-time head).** Already retired in Phase 2B — champS4 / champTa solved the bottleneck problem at training time without inference-time changes.
