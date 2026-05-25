# Plan — Anchored SAE on champTa fc1 → tigerTa

Status: **plan, no implementation yet.** Triggered by [2026-05-22_reframings-audit-tiger.md](2026-05-22_reframings-audit-tiger.md) §"Results" — tiger SAE/LP efficiency exceeds hawk by +25 pp (conv2/Ta) to +51 pp (conv2/S4), firing the pre-registered decision rule "anchored / matryoshka should target tiger, not hawk."

Parent: [`../../RESEARCH-STATUS.md`](../../RESEARCH-STATUS.md) §"Active plan" step 4 (was step 5; promoted after tiger evals landed).
Sibling design note: [2026-05-19_concept-targeted-saes.md](2026-05-19_concept-targeted-saes.md) §3 (Anchored / guided SAEs — original five-candidate design).

Sequencing note: literature review refresh (RESEARCH-STATUS step 3) is owned by the PI and runs in parallel with this plan; implementation work in this doc starts in the *next* session once the lit review lands so the choice of anchor-loss form, anchor-feature topology, and λ_anchor schedule can incorporate any 2025–2026 papers (Anthropic, DeepMind, etc.) that have done anchored / supervised SAEs since the original 2026-05-19 design.

---

## Why this cell

Headline numbers from the tiger evaluation (this batch, 2026-05-22):

| Cell | LP F1-lift | Best SAE F1-lift | Absolute gap | SAE/LP efficiency |
|---|---:|---:|---:|---:|
| champTa fc1 / tigerTa   | **0.301** | F04: 0.158 | **0.143** | 52.5 % |
| champTa conv2 / tigerTa | 0.175 | E05: 0.063 | 0.112 | 36.0 % |
| champS4 fc1 / tigerS4   | 0.179 | F04: 0.112 | 0.067 | 62.5 % |
| champS4 conv2 / tigerS4 | 0.098 | E05: 0.061 | 0.037 | 62.2 % |

S4-fc1 ceiling confirmed via `max_iter=5000` rerun (closed 2026-05-22 PM): 0.1789, unchanged from the `max_iter=1000` headline.

`champTa fc1` has the **largest absolute LP headroom** and the **highest absolute LP ceiling**, while sitting in the middle of the four cells on efficiency. It is the cell where moving the SAE toward LP earns the most coverage points per unit of representational effort. champTa is also the validated minimax-distilled champion (H9 confirmed), so any anchored win on this cell is the one most relevant for the Phase 2D writeup.

`fc1` over `conv2` for the first anchored experiment: tiger results show fc1 > conv2 for agent-relative concepts on both champions, opposite to hawk where conv2 > fc1 on Ta. Anchoring on fc1 also avoids the per-cell-vs-position semantic mismatch that bit earlier conv2 work (CLAUDE.md §"Things that have bitten").

## Anchor target — which BSPs get an anchor slot

All 36 tiger BSPs anchored, with a **two-tier loss weight** to reflect the per-category breakdown from the F04-Ta-fc1 baseline:

| Category | # BSPs | Baseline F1-lift | Anchor priority | Rationale |
|---|---:|---:|---|---|
| `tiger_square_winnable`         | 9  | 0.220 | **high** | Per-square decision concept; already moves the headline. |
| `tiger_offered_completing_attr` | 4  | 0.195 | **high** | Per-attribute decision concept; sparsely encoded by construction. |
| `tiger_line_winnable`           | 10 | 0.147 | **high** | Per-line decision concept; rare base rates (0.045). |
| `tiger_pool_winning_count`      | 4  | 0.124 | medium | High F1 already (0.71), modest lift headroom. |
| `tiger_decision_global`         | 5  | 0.114 | medium | High F1 (0.74), pool-derived. |
| `tiger_pool_safe_count`         | 4  | 0.097 | medium | High F1 (0.80), pool-derived. |

23 BSPs in the **high** tier (the per-cell / per-line / per-attribute concepts), 13 in **medium** (the pool / global aggregates). Two-tier weight schedule: `λ_high = 3·λ_medium`. Total anchored features = 36 out of `expansion · d_act = 8 · 128 = 1024` (3.5 % of the dictionary; the rest stays unsupervised).

## Architecture and base recipe

Base on **F04** since it is the champTa fc1 winner on tigerTa:

```yaml
# A01-champTa-anchored-fc1-jumprelu-t64-exp8-tigerTa-s42.yaml  (placeholder name)
experiment: A01-champTa-anchored-s42
architecture: anchored-jumprelu          # new variant, see "Implementation" below
game: quarto
hook: s4.fc1
data: data/quarto/s4.fc1_amalgam_ta_activations.pt
expansion: 8
batch_size: 4096
num_batches: 25000
lr: 0.0003
seed: 42
log_every: 500
jump_threshold: 0.001
l0_target: 64
patience: 0
min_improvement: 0.001

# anchored-specific
anchor_bsps: data/quarto/bsp_labels-tigerTa_36.pt
anchor_schema: data/quarto/bsp_schema-tiger_36.json
anchor_index_map:        # which dictionary slot anchors which BSP
  mode: prefix           # features [0..35] anchor BSPs [0..35] in schema order
anchor_loss: bce         # BCE(sigmoid(h_pre), y_bsp) or BCE on a 0/1 fired mask
anchor_lambda_high: 0.10
anchor_lambda_medium: 0.033
anchor_high_categories:
  - tiger_square_winnable
  - tiger_offered_completing_attr
  - tiger_line_winnable
```

### Hyperparameter sweep (one champion, one hook)

| Sweep axis | Values | Why |
|---|---|---|
| `λ_high` | {0.03, 0.10, 0.30, 1.0} | Find the regime where anchored features *and* unsupervised features both survive. The 2026-05-19 design note explicitly flags this as the main hyperparameter risk. |
| Base architecture | jumprelu (F04 winner), batchtopk-k16 (anakin fc1 winner) | Verify anchoring works for both top-k-family and threshold-family activations. |
| Seed | 42, 43, 44 | Always — anchored loss has more local minima than pure MSE+L0. |

Total: **4 × 2 × 3 = 24 runs.** Manageable on Deep Brain 3× A6000 in ~6 h.

### What stays fixed

- Hook (`s4.fc1`) and champion (`champTa`). The anchored experiment is the diagnostic on the cell with the highest LP headroom; we test transfer to conv2/Ta and to S4 only *after* a positive result here.
- Expansion = 8. Sweep H ruled out capacity-as-fix on this hook; running anchored at exp = 16+ would conflate "supervision helps" with "more dictionary helps." A separate exp ablation can follow if anchoring wins.
- BSP target = `tigerTa`. Hawk is *not* anchored against in this experiment — the tiger-decision-rule outcome (this entry's §"Why this cell") explicitly says to target tiger first.

## Decision gates

Pre-registered, before implementation:

| Outcome on `champTa fc1` (best λ, best seed) | Interpretation | Next action |
|---|---|---|
| anchored F1-lift ≥ 0.25 on the **23 high-tier BSPs** (vs F04 baseline 0.18 on the same subset) | The wall was allocation. Anchored loss buys back ≥ 60 % of the LP gap on the rare concepts. | Lock the recipe. Run conv2/Ta, S4 transfer. Promote anchored to Phase 2C primary. |
| anchored F1-lift ∈ [0.18, 0.25] on high-tier | Partial win — anchoring nudges but doesn't close. | Increase `λ_high` upper bound; ablate `anchor_index_map.mode` (prefix vs random vs initialized-from-LP-weights); revisit if a second sweep also plateaus. |
| anchored F1-lift unchanged or worse | Either (a) the activation space genuinely lacks sparse-linear directions for these concepts (LP wins via dense affine), or (b) anchored features destroyed unsupervised structure and overall coverage collapsed. Inspect per-category for the latter. | If (a): revisit hook (try conv2 anchored, or earlier conv layer). If (b): drop λ by 10× and retry. |
| unsupervised coverage drops > 0.05 from F04 baseline | Anchored features are *replacing* unsupervised structure, not augmenting it. | Lower λ; consider explicit hidden-orthogonality penalty between anchored and unsupervised feature subspaces. |

## Implementation work items

Ordered by dependency, not priority:

1. **`AnchoredJumpReLUSAE`** in `lib/sae/architectures.py`. Inherits from `JumpReLUSAE`; adds `anchor_bsps_buf`, `anchor_index_map_buf` as buffers; overrides `compute_loss()` to add `L_anchor = Σ_k BCE(h_k_active, y_k) · λ_k`. Register in `ARCHITECTURES`.
2. **`AnchoredBatchTopKSAE`** — same pattern. Two architectures so the anchoring is decoupled from the activation choice.
3. **YAML plumbing in `sae_train.py`.** Read `anchor_bsps`, `anchor_schema`, `anchor_index_map`, `anchor_loss`, `anchor_lambda_*`, `anchor_high_categories` from config. Resolve schema → category-to-index mapping. Pass to SAE constructor.
4. **Eval extension in `sae_eval.py`.** Add a per-feature *anchored vs unsupervised* tag to the cache. Headline coverage stays the standard F1/MCC/lift; per-category breakdown gets a new "anchored subset" row that reports lift only on the 36 anchored BSPs against the 36 anchored features (the diagonal). This is the diagnostic the decision gates need.
5. **Sanity tests.** Extend `tests/test_sae.py` with: anchored loss = 0 when `λ = 0` (must equal baseline jumprelu loss); anchored loss reproduces baseline BCE when called on a deterministic toy tensor; `anchor_index_map` mode = prefix uses features 0..N-1.

Estimated cost: 1.5 – 2 days of code + tests, then 6 h of compute, then ~1 h of eval / table writing.

## What we are *not* doing in this experiment

- **Conv2 anchored.** Deferred until fc1 anchored has a verdict — the conv2 per-cell-vs-position semantic mismatch makes that experiment ambiguous.
- **Anchoring on hawk or gorilla.** The tiger decision rule explicitly fired against this; hawk/gorilla anchoring may be revisited as a *separate* comparison once tiger anchoring has landed, to see whether the target-set mismatch was the whole story or just part of it.
- **Anchored + matryoshka combined.** Each experiment must answer one mechanism question. Matryoshka stays on the list (engineering track 2 once anchored lands).
- **E2E.** Per RESEARCH-STATUS line: starts after at least one of {anchored, matryoshka} has landed.
- **Champion comparison study.** Running anchored on champS4 *after* champTa is the right A/B for H9, not a parallel one. The single-champion-first plan keeps the experiment design clean.

## Open questions (for the lit review the user is running before next session)

- *Form of `L_anchor`.* BCE on sigmoid(`h_pre[k]`) is one option; BCE on a 0/1 fired-mask is another. Anthropic's "Towards Monosemanticity" and recent Bussmann follow-ups may have an answer. Decide before writing the loss.
- *Whether to *initialize* anchored decoder columns from the LP weight vectors.* The LP found a useable affine direction per BSP at F1-lift 0.301 on fc1; warm-starting the anchored columns there could converge faster, but risks the SAE "memorising" the LP rather than learning a sparser code.
- *Whether to anchor on `h_pre` (before activation) or `h_post` (after).* `h_post` is the natural choice for ReLU/JumpReLU (interpretable feature firing); `h_pre` gives a smoother gradient. Test both at one λ.
- *λ schedule.* Constant vs warm-up (start `λ = 0`, ramp over first 5 k steps). The 2026-05-19 design note didn't specify; the lit review should produce a default.
- *How to evaluate fairness.* Anchored features tracking their BSP is by construction — "F1 on the BSP that anchored feature `k` was trained against" is no longer an independent measurement. Hold-out metrics: (a) anchored-feature F1 on a tiger BSP held out of `anchor_index_map`; (b) transfer to gorilla / hawk BSPs not in the anchor set; (c) board-reconstruction accuracy on the unsupervised subset.

## Not in scope for this entry

- The deeper question of *why* champTa's fc1 carries agent-relative tiger information so well (better than conv2) is a representation-mechanism question for a separate diary entry — possibly worth a Phase 2D mechanistic dig once the anchored result lands.
- The probe-grade base-rate-weighted reconstruction loss in RESEARCH-STATUS step 6 stays parked. It attacks the same allocation mechanism as anchored but without supervision; if anchored works, it becomes the cheaper unsupervised alternative worth testing.
