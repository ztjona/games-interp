# Methods & metrics reference

> **Role in the doc structure:** this is the **stable reference** for *how a
> number is computed* — definitions, ranges, ideal values, estimators, and the
> hyperparameters that scale them. It is the answer to "what does this column
> mean?", asked once, permanently.
>
> **It contains no results.** Results live in `docs/diary/`; the running
> narrative lives in `docs/diary/phase-N.md`; the headline ledger is
> `RESEARCH-STATUS.md`. When a diary entry needs a definition it **links here**
> rather than restating it, so a definition can never drift between documents.
>
> **Pruning:** update in place when a *method* changes; bump the rule version
> where one exists. Never append dated entries.

---

## 1. The metric policy

**MCC is the headline metric, reported beside Youden's J. The F1 family is demoted:**
F1 is retained only for comparison with the published literature at write-up, F1-lift
only so older registry rows stay readable, and neither is used to select, rank, gate
or conclude or appears in a headline table.

| | formula | range | base-rate sensitive? |
|---|---|---|---|
| **MCC** (= phi for 2x2) | `(TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))` | −1…1, 0 = chance | **Weakly** — 0 for any constant predictor, but a fixed-quality detector still loses ~25% MCC from p=0.050 to p=0.013. Match prevalence across populations. |
| F1 | `2PR/(P+R)` | 0…1 | **Yes**, through precision |
| F1-lift | `F1 − 2p/(1+p)`, clipped at 0 | 0…1 | partially corrected, still not comparable |

**MCC is not a licence to skip prevalence matching.** For sensitivity $a$, specificity $b$, prevalence $p$: `MCC = sqrt(p(1-p))(a+b-1) / sqrt([ap+(1-b)(1-p)][b(1-p)+(1-a)p])`, which declines with $p$. Comparing two populations (opponent modes, champions, datasets) at different base rates requires matching the number of positives and negatives — see `scripts/investigate_mode_gap.py`. Skipping this produced, and then retracted, a "coverage is 38% worse on-policy" finding on 2026-08-11.

Why F1 is worse still: on `tiger_line_winnable`, the conv2 SAEs of champTa
and champVe score F1 0.157 vs 0.104 — a 34% drop that looks like champVe being
worse. Their MCC is **0.140 vs 0.140**, identical. The entire difference was the
base rate (0.05 vs 0.02). Any conclusion drawn from the F1 pair would have been
an artefact.

`2p/(1+p)` is the F1 of a classifier that says "always true": at base rate 0.5 it
is **0.667**, which is why `offered_piece` F1 ≈ 0.667 is a metrics artefact and
not signal.

### 1.1 Prevalence audit of the metric set

**"Prevalence" and "base rate" are the same quantity** — `P(concept = TRUE)` in
the population being scored. (Epidemiology says prevalence, decision theory says
base rate, ML says class prior.) The one nuance that matters: it is a property of
the **population you sampled**, not of the concept, so it changes when the
population changes — which is exactly how it becomes a confound.

Hold a classifier's quality FIXED (TPR = 0.50, TNR = 0.99) and vary only
prevalence over p in [0.005, 0.100]. Anything that moves is prevalence-dependent.
Reproduce with `python scripts/prevalence_audit.py`.

| metric | p=0.100 | p=0.050 | p=0.020 | p=0.005 | swing | verdict |
|---|---:|---:|---:|---:|---:|---|
| precision | 0.848 | 0.725 | 0.505 | 0.201 | 76% | dependent |
| **R2** (point-biserial²) | 0.389 | 0.342 | 0.243 | 0.097 | **75%** | dependent |
| F1 | 0.629 | 0.592 | 0.503 | 0.287 | 54% | dependent |
| **MCC** | 0.624 | 0.585 | 0.492 | 0.312 | **50%** | dependent |
| **F1-lift** | 0.447 | 0.497 | 0.463 | 0.277 | **44%** | dependent |
| recall (TPR) | 0.500 | 0.500 | 0.500 | 0.500 | 0% | **INVARIANT** |
| specificity (TNR) | 0.990 | 0.990 | 0.990 | 0.990 | 0% | **INVARIANT** |
| **Youden J** (TPR−FPR) | 0.490 | 0.490 | 0.490 | 0.490 | 0% | **INVARIANT** |
| balanced accuracy | 0.745 | 0.745 | 0.745 | 0.745 | 0% | **INVARIANT** |

**F1-lift is NOT prevalence-invariant.** Subtracting the trivial baseline
`2p/(1+p)` corrects the *floor*, not the *scaling*.

**The frozen metric set is THREE numbers**, emitted by every eval:

| key | what it is | use it for |
|---|---|---|
| `coverage_mcc` | raw MCC on this population | what the SAE actually achieves here |
| `coverage_youden_j` | Youden's J = TPR − FPR | prevalence-INVARIANT sanity read |
| `coverage_mcc_at_pref` | MCC standardised to **p_ref = 0.025** (stored per row) | **the only one safe to compare across populations / champions** |

**Report MCC and Youden's J together.** J is exactly the `(a + b − 1)` factor in
the MCC numerator — MCC with the prevalence term stripped out — so
`MCC = J · sqrt(p(1−p)) / sqrt(P(pred+)·P(pred−))`, and **MCC-vs-J divergence is
a direct read on how much of a difference is prevalence rather than quality**.
Emitted as `coverage_youden_j` (needs a re-eval to populate on older entries).

⚠️ **J alone is misleading in the opposite direction**: a latent firing on half
the dataset with TPR 0.9 scores J = 0.4 while being useless at p = 0.02
(precision ≈ 3.5%). The *pair* is what is informative.

### 1.2 Three ways to make a cross-population comparison fair

| method | how | cost |
|---|---|---|
| **Standardise** (cheapest) | MCC is a deterministic function of (TPR, TNR, p), so re-express each feature's MCC at a canonical `p_ref`. No rows discarded, no sampling noise. | free |
| **Subsample-match** | Draw the same number of positives and negatives from both populations, repeat, average. Exact and non-parametric. | discards data, adds sampling noise |
| **Use an invariant metric** | Report Youden's J / TPR+TNR and sidestep matching. | J's blind spot above |

`scripts/investigate_mode_gap.py` implements subsample-matching;
`scripts/prevalence_audit.py --ref` demonstrates standardisation.

### 1.3 Threshold selection: when a threshold must be chosen, maximise J

Where this arises: TopK / BatchTopK / JumpReLU emit **exact structural zeros**,
so `h > 0` is the architecture's own on/off boundary, not a hyperparameter we
picked. **No sweep is needed for any current run.** It only matters for a dense
(Vanilla / Gated) dictionary, or if we deliberately sweep an activation
threshold.

When a threshold must be chosen: **maximise Youden's J** (the classical *Youden
index*, the ROC-optimal operating point). The reason is the confound moving
upstream: maximising **F1 or MCC selects a DIFFERENT threshold at different
prevalences**, so the operating point itself becomes population-dependent before
any metric is computed. J-optimal thresholds are stable across populations by
construction.

**One exception -- reconstruction.** J weights sensitivity and specificity
equally regardless of prevalence, so at p = 0.02 the J-optimal threshold is
liberal: a feature that fires far too often, with precision in the low single
digits. Where a false positive corrupts the output -- board reconstruction --
use a **precision bar** instead (`compute_board_reconstruction` uses
`precision_threshold = 0.9`).

| use | criterion |
|---|---|
| our SAEs, feature-BSP matching | `h > 0` (structural zeros; no choice to make) |
| any deliberate threshold sweep, or a dense dictionary | **maximise J** |
| board reconstruction | **precision bar** |
| replicating Karvonen for the cross-check | their criterion (argmax F1 over the sweep), for comparability only |

### 1.4 Choosing p_ref for standardised MCC

`p_ref` is a substantive choice: the ranking of two detectors *can* flip with it
(a TPR 0.50 / TNR 0.990 detector beats a TPR 0.80 / TNR 0.950 one below p ~ 0.05
and loses above it). Chosen on evidence with `scripts/choose_p_ref.py`:

- **Rank stability** -- over five real runs the ordering was **identical at every
  grid point from p_ref = 0.002 to 0.100 (zero flips)**. For our current spread
  of run quality, p_ref does not change *which* run wins.
- **Regime relevance** -- observed tiger-conjunction base rates: min 0.0186,
  median 0.0233, max 0.0499 (IQR 0.0220-0.0254).
- **Extrapolation distance** -- staying inside [0.0186, 0.0499] avoids
  standardising beyond any measured operating point.

**p_ref = 0.025** (nearest grid point to the median observed base rate; 0.02 is
inside the same stable band and equally defensible). Store `p_ref` in every
registry row so a standardised number is never ambiguous.

⚠️ Rank stability held for *these* runs, which are well separated in quality.
Two closely-matched runs can still swap, so quote `p_ref` whenever a
standardised MCC is compared.

### 1.5 Prevalence and the 3A metrics

| 3A metric | swing over p | safe to compare across populations? |
|---|---:|---|
| `asymptote_r2` | **86%** (fixed-quality planted latent) | **No** — match or standardise first |
| `solo_frac` | 1% on a monosemantic concept, **28% on a diluted one** | **Only away from the threshold** |
| `intrinsic_dim` | **0–1%** | **Yes** |
| `top_phi` (= MCC) | 50% | No |

The verdict rule (3A.2) leans on `solo_frac` and `intrinsic_dim`. `intrinsic_dim`
is robust; `solo_frac` is robust for concentrated concepts but drifts ~28% for
diluted ones, so **a concept sitting near the `captured_solo_frac = 0.70`
boundary can flip verdict on prevalence alone**. Verdicts far from the boundary
(the current situation — unsupervised runs sit at 0.16–0.42) are safe. Treat
near-threshold verdicts as prevalence-sensitive and check them explicitly.

**Base rate** = fraction of positions where a BSP is true. Note the distinction
that matters: at N = 296,045 a base rate of 0.02 still gives ~5,900 positives, so
this is **not** a sample-size problem — estimates are well powered. It is a
*metric comparability* problem, plus a *subsample* problem inside 3A (below).

## 2. BSP vocabulary

| term | meaning |
|---|---|
| **BSP** | Board State Property — a hand-written ground-truth binary fact about a position; the answer key an SAE feature is scored against. |
| **basis** | The concept menu: `gorilla` (164, state-only), `hawk` (173, Nanda-style threat recount), `tiger` (36, agent-relative), `fox` (cells/phase). |
| **suffix** | Which position distribution the labels were computed on. `tigerYb` = tiger concepts on champYb's self-play positions. A numeric suffix (`gorilla156k`) = the unified cross-champion pool. |
| **category** | A family of BSPs inside a basis, e.g. `tiger_line_winnable`. **A category is not one concept** — it is 10 independent BSPs, each with its own MCC. Always report the within-category spread; see §5. |
| **line / square** | The two winning geometries: a line is a row, column or diagonal (10 per board); a square is a 2x2 block (9 per board). |
| **conjunction** | A concept true only when several conditions hold together (three-in-a-line AND shared attribute AND holding the completing piece). The concepts H10 is about. |

**Prevalence spans a 380x range across the menu** (0.0023 to 0.8714 on champYb),
and both extremes distort metrics:

| basis | rarest family | median prevalence | commonest family | median prevalence |
|---|---|---:|---|---:|
| gorilla | `threat_line` / `threat_square_2x2` | 0.0094 | `offered_piece` | 0.5030 |
| hawk | `reframed_(sq_)completable` | **0.0028** | `reframed_global` | 0.2509 |
| tiger | `tiger_square_winnable` | 0.0218 | `tiger_pool_safe_count` | **0.7480** |

Consequences to keep in mind when reading any per-category number:
`tiger_pool_safe_count` has a **trivial F1 of 0.856**, and `offered_piece` of
**0.669** (the known artefact) — F1 there is nearly uninformative. At the other
end, hawk's `completable` families give ~890 positives even in 296k rows, so
conclusions about them are measurement-limited. Regenerate the full table with
`python scripts/bsp_prevalence.py --bsps=<set>...`; audit and implications in
[`diary/2026-08-11_bsp-prevalence-audit.md`](diary/2026-08-11_bsp-prevalence-audit.md).

**A basis is a packaging convention; a CATEGORY is the analysis unit.** Within
`tiger`, `tiger_line_winnable` (base 0.023) and `tiger_pool_safe_count`
(base 0.748) share nothing but a filename, so a whole-basis average mixes
families spanning a 380x prevalence range and is not interpretable. Across bases,
`gorilla/threat_line`, `hawk/reframed_completable` and `tiger/tiger_line_winnable`
are three framings of ONE game fact and are directly comparable once prevalence
is handled — those **triads** are the meaningful comparison. Report and compare
categories; use the basis only to name which framing a category belongs to.
Tool: `scripts/basis_comparison.py`; worked example in
[`diary/2026-08-11_basis-recheck-prevalence.md`](diary/2026-08-11_basis-recheck-prevalence.md).

⚠️ **Selection criterion must match the reported metric.** Selecting features by
J and then reporting MCC@pref is incoherent: at base rate 0.003 the J-optimal
latent fires on 10% of positions (J = 0.85, precision = 0.07), which flatters the
rarest families. Select by `mcc_at_pref` when reporting `mcc_at_pref`.

Full schema: [`BSP-schema-summary.md`](BSP-schema-summary.md).

## 3. Phase 3A — dilution diagnostic

Code: `lib/sae/dilution.py` (game-agnostic) + `scripts/dilution_diagnostic.py`.
Design spec (frozen): [`diary/2026-07-21_3A-dilution-diagnostic.md`](diary/2026-07-21_3A-dilution-diagnostic.md).
Every report JSON embeds this glossary under its `glossary` key.

### 3.1 What it answers

For one concept and one trained SAE: **how is this concept carried in the
dictionary?** Not "how well can one feature detect it" (that is the coverage
metric) — but whether the information is present at all, and if so whether it is
concentrated or spread.

### 3.2 The pipeline

**Stage A — rank candidates.** Binarize the codes (`h > fire_threshold`), compute
signed phi between every *alive* latent's firing and the concept, keep the top
`top_k` by |phi|. This ordering is the support order for the curve.

- "Fires" is **`h > 0`**, not a swept threshold and not relative to `h_max`.
  TopK / BatchTopK / JumpReLU all emit exact structural zeros, so 0 is the
  architecture's own on/off boundary. `fire_threshold` exists in the config for
  a hypothetical dense (Vanilla/Gated) dictionary; it has never been changed.
- "Alive" = firing rate in `[min_freq, max_freq]` = `[1e-4, 0.999]`.
- Stage A state depends only on the codes, not the concept, so it is built once
  per run (`RankingCache`) rather than per BSP.

**Stage B — everything else**, on the `top_k` selected columns only.

*Co-firing couplings.* On the binarized firing of the candidates: correlation
matrix → ridge → invert to the precision matrix `Theta` → partial correlations
`pcorr_ij = −Theta_ij / sqrt(Theta_ii · Theta_jj)`. This is the
Gaussian-graphical-model estimator (an approximation for binary variables — the
**sign structure** is robust, the magnitudes are used only via a threshold).
**Decoder cosine similarity is deliberately not used**: `sae-concept-manifolds`
shows it fails to recover co-firing structure.

*Community.* Build a graph over candidates with an edge where
`|pcorr| > coupling_tau` (0.05); take the deterministic greedy-modularity
community containing the top latent. Fallbacks: connected component, then
singleton.

*Restricted-R2 support curve.* Regress the concept on the first `k` candidates,
`k = 1…top_k`, held out. **Averaged over `n_splits` = 5 independent train/test
resamples**, not one draw — a single split makes `knee_k` (a threshold crossing)
hostage to one sample, and for a base-rate-0.02 concept a 30% test split holds
only a few hundred positives. `asymptote_r2_std` reports the across-split spread
so instability is visible rather than silently priced in.

*Null.* `n_perm` = 5 label shuffles **per split** (25 draws total), at full
support. Only the train labels are shuffled; the test set is untouched.

*Estimator note.* The curve is OLS with intercept, computed by forming the train
Gram matrix once per split and solving each `k x k` prefix — numerically
identical to refitting per prefix (pinned by a test), ~18x faster. The
permutation null reuses the same Gram, so a null draw is one mat-vec.

### 3.3 Row budgets — and why there are two

| knob | default | applies to |
|---|---:|---|
| `max_rows` | 40,000 | Stage A ranking; needs all `d_dict` columns, so rows are memory-expensive |
| `curve_rows` | 120,000 | Stage B; only `top_k` columns, so rows are cheap |
| `min_positives` | 2,000 | raises `curve_rows` until the concept has this many positive rows (capped at N) |

At base rate 0.02, a flat 40k subsample leaves ~800 positives and ~240 in the
held-out split — too few to trust a knee. The `min_positives` rule widens the
budget to ~100k rows for such a concept while **preserving the natural base
rate** (it is a larger uniform sample, not a stratified/enriched one, so R2 stays
interpretable). Reports carry `n_curve_rows` and `n_curve_positives`; a concept
whose `n_curve_positives` is far below `min_positives` is too rare in this
dataset to diagnose and its verdict is provisional.

### 3.4 Metrics

⚠️ **`top_k` is the scale of this diagnostic.** `knee_k`, `community_size` and
`n_candidates` are all bounded by `top_k` (default **64**), and `asymptote_r2` is
measured at exactly that support. A range "1…64" means "1…`top_k`" — it is a
chosen hyperparameter, not a property of the SAE, and raising it would raise
`asymptote_r2` (more predictors) and can only move `knee_k` outward. Always quote
`top_k` with these numbers; it is recorded in every report's `config`.

| metric | range | ideal (for "captured") | meaning |
|---|---|---|---|
| `top_phi` | −1…1 | near ±1 | Signed phi (**= MCC**) of the single best-associated latent. The best single-latent association that exists in the dictionary — so it is directly comparable to `best_mcc_per_bsp` from the eval pipeline. |
| `asymptote_r2` | ≤1 | high | Held-out R2 from **all `top_k`** latents together = total recoverable information, independent of concentration. R2 = fraction of variance explained; 1 = perfect, 0 = no better than the mean. |
| `asymptote_r2_std` | ≥0 | small vs `asymptote_r2` | Across-split spread. Comparable to `asymptote_r2` itself ⇒ the concept's numbers are split noise. |
| `null_r2` | ~0 | 0 | Shuffled-label floor. `asymptote_r2 − null_r2` is the real signal. |
| `solo_r2` | ≤1 | ≈ `asymptote_r2` | Held-out R2 from the single best latent. |
| **`solo_frac`** | 0…1 | **≥ 0.70** | `solo_r2 / asymptote_r2` — share of all recoverable signal carried by ONE latent. **The primary concentration metric.** |
| `knee_k` | 1…`top_k` | 1–2 | Smallest k reaching 90% (`knee_frac`) of `asymptote_r2`. **Brittle** — a ~2% upward drift in the curve's tail pushes the crossing far right even when latent #1 did the work. Never read alone. |
| `community_size` | 1…`top_k` | 1–3 | Latents in the co-firing community. Counts **redundancy** — near-duplicates inflate it without making the concept harder to read. |
| `intrinsic_dim` | 1…`community_size` | ~1 | **PCA participation ratio** `(Σλ)²/Σλ²` of the community's codes, restricted to rows where the concept is TRUE. Continuous "effective dimensionality" — **not** a count of axes above a variance cutoff, so there is no 90%-threshold to choose. 1.0 = one dominant direction; 5.0 = five comparable ones. Computed on the **community**, not on all `top_k`. |
| `support_overlap` | 0…1 | — (selects the failure mode) | Mean **pairwise Jaccard** of community firing supports: for each pair, `|rows where both fire| / |rows where either fires|`, averaged over pairs. **Not** "how many latents fire per position". HIGH = same rows (redundant → diluted); LOW = disjoint rows (shattered → tiled). Neither end is good. A singleton community has no pairs and returns 1.0 by convention — check `singleton_community` before reading 1.0 as redundancy. |
| `neg_coupling_frac` | 0…1 | — | Share of within-community couplings that are negative (latents suppressing each other). High + low overlap = tiled. |
| `knee_over_idim` | ≥1 typically | ~1 | `knee_k / intrinsic_dim`. ≫1 = many more latents needed than the code's own dimensionality implies: a splitting signature. |
| `geometric_frac` | 0…1 | — | `(n_diluted + n_tiled) / n_threat_bsps`. **Gate G-3A**: ≥ 0.50 → 3C proceeds. |

### 3.5 Verdicts and the classification rule

| verdict | plain meaning | what it implies |
|---|---|---|
| **absent** | Codes carry no more signal than the null. | Not a capacity problem — change the hook, or go E2E/supervised. No architecture change on these activations can help. |
| **captured** | Recovered by a small, low-dimensional latent set. | Nothing to fix. |
| **diluted** | Recoverable only by aggregating many **overlapping** latents (feature splitting). | *Geometric.* Info is present; an aggregating/hierarchical readout can get it. |
| **tiled** | Recoverable but spread over near-disjoint, **competing** latents. | *Geometric.* Needs a manifold-aware/bilinear readout. |

**"Geometric"** = diluted **or** tiled = "the information is in the code, the flat
dictionary just isn't presenting it in one atom". This is the H10 outcome and the
*favourable* case — the fix is an architecture change rather than a dead end.

**Rule 3A.2** (`classify`, a pure function of the stored metrics):

```
gap = asymptote_r2 - null_r2
if gap < absent_margin (0.02) or asymptote_r2 < absent_floor (0.02):
    absent
elif knee_k <= captured_k (2) AND community_size <= captured_size (3):        # (a)
    captured
elif solo_frac >= captured_solo_frac (0.70)                                    # (b)
     AND (intrinsic_dim <= captured_idim (2.0) OR knee_k <= captured_k (2)):
    captured
elif support_overlap < tile_overlap (0.15) AND neg_coupling_frac > tile_neg_frac (0.50):
    tiled
else:
    diluted
```

**"A small, low-dimensional set" is a description, not the test.** There are two
*sufficient* conditions and either one is enough:

- **(a)** operationalises "small" as *few latents needed* (`knee_k ≤ 2`) *and*
  *few latents present* (`community_size ≤ 3`).
- **(b)** operationalises it as *the signal is concentrated in one latent*
  (`solo_frac ≥ 0.70`) *and* the carrier is low-dimensional by **either**
  available measure. Either suffices because both are one-sided: `intrinsic_dim`
  is inflated by redundant members, `knee_k` by noise creep.

(b) was added as rule **3A.2** after the 3A.1 rule failed its pre-registered
calibration check — see [`diary/2026-07-27_3A-dilution-results.md`](diary/2026-07-27_3A-dilution-results.md) §3.
`rule_version` is stamped into every report.

Because the verdict is a **pure function of stored metrics**, a threshold change
never needs the multi-GB `_h` caches:

```bash
python scripts/dilution_diagnostic.py reclassify saes/quarto/analysis/*_dilution-*.json --dry-run
```

### 3.6 Controls

- **Permutation null** (always): shuffled labels, same regression. Weak — it only
  destroys the concept's association, not the dictionary's structure.
- **Random-model SAE control** (preferred, currently missing for Ta/Ve/Yb): an
  SAE trained on a *randomly initialised* network's activations. A random CNN
  still yields SAE structure, so the permutation null **understates** what
  "absent" should mean. Reports record `random_control: null` when it was not
  available, so the weaker null is never silently assumed.

## 4. Hypothesis and gate IDs

| ID | statement |
|---|---|
| **H10** | The SAE/LP wall is *geometric*: conjunctive concepts occupy multi-dimensional structure that flat dictionary atoms dilute or tile. Phase 3's founding hypothesis. |
| **H11** | Decision-upstream concepts (tiger) are more linearly readable than spectator concepts (gorilla) at matched base rate. |
| **G11** | The epiphenomenality risk (Balogh & Jelasity): a probe can decode a concept the model never *uses*. **Decodability ≠ causality.** |
| **G-3A** | `geometric_frac ≥ 0.50` over a run's threat BSPs → 3C proceeds; else 3C deprioritised in favour of hooks / E2E. 3B runs regardless. |
| **G-3C** | Beat I04 tigerVe, or close ≥50% of the threat SAE/LP gap, over 3 seeds, with the random control unchanged and centred cross-seed feature-overlap stability. |
| **HP-canonical** | **CLOSED 2026-08-17.** Both axes have interior optima under canonical recipes — conv2 k=32, fc1 exp32 — so the swept ranges are wide enough. Opened 2026-08-16; see §6. |

Full hypothesis table with current status: [`../RESEARCH-STATUS.md`](../RESEARCH-STATUS.md).

## 5. Reporting rules that follow from all this

1. Rank and conclude on **MCC**. F1 is for the literature comparison at write-up.
2. Always show the **random-network control** alongside the trained one.
3. Per-category breakdowns need the **within-category spread**, not just the
   mean — a category is many independent BSPs and they can disagree structurally.
4. Feature→BSP alignment in the eval pipeline is **greedy argmax on
   decodability**, not causality. 3A avoids this by scoring communities (top-K by
   signed phi), never a single feature. Causal claims need 3B-causal / 3D.
5. Every results presentation opens with a glossary — link here rather than
   restating definitions.
6. A pre-registered gate must be shown able to **fail**: run the positive control
   and confirm the rule classifies it as designed before reading the gate.


## 6. HP-canonical — the hyperparameter gate (opened 2026-08-16, CLOSED 2026-08-17)

**Status: CLOSED.** Both axes have interior optima under canonical recipes, so the
swept ranges are wide enough and the settings in use sit at or adjacent to the
optimum. Result table in §6.5; full record in
[`diary/2026-08-17_3A-gate-readable.md`](diary/2026-08-17_3A-gate-readable.md) §4.

### 6.1 What is settled

**Convergence.** 11 of 232 runs early-stopped — all `jumprelu-t32`,
`vanilla-l1_0001` or `batchtopk-k16` seed replicates, and **no panel member or
headline run**. That is a per-run fact and recipe-independent, so it is closed
and stays closed.

### 6.2 What is open, and why the existing sweeps do not answer it

The sweeps on disk were run on **pre-conformance recipes** (see
[`diary/2026-08-15_dead-feature-revival.md`](diary/2026-08-15_dead-feature-revival.md)),
and the canonical points show the curves change *shape*, not just level:

| conv2 TopK, gorillaYb | k=16 | k=32 | k=48 | k=64 | k=96 |
|---|---:|---:|---:|---:|---:|
| legacy (kaiming init) | 0.333 | 0.330 | 0.329 | 0.306 | 0.278 |
| canonical (tied init) | 0.269 | **0.319** | — | — | — |

Legacy declines from k=16, so the optimum reads as at-or-below the low edge.
Canonical *rises* 0.269 → 0.319, so the optimum may be at k=48 or beyond, where
no canonical run exists. The fc1 expansion sweep has the same defect for a
sharper reason: it was JumpReLU with the kaiming init, and the tied init is
precisely the mechanism that stops columns dying — the legacy trend had alive
latents *shrinking* as expansion grew (266 → 184 from exp8 to exp64).

**Second defect: those curves are whole-basis numbers.** Per §2 and `CLAUDE.md`,
a whole-basis mean cannot support a comparison — the best `k` for cell concepts
need not be the best `k` for threats. The redo must be reported **per concept
family**.

### 6.3 What closes it

Four canonical runs, reported per family:

| run | purpose |
|---|---|
| conv2 TopK canonical, k=48 | complete the sparsity curve above the current canonical maximum |
| conv2 TopK canonical, k=64 | confirm the turn-over |
| fc1 JumpReLU canonical, exp16 | complete the expansion curve |
| fc1 JumpReLU canonical, exp32 | confirm the turn-over |

Plus seed replicates on whichever setting comes out closest to the optimum, per
the replication rule (§ replicate where a margin is thin or a claim inverts a
previous result). Estimated ~1 h wall clock across three GPUs.

### 6.4 Why it is a gate rather than a task

**G-3C** is "beat I04 tigerVe 0.255, or close ≥50% of the threat SAE/LP gap". If
the unsupervised baseline is under-tuned, that criterion is too easy and any
geometry-aware 3C variant clears it for the wrong reason. HP-canonical exists so
the baseline cannot be silently weak at the moment 3C is judged against it.

### 6.5 Result [DIRECT] — champYb, gorillaYb

| conv2 sparsity (TopK, exp8) | k=16 | k=32 | k=48 | k=64 |
|---|---:|---:|---:|---:|
| legacy | 0.333 | 0.330 | 0.329 | 0.306 |
| **canonical** | 0.269 | **0.312** | 0.293 | 0.290 |

| fc1 expansion (JumpReLU t64) | exp8 | exp16 | exp32 | exp64 |
|---|---:|---:|---:|---:|
| legacy | 0.424 | 0.407 | 0.385 | 0.327 |
| **canonical** | 0.459 | 0.455 | **0.475** | 0.461 |

Legacy declined monotonically on both axes, implying the optimum lay *outside*
the swept range. Canonical peaks **inside** it on both. The conformance fix did
not merely shift the curves' level, it changed their **shape** — which is why
the pre-conformance sweeps could not answer this and why the gate was opened
rather than closed by inspection.

**Corollary — capacity is not the binding constraint.** Across an 8x larger
dictionary (4,096 -> 32,768 slots) alive latents move 350 -> 400 and effective
expansion 0.68x -> 0.78x, never reaching 1x, for +0.016 then -0.014 coverage. On
conv2, k=16 has the MOST alive latents (364) and the WORST coverage (0.269). The
residual wall cannot be attacked by enlarging the dictionary.
