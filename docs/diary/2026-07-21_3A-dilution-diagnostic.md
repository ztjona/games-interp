# Phase 3A — dilution diagnostic: method specification (2026-07-21)

Status: design + implementation. Frozen after creation (dated diary entry).
Parent ledger: [`phase-3.md`](phase-3.md). Founding rationale + gate
pre-registration: [`2026-06-09_geometric-pivot.md`](2026-06-09_geometric-pivot.md)
§4 (3A). Reassessment against the 2026-07-14 literature v2:
[`phase-3.md`](phase-3.md) ch. "2026-07-21 litv2 reassessment".

This document specifies **what 3A measures, how, and how a verdict is
assigned**, so that any result JSON can be reproduced and audited. Code:
`lib/sae/dilution.py` (game-agnostic core) + `scripts/dilution_diagnostic.py`
(CLI). Runner: `runner3A.sh`. Tests: `tests/test_dilution.py`.

---

## 1. Question

For a trained SAE and a target concept (one BSP column `y`), decide **how the
concept is carried in the dictionary**, distinguishing four outcomes:

| verdict | meaning | fix implied |
|---|---|---|
| **absent** | codes carry no more signal about `y` than a null control | change hook / E2E / supervision — not a capacity problem |
| **captured** | recovered by a small, low-dim set of latents (clean) | nothing — the SAE already has it |
| **diluted** | recoverable only by aggregating **many overlapping** latents (feature splitting) | aggregation / hierarchical readout can recover it (info is present) |
| **tiled** | recoverable but spread over **near-disjoint, competing** latents (shattered manifold) | manifold-aware / bilinear readout |

`diluted` and `tiled` are the two *geometric* outcomes H10 predicts.
`captured` and `absent` are the two non-geometric outcomes (already-solved, or
not-in-the-dictionary). This is the decision tree for 3C: **diluted/tiled →
the information is in the code and an architecture change can extract it;
absent → an architecture change on the same activations cannot.**

The design note (2026-06-09) named three outcomes (diluted / tiled / absent);
`captured` is added here so a threat concept that *is* cleanly represented is
not mislabelled geometric. It never counts toward the gate.

## 2. Inputs (all row-aligned, N positions)

- `saes/<game>/cache/<run_id>_h.pt` — full `(N, d_dict)` SAE codes (the
  post-activation feature values `h`, produced by `sae_eval.py evaluate`).
- `data/<game>/bsp_labels-<bsps>_<count>.pt` — `(N, num_bsps)` binary labels
  (champion-suffixed, e.g. `tigerTa`, so base rates match the activations).
- `data/<game>/bsp_schema-<basis>_<count>.json` — concept menu (id, category).
- *Optional* `saes/<game>/cache/<random_run_id>_h.pt` — codes from the
  **random-model** SAE, row-aligned. Supplies the proper `absent` control (a
  random CNN still yields SAE structure; permutation of `y` is the weaker
  fallback used when this is not provided). Supply it whenever available — it
  is required for a trustworthy `absent` verdict, per the reporting standard's
  random-network-control clause.

## 3. Per-concept pipeline (`diagnose_concept`)

All numeric work runs on a deterministic row subsample (`max_rows`, default
40 000; base rate is still taken on the full N). R2 / couplings / PCA are
stable far below the full N, and this keeps each concept under ~1 s.

1. **Association ranking.** For every *alive* latent (firing rate in
   `[min_freq, max_freq]`), compute the signed phi coefficient (= MCC for 2x2)
   between its firing and `y`. Keep the top-`top_k` by `|phi|`; this order is
   the support order for the R2 curve. The top latent seeds the community.

2. **Signed couplings.** On the binarized firing of the top-`top_k` candidates,
   estimate partial correlations via the ridge-regularized inverse correlation
   matrix: `pcorr_ij = -Theta_ij / sqrt(Theta_ii Theta_jj)`. This is the
   Gaussian-graphical-model estimator (an approximation for binary variables)
   and recovers the **sign structure** the verdict needs. Decoder cosine is
   deliberately **not** used — `sae-concept-manifolds` shows it fails to
   recover co-firing structure.

3. **Community.** Build a graph over candidates with an edge where
   `|pcorr| > coupling_tau`; take the deterministic greedy-modularity community
   containing the seed latent (fallback: connected component, then singleton).
   From it compute: `community_size`; `neg_coupling_frac` (fraction of within-
   community couplings that are negative); `support_overlap` (mean pairwise
   Jaccard of firing supports — high = redundant/overlapping = dilution, low =
   disjoint = tiling); `intrinsic_dim` (PCA participation ratio of the
   concept-conditioned community code).

4. **Restricted-R2 support curve.** With a single held-out split
   (`test_frac`), regress `y` on the first `k` association-ranked latents for
   `k = 1..top_k`; record the held-out R2 curve, its `asymptote_r2` (full
   support), and `knee_k` (smallest `k` reaching `knee_frac` of the asymptote).
   The **permutation null** (`n_perm` shuffles of `y`, R2 at full support) gives
   `null_r2`. Held-out R2 prevents the trivial monotone inflation that in-sample
   R2 would show as `k` grows.

5. **Verdict** (`classify`, a pure function of the metrics + config):

   ```
   gap = asymptote_r2 - null_r2
   if gap < absent_margin or asymptote_r2 < absent_floor:
       absent
   elif knee_k <= captured_k and community_size <= captured_size:
       captured
   elif support_overlap < tile_overlap and neg_coupling_frac > tile_neg_frac:
       tiled
   else:
       diluted
   ```

   Every threshold is a `DilutionConfig` field and is copied verbatim into the
   output JSON, so a verdict can always be recomputed from the raw metrics
   without rerunning the model.

## 4. Gate G-3A

Over the threat/relational BSPs of a run (categories selected by
`--categories`, default `auto` = names containing `threat`, `winnable`,
`completable`, `completing`, or the hawk `reframed_count` families):

```
geometric_frac = (n_diluted + n_tiled) / n_threat_bsps
verdict = "3C-proceeds"      if geometric_frac >= 0.5
          "3C-deprioritized" otherwise
```

Per the founding pre-registration: **>= half of threat BSPs diluted-or-tiled →
geometric capture is the binding constraint and 3C goes ahead; otherwise the
concepts are absent in dictionary-accessible form and 3C is deprioritized in
favour of hooks / E2E.** 3B runs regardless of this verdict.

The gate is evaluated per run and per BSP set; the headline decision uses the
best unsupervised runs (F04 fc1, E05 conv2) on `tigerTa`/`tigerVe`, cross-read
against the anchored I04 (which, being supervised, is expected to sit toward
`captured` on its anchored slots — a positive control that the diagnostic
separates `captured` from `diluted`).

## 5. Output JSON

`saes/<game>/analysis/<run_id>_dilution-<bsps>.json`:

```
{
  "summary":   { run_id, game, bsp_set, n_samples, d_dict, categories,
                 random_control, config: {<all thresholds>} },
  "gate_g3a":  { n_threat_bsps, n_diluted, n_tiled, n_captured, n_absent,
                 geometric_frac, verdict },
  "category_summary": { <cat>: {n, captured, diluted, tiled, absent,
                                mean_asymptote_r2} },
  "concepts":  [ { bsp_index, bsp_id, category, verdict, base_rate,
                   asymptote_r2, null_r2, knee_k, community_size,
                   neg_coupling_frac, support_overlap, intrinsic_dim,
                   knee_over_idim, curve:[...], random_asymptote_r2? } ]
}
```

## 6. Known limitations / calibration notes

- **R2 is base-rate-sensitive.** For very rare concepts (base rate ~0.01) the
  absolute `asymptote_r2` is small even when the signal is real; the
  `gap`-vs-null and the random-model control — not the absolute R2 — are the
  load-bearing quantities. `absent_floor`/`absent_margin` should be sanity-
  checked against the random control on each champion before the gate is read.
- **Binary partial correlations are approximate.** The sign structure is
  robust; exact magnitudes are not, and are used only via thresholds.
- **Verdict thresholds are provisional.** Defaults were set for legibility and
  validated on synthetic planted concepts (`tests/test_dilution.py`) and a
  champAa fc1 smoke run; the champTa/Ve run should confirm they place the
  supervised I04 anchored slots at `captured` and the F04 threat concepts at
  `diluted`/`tiled` (the expected calibration check).
- **Theory dependence.** Dorrell's optimality result (splitting/absorption are
  optimum properties) is derived for the L1 vanilla objective; TopK/JumpReLU
  inherit the qualitative prediction but not the exact conditions. 3A tests the
  prediction empirically with concept hierarchy known by construction; it does
  **not** assume it. 3A says nothing about causality (that is 3B-causal / 3D).
