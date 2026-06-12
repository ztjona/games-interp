# Phase 3 pivot — geometric concept structure (2026-06-09)

Status: design + pre-registration. No new training in this entry.
Parent ledger: [`phase-3.md`](phase-3.md). Predecessor arc:
[`phase-2B.md`](phase-2B.md) (chapters 1–5).

This entry records *why* Phase 3 exists, what evidence triggered it, the
pre-registered plan (steps 3A–3D with gates), and how thesis risk is
managed. Numbers quoted from registries / LP jsons are `[DIRECT]`;
mechanism claims are `[AI-REASONED PROVISIONAL ANALYSIS]` until 3A/3B
test them.

---

## 1. Where the project stood on 2026-06-09 [DIRECT]

The 2B+2C arc (see [`phase-2B.md`](phase-2B.md) ch. 5) closed with the
SAE/LP wall measured on three champions and its non-causes identified:

- **Wall:** best unsupervised SAE F1-lift ≈ 0.19–0.21 (gorilla conv2:
  E05-Ta 0.214, E01-Ve 0.192) vs LP ceiling 0.71–0.76 on the same
  activations. SAE/LP efficiency 25–32% on gorilla/hawk.
- **Not capacity:** Sweep H closed 0% of the gap; dead features rise to
  ~99% with expansion (`phase-2B.md` ch. 4). champVe batchtopk runs:
  98.2–98.3% dead.
- **Partially objective:** anchoring (I04, λ_h=1.0) reaches SAE/LP ≈ 86%
  on tiger — but only on the anchored slots, and tiger's LP ceiling is
  itself low (lift 0.298 fc1/Ve).
- **The failure is category-structured**, not uniform. champVe winners,
  per-category F1 [DIRECT, registry_query category, 2026-06-09]:

| Concept family | fc1 SAE (I03) | conv2 SAE (E01) | LP reference |
|---|:---:|:---:|:---:|
| cell_occupancy | 0.730 | 0.618 | — |
| cell_attribute | 0.503 | 0.685 | — |
| game_phase / global | 0.676 / 0.680 | 0.477 / 0.411 | — |
| **threat_line** | **0.163** | **0.072** | 0.551 (fc1) / 0.887 (conv2) |
| **threat_square_2x2** | **0.233** | **0.073** | — |
| tiger pool counts (I04) | 0.765–0.863 | — | — |
| tiger line/square winnable (I04) | 0.202 / 0.387 | — | — |

Additive / countable concepts (occupancy, attributes, phase, pool
counts) are captured; **relational / conjunctive** concepts (a line's
cells *share an attribute* AND the line is completable) are exactly
where the SAE collapses while the LP does not.

## 2. The trigger: literature synthesis + five papers [DIRECT — papers in DB]

The 2026-06-08 synthesis of the paper database (123 summaries; external:
papers database `synthesis/2026-06-08_sae-landscape.md`) identifies
"directions vs. geometry" as the field's central open fault line, and
flags three gaps this project is positioned to fill: **G1**
(geometry-aware evaluation), **G2** (manifold architectures lack causal
validation), **G9** (board games are the cleanest known-ground-truth
testbed, unused by the 2026 geometry wave).

Five papers were processed into the DB on 2026-06-09 to complete the
theoretical base:

| DB tag | One-line relevance |
|---|---|
| `not-all-language-model-features` | Defines irreducible multi-dim features; separability / ε-mixture indices; **causal subspace-patching recipe** (template for 3D). |
| `geometry-categorical-hierarchical-concepts-large` | Binary features = vectors; categorical = polytopes; **hierarchy = orthogonality; children are co-linear with parents** → theoretical root of absorption (H2). |
| `incorporating-hierarchical-semantics-sparse-autoen` | H-SAE: MoE architecture implementing parent/child geometry; **never benchmarked vs matryoshka** — a gap our testbed fills (3C). |
| `understanding-sparse-autoencoder-scaling-presence` | Brill capacity-allocation model: regime β<α ⇒ SAEs tile common structure and starve rare features; **α and β are unmeasurable in LLMs but measurable here** (3B). |
| `origins-linear-representations-large-language` | Linearity is *objective-specific* (softmax-CE + GD bias). DQN regression heads get no guarantee; distilled CE heads do. |

## 3. The reframing [AI-REASONED PROVISIONAL ANALYSIS — read sceptically]

Hypothesis **H10**: the residual SAE/LP wall is *geometric*. Concretely:

1. **Threats are conjunctions** (product of indicator conditions). A
   linear probe can read a conjunction obliquely (LP threat_line 0.55
   fc1 / 0.89 conv2 [DIRECT]); a *flat dictionary atom* cannot represent
   it — the structure is multi-dimensional in the sense of Engels et
   al., and the SAE responds by **diluting** it across redundant latents
   (Bhalla/Geiger `sae-concept-manifolds`) rather than capturing it.
2. **Sweep H's null is the allocation regime's prediction.** If common
   concepts (occupancy/attribute, base rates 0.17–0.34) sit on structure
   with slowly-decaying per-feature loss, widening the dictionary
   re-tiles them instead of discovering rare threats (base rate ≈ 0.01)
   — matching chapter 4's "0% of gap closed, dead features ↑" and the
   monotone fc1/hawk degradation with width.
3. **Absorption (H2) is hierarchy co-linearity.** Park et al. prove
   child concepts share the parent's direction plus an orthogonal
   residual: `threat_line_tall ≺ line_3_tall ≺ line_has_3` predicts the
   measured 6–10 BSPs/feature on champTa.
4. **Objective-specificity explains the champion gradient.** The
   distilled CE SELECT/PLACE heads (S4/Ta/Ve) fall under the
   linearity-promoting mechanism of Jiang et al.; concepts *upstream of
   the action choice* should linearize best. Consistent: tiger pool
   counts (decision-adjacent) hit SAE F1 0.77–0.86 while spectator
   threat conjunctions lag; gorilla LP rose Aa → Ta → Ve (H9). A DQN
   value-regression model (champAa) had no such guarantee.

Falsifiable alternative: threat concepts are simply **absent** from the
representation in dictionary-accessible form (the champAa-era reading).
3A is designed to distinguish *diluted* / *tiled* / *absent*.

## 4. Phase 3 plan — pre-registered

### 3A — Dilution diagnostic on existing SAEs (cheap; first)

Method, on cached SAE codes (champVe + champTa, F04/I03/I04/E01/E05):
Ising-coupling co-activation communities (NOT decoder cosine — shown to
fail in `sae-concept-manifolds`), restricted-R² support curves, and the
separability / ε-mixture irreducibility indices on concept-conditioned
reconstructions.

Per threat-BSP-group verdicts:
- **Diluted**: community of size ≫ intrinsic dim with mixed-sign
  couplings; restricted-R² plateaus far beyond ambient concept dim.
- **Shattered/tiled**: near-disjoint supports, negative couplings.
- **Absent**: no community aligns with the concept above the
  random-SAE control.

**Gate G-3A:** if ≥ half of threat BSPs show dilution/tiling → geometric
capture is the binding constraint; 3C goes ahead. If absent → 3C is
deprioritized; revisit hooks / E2E first. Either way 3B runs (it does
not depend on the verdict).

### 3B — Ground-truth geometry + allocation-regime measurement

- **α** directly from the BSP base-rate spectrum (known by
  construction — impossible in LLMs).
- **β** from per-concept latent-allocation curves L_c(n) across the
  existing width sweep (H01–H06 + exp8 baselines), counting latents per
  concept community.
- First empirical classification of the Brill–Michaud regime on a
  ground-truth testbed (their stated open question).
- Polytope / hierarchy checks (Park): `game_phase` 2-simplex;
  `offered_piece` polytope vertices vs the 4 attribute axes; hierarchy
  orthogonality cos(ℓ_child − ℓ_parent, ℓ_parent) for threat families on
  LP directions. Note: Park's theory is final-layer; testing it at
  fc1/conv2 of a game CNN is itself a contribution.

### 3C — Geometry-aware SAE variants (gated on G-3A)

All retain the SAE backbone (thesis commitment intact):
1. **Hierarchical anchoring** — extend `_AnchorMixin` from single-slot
   BCE to parent/child slot blocks with child-minus-parent targets
   (Park's orthogonalized basis). Smallest diff, reuses I-series infra.
2. **H-SAE-style experts vs matryoshka head-to-head** — the comparison
   missing from the literature; per-expert sparsity TopK_j (threats
   co-occur; TopK₁ is wrong for us).
3. **Bilinear/quadratic slots** f_j(x) = (L_j^T x)(R_j^T x) for
   conjunction concepts — tractable at d=512 where LLM work cannot go.

**Gate G-3C:** a variant is promoted iff it beats I04 tigerVe lift 0.255
or closes ≥ 50% of the threat-category SAE/LP gap at matched L0, on 3
seeds, with the random-model control unchanged.

### 3D — Causal validation (fills G2)

Engels-style subspace patching adapted to Quarto: fit low-dim probe on
the candidate concept subspace → replace coordinate → average-ablate the
complement → measure **move-change rate / targeted-move rate** vs
random-direction and full-layer-patch controls. Run on the best 3C
variant and on 3A-recovered communities of a baseline SAE.

## 5. Thesis-risk framing

- **Safe path (graduation-sufficient):** 3A + 3B alone constitute a
  geometry-aware evaluation framework on a ground-truth testbed (gaps
  G1 + G9) on top of the already-banked anchored-SAE results. No
  dependency on new architectures working.
- **Novel path (high upside):** 3C/3D — geometric capture inside an SAE
  with causal validation. Gated by G-3A so we spend sweep budget only if
  the mechanism is confirmed.

## 6. What Phase 3 does NOT change

Reporting standard (F1 + MCC + F1-lift, LP ceiling + random control,
count-weighted categories), run-ID scheme, BSP naming, eval pipeline,
and the champion roster all carry over unchanged.
