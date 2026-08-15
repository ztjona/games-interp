# Hook specialisation: conv2 holds the threat information, fc1 factors it (2026-08-14)

Status: frozen self-contained record. Parent ledger: [`phase-3.md`](phase-3.md).

> **Partial correction (2026-08-15).** The champYb `s4.conv2` **SAE** column
> below was measured on a collapsed dictionary (§4 caveat). Retrained with
> dead-feature revival it rises 4–6× on the per-cell families and ~1.5× on the
> threat families, so conv2 threat efficiency is **0.18–0.19**, not 0.12–0.13.
> The conclusion is unchanged and in fact strengthened — the threat gap against
> fc1 (0.62 / 0.91) survives almost intact, and what the broken dictionary hid
> was conv2's competence on *spatial* concepts. Every LP column, and every
> conclusion resting on availability, is unaffected. See
> [`2026-08-15_dead-feature-revival.md`](2026-08-15_dead-feature-revival.md) §4.2.
Metric definitions: [`../methods-reference.md`](../methods-reference.md) §1.
Numbers are `[DIRECT]` from `saes/quarto/eval_registry.json` and the
`*_sae-lp-efficiency.json` reports of the 2026-08-14 basis-verdict run;
interpretation is `[AI-REASONED PROVISIONAL ANALYSIS]`.

Asked while reviewing the basis verdict: *does fc1 consistently beat conv2?*
The pooled answer is "yes", and the pooled answer is misleading. Broken out by
concept family the split is total and has no exceptions, and once the linear
probe is put beside the SAE the two hooks turn out to differ on **isolation**,
not on **information** — which is a different claim, and a more useful one.

## 0. Glossary

| term | range | ideal | meaning |
|---|---|---|---|
| **availability** | −1…1 | high | linear-probe `mcc_at_pref` on the raw activations of that hook. What the layer carries, before any SAE. A property of the model. |
| **isolation** | −1…1 | high | best single SAE latent's `mcc_at_pref`. What the dictionary presents as one atom. |
| **efficiency** | 0…1+ | 1 | isolation ÷ availability, per concept family. |
| **concept family** | — | — | the cross-basis axis (`line_threat`, `square_threat`, `board_attribute`, …), read from the schema's `concept_family` stamp. A *basis* is packaging; the family is the game fact. |
| **matched cell** | — | — | one (champion, SAE recipe) pair for which BOTH hooks were trained with the same recipe, so the hook is the only thing that varies. |

`mcc_at_pref` standardises to p_ref = 0.025. That is inside the measured range
for the threat families (base rate 0.009–0.026) and **far outside it** for
`pool_reasoning` (0.49) and `global_threat` (0.45) — those two rows are marked
and should be read on raw MCC / J instead. See §5.

## 1. Design

Nine matched cells: champions **Ta, Ve, Yb** × recipes **topk-k32-exp8**,
**topk-k64-exp8**, **batchtopk-k32-exp8**, each trained at both `s4.fc1` and
`s4.conv2`. Both hooks are **512-dimensional** here (`s4.conv2` is the flattened
conv output; verified `(289795, 512)` for both), so the SAEs have identical input
width and identical `d_dict = 4096` at exp8. Nothing about the comparison is
confounded by hook width.

Metric: mean best-latent MCC per concept family, count-weighted over BSPs
(`aggregate_per_category_by_family`). Youden's J is shown where both rows carry
it — 6 of the 9 cells; the three champYb cells whose registry rows predate the J
backfill are counted in the MCC columns only.

## 2. The split [DIRECT]

| basis | family | n BSPs | base rate | fc1 MCC | conv2 MCC | Δ | sd(Δ) | fc1 wins | fc1 J | conv2 J |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| gorilla | `board_occupancy` | 16 | 0.3401 | 0.448 | 0.542 | **−0.094** | 0.130 | 2/9 | 0.426 | 0.577 |
| gorilla | `board_attribute` | 64 | 0.1713 | 0.399 | 0.518 | **−0.119** | 0.113 | 2/9 | 0.400 | 0.610 |
| gorilla | `offered_piece_attr` | 4 | 0.5042 | 0.370 | 0.378 | **−0.008** | 0.273 | 3/9 | 0.265 | 0.468 |
| gorilla | `game_phase` | 3 | 0.3333 | 0.432 | 0.230 | **+0.202** | 0.085 | 9/9 | 0.529 | 0.294 |
| gorilla | `global_threat` ‡ | 1 | 0.2559 | 0.383 | 0.201 | **+0.183** | 0.107 | 9/9 | 0.302 | 0.207 |
| gorilla | `line_threat` | 40 | 0.0095 | 0.260 | 0.102 | **+0.158** | 0.120 | 9/9 | 0.571 | 0.456 |
| gorilla | `square_threat` | 36 | 0.0091 | 0.388 | 0.107 | **+0.281** | 0.215 | 9/9 | 0.706 | 0.482 |
| hawk | `global_threat` ‡ | 2 | 0.2525 | 0.360 | 0.200 | **+0.159** | 0.086 | 9/9 | 0.300 | 0.213 |
| hawk | `line_threat` | 90 | 0.0136 | 0.244 | 0.110 | **+0.134** | 0.078 | 9/9 | 0.605 | 0.470 |
| hawk | `square_threat` | 81 | 0.0126 | 0.340 | 0.112 | **+0.228** | 0.123 | 9/9 | 0.748 | 0.503 |
| tiger | `global_threat` ‡ | 5 | 0.4510 | 0.405 | 0.217 | **+0.188** | 0.099 | 9/9 | 0.385 | 0.210 |
| tiger | `line_threat` | 10 | 0.0248 | 0.195 | 0.100 | **+0.094** | 0.029 | 9/9 | 0.432 | 0.301 |
| tiger | `square_threat` | 9 | 0.0229 | 0.273 | 0.101 | **+0.173** | 0.058 | 9/9 | 0.587 | 0.311 |
| tiger | `offered_completion` | 4 | 0.0476 | 0.239 | 0.147 | **+0.092** | 0.072 | 8/9 | 0.344 | 0.368 |
| tiger | `pool_reasoning` ‡ | 8 | 0.4928 | 0.404 | 0.216 | **+0.188** | 0.103 | 9/9 | 0.389 | 0.210 |

‡ base rate far from p_ref; read these on raw MCC and J, not on the standardised column (§5).

**There are no exceptions in either direction.** Every per-cell / spatial family
goes to conv2 (fc1 wins 2/9, 2/9, 3/9); every relational, threat, decision or
pool family goes to fc1 (9/9, or 8/9 for `offered_completion`). The two groups
do not overlap and the sign is stable across three champions and three recipes.

The gap on state threats widens along the competence axis:

| family | champTa Δ | champVe Δ | champYb Δ |
|---|---:|---:|---:|
| gorilla/`square_threat` | +0.093 | +0.190 | **+0.560** |
| gorilla/`line_threat` | +0.049 | +0.105 | **+0.319** |
| hawk/`square_threat` | +0.114 | +0.187 | **+0.382** |
| tiger/`square_threat` | +0.122 | +0.180 | +0.216 |
| gorilla/`board_attribute` | −0.076 | −0.204 | −0.077 |
| gorilla/`board_occupancy` | −0.085 | −0.197 | +0.000 |

## 3. The probe changes what this means [DIRECT]

Put the linear probe beside the SAE and the story is not "fc1 has the threat
information". Over 45 (basis, family, champion) cells of the panel:

- **conv2 has the higher linear availability in 23** — including 11 of the 12
  state-threat cells;
- **fc1 has the higher SAE efficiency in 39.**

The state-threat rows, in full:

| basis / family | champ | LP fc1 | LP conv2 | more available at | SAE fc1 | SAE conv2 | eff fc1 | eff conv2 |
|---|---|---:|---:|:---:|---:|---:|---:|---:|
| gorilla/`line_threat` | Ta | 0.639 | 0.815 | **conv2** | 0.261 | 0.185 | 0.41 | 0.23 |
| gorilla/`line_threat` | Ve | 0.704 | 0.907 | **conv2** | 0.360 | 0.174 | 0.51 | 0.19 |
| gorilla/`line_threat` | Yb | 0.893 | 0.958 | **conv2** | 0.550 | 0.122 | **0.62** | 0.13 |
| gorilla/`square_threat` | Ta | 0.774 | 0.886 | **conv2** | 0.369 | 0.192 | 0.48 | 0.22 |
| gorilla/`square_threat` | Ve | 0.838 | 0.929 | **conv2** | 0.495 | 0.170 | 0.59 | 0.18 |
| gorilla/`square_threat` | Yb | 0.938 | 0.946 | **conv2** | 0.851 | 0.115 | **0.91** | 0.12 |
| hawk/`line_threat` | Ta | 0.712 | 0.840 | **conv2** | 0.299 | 0.189 | 0.42 | 0.23 |
| hawk/`line_threat` | Ve | 0.753 | 0.875 | **conv2** | 0.367 | 0.180 | 0.49 | 0.21 |
| hawk/`line_threat` | Yb | 0.873 | 0.897 | **conv2** | 0.483 | 0.121 | 0.55 | 0.13 |
| hawk/`square_threat` | Ta | 0.824 | 0.873 | **conv2** | 0.424 | 0.196 | 0.52 | 0.22 |
| hawk/`square_threat` | Ve | 0.871 | 0.884 | **conv2** | 0.511 | 0.179 | 0.59 | 0.20 |
| hawk/`square_threat` | Yb | 0.890 | 0.886 | fc1 | 0.702 | 0.115 | 0.79 | 0.13 |

The champYb row is the extreme case: **conv2 carries the square-threat concept
slightly better than fc1 (LP 0.946 vs 0.938) and its dictionary surfaces
one-seventh as much of it (0.115 vs 0.851).**

The agent-relative families behave differently — fc1 wins *both* terms, in all
15 tiger cells. `tiger_square_winnable` on champYb: LP 0.808 at fc1 against 0.338
at conv2.

## 4. Reading [AI-REASONED PROVISIONAL ANALYSIS]

Three regimes, and they line up with what each layer is for.

1. **Per-cell state (occupancy, attributes).** conv2 wins availability *and*
   isolation. Spatial facts stay spatial; the conv feature map is already close
   to one-channel-per-fact, so a dictionary atom finds them easily.

2. **State threats (line, square — gorilla and hawk).** Availability is equal or
   slightly higher at conv2; isolation is 3–7× higher at fc1. The information
   survives the bottleneck essentially intact and *changes form* crossing it:
   distributed over a spatial map at conv2, factored into directions at fc1.
   This is the cleanest evidence the project has that the SAE/LP wall is a
   **factorisation** problem and not an information problem — the same concept,
   measured on the same positions with the same recipe, is 0.95 available at
   both hooks and 7× more isolable at one of them.

3. **Agent-relative / decision concepts (tiger, `game_phase`, `pool_reasoning`).**
   fc1 wins availability too, by a lot on champYb. Expected rather than
   surprising: these are functions of the *offered piece* as well as the board,
   and the offered piece only meets the board state at the fully-connected
   layers. conv2 cannot represent them because it has not been shown the second
   operand yet.

This reframes two standing hypotheses:

- **H6** ("wrong hook — conv2 may be better") should be retired as posed. The
  hook question has no basis-level answer; it has a family-level answer, and the
  answer is a clean partition.
- **H8** ("threat info is spatially encoded, lost at the fc1 bottleneck") is
  overturned in its stronger form for the unified-aux family, which was already
  the 2026-05-19 finding — but the reason is now measured rather than inferred.
  Nothing is lost at the bottleneck; the LP recovers threats at 0.89–0.94 on both
  sides of it. The bottleneck is where the concept becomes *addressable*.

**Caveat that limits regime 2 quantitatively.** The conv2 panel member for
champYb (`E05-champYb-…-s4.conv2`) is a degenerate dictionary — 99.0% dead, 41
alive latents, FVU 0.110 against 0.043/0.051 for the same recipe on Ta/Ve, and
ranked 10th of 11 champYb conv2 runs by coverage MCC. Its SAE columns understate
champYb's conv2 isolation. **The LP columns are unaffected** (the probe reads
activations, not the SAE), so the availability comparison — which is what the
argument in regime 2 rests on — stands for all three champions. The Ta and Ve
efficiency rows use healthy dictionaries and already show the 2–3× gap.

## 5. What this does not say

- **Nothing causal.** Everything here is decodability. A conv2 latent could be
  the one the policy reads and still score badly on this table (G11).
- **The ‡ rows are extrapolated.** `pool_reasoning` (base rate 0.49) and
  `global_threat` (0.45) are standardised to p_ref = 0.025, a ~20× shift. Their
  raw MCC and J tell the same story directionally, but the standardised numbers
  should not be quoted.
- **One seed.** Every run here is `s42`. The sign is consistent across nine
  cells and three recipes, which is a stronger stability argument than a seed
  replicate would be, but the magnitudes are single-seed.
- **Two hooks, not a depth sweep.** `s4.conv1`, `s4.fc2_place` and
  `s4.fc2_select` are unmeasured. Where exactly the change of form happens is
  open.

## 6. Literature [AI-REASONED PROVISIONAL ANALYSIS]

The layer split is the expected shape, and prior work in board games found the
same axis without separating availability from isolation:

- **Lovering et al. 2022, *Evaluation Beyond Task Performance: Analyzing
  Concepts in AlphaZero in Hex* (arXiv:2211.14673)** — short-term concepts
  (immediately winning) probe best in the **final** layers, long-term concepts at
  ~50% depth; the authors propose layer-depth specialisation as "a potentially
  general principle across board-game agents". Probes only, and they state
  explicitly that they establish no causal link.
- **Du et al. 2025, *How GPT learns layer by layer* (arXiv:2501.07108)** —
  OthelloGPT: early layers carry **static board attributes** (edges, shape),
  deeper layers carry **dynamic gameplay features**. SAE-based, so the closest
  prior result to this one. Small model, mostly single-seed.
- **emergent-world-models-chess-llm** — board state peaks mid-to-late, skill at
  the final layer: "different functional roles".
- **metaothello-multiple-world-models** — early layers game-agnostic board state,
  later layers game-specific specialisation.

Counterpoint worth keeping: **scale-alone-does-not-improve-mechanistic** finds
layer depth does *not* predict unit interpretability. That is about neuron-level
interpretability in vision models rather than about which concept class lives
where, so it does not contradict the partition above — but it does argue against
generalising "later is more interpretable" beyond concept classes one has
actually measured.

What is new here relative to all four: they compare *probes* across layers. This
compares **probe and dictionary across layers on the same activations**, which is
what separates "the layer knows it" from "the dictionary can name it". The
methodological warning in **remarkable-robustness-llms-stages-inference** applies
directly — an SAE trained at one depth and one trained at another "are
decomposing structurally different objects, and their features are not
comparable", and most SAE work picks a layer by convenience.

## 7. Consequences

- **Report hook results per family, never pooled.** A whole-basis fc1-vs-conv2
  number averages a +0.28 and a −0.12 and means nothing.
- **The 3B.1 three-term decomposition is now the priority measurement.**
  Availability (LP on activations) → survival (LP on SAE codes) → isolation (best
  latent). Regime 2 says availability is flat across the bottleneck and isolation
  is not; the missing middle term says whether the fc1 dictionary *preserves*
  more or merely *presents* it better.
- **conv2 is the right hook for a factorisation experiment.** It is the case
  where the information is demonstrably present (LP 0.89–0.96) and the flat
  dictionary demonstrably fails to atomise it (efficiency 0.12–0.23). Any 3C
  variant that claims to fix dilution should be tested there first, because
  there the ceiling is known and the gap is largest.
- **`tiger` on conv2 should stop being used for basis comparison.** Every conv2
  tiger number is near the floor (0.10) because the concept is barely available
  there (LP 0.27–0.43), so a basis ranking computed on conv2 is ranking noise.
  That is the mechanism behind the hook-dependent hawk-vs-tiger flip.

Cross-references: panel fragility and the champYb conv2 dictionary are recorded
in the 2026-08-14 review (findings F2, F4); the availability/isolation framing
feeds Phase 3B.1 in [`phase-3.md`](phase-3.md).
