# Phase 3A first run: results, a failed calibration check, and rule 3A.2 (2026-07-27)

Status: frozen self-contained record. Parent ledger: [`phase-3A.md`](phase-3A.md).
Method spec (frozen, still accurate for everything except the `captured`
branch): [`2026-07-21_3A-dilution-diagnostic.md`](2026-07-21_3A-dilution-diagnostic.md).
Numbers from the report JSONs are `[DIRECT]`; interpretation is
`[AI-REASONED PROVISIONAL ANALYSIS]`.

---

## 0. Glossary

**Concepts and phases**

| term | meaning |
|---|---|
| **BSP** | Board State Property — a hand-defined ground-truth binary concept about a Quarto position. The label tensor is the answer key an SAE feature is scored against. |
| **gorilla / hawk / tiger** | The three BSP *bases* (concept menus). gorilla_164 = state-only properties. hawk_173 = the same threats recounted Nanda-style. tiger_36 = agent-relative ("can *I* win with what I hold"). The suffix (`tigerYb`) says which champion's position distribution the labels were computed on. |
| **threat_line / threat_square_2x2** | gorilla threat categories: does a line / 2x2 square have three pieces sharing an attribute, so one more completes it. 40 + 36 BSPs. |
| **tiger_line_winnable / tiger_square_winnable** | The tiger counterparts, agent-relative: can the player to move actually complete that line / square with the piece they were handed. 10 + 9 BSPs. |
| **conjunction** | A concept that is only true when several conditions hold together (three-in-a-line AND a shared attribute AND holding the completing piece). H10's claim is that these need several dimensions, which a flat dictionary of independent atoms cannot hold in one atom. |
| **base rate** | Fraction of positions where a BSP is TRUE. Rare BSPs (~0.02) depress F1 and R2 mechanically, so two champions are only comparable at matched base rates. |
| **H10** | Phase 3 founding hypothesis: the SAE/LP wall is *geometric* — conjunctive concepts occupy multi-dimensional structure that flat dictionary atoms dilute or tile. |
| **G11** | The epiphenomenality risk (Balogh & Jelasity): a probe can decode a concept the model never uses. Decodability != causality. |
| **LP** | Linear probe — the unconstrained linear ceiling on the same activations. |
| **anchored SAE** | A *supervised* SAE (`_AnchorMixin`): some dictionary slots are trained to align with named BSPs. I03/I04 are the anchored runs; F01/F04/E01/E05 are unsupervised. |

**3A metrics** (all also embedded in every report JSON under `glossary`)

| metric | range | ideal for "captured" | meaning |
|---|---|---|---|
| `top_phi` | -1..1 | near ±1 | Signed phi (= MCC for a 2x2 table) between the single best-associated latent's firing and the BSP. The strongest single-latent association that exists in the dictionary. |
| `asymptote_r2` | ≤1 | high | Held-out R2 predicting the BSP from **all 64** candidate latents together. Total recoverable information, regardless of how many latents it takes. |
| `null_r2` | ~0 | 0 | Same regression with shuffled labels — the overfitting floor. `asymptote_r2 - null_r2` is the real signal. |
| `solo_r2` | ≤1 | ≈ `asymptote_r2` | Held-out R2 from the single best latent alone. |
| **`solo_frac`** | 0..1 | ≥ 0.70 | `solo_r2 / asymptote_r2` — the share of all recoverable signal ONE latent already carries. **The primary concentration metric.** |
| `knee_k` | 1..64 | 1-2 | Smallest number of latents reaching 90% of `asymptote_r2`. Intuitive but brittle (see §3). |
| `community_size` | 1..64 | 1-3 | Latents in the co-firing community around the top latent. Counts **redundancy** — near-duplicates inflate it. |
| `intrinsic_dim` | 1..`community_size` | ~1 | PCA participation ratio of the community's codes on positive positions. ~1 = one real direction; 5 = five effective dimensions. |
| `support_overlap` | 0..1 | — | Mean pairwise Jaccard of community firing supports. HIGH = same positions (redundant → diluted); LOW = disjoint (shattered → tiled). |
| `neg_coupling_frac` | 0..1 | — | Share of within-community couplings that are negative (latents suppressing each other). High + low overlap = tiled. |
| `geometric_frac` | 0..1 | — | `(n_diluted + n_tiled) / n_threat_bsps`. **Gate G-3A**: ≥ 0.50 → 3C proceeds. |

**Verdicts** — what each implies for the fix

| verdict | meaning | implication |
|---|---|---|
| **absent** | Codes carry no more signal than the null. | Not a capacity problem — change the hook, or go E2E/supervised. An architecture change on these activations cannot help. |
| **captured** | Recovered by a small, low-dimensional latent set. | Nothing to fix; the SAE already has it. |
| **diluted** | Recoverable only by aggregating many **overlapping** latents (feature splitting). | *Geometric.* The information IS there — an aggregating / hierarchical readout can get it. |
| **tiled** | Recoverable but spread over near-disjoint, **competing** latents (shattered manifold). | *Geometric.* Needs a manifold-aware / bilinear readout. |

"**Geometric**" = diluted **or** tiled = "the information is in the code, the flat
dictionary just isn't presenting it in one atom". That is the H10 outcome and the
precondition for 3C. It is not a synonym for "bad".

---

## 1. What was run [DIRECT]

`runners/3A-dilution.ps1`, 12 run x BSP-set combinations over champTa / champVe /
champYb, unsupervised (F04 fc1, E05/E01 conv2) plus the anchored positive
controls (I03/I04). BSP sets: **gorilla and tiger** — hawk was not included.

**488 concept-verdicts**, which is (BSP x run), not 488 distinct concepts:

- 76 gorilla threat BSPs (`threat_line` 40 + `threat_square_2x2` 36) x 4 gorilla runs = 304
- 23 tiger threat BSPs (`line_winnable` 10 + `square_winnable` 9 + `offered_completing_attr` 4) x 8 tiger runs = 184

so **99 distinct concepts**, each diagnosed in several SAEs. The other BSP
categories (cell occupancy, attributes, pool counts, `offered_piece`, phase) are
deliberately excluded: `--categories auto` keeps only the threat/relational
families, because those are the ones the wall is about.

Tally under the shipped rule: **469 diluted, 13 tiled, 5 absent, 1 captured**.
`geometric_frac` 0.93-1.00 on every run; every gate read "3C-proceeds".

## 2. Why tiger is the focus, and why hawk was missing [AI-REASONED PROVISIONAL]

gorilla *was* included (4 of the 12 runs). Only **hawk** was absent. The reason
is a pre-registered decision from the 2026-05-22 reframing audit: tiger's SAE/LP
efficiency beat hawk's by +25 pp (conv2/Ta) to +51 pp (conv2/S4), so the
*supervision pivot* targets tiger and hawk was deprioritised as an anchoring
target. tiger is also the only basis that is **agent-relative** — decision-
upstream rather than spectator — which is what the causal track (3B-causal / 3D)
needs.

That reasoning is about *what to supervise*, and it does **not** transfer to
"which bases can be diluted". Whether the failure mode is basis-dependent is a
free question here — the codes are already cached and 3A is CPU-only. hawk is
therefore added to the panel (see §5). If hawk threats come out diluted and
gorilla threats come out captured on the same champion and the same SAE, the
verdict is a property of the *concept framing*, not of the dictionary, which
would be a genuinely new result.

## 3. The calibration check failed [DIRECT + AI-REASONED]

The method spec §6 pre-registered: *"the champTa/Ve run should confirm they place
the supervised I04 anchored slots at `captured` and the F04 threat concepts at
`diluted`/`tiled`"*. It did not. `I04-champYb`, whose anchored slots reach
tiger F1 0.774 (the best extraction in the project), came out **22/23 diluted**
— with `community_size` 2, `intrinsic_dim` 1.01 and `top_phi` 0.77, i.e. a
single clean latent per concept.

Two independent causes, both in the `captured` branch of `classify`:

1. **`knee_k <= 2` is brittle.** `knee_k` is where the held-out R2 curve first
   reaches 90% of its final value, and that final value keeps creeping upward by
   ~2% out to k=64 on pure noise. On `I04-champYb` the *first* latent already
   carries 79% of everything recoverable, but reaching 90% takes 17 latents.
2. **`community_size <= 3` conflates redundancy with necessity.**
   `F04-champYb` on gorillaYb has `knee_k = 1` and `solo_frac = 0.99` — one
   latent, all of it — yet 9 near-duplicate latents co-fire, so it failed the
   size test and all 76 concepts were called diluted.

Underneath both: the rule ANDed a statistic measured over the **candidate list**
(`knee_k`) with one measured over the **community** (`community_size`) — two
different scopes.

Consequence: `captured` fired **1 time in 488**. A gate whose non-geometric arm
is effectively unreachable is not a test, and a pre-registration that cannot fail
does not earn any credit when it "passes".

## 4. Rule 3A.2 [DIRECT]

`captured` becomes a two-condition branch (`lib/sae/dilution.py:classify`):

```
(a) knee_k <= captured_k AND community_size <= captured_size          [3A.1, kept]
(b) solo_frac >= captured_solo_frac                                   [new]
    AND (intrinsic_dim <= captured_idim OR knee_k <= captured_k)
```

with `captured_solo_frac = 0.70`, `captured_idim = 2.0`. `solo_frac` is the
necessary term — the concept must genuinely be concentrated in one latent — and
the second term confirms low dimensionality by *either* measure, because both are
one-sided (`intrinsic_dim` inflated by redundant members, `knee_k` by noise
creep). (a) is retained unchanged, so **no verdict that was `captured` can stop
being `captured`**; 3A.2 is strictly more permissive.

`solo_frac` and `solo_r2` are now first-class reported metrics, and every report
embeds the glossary above under a `glossary` key. `DilutionConfig.rule_version`
stamps each report so a stored verdict is always traceable to the rule that made
it. Regression tests pin all four shapes from §3 (`tests/test_dilution.py`, 20
tests).

**Verdicts are a pure function of stored metrics**, so a threshold change never
needs the 4.6-38 GB `_h` caches:

```powershell
python scripts/dilution_diagnostic.py reclassify saes/quarto/analysis/*_dilution-*.json --dry-run
```

### What 3A.2 does to the existing 12 reports [DIRECT, dry-run preview]

| run | BSP set | geom 3A.1 | geom 3A.2 | gate |
|---|---|---:|---:|---|
| F04-champTa fc1 | tigerTa | 1.00 | **1.00** | proceeds |
| F04-champTa fc1 | gorillaTa | 1.00 | **1.00** | proceeds |
| E05-champTa conv2 | tigerTa | 1.00 | **1.00** | proceeds |
| E05-champVe conv2 | tigerVe | 1.00 | **1.00** | proceeds |
| E01-champVe conv2 | gorillaVe | 0.93 | **0.91** | proceeds |
| E05-champYb conv2 | tigerYb | 1.00 | **1.00** | proceeds |
| F04-champYb fc1 | tigerYb | 1.00 | **0.96** | proceeds |
| F04-champYb fc1 | gorillaYb | 1.00 | **0.11** | **deprioritized** |
| I03-champVe anch | gorillaVe | 1.00 | **0.72** | proceeds |
| I04-champTa anch | tigerTa | 1.00 | **1.00** | proceeds |
| I04-champVe anch | tigerVe | 1.00 | **1.00** | proceeds |
| **I04-champYb anch** *(positive control)* | tigerYb | 0.96 | **0.26** | **deprioritized** |

The gate now fires both ways, and the positive control behaves as
pre-registered.

## 5. What this changes about the finding [AI-REASONED PROVISIONAL]

- **G-3A still passes where it matters.** Every *unsupervised* run on **tiger**
  stays 0.96-1.00 geometric. The agent-relative conjunctions are diluted in every
  champion and at both hooks. 3C proceeds, unchanged.
- **New and not visible before: champYb's unsupervised fc1 SAE has essentially
  solved the gorilla state-threats** (0.11 geometric = 68/76 captured), while
  champTa's is at 1.00 diluted on the same concepts. That is a large,
  previously-invisible champion difference, and it makes the wall *specific*:
  what remains diluted on Yb is tiger, not threats-in-general.
- **The wall is now localised to one cell**: agent-relative conjunctions, on the
  strongest champion, unsupervised. That is exactly the 3C target, and it is a
  much sharper pre-registration than "beat 0.255".
- **The strongest H10 evidence is continuous, not categorical.** On tiger
  conjunctions, Ta→Yb: mean `asymptote_r2` 0.211→0.379 (line) and 0.294→0.534
  (square) — the information in the code went *up* ~80% — while the single-feature
  F1 went *down* (0.234→0.180, 0.300→0.268), on a target whose base rate also
  halved (0.05→0.02, 0.04→0.02), which makes the R2 rise conservative. Measured
  on the *same* unsupervised SAE, information-up-while-extraction-down is the
  definition of dilution. Report `solo_frac` / `intrinsic_dim` / `top_phi`
  alongside the 4-way verdict; the thresholded label throws this away.

## 6. Open gaps [DIRECT]

- **No random-model control.** `random_control: null` in all 12 reports, so
  `absent` rests on the permutation null alone. The spec §2 calls the
  random-model SAE required for a trustworthy `absent`, and the reporting
  standard's clause 2 wants it everywhere. The random-model *activations* exist
  for every S4 champion (`s4.{fc1,conv2}_amalgam_{ta,ve,yb}_random_activations.pt`)
  but **no SAE has been trained on them** for Ta/Ve/Yb — only the champAa-era
  A01/A02/G0x controls. That is ~6 short training runs.
  `runners/3A-dilution.ps1` has the `$RANDOM_CONTROLS` map wired and empty.
- **champVe had no unsupervised fc1 row** on tiger — `F04-champVe` was never
  evaluated against `tigerVe` (the only unsupervised tigerVe entry in the
  registry is the conv2 E05). This is why the first cross-champion read could
  only contrast Ta vs Yb. The revised runner adds it and regenerates the missing
  eval via `sae_eval --force`.
- **hawk missing** — added to the revised panel (§2).
- **Cross-champion comparisons are on each champion's own self-play
  distribution**, so base rates differ (tiger conjunctions: Ta 0.04-0.08, Yb
  0.02-0.04). The unified pool (`scripts/unify_positions.py`, `gorilla156k`)
  exists for exactly this and has **not been built on Deep Brain** — no
  `*156k*` labels or activations on disk.

## 7. Revised panel (17 runs) [DIRECT]

Three champions x {fc1 unsupervised, conv2 unsupervised, anchored control} x
{gorilla, hawk, tiger} on the fc1 winners. Dry-run confirms all 17 runnable;
`F04-champVe` regenerates its `_h` and its missing registry entries on the way.
