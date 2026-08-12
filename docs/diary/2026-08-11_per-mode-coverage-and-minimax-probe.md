# Per-mode coverage: the on-policy "gap" is a prevalence artefact (2026-08-11)

Status: frozen self-contained record. Parent ledger: [`phase-3.md`](phase-3.md).
Metric definitions: [`../methods-reference.md`](../methods-reference.md).
Numbers are `[DIRECT]`; interpretation is `[AI-REASONED PROVISIONAL ANALYSIS]`.

> **This entry records a claim that was made and then retracted within the same
> session.** The raw per-mode numbers (§1) appear to show SAE coverage collapsing
> on the positions champions actually play. That reading was wrong. Once
> prevalence is matched (§3) the effect disappears and reverses slightly. The
> wrong version is kept here because the artefact is easy to walk into and the
> control that catches it is the reusable lesson.

## 0. Glossary

| term | meaning |
|---|---|
| **opponent mode** | How a position was generated: `random_v_random`, `model_v_random` / `random_v_model` (champion vs random bot as P1 / P2), `model_v_model` (self-play). |
| **on-policy** | Positions the champion actually reaches. `model_v_model` is purest. |
| **prevalence / base rate** | Fraction of positions where a concept is TRUE. |
| **MCC** | Matthews correlation coefficient. Range −1…1, 0 = chance. **Not** prevalence-invariant — see §2. |
| **pooled** | Coverage computed over the whole amalgam at once (what every banked number is). |

## 1. The raw per-mode numbers [DIRECT] — and why they mislead

Rows tagged by originating opponent mode, matching recomputed per mode, mean
best-MCC over the 23 tiger threat concepts:

| population | base rate | F04-Yb fc1 | E05-Ve conv2 | I04-Yb fc1 (anchored) |
|---|---:|---:|---:|---:|
| pooled | 0.026–0.028 | 0.227 | 0.146 | 0.724 |
| `random_v_random` | 0.050 | 0.270 | 0.188 | 0.754 |
| `model_v_random` | 0.017–0.019 | 0.223 | 0.132 | 0.703 |
| `random_v_model` | 0.017–0.019 | 0.217 | 0.130 | 0.699 |
| **`model_v_model`** | 0.013–0.014 | **0.167** | **0.111** | **0.647** |

Read naively this says coverage is 38–41% worse on the positions the champion
actually plays. **It does not.** Notice that the base rate falls monotonically
in lockstep with the MCC, from 0.050 to 0.013.

## 2. MCC is NOT prevalence-invariant [DIRECT]

This corrects an assertion made elsewhere in this project's docs. MCC is *far*
less prevalence-sensitive than F1, and it is 0 for any constant predictor — but
for a detector of **fixed quality** it still falls as positives get rarer:

$$\mathrm{MCC}=\frac{\sqrt{p(1-p)}\,(a+b-1)}{\sqrt{[ap+(1-b)(1-p)]\,[b(1-p)+(1-a)p]}}$$

with sensitivity $a$, specificity $b$, prevalence $p$. For $a=0.50$, $b=0.99$:

| p | 0.050 | 0.030 | 0.020 | 0.013 |
|---|---:|---:|---:|---:|
| MCC | 0.585 | 0.539 | 0.492 | 0.437 |

**−25% from an identical detector**, purely from prevalence. That alone covers
most of the apparent gap in §1.

## 3. The decisive test: match prevalence and N exactly [DIRECT]

For each concept, draw the same number of positives and the same number of
negatives from both populations, so prevalence and N are identical by
construction; 5 resamples. `scripts/investigate_mode_gap.py`.

| run | matched prevalence | `random_v_random` | `model_v_model` | residual gap |
|---|---:|---:|---:|---:|
| F04-Yb fc1 (unsup) | 0.0132 | 0.1566 | 0.1675 | **+7.0%** |
| E05-Ve conv2 (unsup) | 0.0143 | 0.1026 | 0.1113 | **+8.5%** |
| I04-Yb fc1 (anchored) | 0.0132 | 0.6434 | 0.6485 | **+0.8%** |

**The gap vanishes and reverses.** Coverage is *slightly better* on-policy —
the intuitive direction. Repeating against self-play-**only** rows (overlap
removed) gives the same answer (+4.7% on F04-Yb).

Two other candidate confounds were checked and are not the explanation:

- **Population overlap** — only **3.5–4.4%** of self-play rows are also reachable
  by random play. The populations are genuinely distinct.
- **Game phase** — mean pieces on board is 5.87 (`random_v_random`) vs 5.19–5.67
  (`model_v_model`), essentially the same distribution (median 6 vs 5–6). Not a
  late-game-complexity effect.

## 4. What this means [AI-REASONED PROVISIONAL ANALYSIS]

- **There is no on-policy coverage deficit.** SAE coverage of tiger threats is
  the same, marginally better, on the positions the champion actually plays.
- **The real lesson is about the metric, not the modes.** Any comparison across
  populations with different base rates — opponent modes, champions, datasets —
  needs **prevalence matching**. MCC is a large improvement on F1 but does not
  remove the need for it. This applies directly to the cross-champion story:
  champTa's tiger base rate (0.045) is ~2x champVe/champYb's (0.023), so even
  MCC comparisons between them are not automatically fair.
- **The random-play fraction of the training mix is not harming the dictionary.**
  Combined with the fact that random play carries ~4x the threat density, the
  case for keeping a broad training mix is strengthened, not weakened.
- **Supervision still transfers best**: the anchored run has the smallest
  residual gap (+0.8%), i.e. it is the most population-agnostic. Weak evidence,
  but consistent with anchored features being less distribution-specific.

## 5. MinimaxBot timing probe [DIRECT]

`quartopy.bot.minimax_bot.MinimaxBot` (alpha-beta, depth 2 default) exists and
subclasses the pipeline's `BotAI`. 100 games per pairing, single process, CPU:

| pairing | s/game | 10,000-game run | vs `random_v_random` |
|---|---:|---:|---:|
| `random_v_random` (reference) | 0.0004 | ~0.00 h | 1x |
| `model_v_random` (reference) | 0.017 | 0.05 h | 45x |
| `minimax(d=2)_v_random` | 0.089 | **0.25 h** | 240x |
| `minimax(d=2)_v_model` | 0.528 | **1.47 h** | 1414x |

Affordable; modes are file-disjoint and can run concurrently. Minimax sits in
Tier 2 on design grounds (roster independence), not budget.

Caveat: the probe loop carries a guard for a terminal
select-from-empty-storage state that the production loop does not have, and it
records 16 positions/game versus ~10.5 in real generation. Timing is a
conservative upper bound; position counts are not comparable.

## 6. Saturation: is 10,000 games "enough"? [DIRECT]

| games | champYb self-play | `random_v_random` |
|---:|---|---|
| 500 | 4,635 (9.27 new/game) | 4,840 (9.68 new/game) |
| 2,500 | 22,506 (8.89) | 23,317 (9.16) |
| 5,000 | 44,065 (8.62) | 45,349 (8.81) |
| 10,000 | 86,181 (**8.42**) | 88,524 (**8.54**) |

Nowhere near saturation — marginal yield falls ~9% across the whole run.
Expected against a legal space of 2.07e16 (6.7e12 after the 3,072-element
symmetry group). 10,000 games is a **budget** choice, not a coverage one; keep
it for every new mode for **uniformity**, since the champTa defect was a
composition asymmetry.

## 7. Method note for the future

The control that caught this — matching prevalence and N and resampling — is
cheap and should be standard whenever two populations are compared. Reusable:
`scripts/investigate_mode_gap.py`. The failure mode is seductive because the
uncorrected numbers were monotone, replicated across two hooks, two
architectures, two champions, and both supervised and unsupervised training.
**Consistency across conditions is not evidence against a confound when the
confound is present in every condition.**
