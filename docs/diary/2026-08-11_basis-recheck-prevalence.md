# Basis re-check at matched prevalence: the 2026-05-22 margin does not reproduce (2026-08-11)

Status: frozen self-contained record. Parent ledger: [`phase-3.md`](phase-3.md).
Definitions: [`../methods-reference.md`](../methods-reference.md) §1.
Numbers `[DIRECT]` from `scripts/basis_comparison.py`; interpretation
`[AI-REASONED PROVISIONAL ANALYSIS]`.

The 2026-05-22 reframing audit chose **tiger over hawk** as the supervision
target and redirected the programme. It compared bases whose prevalence differs
by ~7x, using prevalence-dependent metrics. This re-runs the comparison fairly.

## 0. Glossary

| term | meaning |
|---|---|
| **basis** | A whole BSP menu: `gorilla` (state-only), `hawk` (Nanda-style recount), `tiger` (agent-relative). |
| **triad** | The same underlying game fact described once per basis. The meaningful unit of comparison. |
| **MCC@pref** | MCC standardised to p_ref = 0.025 — the prevalence-fair number. |
| **J** | Youden's J = TPR − FPR, prevalence-invariant but precision-blind. |

## 1. Design

**One SAE, one set of codes** (`F04-champYb`, fc1 jumprelu). Same model, same
positions, same dictionary — **only the concept framing changes**. That removes
every confound except the one under test.

Features are selected by **`mcc_at_pref`**, matching the reported metric. A first
attempt selected by J and had to be discarded: at base rate 0.003 J happily picks
a latent firing on 10% of positions (J = 0.85, **precision = 0.07**). The
selection criterion must match the reported metric — J-selection flatters exactly
the rarest families, which are hawk's.

## 2. The triads [DIRECT]

| framing | base rate | raw MCC | J | **MCC@pref** | precision |
|---|---:|---:|---:|---:|---:|
| **LINE threat** | | | | | |
| gorilla `threat_line` | 0.0097 | 0.401 | 0.809 | **0.550** | 0.209 |
| hawk `reframed_completable` | 0.0031 | 0.242 | 0.851 | **0.552** | 0.074 |
| tiger `tiger_line_winnable` | 0.0230 | 0.166 | 0.223 | **0.173** | 0.149 |
| **SQUARE threat** | | | | | |
| gorilla `threat_square_2x2` | 0.0091 | 0.708 | 0.979 | **0.851** | 0.519 |
| hawk `reframed_sq_completable` | 0.0029 | 0.411 | 0.985 | **0.794** | 0.175 |
| tiger `tiger_square_winnable` | 0.0215 | 0.275 | 0.295 | **0.291** | 0.317 |

## 3. What changed [AI-REASONED PROVISIONAL ANALYSIS]

- **gorilla's apparent advantage over hawk was almost entirely prevalence.**
  Raw MCC says gorilla beats hawk by **+66%** (line) and **+72%** (square). At
  matched prevalence the line comparison **inverts** (0.550 vs 0.552 — a tie
  within noise) and the square gap shrinks to **+7%**. Roughly 90–95% of the
  apparent difference was base rate, not framing.
- **tiger's threat categories score far WORSE than both** — 0.173 / 0.291 versus
  0.55 / 0.79–0.85, a factor of 3–4. That is the **opposite direction** to the
  2026-05-22 conclusion.
- **The 2026-05-22 numerical margin therefore does not reproduce**, and its
  metric was prevalence-dependent across bases differing ~7x in base rate. It
  should not be cited as a quantitative result until re-run.

**Three caveats that stop this from being a refutation:**

1. The original audit's statistic was **SAE/LP efficiency** (a ratio involving
   linear-probe ceilings), not per-category MCC. That statistic is not reproduced
   here — it needs LP numbers this run does not have.
2. The original was measured on champS4 and **champTa**, whose dataset is now
   **quarantined** (random-play only). Those inputs cannot be trusted anyway.
3. This is a single SAE on a single champion.

**What survives untouched:** the *conceptual* argument for tiger — it is the only
agent-relative basis, i.e. decision-upstream rather than spectator — which is
independent of any metric and is what the causal track (3B-causal / 3D) needs.
The supervision pivot is not overturned; its numerical justification is.

**Absolute quality is poor for all three.** Precision runs 0.07–0.52, so even the
"winning" detectors fire far more often than the concept occurs. The triads rank
framings; they do not show any framing is well extracted.

## 4. Consequence for how BSPs are grouped

The triads are the informative view, and they cut **across** bases. Within tiger,
`tiger_line_winnable` (base 0.023) and `tiger_pool_safe_count` (base 0.748) share
nothing but a filename. Across bases, `threat_line` / `reframed_completable` /
`tiger_line_winnable` are three descriptions of one game fact and are directly
comparable once prevalence is handled.

**A basis is a packaging convention; a category is a concept.** Whole-basis
averages mix families spanning a 380x prevalence range and are not interpretable.
Report and compare **categories**, and use the basis only to say which framing a
category belongs to. Recorded in
[`../methods-reference.md`](../methods-reference.md) §2.

## 5. Reproduce

```bash
python scripts/basis_comparison.py --run-id=F04-champYb-s42-jumprelu-t64-exp8-s4.fc1 --champ=Yb
```
