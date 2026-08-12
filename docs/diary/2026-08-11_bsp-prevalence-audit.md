# BSP prevalence audit: the menu spans a 380x range, and it matters (2026-08-11)

Status: frozen self-contained record. Parent ledger: [`phase-3.md`](phase-3.md).
Metric definitions: [`../methods-reference.md`](../methods-reference.md) §1.
Numbers are `[DIRECT]` from `scripts/bsp_prevalence.py` on champYb (N=296,045);
interpretation is `[AI-REASONED PROVISIONAL ANALYSIS]`.

Asked after the prevalence work: *if the tiger conjunctions sit at ~2.5%, are our
BSPs too specific?* The answer needs the whole menu, not one family.

## 0. Glossary

| term | meaning |
|---|---|
| **prevalence / base rate** | `P(BSP = TRUE)` in the scored population. |
| **trivial F1** | `2p/(1+p)` — the F1 scored by a classifier that always says TRUE. The floor F1 cannot go below. |
| **VERY RARE** | prevalence < 0.01. MCC and R² are compressed toward 0 by the prevalence factor alone (§1.1 of the methods reference). |
| **TRIVIAL-F1** | prevalence > 0.4, so trivial F1 > 0.57 and F1 carries almost no information. |
| **DEGENERATE** | prevalence < 0.001 or > 0.999 — effectively constant. |

## 1. The table [DIRECT] — champYb, N = 296,045

| basis | category | n | min | median | max | trivial F1 | flag |
|---|---|---:|---:|---:|---:|---:|---|
| **gorilla** | `threat_line` | 40 | 0.0076 | **0.0094** | 0.0130 | 0.019 | VERY RARE |
| | `threat_square_2x2` | 36 | 0.0066 | **0.0094** | 0.0123 | 0.019 | VERY RARE |
| | `cell_attribute` | 64 | 0.1266 | 0.1629 | 0.2570 | 0.280 | |
| | `global` | 1 | 0.2336 | 0.2336 | 0.2336 | 0.379 | |
| | `cell_occupancy` | 16 | 0.2570 | 0.3211 | 0.4923 | 0.486 | TRIVIAL-F1 |
| | `game_phase` | 3 | 0.0174 | 0.4452 | 0.5374 | 0.616 | TRIVIAL-F1 |
| | `offered_piece` | 4 | 0.4942 | 0.5030 | 0.5146 | **0.669** | TRIVIAL-F1 |
| **hawk** | `reframed_sq_completable` | 36 | 0.0023 | **0.0028** | 0.0036 | 0.006 | VERY RARE |
| | `reframed_completable` | 40 | 0.0027 | **0.0030** | 0.0037 | 0.006 | VERY RARE |
| | `reframed_sq_count` | 36 | 0.0108 | 0.0180 | 0.0235 | 0.035 | |
| | `reframed_count` | 40 | 0.0128 | 0.0187 | 0.0290 | 0.037 | |
| | `reframed_sq_any_threat` | 9 | 0.0249 | 0.0337 | 0.0405 | 0.065 | |
| | `reframed_any_threat` | 10 | 0.0290 | 0.0342 | 0.0431 | 0.066 | |
| | `reframed_global` | 2 | 0.1452 | 0.2509 | 0.3567 | 0.401 | |
| **tiger** | `tiger_square_winnable` | 9 | 0.0186 | 0.0218 | 0.0243 | 0.043 | |
| | `tiger_line_winnable` | 10 | 0.0213 | 0.0228 | 0.0257 | 0.045 | |
| | `tiger_offered_completing_attr` | 4 | 0.0428 | 0.0444 | 0.0468 | 0.085 | |
| | `tiger_pool_winning_count` | 4 | 0.1281 | 0.4216 | 0.4656 | 0.593 | TRIVIAL-F1 |
| | `tiger_decision_global` | 5 | 0.1281 | 0.4656 | 0.8714 | 0.635 | TRIVIAL-F1 |
| | `tiger_pool_safe_count` | 4 | 0.1281 | **0.7480** | 0.8714 | **0.856** | TRIVIAL-F1 |

Totals: gorilla 53/164 VERY RARE and 9 TRIVIAL-F1; hawk **76/173 VERY RARE**;
tiger 0 VERY RARE but **8/36 TRIVIAL-F1**. Nothing is DEGENERATE.

## 2. Answering the question [AI-REASONED PROVISIONAL ANALYSIS]

**Not "too specific" — too WIDELY SPREAD.** The menu spans 0.0023 to 0.8714, a
**380x range**, and the problems sit at *both* ends:

- **Too rare to measure well (hawk).** `reframed_completable` and
  `reframed_sq_completable` sit at **0.003** — three positives per thousand
  positions, 76 of hawk's 173 BSPs. At that prevalence every prevalence-dependent
  metric is crushed toward zero regardless of how good the SAE is.
- **Too common to be informative (tiger pool counts, gorilla offered_piece).**
  `tiger_pool_safe_count` has median prevalence **0.748**, so its trivial F1 is
  **0.856**. `offered_piece` sits at 0.50 → trivial F1 **0.669**, which is the
  known 0.667 artefact, now confirmed exactly from the data.
- **tiger's conjunctions are the best-designed family for measurability** —
  0.019–0.047, rare enough to be non-trivial, common enough to estimate.

### 2.1 A banked result that needs re-reading

Phase 2B/3 reported the anchored I04 as *"strong on pool counts / decision_global
(F1 0.83–0.88)"*. With `tiger_pool_safe_count` at trivial F1 = **0.856**, an F1 of
0.879 is **barely above the always-say-yes baseline**. The MCC for that category
(0.702) shows the result is real — but the F1 framing was close to meaningless,
which is exactly the demotion argued in the methods reference.

### 2.2 A decision that needs re-checking

The **2026-05-22 reframing audit** chose tiger over hawk as the supervision
target, and that redirected the whole programme. It compared bases whose
prevalence differs by roughly **7x** (hawk `completable` ~0.003 vs tiger
conjunctions ~0.022). Since MCC is *not* prevalence-invariant, part of hawk's
apparent weakness may have been its base rate rather than its framing.

This does **not** overturn the decision — tiger also won on the agent-relative
argument, which is independent — but the numerical margin is not trustworthy as
stated. **Re-check with `coverage_mcc_at_pref` (p_ref = 0.025) or matched
prevalence** before the comparison appears in a chapter.

## 3. What to do about the menu [AI-REASONED PROVISIONAL]

- **Do not delete rare BSPs.** Rarity is a property of Quarto, not a design
  error: winnable lines *are* rare in competent play, and that is the phenomenon
  under study. Deleting them would delete the research question.
- **Do stop reporting F1 on the high-prevalence families** — `offered_piece`,
  `tiger_pool_safe_count`, `tiger_decision_global`, `tiger_pool_winning_count`,
  `cell_occupancy`, `game_phase`. Already covered by the metric policy; this
  table says exactly which categories it bites.
- **Treat hawk's `completable` families as measurement-limited.** At p = 0.003
  even 296k rows give ~890 positives; conclusions there are weak by construction
  and should carry an explicit caveat.
- **Report prevalence next to every per-category number.** `registry_query.py
  category` already prints a `base` column — it should be read, not skipped.

Regenerate any time with:

```bash
python scripts/bsp_prevalence.py --bsps=gorillaYb --bsps=hawkYb --bsps=tigerYb
```
