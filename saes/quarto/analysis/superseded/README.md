# Superseded 3A dilution reports

Archive of reports that were correct for the panel that produced them, kept as
the audit trail for earlier readings. Established 2026-08-17 for `E05-champYb`;
extended 2026-08-24 with the 14 reports the top-3 panel and the conformance
filter replaced.

These are NOT members of the current panel
(`../3A_panel_runlist.json`), and they live here rather than in `analysis/`
because `summarize_3a_gate.py` and `verdict_stability.py` glob that directory —
leaving them alongside the live panel silently mixed three populations into one
summary (the 2026-08-24 gate summary reported "128 runs" for a 114-entry panel).

Nothing here is deleted, and nothing here should be quoted as a current result.

## Why each was superseded

| report | superseded by |
|---|---|
| `F04-champYb …` (gorillaYb, hawkYb, tigerYb) | **The 2026-08-17 headline numbers** (gorilla 0.03 / hawk 0.17 / tiger 0.96). `F04-champYb` is not in the top-3 panel for any champYb fc1 cell; its condition (jumprelu-t64-exp8) is represented by `K05-champYb-s44`, the seed that scores best on threat-family MCC. Kept because the framing-effect result was first measured here. |
| `F00`, `F03` (champYb fc1) | Dropped 2026-08-23 by `--canonical-only`: batchtopk trained with `aux_loss_weight = None`, the configuration 2026-08-15 showed is genuinely collapsed (F00: FVU 0.086, 98.1% dead). |
| `K02-champYb` (gorillaYb, hawkYb, henYb) | Same conformance filter — batchtopk without the tied init. |
| `C01-champYb` (tigerYb) | Dropped as a **duplicate recipe** of `K07-champYb` (both topk k16 exp8 conv2, both s42), retrained after the tied init landed. Not a conformance failure — a dedup failure, fixed by keying `condition_of` on hyperparameters instead of the run-id prefix. |
| `F01-champYb` (gorillaYb) | Duplicate recipe, same fix. |
| `E05-champYb` (tigerYb) | Retired 2026-08-17 — see its own section below. |
| `E01-champVe`, `E05-champTa`, `F04-champTa`, `F04-champVe` | Members of the single-member panel that the 2026-08-23 top-3 panel replaced. |

## `E05-champYb-…-s4.conv2_dilution-tigerYb.json` (retired 2026-08-17)
The champYb conv2 panel member until the conformance work. `E05-champYb` is a
**collapsed dictionary**: 99.0% dead, 41 alive latents — fewer than the
diagnostic's `top_k = 64`, so it could not even fill the candidate list — FVU
0.110 against 0.043 / 0.051 for the same recipe on champTa / champVe, and ranked
10th of 11 champYb conv2 runs by coverage MCC.

Its verdict (`geometric_frac` 0.17, `3C-deprioritized (absent)`) described the
broken dictionary, not champYb's conv2 representation. Replaced by `K04`, the
same recipe trained in canonical form (FVU 0.0066, 174 alive), which returns
**1.00 on both tigerYb and gorillaYb** — i.e. fully geometric, in line with
champTa and champVe.

The file is retained because "what a collapsed dictionary looks like under 3A"
is itself useful: it is the only cell where the rule-3A.3 learned-signal floor
fired at scale (19 of 23 concepts to `absent`), which is what a genuinely
unlearned signal looks like in this diagnostic.

## Provenance

All were reclassified to rule 3A.5 before archiving, so their verdicts and
bands are on the same footing as the live panel and can be compared directly.
