# Superseded 3A reports

Kept as a record, excluded from `3A_gate_summary.json` (which globs the parent
directory only) so a retired measurement cannot enter the G-3A tally.

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
