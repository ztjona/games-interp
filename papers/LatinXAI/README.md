# LXAI @ NeurIPS 2026 submission

**Deadline 2026-08-20, 23:59:59 AoE.** Double-blind, NeurIPS 2026 format
(desk-reject otherwise). Full paper 6–8 pp or extended abstract ≤4 pp,
references excluded. Submit PDF via
[OpenReview](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/LXAI).
At least one author must identify as LatinX. Archival (JLXAIR, gets a DOI) vs
non-archival is chosen *at acceptance*, not now.

Current draft: **8 pp body + 1 pp references** → full-paper category, at the
page limit. Any addition needs a matching cut; check with
`pdftotext -layout main.pdf -` and confirm "References" still starts at the top
of page 9.

## Build

```bash
python make_tables.py --metric=mean_mcc   # -> table_split.tex, table_availability.tex, numbers.json
python make_3a_tables.py                  # -> table_disjunction.tex, table_ladder.tex, numbers_3a.json
python make_figure.py                     # -> fig_availability.pdf   (reads numbers.json)
python make_figure_3a.py                  # -> fig_disjunction.pdf    (reads the dilution reports)
latexmk -pdf main.tex
```

## Submission abstract (plain unicode, max 2000 chars)

The OpenReview form takes the abstract as **plain unicode text, 2000 characters
max** - no LaTeX. Regenerate it from `main.tex` after any abstract edit, so the
form text and the PDF cannot drift:

```bash
python plain_abstract.py main.tex --out=abstract.txt          # unicode (paste this)
python plain_abstract.py main.tex --ascii --out=abstract-ascii.txt   # fallback
```

Current: **1534 / 2000 characters**, 247 words. The converter resolves LaTeX to
unicode rather than deleting it (`\times` -> U+00D7, ` ``...'' ` -> U+201C/D,
`--` between digits -> U+2013) and reports any leftover markup, so a stray
`\emph{}` cannot reach the form silently.

The LaTeX abstract carries **no `\textbf` or `\emph`** either: OpenReview does
not render them, and keeping the typeset abstract and the submitted text
identical in wording and styling means a reader comparing the two sees no
difference. Math (`$...$`, `\times`) and `\%` stay in `main.tex` - they are
content, not styling - and the converter resolves them to unicode for the form.

`make_tables.py` reads `saes/quarto/eval_registry.json` and the
`*_sae-lp-efficiency.json` reports, and rolls categories up with the project's
own `aggregate_per_category_by_family` (which reads `concept_family` off the
BSP schema). `make_3a_tables.py` reads the frozen Phase 3A dilution reports in
`saes/quarto/analysis/`. Nothing is transcribed by hand, so the paper cannot
drift from the artefacts.

The two scripts share one home for the anonymisation maps: `AGENT_LABEL` and
`BASIS_LABEL` are defined in `make_tables.py` and **imported** by
`make_3a_tables.py`. Do not restate them — a label with two homes drifts.

## What the paper measures

Four quantities, three instruments:

| symbol | what | source |
|---|---|---|
| availability $A$ | linear probe MCC on raw activations | `scripts/linear_probe_baseline.py` → `*_sae-lp-efficiency.json` |
| isolation $I$ | best single latent MCC | `sae_eval.py` → `eval_registry.json` |
| retention $R$ | held-out $R^2$ of the concept on the top-64 latents, vs the random-model floor | `scripts/dilution_diagnostic.py` → `*_dilution-*.json` (`asymptote_r2` / `random_asymptote_r2`) |
| concentration $S$ | share of $R$ in the best latent | same reports (`solo_frac`) |

$k_{90}$ in §4.5 is the reports' `knee_k`. §4.4/§4.5 use the *unsupervised*
dictionaries on champYb / `s4.fc1` only (9 checkpoints, 3,209 concept
verdicts) — the one (champion, hook) whose pinned concepts have modal
`knee_k = 1`, which is the precondition the tiling reading is conditional on.

## Decisions baked into the numbers

- **Table 1 uses raw `mean_mcc`, not `mcc_at_pref`.** Within a matched cell both
  arms share positions and concept, so prevalence is identical by construction
  and standardisation is unnecessary. It is also unavailable: the six champYb
  rows in this panel predate the `mcc_at_pref` backfill. Table 2 crosses
  champions, so it uses `mcc_at_pref` throughout (all six efficiency reports
  carry it).
- **conv2 for champYb is the repaired `K04`, not `E05`.** `E05` is degenerate
  (99.0% dead, FVU 0.110) because BatchTopK lacked the Gao dead-latent aux loss
  until 2026-08-15. Using it would flatter the paper's thesis. `make_tables.py`
  substitutes `K04` (FVU 0.0066) and prints a sign-flip robustness check:
  2 of 15 families flip, both per-cell, both *toward* the partition; all 12
  relational families unchanged (reported in §4.6).
- **champTa rows are post-rebuild** (2026-08-12, all four opponent modes). The
  "provisional" flag in `RESEARCH-STATUS.md` is stale for these rows —
  `_dataset_status.json` records the rebuild and the SAE-sweep retrain.
- **`DISJUNCTS` in `make_3a_tables.py` is a map from BSP category to the number
  of disjuncts that category ORs over, known *by construction* from the BSP
  definitions** — 1 for pinned, 4 for `*_any_threat`, 8 for tiger's `*_winnable`
  (because `tiger == OR(hawk ∪ hen)`, verified with zero violations
  2026-08-21). It is a prediction the $k_{90}$ distribution is compared
  against, never fitted. `tiger_offered_completing_attr` is **excluded**: it ORs
  over locations rather than poles, so the category does not fix its count.
- **Anchored (`I`-series) dictionaries are excluded from every §4.4–4.5 number**
  (`is_unsupervised`); they appear only as the positive control in §4.6.
- **Table 4 (`table_ladder.tex`) pools the three panel bases** per (agent,
  layer). `hen` is not in `3A_panel.json` — it is the mirrored arm, selected by
  copying hawk's choices — so it is absent from that table but present in
  Table 3.
- **The 8× capacity ladder is one seed (s42)** across `K05` exp8 → `K11` exp32 →
  `K12` exp64, so $d_{\text{dict}}$ is the only thing varying. `K12` was only
  run on gorilla and tiger.
- **Every number is from rule 3A.5** (114 cells, 0 provisional, 100% random
  control coverage). Verdicts are a pure function of the stored metrics, so
  `dilution_diagnostic.py reclassify` regenerates them without the `_h` caches.

## Numbers audited against the artefacts

Corrected on 2026-08-25 during the Phase 3A update, after checking every
assertion against `numbers.json` / `numbers_3a.json`:

- pooled pinned capture is **73.2% over n=2,736** (76.7%/n=1,672 is the
  *state*-pinned subgroup only);
- the panel spans **3 architectures in 14 unsupervised recipes and 4 widths**
  (4,096–32,768), not "six recipes";
- the clipped $k_{90}$ bin is "**at 20 or beyond**" (≥20 = 21.6%), not "beyond 20";
- the disjunction contrast is stated on the arm that holds agent-relativity at
  "no" in both rows (`count` pinned/no 81.2% vs `count` OR-4/no 0.0%);
- "captured four times in five" belongs to `count`'s **532** agent-relative
  pinned concepts (79.0%), not to `reframed_completable` alone (66.4%).

## Anonymity

Internal jargon is also identifying, so the paper renames it:
champions `Ta/Ve/Yb` → **M1/M2/M3** (ordered by head-to-head strength), bases
`gorilla/hawk/hen/tiger` → **state/count/count⁻/agent-rel.** The maps live in
`make_tables.py` (`AGENT_LABEL`, `BASIS_LABEL`). Figure subtitles say "agents",
not "champions". The CFP forbids revealing *country* as well as names and
institutions.

`\usepackage[dblblindworkshop]{neurips_2026}` keeps anonymity on. Do **not**
add `final`, `preprint` or `nonanonymous` before acceptance. Camera-ready
switches to the LXAI style file, which is published only after acceptance:
`\usepackage[sglblindworkshop, final]{latinx_2026}`.

## Source

Hook specialisation + probe: [`docs/diary/2026-08-14_hook-specialisation.md`](../../docs/diary/2026-08-14_hook-specialisation.md);
conv2 repair: [`docs/diary/2026-08-15_dead-feature-revival.md`](../../docs/diary/2026-08-15_dead-feature-revival.md) §4.2;
**retention / concentration / the disjunction wall**:
[`docs/diary/2026-08-25_3A-final-report.md`](../../docs/diary/2026-08-25_3A-final-report.md)
(§2 validity audit, §3 stability, §4 the gate, §5 the disjunction result, §6 the
correction to the agent-relativity reading);
`hen` basis: [`docs/diary/2026-08-21_hen-basis-and-rule-3A4.md`](../../docs/diary/2026-08-21_hen-basis-and-rule-3A4.md);
metric definitions: [`docs/methods-reference.md`](../../docs/methods-reference.md) §1 and §3.
