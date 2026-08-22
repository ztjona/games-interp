# Research diary

Long-form notes for the Quarto SAE interpretability project. This folder
is the companion to the top-level docs:

- [`../../RESEARCH-STATUS.md`](../../RESEARCH-STATUS.md) — current phase, hypothesis status, headline numbers.
- [`../../Quarto-specifications.md`](../../Quarto-specifications.md) — game, model, hooks, dataset catalog, naming.
- [`../BSP-schema-summary.md`](../BSP-schema-summary.md) — BSP schema for gorilla / hawk / hen / tiger.
- [`../../CLAUDE.md`](../../CLAUDE.md) — agent operating instructions and pipeline gotchas.

The top-level docs stay **short and current**. Specifics — full tables,
provisional interpretations, design rationale, dead-ends — live in
dated entries here.

## When to add an entry

- A sweep or experiment block produces results that deserve a per-category /
  per-config breakdown (more than ~3 rows of numbers).
- A pivot in research direction whose rationale would clutter
  `RESEARCH-STATUS.md` if inlined.
- A non-obvious pipeline contract, schema, or invariant lands in code.
- An AI-assisted interpretation of results that future readers should be
  able to evaluate sceptically (see "AI-reasoned content" below).

Do **not** add entries for routine status updates — those go in
`RESEARCH-STATUS.md` as a one-line "Project State" bullet that links
back here.

## File naming

- `YYYY-MM-DD_short-slug.md` — date is the day the result / design landed,
  slug is 2–4 words describing the topic.
- `phase-<id>.md` or `series-<id>.md` — running ledger when a phase
  produces multiple entries; date-stamp inside the file.

## AI-reasoned content

Many entries are written with AI assistance. To prevent over-confident
interpretations from compounding across sessions:

- **Numbers** lifted directly from JSONL / registry / json files are
  tagged `[DIRECT — from <source>]` and are reliable.
- **Interpretation** paragraphs are tagged
  `[AI-REASONED PROVISIONAL ANALYSIS]` or `[INFERENTIAL — read sceptically]`.
  Future readers — human or AI — should not treat them as established
  conclusions without independent verification against the cited
  numbers or a fresh look at the underlying data.

If you (a future agent) find yourself agreeing with a provisional claim
without checking it, that is the failure mode this convention exists for.

## Index

### Per-phase ledgers

| Phase | File | Topic |
|---|---|---|
| 1 (A–G) | [phase-1.md](phase-1.md) | Initial SAE family work on champAa: Anakin sweep (28 configs), conv2 LP, competence audit, H1/H6/H7/H8 establishment. |
| 2A | [phase-2A.md](phase-2A.md) | Conv2 architecture sweep on champAa (33 runs, Deep Brain 3× GPU, 2026-05-11). C01 winner; reporting standard (MCC + F1-lift) adopted. |
| 2B | [phase-2B.md](phase-2B.md) | Two-champion era (champS4 / champTa), CLOSED 2026-06-09. Five chapters: champS4 mini-sweep, LP rescoping, 25-config sweep + H9, Sweep H capacity null, close-out folding in the executed "2C" arc (tiger → anchored I/J → champVe). |
| 3 | [phase-3.md](phase-3.md) | OPEN (2026-06-09). Geometric concept structure: H10, dilution diagnostic, α/β allocation regime, geometry-aware SAE variants, causal patching. |

### Design notes

| Date | Entry | Topic |
|---|---|---|
| 2026-05-19 | [concept-targeted-saes](2026-05-19_concept-targeted-saes.md) | Design only, no implementation. Five candidate approaches (wider exp, higher k, anchored, E2E, matryoshka) to close the 11 % conv2/hawk SAE/LP wall; recommended ordering and pre-registered decision gates. |
| 2026-05-22 | [reframings-audit-tiger](2026-05-22_reframings-audit-tiger.md) | `tiger` BSP set design + implementation + **results** (36 BSPs, 6 categories): agent-relative threats and pool-reasoning. Pre-registered decision rule **fired** \u2014 tiger SAE/LP efficiency exceeds hawk by +25 to +51 pp; supervised SAE pivot now targets tiger, not hawk. |
| 2026-05-22 | [anchored-sae-champta-tiger](2026-05-22_anchored-sae-champta-tiger.md) | Plan (**executed**) for anchored jumprelu / batchtopk on champTa fc1, anchored against `tigerTa` BSPs. Anchor topology, lambda sweep, pre-registered decision gates. |
| 2026-05-25 | [anchored-sweep-ij-results](2026-05-25_anchored-sweep-ij-results.md) | **Results** of I/J anchored sweep (24 configs). Winner: I04 anchored-jumprelu lh=1.0 (+62% F1-lift). Decision gate PASS. Per-slot anchor analysis, cross-BSP coverage, J-series negative result, reporting standard for future anchored experiments. |
| 2026-06-09 | [geometric-pivot](2026-06-09_geometric-pivot.md) | Phase 3 founding design note. H10 (the SAE/LP wall is geometric), evidence from 5 newly-ingested papers + 2026-06-08 synthesis, pre-registered plan 3A–3D with gates G-3A / G-3C, thesis-risk framing. |
| 2026-06-18 | [champYb-onboarding](2026-06-18_champYb-onboarding.md) | champYb onboarding (Yb_hotChamp, hot-piece-shaped S4 subclass). New game module `quarto_s4_hot`, config, checkpoint integration, staged pipeline; how it slots into Phase 3 as a piece-safety-trained trunk. Groundwork only — no GPU run. |

### Explanations (formerly "supervisor advances")

`advances-supervisor/` was retired on 2026-08-12. It assumed one
self-contained snapshot per monthly meeting cycle, but each meeting
ended up presenting something too different for that template to fit,
so the files were pruned rather than kept as a half-followed convention.

Standalone explanatory pieces now live in
[`../explanations/`](../explanations/README.md) — written on request,
named by topic rather than by date, and updated in place. Use one when
something needs to be readable on its own by someone who has not read
the diary; use a dated diary entry when you are recording what happened
and when.

## How agents should use this folder

1. Read `RESEARCH-STATUS.md` first for the current phase and live hypotheses.
2. Consult this index for the entry whose topic matches the question.
3. Read that entry directly; **do not quote diary content into other docs**
   — link to the entry by filename instead, so updates propagate.
4. When numbers in an entry conflict with what is in code / registries today,
   trust the live data and update or annotate the entry.
