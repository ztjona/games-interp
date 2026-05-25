# Research diary

Long-form notes for the Quarto SAE interpretability project. This folder
is the companion to the top-level docs:

- [`../../RESEARCH-STATUS.md`](../../RESEARCH-STATUS.md) — current phase, hypothesis status, headline numbers.
- [`../../Quarto-specifications.md`](../../Quarto-specifications.md) — game, model, hooks, dataset catalog, naming.
- [`../../BSP-schema-summary.md`](../../BSP-schema-summary.md) — BSP schema for gorilla and hawk.
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
| 2B | [phase-2B.md](phase-2B.md) | Two-champion era (champS4 / champTa) at shared `QuartoCNNAutoregUnifiedS4` architecture. Three chapters: champS4 mini-sweep (2026-05-18), LP rescoping (2026-05-19 AM), 25-config matched sweep + H9 distillation result (2026-05-19 PM). |

### Design notes

| Date | Entry | Topic |
|---|---|---|
| 2026-05-19 | [concept-targeted-saes](2026-05-19_concept-targeted-saes.md) | Design only, no implementation. Five candidate approaches (wider exp, higher k, anchored, E2E, matryoshka) to close the 11 % conv2/hawk SAE/LP wall; recommended ordering and pre-registered decision gates. |
| 2026-05-22 | [reframings-audit-tiger](2026-05-22_reframings-audit-tiger.md) | `tiger` BSP set design + implementation + **results** (36 BSPs, 6 categories): agent-relative threats and pool-reasoning. Pre-registered decision rule **fired** \u2014 tiger SAE/LP efficiency exceeds hawk by +25 to +51 pp; supervised SAE pivot now targets tiger, not hawk. |
| 2026-05-22 | [anchored-sae-champta-tiger](2026-05-22_anchored-sae-champta-tiger.md) | Plan (no implementation yet) for the next supervised-SAE experiment: anchored jumprelu / batchtopk on champTa fc1, anchored against `tigerTa` BSPs. Anchor topology, \u03bb sweep, pre-registered decision gates, implementation work items, open questions for the lit review. |

### Supervisor advances

Self-contained snapshots prepared for each supervisor meeting cycle.
One file per month under [`advances-supervisor/`](advances-supervisor/);
frozen after the meeting (subsequent updates go in the next month's
file). Each entry defines the metrics, BSP sets, and current state in
a single document so the supervisor can read it standalone.

| Cycle | Entry | Topic |
|---|---|---|
| 2026-05 | [advances-supervisor/2026-05.md](advances-supervisor/2026-05.md) | Phase 2B complete — champAa / champS4 / champTa, H9 confirmed, 11 % conv2/hawk SAE/LP wall identified, Phase 2C plan, Karvonen-framework comparison. |

## How agents should use this folder

1. Read `RESEARCH-STATUS.md` first for the current phase and live hypotheses.
2. Consult this index for the entry whose topic matches the question.
3. Read that entry directly; **do not quote diary content into other docs**
   — link to the entry by filename instead, so updates propagate.
4. When numbers in an entry conflict with what is in code / registries today,
   trust the live data and update or annotate the entry.
