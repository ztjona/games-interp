# Explanations

Standalone explanatory pieces, written **on request**: a description of how
something works, an explanation of why a choice was made, or a progress
write-up meant to be read by someone who has not been following the diary.

This folder replaces `docs/diary/advances-supervisor/` (retired 2026-08-12).
That folder assumed one self-contained snapshot per monthly supervisor cycle,
but each meeting presented something different enough that the monthly template
was never really followed; a convention that is half-followed is worse than
none, so the files were pruned and the idea reframed.

## What belongs here

- **Self-contained.** Assume the reader has not read `RESEARCH-STATUS.md` or the
  diary. Define the metrics, BSP sets and champions you use, or link to
  [`../methods-reference.md`](../methods-reference.md) for definitions.
- **Topic-named, not date-named.** `sae-evaluation-pipeline.md`, not
  `2026-08-12_sae-evaluation-pipeline.md`. These are living documents.
- **Updated in place**, and deleted once superseded.

## What does *not* belong here

| If you are… | Write instead |
|---|---|
| recording what happened on a given day, with tables | a dated entry in [`../diary/`](../diary/README.md) — frozen after creation |
| defining **how a number is computed** | [`../methods-reference.md`](../methods-reference.md) — no results, no dated entries |
| noting a phase change or a headline metric | a one-line bullet in [`../../RESEARCH-STATUS.md`](../../RESEARCH-STATUS.md) |
| documenting a naming or pipeline **convention** | `Quarto-specifications.md` or `CLAUDE.md` |

Do not duplicate content across these — link instead. The diary is the audit
trail of *what happened*; this folder is for *what it means*.

## Index

| Topic | Written | Summary |
|---|---|---|
| [new-champion-recommendations](new-champion-recommendations.md) | 2026-09-12 | How to train the next Quarto champion so every analysis re-runs on it: the comparability contract, legality via synthetic illegal-move = loss transitions, a controlled L0/L1 pair with matched seeds, provenance, and competence-only acceptance criteria. |
