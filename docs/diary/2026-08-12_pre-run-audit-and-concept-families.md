# 2026-08-12 — Pre-run audit of the phase3A branch, + `concept_family` in the BSP schema

Audit performed before launching the planned chain
`champTa-rebuild -SkipTrain → unified-pool → 3A-prep → 3A-dilution → R² re-check
→ registry backfill`. Scope: are the new files correct, still relevant, and does
the chain actually work in the intended order?

Verdict: **the chain would not have produced correct results as written.** Two
blocking defects, both silent (no error, wrong numbers), plus a set of smaller
gaps. Blockers are fixed in this commit; the remaining items are listed with a
recommendation each.

---

## A. Blocking defects (fixed)

### A1 — `unify_positions.py` would have skipped the rebuild entirely

Staleness was decided by comparing the *set of source paths* against the pool's
recorded provenance. `champTa-rebuild.ps1` **corrects a file in place**; it does
not add or remove one. So the source set is unchanged, `need_rebuild` stays
`False`, and the run prints

```
Already up-to-date: 677744 positions (677k)
```

leaving the old, defective pool on disk. Every subsequent stage — unified
activations, unified labels, all matched-base-rate cross-champion numbers —
would then have been computed on a pool that still contains champTa's
random-play positions, which is the exact defect the rebuild exists to remove.

Verified before the fix: the existing pool's five source names are set-equal to
the five amalgams currently on disk.

**Fix:** make-style staleness check — rebuild if any input's mtime is newer than
the pool's. Also normalised the path separator when comparing source names, so a
pool merged on another box is not reported as entirely new.

### A2 — `unified-pool.ps1` suffix detection throws *after* the GPU work

The pool-size suffix was resolved by globbing `bsp_labels-gorilla[0-9]*k_*.pt`
and requiring exactly one match. The superseded `bsp_labels-gorilla677k_164.pt`
is still on disk (correctly — it is superseded, not deleted), so after a rebuild
to a new N there would be two matches and the runner would `throw` — at stage 3,
having already spent all of stage 2's activation collection.

**Fix:** read the suffix from the pool file itself (`N // 1000`), and assert that
labels for *that* suffix exist before evaluating anything.

---

## B. Gaps that block a *downstream* step (fixed)

### B1 — `pytest` with no arguments ran zero project tests

`repositories/SAE_BoardGameEval` (vendored, untracked) contributes four test
modules importing `circuits` and `pandas`. Collection errors abort the entire
run, so `pytest` exited on 4 errors and none of the project's own 147 tests
executed. Anyone using a bare `pytest` as a pre-flight gate got a red result
unrelated to their change — or, worse, learned to ignore it.

**Fix:** `pytest.ini` (`testpaths = tests`, `norecursedirs` incl. `repositories`)
and `repositories/` added to `.gitignore`. Bare `pytest` now runs 159 tests.

### B2 — `linear_probe_baseline.py` could not support the planned verdict

The LP baseline is the upper bound the SAE is measured against, but it emitted
only F1 / MCC / F1-lift — no Youden's J, no `mcc_at_pref`, no base-rate block,
and no family rollup. A hawk-vs-tiger comparison built on it would have been
made on prevalence-dependent numbers across bases whose base rates differ ~7x,
which is precisely the defect that makes the 2026-05-22 audit unreliable.

**Fix:** LP now emits the same metric schema as `sae_eval` (`coverage_mcc`
headline, `coverage_youden_j`, `coverage_mcc_at_pref`, base-rate block) with
per-category key names identical to `compute_per_category_coverage`, plus a
`per_family` rollup. LP and SAE reports are now diffable field-by-field.

---

## C. Resolved in the follow-up pass (2026-08-12, later)

Everything in this section was raised as open in the first pass and settled
after review. Kept as written, with the resolution appended to each.

### C1 — `scripts/backfill_eval_metrics.py` is stale and will silently no-op

Two independent reasons the planned "registry backfill (J + mcc_at_pref +
prevalence + trimmed schema)" will not work as written:

1. The registry-level skip is
   `if not force and "coverage_mcc" in metrics and "coverage_f1_lift" in metrics`.
   497 of 523 rows satisfy both, so they are skipped.
2. `--force-recompute` does not help. `_augment_matching` only rebuilds from the
   `_h` cache when `best_mcc_per_bsp is None`; for those same 497 rows MCC is
   present, so it never rebuilds, `youden_j` / `mcc_at_pref` stay `None`, and
   `compute_coverage` emits neither.

It also still writes the F1-lift family into the matching cache and never
removes the now-dropped keys (`coverage_above_50`, `min_f1`, `median_f1`,
`coverage_f1_lift*`, `median_f1_lift`) from existing rows, so old and new rows
would not share one schema — the "trimmed schema" half of the task.

Measured scope: 240 in-scope (champVe/champYb) rows lack `coverage_mcc_at_pref`.

**Recommendation:** rewrite the backfill before running it — (a) skip on
`coverage_youden_j`, not `coverage_f1_lift`; (b) rebuild from `_h` whenever any
new field is missing; (c) persist the new tensors in the matching cache;
(d) delete the retired keys. Until then the `family` / efficiency views show
`-` for J and MCC@pref, which is honest but useless.

### C2 — `unify_positions.py` merges RETIRED champS4 and non-panel champAa

`discover_amalgams()` globs every `positions-amalgam*_unique.pt`. That includes
`positions-amalgam_s4_unique.pt` (RETIRED, byte-identical to champAa's) and
champAa's own file. After dedup champS4 contributes **zero** unique rows, so
pool *content* is unaffected — but the provenance lists a retired dataset, and
champAa's ~200k positions come from a different architecture (`QuartoCNN`,
fc1=128) than the S4-family panel the pool is used to compare.

**This is a research decision, not a bug**, so it is left alone: whether the
cross-champion pool should span architectures is the user's call. If it should
not, add an exclusion list driven by `_dataset_status.json`.

### C3 — `unified-pool.ps1` re-encodes `_h` three times per SAE

Stage 3 loops `sae_eval.py evaluate --force` per basis, so the pool's codes are
encoded 3x per SAE (27 encodes over a 677k+ pool for 9 SAEs).
`scripts/eval_unified.py` already exists and encodes once per SAE, reusing the
codes across bases — but its tag inference predates champYb and would mis-tag Yb
as `aa`. **Recommendation:** teach `eval_unified.py` about Yb and call it from
the runner, or accept the 3x cost knowingly.

### C4 — smaller items

- `docs/diary/advances-supervisor/2026-05.md` and `docs/advances-supervisor/2026-05.md`
  are both staged as deletions, with no supervisor file left anywhere in git.
  The doc contract marks that directory **not prunable**. Restore or record why.
- `runners/_missing_evals.txt` is staged as a tracked file. It is a regenerable
  scratch manifest (`3A-prep.ps1 -Rescan`); it reads as an input but is derived.
- `cheatsheet.md` (1 line, repo root) is not in the documentation contract table.
- Suffixed schemas violate the basis-only convention: `bsp_schema-gorilla677k_164.json`,
  `-gorillaVe_164`, `-gorillaYb_164`, `-hawk677k`, `-hawkVe`, `-hawkYb`,
  `-tiger677k`. `unify_positions.py` creates them by passing an explicit
  `--schema-out`; `compute_bsp_labels.py` keys schemas by basis by default.
  They are byte-identical duplicates and will proliferate one set per pool size.
- `bsp_schema-hawk_92.json` is a stale artefact of the pre-173 hawk basis.
  `unify_positions.py` picks `schema_matches[0]`, which happens to sort to
  `_173` before `_92` — correct today, by luck.
- `3A-prep.ps1` excludes QUARANTINED champions from the tiger backfill (stage 1)
  but **not** from the control training/eval (stages 2–3), so
  `R{1,2}-champTarandom-*` would still be trained on quarantined activations.

---

## D. Ordering hazard in the planned chain

`champTa-rebuild.ps1 -SkipTrain` rebuilds champTa's positions, labels and
activations but **deliberately does not retrain its SAEs** — the runner says so.
The champTa checkpoints on disk therefore remain trained on the old random-play
activations.

`unified-pool.ps1` then re-evaluates `F04/E05/I04-champTa` against the unified
pool. That is not meaningless (the SAE is a fixed function; evaluating it on new
positions is well-defined), but the resulting rows describe *a dictionary learned
from random play, scored on mixed play*. They are **not** the champTa numbers a
reader will assume they are.

Note also that `champTa-rebuild.ps1` never flips champTa's status in
`_dataset_status.json`; its stage 2b prints a note and leaves it to a human. Since
the champTa *SAEs* stay stale under `-SkipTrain`, leaving the champion
QUARANTINED is the correct state until the retrain happens — but then
`unified-pool.ps1`'s own gate does not exclude champTa's SAEs from stage 3.

**Recommendation:** either run `champTa-rebuild.ps1` without `-SkipTrain`, or
drop the three champTa rows from `unified-pool.ps1`'s `$RUNS` for this pass and
add them after the retrain.

---

## E. `concept_family` — implemented

Added as an optional-but-now-populated schema field, so cross-basis comparison is
a property of the data rather than a hardcoded dict in one analysis script.

- **Source of truth:** `CONCEPT_FAMILIES` in `scripts/games/quarto.py`
  (re-exported by `quarto_s4` / `quarto_s4_hot`), mapping
  `category -> (concept_family, family_role)`.
- **`family_role`** is `state` | `state_any` | `agent_relative` — the level of
  agent-relativity, which is the axis the 2026-05-22 reframing varied. A *triad*
  is one category per basis per family, at the highest agent-relativity that
  basis offers.
- **Stamped into the data** by `compute_bsp_labels.py` (per-BSP
  `concept_family` / `family_role`, plus schema-level `concept_families` and
  `category_families` rollups), and retrofitted onto the 11 schemas already on
  disk by `scripts/stamp_concept_families.py` — no label recompute, idempotent,
  refuses to alter the BSP list.
- **Read, never re-derived**, via `lib/sae/eval.py`: `category_families`,
  `family_of_category`, `derive_triads`, `aggregate_per_category_by_family`
  (count-weighted, so a family mean is a mean over BSPs, not over categories).
- **Consumers:** `basis_comparison.py` now derives its triads from the schema
  (its hardcoded `TRIADS` dict is gone); `registry_query.py` gains `family` and
  `triads` subcommands; `linear_probe_baseline.py` emits `per_family`.

Derived triads reproduce the two the audit used, and surface a third for free:

| concept_family | gorilla | hawk | tiger |
|---|---|---|---|
| `line_threat` | `threat_line` | `reframed_completable` | `tiger_line_winnable` |
| `square_threat` | `threat_square_2x2` | `reframed_sq_completable` | `tiger_square_winnable` |
| `global_threat` | `global` | `reframed_global` | `tiger_decision_global` |

Single-basis families (no cross-basis comparison exists, and reporting one would
mislead): `board_attribute`, `board_occupancy`, `game_phase`,
`offered_completion`, `offered_piece_attr`, `pool_reasoning`.

Deliberate split: gorilla `offered_piece` (a 4-bit readout, the source of the
F1 ≈ 0.667 trivial-baseline artefact) is `offered_piece_attr`, kept apart from
`tiger_offered_completing_attr` (`offered_completion`, a board-conditioned
conjunction). Folding them together would average a trivial readout into a real
threat concept.

Guarded by `tests/test_bsp_logic.py::TestConceptFamilies` (every category in
every basis has a family; roles are known; triads are 1:1 per basis; the
line/square triads match the audit) and
`tests/test_sae_eval.py::TestConceptFamilyRollups` /
`::TestShippedSchemasCarryFamilies` (schema-driven only; unstamped schema
reports "nothing declared" rather than guessing; family means are count-weighted;
the real schemas on disk are stamped).

---

## F. New runner — `runners/basis-verdict.ps1`

Settles hawk vs tiger on **efficiency** rather than raw score.

`scripts/sae_lp_efficiency.py` joins the LP reports with the registry rows and
reports, per concept family per basis: `lp_mcc_at_pref` (availability — is the
concept linearly decodable at all), `sae_mcc_at_pref` (captured), and their
ratio (efficiency), with an LP floor of 0.05 below which the ratio is suppressed
rather than reported as a loud, meaningless number.

The distinction matters because a basis can win on raw SAE score purely because
its concepts are easier to decode, which says nothing about whether an
unsupervised dictionary finds them — and availability-vs-efficiency was never
separated in the original audit.

Stage order is load-bearing, and each stage has a failure mode that is silent
rather than loud:

| # | stage | why it must precede the next |
|---|---|---|
| 0 | dataset provenance gate + QUARANTINED/RETIRED champion exclusion | else the verdict rests on random-play data; champTa is excluded until rebuilt **and** retrained |
| 1 | `stamp_concept_families.py` | else no family rollups and both sides fall back to whole-basis averages — the original defect |
| 2 | LP baselines (slow, CPU, fans out per champion) | the denominator of every ratio |
| 3 | assert registry rows carry `mcc_at_pref` | a stale row rolls up fine and yields `n/a` everywhere with no reason given |
| 4 | `basis_comparison.py` per champion | one SAE, one code set, three framings — only the framing varies |
| 5 | `sae_lp_efficiency.py` | the verdict |

Stage 3 is advisory, not fatal, so the expensive stage-2 work is never wasted by
a registry problem; stage 5 then excludes any basis whose row is stale and names
it. Verified by dry-run: it correctly reports 240 stale in-scope rows (see C1)
and notes that champVe/`s4.fc1` currently has no unsupervised `_h` cache and
would fall back to the anchored control — the very cell `3A-prep`'s tiger
backfill fills, which is why `basis-verdict` must run after it.

---

## G. Resolutions (follow-up pass, 2026-08-12)

### G1 — backfill rewritten (resolves C1)

- Staleness now keys on `CURRENT_METRIC_KEYS` (`coverage_mcc`,
  `coverage_youden_j`, `coverage_mcc_at_pref`, `mean_base_rate`) — the *newest*
  metrics, so adding one in future re-opens the rows that lack it automatically.
- `_augment_matching` rebuilds from `_h` whenever **any** field in
  `REQUIRED_MATCHING_FIELDS` is missing, not just when MCC is absent. J and
  `mcc_at_pref` are functions of the full `(d_dict, num_bsps)` rate tensors, so
  they genuinely cannot be derived from the stored per-BSP bests.
- `_save_matching` writes **every** field the dataclass carries. The old
  hand-listed subset predated J and `mcc_at_pref`, so a rebuilt matching lost
  exactly the fields the rebuild existed to produce — and the next run rebuilt
  it again.
- Retired keys are now removed (`RETIRED_METRIC_KEYS` +
  `RETIRED_PER_CATEGORY_KEYS`), including from rows that are otherwise current.
- A row needing a rebuild with no `_h` on disk raises `FileNotFoundError` and is
  reported **by name** with the exact regenerating command, batched one call per
  checkpoint with all bases comma-separated. Previously it fell into a generic
  `except Exception` and was indistinguishable from a real failure.
- `--dry-run` now inspects the caches instead of short-circuiting, so its counts
  match what the real run will do.

Measured on the current registry: **116 rows can be backfilled from caches on
disk; 371 need `_h` re-encoding; 36 have no matching cache.** By champion —
champYb 96/129 backfillable, champVe 11/111, champTa 9/118 (moot, it is being
retrained), champS4 + legacy 123 rows not worth re-encoding. The champVe gap
closes on its own: `3A-prep` and `unified-pool` re-evaluate those checkpoints,
and current `sae_eval` writes the new metrics natively.

### G2 — champTa retrains in full (resolves D)

Verified the premise: champTa's 43 sweep configs are recipe-identical to
champVe's and champYb's (same ids modulo the champion tag), so a full retrain is
what puts all three on equal footing. `-SkipTrain` is now documented as a
validation escape hatch only, and **stage 7 clears the quarantine** — re-running
`validate_datasets.py` and, only if it passes cleanly, flipping champTa to `OK`
in `_dataset_status.json`. Without that flip every downstream runner keeps
excluding champTa and the rebuild silently accomplishes nothing.

### G3 — quarantine exclusion is now consistent across all of `3A-prep` (resolves C4)

The excluded-champion set is resolved once, up front, and applied to all three
stages. Previously it gated only the tiger backfill, so stages 2–3 still trained
and evaluated `R{1,2}-champTarandom-*` on quarantined activations: 2 SAE
trainings, 6 evals and 2 multi-GB `_h` writes producing a control that could not
be used but would sit in the registry looking like any other row.

Stage 2 also no longer goes through `run_sweep.py --configs=configs/controls`,
which globs the whole directory and would have trained the excluded configs
regardless of the filter; the 6-config set is driven directly instead.

### G4 — `_h` is encoded once per checkpoint, not once per basis (resolves C3)

`sae_eval.py --bsps` already accepts a comma-separated list and encodes the
codes **once**, reusing them across every BSP set in the call. Three runners
looped per basis with `--force`, which re-encodes the whole dataset and rewrites
the same multi-GB file each time. Nothing was being deleted and re-created — the
same path was simply overwritten three times.

Not negligible: a single conv2 `_h` is 1.4–4.8 GB (`saes/quarto/cache` is 272 GB
in total), and the unified pool is larger still. Fixed in `unified-pool.ps1`
(27 encodes → 9), `champTa-rebuild.ps1` (132 → 44) and `3A-prep.ps1` (18 → 6).

### G5 — `advances-supervisor/` retired, `docs/explanations/` introduced (resolves C4)

The monthly-snapshot template was never really followed, so the files were
pruned deliberately rather than left as a half-kept convention. The doc contract
in `CLAUDE.md` now describes `docs/explanations/` instead: standalone pieces
written **on request**, topic-named rather than date-named, self-contained,
updated in place. `docs/explanations/README.md` states what belongs there and,
more usefully, what belongs in the diary / methods-reference / RESEARCH-STATUS
instead. The dangling links in `docs/diary/README.md`, `phase-2B.md` and
`RESEARCH-STATUS.md` are repointed.

### G6 — schema naming: one file per basis (resolves C4)

**Root cause found.** `_ANIMAL_BASIS_RE` only stripped an upper-case-initial
champion suffix, so `animal_to_basis("gorilla677k")` returned `gorilla677k`
unchanged, the basis fallback in `resolve_schema_path` never fired, and
`unify_positions.py` *had* to write a suffixed schema copy per pool size just to
be findable. The duplicates were a symptom, not the disease.

Fixed the regex to strip both suffix kinds (`<digits>k` and `[A-Z].*`), removed
the explicit `--schema-out` from `unify_positions.py`, and deleted the eight
redundant files after verifying all seven suffixed copies were **content-identical**
to their basis schema:

| deleted | reason |
|---|---|
| `bsp_schema-{gorilla,hawk,tiger}677k_*.json` | identical to basis; would proliferate one set per pool size |
| `bsp_schema-{gorilla,hawk}{Ve,Yb}_*.json` | identical to basis; per-champion copies |
| `bsp_schema-hawk_92.json` | orphaned pre-173 hawk basis — **zero** registry rows use it (all 191 hawk rows are 173) |

`bsp_schema-hawk_92.json` was the sharpest hazard: `resolve_schema_path` globs
`bsp_schema-hawk_[0-9]*.json` and takes `sorted()[0]`, which returned `_173`
before `_92` only because `"1" < "9"` lexically. Correct by luck.

Three schemas now remain, one per basis, and
`tests/test_sae_eval.py::TestSchemaNamingConvention` fails on any suffixed file
or any basis with more than one schema.

**Storage note:** the superseded `bsp_labels-*677k_*.pt` label tensors are
*not* deleted here — they are per-distribution data, not duplicated metadata,
and `_dataset_status.json` already marks them superseded. Delete them once the
new pool's labels exist, if disk is needed.
