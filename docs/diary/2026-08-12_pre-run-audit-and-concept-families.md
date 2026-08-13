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

---

## H. Incident: the first champTa-rebuild run (2026-08-12, 13:39–16:33)

The runner reported success and cleared the quarantine. It had done almost none
of its job, and had destroyed two files on the way. Three defects, none of which
raised an error.

### H1 — stage 3 wrote labels under the wrong names, overwriting champAa's

```powershell
foreach ($name in @('gorilla','hawk','tiger')) {
    python scripts/compute_bsp_labels.py $OUT --game $GAME --name $name
}
```

Labels are **per-distribution** and keyed by the *suffixed* animal; only the
schema is basis-keyed. Passing the bare basis meant champTa's 290,147 rows were
written to `bsp_labels-gorilla_164.pt` and `bsp_labels-hawk_173.pt` — which
belong to **champAa**, the un-tagged baseline champion. champTa's own `*Ta`
files were never touched and stayed at 88,524 rows.

The naming convention is documented in CLAUDE.md and the tool implements it
correctly; the runner simply did not follow it. Recovered by recomputing
champAa's two label sets from `positions-amalgam_unique.pt` (~3.5 min each);
the stray `bsp_labels-tiger_36.pt` was deleted (champAa never had one).

### H2 — stage 4 silently skipped every activation collection

```
[4/6] Collecting activations (trained + random model, both hooks)...
  trained:
  random :
  [SKIP] no model for suffix ''        x4
```

The inline resolver read `model_path` / `random_model_path`; the champion YAML
keys are **`path`** and **`random_path`** (`unify_positions.py` reads them
correctly). Both resolved to empty strings, and the guard was
`if (-not $mdl) { ...; continue }` — **non-fatal**.

So stage 5 retrained all 43 SAEs on the OLD 88,524-row random-play activations,
and stage 6 evaluated them against the stale `*Ta` labels — which match those
activations, so every number came out internally consistent and entirely wrong.
~2.5 h of GPU wasted, and the checkpoints now carry fresh timestamps, which
makes them more dangerous than before the run.

### H3 — the stage-7 quarantine gate could not fail

Written (in this same session) to validate *before* flipping the status. While
champTa is `QUARANTINED`, `validate_datasets.py` treats its problems as
**declared** and exits 0 — so the gate passed on the strength of the very
quarantine it was about to lift, then the flip made the same problem unflagged.
A dataset that fails validation was marked `OK`.

This is the "a pre-registered gate whose other arm never fires is not a test"
entry already in CLAUDE.md, reintroduced verbatim in a new place. The rule
generalises: **a gate must be evaluated against the state it is authorising, not
the state it is leaving.**

### H4 — provenance `game` field

`deduplicate_positions.py` auto-detects `game` from the *first* source file.
champTa has no `random_v_random_raw.pt` of its own (only Ve and Yb do), so the
runner used champAa's shared one and the merged dataset claimed
`game='quarto'`, failing validator check 3. The positions are fine —
random-vs-random involves no model, so that file is legitimately reusable — so
the fix is to pass `--game quarto_s4` explicitly.

### Fixes applied

| # | fix |
|---|---|
| H1 | `--name "${basis}${CHAMP}"`; after each basis, assert the label file exists **and** has `$N_POSITIONS` rows, else `throw` |
| H2 | read `path` / `random_path`; `sys.exit()` in the resolver if either is missing; `throw` on a missing model file; assert each activation file's row count, else `throw` |
| H3 | flip first, validate the **post-flip** state, restore from a `.pre-clear` copy and `throw` on failure |
| H4 | `--game $GAME` passed explicitly to `deduplicate_positions.py` |

`$N_POSITIONS` is read from the rebuilt file itself after stage 1 and is the
single number every later stage checks against.

Guarded by `tests/test_bsp_logic.py::TestRunnerLabelNaming` — no runner may pass
a bare basis to `compute_bsp_labels --name`, and any runner that *replaces* an
existing distribution must check label row counts. (Scoped to rebuild runners:
a first-time build has no stale file to shadow it, and failing the suite on
`champYb.ps1`, which is not broken, would just train us to ignore it.)

### H5 — why the log looked finished while it kept running

`Receive-Job -Wait -AutoRemoveJob` buffers each job's entire output and flushes
it only when that job ends. With three GPU jobs the transcript showed GPU1's
whole 13:57→15:25 block, then jumped **back** to 13:57 for GPU2, then back again
for GPU0 (which ran to 16:13). Timestamps ran backwards and each block ended
with what looked like a completed sweep.

Stage 5 now drains the jobs incrementally on a 5s poll, tags every line with its
GPU (`[gpu0]`), and prints a heartbeat every 60s with trained/total, elapsed and
ETA. Stage 6 prints `(k/N)` with elapsed and ETA per checkpoint. Every stage now
opens with a banner naming the stage, the wall clock, the elapsed time and the
**remaining stages**, so no line in the middle of the log can be mistaken for the
end of the run.

### H6 — second run: stages 1–3 correct, stage 4 failed loudly on a masked bug

The H1–H4 fixes all held. Stage 3 wrote `bsp_labels-{gorilla,hawk,tiger}Ta_*.pt`
at 290,147 rows each and the row-count assertion passed on all three; champAa's
`bsp_labels-gorilla_164.pt` / `-hawk_173.pt` were untouched; champTa stayed
`QUARANTINED` because stage 7 was never reached.

Stage 4 then failed — **loudly**, which is the point of the H2 fix — on a
different bug that the empty-model bug had been hiding:

```powershell
$flat = if ($hook -eq 's4.conv2') { '--flatten-position' } else { '' }
python scripts/collect_activations.py $mdl --hook $hook ... --device cuda $flat
```

In bash an unquoted empty variable expands to nothing. **PowerShell passes `''`
as a real, empty argument.** docopt reads it as a stray positional, prints the
usage block and exits 1. Verified directly against the module's own docstring:

| argv | result |
|---|---|
| splatted, fc1 | PARSES OK |
| splatted, conv2 + `--flatten-position` | PARSES OK |
| same + trailing `''` | REJECTED (usage error, exit 1) |

Fixed by building the argument list and splatting it (`python @actArgs`), so the
flag is genuinely absent rather than present-and-empty. `grep` confirms this was
the only `... else { '' }` argument-interpolation in `runners/`.

**Generalisable:** never interpolate a maybe-empty variable as a command-line
argument in PowerShell. Build an array, append conditionally, splat.

Also worth noting: two of the three bugs in this rebuild were only reachable
once the previous one was fixed. A runner that fails soft doesn't just lose the
current stage — it hides every bug behind it.

### H7 — third run: `$OUT` vs `$out`, a case-insensitive variable collision

Stage 4 ran this time (the H6 splat fix held) and failed on the next masked bug:

```
Loading positions from: data/quarto/s4.fc1_amalgam_ta_activations.pt
IndexError: too many indices for tensor of dimension 2
```

It was handed **its own output file** as `--positions-file`. Cause:

```powershell
$OUT  = 'data/quarto/positions-amalgam_ta_unique.pt'   # top of the runner
...
$out  = "data/quarto/${hook}_amalgam_${TAG}${sfx}_activations.pt"   # in the loop
```

**PowerShell variable names are case-insensitive**, so `$OUT` and `$out` are one
variable. The loop assignment silently overwrote the positions path. The symptom
appeared several frames away, inside torch indexing, with nothing pointing at
the real cause. This bug predates the rewrite — it was in the original runner
too, unreachable because stage 4 always skipped.

Fixed by renaming the loop variable to `$actOut`. Guarded by
`tests/test_bsp_logic.py::TestRunnerVariableHygiene`, which parses every
`runners/*.ps1`, groups assigned variable names case-insensitively and fails on
any name with two spellings. All seven runners are currently clean.

**Generalisable:** in PowerShell, two spellings of one variable name is never
intentional — it is always a silent clobber.

### H8 — the backup was overwritten by the retries

`Move-Item -Path $OUT -Destination $BACKUP -Force` ran on every attempt, so
after two failed retries `legacy_wrong_distribution/positions-amalgam_ta_unique.pt`
held a **correct 290,147-row rebuild** rather than the original 88,524-row
random-play dataset it exists to preserve. The old champTa registry rows are no
longer reproducible from it.

Reconstructible if ever needed — the original was a dedup of
`positions-random_v_random_raw.pt` alone:

```bash
python scripts/deduplicate_positions.py data/quarto/positions-random_v_random_raw.pt \
    --output data/quarto/legacy_wrong_distribution/positions-amalgam_ta_unique.pt
```

Stage 1 now refuses to overwrite an existing backup and discards the current
file instead.

### Verification status after the fixes

| stage | evidence |
|---|---|
| 1–3 | ran clean on 2026-08-12 20:03; labels `*Ta` at 290,147, row-checks passed, champAa untouched |
| 4 | fc1/trained collection run directly to completion: `(290147, 512)`; conv2 `--flatten-position` argv verified against the docopt spec |
| 5 | drain/heartbeat pattern exercised in isolation with three dummy jobs — interleaved, tagged, no lost output, failure detection works |
| 6–7 | **not yet run in their current form** |

Three consecutive failures, each revealed only by fixing the one before it. That
is the cost of fail-soft: a `continue` where a `throw` belongs does not lose one
stage, it hides every bug behind it.

---

## I. champTa rebuild verified; pipeline re-costed (2026-08-13)

### I1 — the rebuild is correct

All seven stages verified independently of the runner's own success message:
43 checkpoints retrained 08-12 21:18→23:32 with **zero stale**, trained on
`s4.{fc1,conv2}_amalgam_ta_activations.pt`; 129 registry rows (43 x 3 bases)
written 23:33→00:28, **none** missing MCC/J/MCC@pref/base-rate and **none**
carrying retired F1 keys; `validate_datasets` clean with `Ta OK 290147
quarto_s4`; no leftover `.pre-clear`.

**The confound is gone** — tiger conjunctions (19 BSPs, line+square winnable):

| champion | before | after |
|---|---|---|
| Ta | 0.045 | **0.0255** |
| Ve | — | 0.0239 |
| Yb | — | 0.0223 |

with matched N (290,147 / 289,795 / 296,045). That was the entire scientific
purpose of the rebuild.

One cosmetic fix: the stage-7 banner advertised "remaining: clear quarantine"
while running stage 7, because PowerShell **reverses** `$a[7..6]` instead of
returning empty.

### I2 — `_h` is keyed by run_id only, which constrains the order

`sae_eval.py` caches codes as `{run_id}_h.pt` with **no dataset in the name**.
`unified-pool.ps1` re-evaluates the 9 panel SAEs with `--force` against the
unified activations, so it overwrites ~36 GB of per-champion codes that
`3A-dilution`, `basis_comparison` and `backfill_eval_metrics` read.

Worse, `3A-dilution.ps1` tests only whether `_h` *exists* before deciding to
regenerate it, so it would silently consume unified-pool codes against
per-champion labels. **unified-pool must therefore run last.**

Its original justification is also largely spent: the pool existed because
champTa's base rates were ~2x everyone else's, which I2's rebuild fixed, and
`coverage_mcc_at_pref` already standardises prevalence analytically. Recorded in
the runner header as a decision to take immediately before running it.

### I3 — disk is now the binding constraint

`saes/quarto/cache` is **549 GB** (champTa 288 / champYb 244 / champVe 18) with
**231 GB free**. 3A-prep stage 1 needs 17 new `_h` caches — dominated by the
H-series, where exp64 at 289,795 rows is 35 GB apiece:

```
  exp8  (d_dict  4096)   4.4 GB     x11
  exp16 (d_dict  8192)   8.8 GB     x2
  exp32 (d_dict 16384)  17.7 GB     x2
  exp64 (d_dict 32768)  35.4 GB     x2      total 172.5 GB
```

3A-prep now sizes this up front (`Disk pre-flight`) and refuses to start rather
than truncating a cache mid-write — a partially written `_h` loads later as a
corrupt tensor. Current margin: 197.5 GB needed vs 231.2 GB free.

**Cheapest reclaim: 274 GB from champTa's 40 non-panel `_h` caches.** Safe
because champTa's 129 registry rows are *already* current (the rebuild wrote
J and MCC@pref natively), so the backfill never re-reads them, and 3A-dilution
plus basis-verdict need only F04 / E05 / I04:

```bash
python - <<'PY'
import glob, os
KEEP = {"F04-champTa-s42-jumprelu-t64-exp8-s4.fc1",
        "E05-champTa-s42-batchtopk-k32-exp8-s4.conv2",
        "I04-champTa-lh100-s43-anchored-jumprelu-t64-exp8-s4.fc1"}
for p in glob.glob("saes/quarto/cache/*champTa*_h.pt"):
    if os.path.basename(p)[:-5] not in KEEP:
        os.remove(p)
PY
```

Do **not** prune champYb's yet: 96 of its rows are backfillable only while their
`_h` survives.

### I4 — 3A-prep audited against every failure mode from H

| class | finding |
|---|---|
| case-only variable collision | clean (guarded by `TestRunnerVariableHygiene`) |
| empty-string argument | none — no conditional-flag interpolation |
| bare-basis `--name` | n/a, does not compute labels |
| fail-soft `continue` | training jobs now `throw` on `Failed` |
| gate that cannot fail | exclusion resolved once, applied to all three stages |
| buffered job output | fixed — incremental drain, `[gpu0]` tags |

Two real bugs fixed. **Stage 3 evaluated every champion twice**: it looped over
6 configs, but each champion's two configs (R1 fc1 + R2 conv2) both resolve to
the same `*champ<X>random*.pt` glob, which already returns both checkpoints. Now
iterates the 3 unique champions. And the round-robin partition was verified by
execution (`gpu0: Ta, gpu1: Ve, gpu2: Yb`), not by reading.

`-Rescan` took **1:19** and dropped champTa's 11 now-completed cells: the
manifest is 32 (14 champS4, excluded at runtime as RETIRED, + 18 champVe).

### I5 — LP subsampling for basis-verdict

`--max-train` added to `linear_probe_baseline.py` (train split only; the test
split is never subsampled) and surfaced as `-MaxTrain` on the runner. Measured
on champTa / s4.fc1 / tiger:

| max_train | s/BSP | MCC (base 0.28) | MCC (base 0.12) | est. wall |
|---:|---:|---:|---:|---:|
| 232,117 | 31.8 | 0.6365 | 0.5754 | ~7 h |
| 100,000 | 11.2 | 0.6350 | 0.5699 | ~2.3 h |
| 50,000 | 4.2 | 0.6326 | 0.5643 | **~50 min** |
| 25,000 | 1.8 | 0.6296 | 0.5632 | ~22 min |

The probe is an upper bound, not a precision estimate, and 512 features do not
need 232k rows. Caveat recorded in the runner and enforced in code: rare
concepts degrade first (hawk `completable` at base rate 0.003 keeps ~150
positives at 50k), so the script **names every BSP left with <50 positives** and
the report records `max_train` / `train_subsampled` / `n_train_full` so a
subsampled run can never be mistaken for a full one.

`config.n_train` is now the *effective* count, and the LP jobs drain
incrementally with a 120s heartbeat like the other runners.

### I6 — 3A-prep OOM: three full-size copies of `h` in one eval

Stage 1 completed 15 of 18 cells and died on the first exp64 dictionary:

```
Samples: 289795, BSPs: 36
RuntimeError: DefaultCPUAllocator: not enough memory:
  you tried to allocate 37984010240 bytes
```

289,795 x 32,768 x 4 B = **35 GB**, and one eval wanted three tensors that size
on a 256 GB box:

| site | allocation | why it existed |
|---|---|---|
| `match_features_to_bsps` | `fires = (h > 0).float()` | 35 GB |
| `match_features_to_bsps` | `(1.0 - fires).T @ labels` | another 35 GB, for `fn` |
| `compute_board_reconstruction` | `fires = (h > 0).float()` | 35 GB — to read **one column per BSP** |

plus `h` itself at 35 GB, so ~140 GB peak for a job whose real working set is a
few hundred MB.

**Fixes.** `fp` and `fn` follow from `tp` by identity, so no complement matrix is
needed at all:

```
fp = fires.T @ (1 - labels) = fires_sum[:, None] - tp
fn = (1 - fires).T @ labels = label_sum[None, :] - tp
```

and `tp` is now accumulated over row chunks sized to ~256 MB
(`chunk_rows = 2**26 // d_dict`). `compute_board_reconstruction` binarises one
column at a time. Peak extra memory for the failing shape: **71 GB -> 0.25 GB**.

Verified equivalent to the old formulation on five shapes including
`N` not a multiple of the chunk size and `d_dict > N`, then confirmed on the
real case: `H03-champVe … exp64 / tigerVe` now completes
(`coverage_mcc 0.3017, J 0.4933, MCC@pref 0.2326`). Guarded by
`tests/test_sae_eval.py::TestMatchingMemoryFootprint`.

**Also fixed while in there:** `sae_eval.py` wrote a matching cache missing
`youden_j`, `mcc_at_pref`, their best-per-BSP indices and `p_ref` — exactly the
fields `backfill_eval_metrics.py`'s `REQUIRED_MATCHING_FIELDS` checks — so every
cache it wrote looked stale and would have been rebuilt from `_h` for nothing.

**Note on cost:** this was reached only because 3A-prep evaluates the full
H-series sweep. The exp32/exp64 members cost 18–35 GB of cache each and are not
in the 3A panel; they are in the manifest only because the tiger basis
post-dates the sweep.

---

## J. 3A-prep verified (2026-08-13)

### J1 — completeness

18/18 champVe tiger cells present; the 14 champS4 cells correctly excluded as
RETIRED. All 6 random-model controls trained, and 18 control registry rows
(6 x 3 bases) written with **zero** missing MCC / J / MCC@pref / base-rate.

Every one of the 11 panel runs in `3A-dilution.ps1` resolves to a real control
whose `_h` cache exists, so **no run falls back to the permutation null** —
which was the whole point of the stage. Verified by executing the runner's own
regex against the run list rather than reading it.

The runner's closing line used to say "populate `$RANDOM_CONTROLS`"; they were
already populated, so it now *checks* instead — a control whose `_h` is missing
does not fail, it silently degrades to the weaker null.

### J2 — the controls are not degenerate

Dead-feature rates are comparable between control and recipe-matched trained SAE
(97–98% on both), so the controls are not a training collapse. FVU is much
higher for the controls (0.17–0.20 vs 0.004–0.11), which is what a proper null
should look like: random-network activations carry less sparsely-reconstructable
structure.

### J3 — the headline finding: conv2 is largely NOT learned

Learned-gap fraction, control vs recipe-matched trained SAE. Within a cell the
control and the trained SAE score the *same positions against the same labels*,
so the base rate is identical and raw MCC is a fair comparison; `mcc_at_pref`
only matters across cells.

| champ / hook | gorilla | hawk | tiger |
|---|---:|---:|---:|
| Ta / fc1 | 99.4% | 99.5% | 99.1% |
| Ve / fc1 | 99.3% | 99.5% | 99.2% |
| Yb / fc1 | 99.3% | 99.7% | 98.2% |
| Ta / conv2 | 61.5% | 48.8% | 29.9% |
| Ve / conv2 | 62.4% | 45.1% | 33.0% |
| **Yb / conv2** | **7.1%** | **13.4%** | **16.3%** |

**fc1 is clean** — the random-network control scores 0.0013–0.0043, essentially
nothing, so ~99% of the trained SAE's score is learned.

**conv2 is not.** The control reaches 0.065–0.151, and on champYb the trained
SAE beats it by only 7–16%. Most of what a conv2 SAE "finds" is available in a
randomly-initialised network's activations — i.e. it is structure in a random
projection of the board, not a learned representation.

This is exactly the failure the reporting standard's learned-gap clause exists
to catch, and it fires hardest on the strongest champion. Every conv2 row in the
3A panel (E01/E05) should be read against it: 3A can still say whether a concept
is diluted *in that dictionary*, but a conv2 dilution verdict says much less
about the trained model than an fc1 one does.

Caveat: 11 of 18 cells fell back to raw MCC because the *trained* row predates
`mcc_at_pref` (champVe/champYb gorilla+hawk). The comparison is still valid for
the reasons above; the backfill will make the columns uniform.

Minor: `E01-champVe` (topk k32) is scored against the batchtopk k32 control,
since the controls were recipe-matched to E05. Same hook, width and sparsity
level, different top-k variant — close but not exact.

### J4 — disk is now critical

**32 GB free; `saes/quarto/cache` is 755 GB.** Recomputed the safe prune against
what is actually still needed — a run's `_h` is needed only if it is in the 3A
panel / a control, or if its registry row still lacks the current metrics (i.e.
the backfill has not consumed it yet):

```
  total _h       : 749 GB
  must keep      : 474 GB  (62 files)
  SAFE TO DELETE : 274 GB  (40 files)
```

The 40 are champTa non-panel caches: champTa's 129 rows were written by the
rebuild with J and MCC@pref already present, so the backfill never re-reads
them, and 3A/basis-verdict need only F04 / E05 / I04.

`unified-pool` needs ~160 GB and cannot run until this is reclaimed.
