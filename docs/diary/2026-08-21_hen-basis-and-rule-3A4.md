# `hen` built and verified; rule 3A.4 bands the verdict; panel wired (2026-08-21 PM)

Status: frozen record. Parent ledger: [`phase-3.md`](phase-3.md). Metric and
rule definitions: [`../methods-reference.md`](../methods-reference.md) §3.5.
Numbers are `[DIRECT]`; interpretation is `[AI-REASONED PROVISIONAL ANALYSIS]`.

Executes the handoff in
[`2026-08-21_3A-residuals-and-handoff.md`](2026-08-21_3A-residuals-and-handoff.md) §8.
Two of that entry's operational claims did not survive checking; both are
recorded in §1 and both are now fixed rather than worked around.

Every table states its metric and units.

## 0. Review of the handoff before executing it

Re-derived the load-bearing numbers from source rather than trusting the entry.

| handoff § | claim | outcome |
|---|---|---|
| 2 | per-seed `geometric_frac` 0.16/0.13/0.18 and 1.00/0.87/0.87 | **reproduces** |
| 3 | K03 W=1 85.2 → never-fires 54.5; artefact 30.7 pp; 1,864 alive = 3.64× | **reproduces** |
| 4 | 3.4% undecided at ±0.033 | **reproduces** (49/1450 concept points) |
| 5 | ladder 0.62/0.41/0.25 and 0.64/0.53/0.23, spreads 0.37/0.41 | **reproduces** |
| 6 | tiger 68,041 / OR(hawk) 35,633 / t∧¬h 32,408 / h∧¬t **0** | **reproduces** |

Also checked and sound: `d_dict = 512 × exp` is right for **both** hooks
(`W_enc` is `(512, 4096)` for `s4.conv2` too, so the panel's health arithmetic
holds); all six random-model controls have their `_h`; `rule_version` is stamped
in `summary.config`.

## 1. Two corrections to the handoff [DIRECT]

**(a) "All 15 missing `_h` encodes are DONE — the run is CPU-only from here"
was false.** All 15 were absent from `saes/quarto/cache/`, and
`h_prune_plan.json`'s `kept` list (36 entries) contained **none** of them —
despite its own `rationale` string claiming "the runs still needing an `_h`
encode" were kept. The prune was computed from what was on disk, so runs with no
cache had nothing to keep; the rationale described an intention, not the plan.
The four **anchored positive controls** were dropped by the same prune.

The consequence was worse than the lost GPU time. `check_sae_usable.py` exits
**2** on a missing `_h`, and the runner throws on any non-zero exit from that
gate — which sits *before* its own `_h`-regeneration step. A `-Panel` mode
written as the handoff described would have died at the gate on a freshly pruned
cache. Fixed by ordering: **encode → gate → diagnose** (§4).

All 19 encodes are now done (15 panel members + 4 anchored controls).

**(b) The specified tri-state contradicted the handoff's own §2, and was blind
to half the measured instability.** `±3 × 0.0110` gives 3.4% undecided, but
direct seed replication measured **12–17%** of per-concept verdicts flipping.
Two independent causes:

- `0.0110` is the **median** per-concept seed sd of a heavily right-skewed
  distribution (mean 0.0523, p90 0.12–0.17). A band is a claim about the tail.
- `solo_frac` is only one of `classify`'s inputs. Decomposing the flips
  (units: concepts, over seeds 42/43/44):

| condition | stable | flips via `solo_frac` | flips via OTHER terms |
|---|---:|---:|---:|
| K03 TopK fc1 / gorillaYb | 67 | 8 | 1 |
| **K04 BatchTopK conv2 / tigerYb** | 19 | **0** | **4** |

All four of K04's flips are `absent ↔ spread`, i.e. rule 3A.3's learned-signal
floor — which a `solo_frac` band never touches. Built the superset instead (§3).

## 2. `hen` — BUILT and VERIFIED [DIRECT]

173 BSPs mirroring hawk category-for-category on the four **negative** attribute
poles (`little`, `white`, `circle`, `without_hole`). A **new basis**, not an
extension of hawk — extending hawk would silently change every banked hawk
number, and hawk-vs-hen on one dictionary is the matched pair that isolates
polarity.

| hen category | n | hawk counterpart | n |
|---|---:|---|---:|
| `neg_count` | 40 | `reframed_count` | 40 |
| `neg_completable` | 40 | `reframed_completable` | 40 |
| `neg_any_threat` | 10 | `reframed_any_threat` | 10 |
| `neg_sq_count` | 36 | `reframed_sq_count` | 36 |
| `neg_sq_completable` | 36 | `reframed_sq_completable` | 36 |
| `neg_sq_any_threat` | 9 | `reframed_sq_any_threat` | 9 |
| `neg_global` | 2 | `reframed_global` | 2 |
| **total** | **173** | | **173** |

The BSP menu goes 373 → 546. hen's categories share hawk's
`(concept_family, family_role)` pairs, so the family rollup compares them
directly, and `--categories=auto` selects the same 6 of 7 in both (the global
pair is excluded in both).

**The identity holds exactly.** Units: positions. Lines and 2×2 squares, every
row of each champion's amalgam:

| champion | rows | tiger | OR(hawk) | OR(hen) | hen-only | share of tiger | **violations** |
|---|---:|---:|---:|---:|---:|---:|---:|
| champTa | 290,147 | 140,494 | 72,605 | 70,828 | 67,889 | 48.3% | **0** |
| champVe | 289,795 | 131,724 | 67,571 | 66,918 | 64,153 | 48.7% | **0** |
| champYb | 296,045 | 125,344 | 65,450 | 62,576 | 59,894 | 47.8% | **0** |

`tiger == OR(hawk ∪ hen)` with zero counterexamples on ~290k positions per
champion. The 47.8% for champYb reproduces the handoff's 47.6% (that figure was
lines only; these totals are lines + squares).

**Regression check**: recomputed hawk labels on 3,000 sampled champYb rows
against the banked `bsp_labels-hawkYb_173.pt` — **0 mismatches**, and the BSP id
order is unchanged, so no banked tensor misaligns. The refactor that made the
threat helpers pole-keyed (`ATTR_POLES`) is behaviour-preserving for the
positive poles by construction.

Guarded by `tests/test_bsp_logic.py::TestHenNegativePoles` (6 tests), including
the identity on 400 randomised dense boards and a dispatch-ordering test —
`row_0_neg_any_threat` also ends in `_any_threat`, so an ordering slip routes it
to the positive-pole helper and the bug would be invisible in the counts.

### First matched hawk-vs-hen cell [DIRECT]

One entry run as an end-to-end check of the hen path. Same dictionary, same
positions, same random-model control — only the attribute polarity differs.
`geom` = `geometric_frac` (dimensionless 0–1), `solo` = mean `solo_frac`.

| basis | n threat BSPs | geom | band | undecided | captured | absent | mean R² | solo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hawkTa | 171 | 0.99 | [0.96, 0.99] | 4% | 0 | 1 | 0.208 | 0.44 |
| henTa | 171 | 0.98 | [0.93, 0.99] | 6% | 0 | 4 | 0.190 | 0.46 |

Per family — hawk `line_threat` 0.99/0.38, `square_threat` 1.00/0.49; hen
0.96/0.42 and 1.00/0.50.

`[AI-REASONED PROVISIONAL ANALYSIS]` **hawk and hen are indistinguishable here,
and this cell cannot test the polarity hypothesis.** On champTa's fc1
dictionary *neither* basis is captured — both sit at ~0.98 geometric with zero
`captured` verdicts — so there is no headroom for hen to be "markedly less
captured than hawk". The hypothesis needs a cell where the positive poles ARE
captured, and exactly one exists: **champYb fc1**, where 2026-08-17 measured
gorilla 0.03 and hawk 0.17 against tiger 0.96. If polarity explains tiger's
residual, `henYb` on that dictionary should land near tiger, not near hawk.
That entry is in the panel; it can also be run alone in ~20 min with
`-Only henYb`.

What this cell *does* establish: the hen path runs end-to-end, produces 171
threat BSPs (matching hawk exactly), stamps rule 3A.4, applies the
random-model control to 100% of concepts, and populates the band across **both**
boundaries (6 verdicts resting on `asymptote_r2`, 5 on `solo_frac`).

**The polarity hypothesis itself is still open.** It needs no new GPU work:
`_h` does not depend on the BSP set.

## 3. Rule 3A.4 — the verdict stability band [DIRECT]

`classify_with_stability` pushes every quantity `classify` thresholds on to
±`band_sds` (3) sd and re-runs `classify` at each of the 2^k corners. Agreement
everywhere → `confident`; otherwise `undecided`, with `verdict_flips_on` naming
the quantities that flip it when moved alone. **The point verdict is unchanged
at 3A.4** — reclassifying all 23 banked reports produced 0 verdict changes.

| banded quantity | sd used | kind |
|---|---|---|
| `asymptote_r2` | `asymptote_r2_std / √n_splits` | within-run **split** sd |
| `solo_frac` | `solo_frac_seed_sd` = 0.0523 | **cross-seed** sd, pooled constant |

Two corrections found while building it:

- **The mean, not the median**, for `solo_frac_seed_sd` (§1b).
- **A √n_splits scale error.** `asymptote_r2` is the *mean* over `n_splits = 5`
  resamples while `asymptote_r2_std` is the sd *across* them, so the uncertainty
  of the stored number is `sd/√5`, not `sd`. Banding the mean at the
  across-split sd read K04 at **46.4%** undecided against a measured 17.4%.

**Calibration against the two seed-replicated conditions.** Units: % of
per-concept verdicts.

| condition | band says `undecided` | **measured** seed flip rate |
|---|---:|---:|
| K03 TopK fc1 / gorillaYb | 5.3% | 11.8% |
| K04 BatchTopK conv2 / tigerYb | 11.6% | 17.4% |
| all 23 banked reports | 7.2% | (no seed replicates) |

`[AI-REASONED PROVISIONAL ANALYSIS]` The band **under-calls by roughly 1.5–2×**,
in a direction that is expected and should be stated rather than tuned away:
both sds are lower bounds on seed-to-seed movement (split resampling does not
retrain the SAE; the `solo_frac` constant is pooled rather than per-concept).
Fitting the constant to two conditions would buy agreement at the cost of the
one honest property the band has — that each sd is a measured quantity. **Where
seeds exist, quote the measured flip rate, not the band.** Read `undecided` as
"near a boundary relative to how much this number is known to move", never as a
significance test.

Of the 105 undecided verdicts across the banked corpus, **83 rest on
`solo_frac` and 22 on `asymptote_r2`** — so the second boundary carries a fifth
of the instability that the specified design would have reported as zero.

**The band immediately changes a reading.** The gate now reports
`geometric_frac_lo`/`_hi` (resolving every undecided concept the least- and
most-geometric way) and refuses to certify a run whose band straddles 0.50:

```
I04-champYb-lh100-s42-anchored-…  tigerYb   geom 0.30   [0.17, 0.96]   78% undecided
```

One of 23 entries straddles — the anchored champYb positive control, which the
handoff §2 had already flagged as "within one seed-range of the 0.50 gate" and
§4 as the concentration point of the undecided mass (21.7% there). The native
band puts it far more strongly: **that entry does not determine its gate at
all**, and its `3C-deprioritized (captured)` verdict must not be quoted. Every
other banked entry is stable, so the aggregate G-3A reading is untouched.

## 4. The panel run — WIRED, not yet launched [DIRECT]

`scripts/build_3a_panel_runlist.py` expands `analysis/3A_panel.json` into the
exact entry list; `runners/3A-dilution.ps1 -Panel [-WithHen]` sequences it.

| role | entries |
|---|---:|
| panel (18 cells × top-3 conditions) | 54 |
| panel-hen (hawk's selections, mirrored onto `hen`) | 18 |
| anchored positive control | 4 |
| **total** | **76** over 37 checkpoints |

(The handoff §8 and `RESEARCH-STATUS.md` said "~40+ panel entries"; the panel
alone is 54.)

Four things the builder settles that a PowerShell array could not:

1. **The anchored positive control is carried through.** `select_3a_panel.py`
   excludes anchored runs by design — they are supervised, so they are not panel
   members — but they are the gate's other arm, and rule 3A.2 exists *only*
   because the anchored control came out `diluted` 22/23 times under 3A.1.
   Dropping them would remove the calibration check while making the gate look
   healthier.
2. **`hen` runs on hawk's own dictionaries**, because the polarity question is a
   matched-pair question — and hen has no eval-registry history to rank on.
3. **The random-control mapping lives in one place**, not as two copies of one
   regex in the runner's pre-flight and its exec loop.
4. **Which checkpoints still need an `_h`**, so the runner can encode first (§1a).

## 5. What is NOT done

- **The panel has not been run** -- it is prepared, and left for the user to
  start (see below). Cost is set by CONCEPTS, not entries, and the bases differ
  by 7x: gorilla 76 threat BSPs, hawk 171, hen 171, tiger 23. The 76 entries
  carry **8,083 concepts** (hawk 3,078 + hen 3,078 + gorilla 1,444 + tiger 483).
  At the measured **6.7 s/concept** (2026-08-17: 1,176 concepts in 2h12m) that
  is **~15 h** plus ~20 min for the usability gate -- not the ~6 h an
  entry-count estimate suggests. The diagnostic is CPU-bound and file-disjoint
  per `(run_id, bsps)`, so it could be parallelised; it is kept sequential
  because a detached overnight run is cheap and process orchestration in the
  runner is not.
- **The polarity hypothesis is untested** until the hen entries execute (§2).
- **`geometric_frac` with a cross-seed band** exists only for the two replicated
  conditions; the 3A.4 band is a within-run proxy everywhere else (§3).
- **`dead_features_pct` still gates panel health** through
  `check_sae_usable.py`, which under-counts alive latents for TopK by up to
  31 pp (handoff §3). That direction is conservative; the OTHER direction was
  not, and it killed the first panel run — see §7.
- **"cell" → "panel entry" has not been corrected** across older diary entries
  (handoff §8 item 4).

### How to start it

Everything GPU is already done: all 37 checkpoints have their `_h`, all six
random-model controls are present, and hen's labels exist for Ta/Ve/Yb. The run
is pure CPU from here.

```powershell
pwsh -File runners\launch.ps1 3A-dilution -Panel -WithHen
Get-Content logs\3A-dilution.transcript.log -Wait -Tail 40    # follow
```

`launch.ps1` spawns via WMI, so it survives an SSH disconnect, and refuses to
start a second copy while one is running. The runner re-checks its inputs
before doing any work: it rebuilds the runlist, validates dataset provenance,
finds nothing to encode, gates all 37 dictionaries on `alive >= top_k`, then
runs the 76 entries. It writes one
`analysis/{run_id}_dilution-{bsp_set}.json` per entry plus
`3A_gate_summary.json`, and ends with `stage_3A-dilution.md` listing what to
commit.

To sanity-check the plan without committing the hours, `-DryRun` prints the
entry list and runs the same usability gate -- but the gate loads 37 caches of
1.4-4.8 GB each, so budget ~20 min for it too.

Partial re-runs are first-class: `-Only henYb` or `-Only K04` filters the entry
list, and the filter is applied before the encode and gate stages.

## 7. POST-RUN: the first panel launch produced nothing [DIRECT]

The run was launched and **threw at the usability gate after ~40 min, having
written 0 of 76 reports**. One panel member failed:

```
E06-champYb-s42-batchtopk-k64-exp8-s4.conv2   38 alive   97.8% dead   47 fire>99.9%   UNUSABLE
```

`$ErrorActionPreference = 'Stop'` turns a non-zero exit from
`check_sae_usable.py` into a terminating error, which is the intended
behaviour — a panel measured at a smaller support than its peers is not a
panel. The defect is upstream, in **selection**.

**Two definitions of "alive" had drifted apart:**

| where | definition | E06-champYb |
|---|---|---:|
| `select_3a_panel.py` | `d_dict x (1 - dead_features_pct/100)`, from the registry | **85** — passes |
| `RankingCache` (diagnostic, `check_sae_usable`) | `min_freq < freq <= max_freq` — **excludes always-on latents** | **38** — fails |

85 − 47 always-on = 38, exactly. A latent firing on >99.9% of rows is a
constant column: it carries no information and cannot be a candidate, so the
diagnostic is right to exclude it and the selection proxy was wrong to count
it. The registry estimate is an **upper bound**, not a conservative one.

`[AI-REASONED PROVISIONAL ANALYSIS]` The general form is the one CLAUDE.md
already records for thresholds — *mixing measurement scopes inside one rule* —
here across two scripts rather than inside one. A selection filter and the gate
it is meant to anticipate must share an implementation, not a description of
one. §5 of this entry asserted the health proxy was "conservative, therefore
safe"; that was checked only for the TopK-deadness direction and is false for
the always-on direction.

**Fix.** `select_3a_panel.py` now MEASURES alive with `lib.sae.dilution.RankingCache`
— the same class the gate and the diagnostic use, so the two cannot drift —
wherever the `_h` cache exists, and falls back to the registry estimate only
for un-cached candidates, labelling those `alive_source: "estimate"` in the
panel JSON. A candidate that fails the measured check is recorded under
`rejected_on_measured_health` and the next-ranked condition is promoted from a
reserve list (`--reserve`, default 5), so a cell keeps 3 conditions rather than
silently shrinking to 2. The runner's gate message now names the re-selection
command, and says what happened here.

**Result**: E06-champYb rejected (est 85 / measured 38);
`C01-champYb-s42-topk-k16-exp8-s4.conv2` promoted into `Yb/s4.conv2/tiger`,
encoded, and measured at **230 alive**. The re-selected panel is 18 cells x 3
conditions with **every member measured** (no cell short, none below the floor,
tightest margin E05-champVe at 66 against `top_k = 64`), and all **43**
checkpoints the runner will gate — 33 panel + 4 anchored + 6 controls — now
have a measured alive at or above 64. The gate cannot fire on this panel.

`[AI-REASONED PROVISIONAL ANALYSIS]` Two members sit within 2 latents of the
floor (E05-champVe, 66). They pass, but `top_k` is not a natural constant, and
a panel whose weakest member can only just fill the candidate list is measuring
that member at the edge of its usable range. Worth revisiting if any
champVe/conv2 conclusion turns out to hinge on E05.

**Process note**: a `-DryRun` runs this exact gate and would have caught it in
~20 min. It was described as skippable. It is not.

## 6. Standing corrections not to re-break

As in the handoff, plus one. Never compare runs on a whole-basis scalar (compare
per concept family); `dead_features_pct` is architecture-dependent; tiger is
*mover-relative*, not owner-relative — Quarto has no owned pieces; every table
states its metric and units; **and a per-concept verdict is now quoted with its
stability annotation or not at all.**
