# Phase 3B-causal — Wave 1b built: gold sets, rule 3B.C2, dry runs (2026-09-14)

Build of the [Wave 1b pre-registration](2026-09-14_3B-causal-wave1b-preregistration.md)
(frozen at c29d859), its §14 steps 1–6. **No Wave-1b interchange score exists.**
Everything below is design-stage: self-play, labels, pair counts, and dry runs
that stop before any score. The pilot's R1–R6 stay sealed.

## Glossary

| term | range / ideal | meaning |
|---|---|---|
| **gold*k*** | — | champYb against itself: random through the *k*-th placement and the hand-over after it, then legal argmax for both sides; placements *k*+1 onward recorded |
| **distinct games** | [1, games] | distinct (board, piece) at the (*k*+1)-th placement; the continuation is deterministic, so this counts different games |
| **pos/game** | [0, 16 − *k*] | recorded placement decisions per game, before dedup |
| **not fresh** | [0, unique] | positions whose board, up to the 8 symmetries, is a pilot pair's board (removed, §4.3) |
| **any threat** | [0, 1] | share of positions with a line/square of three sharing a value and one empty cell |
| **win available** | [0, 1] | share where the piece in hand wins now (`tiger_win_now_exists`) |
| **powered in every arm** | [0, 176], ideal ≥ 88 | concepts with switch-on ≥ 100, specificity ≥ 100 and switch-off ≥ 50 pairs (design stage) |
| **feasibility** | pass / fail | ≥ 50 % of the 176 concepts powered in every arm (§4.4); a failing set is dropped |

## 1. What was built

| §14 step | artefact | verified by |
|---|---|---|
| 1 generator | `--opponents gold<k>` in `scripts/games/quarto_s4.py` (`gold_prefix`, `PrefixGreedyBot`) and `generate_positions.py` | `tests/test_gold_positions.py` (15): only placements *k*+1 on recorded; every later move is the legal argmax (re-derived from the network, move by move); the prefix is random and seeded; `gold0` has ≤ 16 distinct games. Two mutants caught (sampled continuation; random player leaving one hand-over early) |
| 2 sweep | `scripts/gold_prefix_sweep.py` → `saes/quarto/analysis/3B-causal_champYb_gold-prefix-sweep.json` | uses the run's own pair and power code |
| 3 sets | `scripts/build_gold_sets.py` (generate → dedup → `scripts/freshness_filter.py` → labels → orbit IDs), board-only key `compute_orbit_ids.board_keys` | `tests/test_freshness.py`; orbit IDs of the amalgam reproduce the saved file exactly; generator guard 0 mismatches |
| 4 rule 3B.C2 | `lib/sae/interchange.py` (`classify(rule=)`, `relative_leak`, `covariance_matched_directions`, `train_das_direction_multi`, `feasibility`), `scripts/interchange_3b.py` (`summarize_c2`, rule from the config) | `tests/test_interchange.py`, `tests/test_interchange_3b.py`; 3B.C1 regression (§5) |
| 5 configs | `configs/3B-causal/champYb-gold{3,5}.yaml` (`extends: champYb.yaml`) | inherit every Wave-1 key except the position set (test) |
| 6 dry runs | `runners/3B-causal.ps1 -Set gold3,gold5`, launched detached | §4 |

## 2. The descriptive prefix sweep (§4.2) — cannot change *k*

2,000 games per *k*, seed 4000 + *k*, CPU. Pair counts at this size, freshness applied.

| *k* | distinct games | pos/game | unique | not fresh | any threat | win available | median pairs on / off / spec | powered in every arm |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 1,535 | 9.85 | 14,654 | 50 | 0.380 | 0.107 | 52 / 0 / 1000 | 1 |
| 2 | 1,997 | 8.84 | 17,629 | 45 | 0.456 | 0.122 | 111 / 1 / 1000 | 4 |
| **3** | 2,000 | 7.42 | 14,839 | 41 | 0.536 | 0.144 | 132 / 5 / 1000 | 10 |
| 4 | 1,992 | 5.95 | 11,892 | 3 | 0.620 | 0.179 | 136 / 11 / 1000 | 22 |
| **5** | 1,968 | 4.50 | 8,999 | 0 | 0.725 | 0.231 | 105 / 15 / 1000 | 24 |
| 6 | 1,909 | 3.04 | 6,072 | 0 | 0.823 | 0.333 | 69 / 16 / 1000 | 24 |
| 8 | 1,647 | 1.39 | 2,786 | 0 | 0.963 | 0.618 | 18 / 6 / 509 | 12 |

[AI-REASONED PROVISIONAL ANALYSIS] Switch-off is the binding arm at every *k*:
greedy play rarely hands over a winning piece, so switch-off bases come from the
random hand-over and from blunders (§13 anticipated this). Longer prefixes put
more threats on the board but record fewer positions per game; the product
peaks around *k* = 4–6. None of this changes *k*; the gold sets are 15× larger,
and their own counts (§4) decide feasibility.

## 3. The gold sets (§4.1, §4.3)

| set | games | seed | raw | after dedup | not fresh (removed) | N | labels |
|---|---:|---:|---:|---:|---:|---:|---|
| gold3 | 30,000 | 3003 | 226,043 | 225,755 | 558 (0.25 %) | **225,197** | `{hawk,hen}YbGold3_173`, `tigerYbGold3_36` |
| gold5 | 30,000 | 3005 | 132,274 | 132,274 | 0 | **132,274** | `…YbGold5…` |

Generated on **CPU**: near-tied Q-values break differently on CUDA (one seed,
200 games: 1,488 positions on CPU, 1,489 on CUDA), so the device is part of the
recipe and of each file's provenance. The pilot's pairs covered 119,629 amalgam
rows (119,311 board orbits); gold boards almost never coincide with them.

## 4. Dry runs (§4.4) — design stage, no score

`pwsh -File runners\launch.ps1 3B-causal -Set gold3,gold5 -DryRun`, detached.

| set | guard | freshness re-check | Tier A | powered in every arm | per family (hawk / hen pinned, tiger) | feasibility |
|---|---|---|---|---:|---|---|
| gold3 | 0 mismatches × 176 | 0 stale | A1a, A1b, A2, A3 pass; closed form | **128 / 176 (73 %)** | 52/76, 52/76, 24/24 — the 48 short are switch-off only (median 67, min 4) | **pass** |
| gold5 | 0 mismatches × 176 | 0 stale | all pass; closed form | **176 / 176** | all | **pass** |

Both sets enter the analysis. Switch-on and specificity reach their caps (1,000)
for every concept in both sets. In gold3 the 48 pinned concepts without
switch-off power can still earn `concept-consistent (on-only)`; that is the
registered rule, not a new allowance.

## 5. Verification of the build

- **3B.C1 unchanged**: the pre-refactor code (from HEAD) and the new code ran
  Wave 1 on the **untrained twin** (5 concepts, 200 null draws): every score and
  every per-pair record identical. The only differences were intended — a
  `rule_version` field, the Wave-1b pre-registration in the freeze stamps, and
  two verdicts moving off-target → inert (§6).
- **3B.C2 end to end**: untrained twin on gold3 (5 concepts, replicates on):
  every registered field present (network-own IIA_net\*, F / E / ρ, the oracle
  score, the isotropic null beside B1′, per-kind ceilings under both targets;
  records carry D(b) and D(s)). Numbers meaningless by construction.
- **Timing**: ~15–19 s per concept at 1,000 null draws → about 1 h per set;
  the runner runs both sets in parallel, one GPU each.
- Suite: 1,075 passed, 13 skipped.

## 6. A correction to the 3B.C1 code: off-target

Amendment 1 §A3 defines off-target as "switch-on **not significant**, but the
flip rate above its null's 95th percentile". `interchange.classify` omitted
"not significant", so a switch-on that was BH-significant but below the 0.20
floor, with a significant flip rate, read off-target instead of inert. The
truth-table test had been written from the code, not the text. Fixed for both
rules; guarded by `test_off_target_needs_an_insignificant_switch_on`.

Effect on the pilot (R7 only; R1–R6 not examined): **one verdict**,
`tiger_win_now_exists`, off-target → inert. The gate is unaffected (neither is
concept-consistent). The pilot's JSON is left as recorded; verdicts are a pure
function of its stored arm statistics, so its R1–R6, when unsealed as
exploratory under 3B.C1 (§12), are derived with the corrected rule.

## 7. Choices the pre-registration leaves open (no design change)

- CPU generation; the device recorded (§3).
- B1′ sampled exactly as ε·Δ_c/√(m−1), ε ~ N(0, I): no factorisation, and
  Σ_Δ is singular (m pairs span at most m − 1 dimensions).
- DAS-1's per-kind averaging skips a kind with no training pairs in a fold.
- Specificity CI: orbit bootstrap of F.
- An infeasible set would be written as a DROPPED record with no score.
- Gold labels use `compute_bsp_labels.py --schema-out none`: the tracked basis
  schemas are not rewritten; the label columns are checked against them.
- The shortlists, 3A `knee_k` and probes stay the amalgam's (§13): the gold
  configs keep `bsp_suffix: Yb` and add `labels_suffix: YbGold3/5` for the guard.

## 8. Next

The two runs (about an hour, in parallel). `--require-frozen` checks the design
documents, but each run also records `git HEAD`, so the implementation should be
committed first for that stamp to name the code that ran.
