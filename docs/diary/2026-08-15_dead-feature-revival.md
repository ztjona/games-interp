# Dead-feature revival existed in one architecture out of three (2026-08-15)

Status: frozen self-contained record. Parent ledger: [`phase-3.md`](phase-3.md).
Metric definitions: [`../methods-reference.md`](../methods-reference.md) §1.
Code state is `[DIRECT]` from `lib/sae/architectures.py` and `sae_train.py`;
the A/B numbers are `[DIRECT]` from the K01/K02 runs; interpretation is
`[AI-REASONED PROVISIONAL ANALYSIS]`.

## 0. Why this side work exists

Phase 3B.0A was about to start when a simpler question got asked: *are the SAE
hyperparameters wide enough? Every conclusion rests on these dictionaries having
been trained correctly.* Before spending compute on a re-sweep, the cheap check
is whether the training code does the same thing for every architecture.

It does not. **`TopKSAE` carried a dead-feature auxiliary loss; `BatchTopKSAE`
and `JumpReLUSAE` had no such mechanism at all.** So an architecture comparison
in this project has been, in part, a comparison of training machinery — and the
two architectures pinned as 3A / basis-verdict panel members are the two that
lacked it.

That is not a hyperparameter-width problem. It is a correctness problem
upstream of it, and it has to be resolved before a sweep can be interpreted,
because a sweep over `k` and expansion cannot distinguish "this architecture is
worse here" from "this architecture cannot recover a feature once it dies".

## 1. The finding [DIRECT]

`lib/sae/architectures.py`, before this change:

| architecture | dead-feature revival | note |
|---|---|---|
| `TopKSAE` | **yes** — Gao et al. 2024 aux loss, `aux_loss_weight = 1e-2` | carries a comment recording an earlier bug where the term was a zero-gradient step function, "so dead features could never revive" |
| `BatchTopKSAE` | **no** | `compute_loss` returns reconstruction only |
| `JumpReLUSAE` | **no** | `compute_loss` returns reconstruction + the L0 penalty |
| `GatedSAE` | n/a | its `l_aux` is the via-gate reconstruction of Rajamanoharan et al. 2024a — a different mechanism, routing gradient to `W_gate`, not reviving dead columns |
| `VanillaSAE`, `PAnnealingSAE` | no | dense/L1 variants; less exposed, since nothing structurally zeroes a column |

`sae_train.py` recorded the asymmetry in a comment rather than fixing it —
`# BatchTopKSAE has no aux_loss_weight parameter` — while the module docstring
advertised `--aux-loss-weight` as "TopK/BatchTopK", so the CLI help and the code
disagreed.

The dead-feature census across the champion era lines up with it:

| architecture | revival | dead % range | example |
|---|---|---|---|
| TopK | yes | 61.7 – 95.8 | `F01-champYb` fc1: **85.6%**, coverage MCC **0.502** |
| BatchTopK | no | 91.8 – 99.8 | `E05-champYb` conv2: 99.0%, MCC 0.163 |
| JumpReLU | no | 92.7 – 99.4 | `F04-champYb` fc1: 93.5%, MCC 0.424 |

At 4096 dictionary slots, 99.0% dead is **41 alive latents**. 3A's `top_k = 64`
cannot even be filled, so that run's `asymptote_r2` is measured at a smaller
support than every other cell in the panel.

## 2. What was also wrong with the mechanism that did exist

Two deviations from Gao et al. 2024, both of which make the existing TopK
revival weaker than the reference — and both of which affect the *reported*
`dead_features_pct` as well, since the metric uses the same definition:

1. **Aliveness was per-batch.** `alive_mask = (h > 0).any(dim=0)` counts a
   feature dead if it did not fire in *this* batch. Gao et al. define dead as
   "has not fired in the last N tokens" (~10M). At `batch_size = 4096` with
   `k = 32`, only `B*k` of `B*d_dict` slots can fire at all, so a healthy but
   rare feature reads as dead in most batches. **Part of the 85–99% figure is
   therefore a measurement artefact rather than a training outcome** — how much
   is not yet known.
2. **`k_aux` was the model's own `k`** (32), where Gao et al. use a separate,
   much larger value (512). Revival capacity was ~16× smaller than the
   reference.

## 3. The change [DIRECT]

The aux term moved to **`BaseSAE`** as `_aux_dead_loss` + `_update_dead_mask`,
and `TopKSAE`, `BatchTopKSAE` and `JumpReLUSAE` each call it. One
implementation, so the next fix lands in all of them at once — which is exactly
the failure mode that produced this entry.

New knobs, all defaulting to **current behaviour**:

| knob | default | effect |
|---|---|---|
| `aux_loss_weight` (TopK) | `1e-2` | unchanged |
| `dead_revival` → `aux_loss_weight` (BatchTopK, JumpReLU) | **`0.0` = OFF** | a *deliberately separate* CLI flag from `--aux-loss-weight`, so a config that does not mention it trains exactly as before |
| `dead_window` | `0` | `0` reproduces per-batch aliveness bit-for-bit; a positive value is the Gao et al. "has not fired in N steps" definition |
| `aux_k` | `None` | falls back to `k` (TopK/BatchTopK) or `l0_target` (JumpReLU) — the previous behaviour |

**Every checkpoint already on disk still loads and every banked run still
reproduces.** The revival counter is registered `persistent=False`, so it is
absent from `state_dict` and cannot break `load_state_dict`, which
`lib/sae/train.load_checkpoint` calls in strict mode. Verified directly on
`F04-champYb`, `E05-champYb` and `F01-champYb`.

**A latent device bug surfaced and was fixed in passing.** Both anchored
constructors called their base with `device` as a *positional* argument, so
inserting a parameter before it silently routed `device` into another slot and
every tensor landed on the default `cuda`. Both calls are now keyword-form. This
was caught by the existing `TestAnchoredSAE` tests, which is the system working.

### Guards added (`tests/test_sae.py`)

- `test_every_sparse_arch_can_revive_dead_features` — parametrised over topk /
  batchtopk / jumprelu; fails if any sparse variant stops sending gradient to a
  dead column. **This is the test that would have caught the original
  asymmetry.**
- `test_revival_is_off_by_default` — the fix adds a capability, it does not
  silently change the objective of checkpoints already in the registry.
- `test_dead_window_zero_matches_per_batch_aliveness` and
  `test_dead_window_requires_sustained_silence` — pin both definitions.
- `test_revival_counter_is_not_in_state_dict` — pins the load-compatibility
  guarantee above.

Suite: 183 passed, 2 skipped (was 176 passed, 1 skipped).

## 4. The A/B [DIRECT]

Two runs, campaign **K**, each identical to its partner in every field except
the revival term:

| run | partner | arch | hook |
|---|---|---|---|
| `K01-champYb-s42-jumprelu-t64-exp8-s4.fc1` | `F04-champYb` | JumpReLU t64 | `s4.fc1` |
| `K02-champYb-s42-batchtopk-k32-exp8-s4.conv2` | `E05-champYb` | BatchTopK k32 | `s4.conv2` |

`dead_window` stays at 0 in both, so the A/B isolates *revival present vs
absent* and does not simultaneously change the definition of dead. champYb was
chosen because it is the champion carrying the phase's headline result and
because `E05-champYb` is the worst-behaved dictionary in the panel — the largest
available signal if revival is the cause.

### 4.1 Result [DIRECT]

**The two architectures were failing in completely different ways, and only one
of them was failing because of this.**

| | E05 → K02 (BatchTopK, conv2) | F04 → K01 (JumpReLU, fc1) |
|---|---|---|
| FVU | **0.1079 → 0.0076** (14× better) | 0.0099 → 0.0089 |
| MSE | **61.63 → 4.35** | 1.05 → 1.15 |
| dead % | **99.0 → 96.1** (42 → 158 alive) | 93.5 → 93.7 (267 → 258 alive) |
| L0 | 32.0 → 32.0 | 66.1 → 67.6 |
| `coverage_mcc` gorillaYb | **0.163 → 0.343** (+111%) | 0.424 → 0.421 |
| `coverage_mcc` hawkYb | **0.079 → 0.114** (+44%) | 0.419 → 0.411 |
| `coverage_mcc` tigerYb | 0.128 → 0.111 (−13%) | 0.238 → 0.236 |
| `coverage_youden_j` gorillaYb | 0.330 → 0.428 | 0.569 → 0.573 |

Every K01 change is within ±0.013, i.e. seed-level. Every substantive K02 change
is large.

**BatchTopK was genuinely collapsed; JumpReLU was not.** E05's 99.0% dead came
with FVU 0.108 — the dictionary was failing to reconstruct at all. F04's 93.5%
dead comes with FVU 0.0099, excellent reconstruction: its dead features are
*unneeded*, not collapsed, and giving it a revival term changes nothing.

**Diagnostic consequence:** high `dead_features_pct` is not by itself a fault
signal. The pair `dead% AND FVU` is. A health gate on dead% alone would have
passed F04 (correctly) and only barely caught E05.

### 4.2 What repairing conv2 does to the hook comparison [DIRECT]

Efficiency = SAE `mcc_at_pref` ÷ LP `mcc_at_pref`, champYb, `s4.conv2`, gorilla:

| family | n | LP conv2 | E05 SAE | K02 SAE | eff E05 | eff K02 |
|---|---:|---:|---:|---:|---:|---:|
| `board_attribute` | 64 | 0.985 | 0.108 | 0.465 | 0.11 | **0.47** |
| `board_occupancy` | 16 | 0.977 | 0.075 | 0.514 | 0.08 | **0.53** |
| `offered_piece_attr` | 4 | 0.982 | 0.011 | 0.086 | 0.01 | 0.09 |
| `game_phase` | 3 | 0.781 | 0.086 | 0.099 | 0.11 | 0.13 |
| `line_threat` | 40 | 0.958 | 0.122 | 0.174 | 0.13 | **0.18** |
| `square_threat` | 36 | 0.946 | 0.115 | 0.175 | 0.12 | **0.19** |
| `global_threat` | 1 | 0.244 | 0.093 | 0.077 | 0.38 | 0.32 |

The per-cell families recover 4–6× and land back in the range champTa/champVe
conv2 already showed (0.28–0.53). **The threat families barely move** — 0.13 →
0.18 and 0.12 → 0.19, against fc1's 0.62 and 0.91.

So the [hook-specialisation result](2026-08-14_hook-specialisation.md) is
**strengthened, not weakened**: with a healthy conv2 dictionary the threat
isolation gap survives almost intact, and what the broken dictionary had been
hiding was conv2's competence on *spatial* concepts, not on threats. The caveat
recorded in §4 of that entry is now discharged with data rather than a hedge.
Its champYb conv2 SAE column understates by 4–6× on the per-cell rows and by
~1.5× on the threat rows; the LP columns and every conclusion drawn from them
stand as written.

### 4.3 Scope of the damage [DIRECT]

Distinct checkpoints in the registry, by architecture:

| family | n | status |
|---|---:|---|
| `batchtopk` + `anchored-batchtopk` | **90** (61 champion-era) | trained without revival; K02 shows the effect can be large |
| `topk` | 55 | always had it — unaffected |
| `jumprelu` + `anchored-jumprelu` | 69 | lacked it, but K01 measures the effect as null at fc1 |
| `vanilla`, `gated`, `p-annealing` | 12 | dense / L1 variants, not structurally exposed |

**Limits of this A/B.** One seed, one champion, and the OFF arm is a run that was
an outlier even among its own recipe siblings (champTa and champVe at identical
settings sat at FVU 0.043 / 0.051, not 0.108). The honest claim is *"revival
fixes this collapse and the collapse was caused by its absence"*, not *"revival
is worth 14× on BatchTopK in general"*. Whether the milder champTa/champVe
BatchTopK runs are also depressed needs the same A/B on one of them — two runs,
and it decides whether the 90 checkpoints need retraining or only a caveat.

The tiger regression on K02 (0.128 → 0.111) is real but sits where conv2 tiger
is near the floor anyway (LP availability 0.27–0.43), so it is close to noise
and should not be read as revival harming agent-relative concepts without a
second seed.

## 5. What this does and does not invalidate

- **It does not touch the linear-probe numbers.** Availability is measured on
  raw activations; no SAE is involved. So the 2026-08-14 hook result
  ([`2026-08-14_hook-specialisation.md`](2026-08-14_hook-specialisation.md)) —
  conv2 carries state threats at least as well as fc1 — is unaffected.
- **It does put every cross-*architecture* claim on notice**, including "TopK
  wins tiger, JumpReLU wins hawk", which is now partly attributable to one
  architecture having a mechanism the other lacked.
- **Within-architecture, cross-champion comparisons are unaffected** — the same
  code ran for all three champions. The champYb-vs-champVe-vs-champTa results
  stand as comparisons.
- **`dead_features_pct` should be read with §2.1 in mind** everywhere it appears
  in the registry and in the atlas.

## 6. Consequences for the plan

1. The panel health gate proposed in the 2026-08-14 review (`dead ≤ 98%`,
   `alive ≥ top_k`) is necessary but not sufficient — a panel member should also
   be trained with revival available, or the gate just filters for architectures
   that happen to have the mechanism.
2. The hyperparameter-width question is **not yet answered**; this was the
   prerequisite. It resumes with: did runs stop early under `patience`, and is
   coverage still climbing at the edge of the swept `k` / expansion range?
3. Phase 3B.0A follows.

Bite-mark added to `CLAUDE.md`: a training mechanism added to one architecture
must be added to all of them, or an architecture comparison measures the
implementation.

## 7. Reference conformance — the full audit

The question that produced this section: *if an architecture is prone to dead
latents, that is a property of the method and an argument against it — why patch
it?* The objection is correct in principle, and it makes the decisive question
empirical: **is the mechanism part of the published method, or are we adding
one?** Checked against the papers:

| | auxiliary loss | `W_enc = W_dec^T` init | our recipe before 2026-08-15 |
|---|---|---|---|
| **TopK** — Gao et al. 2024 | required | **required** (their mitigation *(1)*, listed first) | aux ✓, **init ✗** |
| **BatchTopK** — Bussmann et al. 2024 | **required**, "auxiliary loss (same as TopK) retained" | inherits TopK's setup | **aux ✗, init ✗** |
| **JumpReLU** — Rajamanoharan et al. 2024b | **none, by design**: "No auxiliary losses, no resampling" | **not specified** | aux ✓ (correctly absent), init **underdetermined** |

So the answer differs per architecture, and only one of them is a patch:

- **BatchTopK**: we implemented the method minus a component it specifies.
  Restoring it is a *correction*, and K02 measures the size of the omission.
- **TopK**: conforming on the aux loss, non-conforming on the init.
- **JumpReLU**: adding an auxiliary loss would be reporting a JumpReLU result
  for a variant of JumpReLU. The default stays **off**, on principle — and the
  A/B independently measures the effect as null, so nothing is being given up.

**A correction to the first reading of this table.** "The paper does not specify
an init" is not the same as "the paper specifies kaiming". JumpReLU's init is
*underdetermined* by its source, so our independent-kaiming choice was arbitrary
rather than canonical — which means adopting the tied init for JumpReLU would
not be a deviation from the paper, only a choice among options it leaves open.
That matters because K05 shows the choice is not free (§7.2).

### 7.1 What "canonical by default" now means

Defaults are the published form; every deviation must be written into the
config. That inverts the failure mode — before, the deviation lived in the code
and nothing recorded it.

| architecture | default aux | default init |
|---|---|---|
| `topk`, `batchtopk`, `anchored-batchtopk` | `1e-2` | `decoder_transpose` |
| `jumprelu`, `anchored-jumprelu` | `0.0` | `kaiming` |

**230 existing configs were stamped** with the settings they were actually
trained with (`init_mode: kaiming` on all 230; `aux_loss_weight: 0.0` on the 91
batchtopk-family ones), so a rerun reproduces its own recipe rather than
inheriting today's defaults. `--dead-revival` remains as a JumpReLU-only,
explicitly non-canonical experiment flag.

`tests/test_configs.py` guards this: every config must pin `init_mode`, the
TopK family must pin `aux_loss_weight`, no two configs may resolve to the same
run-id, and `experiment:` may not duplicate arch or hook. Suite: 870 passed.

Two things it caught on its first run — a run-id "collision" that was the
*test's* bug (it read `l1_weight` where gated configs carry `gated_l1`; it now
drives the trainer's own `get_arch_kwargs`), and four D-campaign experiment IDs
that violate the 2026-04-24 naming rule, grandfathered explicitly rather than by
weakening the check.

### 7.2 Campaign K in full [DIRECT]

champYb. `alive` and FVU are eval-time; `d_input = 512`, so **effective
expansion = alive / 512** — the number that says whether a nominally 8×
overcomplete dictionary is overcomplete at all.

⚠️ **The gorilla / hawk / tiger columns below are whole-basis `coverage_mcc` and
must NOT be used to compare runs.** They are retained only as a coarse
health/regression signal (did this run collapse?), because that is a
within-run reading. Every comparison in this entry is made in §7.3 on the
**concept-family** rollup. A whole-basis mean over gorilla is 39%
`cell_attribute` and 46% threats, so it can move for reasons unrelated to the
concepts under study — see the correction notice in §7.3(c).

**s4.fc1**

| run | recipe | gorilla | hawk | tiger | FVU | alive | eff exp |
|---|---|---:|---:|---:|---:|---:|---:|
| F01 | TopK k32, legacy | **0.502** | **0.454** | **0.268** | 0.0159 | 590 | 1.15× |
| K03 | TopK k32, canonical | 0.500 | 0.451 | 0.252 | 0.0151 | 646 | 1.26× |
| F04 | JumpReLU t64, legacy | 0.424 | 0.419 | 0.238 | 0.0081 | 266 | 0.52× |
| K01 | JumpReLU + aux | 0.421 | 0.411 | 0.236 | 0.0088 | 252 | 0.49× |
| K05 | JumpReLU + tied init | **0.459** | **0.445** | **0.257** | 0.0059 | 350 | 0.68× |

**s4.conv2**

| run | recipe | gorilla | hawk | tiger | FVU | alive | eff exp |
|---|---|---:|---:|---:|---:|---:|---:|
| E01 | TopK k32, legacy | 0.330 | 0.102 | 0.133 | 0.0105 | 172 | 0.34× |
| K06 | TopK k32, canonical | 0.312 | 0.098 | 0.119 | 0.0094 | 204 | 0.40× |
| C01 | TopK k16, legacy | 0.333 | 0.103 | 0.150 | 0.0515 | 227 | 0.44× |
| K07 | TopK k16, canonical | 0.269 | 0.092 | 0.153 | 0.0241 | 364 | 0.71× |
| E05 | BatchTopK k32, legacy | 0.163 | 0.079 | 0.128 | 0.1100 | 42 | 0.08× |
| K02 | BatchTopK + aux only | 0.343 | 0.114 | 0.111 | 0.0081 | 158 | 0.31× |
| K04 | BatchTopK k32, canonical | **0.367** | **0.119** | 0.115 | 0.0066 | 174 | 0.34× |

**Seed replication** (conditions carrying a load-bearing comparison; whole-basis
figures, health signal only):

| condition | gorillaYb s42 / s43 / s44 | mean | sd |
|---|---|---:|---:|
| K03 TopK fc1 canonical | 0.500 · 0.507 · 0.495 | 0.501 | 0.005 |
| K05 JumpReLU fc1 + init | 0.459 · 0.462 · 0.463 | 0.462 | 0.002 |
| K06 TopK conv2 canonical | 0.312 · 0.327 · 0.318 | 0.319 | 0.006 |
| K04 BatchTopK conv2 canonical | 0.367 · 0.368 · 0.365 | 0.367 | 0.001 |

Seed noise is 0.001–0.006, so single-seed campaign-K results were safe — now
demonstrated rather than assumed. The per-family seed sds in §7.3 are the ones
that govern whether a family-level gap is real.

### 7.3 Three findings [AI-REASONED PROVISIONAL ANALYSIS]

**(a) The two mitigations do different jobs.** The auxiliary loss rescues a
*collapse*; the tied init prevents columns from *starting* in the trap. E05's
BatchTopK had collapsed (FVU 0.110 — it was not reconstructing at all) and the
aux loss alone recovers 17× of that. JumpReLU never collapsed (FVU 0.0081), the
aux loss is inert for it (K01, null) — mechanistically coherent, since a
JumpReLU latent dies by `z < theta` everywhere and the `relu(z)` the aux term
feeds on is already ~0 — and only the init moves it.

**(b) Capacity is not the binding constraint, and on conv2 more capacity is
actively worse.** K03 adds 56 live latents on fc1 for zero coverage change
(0.502 → 0.500). On conv2 the tied init adds capacity and *costs* coverage,
replicated at two sparsity levels: k32 172 → 204 alive for −0.018 gorilla; k16
**227 → 364 alive for −0.064**. So "our dictionaries are secretly undercomplete"
is true as a description and is **not** the explanation for the SAE/LP wall.

The conv2 direction is the interesting one: more live atoms, worse
single-latent recovery, is exactly what feature splitting predicts — the same
concept spread across more atoms leaves the best *single* one weaker. That is
the phenomenon 3A exists to measure, showing up here as a side effect of an
initialisation change. It is a candidate handle on dilution that costs one
training run to manipulate.

**(c) The conv2 architecture ranking reverses under conformance — on the cell
families. On threats the reversal is real but an order of magnitude smaller.**

⚠️ **The first version of this section compared whole-basis `coverage_mcc`,
which this project forbids** (`CLAUDE.md`: "whole-basis averages are not
interpretable"; `methods-reference.md` §2: "a basis is a packaging convention;
a CATEGORY is the analysis unit"). Holding the basis fixed across the two arms
is not a defence: gorilla is 39% `cell_attribute` and 46% threats, so a
whole-basis number can be carried entirely by families the research is not
about. It was. Corrected below; all numbers are **means over 3 seeds** (42/43/44)
and **micro-averaged over BSPs within a family**, which is what
`aggregate_per_category_by_family` computes.

**conv2, k = 32, canonical vs canonical — BatchTopK (K04) minus TopK (K06)**

| basis / family | n BSPs | K04 | K06 | gap | seed sd |
|---|---:|---:|---:|---:|---:|
| gorilla/`board_attribute` | 64 | 0.643 | 0.556 | **+0.088** | 0.011 |
| gorilla/`board_occupancy` | 16 | 0.526 | 0.451 | **+0.075** | 0.019 |
| gorilla/`line_threat` | 40 | 0.114 | 0.099 | +0.014 | 0.002 |
| gorilla/`square_threat` | 36 | 0.109 | 0.096 | +0.014 | 0.002 |
| hawk/`line_threat` | 90 | 0.119 | 0.102 | +0.017 | 0.001 |
| hawk/`square_threat` | 81 | 0.114 | 0.097 | +0.017 | 0.002 |
| tiger/`line_threat` | 10 | 0.092 | 0.071 | +0.020 | 0.004 |
| tiger/`square_threat` | 9 | 0.092 | 0.071 | +0.021 | 0.004 |
| tiger/`offered_completion` | 4 | 0.139 | 0.086 | +0.053 | 0.015 |
| gorilla/`game_phase` | 3 | 0.219 | 0.264 | −0.045 | 0.069 |
| gorilla/`offered_piece_attr` | 4 | 0.289 | 0.316 | −0.027 | 0.050 |
| tiger/`pool_reasoning` | 8 | 0.202 | 0.218 | −0.017 | 0.056 |

BatchTopK wins **every threat family** (+0.014 to +0.021, 5–10× the seed sd) and
wins the cell families by 4–6× as much. So the reversal is genuine and
consistent, but its magnitude lives in `board_attribute` and `board_occupancy`.
The three cells where TopK leads are the smallest families (n = 3, 4, 8) and
carry the largest seed sd (0.050–0.069) — noise, not signal.

**fc1, canonical vs canonical — TopK (K03) minus JumpReLU (K05)**

| basis / family | n BSPs | K03 | K05 | gap | seed sd |
|---|---:|---:|---:|---:|---:|
| gorilla/`offered_piece_attr` | 4 | 0.488 | 0.245 | +0.242 | 0.049 |
| gorilla/`game_phase` | 3 | 0.409 | 0.222 | +0.187 | 0.045 |
| gorilla/`board_attribute` | 64 | 0.403 | 0.357 | +0.046 | 0.003 |
| gorilla/`board_occupancy` | 16 | 0.389 | 0.345 | +0.044 | 0.007 |
| gorilla/`line_threat` | 40 | 0.476 | 0.455 | +0.021 | 0.012 |
| gorilla/`square_threat` | 36 | 0.768 | 0.754 | +0.014 | 0.009 |
| hawk/`line_threat` | 90 | 0.374 | 0.369 | +0.004 | 0.007 |
| hawk/`square_threat` | 81 | 0.541 | 0.543 | **−0.002** | 0.004 |
| tiger/`line_threat` | 10 | 0.175 | 0.172 | +0.004 | 0.009 |
| tiger/`square_threat` | 9 | 0.248 | 0.266 | **−0.018** | 0.009 |
| tiger/`offered_completion` | 4 | 0.213 | 0.295 | **−0.082** | 0.025 |

**"TopK beats JumpReLU on fc1" is RETRACTED as a statement about threat
concepts.** TopK's whole-basis lead is carried by `offered_piece_attr` (+0.242,
n = 4, and the known F1 ≈ 0.667 trivial-baseline family), `game_phase` (+0.187,
n = 3) and the cell families. On the threat families the two are **tied** —
hawk square −0.002, hawk line +0.004, tiger line +0.004 — and **JumpReLU leads
on the agent-relative ones** (tiger square −0.018, tiger `offered_completion`
−0.082).

That matters for the panel: F04/K05 (JumpReLU) is the fc1 panel member, and on
the concepts this project is about it is not the weaker architecture.

### 7.4 What this means for the banked results

| claim type | status |
|---|---|
| cross-**champion** (champTa/Ve/Yb) | **stand** — identical code for every champion |
| cross-**hook** (fc1 vs conv2) | **stand** — identical code for both, and the LP side has no SAE in it |
| cross-**architecture** | **unsafe** — a rank reversal is not a footnote. Any such claim needs canonical runs on both arms |
| absolute coverage values | mildly understated for JumpReLU (−8%), badly understated for legacy BatchTopK |

This is a narrower blast radius than "90 checkpoints must be retrained", and a
wider one than the footnote proposed before K06 existed. The rule now: **an
architecture comparison requires both arms in canonical form**; everything else
is untouched.

### 7.5 The JumpReLU init decision

Adopted `decoder_transpose` as the default for `jumprelu` and
`anchored-jumprelu`, on the evidence in §7.2 (+6–8% on all three BSP sets, 32%
more live latents, 27% lower FVU).

**This is a choice, not conformance, and the code says so.** The map in
`sae_train.py` was renamed `CANONICAL_INIT` → `DEFAULT_INIT` with per-row
provenance, because the TopK-family entries are specified by their papers while
the JumpReLU entries are a decision on an axis Rajamanoharan et al. 2024b leave
open. Pinned by `test_jumprelu_init_is_a_recorded_choice_not_a_spec`. The
auxiliary loss stays **off** for JumpReLU: that one *is* specified, and K01
measured adding it as null.

The change surfaced one more instance of the failure this whole entry is about:
`AnchoredJumpReLUSAE` kept its own `init_mode` default and would have silently
trained a differently-initialised dictionary than the base class it is meant to
be a supervised version of. Fixed, and guarded by
`test_anchored_defaults_track_their_base_class`. Suite: 888 passed, 11 skipped.

Even fully canonical, K05 sits at 0.68× and K04 at 0.34× — **still
undercomplete** against a nominal 8×. Conformance was necessary and is not the
answer to the hyperparameter question; that resumes with early stopping and the
swept `k` / expansion range.
