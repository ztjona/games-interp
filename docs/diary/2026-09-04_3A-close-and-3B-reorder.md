# Phase 3A closed for real: paper reconciliation, plan reorder, top-K export (2026-09-04)

Phase 3A's *science* closed on 2026-08-25. This entry closes its *bookkeeping*:
the LXAI paper branch is folded back in, the one substitution it introduced is
promoted from a paper-build workaround to a project artefact, the Phase 3 plan
is reordered so the causal step runs first, `subspace-aware SAEs` (SASA) is
adopted as 3C's lead arm, and the top-K candidate export that 3B-causal depends
on is built.

Nothing here changes a 3A number. Everything here removes a way for the docs to
contradict themselves.

## 0. Glossary

| Term | Range / ideal | Meaning |
|---|---|---|
| availability *A* | [0, 1], ideal 1 | linear-probe MCC on raw activations — is the concept present at all |
| isolation *I* | [0, 1], ideal 1 | best single-latent MCC — does one dictionary atom carry it |
| efficiency | *I / A*, [0, 1], ideal 1 | share of available signal the dictionary factors into one atom |
| FVU | [0, 1], ideal 0 | fraction of variance unexplained by the SAE reconstruction |
| `mcc_at_pref` | [-1, 1], ideal 1 | MCC restated at p_ref = 0.025; the cross-population-comparable number |
| `knee_k` | integer ≥ 1, ideal 1 | latents needed before restricted-R² plateaus |
| `argmax_f1_rank` | integer ≥ -1, ideal 0 | where the F1-argmax feature sits inside the MCC shortlist; -1 = outside it |
| d_i | integer ≥ 1 | intrinsic dimension of a feature (SASA); our analogue is the disjunct count |
| G11 | gate | causal validation — decodability ≠ use. Still OPEN. |

Standing definitions: [`../methods-reference.md`](../methods-reference.md) §1, §3.

## 1. The paper branch, folded back in [DIRECT]

`origin/paper-phase3A` was **3 commits strictly ahead of `phase3A`, zero
divergence** — a fast-forward. It adds `papers/LatinXAI/` (the LXAI @ NeurIPS
2026 submission) and touches exactly one non-paper file, `RESEARCH-STATUS.md`,
in two places. Both edits are corrections, and both have the same root cause.

**champYb's conv2 numbers were computed on a dictionary the project had already
condemned.** `E05-champYb-…s4.conv2` is degenerate — FVU 0.110, 99.0 % dead, 41
alive — because BatchTopK lacked the Gao et al. dead-feature auxiliary loss
until 2026-08-15. 3A *already* replaced it: on 2026-08-17 the panel member went
`E05` → `K04` (FVU 0.0066, 174 alive), because 41 alive < `top_k = 64` meant
`E05` could not even fill the diagnostic's candidate list.

That replacement was never propagated to the **hook-specialisation /
efficiency** numbers, which were computed on 2026-08-14, three days before the
repair existed. So the docs carried `E05`-based efficiency figures next to
`K04`-based 3A verdicts. The paper caught it. The corrections:

| Claim | Was (`E05`) | Is (`K04`) |
|---|---|---|
| champYb `square_threat` conv2 best latent | 0.115 | **0.178** |
| fc1-over-conv2 isolation, 12 state-threat cells | "3–7×" | **1.8–4.8×** |

**Verified independently, not taken on trust.** Recomputing all 12 matched
state-threat cells from `papers/LatinXAI/numbers.json` gives a range of
**1.79× (Ta gorilla/`line_threat`) to 4.81× (Yb gorilla/`square_threat`)**. So
`1.8–4.8×` is right — and the superseded "3–7×" was never right at the low end
either, since the minimum was 1.79× even before the repair.

**One inconsistency found inside the paper branch and fixed here.** Its two
edits disagreed with each other: `RESEARCH-STATUS.md` line 36 said `1.4–4.8×`
while the H6b row said `1.8–4.8×`. `1.8` is correct; line 36 is fixed.

The paper README also logs five wording corrections made against
`numbers.json` / `numbers_3a.json` during the 2026-08-25 update (pooled pinned
capture is 73.2 % over n = 2,736, not 76.7 % over n = 1,672; the panel spans 14
unsupervised recipes, not six; the clipped bin is "at 20 or beyond"). **Each was
checked against this diary and `RESEARCH-STATUS.md`: none of those overstatements
exist here.** They were paper-draft-only. §5.1 of the final report recomputes
exactly — 73.2 % is the n-weighted mean of its own 76.7 % / 67.8 % rows.

## 2. The substitution is now an artefact, not a build-time patch [DIRECT]

**Do we agree with `K04` for `E05`? Yes — and it is not a new decision.** It is
3A's own 2026-08-17 panel decision, applied to the one set of numbers that
predates it. Three independent reasons, in order of weight:

1. **`E05` fails the project's own usability gate.** 41 alive latents against
   `top_k = 64`: it cannot fill the candidate list, so its verdict describes a
   broken dictionary rather than the hook. `scripts/check_sae_usable.py` rejects
   it. A number no gate would accept cannot be a number a claim rests on.
2. **It is the worse dictionary on every within-run health signal.** FVU
   0.110 → 0.0066 (17×), dead 98.97 % → 95.75 %, `coverage_mcc` on gorillaYb
   0.163 → 0.367. (Whole-basis scalars are not a basis for *comparing runs* —
   CLAUDE.md — but they are exactly the within-run "did this dictionary
   collapse?" signal that rule permits.)
3. **Using it would flatter the paper's own thesis.** The claim is that fc1
   isolates state threats and conv2 does not; representing conv2 by its *worst*
   dictionary inflates that gap. `K04` is the conservative choice.

Honest detail: `K04` is not uniformly better. `coverage_mcc` on **tigerYb** goes
0.128 → **0.115**, slightly *worse*. The substitution is not cherry-picked for
outcome; it is made on dictionary health and applied wherever champYb conv2
appears.

**But the paper's *mechanism* for the substitution was wrong for the project.**
`make_tables.py::apply_conv2_repair` swapped the column at table-build time,
reading `K04` out of `eval_registry.json`. That left
`saes/quarto/analysis/E05-…conv2_sae-lp-efficiency.json` on disk as the only
conv2 efficiency artefact, so every *future* consumer re-inherits the bug and
only the paper knows better. **Fixed at the source:**

```bash
python scripts/sae_lp_efficiency.py --champ=Yb --hook=s4.conv2 \
    --run-id=K04-champYb-s42-batchtopk-k32-exp8-s4.conv2
git mv saes/quarto/analysis/E05-champYb-…conv2_sae-lp-efficiency.json \
       saes/quarto/analysis/superseded/
```

The generated report reproduces the paper exactly — gorilla `square_threat`
SAE 0.178 / LP 0.946 / efficiency 0.19 — which independently confirms the
paper's registry-based swap was computing the right thing.

**Reproducibility check.** Rebuilding the paper tables against the new artefact
set gives 90 availability rows, identical keys, and **zero differences in every
reported field** (`lp`, `sae`, `efficiency`, `run_id`, `n_bsps`, `base_rate`).
The only values that moved are the `sae_legacy` / `efficiency_legacy`
bookkeeping fields, which now record `K04`'s own numbers instead of `E05`'s
because the rows no longer originate from the degenerate report. The paper's
§4.6 sign-flip robustness check reads the registry directly and is unaffected:
still **2 flips of 15 families, both per-cell, all 12 relational families
unchanged**.

`apply_conv2_repair` is kept as an idempotent guard — if anyone regenerates the
`E05` report it still corrects — but its docstring and the paper README now say
the substitution lives upstream.

### General rule this earns

> **A repaired dictionary must be substituted in the ARTEFACT, not in the
> script that reads it.** A build-time patch fixes one consumer and hides the
> defect from every other one. If a report on disk is wrong, regenerate it and
> move the old one to `superseded/`.

## 3. Plan reorder: 3B-causal runs first [DECISION]

**Decision: the causal step runs before the ground-truth-geometry step.** This
was already the standing instruction in three places (`2026-07-21` litv2
reassessment "runs before any 3C compute"; final report §9 "3B-causal runs
first"; `phase-3A.md` "3B-causal still runs first (G11)") but the plan's own
numbered list contradicted it by putting geometry at 2 and causal at 3. That
contradiction is now removed.

**Why causal first, restated so it is checkable:**

1. **It can falsify the headline; geometry cannot.** Every 3A number is
   *decodability*. Final report §9 sharpens the question: are the clean pinned
   atoms the ones the policy uses, or is the disjunction computed somewhere the
   dictionary never sees? If the latter, "the dictionary reconstructs a 4-way
   disjunction from exactly its four disjuncts" is a fact about a decoder, not
   about the network. The literature prior is live, not hypothetical:
   `adversarial-world-model-verification` found high-accuracy chess board-state
   probes to be causally epiphenomenal (probe-vs-next-move gradients at cosine
   distance ~0.99 across 24 models).
2. **It sizes 3C's budget**, which is the expensive phase.
3. **Geometry will read the same in three months.** α is known by construction
   and β comes from the width sweep already on disk; it is bankable write-up
   material with no null risk, and the causal answer may change which concepts
   are worth measuring it for.

### How the change is reported: NO rename

The obvious move — renaming `3B-causal` to `3B.1` to mark it as first — is
**rejected**, for two concrete reasons:

- **`3B.1` is already taken.** It denotes the three-term decomposition
  (availability → survival → isolation), defined in
  [`2026-08-14_hook-specialisation.md`](2026-08-14_hook-specialisation.md) §7
  and referenced by H6b. That work sits on the *geometry* side, so promoting the
  causal track to `3B.1` would collide head-on.
- **Diary entries are frozen by contract**, and eight of them reference
  `3B-causal` with its current meaning. A rename makes every one of them
  silently wrong — manufacturing exactly the contradictions this pass exists to
  remove.

**Instead: the IDs are stable, the ORDER is made explicit.** The change is an
ordering change, so it is encoded as ordering:

- the working-steps table in [`phase-3.md`](phase-3.md) gains a **`Runs`**
  column (1 / 2 / 3 / 4) and its rows are sorted by it;
- `RESEARCH-STATUS.md`'s numbered plan is re-sorted to match;
- the geometry track gains the **display alias `3B-geom`** for new prose, since
  a bare "3B" is ambiguous once a sibling runs before it. This is an *alias*,
  not a rename: `3B` remains valid and every existing reference stays correct.
  Entries dated before 2026-09-04 that say "3B" mean `3B-geom`.

## 4. SASA adopted as 3C's lead arm [DECISION]

14 paper summaries entered the DB since the 2026-07-14 synthesis v2. Thirteen
are agentic-interp, LLM-behaviour or off-domain. One is load-bearing:

**`subspace-aware-sparse-autoencoders-effective`** (Dalili & Mahdavi, arXiv
2606.06333, DB topic `sparse-autoencoders`) — SASA.

It supplies the theory 3A's result was missing and the architecture 3C's brief
was groping for:

- **Splitting is provably objective-driven, not an artefact.** With a
  single-direction decoder per latent, a feature of intrinsic dimension
  d_i ≥ 2 forces splitting two ways: a covering-number bound
  (Ω((t/ε)^((d_i−k)/k)) atoms) and a basis-instability theorem — the true
  d_i-dimensional basis is a *strict saddle* of the ℓ1 SAE objective, with a
  continuous path to lower risk leading away from it. The ℓ1 objective **wants**
  to split.
- **The converse gives the fix.** Once block size r ≥ d_i, a single group is the
  global minimiser and recovers the feature subspace exactly.
- **The architecture:** block/group decoders, Top-*s* group gating, a
  nuclear-norm penalty for rank adaptation, a dead-group aux loss. Code
  released. Empirically GPT-2 absorption 37.2 % → 6.6 %, and Engels et al.'s
  35-atom temporal cluster collapses to one rank-6 group.

**Why it belongs here specifically.** SASA's central quantity is d_i, and the
paper cannot measure it — in an LLM you never know a feature's true intrinsic
dimension, so the converse theorem is untestable and r is a guessed
hyperparameter. **Quarto knows d_i by construction, and 3A already measured its
proxy**: `knee_k` mode = 1 / 4 / 9 against disjunct counts of 1 / 4 / 8
(§5.2 of the final report). That makes `r = disjunct count` a *pre-registered*
prediction rather than a swept hyperparameter — the same "measurable here,
unmeasurable in LLMs" argument that motivates Brill's α/β, applied to a sharper
and more recent theorem.

It also partly repairs a stated limitation. The `diluted`/`tiled` decline
(2026-08-24) forced the concession that *3A establishes that the geometry binds
but not which geometry, so the 3C arm choice rests on literature priors rather
than our measurement.* SASA narrows that: it gives a principled reason to prefer
an aggregating/block readout over a manifold method, and its knob is the
quantity we measured.

**Caveats, recorded now so they are not discovered later:**

- **The mapping is an analogy, not an identity.** SASA models a feature
  occupying a d_i-dimensional *subspace*; our concept is a *disjunction* over
  k poles — a union of conditions. A union of k conditions spans a k-dimensional
  subspace of the code, so the block decoder should apply and `r ≈ knee_k` is
  the natural prediction, but this needs testing before 3C compute is committed
  to it. This project has twice been burned by exactly this class of over-read.
- **The paper's empirics are thin**: two models, single runs, no seeds or error
  bars, and the headline token-budget comparison is not fully apples-to-apples
  (SASA at half budget vs externally-trained baselines). Treat the theory as
  strong and the numbers as suggestive.
- SASA's own limitation section notes it recovers a subspace's *span* but not an
  interpretable coordinate system inside it.

**Consequence for 3C.** The arm list becomes **SASA-style block decoder (lead)**
vs Matryoshka / H-SAE / MP-SAE (hierarchical) vs bilinear slots (manifold). The
DB flags SASA, `finding-manifolds-bilinear-autoencoders` and `smixae` as three
independent 2026 approaches converging on one diagnosis; a **focused three-paper
note** comparing them is warranted before 3C. A full synthesis v3 is **not** —
one core paper in fourteen does not justify re-reading 278 summaries.

## 5. Top-K candidate export — BUILT [DIRECT]

Plan item 7, and 3B-causal's hard prerequisite. Reporting standard 4 has said
since 2026-07-27 that *feature→BSP alignment is greedy argmax on decodability,
not causality* — the matched feature may be a spectator while the causally-used
one ranks #2 or lower — but `lib/sae/eval.py` persisted only the argmax, so the
rule had no artefact to act on.

**Design decision: derive, don't re-run.** The `_matching-<animal>.pt` cache
already holds the full `(d_dict, num_bsps)` metric matrices. A top-K shortlist is
therefore a `torch.topk` over data already on disk — **no `_h` cache, no GPU, no
re-encode, and no change to the cache format**, so all 598 existing caches work
unchanged and no `--force` re-eval (which *would* need `_h`) is triggered.

- `lib/sae/eval.py`: `TopKMatches` + `top_k_features_per_bsp(matching, k, metric)`.
  Game-agnostic, tensors only. Ranks by **MCC** by default (standard 1 makes it
  the headline; standard 4 prefers it at base rate ~0.02), with
  `mcc_at_pref` / `youden_j` / `f1` available. `k` clamps to the dictionary size.
  Companion metrics are gathered *at the selected features*, so a row's
  precision/recall/F1 always describe that row's candidates.
- `scripts/export_topk_matches.py`: CLI → `{run_id}_topk-{animal}.json`, with an
  embedded `glossary` key (standard 6). `--all [--filter=<glob>]` batches.
- `tests/test_sae_eval.py::TestTopKFeaturesPerBSP` — 9 tests, incl. a planted
  feature, descending order, companion-gather correctness, and an explicit test
  that F1-ranked and MCC-ranked shortlists **differ** (if they ever stop
  differing, the fixture has stopped exercising the thing the export exists for).
  Suite 961 → 969.

### `argmax_f1_rank`: standard 4's flag, now measured

Each BSP records where the F1-argmax feature lands inside the MCC shortlist —
`0` = the metrics agree, `-1` = F1's pick is not in the top-16 at all. The
standing warning turns out to be **understated**:

| cell | BSPs | F1-argmax ≠ MCC-argmax | F1-argmax outside top-16 |
|---|---:|---:|---:|
| K05-champYb s4.fc1 / gorillaYb | 164 | 59 | 11 |
| K05-champYb s4.fc1 / tigerYb | 36 | 12 | 0 |
| K03-champYb s4.fc1 / gorillaYb | 164 | 49 | **16** |
| K04-champYb s4.conv2 / hawkYb | 173 | 57 | 7 |
| K04-champYb s4.conv2 / tigerYb | 36 | 19 | **5** |
| R3-champYb**random** s4.fc1 / gorillaYb | 164 | 81 | 11 |

So on roughly **a third of concepts the two metrics disagree on the single best
feature**, and on up to 10 % the F1 pick is not among the sixteen best by MCC.
Any causal experiment seeded from the F1 argmax would have started from the
wrong feature that often. The random-model control disagreeing *most* (81/164)
is the expected direction — with no learned structure, both rankings are noise.

Reports generated for the 3B-causal target cells (champYb: `F04`/`K05` on
`s4.fc1`, `K04` on `s4.conv2`, × gorilla/hawk/tiger — 9 files, 2.4 MB).
Everything else regenerates in seconds:

```bash
python scripts/export_topk_matches.py --all --filter='*champYb*' --dry-run
```

**This is a candidate list, not a causal result.** The ordering is still
decodability; 3B-causal re-ranks it by intervention effect. Nothing in the
export decides which feature the network uses.

## 6. What is deliberately NOT done

- **No `_h` cache pruning.** Reviewer comments on the LXAI submission have not
  arrived, and re-measurement may be needed; 354 GB stays until 3B-causal starts.
- **No re-run of any 3A verdict.** Verdicts are a pure function of stored
  metrics and none of those metrics moved.
- **No synthesis v3** (§4).
- **The `knee_k` dose-response ladder is NOT scheduled** — see §7.

## 7. The k-dose ladder: assessed, and deferred [DECISION]

Considered: build BSP categories that OR over 2, 3, 5, 6, 7 attribute poles
(from the 8 available — 4 hawk positive + 4 hen negative) and check whether
`knee_k` tracks the disjunct count across the whole range, instead of at the
three points 3A measured (1, 4, 8).

**It is cheap** — labels are CPU work, and the diagnostic runs against `_h`
caches already on disk at 6.7 s/concept, so ~50 concepts × 9 dictionaries is
well under an hour.

**It is deferred anyway**, for three reasons:

1. **It reopens a closed phase.** 3A is closed, its 114 reports are frozen, and
   the submitted paper's Table 4 / §4.5 rest on them. Adding BSP categories to
   a closed basis is the same class of move as adding BSPs to hawk — which
   CLAUDE.md already forbids because it silently changes banked numbers. A new
   basis would be required, with its own labels and its own panel run.
2. **The confound is real and not yet designed away.** Base rate rises
   monotonically with the number of OR'd poles, so prevalence is collinear with
   the dose. `knee_k ∝ k` and `knee_k ∝ base rate` would be indistinguishable
   without a matched-prevalence design (`mcc_at_pref`, or comparing across
   different pole *subsets* at fixed k). Running it undesigned would produce a
   pretty curve that means nothing — the 2026-08-11 per-mode-coverage retraction
   is exactly this mistake.
3. **It is not on the critical path.** It strengthens a result already
   published; it cannot change what 3C should be, and it does not test G11.

**Where it does belong:** as the empirical half of the planted positive control
the `diluted`/`tiled` decline asked for, and as a **direct test of SASA's
`r = d_i` prediction** (§4) — which is a 3C question, not a 3A one. Revisit it
when 3C needs to choose block ranks, with the prevalence design settled first.
If reviewers ask for a dose-response, this is the answer, and it is ~1 hour of
compute away.

## 8. Reproducing

```bash
python scripts/sae_lp_efficiency.py --champ=Yb --hook=s4.conv2 \
    --run-id=K04-champYb-s42-batchtopk-k32-exp8-s4.conv2
python scripts/export_topk_matches.py --all --filter='*champYb*' --dry-run
python -m pytest tests/test_sae_eval.py -k TopK -q
cd papers/LatinXAI && python make_tables.py --metric=mean_mcc   # numbers unchanged
```

## 9. Handoff

- **Next: 3B-causal.** Gradient-alignment screen (∂decision-margin/∂activation
  vs the LP direction, against a random-direction null) over the exported top-K
  shortlists, then clamp/steer as the discriminator. Build the **planted
  positive control first** — a case where the causally-used feature is known —
  before pointing the screen at real dictionaries.
- **Then 3B-geom** (α/β, κ_ms, Park polytopes, H11), which needs no new compute.
- **3C** with SASA as lead arm, after the three-paper architecture note.
- **Open, carried forward from 3A:** the `diluted`/`tiled` statistic with a
  planted control; champTa/Ve conformance; a power flag for under-powered
  concepts; the `instrument/metrics` track.
