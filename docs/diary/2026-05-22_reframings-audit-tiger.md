# Design + scope — `tiger` BSP reframing (agent-relative threats)

Status: design + initial implementation (this entry). Evaluation pending — runs
after `bsp_labels-tiger*_<N>.pt` are computed on Deep Brain and the existing
H + F + E winners are re-evaluated against the new set.

Parent: [`../../RESEARCH-STATUS.md`](../../RESEARCH-STATUS.md) § "Active plan"
step 3. Triggered by [`phase-2B.md`](phase-2B.md) ch. 4 — Sweep H closed
the capacity branch of the decision gate; before committing supervised SAE
objectives (anchored / matryoshka / E2E) to a *fixed* BSP target, we should
validate whether gorilla / hawk are the right targets at all.

## Motivation [INFERENTIAL — competence-audit + capacity-scan combined]

Two facts in tension:

1. **champTa's loss-avoidance gain is +24 pp vs champS4** (model-competence
   audit, [`phase-2B.md`](phase-2B.md) ch. 3, test B). Minimax distillation
   *demonstrably* trained an asymmetric defensive sense into the model —
   the model is much better at *not letting the opponent win* than at
   *winning itself*.
2. **gorilla and hawk are state-only BSP sets.** They encode "there is a
   threat on line X" / "the offered piece shares attribute Y," not
   "**I am about to lose** if I place this piece anywhere," not
   "**every piece I can offer** gives the opponent a winning line." The
   target sets carry no signal about *whose* threat this is or *what the
   agent's options are*.

If champTa's representations are agent-relative (which the behavioural
gain strongly suggests), then a coverage measurement against gorilla / hawk
would systematically under-estimate the SAE's success — the SAE could be
encoding rich agent-relative concepts that simply *don't match* the
state-only BSP catalogue. The 11 % SAE / LP wall on conv2 / hawk could
then be partly a target-set mismatch, not a representation deficit.

`tiger` is the BSP set designed to test that hypothesis: it asks **what is
the agent's decision-relevant state**, not **what threats exist on the
board**.

## Scope — what `tiger` includes (and what it deliberately doesn't)

### Included (v1, ~30–40 BSPs)

| Category | # BSPs | What it captures |
|---|---:|---|
| `tiger_offered_disposition` | 3 | Mutually exclusive: offered piece is a `winner` (placeable to win this turn) / `poison` (every placement loses immediately) / `neutral`. |
| `tiger_offered_attr_match` | 4 | Per-attribute: the offered piece has attribute X (tall / black / square / with_hole) AND ≥1 line on the board has 3 cells matching X. Decomposes "offered is a winning completer" by which attribute is doing the work. |
| `tiger_decision_global` | 5 | `win_now_exists` (any cell-placement of offered wins) / `lose_next_forced` (every pool piece offers opponent a win) / `safe_offer_exists` / `opp_winning_after_some_offer` / `every_offer_safe`. |
| `tiger_pool_winning_count` | 4 | Bucketed: number of pool pieces that, if offered to opponent, would let them win. Buckets `≥1`, `≥2`, `≥4`, `all`. Captures "how poisoned is the pool". |
| `tiger_pool_safe_count` | 4 | Bucketed: number of pool pieces I could safely offer. Buckets `≥1`, `≥2`, `≥4`, `=0`. The `=0` bucket is the forced-loss signal — same as `lose_next_forced` but kept separately so per-category breakdowns are clean. |
| `tiger_line_winnable` | 10 | Per-line: placing the offered piece on this line's empty cell wins (the line is a *self-winning* line). |
| `tiger_square_winnable` | 9 | Per-2×2-square: placing the offered piece in this square's empty cell wins. |

**Total: 39 BSPs.** Comparable in size to hawk (173) divided into per-square
units; the count is small on purpose — these are the *agent-relative* signals
that gorilla / hawk lack, not a redundant catalogue.

### Deliberately excluded from v1

- **Depth-≥2 minimax-tree concepts** ("is this position forced 2 plies ahead"). Compute is expensive (≈ 16 × pool_size² per position over the 275 k amalgam); we can add a `tiger2` or `eagle` set later if v1 shows traction.
- **Turn-history concepts** ("did the opponent leave this line threatening"). Requires re-keying positions by game-progression; gorilla and hawk don't track this either, and our position dataset is per-position, not per-trajectory.
- **Defensive-blocking concepts** ("placing the offered piece *here* blocks a 3-line"). Inverse of `tiger_line_winnable`. Likely correlates with existing `threat_line`; defer until v1 evaluation tells us whether the agent-relative axis matters at all.

### Design choices worth flagging

- **`tiger` is a single basis**, not a gorilla ↔ hawk style pair. Unlike the threat reframing where gorilla = per-line and hawk = per-attribute-count, no obvious dual reframing exists for "agent's decision state." A second basis can be added if the v1 evaluation produces a clear axis to swap (e.g., "self" vs "opponent" stake).
- **No `offered_piece` reuse from gorilla.** Gorilla's `offered_piece` (4 BSPs at base-rate ≈ 0.5) is the trivial all-positive baseline — it carries no signal. Tiger's `offered_disposition` replaces it with a 3-way mutually exclusive split that is *not* trivially predictable.
- **Pool reasoning is the most distinctive part.** Of the seven categories, `tiger_pool_winning_count` and `tiger_pool_safe_count` (8 BSPs total) are the most clearly orthogonal to anything gorilla / hawk can express. If a single category is going to *move* the SAE coverage number, it will be one of these.

## Predicted outcomes [INFERENTIAL — pre-registered for honest comparison]

Two predictions, in order of confidence:

1. **`tiger_offered_disposition` and `tiger_decision_global` will have high SAE coverage** — the model is *required* to compute these to play at all, and the LP almost certainly reads them out cleanly. If the SAE fails on these, it's a feature-allocation failure, not a target-set issue. Confidence: high.
2. **Pool-reasoning categories will show the largest SAE/LP gap** — these are the most distinctive concepts, the model uses them (loss-avoidance gain), and they have rare base rates. If anything is going to behave like the conv2/hawk wall, it's these. Confidence: medium.

The interesting case is the *combination*: if tiger shows higher overall SAE coverage than hawk at the same SAE budget, then **a meaningful chunk of the conv2/hawk wall was target-set mismatch, not representation deficit**. That changes the design-note pivot: supervised SAEs would target tiger, not hawk, as the primary signal.

Conversely, if tiger sits at the same 10–15 % SAE/LP efficiency as hawk, then the wall is in the SAE objective itself (not the target framing), and the anchored/matryoshka/E2E pivot from the 2026-05-19 design note proceeds as planned.

Either way is informative.

## Evaluation plan

1. Implement tiger BSP definitions + compute functions in
   `scripts/games/quarto.py`. **(done, this entry)**
2. Register `tiger` in `BSP_SETS` so `compute_bsp_labels.py --name tiger`
   auto-resolves the category filter. **(done, this entry)**
3. On Deep Brain:

   ```bash
   python scripts/compute_bsp_labels.py \
       data/quarto/positions-amalgam_s4_unique.pt \
       --game quarto --name tigerS4
   python scripts/compute_bsp_labels.py \
       data/quarto/positions-amalgam_ta_unique.pt \
       --game quarto --name tigerTa
   ```

   Produces `data/quarto/bsp_labels-tigerS4_36.pt`, `bsp_labels-tigerTa_36.pt`,
   and the basis schema `bsp_schema-tiger_36.json` (shared, written once).
   `--name tigerS4` auto-resolves to the `tiger` category list via the
   basis-prefix lookup in `BSP_SETS`.

4. Run an **LP baseline** against tiger first:

   ```bash
   python scripts/linear_probe_baseline.py \
       data/quarto/s4.fc1_amalgam_s4_activations.pt \
       data/quarto/bsp_labels-tigerS4_36.pt \
       data/quarto/bsp_schema-tiger_36.json
   # ...and the conv2 / Ta variants
   ```

   The LP ceiling tells us whether the activations *contain* the
   tiger-readable directions before we ask the SAE about them.

5. **Re-evaluate the Sweep H + F + E winners against tiger** via the
   multi-bsps API:

   ```bash
   python sae_eval.py evaluate <ckpt> --bsps=tigerS4   # for champS4 runs
   python sae_eval.py evaluate <ckpt> --bsps=tigerTa   # for champTa runs
   ```

   Or piggy-back on the next run: `--bsps=gorillaS4,hawkS4,tigerS4`.

6. **Decision rule for the comparison:**

   | Outcome | Interpretation | Next action |
   |---|---|---|
   | tiger SAE / LP efficiency *higher* than hawk by > 0.10 | Target-set mismatch was contributing; agent-relative framing recovers some of the wall. | Anchored / matryoshka should target tiger, not hawk, as the supervision signal. |
   | tiger SAE / LP efficiency *within* ±0.05 of hawk | The wall is objective-side, not target-side. | Proceed with hawk-targeted anchored / matryoshka as in the 2026-05-19 design note. |
   | tiger SAE / LP efficiency *lower* than hawk | Tiger concepts are *less* SAE-readable — possibly because they require integration over the pool, which the activations might not encode cleanly. | Investigate where in the network pool-information lives (potentially a new hook); re-examine champTa's loss-avoidance mechanism. |

## Open questions

- *Pool-piece enumeration is metadata-derived, not from a quartopy API.* The current `metadata['cells']` and `metadata['offered_piece']` in `generate_positions` does not record the unplaced-piece pool explicitly. The implementation derives it from the 16-piece universe minus placed cells minus the offered piece — correct for canonical Quarto but depends on `mode_2x2=True` semantics. Verify the derivation on a few sample positions before running the full label generation.
- *Compute cost.* Pool-relative BSPs require ≈ pool_size × empty_cells × line_count = ~10 × 12 × 14 ≈ 1700 line-completion checks per position × 275 k positions ≈ 470 M ops. Pure Python this is 10–30 min. If it's worse than that, vectorise the `_would_win_at_position` helper or factor out a per-position pool-evaluation cache.
- *Cross-champion comparability.* tigerS4 and tigerTa label tensors are computed on each champion's self-play distribution. The base rates of `lose_next_forced` will differ — champTa engineers fewer of these — so direct cross-champion F1 comparisons are not apples-to-apples without conditioning on the base rates. The MCC and F1-lift columns largely correct for this (per the 2026-05-11 reporting standard).
- *Whether to anchor on tiger or hawk* in the anchored-SAE pivot is the question this evaluation is meant to answer. Pre-registering the decision rule above before seeing the numbers.

## Not on this list (and why)

- **Defining tiger as an extension of hawk** (gorilla → hawk → tiger as a chain of reframings). They attack different axes; chaining would conflate "rare-concept reframing" with "agent-relative reframing" and we'd lose the diagnostic separation.
- **Computing tiger labels for champAa.** champAa lacks the loss-avoidance behaviour (attack-recognition / defence-blindness asymmetry per Phase 1G); tigerAa would be theoretically computable but the diagnostic interest is on the champions that *do* show the behaviour.
- **Per-cell tiger.** Same reason as the per-cell hawk concern (CLAUDE.md § Things that have bitten): not on the current `--flatten-position` path.

---

## Results (closed 2026-05-22 PM) [DIRECT — from `saes/quarto/eval_registry.json`]

Labels: `bsp_labels-tigerS4_36.pt` (275 916 × 36, base-rates 0.028 – 0.887, 0 degenerate columns), `bsp_labels-tigerTa_36.pt` (88 524 × 36, base-rates 0.040 – 0.814, 0 degenerate columns). Sanity-check passed; tiger is a real signal axis.

### Headline table (F1-lift, count-weighted across 36 BSPs)

| Champion | Hook | LP-trained (F1 / MCC / lift) | Best SAE | LP-random (F1 / MCC / lift) | SAE/LP efficiency (lift) |
|---|---|---|---|---|---:|
| S4 | fc1   | 0.440 / 0.380 / **0.179**   | F04 — 0.376 / 0.201 / **0.112** | 0.258 / 0.178 / 0.038 | **62.5 %** |
| S4 | conv2 | 0.335 / 0.262 / **0.098**   | E05 — 0.325 / 0.161 / **0.061** | 0.271 / 0.197 / 0.046 | **62.2 %** |
| **Ta** | fc1   | 0.591 / 0.537 / **0.301** | **F04 — 0.449 / 0.333 / 0.158** | 0.278 / 0.206 / 0.048 | **52.5 %** |
| **Ta** | conv2 | 0.463 / 0.405 / **0.175** | E05 — 0.353 / 0.165 / 0.063 | 0.293 / 0.230 / 0.058 | **36.0 %** |

S4-fc1 LP was rerun at `max_iter=5000` to address 5 cosmetically-unconverged BSPs from the original `max_iter=1000` pass; the rerun returned 0.4402 / 0.3797 / 0.1789 (vs original 0.440 / 0.380 / 0.179) — numbers unchanged in the third decimal. The headline ceiling stands.

### Decision-rule outcome — **target-set mismatch was contributing**

Comparison anchor: phase-2B's "11 % conv2/hawk SAE/LP wall" (champTa). Tiger conv2/Ta sits at **36 %** efficiency, **+25 pp above** the hawk wall. Tiger conv2/S4 sits at **62 %**, **+51 pp above**. Both cells clear the pre-registered `+0.10` threshold by a wide margin.

Per the decision rule (this entry, §"Decision rule"):

> tiger SAE / LP efficiency *higher* than hawk by > 0.10 → Target-set mismatch was contributing; agent-relative framing recovers some of the wall. **Anchored / matryoshka should target tiger, not hawk, as the supervision signal.**

The two compatible mechanisms predicted in §"Why" of the 2026-05-19 concept-targeted design note remain in play; what tiger settles is *which target set* to anchor / matryoshka against, not whether to do them.

### Sub-claims supported by the per-category breakdown

- **fc1 > conv2 for tiger on both champions** (opposite of hawk on Ta, where conv2 > fc1). Agent-relative concepts live at fc1 — the bottleneck where the model integrates board + offered-piece + pool into a decision. This matches the H8-revised picture: unified-aux fc1 carries decision-relevant information.
- **Champion ordering Ta > S4 holds for tiger lift** (Δ +0.046 fc1, +0.077 conv2), consistent with H9 — minimax distillation transfers agent-relative competence, not just threat detection.
- **Sweep H null reproduces on tiger.** Expansion ladder (`H01`/`H02`/`H03` exp ∈ {16, 32, 64} on fc1; `H04`/`H05`/`H06` on conv2) is flat-to-negative on tigerTa F1-lift relative to the F04/E05 baselines. Capacity alone does not buy tiger coverage either.
- **F04-Ta-fc1 per-category lifts** (tigerTa): `tiger_square_winnable` **0.220**, `tiger_offered_completing_attr` **0.195**, `tiger_line_winnable` **0.147**, pool-count categories ≤ 0.124. The rare per-cell concepts move the headline; the pool-count categories sit at high F1 (≥ 0.71) but low lift because base rates are high. Pre-registered prediction #1 (decision-global + offered-disposition are well-covered) **lands**; prediction #2 (pool-reasoning shows the largest SAE/LP gap) is **mixed** — pool-safe-count has the largest gap, pool-winning-count does not.

### Feature-sharing observation (motivates matryoshka, separately)

F04-champTa-fc1 on tigerTa: `mean_bsps_per_feature = 1.44`, `max_bsps_per_feature = 6`. The conv2 winners go higher (E05-Ta conv2 max ≈ 10). Real feature absorption exists in the current champion-era SAEs — this contradicts the older champAa-based deprioritization of matryoshka ("missing concepts aren't being absorbed, they're never represented"). On champTa, *some* are being absorbed. Matryoshka stays on the active list, but anchored is the cheaper first move.

### Implication for next session

**Anchored SAE on `champTa fc1 → tigerTa`** is the highest-prior next step:

- Largest LP headroom in any cell where we have an LP win (F1-lift 0.301 vs SAE 0.158 → 0.143 absolute gap to close).
- Per-cell concept categories (`tiger_line_winnable` 10 BSPs, `tiger_square_winnable` 9 BSPs, `tiger_offered_completing_attr` 4 BSPs) match the candidate anchor-feature-count budget (23 anchored slots out of 1 024 = 2.2 % of the dictionary).
- Implementation cost is the lowest of the three supervised candidates (~2 days; see [2026-05-19_concept-targeted-saes.md](2026-05-19_concept-targeted-saes.md) §3).
- Diagnostic is sharp: if anchored features approach LP F1 on these 23 BSPs at λ_anchor ≪ 1, the wall was allocation; if they plateau well below LP, the activation space lacks sparse linear directions for those concepts and we revisit hook choice.

Detailed plan: [2026-05-22_anchored-sae-champta-tiger.md](2026-05-22_anchored-sae-champta-tiger.md).
