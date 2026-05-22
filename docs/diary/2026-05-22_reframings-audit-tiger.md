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
       data/quarto/positions-amalgam_s4.pt \
       --game quarto --name tigerS4
   python scripts/compute_bsp_labels.py \
       data/quarto/positions-amalgam_ta.pt \
       --game quarto --name tigerTa
   ```

   Produces `data/quarto/bsp_labels-tiger{S4,Ta}_<N>.pt` and a basis
   schema `bsp_schema-tiger_<N>.json`.

4. Run an **LP baseline** against tiger first:

   ```bash
   python scripts/linear_probe_baseline.py \
       data/quarto/s4.fc1_amalgam_s4_activations.pt \
       data/quarto/bsp_labels-tigerS4_<N>.pt \
       data/quarto/bsp_schema-tiger_<N>.json
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
