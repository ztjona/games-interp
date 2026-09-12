# Training a new Quarto champion — recommendations from the interpretability side

Written 2026-09-12, on request, for the training run happening in parallel with
Phase 3B-causal. Self-contained: it assumes the reader has not followed the
interpretability diary. Hand it to the training agent as-is.

## Why a new champion, in three lines

The current champion, **champYb** (`Yb_hotChamp(3)`, hot-piece shaping,
λ = 1.0, seed B, 10k epochs), never learned which moves are legal. Its place
head's raw top-1 is an occupied cell **67.2 %** of the time and its select
head's **28.5 %** (competence-audit Test F); the bot filters illegal moves at inference, and every training
loss masks them, so their Q-values receive no gradient and drift freely. That
is fine for play, and the interpretability work is designed around it — but a
champion that *does* know legality is worth having as a comparison.

The single most important recommendation is in §3: **train it as a controlled
pair**, so the new champion is an experiment rather than a confound.

## 1. The comparability contract — keep these fixed

Every analysis in this project (SAE sweeps, the 3A dilution diagnostic, the
3B-causal interchange) must re-run on the new champion **unchanged**, and its
numbers must be comparable to champYb's. That holds only if these stay fixed.

| keep fixed | why |
|---|---|
| **Trunk shape**: conv1 16 ch → conv2 32 ch on the 4×4 grid → `fc1` 512 → two linear heads (`fc2_place`, `fc2_select`, 16 each) | hook names (`s4.conv2`, `s4.fc1`) and dimensions are baked into configs, SAE recipes and runners |
| **Nothing between `fc1` and the heads** — no extra layer, no skip connection into the heads | this is what makes the downstream map from `fc1` exactly `tanh(W·relu(z) + b)`: all causal influence lives in a known 32-dim subspace, which 3B-causal exploits. An extra layer destroys that |
| **Input encoding**: board one-hot (16 × 4 × 4) + 32-d aux (`offered_one_hot ⊕ available_mask`) | BSP labels and counterfactual inputs (swapping the offered piece) are generated against this encoding |
| **Functional ReLU** (`F.relu(self.fc1(x))`, no `nn.ReLU` modules) | hooks capture the module output, i.e. the **pre-activation**. If this changes, say so — every intervention depends on which side of the ReLU the hook sits |
| **Output activation** (tanh) and the two-head, phase-routed design | Q-value ranges and the head split are assumed by the audits |
| **The game engine** (Quartopy's retry-on-invalid `QuartoGame`) and the position protocol: 4 opponent modes, aggregate then deduplicate, `mode_2x2=True` | self-play positions are the dataset every analysis runs on |

**If something must change, change one thing at a time**, and name what changed
in the checkpoint's metadata.

## 2. Legality: make an illegal move a loss — without playing it

**Keep the existing masked RL losses as they are.** In particular the TD
target's max stays over **legal** next moves, whatever else changes: before
legality is learned, the illegal Q-values are arbitrary, and bootstrapping from
them would feed that noise into every legal value.

Then teach legality. Three ways, in order of preference:

**A. Illegal move = loss, as *synthetic* transitions (recommended).** For any
state already in the replay buffer, the outcome of each illegal action is known
without playing it: the game ends and the mover loses. So add those transitions
to the TD loss directly — `(s, illegal a) → terminal, reward = loss` — for every
illegal action of every sampled state.

- **Dense**: every state supervises every illegal action on both heads, instead
  of only the action actually taken.
- **No wasted games**: nothing illegal is ever played; the game the agent
  learns on is unchanged.
- **The signal never goes away**: every batch keeps supervising Q(illegal), so
  it cannot drift back once learned (option B's failure mode).
- **Q on illegal actions becomes a real value** — the value of forfeiting —
  so all 16 outputs of each head mean something. That is the best outcome for
  interpretability: champYb's illegal Q-values are untrained noise, and every
  analysis has to exclude them.

Weight or cap the synthetic transitions per batch so they do not swamp the real
ones. The fit near −1 is slow under tanh (the gradient `(1 − q²)` shrinks), but
it only has to push Q(illegal) below every legal Q, which happens long before
saturation.

**One subtlety for acceptance.** In a forced-loss position every legal move
loses too, a little later: its value is about −γᵏ, just above the forfeit's −1.
The ordering is right but the margin is tiny, so an illegal top-1 can survive
there. It changes no game outcome (every move loses anyway), so **measure Test F
excluding forced-loss positions**.

**B. Illegal move = loss, played for real.** Let the agent choose illegal moves
during training and end the game as a loss when it does. This is the literal
version, and it works, but it costs more:

- **It needs action selection unmasked.** Today the bot filters illegal moves
  before they are played, so a forfeit reward would never fire.
- **Wasted games.** With ε-greedy exploration over all 16 actions at ε = 0.1,
  roughly 30 % of full-length games would end in a random forfeit from place
  moves alone [AI-REASONED ESTIMATE: per-move forfeit probability ε·(occupied
  cells / 16), compounded over one player's 8 placements]. Early on it is much
  worse: champYb's greedy choice is illegal 67 % of the time.
- **The signal disappears once learned.** When the greedy policy stops picking
  illegal moves, Q(illegal) stops receiving gradient and can drift again as the
  representation keeps changing — champYb's problem, returning slowly.
- Early value targets are dominated by forfeit outcomes, which slows the
  learning of actual strategy.

**C. A margin loss on the pre-tanh logits.** For each illegal action *a*:
`max(0, logit(s,a) − min_legal logit(s) + m)`. Dense, and it avoids tanh
saturation by acting on logits. But it only enforces *ordering*: Q(illegal)
means "below the legal moves", not a value. Use it if A is awkward to
implement.

Whichever you use, **accept the smallest weight that reaches raw illegal top-1
< 1 % on both heads on held-out positions** (forced-loss positions excluded)
with no loss of competence (§5).

**Learn from the earlier attempt first.** `QC_unifiedNoMask` (2026-05-11) had a
legality loss and was rejected because its illegal top-1 stayed at **0.32**.
Before training, find out why: the loss weight, tanh saturation (a −1 target on
the tanh output stops pushing as q → −1), or too few epochs. A loss that plateaus at 32 % is a symptom, not a property of the
task — occupancy is directly in the input one-hot, so legality is trivially
available to the network.

**Measure it the way the analysis will.** Report raw illegal top-1 **per head**
and **by game phase** (pieces on board: 1–4 / 5–8 / 9–12 / 13–15). The failure
concentrates mid-to-late game (champYb: 82.8 % illegal at 5–8 pieces).

## 3. Train a controlled pair, not a single champion

The most valuable design for the research: **two recipes that differ only in
the legality term** (§2, option A preferred), same everything else.

| arm | legality loss | seeds |
|---|---|---|
| **L0** (reference) | off — the current recipe | ≥ 2 |
| **L1** (treatment) | on (§2, option A) | ≥ 2, **the same seeds** as L0 |

Same initialisation per seed, same number of epochs, same opponents, same
hot-piece shaping. The question this answers is then clean: *does learning
legality change how the network represents and uses threats?* [AI-REASONED
PROVISIONAL ANALYSIS] It might: to suppress 16 place outputs on occupied cells,
the network must route occupancy into the same 32-dim subspace the heads read,
where threats would then compete for it.

**Why ≥ 2 seeds.** `Ya` and `Yb` are seed siblings of one recipe and tie
head-to-head (49.1 %); seed variation is as large as many recipe effects.
Without seeds, a representation difference between L0 and L1 cannot be told
apart from seed noise.

**Optional second factor — dropout at `fc1`.** The S4 trunk applies
`nn.Dropout(0.5)` directly after `relu(fc1)` during training — the very layer
the SAEs analyse. [AI-REASONED PROVISIONAL ANALYSIS] Dropout rewards redundancy
(a concept spread across many units survives any one being dropped), which could
contribute to the "spread" representations Phase 3A measured. A
`QuartoCNNAutoregLowDropout` class already exists in `models/quarto/CNN_autoreg.py`.
If compute allows, a full 2 × 2 (legality × dropout 0.5 / 0.1) is the clean
design. If not, **keep dropout at 0.5** so the pair stays comparable to champYb,
and leave dropout for later. Do not change both at once in a single arm.

## 4. Save the provenance

Each checkpoint should carry, or ship next to it:

- [ ] full training config, seed, number of epochs;
- [ ] git commit of the trainer **and** the Quartopy version;
- [ ] every loss term with its weight, and an explicit masking record, e.g.
      `legal_masking: {td_target_max: true, select_loss: true, hot_loss: true, legality_loss_weight: 0.0}`;
- [ ] the **untrained twin** (`E_0000`) from the same initialisation — the
      random-model control every analysis uses;
- [ ] **intermediate checkpoints** (e.g. every 1,000 epochs). Cheap, and they
      let us ask *when* during training a concept or legality appears;
- [ ] the benchmark results (JSONL, as in `champion-results.jsonl`).

## 5. Acceptance criteria

Select the champion on **competence only**. **Never select on
interpretability outcomes** (SAE coverage, 3A verdicts, 3B-causal results):
choosing a champion because it makes the analysis look good would make every
later comparison circular.

| criterion | champYb, for reference | target |
|---|---|---|
| head-to-head vs champYb | — | ≥ 50 % |
| vs MinimaxBot depth 2 | 44.5 % | ≥ 44.5 % |
| audit A — takes an immediate win | 97.2 % | ≥ 97 % |
| audit B — avoids handing over a losing piece | 98.6 % | ≥ 98 % |
| audit D — Q(empty) − Q(occupied) | −0.31 | > 0 (L1 arm) |
| audit F — raw illegal top-1, place / select | 67.2 % / 28.5 % | < 1 % / < 1 % (L1 arm, forced-loss positions excluded); report for L0 |

The audit is `scripts/model_competence_audit.py` in the interpretability repo
(tests A–F; F, the raw illegal rate per head, was added 2026-09-12).

## 6. Handoff to the interpretability repo

For each accepted checkpoint (and its untrained twin):

1. copy both into `models/quarto/`;
2. write `configs/models/champ<Tag>.yaml` (path, random path, game module,
   benchmarks, and the masking record from §4);
3. reuse the `quarto_s4_hot` game module if the class is unchanged; otherwise
   add a sibling module;
4. `runners/champ<Tag>.ps1` then produces positions → activations → the SAE
   sweep → evaluation, and the 3A and 3B-causal runners take `-Champ <Tag>` and
   re-run unchanged.

Anything in §1 that changed must be named in the yaml, because it decides which
analyses are still comparable.
