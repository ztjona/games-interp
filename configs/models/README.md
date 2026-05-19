# Adding a new champion

A "champion" is a (game, model checkpoint) pair that the interpretability
pipeline is run against. This README is the checklist for onboarding a new
one without breaking existing artifacts.

Existing entries are good templates: `champAa.yaml` (CNN_uncoupled,
fc1=128) and `champS4.yaml` (unified-aux autoregressive, fc1=512).

## TL;DR

1. Pick a short CamelCase tag (e.g. `S4`, `Bb`, `T5`). It propagates into
   filenames, BSP-set names, and SAE `experiment:` fields. Avoid lowercase
   suffixes that collide with the metals/animals theme.
2. Register the champion: write `configs/models/champ<X>.yaml`.
3. If the architecture is new, add a game module at
   `scripts/games/quarto_<arch>.py` and wire it into
   `scripts/games/__init__.py`.
4. Stage one phase's pipeline (positions → BSP labels → activations →
   SAE sweep → evals) in the transient `commands.sh` at the repo root.
5. Create a `configs/champ<X>/` directory with the SAE YAMLs for the
   phase. Every `experiment:` field must include `champ<X>`.
6. Verify with the checklist at the bottom.

## 1. Champion YAML — `configs/models/champ<X>.yaml`

Minimal schema (kept in sync with `model_competence_audit.py`'s
`--model-config`):

```yaml
name: champ<X>                                  # CamelCase tag, matches filename
display_name: "<one-line human description>"
game: quarto | quarto_<arch>                    # which game module to load
path: models/quarto/<trained_checkpoint>.pt
random_path: models/quarto/<epoch_0_random>.pt  # used for random-net controls
# Informational comments:
#   - hookable layers
#   - d_act per hook (helps when sizing SAE expansion)
```

`name` and the filename's `<X>` suffix must match. Both `path` and
`random_path` must already live in `models/quarto/` and be tracked in git
(force-add with `git add -f` since `models/**/*.pt` is gitignored by
pattern).

## 2. Game module — only if architecture differs

If the new champion uses the same architecture as `champAa` (the
`CNN_uncoupled` family), set `game: quarto` and you're done — reuse the
existing `scripts/games/quarto.py`. **No new module needed.**

If the architecture differs (new hook names, new forward signature, new
hook namespacing), add `scripts/games/quarto_<arch>.py` that exposes:

```python
def generate_positions(num_games, seed) -> (boards, pieces, metadata): ...
def load_model(model_path, device) -> nn.Module: ...
# Plus the BSP set registry used by compute_bsp_labels.py --name:
BSP_SETS = {"gorilla": [...categories...], "hawk": [...]}
get_all_bsp_definitions()
compute_bsp_vector(metadata, bsp_ids)
```

Then register it in `scripts/games/__init__.py` (add to `AVAILABLE_GAMES`
plus an `elif` in `get_game_module`). Hook names should be namespaced —
champS4's are `s4.conv2`, `s4.fc1`, etc. — so they don't collide with
sibling modules in the same workspace.

## 3. Naming tags — propagate `<X>` everywhere downstream

| Artifact | Pattern | Example (`<X>=S4`) |
|---|---|---|
| Positions | `positions-amalgam_<x>_unique.pt` (lowercased tag, underscore) | `positions-amalgam_s4_unique.pt` |
| Activations | `<hook>_amalgam_<x>{,_random}_activations.pt` | `s4.conv2_amalgam_s4_activations.pt`, `s4.conv2_amalgam_s4_random_activations.pt` |
| BSP labels | `bsp_labels-<animal><X>_<count>.pt` (CamelCase tag, no separator) | `bsp_labels-gorillaS4_164.pt` |
| BSP schema | `bsp_schema-<animal><X>_<count>.json` | `bsp_schema-gorillaS4_164.json` |
| SAE checkpoint | trainer-generated; `experiment:` must contain `champ<X>` | `C01-champS4-s42-topk-k16-exp8-s4.conv2.pt` |
| Eval registry key | `run_id:<bsp_set>` (no champion glue needed) | `C01-champS4-s42-...:gorillaS4` |

Lower-case-vs-CamelCase tagging is deliberate: the underscore-lowercase
form keeps the metals theme readable in glob patterns, while the
CamelCase form on BSP labels prevents `bsp_labels-gorilla_[0-9]*.pt`
from matching `bsp_labels-gorillaS4_164.pt`. **Do not invent a new
convention.**

When invoking `compute_bsp_labels.py`, pass `--output` and `--schema-out`
explicitly to get the `<X>` tag into the filename — the script does not
auto-tag (yet).

## 4. SAE configs — `configs/champ<X>/*.yaml`

One YAML per (architecture × sparsity × hook) condition. Required
fields specific to a non-baseline champion:

- `experiment:` — must contain `champ<X>` (e.g. `C01-champS4-s42`).
  The trainer appends `-{arch}-{sparsity}-exp{E}-{hook}` to form the
  run_id; keep `experiment:` short.
- `data:` — point at the champion-specific activation file (e.g.
  `data/quarto/s4.conv2_amalgam_s4_activations.pt`). Stored in
  checkpoint metadata so `sae_eval` auto-resolves at eval time.
- `game: quarto` — write `quarto` even for `champS4`-style sub-games;
  this keeps every champion's registry entries in the same
  `saes/quarto/eval_registry.json` and the `run_id:bsp_set` key
  prevents collisions.

Expansion sizing reminder: SAEs match either *dictionary size* or
*per-input-dim feature budget*, not both, when `d_act` changes between
champions. champS4 fc1 hit 96.6% dead features at `expansion=2` because
it matched dictionary size but not per-dim budget. When in doubt,
match per-dim budget (i.e. keep `expansion` constant, accept a larger
dictionary).

## 5. Pipeline — stage in `commands.sh`

`commands.sh` at the repo root is the user's transient per-phase
execution recipe (see CLAUDE.md). For a new champion, write blocks for:

1. Competence audit: `scripts/model_competence_audit.py
   --model-config=configs/models/champ<X>.yaml ...` for both the new
   champion and a champAa re-baseline.
2. Position generation (4 opponent modes, then `deduplicate_positions.py`).
3. BSP labels (gorilla<X>, hawk<X>) — remember to set `--output` /
   `--schema-out`.
4. Activations for the trained + random checkpoint at each hook of
   interest.
5. SAE sweep (`run_sweep.py --configs=configs/champ<X> ...`) followed
   by explicit `sae_eval` against the new BSP sets (the sweep's default
   `--bsps=gorilla` will pick the wrong labels otherwise).
6. Linear-probe baseline block on the new activations × BSP sets — see
   the existing block in `commands.sh` for the format.
7. `registry_query.py` reports (top vs `gorilla<X>`, pairwise compare
   vs champAa twin runs).

`commands.sh` is not committed canonical infrastructure; rewrite it as
phases change.

## 6. Verification checklist

Before merging the onboarding PR, confirm:

- [ ] `python -c "from scripts.games import get_game_module;
      get_game_module('<game>')"` succeeds.
- [ ] `python scripts/model_competence_audit.py
      --model-config=configs/models/champ<X>.yaml --num-positions=100`
      runs to completion (sanity check, not a full audit).
- [ ] `python scripts/run_id.py configs/champ<X>/<some>.yaml --checkpoint`
      returns a path that matches the trainer's filename rule.
- [ ] `sae_eval`'s glob `bsp_labels-{animal}_[0-9]*.pt` does not match
      your new champion-tagged files when called with the *other*
      champion's animal name (i.e. `--bsps=gorilla` shouldn't pick up
      `bsp_labels-gorilla<X>_164.pt`).
- [ ] The new champion's YAML, game module (if added), and `configs/champ<X>/`
      are committed; the `models/quarto/*.pt` files are force-added.
- [ ] `RESEARCH-STATUS.md` has a one-paragraph entry describing the
      new champion and the active phase. `Quarto-specifications.md`'s
      Dataset Catalog and Hookable-layers sections are updated.
