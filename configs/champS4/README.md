# champS4 follow-up sweep configs
#
# These mirror the top-ranked Aa_replay runs (per `registry_query.py top --bsps=gorilla`)
# against the new S4 champion. The S4 model uses a 32-d aux input internally; SAE
# training is unchanged because the `quarto_s4` game module wraps the model and
# preserves the `(board, piece16)` forward signature used by activation collection.
#
# How to run
# ----------
# Use `commands.sh` at the repo root — it executes the full pipeline end-to-end.
#
# Naming & run-ids
# ----------------
# Filename pattern in this folder (human-readable):
#     {Major}{Minor}-champS4-{hook}-{arch}-{sparsity}-exp{E}-s{seed}.yaml
#
# The *trainer* writes checkpoints as:
#     saes/{game}/{experiment}-{arch}-{sparsity}-exp{E}-{hook}.pt
# where ``{sparsity}`` is:
#     topk, batchtopk         -> k{k}
#     vanilla, gated          -> l1_{l1_weight as digits}
#     jumprelu                -> t{l0_target as int}
#     panneal                 -> (omitted)
# This rule lives in `sae_train.build_filename_suffix()`; it is the single
# source of truth.
#
# Do NOT reconstruct run_ids in shell scripts by hand. Use:
#     python scripts/run_id.py <config.yaml>             # prints run_id stem
#     python scripts/run_id.py <config.yaml> --checkpoint  # full .pt path
#     python scripts/run_id.py <config.yaml> --metrics     # metrics jsonl path
#
# The `experiment:` field is the *only* free string; everything after it is
# derived deterministically from the YAML. Keep `experiment:` short — it is
# the run_id prefix and gets re-used as the eval-registry key.
#
# Top runs being mirrored (gorilla, F1-lift):
#   1. C01-c2arch-s42      conv2 topk-k16-exp8     0.155
#   2. anakin-batchtopk... fc1   batchtopk-k16-exp8 0.141  -> A01-champS4 (exp=2)
#   3. C07-c2arch-s42      conv2 jumprelu-t32-exp8 0.139
#   5. D02-c2exp16-s42     conv2 topk-k64-exp16    0.134
#
# Notes
# -----
# - conv2 d_act is identical for champAa and champS4 (32 channels), so configs
#   transfer directly modulo data path.
# - fc1 d_act differs (128 -> 512); A01-champS4 uses expansion=2 so d_dict=1024
#   matches the champAa A01 feature budget for like-for-like comparison.
# - Positions file is `data/quarto/positions-amalgam_s4_unique.pt` (S4 self-play
#   amalgam), generated separately. BSP labels are `bsp_labels-{gorillaS4,hawkS4}_*.pt`
#   computed against the same positions.
#
# One registry, distinct BSP-set names
# ------------------------------------
# All champion SAEs (champAa, champS4, future) write to a single registry at
# `saes/quarto/eval_registry.json` (the YAMLs set `game: quarto`, NOT
# `quarto_s4`). The S4 distribution is distinguished at the BSP-set level:
#
#   champAa runs  -> evaluated against `gorilla` / `hawk` (positions-amalgam_unique.pt)
#   champS4 runs  -> evaluated against `gorillaS4` / `hawkS4` (positions-amalgam_s4_unique.pt)
#
# Registry keys therefore look like `<run_id>:gorilla` and `<run_id>:gorillaS4`.
# Cross-champion compare in one call:
#
#   python scripts/registry_query.py compare \
#       C01-champS4-s42-topk-k16-exp8-s4.conv2 \
#       C01-c2arch-s42-topk-k16-exp8-conv2 \
#       --bsps=gorillaS4 --bsps-b=gorilla
#
# The S4 SAE YAMLs deliberately point `data:` at the S4 activation files; the
# trainer stores that path in checkpoint metadata so `sae_eval` always picks
# the right activations regardless of the `game` field.
