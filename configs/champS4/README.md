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
#
# Results — first pass (2026-05-18)
# ---------------------------------
# Ranked by F1-lift on gorillaS4 (only the 4 mirrored configs were trained):
#
#   run                                          F1-lift  cov    MCC    FVU    L0    dead%
#   D02 topk-k64-exp16  s4.conv2                 0.128    0.327  0.282  0.012  64    93.1
#   A01 batchtopk-k16-exp2  s4.fc1               0.125    0.324  0.292  0.058  16    96.6
#   C07 jumprelu-t32-exp8  s4.conv2              0.106    0.305  0.247  0.075  45    95.3
#   C01 topk-k16-exp8  s4.conv2                  0.039    0.225  0.226  0.061  16    58.0
#
# All four trail their champAa twins on F1-lift (-0.006 to -0.116). C01 in
# particular collapsed: cov_above_50 falls from 0.40 (Aa) to 0.04 (S4) with
# identical hyperparameters. hawkS4 is uniformly weaker (0.058-0.074 vs Aa
# peak 0.105).
#
# Competence audit gave the opposite signal: champS4 plays *better* (test B
# losing-piece avoidance 0.62 vs 0.40, test A 0.79 vs 0.70). The model is
# stronger but its activations are harder to interpret with the recipes that
# won on Aa.
#
# Open question and next steps
# ----------------------------
# Before re-launching a sweep, run two cheap diagnostics:
#
#   1. Linear-probe baseline on champS4 activations. Invoked from the LP
#      block at the bottom of the user's `commands.sh` (transient working
#      file — see CLAUDE.md). It runs 8 LP configs ({s4.fc1, s4.conv2} x
#      {trained, random} x {gorillaS4, hawkS4}) and reports F1, MCC, F1-lift
#      in the same schema as `sae_eval` (LP script was extended 2026-05-18).
#      If LP coverage on S4 falls in line with LP on Aa, the gap is the SAE
#      recipes; if LP itself drops, no SAE recipe will close it.
#
#   2. Fix C07's jumprelu early stop. Its *_metrics.jsonl has 7 rows vs the
#      other three configs' 51 rows; lower `min_improvement` or extend
#      `patience` for jumprelu only.
#
# Targeted retraining (after the LP results):
#
#   3. Rerun A01 at expansion=8 (d_dict=4096) instead of expansion=2. The
#      current A01-S4 matches dictionary *size* (1024) but not feature *budget
#      per activation dimension* — fc1 widened from 128 to 512, so exp=2 gives
#      4x less capacity per dim than the champAa A01. Consistent with the
#      observed 96.6% dead-features rate.
#
#   4. Only if (1)-(3) leave a residual gap, run a scoped conv2-S4 sweep:
#      topk-k in {16, 24, 32} at exp=8, plus batchtopk variants at the same
#      k. fc1-S4 stays deprioritized beyond step 3 (per the fc1-only-for-
#      bottleneck-comparison rule).
#
# LP results (2026-05-19) — diagnosis flipped
# -------------------------------------------
# The LP baseline came back FAR higher than expected. champS4 activations are
# more linearly separable than champAa's at every hook+BSP, by a wide margin:
#
#   hook x BSP set     LP F1   LP F1-lift   random F1-lift
#   fc1 x gorillaS4    0.719   0.520        0.105
#   fc1 x hawkS4       0.584   0.548        0.001
#   conv2 x gorillaS4  0.877   0.678        0.206
#   conv2 x hawkS4     0.750   0.715        0.053
#
# So the four-run sweep underperformed not because the model is opaque, but
# because the same SAE budgets recover ~3x less of the available signal
# (SAE/LP efficiency dropped from ~70% on Aa to ~20% on S4). Two policy
# updates land here:
#
#   - fc1 deprioritization is REVERSED for champS4 (hawk fc1 LP lift 0.548
#     vs ~0.005 for Aa). Threats are preserved through the bottleneck.
#   - The Phase 4 architectural-fix proposal (auxiliary threat head) is no
#     longer needed; champS4's unified-aux training already does that work.
#
# The revised sweep is documented in RESEARCH-STATUS.md "Phase 2B LP results +
# revised sweep" — 11 new configs spanning targeted A01/C07 fixes, a conv2 k
# sweep (E01-E05), and fc1 reactivation (F01-F04). Target: SAE F1-lift >= 50%
# of LP F1-lift on the same hook x BSP.
