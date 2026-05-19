# champTa follow-up sweep configs
#
# champTa = Ta_minimaxSelect(1) [depth=2, E=4350]. Same architecture class as
# champS4 (QuartoCNNAutoregUnifiedS4); the difference is the training objective
# — minimax depth-2 selection during self-play instead of uniform sampling.
# Benchmark vs prior champions (hierarchical-SAE/champion-results.jsonl):
#     vs Random          90.8%
#     vs Loss_BT         82.7%
#     vs Aa_replay       73.1%
#     vs ME_endgame      66.2%
#     vs Sa_archScan(3)  66.6%   <- previous champion (champS4)
#
# Because the architecture is identical to champS4, the comparison champS4 vs
# champTa is a controlled experiment on the training procedure. All hook names,
# the game module (quarto_s4), and the d_act values transfer unchanged. Only
# the dataset distribution and BSP label tags change.
#
# Setup
# -----
# Checkpoints are already in models/quarto/ (force-added since
# models/**/*.pt is gitignored):
#
#     models/quarto/20260516_1452-Ta_minimaxSelect(1)0514_DEPTH_2_E_4350.pt  (trained)
#     models/quarto/20260515_1110-Ta_minimaxSelect(1)0514_DEPTH_2_E_0000.pt  (random / epoch 0)
#
# To launch the data + LP + sweep pipeline, run the recipe at the repo root:
#
#     bash commands.sh
#
# Sweep design (no seed replicates this pass — see RESEARCH-STATUS.md)
# --------------------------------------------------------------------
# 13 SAE configs. The E*/F* IDs mirror the corresponding champS4 configs so
# `registry_query.py compare A B --bsps=gorillaTa --bsps-b=gorillaS4` works
# for every pair.
#
#   ID                          hook       arch       k / t   exp
#   C01-champTa-s42             s4.conv2   topk       k=16    8    (collapse-zone probe)
#   E01-champTa-s42             s4.conv2   topk       k=32    8
#   E02-champTa-s42             s4.conv2   topk       k=48    8
#   E03-champTa-s42             s4.conv2   topk       k=64    8
#   E04-champTa-s42             s4.conv2   topk       k=96    8
#   E05-champTa-s42             s4.conv2   batchtopk  k=32    8
#   E06-champTa-s42             s4.conv2   batchtopk  k=64    8
#   E07-champTa-s42             s4.conv2   jumprelu   t=32    8    (patience fix: min_improvement=0.005)
#   F00-champTa-s42             s4.fc1     batchtopk  k=16    8    (= champS4 A01v2 setting)
#   F01-champTa-s42             s4.fc1     topk       k=32    8
#   F02-champTa-s42             s4.fc1     topk       k=64    8
#   F03-champTa-s42             s4.fc1     batchtopk  k=32    8
#   F04-champTa-s42             s4.fc1     jumprelu   t=64    8    (patience fix)
#
# All YAMLs set `game: quarto` (not `quarto_s4`) so registries stay unified.
# Eval against `gorillaTa` / `hawkTa` (distinct from gorillaS4 / hawkS4); the
# `run_id:bsp_set` key keeps champion lineages from colliding.
#
# Naming & file conventions: see configs/models/README.md.
