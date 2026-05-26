# champVe follow-up sweep configs

champVe = Ve_oracleAblation(4) [DISABLE_NEVER, E=10000]. Same architecture
class as champS4 / champTa (QuartoCNNAutoregUnifiedS4); the difference from
champTa is the training procedure — minimax-oracle SELECT distillation
*never disabled* and trained 2.3x longer (10000 vs 4350 epochs).

Benchmark vs prior champions (hierarchical-SAE/champion-results.jsonl,
2026-05-24, 500 matches each direction):

    vs Random Baseline           94.3%
    vs Loss_BT                   88.3%
    vs Aa_replay(2)              80.9%
    vs ME_endgame(2)             72.9%
    vs Sa_archScan(3) [S4]       73.7%
    vs Ta_minimaxSelect(1) [Ta]  59.4%   <- previous champion

Because the architecture is identical to champS4 and champTa, the comparisons
{champTa vs champVe} and {champS4 vs champVe} are controlled experiments on
the training procedure. All hook names, the game module (quarto_s4), and the
d_act values transfer unchanged. Only the dataset distribution and BSP label
tags change.

Setup
-----
Checkpoints already in `models/quarto/` (force-add since `models/**/*.pt` is
gitignored):

    models/quarto/20260524_1904-Ve_oracleAblation(4)0522_DISABLE_NEVER_10k_E_10000.pt  (trained)
    models/quarto/20260522_1832-Ve_oracleAblation(4)0522_DISABLE_NEVER_10k_E_0000.pt   (random / epoch 0)

To launch the data + LP + sweep pipeline, run the recipe at the repo root:

    bash commands.sh

Sweep design (mirrors champTa non-anchored + anchored set; no seed replicates for C/E/F/H)
----------------------------------------------------------------------------------------
43 SAE configs total: 19 non-anchored (s42 only) + 24 anchored (I01-I04 anchored-jumprelu
+ J01-J04 anchored-batchtopk, 3 seeds each: s42/s43/s44). The IDs mirror the corresponding
champTa configs so `registry_query.py compare A B --bsps=gorillaVe --bsps-b=gorillaTa`
works for every pair.

  ID                          hook       arch                k / t   exp
  C01-champVe-s42             s4.conv2   topk                k=16    8    (collapse-zone probe)
  E01-champVe-s42             s4.conv2   topk                k=32    8
  E02-champVe-s42             s4.conv2   topk                k=48    8
  E03-champVe-s42             s4.conv2   topk                k=64    8
  E04-champVe-s42             s4.conv2   topk                k=96    8
  E05-champVe-s42             s4.conv2   batchtopk           k=32    8
  E06-champVe-s42             s4.conv2   batchtopk           k=64    8
  E07-champVe-s42             s4.conv2   jumprelu            t=32    8    (patience fix)
  F00-champVe-s42             s4.fc1     batchtopk           k=16    8
  F01-champVe-s42             s4.fc1     topk                k=32    8
  F02-champVe-s42             s4.fc1     topk                k=64    8
  F03-champVe-s42             s4.fc1     batchtopk           k=32    8
  F04-champVe-s42             s4.fc1     jumprelu            t=64    8    (patience fix)
  H01-champVe-s42             s4.fc1     jumprelu            t=64    16
  H02-champVe-s42             s4.fc1     jumprelu            t=64    32
  H03-champVe-s42             s4.fc1     jumprelu            t=64    64
  H04-champVe-s42             s4.conv2   batchtopk           k=32    16
  H05-champVe-s42             s4.conv2   batchtopk           k=32    32
  H06-champVe-s42             s4.conv2   batchtopk           k=32    64
  I01-champVe-lh003-{s42-44}  s4.fc1     anchored-jumprelu   t=64    8    lambda_high=0.03
  I02-champVe-lh010-{s42-44}  s4.fc1     anchored-jumprelu   t=64    8    lambda_high=0.10
  I03-champVe-lh030-{s42-44}  s4.fc1     anchored-jumprelu   t=64    8    lambda_high=0.30
  I04-champVe-lh100-{s42-44}  s4.fc1     anchored-jumprelu   t=64    8    lambda_high=1.00
  J01-champVe-lh003-{s42-44}  s4.fc1     anchored-batchtopk  k=16    8    lambda_high=0.03
  J02-champVe-lh010-{s42-44}  s4.fc1     anchored-batchtopk  k=16    8    lambda_high=0.10
  J03-champVe-lh030-{s42-44}  s4.fc1     anchored-batchtopk  k=16    8    lambda_high=0.30
  J04-champVe-lh100-{s42-44}  s4.fc1     anchored-batchtopk  k=16    8    lambda_high=1.00

Anchored I/J configs reference `data/quarto/bsp_labels-tigerVe_36.pt` (champion-tagged)
and `data/quarto/bsp_schema-tiger_36.json` (basis-only, shared across champions per the
gorilla/hawk/tiger convention).

All YAMLs set `game: quarto` (not `quarto_s4`) so registries stay unified.
Eval against `gorillaVe` / `hawkVe` for C/E/F/H, and `tigerVe` (primary) plus
`gorillaVe` / `hawkVe` (regression check) for I/J. The `run_id:bsp_set` key
keeps champion lineages from colliding.

Naming & file conventions: see `configs/models/README.md`.
