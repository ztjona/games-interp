# champYb follow-up sweep configs

champYb = Yb_hotChamp(3) [hot lambda=1.0, seedB, E=10000]. Same architecture
class as champS4 / champTa / champVe (QuartoCNNAutoregUnifiedS4); the model
class is the subclass QuartoCNNAutoregUnifiedS4Hot, which adds a
training-time-only auxiliary head (fc_hot, depth-1 hot-piece BCE scaffold).
That head is never read at inference and does not touch any SAE hook point, so
the hookable trunk (s4.conv1, s4.conv2, s4.fc1=512, s4.fc2_place,
s4.fc2_select) and the d_act values are byte-identical to the S4 family.

Difference from champVe: where champVe deepens *state* encoding via minimax
oracle SELECT distillation (DISABLE_NEVER), champYb shapes the shared trunk
with a dense piece-safety ("hot-piece") signal. Because the architecture is
identical, {champYb vs champVe/Ta/S4} are controlled experiments on the
training procedure. Only the dataset distribution and BSP label tags change.

Benchmark vs prior champions (hierarchical-SAE/champion-results.jsonl,
2026-06-16, 500 matches each direction):

    vs Random Baseline           97.5%
    vs Loss_BT                   95.8%
    vs Aa_replay(2)              91.2%
    vs ME_endgame(2)             85.4%
    vs Sa_archScan(3) [S4]       90.0%
    vs Ta_minimaxSelect(1) [Ta]  75.3%
    vs Ve_oracleAblation(4) [Ve] 70.5%   <- previous champion
    vs Ya_hotHead(4) [seedA]     49.1%   (sibling, statistical tie)
    vs MinimaxBot (depth=2)      44.5%   (still below depth-2 minimax)

champYb is the strongest *learned* champion to date.

Setup
-----
Checkpoints already in `models/quarto/` (force-add if `models/**/*.pt` is
gitignored on your branch):

    models/quarto/20260614_2031-Yb_hotChamp(3)0612_HOT_1.0_seedB_E_10000.pt  (trained)
    models/quarto/20260612_1414-Yb_hotChamp(3)0612_HOT_1.0_seedB_E_0000.pt   (random / epoch 0)

Loaded via the `quarto_s4_hot` game module. To launch the full data + LP +
sweep + unify pipeline, run the recipe at the repo root (on Deep Brain):

    bash commands.sh

Sweep design (mirrors champVe exactly so cross-champion compare lines up)
------------------------------------------------------------------------
43 SAE configs: 19 non-anchored (s42 only) + 24 anchored (I01-I04
anchored-jumprelu + J01-J04 anchored-batchtopk, 3 seeds each: s42/s43/s44). The
IDs mirror the corresponding champVe/champTa configs so
`registry_query.py compare A B --bsps=gorillaYb --bsps-b=gorillaVe` works for
every pair.

  ID                          hook       arch                k / t   exp
  C01-champYb-s42             s4.conv2   topk                k=16    8    (collapse-zone probe)
  E01-champYb-s42             s4.conv2   topk                k=32    8
  E02-champYb-s42             s4.conv2   topk                k=48    8
  E03-champYb-s42             s4.conv2   topk                k=64    8
  E04-champYb-s42             s4.conv2   topk                k=96    8
  E05-champYb-s42             s4.conv2   batchtopk           k=32    8
  E06-champYb-s42             s4.conv2   batchtopk           k=64    8
  E07-champYb-s42             s4.conv2   jumprelu            t=32    8
  F00-champYb-s42             s4.fc1     batchtopk           k=16    8
  F01-champYb-s42             s4.fc1     topk                k=32    8
  F02-champYb-s42             s4.fc1     topk                k=64    8
  F03-champYb-s42             s4.fc1     batchtopk           k=32    8
  F04-champYb-s42             s4.fc1     jumprelu            t=64    8
  H01-H03-champYb-s42         s4.fc1     jumprelu            t=64    16/32/64
  H04-H06-champYb-s42         s4.conv2   batchtopk           k=32    16/32/64
  I01-I04-champYb-{s42-44}    s4.fc1     anchored-jumprelu   t=64    8    lambda_high in {0.03,0.10,0.30,1.0}
  J01-J04-champYb-{s42-44}    s4.fc1     anchored-batchtopk  k=16    8    lambda_high in {0.03,0.10,0.30,1.0}

The I/J base architecture (jumprelu-t64 / batchtopk-k16) is inherited from the
champVe recipe (F04 was Ve's fc1 winner); champYb's own fc1 winner is confirmed
from the F-series eval before reading too much into the anchored runs.

Anchored I/J configs reference `data/quarto/bsp_labels-tigerYb_36.pt`
(champion-tagged) and `data/quarto/bsp_schema-tiger_36.json` (basis-only,
shared across champions per the gorilla/hawk/tiger convention).

All YAMLs set `game: quarto` (not `quarto_s4_hot`) so registries stay unified;
the champion is carried by the `data:` path (`_amalgam_yb_`) and the anchor
label tag. Eval against `gorillaYb` / `hawkYb` for C/E/F/H, and `tigerYb`
(primary) plus `gorillaYb` / `hawkYb` (regression check) for I/J. The
`run_id:bsp_set` registry key keeps champion lineages from colliding.

Naming & file conventions: see `configs/models/README.md`.
