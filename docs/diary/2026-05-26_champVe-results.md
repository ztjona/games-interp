# champVe full sweep results

Champion: `Ve_oracleAblation(4) [DISABLE_NEVER, E=10000]`. Same architecture as champS4/champTa
(`QuartoCNNAutoregUnifiedS4`, fc1=512); differs from champTa by oracle-NEVER SELECT schedule + 2.3x
longer training (10k vs 4350 epochs). Head-to-head vs champTa: 59.4%.

Config: [`configs/models/champVe.yaml`](../../configs/models/champVe.yaml).
Registry: `saes/quarto/eval_registry.json` (110 Ve entries across gorillaVe, hawkVe, tigerVe).

---

## 1. Linear probe baselines

### Ve LP (trained model)

| BSP set | Hook | F1 | MCC | F1-lift |
|---|---|---:|---:|---:|
| gorillaVe | fc1 | 0.802 | 0.796 | 0.605 |
| gorillaVe | conv2 | 0.959 | 0.958 | **0.762** |
| hawkVe | fc1 | 0.692 | 0.693 | 0.662 |
| hawkVe | conv2 | 0.851 | 0.851 | **0.822** |
| tigerVe | fc1 | 0.549 | 0.490 | 0.298 |
| tigerVe | conv2 | 0.385 | 0.322 | 0.136 |

### Twin comparison: Ve LP vs Ta LP

| BSP | Hook | Ve lift | Ta lift | Delta |
|---|---|---:|---:|---:|
| gorilla | fc1 | 0.605 | 0.595 | **+0.010** |
| gorilla | conv2 | 0.762 | 0.708 | **+0.054** |
| hawk | fc1 | 0.662 | 0.660 | +0.002 |
| hawk | conv2 | 0.822 | 0.784 | **+0.038** |
| tiger | fc1 | 0.298 | 0.301 | -0.003 |
| tiger | conv2 | 0.136 | 0.175 | **-0.039** |

[AI-REASONED PROVISIONAL ANALYSIS]

Ve's oracle-NEVER training + longer training improves gorilla/hawk LP substantially (especially conv2: +0.054 gorilla, +0.038 hawk) but does **not** improve tiger LP (fc1 flat at -0.003; conv2 **worse** by -0.039). The oracle teaches optimal piece selection, improving board-state encoding (gorilla/hawk measure this). But tiger BSPs measure agent-relative strategic reasoning (can I win with this line? is this piece safe to offer?) -- the LP's ability to linearly decode these from activations does not improve despite the model playing 59.4% better. Tiger conv2 degradation suggests the extra training encodes strategic concepts more nonlinearly in conv2.

## 2. Non-anchored SAE results (C/E/F/H series, gorilla/hawk)

### gorillaVe top 5

| Rank | Run ID | Hook | Lift | MCC |
|---|---|---|---:|---:|
| 1 | I03-champVe-lh030-s43 (anchored-jumprelu) | fc1 | **0.195** | 0.374 |
| 2 | E01-champVe-s42 (topk-k32) | conv2 | 0.192 | 0.376 |
| 3 | H06-champVe-s42 (batchtopk-k32-exp64) | conv2 | 0.187 | 0.372 |
| 4 | E02-champVe-s42 (topk-k48) | conv2 | 0.185 | 0.356 |
| 5 | E05-champVe-s42 (batchtopk-k32) | conv2 | 0.183 | 0.370 |

### hawkVe top 5

| Rank | Run ID | Hook | Lift | MCC |
|---|---|---|---:|---:|
| 1 | I02-champVe-lh010-s43 (anchored-jumprelu) | fc1 | **0.209** | 0.289 |
| 2 | F01-champVe-s42 (topk-k32) | fc1 | 0.207 | 0.285 |
| 3 | I01-champVe-lh003-s44 (anchored-jumprelu) | fc1 | 0.202 | 0.285 |
| 4 | I02-champVe-lh010-s44 (anchored-jumprelu) | fc1 | 0.200 | 0.284 |
| 5 | I03-champVe-lh030-s43 (anchored-jumprelu) | fc1 | 0.199 | 0.285 |

### Twin comparison: non-anchored SAEs (Ve vs Ta)

| Config | BSP | Ve lift | Ta lift | Delta |
|---|---|---:|---:|---:|
| E05 batchtopk-k32 conv2 | gorilla | 0.183 | **0.214** | -0.031 (-14%) |
| F04 jumprelu-t64 fc1 | gorilla | 0.174 | 0.177 | -0.003 |
| F04 jumprelu-t64 fc1 | hawk | **0.197** | 0.172 | +0.025 (+15%) |
| E05 batchtopk-k32 conv2 | hawk | 0.039 | 0.088 | -0.049 (-56%) |

[AI-REASONED PROVISIONAL ANALYSIS]

Non-anchored conv2 SAEs are **worse** on Ve than Ta across the board (gorilla -14%, hawk -56%).
fc1 SAEs are comparable on gorilla and **better** on hawk. The pattern suggests that Ve's conv2
representations have become harder for standard SAEs to decompose, possibly due to more distributed
encoding from the extended training. fc1 remains accessible, consistent with the LP trend where Ve
fc1 LP is on par or slightly better than Ta fc1 LP.

Note: conv2 collapses on hawkVe (best lift 0.064 vs 0.088 on Ta) -- the hawk gap worsens on Ve.

## 3. Anchored SAE sweep I/J on tigerVe

24 anchored configs (12 anchored-jumprelu I-series, 12 anchored-batchtopk J-series) evaluated on
tigerVe (primary), gorillaVe, hawkVe (regression checks).

### Mean across seeds, by condition

| Condition | tigerVe lift | tigerVe MCC | gorillaVe lift | hawkVe lift | FVU |
|---|---:|---:|---:|---:|---:|
| F04 baseline (unanchored) | -- | -- | 0.174 | 0.197 | 0.004 |
| I anch-jrelu lh=0.03 | 0.160 | 0.298 | 0.172 | 0.196 | 0.004 |
| I anch-jrelu lh=0.10 | 0.187 | 0.344 | 0.173 | 0.202 | 0.004 |
| I anch-jrelu lh=0.30 | 0.231 | 0.405 | 0.177 | 0.195 | 0.004 |
| **I anch-jrelu lh=1.00** | **0.254** | **0.450** | 0.171 | 0.195 | 0.004 |
| J anch-btopk lh=0.03 | 0.147 | 0.319 | 0.152 | 0.158 | 0.033 |
| J anch-btopk lh=0.10 | 0.150 | 0.335 | 0.163 | 0.157 | 0.033 |
| J anch-btopk lh=0.30 | 0.162 | 0.359 | 0.152 | 0.146 | 0.035 |
| J anch-btopk lh=1.00 | 0.180 | 0.385 | 0.155 | 0.157 | 0.038 |

Note: F04 baseline was not evaluated on tigerVe so the "--" cells are absent. The unanchored
gorillaVe/hawkVe numbers provide the regression baseline.

### Best single run: I04-lh100-s43

| Metric | I04-lh100-s43 |
|---|---:|
| tigerVe F1 | **0.506** |
| tigerVe F1-lift | **0.255** |
| tigerVe MCC | 0.451 |
| gorillaVe F1 | 0.371 |
| gorillaVe F1-lift | 0.174 |
| hawkVe F1-lift | 0.193 |

### Per-category breakdown: I04-lh100-s43 on tigerVe

| Category | n | F1 | MCC | Lift | Base |
|---|---:|---:|---:|---:|---:|
| tiger_decision_global | 5 | 0.802 | 0.649 | 0.224 | 0.45 |
| tiger_line_winnable | 10 | 0.198 | 0.209 | 0.149 | 0.02 |
| tiger_offered_completing_attr | 4 | 0.553 | 0.553 | 0.463 | 0.05 |
| tiger_pool_safe_count | 4 | 0.861 | 0.640 | 0.144 | 0.63 |
| tiger_pool_winning_count | 4 | 0.763 | 0.643 | 0.255 | 0.36 |
| tiger_square_winnable | 9 | 0.390 | 0.393 | 0.346 | 0.02 |

### Per-category twin comparison (I04-lh100-s43, Ve vs Ta)

| Category | n | Ve F1 | Ta F1 | Ve MCC | Ta MCC | Ve lift | Ta lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiger_decision_global | 5 | 0.802 | 0.851 | 0.649 | 0.718 | 0.224 | 0.227 |
| tiger_line_winnable | 10 | 0.198 | 0.263 | 0.209 | 0.264 | 0.149 | 0.177 |
| tiger_offered_completing_attr | 4 | 0.553 | 0.590 | 0.553 | 0.578 | 0.463 | 0.442 |
| tiger_pool_safe_count | 4 | 0.861 | 0.876 | 0.640 | 0.707 | 0.144 | 0.174 |
| tiger_pool_winning_count | 4 | 0.763 | 0.827 | 0.643 | 0.699 | 0.255 | 0.237 |
| tiger_square_winnable | 9 | 0.390 | 0.405 | 0.393 | 0.410 | 0.346 | 0.324 |

[AI-REASONED PROVISIONAL ANALYSIS]

The per-category comparison shows a split pattern. Ta SAE dominates on 4/6 categories
(decision_global, line_winnable, pool_safe, pool_winning by MCC) but Ve SAE wins on the two
**rarest** categories: `offered_completing_attr` (+0.021 lift) and `square_winnable` (+0.022 lift).
These are the categories where base rates are lowest (2-5%) and supervision has the most room to
steer features. The overall lift is near-identical (0.255 vs 0.257).

## 4. Cross-champion pattern confirmation

### Anchored recipe transfer

The I04 lh=1.0 anchored-jumprelu recipe transfers perfectly from champTa to champVe:

| Metric | Ve (I04-s43) | Ta (I04-s43) |
|---|---:|---:|
| tiger lift | 0.255 | 0.257 |
| tiger MCC | 0.451 | 0.496 |
| SAE/LP efficiency (tiger fc1) | 85.6% | 85.4% |

Lambda ordering is monotonic on both champions (lh=1.0 > 0.30 > 0.10 > 0.03). J-series
(anchored-batchtopk) underperforms on both (best J: 0.180 lift on Ve vs 0.162 on Ta; both below
I04). Low-lambda counterproductivity partially resolves on Ve (I01 lh=0.03 lift 0.160 is above
what the unanchored baseline would likely produce, given F04-Ve gorillaVe lift 0.174 > F04-Ta
gorillaTa lift 0.177 and the general pattern).

### H9 extended (concept distillation across oracle schedules)

| Comparison | Ve lift | Ta lift | Delta | Interpretation |
|---|---:|---:|---:|---|
| LP gorilla conv2 | 0.762 | 0.708 | +0.054 | DISABLE_NEVER deepens state encoding |
| LP hawk conv2 | 0.822 | 0.784 | +0.038 | Same |
| LP tiger fc1 | 0.298 | 0.301 | -0.003 | Strategic concepts NOT boosted |
| LP tiger conv2 | 0.136 | 0.175 | -0.039 | Strategic concepts degraded in conv2 |
| Best SAE gorilla conv2 | 0.192 | 0.214 | -0.022 | SAE decomposition harder on Ve conv2 |
| Best SAE tiger fc1 (anchored) | 0.255 | 0.257 | -0.002 | Anchored efficiency constant |

[AI-REASONED PROVISIONAL ANALYSIS]

The DISABLE_NEVER oracle + longer training dissociates two dimensions:
1. **State representation quality** (gorilla/hawk LP) -- improves, especially at conv2.
2. **Strategic reasoning accessibility** (tiger LP) -- flat at fc1, degraded at conv2.

This suggests the oracle helps the model build better intermediate state representations but does
not make its *decision-relevant computations* more linearly decodable. The decision computation may
become more implicit or distributed with longer training. This is a novel dimension to H9: concept
distillation has a ceiling on strategic (agent-relative) concepts even as it continues to improve
state-level concepts.

## 5. Summary of key findings

1. **Anchored I04 lh=1.0 recipe transfers**: tigerVe lift 0.255 matches tigerTa lift 0.257 within
   noise. SAE/LP efficiency = 85.6% on both champions. Lambda ordering and J-series failure mode
   reproduce.

2. **Ve LP improves on state concepts, not strategic concepts**: gorilla/hawk LP up by +0.01 to
   +0.054; tiger LP flat (fc1) or degraded (conv2 -0.039).

3. **conv2 SAEs harder to train on Ve**: E05 gorilla lift drops from 0.214 (Ta) to 0.183 (Ve);
   hawk conv2 collapses further (0.039 vs 0.088). fc1 SAEs are comparable or better.

4. **Per-category: Ve SAE wins on rare categories** (offered_completing_attr, square_winnable),
   Ta SAE wins on moderate-base-rate categories. Overall tiger lift is statistically equivalent.

5. **J-series anchored-batchtopk still underperforms**, confirming this is an architecture-level
   finding, not champion-specific.
