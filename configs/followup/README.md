# Follow-up Config Naming

From 2026-04-24 onward, follow-up SAE runs use prefixed experiment IDs:

- Format: `{Major}{Minor}-{tag}-s{seed}`
- `Major`: campaign family (`A`, `B`, `C`, ...)
- `Minor`: zero-padded condition index inside that campaign (`01`, `02`, ...)
- `tag`: short semantic label (`random-control`, `conv2-completion`, `seedpanel`, ...)
- `s{seed}`: explicit seed suffix used in the checkpoint stem

The trainer appends architecture and hook automatically, so the `experiment:` field should stay short.

Current mapping:

| ID | Purpose | Config |
|---|---|---|
| `A01` | fc1 random-model control | `A01-fc1-batchtopk-k16-exp8-random-control-s42.yaml` |
| `A02` | conv2 random-model control | `A02-conv2-topk-k32-exp8-random-control-s42.yaml` |
| `B01` | conv2 completion: BatchTopK exp4 | `B01-conv2-batchtopk-k64-exp4-completion-s42.yaml` |
| `B02` | conv2 completion: p-Annealing exp4 | `B02-conv2-panneal-exp4-completion-s42.yaml` |
| `B03` | conv2 completion: TopK exp2 | `B03-conv2-topk-k64-exp2-completion-s42.yaml` |
| `B04` | conv2 completion: Vanilla exp4 | `B04-conv2-vanilla-l1_001-exp4-completion-s42.yaml` |

## Campaign C — conv2-512 full architecture sweep at exp8 (2026-04-29)

All 6 architectures on conv2-512 at exp8, seed=42. Fills the gap left by Anakin
which only tested topk/jumprelu/gated on conv2, and only at exp4 except for two topk runs.

| ID | Arch | Key param | Expected run_id |
|---|---|---|---|
| `C01` | topk | k=16, exp8 | `C01-c2arch-s42-topk-k16-exp8-conv2` |
| `C02` | topk | k=128, exp8 | `C02-c2arch-s42-topk-k128-exp8-conv2` |
| `C03` | batchtopk | k=16, exp8 | `C03-c2arch-s42-batchtopk-k16-exp8-conv2` |
| `C04` | batchtopk | k=32, exp8 | `C04-c2arch-s42-batchtopk-k32-exp8-conv2` |
| `C05` | batchtopk | k=64, exp8 | `C05-c2arch-s42-batchtopk-k64-exp8-conv2` |
| `C06` | batchtopk | k=128, exp8 | `C06-c2arch-s42-batchtopk-k128-exp8-conv2` |
| `C07` | jumprelu | l0=32, exp8 | `C07-c2arch-s42-jumprelu-t32-exp8-conv2` |
| `C08` | jumprelu | l0=64, exp8 | `C08-c2arch-s42-jumprelu-t64-exp8-conv2` |
| `C09` | jumprelu | l0=128, exp8 | `C09-c2arch-s42-jumprelu-t128-exp8-conv2` |
| `C10` | gated | l1=5e-4, exp8 | `C10-c2arch-s42-gated-l1_0005-exp8-conv2` |
| `C11` | gated | l1=2e-3, exp8 | `C11-c2arch-s42-gated-l1_002-exp8-conv2` |
| `C12` | vanilla | l1=1e-4, exp8 | `C12-c2arch-s42-vanilla-l1_0001-exp8-conv2` |
| `C13` | vanilla | l1=1e-3, exp8 | `C13-c2arch-s42-vanilla-l1_001-exp8-conv2` |
| `C14` | p-annealing | exp8 | `C14-c2arch-s42-p-annealing-exp8-conv2` |

## Campaign D — conv2-512 expansion factor exp16 (2026-04-29)

Tests whether 512d input benefits from larger dictionaries (exp8→exp16, d_dict=8192).

| ID | Arch | Key param | Expected run_id |
|---|---|---|---|
| `D01` | topk | k=32, exp16 | `D01-c2exp16-s42-topk-k32-exp16-conv2` |
| `D02` | topk | k=64, exp16 | `D02-c2exp16-s42-topk-k64-exp16-conv2` |
| `D03` | batchtopk | k=32, exp16 | `D03-c2exp16-s42-batchtopk-k32-exp16-conv2` |
| `D04` | batchtopk | k=64, exp16 | `D04-c2exp16-s42-batchtopk-k64-exp16-conv2` |

## Campaign F — seed stability for conv2 (2026-04-29)

Seeds 43+44 for the 5 most promising conv2 architectures.
Provides σ estimate at conv2 (currently unknown) and identifies stable winners.

| ID | Arch | k | Seed | Expected run_id |
|---|---|---|---|---|
| `F01` | topk | k=32 | 43 | `F01-c2seeds-s43-topk-k32-exp8-conv2` |
| `F02` | topk | k=32 | 44 | `F02-c2seeds-s44-topk-k32-exp8-conv2` |
| `F03` | topk | k=64 | 43 | `F03-c2seeds-s43-topk-k64-exp8-conv2` |
| `F04` | topk | k=64 | 44 | `F04-c2seeds-s44-topk-k64-exp8-conv2` |
| `F05` | batchtopk | k=16 | 43 | `F05-c2seeds-s43-batchtopk-k16-exp8-conv2` |
| `F06` | batchtopk | k=16 | 44 | `F06-c2seeds-s44-batchtopk-k16-exp8-conv2` |
| `F07` | batchtopk | k=32 | 43 | `F07-c2seeds-s43-batchtopk-k32-exp8-conv2` |
| `F08` | batchtopk | k=32 | 44 | `F08-c2seeds-s44-batchtopk-k32-exp8-conv2` |
| `F09` | batchtopk | k=64 | 43 | `F09-c2seeds-s43-batchtopk-k64-exp8-conv2` |
| `F10` | batchtopk | k=64 | 44 | `F10-c2seeds-s44-batchtopk-k64-exp8-conv2` |

## Campaign G — random-model controls for conv2-512 (2026-04-29)

Controls for the architectures most likely to surface in Phase 2A analysis.
Uses `conv2_512_amalgam_random_activations.pt` (trained model epoch 0).

| ID | Arch | k | Controls for | Expected run_id |
|---|---|---|---|---|
| `G01` | topk | k=16, exp8 | C01 | `G01-c2random-s42-topk-k16-exp8-conv2` |
| `G02` | batchtopk | k=16, exp8 | C03 | `G02-c2random-s42-batchtopk-k16-exp8-conv2` |
| `G03` | batchtopk | k=32, exp8 | C04 | `G03-c2random-s42-batchtopk-k32-exp8-conv2` |
| `G04` | topk | k=64, exp16 | D02 | `G04-c2random-s42-topk-k64-exp16-conv2` |
| `G05` | batchtopk | k=64, exp16 | D04 | `G05-c2random-s42-batchtopk-k64-exp16-conv2` |
