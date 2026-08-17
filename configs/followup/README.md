# Follow-up Config Naming

From 2026-04-24 onward, follow-up SAE runs use prefixed experiment IDs:

**Three naming levels, deliberately separate** (settled 2026-08-15, because
"3B.0A" welded a phase number to a sub-item letter and collided with this one):

| level | what it names | examples |
|---|---|---|
| **phase** | a research phase | `3A`, `3B`, `3C`, `3D` |
| **track** | instrument work that must precede a phase | `instrument/metrics`, `instrument/conformance` |
| **campaign** | the `{Major}` letter of a run-id — registered *in this file* | `A`…`K` |

A campaign that is not listed below does not exist. Campaign `K` was minted on
2026-08-15 and initially recorded only in a diary, which is the same
"convention in one place, deviation in another" failure the conformance track
was about.

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

## Campaign K — reference conformance A/B (2026-08-15)

Track: **`instrument/conformance`**. Full record and results:
[`../../docs/diary/2026-08-15_dead-feature-revival.md`](../../docs/diary/2026-08-15_dead-feature-revival.md).

Why it exists: dead-feature revival (Gao et al. 2024 auxiliary loss) was
implemented **only on `TopKSAE`**, and `W_enc = W_dec^T` init — Gao's *first*
listed mitigation — was absent from every architecture. So an architecture
comparison was partly a comparison of training machinery. Campaign K measures
what changes when each architecture is trained in its published form.

Configs live in `configs/champYb/` (not here) because every member is a champYb
A/B partner of an existing champYb run. All are `exp8`, `lr 3e-4`,
`25 000 batches`, `batch 4096`, `dead_window 0`.

| ID | Arch / hook | Change vs partner | Partner | Seeds |
|---|---|---|---|---|
| `K01` | jumprelu / `s4.fc1` | + aux loss (**non-canonical** probe) | `F04-champYb` | 42 |
| `K02` | batchtopk / `s4.conv2` | + aux loss only | `E05-champYb` | 42 |
| `K03` | topk / `s4.fc1` | + tied init (→ fully canonical) | `F01-champYb` | 42, 43, 44 |
| `K04` | batchtopk / `s4.conv2` | + aux + tied init (→ fully canonical) | `E05-champYb` | 42, 43, 44 |
| `K05` | jumprelu / `s4.fc1` | + tied init (a **choice**, see below) | `F04-champYb` | 42, 43, 44 |
| `K06` | topk / `s4.conv2` | + tied init (→ fully canonical) | `E01-champYb` | 42, 43, 44 |
| `K07` | topk / `s4.conv2` k=16 | + tied init (→ fully canonical) | `C01-champYb` | 42 |

Seeds 43/44 were run only for the four conditions carrying a load-bearing
comparison (the conv2 rank reversal `K04` vs `K06`, and the fc1 ranking `K03`
vs `K05`), per the rule that seed replication is required where a margin is thin
or a claim inverts a previous result — not for every cell.

`K05` is not a conformance fix: Rajamanoharan et al. 2024b specify **no** init
for JumpReLU, so that axis is underdetermined and `decoder_transpose` is a
recorded choice made on K05's evidence. Its auxiliary loss stays **off**, which
*is* specified. See `DEFAULT_INIT` in `sae_train.py` — deliberately not named
`CANONICAL_INIT`, because only the TopK-family rows are canonical.
