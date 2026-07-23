# champYb onboarding — hot-piece-shaped S4 champion (2026-06-18)

Status: integration groundwork only. No GPU pipeline run, no SAE training in
this entry. Parent ledger: [`phase-3.md`](phase-3.md). Onboarding checklist
followed: [`../../configs/models/README.md`](../../configs/models/README.md).

This entry records what the new champion is, how it maps onto the existing
S4-family infrastructure, the files added/modified, and how it slots into the
open Phase 3 plan. Numbers from `champion-results.jsonl` are
`[DIRECT — from hierarchical-SAE/champion-results.jsonl]`; design judgements are
`[AI-REASONED PROVISIONAL ANALYSIS]`.

---

## 1. What champYb is [DIRECT]

- **Source:** sibling repo `hierarchical-SAE/`, checkpoint dir
  `CHECKPOINTS/Yb_hotChamp(3)0612_HOT_1.0_seedB/`.
  - trained: `20260614_2031-Yb_hotChamp(3)0612_HOT_1.0_seedB_E_10000.pt`
  - random (E=0): `20260612_1414-Yb_hotChamp(3)0612_HOT_1.0_seedB_E_0000.pt`
- **Model class:** `models.quarto.CNN_autoreg_sa.QuartoCNNAutoregUnifiedS4Hot`
  — a subclass of `QuartoCNNAutoregUnifiedS4` (the champS4/Ta/Ve trunk) with
  one extra head `fc_hot` (Linear 512->16). `fc_hot` is a *training-time-only*
  depth-1 hot-piece BCE scaffold (1 = giving this piece loses to an immediate
  completion); `forward` still returns `(q_place, q_select)` and `fc_hot` is
  **never read at inference** and never touches any SAE hook point.
- **Training procedure:** "hot-piece" select-safety shaping, lambda=1.0, seed B,
  10,000 epochs. The dense BCE signal forces the shared trunk to encode
  piece-safety, which the unchanged `fc2_select` head reads.
- **Strength (2026-06-16, 500 matches/direction):** beats every prior champion
  head-to-head — vs Ve 70.5%, vs Ta 75.3%, vs S4 90.0%, vs Aa 91.2%, vs random
  97.5%. Statistical tie with its seed-A sibling `Ya_hotHead` (49.1%). Still
  below depth-2 `MinimaxBot` (44.5%) — i.e. strongest *learned* champion to
  date but not superhuman vs search.

## 2. Architecture mapping — clean reuse of the S4 family [DIRECT]

The Yb checkpoint state_dict has 14 tensors; the symmetric difference against a
freshly-instantiated `QuartoCNNAutoregUnifiedS4Hot` is empty. The trunk is
byte-identical to S4:

| layer | shape | hook name (via `S4Wrapper`) |
|---|---|---|
| `conv1` | (16,18,3,3) | `s4.conv1` |
| `conv2` | (32,16,3,3) | `s4.conv2` (flatten -> 512) |
| `fc1` | (512,512) | `s4.fc1` (d_act=512) |
| `fc2_place` | (16,512) | `s4.fc2_place` |
| `fc2_select` | (16,512) | `s4.fc2_select` |
| `fc_hot` | (16,512) | (not hooked; inference-unused) |
| `fc_in_aux` | (32,32) | (32-d aux: offered16 + available16) |

End-to-end load + forward through the new game module verified: model loads as
`S4Wrapper(QuartoCNN_autoreg_unified_S4_hot)`, all five `s4.*` hooks resolve,
forward returns `(B,16),(B,16)`, `s4.fc1` activation is `(B,512)`. The 32-d aux
contract (`S4Wrapper._build_aux32`) and `mode_2x2=True` position generation are
inherited unchanged from `quarto_s4`, so positions / activations / BSP labels
remain interoperable with `collect_activations.py`, `compute_bsp_labels.py`,
`sae_eval.py`.

## 3. Why a new game module (and not a shared loader) [AI-REASONED PROVISIONAL ANALYSIS]

`NN_abstract.from_file` is `cls()` + `load_state_dict` — class-driven, strict on
keys. The Hot checkpoint carries `fc_hot.{weight,bias}`, so it must be loaded by
the Hot subclass; loading into the plain S4 class would raise on unexpected
keys. The cleanest, lowest-risk option (per the onboarding README's "only if
architecture differs" guidance, where hook names / forward / namespacing are
*unchanged* here) is a thin module `quarto_s4_hot` that **re-exports everything
from `quarto_s4`** and overrides only `load_model` to instantiate the Hot
subclass. This keeps `quarto_s4`'s loader strict and a single source of truth
for the trunk logic.

## 4. Naming choices [AI-REASONED PROVISIONAL ANALYSIS — pending PI confirmation]

Following the existing S4-family convention (Tag derived from the source
experiment family `Yb_hotChamp`):

- **Champion Tag:** `Yb` — config `configs/models/champYb.yaml`; SAE
  `experiment:` fields will carry `champYb`.
- **Position suffix (lowercase):** `yb` — `positions-amalgam_yb_unique.pt`,
  `<hook>_amalgam_yb{,_random}_activations.pt`.
- **BSP animal suffix (CamelCase):** `Yb` — `bsp_labels-gorillaYb_164.pt`,
  `hawkYb_173`, `tigerYb_36`. The basis schema stays basis-only
  (`bsp_schema-gorilla_164.json` etc.), as for every other champion.

This reuses the metals/animals theme exactly as `S4`/`Ta`/`Ve` did. No new
metal or animal *basis* is introduced — only a champion suffix — so there is
nothing thematically novel to name. (The metals theme is per opponent-mode
mixture, which is unchanged; champion identity is carried by the suffix, per
[`../../Quarto-specifications.md`](../../Quarto-specifications.md) Naming
Conventions.) **PI: confirm `Yb` / `yb` / `Yb` before the pipeline runs.**

## 5. Files added / modified

Added:
- `configs/models/champYb.yaml` — champion registration (paths, benchmarks, hooks).
- `scripts/games/quarto_s4_hot.py` — thin game module over `quarto_s4`.
- `models/quarto/20260614_2031-Yb_hotChamp(3)..._E_10000.pt` and
  `..._E_0000.pt` — copied from `hierarchical-SAE/CHECKPOINTS/` (gitignored by
  pattern; force-add).

Modified:
- `models/quarto/CNN_autoreg_sa.py` — appended `QuartoCNNAutoregUnifiedS4Hot`
  (ASCII-only addition; pre-existing unicode in sibling docstrings left alone).
- `scripts/games/__init__.py` — registered `quarto_s4_hot`.
- `commands.sh` — rewritten as the champYb onboarding pipeline (transient).

Tests: full suite **116 passed, 1 skipped**. Game-module import, end-to-end
model load + forward, and state_dict key parity all verified.

## 6. How champYb slots into Phase 3 [AI-REASONED PROVISIONAL ANALYSIS]

Phase 3 (H10: the SAE/LP wall is geometric — see
[`2026-06-09_geometric-pivot.md`](2026-06-09_geometric-pivot.md)) is currently
measured on champVe/champTa. champYb adds a fourth point on the
training-procedure axis that is *orthogonal* to the existing one: where Ve
deepens state encoding via oracle distillation (H9 ceiling: helps gorilla, not
tiger), Yb shapes the *trunk* with a dense piece-safety (hot-piece) signal that
is conceptually close to the tiger "poison/safe pool" concepts. The interesting
Phase-3 question this enables: does a trunk explicitly trained to encode
piece-safety make those agent-relative (tiger) concepts *geometrically simpler*
— i.e. lower-dimensional / less diluted — than in Ve, raising both the tiger LP
ceiling and the SAE/LP efficiency? This is a hypothesis to test in 3A/3B once
the data pipeline reaches parity, not a result.

## 7. Remaining work (must run on Deep Brain)

The data-pipeline-to-parity is staged in `commands.sh` (steps 0-5). GPU-heavy
steps (1 positions, 4 activations) should run on Deep Brain (3x A6000,
`training/arnold`). After parity, SAE work joins the Phase-3 track (configs
`configs/champYb/` to be authored as part of 3A-3D). The competence-audit
`--help` crashes under cp1252 on a pre-existing unicode minus in its docstring
(not Yb-specific; the audit run itself is unaffected) — flagged for a separate
cleanup.
