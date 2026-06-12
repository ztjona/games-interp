# Phase 3 — Geometric concept structure (2026-06-09 → )

Status: OPEN. Founding design note (rationale, evidence, H10, gates):
[`2026-06-09_geometric-pivot.md`](2026-06-09_geometric-pivot.md).

Parent: [`../../RESEARCH-STATUS.md`](../../RESEARCH-STATUS.md).
Predecessor: [`phase-2B.md`](phase-2B.md) (closed 2026-06-09, ch. 5 —
includes the executed-in-place "Phase 2C" concept-targeting arc).

## Scope

Phase 2 established *that* unsupervised SAEs hit a wall on relational
concepts and that neither capacity (Sweep H) nor target reframing alone
(tiger) removes it; anchoring buys back targeted slots only. Phase 3
asks **why**, geometrically, and **what architecture follows**:

- **H10**: the wall is geometric — conjunction/threat concepts occupy
  multi-dimensional structure that flat dictionary atoms dilute or tile;
  hierarchy co-linearity drives the measured absorption.

Working steps (gates and full method in the founding note):

| Step | What | Gate | Status |
|---|---|---|---|
| 3A | Dilution diagnostic (Ising communities, restricted-R², irreducibility indices) on existing champVe/champTa SAE caches | G-3A: diluted/tiled vs absent | pending |
| 3B | Ground-truth geometry: measure α/β allocation regime; polytope + hierarchy-orthogonality checks | — (always runs) | pending |
| 3C | Geometry-aware SAE variants: hierarchical anchoring, H-SAE-vs-matryoshka, bilinear slots | G-3C: beat I04 0.255 or ≥50% threat-gap closure, 3 seeds | gated on 3A |
| 3D | Causal subspace patching → move-change rate | — | after 3C |

## Chapters

*(appended as results land; date-stamped)*

## Pointers

- Literature base: 5 papers ingested 2026-06-09 (tags in founding note);
  synthesis at papers-DB `synthesis/2026-06-08_sae-landscape.md`.
- Eval caches for 3A: `saes/quarto/cache/{run_id}_h.pt`,
  `{run_id}_matching-{bsp_set}.pt` (Deep Brain; gitignored locally).
- Cross-check claims: `python scripts/registry_query.py top|category|compare`.
