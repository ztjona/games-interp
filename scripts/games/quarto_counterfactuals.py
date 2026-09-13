"""Counterfactual pairs for Phase 3B-causal, Wave 1: the offered-piece swap.

The Quarto-specific half of 3B-causal; the game-agnostic half is
``lib/sae/interchange.py``. Design: ``docs/diary/2026-09-12_3B-causal-
preregistration.md`` S5.2-5.3, as amended pre-data by amendments 1 and 2.

A pair is (base, source) = ((B, p), (B, p')): the SAME board, a different piece
in hand, p' drawn from the pieces still available at the base. The board is
untouched, so the legal set (the empty cells) is identical in base and source
BY CONSTRUCTION -- which keeps champYb's untrained illegal Q-values out of every
score (pre-registration S2).

Concepts are recomputed here in vectorised form for every (board, piece)
combination: the project's per-position BSP code is far too slow for millions
of counterfactuals. :func:`check_against_labels` is the guard -- on the natural
positions the vectorised values must equal the stored BSP labels EXACTLY, or
nothing downstream runs. Groups and poles are read from ``scripts/games/quarto``
itself, so this file never restates a line, square or attribute.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from functools import lru_cache

import torch

from scripts.games import quarto as _q

# BSP suffix of each of the 8 attribute poles, in the project's own naming
# (ATTR_POLES): hawk probes the first four, hen the last four.
POLE_SUFFIXES: tuple[str, ...] = tuple(_q.ATTR_POLES)
assert POLE_SUFFIXES == ("tall", "black", "square", "with_hole",
                         "little", "white", "circle", "without_hole"), POLE_SUFFIXES

# quartopy enum name -> BSP suffix
_ENUM_TO_SUFFIX = {"TALL": "tall", "BLACK": "black", "SQUARE": "square",
                   "WITH": "with_hole", "LITTLE": "little", "WHITE": "white",
                   "CIRCLE": "circle", "WITHOUT": "without_hole"}

PAIR_KINDS = ("switch_on", "switch_off", "specificity")


@lru_cache(maxsize=1)
def piece_pole_table() -> torch.Tensor:
    """(16, 8) bool: piece index -> which of the 8 poles it carries."""
    from quartopy import Piece

    t = torch.zeros(16, 8, dtype=torch.bool)
    for i in range(16):
        p = Piece.from_index(i)
        names = {_ENUM_TO_SUFFIX[v.name] for v in (p.size, p.coloration, p.shape, p.hole)}
        for k, s in enumerate(POLE_SUFFIXES):
            t[i, k] = s in names
    assert bool((t[:, :4] ^ t[:, 4:]).all()), "every attribute has exactly one pole"
    return t


@lru_cache(maxsize=1)
def groups() -> tuple[tuple[str, tuple[tuple[int, int], ...]], ...]:
    """The 19 winning groups (10 lines, 9 2x2 squares), from quarto.py."""
    out = [(f"{kind}_{idx}", tuple(coords)) for kind, idx, coords in _q._ALL_LINE_COORDS]
    out += [(f"square_{r}_{c}", tuple(coords)) for r, c, coords in _q._ALL_SQUARE_COORDS]
    return tuple(out)


def group_index(coords) -> int:
    key = tuple(sorted(tuple(x) for x in coords))
    for g, (_, cs) in enumerate(groups()):
        if tuple(sorted(cs)) == key:
            return g
    raise KeyError(f"no winning group with cells {coords}")


# ---------------------------------------------------------------------------
# Per-board tables
# ---------------------------------------------------------------------------


@dataclass
class BoardTables:
    occ: torch.Tensor          # (N, 16) bool, cell occupied
    one_empty: torch.Tensor    # (N, G) bool, group has exactly one empty cell
    empty_cell: torch.Tensor   # (N, G) long, that cell (valid where one_empty)
    threat: torch.Tensor       # (N, G, 8) bool, one empty AND the 3 pieces share pole k
    offered: torch.Tensor      # (N,) long, offered piece (0 where none)
    has_offer: torch.Tensor    # (N,) bool
    avail: torch.Tensor        # (N, 16) bool, pieces that could be offered instead

    @property
    def n(self) -> int:
        return int(self.occ.shape[0])

    def offered_poles(self) -> torch.Tensor:
        return piece_pole_table()[self.offered] & self.has_offer.unsqueeze(-1)


def board_tables(boards: torch.Tensor, pieces: torch.Tensor) -> BoardTables:
    """``boards`` (N, 16, 4, 4) one-hot by piece index; ``pieces`` (N, 16) one-hot."""
    N = boards.shape[0]
    flat = boards.reshape(N, 16, 16) > 0                       # (N, piece, cell)
    occ = flat.any(dim=1)
    pidx = flat.float().argmax(dim=1)                          # (N, cell)
    cell_poles = piece_pole_table()[pidx] & occ.unsqueeze(-1)  # (N, cell, 8)

    cells = torch.tensor([[r * 4 + c for (r, c) in cs] for _, cs in groups()])  # (G, 4)
    g_occ = occ[:, cells]                                      # (N, G, 4)
    one_empty = g_occ.sum(-1) == 3
    slot = (~g_occ).float().argmax(-1)                         # (N, G) position of the empty
    empty_cell = cells.unsqueeze(0).expand(N, -1, -1).gather(2, slot.unsqueeze(-1)).squeeze(-1)
    shared = cell_poles[:, cells].sum(dim=2) == 3              # (N, G, 8)
    threat = one_empty.unsqueeze(-1) & shared

    has_offer = pieces.reshape(N, 16).any(dim=1)
    offered = pieces.reshape(N, 16).float().argmax(dim=1)
    on_board = flat.any(dim=2)                                 # (N, piece)
    avail = ~on_board
    avail[torch.arange(N), offered] &= ~has_offer
    return BoardTables(occ, one_empty, empty_cell, threat, offered, has_offer, avail)


def _scatter_groups(t: BoardTables, per_group: torch.Tensor) -> torch.Tensor:
    """(N, G) bool -> (N, 16) bool: mark the empty cell of every flagged group."""
    out = torch.zeros(t.n, 16, dtype=torch.bool)
    for g in range(per_group.shape[1]):
        rows = torch.nonzero(per_group[:, g], as_tuple=False).flatten()
        out[rows, t.empty_cell[rows, g]] = True
    return out


def win_cells(t: BoardTables, poles: torch.Tensor) -> torch.Tensor:
    """(N, 16) bool: cells where a piece with ``poles`` (N, 8) would win now."""
    return _scatter_groups(t, (t.threat & poles.unsqueeze(1)).any(-1))


def all_source_wins(t: BoardTables) -> torch.Tensor:
    """(N, 16 pieces, 16 cells) bool: where each piece would win on each board."""
    P = piece_pole_table()
    return torch.stack([win_cells(t, P[q].expand(t.n, -1)) for q in range(16)], dim=1)


def threat_cells(t: BoardTables, pole: int | None = None) -> torch.Tensor:
    """(N, 16): cells completing a threat in ``pole`` (any pole if None)."""
    per_group = t.threat[..., pole] if pole is not None else t.threat.any(-1)
    return _scatter_groups(t, per_group)


# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConceptSpec:
    bsp_id: str
    basis: str
    kind: str                # pinned | winnable | offered_attr | win_now
    group: int | None = None
    pole: int | None = None

    @property
    def disjuncts(self) -> int:
        """Poles the concept ORs over, known by construction (3A S5.2)."""
        return {"pinned": 1, "winnable": 8}.get(self.kind, 0)


def parse_concept(bsp_id: str, basis: str) -> ConceptSpec:
    if bsp_id == "tiger_win_now_exists":
        return ConceptSpec(bsp_id, basis, "win_now")
    if bsp_id.startswith("tiger_offered_completes_"):
        suffix = bsp_id[len("tiger_offered_completes_"):]
        return ConceptSpec(bsp_id, basis, "offered_attr", pole=POLE_SUFFIXES.index(suffix))
    if bsp_id.startswith("tiger_line_") and bsp_id.endswith("_winnable"):
        coords = _q._parse_line_coords(bsp_id.split("_")[2:4])
        return ConceptSpec(bsp_id, basis, "winnable", group=group_index(coords))
    if bsp_id.startswith("tiger_square_") and bsp_id.endswith("_winnable"):
        r, c = bsp_id.split("_")[2:4]
        coords = _q._parse_square_coords(["square", r, c])
        return ConceptSpec(bsp_id, basis, "winnable", group=group_index(coords))
    if "_completable_" in bsp_id:
        head, suffix = bsp_id.split("_completable_")
        parts = head.split("_")
        coords = (_q._parse_square_coords(parts) if parts[0] == "square"
                  else _q._parse_line_coords(parts))
        return ConceptSpec(bsp_id, basis, "pinned", group=group_index(coords),
                           pole=POLE_SUFFIXES.index(suffix))
    raise ValueError(f"{bsp_id!r} is not a Wave-1 concept")


def concept_value(t: BoardTables, spec: ConceptSpec, piece_poles: torch.Tensor) -> torch.Tensor:
    """(N,) bool: the concept on each board with a piece carrying ``piece_poles``."""
    if spec.kind == "pinned":
        return t.threat[:, spec.group, spec.pole] & piece_poles[:, spec.pole]
    if spec.kind == "winnable":
        return (t.threat[:, spec.group, :] & piece_poles).any(-1)
    if spec.kind == "offered_attr":
        return piece_poles[:, spec.pole] & t.threat[..., spec.pole].any(-1)
    if spec.kind == "win_now":
        return win_cells(t, piece_poles).any(-1)
    raise ValueError(spec.kind)


def check_against_labels(t: BoardTables, specs: list[ConceptSpec],
                         label_cols: dict[str, torch.Tensor]) -> dict[str, int]:
    """Mismatches between the vectorised concept and the stored BSP label, per
    concept, on the natural positions (the piece actually offered). Every value
    must be 0 before anything downstream runs."""
    natural = t.offered_poles()
    return {s.bsp_id: int((concept_value(t, s, natural) != (label_cols[s.bsp_id] > 0.5)).sum())
            for s in specs}


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------


@dataclass
class PairSet:
    kind: str
    base: torch.Tensor          # (n,) long, row into the position set
    src_piece: torch.Tensor     # (n,) long, piece offered in the source
    target: torch.Tensor        # (n, 16) bool
    legal: torch.Tensor         # (n, 16) bool -- identical for base and source
    expect_base: torch.Tensor   # (n, 16) bool, switch_off: where the base should play
    src_poles: torch.Tensor     # (n, 8) bool, poles through which the source completes
    threat_present: torch.Tensor  # (n,) bool, pinned: the concept's threat is on the board

    @property
    def n(self) -> int:
        return int(self.base.numel())

    def select(self, keep: torch.Tensor) -> "PairSet":
        return PairSet(self.kind, *(x[keep] for x in (
            self.base, self.src_piece, self.target, self.legal, self.expect_base,
            self.src_poles, self.threat_present)))


def _seed(bsp_id: str, kind: str, seed: int) -> int:
    return (zlib.crc32(f"{bsp_id}:{kind}".encode()) + seed) % (2 ** 31)


def cap_pairs(ps: PairSet, cap: int, bsp_id: str, seed: int) -> PairSet:
    """Deterministic uniform subsample to ``cap`` pairs (amendment 2): seeded
    per (concept, pair kind), so a concept's sample never depends on which
    other concepts ran."""
    if ps.n <= cap:
        return ps
    g = torch.Generator().manual_seed(_seed(bsp_id, ps.kind, seed))
    keep = torch.randperm(ps.n, generator=g)[:cap].sort().values
    return ps.select(keep)


def build_pairs(t: BoardTables, spec: ConceptSpec, src_wins: torch.Tensor,
                max_candidates: int = 200_000, seed: int = 0) -> dict[str, PairSet]:
    """All eligible pairs of each kind for one concept (pre-registration S5.3 and
    amendment 2). Switch-off pairs are CANDIDATES: the caller keeps only those
    whose unpatched base decision lands in ``expect_base``. ``max_candidates``
    bounds memory on the very common specificity kind; it is a pre-cap
    subsample, still deterministic."""
    N = t.n
    P = piece_pole_table()
    ar = torch.arange(N)
    legal_all = ~t.occ
    p_poles = t.offered_poles()
    base_wins = win_cells(t, p_poles)
    base_no_win = t.has_offer & ~base_wins.any(1)
    base_any_win = t.has_offer & base_wins.any(1)
    C_base = concept_value(t, spec, p_poles) & t.has_offer

    if spec.kind in ("pinned", "winnable"):
        c = t.empty_cell[:, spec.group]
        cell_c = torch.zeros(N, 16, dtype=torch.bool)
        cell_c[ar, c] = True
        in_group = t.one_empty[:, spec.group]
        base_unique_c = t.has_offer & base_wins[ar, c] & (base_wins.sum(1) == 1)
        tgt = {"switch_on": cell_c, "specificity": cell_c,
               "switch_off": legal_all & ~cell_c}
        expect = cell_c
        spec_ctx = in_group
    elif spec.kind == "offered_attr":
        T = threat_cells(t, spec.pole)
        tgt = {"specificity": T, "switch_off": legal_all & ~base_wins}
        expect = base_wins
        spec_ctx = T.any(1)
    else:  # win_now
        T = threat_cells(t, None)
        tgt = {"specificity": T, "switch_off": legal_all & ~base_wins}
        expect = base_wins
        spec_ctx = T.any(1)

    rows: dict[str, list[tuple[torch.Tensor, int]]] = {k: [] for k in PAIR_KINDS}
    for q in range(16):
        wq = src_wins[:, q]
        q_no_win = ~wq.any(1)
        qp = P[q].expand(N, -1)
        C_src = concept_value(t, spec, qp)
        ok = t.avail[:, q] & t.has_offer & (t.offered != q)
        if spec.kind in ("pinned", "winnable"):
            q_unique_c = wq[ar, c] & (wq.sum(1) == 1)
            on = ok & base_no_win & ~C_base & C_src & q_unique_c
            off = ok & C_base & base_unique_c & ~C_src & q_no_win
        else:
            on = ok & base_no_win & ~C_base & C_src & wq.any(1)
            off = ok & C_base & base_any_win & ~C_src & q_no_win
        sp = ok & spec_ctx & base_no_win & q_no_win
        for kind, m in (("switch_on", on), ("switch_off", off), ("specificity", sp)):
            idx = torch.nonzero(m, as_tuple=False).flatten()
            if idx.numel():
                rows[kind].append((idx, q))

    out: dict[str, PairSet] = {}
    for kind in PAIR_KINDS:
        if not rows[kind]:
            base = torch.zeros(0, dtype=torch.long)
            srcp = torch.zeros(0, dtype=torch.long)
        else:
            base = torch.cat([i for i, _ in rows[kind]])
            srcp = torch.cat([torch.full_like(i, q) for i, q in rows[kind]])
        if base.numel() > max_candidates:
            g = torch.Generator().manual_seed(_seed(spec.bsp_id, kind + ":pre", seed))
            keep = torch.randperm(base.numel(), generator=g)[:max_candidates].sort().values
            base, srcp = base[keep], srcp[keep]
        if kind == "switch_on" and spec.kind in ("offered_attr", "win_now"):
            target = src_wins[base, srcp]
        else:
            target = tgt[kind][base]
        if spec.kind in ("pinned", "winnable"):
            src_poles = t.threat[base, spec.group, :] & P[srcp]
        else:
            src_poles = torch.zeros(base.numel(), 8, dtype=torch.bool)
        threat_present = (t.threat[base, spec.group, spec.pole] if spec.kind == "pinned"
                          else torch.ones(base.numel(), dtype=torch.bool))
        out[kind] = PairSet(kind, base, srcp, target, legal_all[base], expect[base],
                            src_poles, threat_present)
    return out
