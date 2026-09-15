"""Phase 3B-causal: interchange interventions (offered-piece swap), Waves 1 and 1b.

The config's ``rule`` picks the design:
  3B.C1 (default) -- Wave 1: docs/diary/2026-09-12_3B-causal-preregistration.md
          as amended (pre-data) by amendments 1 and 2; oracle targets.
  3B.C2 -- Wave 1b: docs/diary/2026-09-14_3B-causal-wave1b-preregistration.md;
          network-own targets, specificity F / E / rho, B1' null, DAS-1 on all
          three kinds, on a gold position set (freshness re-checked, feasibility
          rule applied before any score).
Game-agnostic machinery lives in lib/sae/interchange.py; Quarto pairs in
scripts/games/quarto_counterfactuals.py; everything champion- or set-specific
comes from ONE config file (configs/3B-causal/champ<Tag>[-<set>].yaml, which may
``extends:`` another) -- this script names no champion.

Stages (a --dry-run stops after 4 and computes no interchange score):
  1. freeze stamps: SHA-256 of every 3B-causal pre-registration and amendment
  2. generator guard: vectorised concepts == stored BSP labels, exactly
     (3B.C2 also: the freshness re-check against the pilot's pair boards)
  3. Tier A on the real model: A1a, A1b, A2, A3 (amendment 1) + readout choice
  4. pairs, switch-off filter, caps, power table (3B.C2: + feasibility)
  5. per concept: R1-R7 on every pair kind, nulls, p-values, CIs
  6. BH, verdicts (the config's rule), the DAS-1 gate, replicate dictionaries

Outputs (in --out-dir):
  3B-causal_<champ>_wave1[_pairs.pt|_dryrun].json      rule 3B.C1
  3B-causal_<tag>_wave1b[_pairs.pt|_dryrun].json       rule 3B.C2 (tag from the config)

Usage:
    interchange_3b.py --config=<yaml> [options]
    interchange_3b.py (-h | --help)

Options:
    -h --help            Show this help message.
    --config=<yaml>      configs/3B-causal/champ<Tag>.yaml
    --dry-run            Stop after the power table; compute no interchange score.
    --untrained          Run on the champion's UNTRAINED twin: a pipeline smoke
                         test whose numbers are meaningless by construction.
    --device=<d>         torch device [default: cuda]
    --cap=<n>            Pairs per (concept, kind) [default: 1000]
    --null-draws=<n>     Draws per null distribution [default: 1000]
    --concepts=<globs>   Only concepts whose bsp_id matches one of these
                         comma-separated globs (smoke tests only; BH needs
                         every concept).
    --no-replicates      Skip the replicate / seed dictionaries.
    --prereqs            Check every input the config names; print what is
                         missing with the command that produces it (lines
                         starting "RUN: " are safe to execute). Exit 0 if
                         nothing is missing, 3 if every missing input has a
                         RUN command, 4 if any does not. Computes nothing.
    --require-frozen     Refuse to run unless the pre-registration and every
                         amendment are committed and unmodified (the runner
                         passes this for every real run).
    --out-dir=<dir>      [default: saes/quarto/analysis]
    --seed=<int>         [default: 0]
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from docopt import docopt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from lib.sae import interchange as ix  # noqa: E402
from lib.sae.eval import match_features_to_bsps, top_k_features_per_bsp  # noqa: E402
from lib.sae.train import load_checkpoint  # noqa: E402
from scripts.games import get_game_module  # noqa: E402
from scripts.games import quarto_counterfactuals as qc  # noqa: E402

TOPK = 16
N_FOLDS = 5
DAS_STEPS, DAS_LR = 400, 0.05                     # amendment 2, B4
ANALYSIS = ROOT / "saes/quarto/analysis"          # 3A reports and top-K exports live here
PREREG_GLOB = "docs/diary/*_3B-causal-preregistration.md"
AMEND_GLOB = "docs/diary/*_3B-causal-amendment-*.md"
WAVE1B_GLOBS = ("docs/diary/*_3B-causal-wave1b-preregistration.md",
                "docs/diary/*_3B-causal-wave1b-amendment-*.md")
ROLES = ("R1", "R2", "R3", "R4", "R5", "R6", "R7")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def load_config(path: str | Path) -> dict:
    """A run config. ``extends: <file>`` (relative to this file) inherits every
    top-level key not set here, so a position-set config (Wave 1b's gold sets)
    cannot drift from its champion's dictionaries, probes or concepts."""
    p = ROOT / path
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    parent = cfg.pop("extends", None)
    if parent:
        merged = load_config(p.parent / parent)
        merged.update(cfg)
        cfg = merged
    return cfg


def freeze_stamps() -> dict:
    """SHA-256 of the design documents, read before any score exists. Wave 1b
    builds on Wave 1's design, so every 3B-causal document is stamped."""
    files = (sorted(ROOT.glob(PREREG_GLOB)) + sorted(ROOT.glob(AMEND_GLOB))
             + [f for g in WAVE1B_GLOBS for f in sorted(ROOT.glob(g))])
    stamps = {}
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        # Hash LINE-NORMALISED content: git on Windows may check a file out with
        # CRLF while the committed blob has LF, and a raw-byte hash then fails to
        # match `git show` even though nothing changed (it did, on Wave 1's
        # pre-registration). Normalising makes the stamp comparable to the blob.
        content = f.read_bytes().replace(b"\r\n", b"\n")
        stamps[rel] = {"sha256_lf": hashlib.sha256(content).hexdigest(),
                       "committed_and_clean": dirty == ""}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    return {"git_head": head, "files": stamps}


def check_prereqs(cfg: dict, cfg_path: str | None = None) -> list[dict]:
    """Every input a run needs, and how to produce the missing ones. This is
    what makes a new champion -- or a new position set -- a new config file: the
    runner reads this list. A config with a ``positions_recipe`` (Wave 1b's gold
    sets) is built by scripts/build_gold_sets.py, so its positions, labels and
    orbit IDs come with a RUN command."""
    champ = yaml.safe_load((ROOT / cfg["champion"]).read_text(encoding="utf-8"))
    suffix, missing = cfg["bsp_suffix"], []
    lsuf = cfg.get("labels_suffix", suffix)
    build = (f"RUN: python scripts/build_gold_sets.py --config={cfg_path}"
             if cfg.get("positions_recipe") and cfg_path else None)

    def need(item, path, command=None):
        if not (ROOT / path).exists():
            missing.append({"item": item, "path": str(path), "command": command})

    need("champion checkpoint", champ["path"], "train it (see configs/models/)")
    need("untrained twin", champ["random_path"], "save the E_0000 checkpoint")
    if cfg.get("pilot_run"):
        need("pilot run (freshness filter)", cfg["pilot_run"], "the Wave-1 pilot run")
    need("positions", cfg["positions"], build or "runners/champ<Tag>.ps1 (position generation)")
    need("activations", cfg["activations"], "runners/champ<Tag>.ps1 (activation collection)")
    need("orbit ids", cfg["orbit_ids"], build or "python scripts/compute_orbit_ids.py ...")
    dicts = cfg["dictionaries"]
    for rid in ([dicts["primary"]] + list(dicts.get("replicate") or []) + list(dicts.get("seeds") or [])
                + ([dicts["anchored"]] if dicts.get("anchored") else [])):
        need(f"dictionary {rid}", f"saes/quarto/{rid}.pt", "the champion's SAE sweep")
    for basis in cfg["wave1"]["bases"]:
        schema = sorted(ROOT.glob(f"data/quarto/bsp_schema-{basis}_*.json"))
        if not schema:
            missing.append({"item": f"schema {basis}", "path": f"data/quarto/bsp_schema-{basis}_*.json",
                            "command": None})
            continue
        n = schema[0].stem.rsplit("_", 1)[-1]
        labels = f"data/quarto/bsp_labels-{basis}{suffix}_{n}.pt"     # the amalgam's
        need(f"labels {basis}{suffix}", labels,
             f"python scripts/compute_bsp_labels.py {cfg['positions']} --game quarto --name {basis}")
        if lsuf != suffix:
            need(f"labels {basis}{lsuf} (the position set's own, for the guard)",
                 f"data/quarto/bsp_labels-{basis}{lsuf}_{n}.pt", build)
        need(f"3A report {basis} (knee_k for R2; R2 is not-run without it)",
             f"saes/quarto/analysis/{dicts['primary']}_dilution-{basis}{suffix}.json",
             f"python scripts/dilution_diagnostic.py --run-id={dicts['primary']} --bsps={basis}{suffix}")
        probe = cfg["linear_probes"].get(basis)
        if probe:
            directions = probe.replace("_results.json", "_directions.pt")
            need(f"probe directions {basis} (R5)", directions,
                 f"RUN: python scripts/linear_probe_baseline.py {cfg['activations']} {labels} "
                 f"{schema[0].relative_to(ROOT).as_posix()} --max-train=50000 --seed=42 --output={probe}")
    return missing


def onehot(idx: torch.Tensor, n: int = 16) -> torch.Tensor:
    return torch.nn.functional.one_hot(idx.long(), n).float()


# ---------------------------------------------------------------------------
# Champion: model, hook values, readouts
# ---------------------------------------------------------------------------


class Champion:
    def __init__(self, cfg: dict, untrained: bool, device: str):
        champ = yaml.safe_load((ROOT / cfg["champion"]).read_text(encoding="utf-8"))
        self.name = champ["name"] + ("-UNTRAINED" if untrained else "")
        path = champ["random_path"] if untrained else champ["path"]
        self.model = get_game_module(champ["game"]).load_model(str(ROOT / path), device=device)
        self.model.eval()
        self.device = device
        self.hook = ix.resolve_module(self.model, cfg["hook"])
        head = ix.resolve_module(self.model, cfg["heads"][cfg["wave1"]["head"]])
        self.closed = ix.LinearReluReadout(head.weight.detach().float(), head.bias.detach().float())
        self.forward = ix.ForwardHookReadout(self.model, self.hook, head,
                                             run=lambda inp: self.model(*inp))
        self.readout: ix.Readout = self.closed

    @torch.no_grad()
    def hook_values(self, boards: torch.Tensor, pieces: torch.Tensor, bs: int = 8192) -> torch.Tensor:
        """Hook value z for each (board, offered piece), on the CPU."""
        out, cap = [], {}
        h = self.hook.register_forward_hook(lambda m, i, o: cap.__setitem__("z", o.detach()))
        try:
            for s in range(0, len(boards), bs):
                self.model(boards[s:s + bs].to(self.device), pieces[s:s + bs].to(self.device))
                out.append(cap["z"].float().cpu())
        finally:
            h.remove()
        return torch.cat(out)

    def logits(self, zp: torch.Tensor, inputs) -> torch.Tensor:
        """Head logits for patched hook values; ``zp`` is (n, d) or (draws, n, d).
        The closed form batches over draws; the generic path loops, so a
        champion whose downstream map is not linear still runs, only slower."""
        ro = self.readout
        if isinstance(ro, ix.LinearReluReadout):
            return torch.relu(zp) @ ro.W.T + ro.b
        if zp.dim() == 3:
            return torch.stack([ro(zp[k], inputs) for k in range(zp.shape[0])])
        return ro(zp, inputs)


@torch.no_grad()
def tier_a(ch: Champion, z_b, z_s, legal, base_inp, src_inp, mode: str, seed: int) -> dict:
    """Amendment 1, A1: the exact controls, on real pairs of the real model."""
    g = torch.Generator().manual_seed(seed)
    res = {}
    W = ch.closed.W.double()
    _, _, Vh = torch.linalg.svd(W.cpu(), full_matrices=True)
    null = Vh[W.shape[0]:].to(z_b.device)
    h = torch.relu(z_b).double()
    v = null[torch.randint(len(null), (1,), generator=g).item()] * 3.0
    res["A1a_null_space_on_h_max_abs"] = float((((h + v) @ W.T) - (h @ W.T)).abs().max())
    z2 = torch.where(z_b < 0, z_b - torch.rand(z_b.shape, generator=g).to(z_b.device), z_b)
    res["A1b_dead_units_on_z_max_abs"] = float((ch.closed(z2) - ch.closed(z_b)).abs().max())
    net_on_source = ix.legal_argmax(ch.forward(z_s, src_inp), legal)   # natural forward on s
    res["A2_full_patch_agreement"] = float(
        (ix.legal_argmax(ch.closed(ix.patch_full(z_b, z_s)), legal) == net_on_source).float().mean())
    zp = z_b + torch.randn(z_b.shape, generator=g).to(z_b.device)
    lc, lf = ch.closed(zp), ch.forward(zp, base_inp)
    res["A3_closed_vs_forward_max_abs"] = float((lc - lf).abs().max())
    res["A3_decisions_identical"] = bool(torch.equal(ix.legal_argmax(lc, legal),
                                                     ix.legal_argmax(lf, legal)))
    ok = {"A1a": res["A1a_null_space_on_h_max_abs"] < 1e-5,
          "A1b": res["A1b_dead_units_on_z_max_abs"] < 1e-5,
          "A2": res["A2_full_patch_agreement"] == 1.0,
          "A3": res["A3_closed_vs_forward_max_abs"] < 1e-5 and res["A3_decisions_identical"]}
    res["passed"] = ok
    if mode == "closed_form" and not ok["A3"]:
        raise SystemExit("A3 failed: the closed form does not match this architecture")
    if mode == "forward_hook" or not ok["A3"]:
        ch.readout = ch.forward
    else:
        ch.readout = ch.closed
    res["readout"] = ch.readout.kind + ("" if ok["A3"] else " (A3 failed for the closed form)")
    return res


# ---------------------------------------------------------------------------
# Design: concepts, pairs, power (shared with scripts/gold_prefix_sweep.py)
# ---------------------------------------------------------------------------


def load_schemas(cfg: dict) -> dict[str, dict]:
    return {basis: json.loads(sorted(ROOT.glob(f"data/quarto/bsp_schema-{basis}_*.json"))[0]
                              .read_text(encoding="utf-8"))
            for basis in cfg["wave1"]["bases"]}


def wave_concepts(cfg: dict, schemas: dict, globs: str | None = None) -> list[tuple[qc.ConceptSpec, int]]:
    """The pre-registered concept list, as (spec, column in its basis's label file)."""
    w1, out = cfg["wave1"], []
    for basis in w1["bases"]:
        cats = set(w1["categories"].get(basis, []))
        extra = set((w1.get("extra_bsps") or {}).get(basis, []))
        for i, b in enumerate(schemas[basis]["bsps"]):
            if (b["category"] in cats or b["id"] in extra) and (
                    not globs or any(fnmatch.fnmatch(b["id"], g) for g in globs.split(","))):
                out.append((qc.parse_concept(b["id"], basis), i))
    return out


@torch.no_grad()
def design_pairs(ch: Champion, specs, raw: dict, boards, pieces, Z_all, cap: int, seed: int,
                 dev: str) -> tuple[dict, dict]:
    """Switch-off filter, then caps (amendment 2, B2); consumes ``raw``.

    Switch-off keeps the pairs whose UNPATCHED base decision is the expected one,
    made with the champion's current readout (Tier A's choice in a run). Returns
    the capped pairs and the design-stage power table."""
    pairs, power = {}, {}
    for s in specs:
        ps_all = raw.pop(s.bsp_id)
        off = ps_all["switch_off"]
        if off.n:
            inp = (boards[off.base].to(dev), pieces[off.base].to(dev))
            dec = ix.legal_argmax(ch.logits(Z_all[off.base].to(dev), inp), off.legal.to(dev)).cpu()
            ps_all["switch_off"] = off.select(off.expect_base[torch.arange(off.n), dec])
        pairs[s.bsp_id] = {k: qc.cap_pairs(v, cap, s.bsp_id, seed) for k, v in ps_all.items()}
        power[s.bsp_id] = {k: {"n": v.n, "boards": int(torch.unique(v.base).numel())}
                           for k, v in pairs[s.bsp_id].items()}
    return pairs, power


# ---------------------------------------------------------------------------
# Dictionaries
# ---------------------------------------------------------------------------


class Dictionary:
    """An SAE, its per-sample encoder, firing frequencies and top-16 lists."""

    def __init__(self, run_id: str, device: str):
        self.run_id = run_id
        self.sae, _ = load_checkpoint(ROOT / f"saes/quarto/{run_id}.pt", device=device)
        self.sae.eval()
        self.W_dec = self.sae.W_dec.detach().float()
        self.device = device
        self.freq: torch.Tensor | None = None

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        return self.sae.encode(z.to(self.device)).float()

    @torch.no_grad()
    def all_codes(self, activations: torch.Tensor, bs: int = 16384) -> torch.Tensor:
        """Codes over the whole position set: the sae_eval `_h` cache when it
        exists (the exact codes the committed exports came from), else encoded."""
        cache = ROOT / f"saes/quarto/cache/{self.run_id}_h.pt"
        if cache.exists():
            h = torch.load(cache, map_location="cpu", weights_only=False)
            return (h["h"] if isinstance(h, dict) else h).float()
        return torch.cat([self.encode(activations[s:s + bs]).cpu()
                          for s in range(0, len(activations), bs)])

    def shortlist(self, animal: str, schema: dict, labels: torch.Tensor,
                  activations: torch.Tensor) -> dict:
        """bsp_id -> {"top": top-16 latents by MCC, "abs_mcc": |MCC| of every latent}.
        Reuses sae_eval's matching cache when present; otherwise computes the
        matching from codes and writes the export (amendment 2, B4)."""
        from scripts.export_topk_matches import build_report_from_matching, load_matching

        cache = ROOT / f"saes/quarto/cache/{self.run_id}_matching-{animal}.pt"
        if cache.exists():
            m = load_matching(cache)
        else:
            log(f"    {self.run_id}:{animal} has no matching cache -- computing from codes")
            h = self.all_codes(activations)
            if self.freq is None:
                self.freq = (h > 0).float().mean(0)
            m = match_features_to_bsps(h, labels.float())
            del h
            rep = build_report_from_matching(m, self.run_id, animal, "quarto", TOPK, "mcc",
                                             source="computed by scripts/interchange_3b.py")
            out = ANALYSIS / f"{self.run_id}_topk-{animal}.json"
            if not out.exists():
                out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
                log(f"    wrote {out.relative_to(ROOT)}")
        tk = top_k_features_per_bsp(m, k=TOPK, metric="mcc")
        return {b["id"]: {"top": tk.indices[i].tolist(), "abs_mcc": m.mcc[:, i].abs()}
                for i, b in enumerate(schema["bsps"])}

    def ensure_freq(self, activations: torch.Tensor) -> torch.Tensor:
        if self.freq is None:
            h = self.all_codes(activations)
            self.freq = (h > 0).float().mean(0)
            del h
        return self.freq


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _decide(logits, legal, target):
    """(hits, decisions); broadcasts over a leading draws dimension."""
    dec = logits.masked_fill(~legal, float("-inf")).argmax(-1)
    tgt = target.expand(*dec.shape, target.shape[-1]) if dec.dim() > target.dim() - 1 else target
    return tgt.gather(-1, dec.unsqueeze(-1)).squeeze(-1), dec


@torch.no_grad()
def null_directions(ch, d, dirs, chunk=64):
    t = (d["zs"] - d["zb"]) @ dirs.T                         # (n, D)
    H, DEC = [], []
    for s in range(0, len(dirs), chunk):
        w = dirs[s:s + chunk]
        zp = d["zb"].unsqueeze(0) + t[:, s:s + chunk].T.unsqueeze(-1) * w.unsqueeze(1)
        h, dec = _decide(ch.logits(zp, d["inp"]), d["legal"], d["target"])
        H.append(h)
        DEC.append(dec)
    return torch.cat(H), torch.cat(DEC)


@torch.no_grad()
def null_latent_sets(ch, d, W_dec, sets, chunk=32):
    H, DEC = [], []
    for s in range(0, len(sets), chunk):
        idx = sets[s:s + chunk].to(d["dA"].device)          # (c, k)
        delta = torch.einsum("nck,ckd->cnd", d["dA"][:, idx], W_dec[idx])
        h, dec = _decide(ch.logits(d["zb"].unsqueeze(0) + delta, d["inp"]), d["legal"], d["target"])
        H.append(h)
        DEC.append(dec)
    return torch.cat(H), torch.cat(DEC)


def summarize(hp, hb, dp, db, groups, null_h=None, null_d=None, seed=0) -> dict:
    """One arm under rule 3B.C1 (Wave 1): the oracle target."""
    n = int(hp.numel())
    iia, r0 = float(hp.float().mean()), float(hb.float().mean())
    s = ix.chance_corrected(iia, r0)
    flip = float((dp != db).float().mean())
    lo, hi = ix.bootstrap_iia_star(hp.cpu(), hb.cpu(), groups.cpu(), n_boot=1000, seed=seed)
    out = {"n": n, "iia": iia, "r0": r0, "iia_star": s, "ci95": [lo, hi], "flip_rate": flip}
    if null_h is not None and s is not None:
        null_star = (null_h.float().mean(1) - r0) / (1 - r0)
        null_flip = (null_d != db.unsqueeze(0)).float().mean(1)
        p, par = ix.empirical_p(s, null_star.cpu())
        pf, _ = ix.empirical_p(flip, null_flip.cpu())
        p5, p95 = float(null_star.quantile(0.05)), float(null_star.quantile(0.95))
        out.update({"p": p, "parametric_tail": par, "null_p5": p5, "null_p95": p95,
                    "below_null_p5": bool(s < p5), "flip_p": pf})
    return out


def null_stats_c2(kind: str, observed: float, dp, db, ds, null_d, r0) -> dict:
    """The null block of one 3B.C2 arm, from the null's DECISIONS (draws, n).

    Greater-tail p on IIA_net* (switch-on/off) or on F (specificity); E = F minus
    the null's median F; anti-consistent and off-target flags (Wave 1b S8)."""
    null_flip = (null_d != db.unsqueeze(0)).float().mean(1).cpu()
    if kind == "specificity":
        med = float(null_flip.median())
        p, par = ix.empirical_p(observed, null_flip)
        return {"p": p, "parametric_tail": par, "null_median_F": med, "E": observed - med,
                "null_p95": float(null_flip.quantile(0.95))}
    null_star = ((null_d == ds.unsqueeze(0)).float().mean(1).cpu() - r0) / (1 - r0)
    p, par = ix.empirical_p(observed, null_star)
    p5, p95 = float(null_star.quantile(0.05)), float(null_star.quantile(0.95))
    flip = float((dp != db).float().mean())
    fp, _ = ix.empirical_p(flip, null_flip)
    return {"p": p, "parametric_tail": par, "null_p5": p5, "null_p95": p95,
            "below_null_p5": bool(observed < p5), "flip_p": fp,
            "flip_above_null_p95": bool(flip > float(null_flip.quantile(0.95)))}


def summarize_c2(kind: str, A: dict, seed: int = 0) -> dict:
    """One arm under rule 3B.C2 (Wave 1b S6): the network-own score, Wave 1's
    oracle score for continuity, the flip rate and target margin; the primary
    null (``nd``) and, for directions, the isotropic one (``nd_iso``)."""
    dp, db, ds, grp = A["dp"], A["db"], A["ds"], A["grp"].cpu()
    flip = dp != db
    oiia, or0 = float(A["hp"].float().mean()), float(A["hb"].float().mean())
    out = {"n": int(dp.numel()), "flip_rate": float(flip.float().mean()),
           "oracle": {"iia": oiia, "r0": or0, "iia_star": ix.chance_corrected(oiia, or0)}}
    if A.get("margin") is not None:
        out["target_margin_mean"] = float(A["margin"].mean())
    if kind == "specificity":
        lo, hi = ix.bootstrap_rate(flip.cpu(), grp, n_boot=1000, seed=seed)
        out.update({"F": out["flip_rate"], "iia_star": None, "ci95": [lo, hi]})
        observed, r0 = out["flip_rate"], None
    else:
        hp, hb = dp == ds, db == ds
        iia, r0 = float(hp.float().mean()), float(hb.float().mean())
        observed = ix.chance_corrected(iia, r0)
        lo, hi = ix.bootstrap_iia_star(hp.cpu(), hb.cpu(), grp, n_boot=1000, seed=seed)
        out.update({"iia": iia, "r0": r0, "iia_star": observed, "ci95": [lo, hi]})
    if A.get("nd") is not None and observed is not None:
        out.update(null_stats_c2(kind, observed, dp, db, ds, A["nd"], r0))
        if A.get("nd_iso") is not None:
            out["isotropic_null"] = null_stats_c2(kind, observed, dp, db, ds, A["nd_iso"], r0)
    return out


class Records:
    """Per-pair records, so every figure is regenerated rather than transcribed.
    ``hit_*`` are against the ORACLE target (Wave 1's); a 3B.C2 record also
    carries the network's own decisions on base and source."""

    def __init__(self):
        self.rows: dict[str, dict] = {}

    def add(self, key, A: dict):
        row = {"base": A["base"].cpu().int(), "src_piece": A["src"].cpu().to(torch.int8),
               "hit_patched": A["hp"].cpu().bool(), "hit_base": A["hb"].cpu().bool(),
               "dec_patched": A["dp"].cpu().to(torch.int8)}
        if A.get("ds") is not None:
            row.update(dec_base=A["db"].cpu().to(torch.int8), dec_source=A["ds"].cpu().to(torch.int8))
        self.rows[key] = row


def _cat_arms(parts: list[dict]) -> dict:
    """Concatenate one arm's held-out folds; null tensors are (draws, n)."""
    out = {}
    for key in parts[0]:
        vals = [p[key] for p in parts]
        out[key] = None if vals[0] is None else torch.cat(vals, 1 if key in ("nh", "nd", "nd_iso") else 0)
    return out


# ---------------------------------------------------------------------------
# One concept
# ---------------------------------------------------------------------------


def run_concept(spec, pairs, ch, Z_all, boards, pieces, orbit, D, sl, knee, lp, anch,
                anch_idx, n_draws, seed, dev, rec: Records, rule: str = ix.RULE_VERSION) -> dict:
    c2 = rule == ix.RULE_C2
    rng = torch.Generator().manual_seed(qc._seed(spec.bsp_id, "null", seed))
    res = {"bsp_id": spec.bsp_id, "basis": spec.basis, "kind": spec.kind,
           "n_pairs": {k: v.n for k, v in pairs.items()}, "roles": {}}
    if c2:
        res["ceiling"] = {}
    data = {}
    for kind, ps in pairs.items():
        if ps.n == 0:
            continue
        inp = (boards[ps.base].to(dev), pieces[ps.base].to(dev))
        zb = Z_all[ps.base].to(dev)
        zs = ch.hook_values(boards[ps.base], onehot(ps.src_piece)).to(dev)
        legal, target = ps.legal.to(dev), ps.target.to(dev)
        lb = ch.logits(zb, inp)
        hb, db = _decide(lb, legal, target)
        d = dict(zb=zb, zs=zs, inp=inp, legal=legal, target=target, hb=hb, db=db,
                 dA=D.encode(zs) - D.encode(zb), grp=orbit[ps.base], ps=ps)
        if c2:
            # D(s): the network's own decision on the source, natural forward
            src_inp = (boards[ps.base].to(dev), onehot(ps.src_piece).to(dev))
            hs, ds = _decide(ch.logits(zs, src_inp), legal, target)
            own = ds if kind != "specificity" else db       # network-own target (S6)
            d.update(ds=ds, lb=lb, tnet=torch.nn.functional.one_hot(own, legal.shape[1]).bool())
            # the ceiling: the full-activation patch, under both targets (S6)
            if kind == "specificity":
                net = {"F": float((ds != db).float().mean())}
            else:
                a0 = float((db == ds).float().mean())
                net = {"iia_star": ix.chance_corrected(1.0, a0), "A0": a0}
            oi, o0 = float(hs.float().mean()), float(hb.float().mean())
            res["ceiling"][kind] = {"network_own": net, "oracle": {
                "iia": oi, "r0": o0, "iia_star": ix.chance_corrected(oi, o0)}}
        data[kind] = d

    # folds grouped by orbit, one assignment shared by every pair kind (amendment 2, B4)
    uniq = torch.unique(torch.cat([d["grp"] for d in data.values()]))
    perm = uniq[torch.randperm(len(uniq), generator=torch.Generator().manual_seed(
        qc._seed(spec.bsp_id, "folds", seed)))]
    fold_of = {int(u): i % N_FOLDS for i, u in enumerate(perm)}
    for d in data.values():
        d["fold"] = torch.tensor([fold_of[int(x)] for x in d["grp"]], device=dev)

    dirs = ix.random_unit_directions(n_draws, Z_all.shape[1], rng).to(dev)
    b1 = {k: null_directions(ch, d, dirs) for k, d in data.items()}   # B1, isotropic
    b1c = None
    if c2:   # B1': covariance of z_s - z_b over the concept's pairs of all kinds (S8)
        delta = torch.cat([d["zs"] - d["zb"] for d in data.values()])
        dirs_c = ix.covariance_matched_directions(delta, n_draws, rng).to(dev)
        b1c = {k: null_directions(ch, d, dirs_c) for k, d in data.items()}
    freq = D.freq.cpu()
    alive = freq > 0

    def pack(d, lp_, dp, hp, te=None, nh=None, nd=None, nd_iso=None) -> dict:
        """Everything one arm's score and record need (held-out rows ``te``)."""
        g = (lambda x: x) if te is None else (lambda x: x[te])
        gc = (lambda x: x) if te is None else (lambda x: x[te.cpu()])
        A = {"dp": dp, "hp": hp, "hb": g(d["hb"]), "db": g(d["db"]), "grp": gc(d["grp"]),
             "base": gc(d["ps"].base), "src": gc(d["ps"].src_piece), "nh": nh, "nd": nd,
             "nd_iso": nd_iso}
        if c2:
            A["ds"] = g(d["ds"])
            A["margin"] = ix.target_margin(g(d["lb"]), lp_, g(d["tnet"]), g(d["legal"]))
        return A

    def score(kind, A) -> dict:
        if c2:
            return summarize_c2(kind, A, seed)
        return summarize(A["hp"], A["hb"], A["dp"], A["db"], A["grp"], A["nh"], A["nd"], seed)

    def direction_nulls(k, te=None):
        """(nh, nd, nd_iso) for a direction role: B1' primary under 3B.C2."""
        cut = (lambda x: x) if te is None else (lambda x: x[:, te])
        if c2:
            return cut(b1c[k][0]), cut(b1c[k][1]), cut(b1[k][1])
        return cut(b1[k][0]), cut(b1[k][1]), None

    def patch_latents(d, J, Wd=None, mask=None):
        Wd = D.W_dec if Wd is None else Wd
        zb, dA = (d["zb"], d["dA"]) if mask is None else (d["zb"][mask], d["dA"][mask])
        return zb + dA[:, J] @ Wd[J]

    def latent_role(J, name):
        Jt = torch.tensor(J, dtype=torch.long, device=dev)
        sets = torch.stack(ix.frequency_matched_sets(J, freq, alive, n_draws, rng))
        out = {"latents": list(J)}
        for k, d in data.items():
            lp_ = ch.logits(patch_latents(d, Jt), d["inp"])
            hp, dp = _decide(lp_, d["legal"], d["target"])
            nh, nd = null_latent_sets(ch, d, D.W_dec, sets)
            A = pack(d, lp_, dp, hp, nh=nh, nd=nd)
            out[k] = score(k, A)
            rec.add(f"{spec.bsp_id}|{name}|{k}", A)
        res["roles"][name] = out

    def crossfit_role(name, pick, null_for_fold):
        """Select on training folds (pick(f)), score every kind held out."""
        parts = {k: [] for k in data}
        picks = []
        for f in range(N_FOLDS):
            chosen, patch = pick(f)
            picks.append(chosen)
            nulls = null_for_fold(chosen)
            for k, d in data.items():
                te = d["fold"] == f
                if not bool(te.any()):
                    continue
                inp_te = (d["inp"][0][te], d["inp"][1][te])
                lp_ = ch.logits(patch(d, te), inp_te)
                hp, dp = _decide(lp_, d["legal"][te], d["target"][te])
                sub = {"zb": d["zb"][te], "zs": d["zs"][te], "dA": d["dA"][te], "inp": inp_te,
                       "legal": d["legal"][te], "target": d["target"][te]}
                nh, nd, nd_iso = nulls(sub, k, te)
                parts[k].append(pack(d, lp_, dp, hp, te=te, nh=nh, nd=nd, nd_iso=nd_iso))
        out = {"picked_per_fold": picks}
        for k, pk in parts.items():
            if not pk:
                continue
            A = _cat_arms(pk)
            out[k] = score(k, A)
            rec.add(f"{spec.bsp_id}|{name}|{k}", A)
        res["roles"][name] = out

    if sl is not None:
        top = sl["top"]
        latent_role(top[:1], "R1")
        if knee is not None:
            latent_role(top[:max(1, min(TOPK, int(knee)))], "R2")
        else:
            res["roles"]["R2"] = "not-run: no 3A knee_k"
        latent_role(top[:TOPK], "R3")
        # B3 control: the frequency-matched latent with the lowest |MCC| with C
        f1 = float(freq[top[0]])
        band = torch.nonzero(alive & (freq >= f1 * 0.8) & (freq <= f1 * 1.2)).flatten()
        band = band[band != top[0]]
        if band.numel() and "switch_on" in data:
            b3 = int(band[sl["abs_mcc"][band].argmin()])
            d = data["switch_on"]
            lp_ = ch.logits(patch_latents(d, torch.tensor([b3], device=dev)), d["inp"])
            hp, dp = _decide(lp_, d["legal"], d["target"])
            res["roles"]["R1"]["B3_control_switch_on"] = {
                "latent": b3, **score("switch_on", pack(d, lp_, dp, hp))}

        if "switch_on" in data:
            def pick_r4(f):
                """The top-16 latent with the best switch-on score on the training
                folds: oracle IIA* (3B.C1) or network-own IIA_net* (3B.C2, S7)."""
                on = data["switch_on"]
                tr = on["fold"] != f
                best, best_s = top[0], -1e9
                inp_tr = (on["inp"][0][tr], on["inp"][1][tr])
                r0 = float((on["db"][tr] == on["ds"][tr]).float().mean()) if c2 \
                    else float(on["hb"][tr].float().mean())
                for j in top[:TOPK]:
                    jt = torch.tensor([j], device=dev)
                    hp, dp = _decide(ch.logits(patch_latents(on, jt, mask=tr), inp_tr),
                                     on["legal"][tr], on["target"][tr])
                    a = float((dp == on["ds"][tr]).float().mean()) if c2 else float(hp.float().mean())
                    s = ix.chance_corrected(a, r0)
                    if s is not None and s > best_s:
                        best, best_s = j, s
                jt = torch.tensor([best], device=dev)
                return best, (lambda d, te: patch_latents(d, jt, mask=te))

            def r4_null(chosen):
                sets = torch.stack(ix.frequency_matched_sets([chosen], freq, alive, n_draws, rng))
                return lambda sub, k, te: (*null_latent_sets(ch, sub, D.W_dec, sets), None)
            crossfit_role("R4", pick_r4, r4_null)
    else:
        for r in ("R1", "R2", "R3", "R4"):
            res["roles"][r] = "not-run: no shortlist"

    def direction_role(w, name):
        w = w.to(dev).float()
        out = {}
        for k, d in data.items():
            lp_ = ch.logits(ix.patch_direction(d["zb"], d["zs"], w), d["inp"])
            hp, dp = _decide(lp_, d["legal"], d["target"])
            nh, nd, nd_iso = direction_nulls(k)
            A = pack(d, lp_, dp, hp, nh=nh, nd=nd, nd_iso=nd_iso)
            out[k] = score(k, A)
            rec.add(f"{spec.bsp_id}|{name}|{k}", A)
        res["roles"][name] = out

    if lp is not None:
        direction_role(lp, "R5")
    else:
        res["roles"]["R5"] = "not-run: probe not fitted"

    if anch is not None and anch_idx is not None:
        out = {"latent": anch_idx}
        j = torch.tensor([anch_idx], device=dev)
        for k, d in data.items():
            dA6 = anch.encode(d["zs"]) - anch.encode(d["zb"])
            lp_ = ch.logits(d["zb"] + dA6[:, j] @ anch.W_dec[j], d["inp"])
            hp, dp = _decide(lp_, d["legal"], d["target"])
            nh, nd, nd_iso = direction_nulls(k)
            A = pack(d, lp_, dp, hp, nh=nh, nd=nd, nd_iso=nd_iso)
            out[k] = score(k, A)
            rec.add(f"{spec.bsp_id}|R6|{k}", A)
        res["roles"]["R6"] = out
    else:
        res["roles"]["R6"] = "not-run: no anchored slot for this basis"

    if "switch_on" in data:
        cos = []

        def pick_r7(f):
            """DAS-1 on the training folds. 3B.C1: switch-on pairs toward the oracle
            target. 3B.C2 (S7): all three kinds toward the network-own targets,
            each kind's loss averaged separately, then the kinds averaged."""
            with torch.enable_grad():
                if c2:
                    batches = []
                    for d in data.values():
                        tr = d["fold"] != f
                        batches.append(ix.DasBatch(d["zb"][tr], d["zs"][tr], d["tnet"][tr], d["legal"][tr],
                                                   (d["inp"][0][tr], d["inp"][1][tr])))
                    w = ix.train_das_direction_multi(batches, ch.readout, steps=DAS_STEPS, lr=DAS_LR,
                                                     seed=qc._seed(spec.bsp_id, f"das{f}", seed))
                else:
                    on = data["switch_on"]
                    tr = on["fold"] != f
                    w = ix.train_das_direction(on["zb"][tr], on["zs"][tr], ch.readout, on["target"][tr],
                                               on["legal"][tr],
                                               inputs=(on["inp"][0][tr], on["inp"][1][tr]),
                                               steps=DAS_STEPS, lr=DAS_LR,
                                               seed=qc._seed(spec.bsp_id, f"das{f}", seed))
            if lp is not None:
                cos.append(float(torch.nn.functional.cosine_similarity(w, lp.to(dev).float(), dim=0)))
            return f"fold{f}", (lambda d, te: ix.patch_direction(d["zb"][te], d["zs"][te], w))

        def r7_null(_):
            return lambda sub, k, te: direction_nulls(k, te)
        crossfit_role("R7", pick_r7, r7_null)
        res["roles"]["R7"]["cos_with_lp_per_fold"] = cos

    # H-C3(b): winnable concepts, switch-on stratified by the single completing pole
    if spec.kind == "winnable" and "switch_on" in data and sl is not None:
        d = data["switch_on"]
        sp = d["ps"].src_poles.to(dev)
        single = sp.sum(1) == 1
        strat = {}
        for rname, J in (("R1", sl["top"][:1]),
                         ("R2", sl["top"][:max(1, min(TOPK, int(knee)))] if knee else None)):
            if J is None:
                continue
            Jt = torch.tensor(J, dtype=torch.long, device=dev)
            hp, dp = _decide(ch.logits(patch_latents(d, Jt), d["inp"]), d["legal"], d["target"])
            hit, base = ((dp == d["ds"]), (d["db"] == d["ds"])) if c2 else (hp, d["hb"])
            per_pole = {}
            for kp, suffix in enumerate(qc.POLE_SUFFIXES):
                m = single & sp[:, kp]
                if int(m.sum()) >= 20:
                    per_pole[suffix] = {"n": int(m.sum()), "iia_star": ix.chance_corrected(
                        float(hit[m].float().mean()), float(base[m].float().mean()))}
            strat[rname] = per_pole
        res["single_pole_switch_on"] = strat
    return res


# ---------------------------------------------------------------------------
# BH, verdicts, the gate
# ---------------------------------------------------------------------------


def assign_verdicts(results: list[dict], rule: str = ix.RULE_VERSION) -> dict:
    """BH within (representation, arm) across concepts, then the ordered rule."""
    c2 = rule == ix.RULE_C2
    for role in ROLES:
        for kind in qc.PAIR_KINDS:
            rows = [r["roles"][role][kind] for r in results
                    if isinstance(r["roles"].get(role), dict)
                    and isinstance(r["roles"][role].get(kind), dict)
                    and "p" in r["roles"][role][kind]]
            for x, sig in zip(rows, ix.benjamini_hochberg([x["p"] for x in rows], ix.BH_Q)):
                x["bh_significant"] = bool(sig)
    for r in results:
        for role in ROLES:
            rr = r["roles"].get(role)
            if not isinstance(rr, dict):
                continue

            def arm(k):
                a = rr.get(k)
                if not isinstance(a, dict):
                    return None
                if c2:
                    return ix.ArmResult(n=a["n"], iia_star=a.get("iia_star"),
                                        significant=a.get("bh_significant", False),
                                        below_null_p5=a.get("below_null_p5", False),
                                        flip_significant=a.get("flip_above_null_p95", False),
                                        excess=a.get("E"))
                return ix.ArmResult(n=a["n"], iia_star=a["iia_star"],
                                    significant=a.get("bh_significant", False),
                                    below_null_p5=a.get("below_null_p5", False),
                                    flip_significant=a.get("flip_p", 1.0) < 0.05)
            on, sp, off = arm("switch_on"), arm("specificity"), arm("switch_off")
            rr["verdict"] = ("underpowered" if on is None or sp is None
                             else ix.classify(on, sp, off, rule=rule))
            if c2 and on is not None and sp is not None:
                rr["rho"] = ix.relative_leak(sp, on)
    powered = [r for r in results if isinstance(r["roles"].get("R7"), dict)
               and r["roles"]["R7"].get("verdict", "underpowered") != "underpowered"]
    cc = [r for r in powered if r["roles"]["R7"]["verdict"] in
          ("concept-consistent", "concept-consistent (on-only)")]
    frac = len(cc) / len(powered) if powered else 0.0
    where = ("Wave 1b pre-registration S10, per gold set" if c2
             else "pre-registration S7 C1; amendment 2 B6")
    return {"C1_das_concept_consistent": len(cc), "C1_powered": len(powered),
            "C1_fraction": frac, "passed": frac >= 0.5, "rule_version": rule,
            "rule": f"R7 concept-consistent or (on-only) for >= 50% of powered concepts ({where})"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = docopt(__doc__)
    dev = args["--device"] if (args["--device"] == "cpu" or torch.cuda.is_available()) else "cpu"
    cap, n_draws, seed = int(args["--cap"]), int(args["--null-draws"]), int(args["--seed"])
    cfg = load_config(args["--config"])
    rule = cfg.get("rule", ix.RULE_VERSION)
    if rule not in ix.RULES:
        raise SystemExit(f"unknown rule {rule!r} in {args['--config']}")
    c2 = rule == ix.RULE_C2
    out_dir = ROOT / args["--out-dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    if args["--prereqs"]:
        missing = check_prereqs(cfg, args["--config"])
        printed = set()
        for m in missing:
            print(f"MISSING {m['item']}: {m['path']}")
            cmd = m["command"]
            if cmd and cmd.startswith("RUN: "):
                if cmd not in printed:
                    print(cmd)
                    printed.add(cmd)
            elif cmd:
                print(f"  -> {cmd}")
        print("prerequisites: all present" if not missing else f"{len(missing)} missing")
        if not missing:
            return 0
        return 3 if all((m["command"] or "").startswith("RUN: ") for m in missing) else 4

    stamps = freeze_stamps()
    frozen = all(v["committed_and_clean"] for v in stamps["files"].values())
    log(f"freeze stamps: {len(stamps['files'])} design files; all committed and clean: {frozen}")
    if args["--require-frozen"] and not frozen:
        dirty = [k for k, v in stamps["files"].items() if not v["committed_and_clean"]]
        raise SystemExit(f"REFUSING TO RUN: design files not committed/clean: {dirty}. "
                         "Commit them first -- they must be frozen before any score exists.")

    ch = Champion(cfg, args["--untrained"], dev)
    log(f"champion {ch.name}, hook {cfg['hook']}, rule {rule}, positions {cfg['positions']}, device {dev}")
    pos = torch.load(ROOT / cfg["positions"], map_location="cpu", weights_only=False)
    boards, pieces = pos["boards"].float(), pos["pieces"].float()
    T = qc.board_tables(boards, pieces)
    orbit = torch.load(ROOT / cfg["orbit_ids"], map_location="cpu", weights_only=False)
    orbit = torch.as_tensor(orbit["orbit_ids"] if isinstance(orbit, dict) else orbit)
    if len(orbit) != T.n:
        raise SystemExit(f"orbit ids ({len(orbit)}) do not match positions ({T.n})")

    fresh = None
    if c2:   # Wave 1b S4.3: the set must share no board with a pilot pair, up to symmetry
        from scripts.freshness_filter import fresh_mask, pilot_board_keys
        keys, info = pilot_board_keys(cfg["pilot_run"])
        stale = int((~fresh_mask(boards, keys)).sum())
        if stale:
            raise SystemExit(f"FRESHNESS CHECK FAILED: {stale} positions share a board with a "
                             f"pilot pair (Wave 1b pre-registration S4.3)")
        fresh = {**info, "stale_positions_at_run": 0,
                 "filter": (pos.get("provenance") or {}).get("freshness")}
        log(f"freshness: 0 of {T.n:,} positions share a board with the pilot's "
            f"{info['pilot_board_orbits']:,} pair-board orbits")

    w1, suffix = cfg["wave1"], cfg["bsp_suffix"]
    lsuf = cfg.get("labels_suffix", suffix)     # the position set's labels (guard)
    schemas = load_schemas(cfg)

    def load_labels(sfx: str) -> dict:
        out = {}
        for basis in w1["bases"]:
            n = len(schemas[basis]["bsps"])
            lab = torch.load(ROOT / f"data/quarto/bsp_labels-{basis}{sfx}_{n}.pt",
                             map_location="cpu", weights_only=False)
            out[basis] = (lab["labels"] if isinstance(lab, dict) else lab).float()
        return out

    labels = load_labels(lsuf)
    chosen = wave_concepts(cfg, schemas, args["--concepts"])
    specs = [s for s, _ in chosen]
    cols = {s.bsp_id: labels[s.basis][:, i] for s, i in chosen}
    mism = qc.check_against_labels(T, specs, cols)
    if any(mism.values()):
        raise SystemExit(f"GENERATOR GUARD FAILED: { {k: v for k, v in mism.items() if v} }")
    log(f"generator guard: 0 mismatches over {len(specs)} concepts x {T.n:,} positions")

    Z_all = ch.hook_values(boards, pieces)
    SW = qc.all_source_wins(T)
    raw = {s.bsp_id: qc.build_pairs(T, s, SW, seed=seed) for s in specs}
    log(f"pairs built for {len(specs)} concepts")

    # Tier A FIRST: it decides the readout every later decision is made with
    probe = next(s for s in specs if raw[s.bsp_id]["switch_on"].n >= 50)
    ps = raw[probe.bsp_id]["switch_on"]
    sel = torch.arange(min(ps.n, 512))
    base_inp = (boards[ps.base[sel]].to(dev), pieces[ps.base[sel]].to(dev))
    src_inp = (boards[ps.base[sel]].to(dev), onehot(ps.src_piece[sel]).to(dev))
    zb = Z_all[ps.base[sel]].to(dev)
    zs = ch.hook_values(*src_inp).to(dev)
    tierA = tier_a(ch, zb, zs, ps.legal[sel].to(dev), base_inp, src_inp,
                   cfg.get("readout", "auto"), seed)
    log(f"Tier A: {tierA['passed']}  readout = {tierA['readout']}")
    if not (tierA["passed"]["A1a"] and tierA["passed"]["A1b"] and tierA["passed"]["A2"]):
        raise SystemExit(f"TIER A FAILED: {tierA}")

    pairs, power = design_pairs(ch, specs, raw, boards, pieces, Z_all, cap, seed, dev)
    under = ix.underpowered_arms(power)
    header = {"rule_version": rule, "wave": "1b" if c2 else 1, "champion": ch.name,
              "untrained_twin": bool(args["--untrained"]), "config": args["--config"],
              "hook": cfg["hook"], "readout": tierA["readout"], "freeze": stamps,
              "tier_a": tierA, "cap": cap, "null_draws": n_draws, "seed": seed,
              "concept_filter": args["--concepts"], "n_concepts": len(specs),
              "power": power, "underpowered_arms": {b: v for b, v in under.items() if v},
              "glossary": {**ix.GLOSSARY, **ix.GLOSSARY_C2} if c2 else ix.GLOSSARY}
    if c2:
        feas = ix.feasibility(power)
        header.update({"position_set": cfg["tag"], "positions": cfg["positions"],
                       "n_positions": T.n, "freshness": fresh, "feasibility": feas})
        log(f"feasibility: {feas['powered_in_every_arm']}/{feas['concepts']} concepts powered in "
            f"every arm ({feas['fraction']:.0%}; per arm {feas['powered_per_arm']}); "
            f"passed={feas['passed']}")
        tag = f"3B-causal_{cfg['tag']}{'-UNTRAINED' if args['--untrained'] else ''}_wave1b"
    else:
        tag = f"3B-causal_{ch.name}_wave1"
    if args["--dry-run"]:
        out = out_dir / f"{tag}_dryrun.json"
        out.write_text(json.dumps(header, indent=1), encoding="utf-8")
        log(f"DRY RUN: no interchange score computed. Wrote {rel(out)}")
        return 0
    if c2 and not header["feasibility"]["passed"] and not args["--concepts"]:
        header["status"] = ("DROPPED: fewer than 50% of concepts powered in every arm "
                            "(Wave 1b pre-registration S4.4). No score computed.")
        out = out_dir / f"{tag}.json"
        out.write_text(json.dumps(header, indent=1), encoding="utf-8")
        log(f"SET DROPPED as infeasible (S4.4); no score computed. Wrote {rel(out)}")
        return 0

    act = torch.load(ROOT / cfg["activations"], map_location="cpu", weights_only=False)
    act = (act["activations"] if isinstance(act, dict) else act).float()
    # shortlists are the AMALGAM's, where the dictionaries were trained (Wave 1b S13)
    amalgam_labels = labels if lsuf == suffix else load_labels(suffix)
    dicts = cfg["dictionaries"]
    D = Dictionary(dicts["primary"], dev)
    ix.check_encoder_is_per_sample(D.encode, Z_all[:64].to(dev))
    shortlists = {b: D.shortlist(f"{b}{suffix}", schemas[b], amalgam_labels[b], act) for b in w1["bases"]}
    D.ensure_freq(act)
    knees = {}
    for b in w1["bases"]:
        rep = ANALYSIS / f"{dicts['primary']}_dilution-{b}{suffix}.json"
        if rep.exists():
            knees[b] = {c["bsp_id"]: (c["knee_k"], c["verdict"])
                        for c in json.loads(rep.read_text(encoding="utf-8"))["concepts"]}
    lps = {}
    for b, path in cfg["linear_probes"].items():
        dp = ROOT / path.replace("_results.json", "_directions.pt")
        if dp.exists():
            dd = torch.load(dp, map_location="cpu", weights_only=False)
            lps[b] = {bid: (dd["coef"][i] if bool(dd["fitted"][i]) else None)
                      for i, bid in enumerate(dd["bsp_ids"])}
    anch = Dictionary(dicts["anchored"], dev) if dicts.get("anchored") else None
    if anch is not None:
        ix.check_encoder_is_per_sample(anch.encode, Z_all[:64].to(dev))
    log(f"dictionaries ready: {D.run_id}; probes {sorted(lps)}; anchored {anch.run_id if anch else None}")

    rec = Records()
    results = []
    tiger_ids = [b["id"] for b in schemas["tiger"]["bsps"]] if "tiger" in schemas else []
    for i, s in enumerate(specs):
        t0 = time.time()
        kn = knees.get(s.basis, {}).get(s.bsp_id, (None, None))
        anch_idx = tiger_ids.index(s.bsp_id) if (anch is not None and s.basis == "tiger") else None
        r = run_concept(s, pairs[s.bsp_id], ch, Z_all, boards, pieces, orbit, D,
                        shortlists[s.basis].get(s.bsp_id), kn[0],
                        lps.get(s.basis, {}).get(s.bsp_id), anch, anch_idx, n_draws, seed, dev, rec,
                        rule=rule)
        r["knee_k_3A"], r["verdict_3A"], r["disjuncts"] = kn[0], kn[1], s.disjuncts
        results.append(r)
        log(f"  [{i + 1}/{len(specs)}] {s.bsp_id}  {time.time() - t0:.1f}s")

    gate = assign_verdicts(results, rule)
    for r in results:
        for role in ("R1", "R2", "R3", "R4", "R5", "R6"):
            if isinstance(r["roles"].get(role), dict):
                r["roles"][role]["readable"] = gate["passed"]
    log(f"gate ({rule}): {gate['C1_das_concept_consistent']}/{gate['C1_powered']} "
        f"({gate['C1_fraction']:.2f}) passed={gate['passed']}")

    replicates = {}
    if not args["--no-replicates"]:
        for rid in list(dicts.get("replicate") or []) + list(dicts.get("seeds") or []):
            Dr = Dictionary(rid, dev)
            rsl = {b: Dr.shortlist(f"{b}{suffix}", schemas[b], amalgam_labels[b], act) for b in w1["bases"]}
            rknee = {}                     # R2 uses THIS dictionary's own 3A knee_k (amendment 2, B4/B7)
            for b in w1["bases"]:
                rep = ANALYSIS / f"{rid}_dilution-{b}{suffix}.json"
                if rep.exists():
                    rknee.update({c["bsp_id"]: c["knee_k"]
                                  for c in json.loads(rep.read_text(encoding="utf-8"))["concepts"]})
            rows = {}
            for s in specs:
                sl = rsl[s.basis].get(s.bsp_id)
                if sl is None:
                    continue
                row = {}
                for kind, ps in pairs[s.bsp_id].items():
                    if ps.n == 0:
                        continue
                    inp = (boards[ps.base].to(dev), pieces[ps.base].to(dev))
                    zb = Z_all[ps.base].to(dev)
                    zs = ch.hook_values(boards[ps.base], onehot(ps.src_piece)).to(dev)
                    dA = Dr.encode(zs) - Dr.encode(zb)
                    legal, target = ps.legal.to(dev), ps.target.to(dev)
                    hb, db = _decide(ch.logits(zb, inp), legal, target)
                    ds = None
                    if c2:
                        s_inp = (boards[ps.base].to(dev), onehot(ps.src_piece).to(dev))
                        ds = _decide(ch.logits(zs, s_inp), legal, target)[1]
                    roles = [("R1", sl["top"][:1]), ("R3", sl["top"][:TOPK])]
                    if rknee.get(s.bsp_id) is not None:
                        roles.insert(1, ("R2", sl["top"][:max(1, min(TOPK, int(rknee[s.bsp_id])))]))
                    for name, J in roles:
                        Jt = torch.tensor(J, device=dev)
                        hp, dp = _decide(ch.logits(zb + dA[:, Jt] @ Dr.W_dec[Jt], inp), legal, target)
                        row.setdefault(name, {})[kind] = (
                            summarize_c2(kind, {"dp": dp, "hp": hp, "hb": hb, "db": db, "ds": ds,
                                                "grp": orbit[ps.base]}, seed) if c2
                            else summarize(hp, hb, dp, db, orbit[ps.base], seed=seed))
                rows[s.bsp_id] = row
            replicates[rid] = rows
            log(f"replicate {rid} done")

    header.update({"gate": gate, "results": results, "replicates": replicates,
                   "elapsed_s": round(time.time() - t_start, 1)})
    out = out_dir / f"{tag}.json"
    out.write_text(json.dumps(header, indent=1, default=float), encoding="utf-8")
    torch.save(rec.rows, out_dir / f"{tag}_pairs.pt")
    log(f"wrote {rel(out)} and {tag}_pairs.pt  ({header['elapsed_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
