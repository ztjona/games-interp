"""Phase 3A dilution diagnostic CLI.

Classify each threat/relational BSP as absent / captured / diluted / tiled in a
trained SAE, from its cached codes. Writes a parseable JSON report and prints
the Gate G-3A tally (>= half of threat BSPs diluted-or-tiled -> 3C proceeds).

Method + thresholds: docs/diary/2026-07-21_3A-dilution-diagnostic.md
Core (game-agnostic): lib/sae/dilution.py

``reclassify`` re-runs the verdict rule on reports that already exist, in place.
The verdict is a pure function of the stored metrics, so changing a threshold
never requires the (multi-GB) _h code caches or a rerun.

Usage:
    dilution_diagnostic.py reclassify <report>... [options]
    dilution_diagnostic.py <checkpoint> [options]
    dilution_diagnostic.py --run-id=<id> [options]
    dilution_diagnostic.py -h | --help

Arguments:
    <checkpoint>          Path to the SAE checkpoint (.pt); stem == run_id.
    <report>              Existing *_dilution-*.json report(s) to re-verdict.

Options:
    --run-id=<id>         Run ID (checkpoint stem) instead of a path.
    --game=<game>        Game name [default: quarto].
    --bsps=<name>        BSP set to diagnose (e.g. tigerTa, gorillaVe). Required.
    --categories=<list>  Comma-separated categories to include
                         [default: auto] (auto = relational/threat categories).
    --random-run-id=<id> Random-model SAE run_id for the absent control
                         [default: none].
    --top-k=<n>          Candidate latents per concept [default: 64].
    --captured-solo-frac=<f>  Share of recoverable signal one latent must carry
                         for a captured verdict [default: 0.70].
    --captured-idim=<f>  Max intrinsic dimension for a captured verdict
                         [default: 2.0].
    --band-sds=<f>       Rule 3A.4 stability band, in sd, applied to every
                         quantity classify thresholds on [default: 3.0].
    --solo-frac-seed-sd=<f>  Per-concept CROSS-SEED sd of solo_frac. Cannot be
                         estimated from one run, so it is a measured constant:
                         the MEAN over seeds 42/43/44 on K03/K04-champYb
                         [default: 0.0523].
    --orbit-ids=<path>   Symmetry-orbit IDs so a position and its board
                         symmetries stay on the SAME side of every train/test
                         split [default: auto]. auto = resolve from the
                         checkpoint's positions file; use 'none' to disable.
    --output=<path>      Output JSON path [default: auto].
    --dry-run            reclassify: report the changes but do not write.
    --quiet              Suppress the stdout summary.
    -h --help            Show this help.

Inputs (all must be row-aligned, N positions):
    saes/<game>/cache/<run_id>_h.pt          full (N, d_dict) SAE codes
    data/<game>/bsp_labels-<bsps>_<count>.pt (N, num_bsps) binary labels
    data/<game>/bsp_schema-<basis>_<count>.json
Optional:
    saes/<game>/cache/<random_run_id>_h.pt   random-model SAE codes
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from docopt import docopt

from lib.sae.eval import category_families, resolve_schema_path
from lib.sae.dilution import (
    GLOSSARY,
    VERDICT_GLOSSARY,
    DilutionConfig,
    RankingCache,
    classify,
    classify_with_stability,
    diagnose_concept,
)

# categories treated as relational/threat by --categories=auto. Substring match
# on the category name; covers gorilla (threat_*), hawk (reframed_*), tiger
# (*_winnable, *_completing_attr).
AUTO_MARKERS = ("threat", "winnable", "completable", "completing",
                "reframed_count", "reframed_sq_count", "any_threat",
                # hen's count categories, for the same reason as hawk's: the
                # name contains no marker substring, so without these two the
                # basis would silently lose 76 of its 173 BSPs from --auto.
                "neg_count", "neg_sq_count")

VERDICTS = ("absent", "captured", "spread")
# Pre-3A.3 verdicts, still countable when summarising an old report
# that has not been reclassified yet.
LEGACY_VERDICTS = ("diluted", "tiled")


def load_h(cache_dir: Path, run_id: str) -> torch.Tensor:
    path = cache_dir / f"{run_id}_h.pt"
    if not path.exists():
        print(f"Error: SAE code cache not found: {path}", file=sys.stderr)
        print("  Run `sae_eval.py evaluate <ckpt> --bsps=<set>` first to build it.",
              file=sys.stderr)
        sys.exit(1)
    return torch.load(path, map_location="cpu", weights_only=True)


def resolve_labels(data_dir: Path, bsps: str) -> Path:
    matches = sorted(data_dir.glob(f"bsp_labels-{bsps}_[0-9]*.pt"))
    if not matches:
        avail = sorted(p.stem.split("-", 1)[1].rsplit("_", 1)[0]
                       for p in data_dir.glob("bsp_labels-*.pt"))
        print(f"Error: no BSP labels for '{bsps}' in {data_dir}", file=sys.stderr)
        if avail:
            print(f"  Available: {', '.join(avail)}", file=sys.stderr)
        sys.exit(1)
    return matches[0]


def is_auto_category(cat: str) -> bool:
    return any(mark in cat for mark in AUTO_MARKERS)


def summarize(concepts, cat_fam=None, cfg=None):
    """(category_summary, family_summary, gate_g3a) from diagnosed concepts.

    Shared by ``run`` and ``reclassify_report`` so a re-verdict cannot drift
    from a fresh run.

    ``cfg`` enables the rule-3A.4 stability band: each concept's verdict is
    re-tested at +/- ``band_sds`` sd on every quantity ``classify`` thresholds
    on, and the gate reports how many verdicts survive. Without a band the
    tally is a point estimate of a quantity measured to move 12-17% on seed
    alone (docs/diary/2026-08-21_3A-residuals-and-handoff.md §2), so the band
    is part of the result, not a footnote.

    ``cat_fam`` maps category -> concept family (from the schema's stamp). With
    it, the report also carries a CONCEPT-FAMILY rollup. Without that rollup the
    only cross-cell number is ``geometric_frac``, which is computed over a whole
    basis -- and the bases partition the threat menu differently (gorilla 76 =
    40 line + 36 square; hawk 171 = 90 + 81; tiger 23 = 10 + 9 + 4), so
    comparing bases on it compares partitions as much as dictionaries. Same
    error the campaign-K architecture comparison made on 2026-08-15; see
    CLAUDE.md, "never compare two runs on a whole-basis scalar".
    """
    cat_summary = {}
    for c in concepts:
        cat = c["category"]
        d = cat_summary.setdefault(cat, {v: 0 for v in VERDICTS})
        d["n"] = d.get("n", 0) + 1
        d[c["verdict"]] = d.get(c["verdict"], 0) + 1
    for cat, d in cat_summary.items():
        members = [c for c in concepts
                   if c["category"] == cat and "asymptote_r2" in c]
        rec = [c["asymptote_r2"] for c in members]
        d["mean_asymptote_r2"] = round(sum(rec) / len(rec), 4) if rec else 0.0
        solo = [c.get("solo_frac", 0.0) for c in members]
        d["mean_solo_frac"] = round(sum(solo) / len(solo), 4) if solo else 0.0

        # WITHIN-CATEGORY CONSISTENCY. Each BSP in a category is an independent
        # concept with its own association, and a category mean can hide a large
        # split among them (on champYb's anchored SAE the eight row/column
        # line_winnable BSPs sit at |phi| 0.66-0.83 while the two DIAGONALS sit
        # at 0.20 -- a structural blind spot invisible in the 0.60 mean). Report
        # the spread and name the extremes so it cannot hide again.
        phis = sorted(((abs(c["top_phi"]), c["bsp_id"]) for c in members))
        if phis:
            vals = [p for p, _ in phis]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            d["mean_top_phi"] = round(mean, 4)
            d["sd_top_phi"] = round(var ** 0.5, 4)
            d["min_top_phi"] = {"bsp_id": phis[0][1], "top_phi": round(phis[0][0], 4)}
            d["max_top_phi"] = {"bsp_id": phis[-1][1], "top_phi": round(phis[-1][0], 4)}
            # Flag a category whose members disagree enough that the mean is
            # not a fair summary of any of them.
            d["heterogeneous"] = bool(vals[-1] - vals[0] > 0.30)

    # Concept-family rollup: the unit that IS comparable across bases.
    family_summary = {}
    if cat_fam:
        for c in concepts:
            fam = cat_fam.get(c["category"])
            if not fam:
                continue
            d = family_summary.setdefault(fam, {v: 0 for v in VERDICTS})
            d["n"] = d.get("n", 0) + 1
            d[c["verdict"]] = d.get(c["verdict"], 0) + 1
        for fam, d in family_summary.items():
            geo = sum(d.get(v, 0) for v in ("spread", "diluted", "tiled"))
            d["n_spread"] = geo
            d["geometric_frac"] = round(geo / d["n"], 4) if d["n"] else 0.0
            members = [c for c in concepts
                       if cat_fam.get(c["category"]) == fam and "asymptote_r2" in c]
            if members:
                d["mean_asymptote_r2"] = round(
                    sum(c["asymptote_r2"] for c in members) / len(members), 4)
                d["mean_solo_frac"] = round(
                    sum(c.get("solo_frac", 0.0) for c in members) / len(members), 4)

    n_threat = len(concepts)
    # `spread` is the geometric verdict; the retired diluted/tiled labels are
    # still counted here so a not-yet-reclassified report summarises correctly.
    n_geo = sum(1 for c in concepts
                if c["verdict"] in ("spread", "diluted", "tiled"))
    geo_frac = n_geo / n_threat if n_threat else 0.0
    # How many verdicts were actually tested against the LEARNED-signal floor.
    # A gate read on concepts that never met a random-model control is
    # provisional, and this is what says so.
    n_tested = sum(1 for c in concepts if c.get("random_control_applied"))
    gate = {
        "n_threat_bsps": n_threat,
        "n_spread": n_geo,
        "n_captured": sum(1 for c in concepts if c["verdict"] == "captured"),
        "n_absent": sum(1 for c in concepts if c["verdict"] == "absent"),
        "n_random_control_applied": n_tested,
        "random_control_coverage": round(n_tested / n_threat, 4) if n_threat else 0.0,
        "geometric_frac": round(geo_frac, 4),
        "verdict": "3C-proceeds" if geo_frac >= 0.5 else "3C-deprioritized",
        # WHY a cell is deprioritized, because the single geometric_frac
        # threshold collapses two OPPOSITE outcomes into one string:
        #   "captured" -- the concepts are already cleanly recovered, so there
        #                 is nothing for a geometry-aware variant to fix. Good.
        #   "absent"   -- the concepts are not recoverable from this dictionary
        #                 at all (or not above the learned-signal floor), so no
        #                 architecture change on these activations can help.
        # Reading a bare "3C-deprioritized" without this is how a success and a
        # dead end get filed as the same result.
        "deprioritized_because": (
            None if geo_frac >= 0.5
            else "captured"
            if sum(1 for c in concepts if c["verdict"] == "captured")
            >= sum(1 for c in concepts if c["verdict"] == "absent")
            else "absent"
        ),
        # A gate whose learned-signal floor never fired is not a test of it.
        "gate_is_provisional": n_tested < n_threat,
    }

    # --- rule 3A.4: the stability band -------------------------------------
    # Recomputed here rather than read off the concepts, so a `reclassify` with
    # different thresholds re-bands too (the band moves with the boundary).
    if cfg is not None:
        n_undecided = 0
        flips_on = {}
        for c in concepts:
            if c.get("degenerate") or "asymptote_r2" not in c:
                c.setdefault("verdict_stability", "confident")
                c.setdefault("verdict_flips_on", [])
                continue
            _, st = classify_with_stability(c, cfg)
            c["verdict_stability"] = st["stability"]
            c["verdict_flips_on"] = st["flips_on"]
            if st["stability"] == "undecided":
                n_undecided += 1
                for k in st["flips_on"]:
                    flips_on[k] = flips_on.get(k, 0) + 1

        # The band on the GATE quantity: resolve every undecided concept the
        # least- and most-geometric way. If the two ends straddle 0.50 the run
        # does not determine the gate, and saying so is the whole point.
        undecided_geo = sum(
            1 for c in concepts
            if c.get("verdict_stability") == "undecided"
            and c["verdict"] in ("spread", "diluted", "tiled"))
        undecided_non_geo = n_undecided - undecided_geo
        lo = (n_geo - undecided_geo) / n_threat if n_threat else 0.0
        hi = (n_geo + undecided_non_geo) / n_threat if n_threat else 0.0
        gate.update({
            "n_undecided": n_undecided,
            "undecided_frac": round(n_undecided / n_threat, 4) if n_threat else 0.0,
            # WHICH threshold the undecided verdicts rest on. `solo_frac` means
            # the captured boundary; `asymptote_r2` the absent / learned-signal
            # floor. Reported separately because a band on solo_frac alone is
            # blind to the second, which is how ALL FOUR of K04's measured seed
            # flips happened.
            "undecided_flips_on": dict(sorted(flips_on.items())),
            "geometric_frac_lo": round(lo, 4),
            "geometric_frac_hi": round(hi, 4),
            # The gate verdict is only quotable when the band does not straddle
            # the 0.50 boundary.
            "gate_verdict_is_stable": bool((lo >= 0.5) == (hi >= 0.5)),
            "band": {"sds": cfg.band_sds,
                     "solo_frac_seed_sd": cfg.solo_frac_seed_sd},
        })

    return cat_summary, family_summary, gate


def reclassify_report(path: Path, cfg: DilutionConfig, dry_run: bool) -> dict:
    """Re-run the verdict rule over a stored report, in place.

    ``classify`` is a pure function of the per-concept metrics, which the report
    already stores in full -- so a threshold change never needs the SAE code
    caches (4.6-38 GB) or a rerun. Degenerate concepts (all-0 / all-1 labels)
    have no metrics and keep their 'absent' verdict.
    """
    with open(path) as f:
        report = json.load(f)

    changes, concepts = [], report["concepts"]
    for c in concepts:
        if c.get("degenerate") or "asymptote_r2" not in c:
            c.setdefault("random_control_applied", False)
            continue
        # Reports written before rule 3A.3 have no such flag; derive it from
        # whether a control number was actually stored, so the gate's
        # random_control_coverage is honest for legacy reports too.
        c["random_control_applied"] = c.get("random_asymptote_r2") is not None
        old = c["verdict"]
        new = classify(c, cfg)
        if new != old:
            changes.append((c["bsp_id"], old, new))
        c["verdict"] = new

    old_gate = report.get("gate_g3a", {})
    # Resolve the schema so a re-verdict gains the family rollup too, rather
    # than only fresh runs having it.
    cat_fam = {}
    sp = resolve_schema_path(Path("data") / report["summary"].get("game", "quarto"),
                             report["summary"]["bsp_set"])
    if sp:
        with open(sp) as f:
            cat_fam = {k: v["concept_family"]
                       for k, v in category_families(json.load(f)).items()
                       if v.get("concept_family")}
    cat_summary, family_summary, gate = summarize(concepts, cat_fam, cfg)
    report["category_summary"] = cat_summary
    report["family_summary"] = family_summary
    report["gate_g3a"] = gate
    report["summary"]["config"] = cfg.to_dict()
    report["glossary"] = {"metrics": GLOSSARY, "verdicts": VERDICT_GLOSSARY}

    if not dry_run:
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    return {
        "path": str(path),
        "run_id": report["summary"]["run_id"],
        "bsp_set": report["summary"]["bsp_set"],
        "n_changed": len(changes),
        "changes": changes,
        "geometric_frac_before": old_gate.get("geometric_frac"),
        "geometric_frac_after": gate["geometric_frac"],
        "verdict_before": old_gate.get("verdict"),
        "verdict_after": gate["verdict"],
    }


def resolve_orbit_ids(data_dir: Path, spec: str, n_rows: int):
    """(N,) orbit IDs, or None. ``auto`` picks the orbit file whose length
    matches the code cache, so a mismatched dataset can never be applied."""
    if spec == "none":
        return None
    if spec != "auto":
        ids = torch.load(Path(spec), map_location="cpu", weights_only=True).numpy()
    else:
        cands = sorted(data_dir.glob("orbit_ids-*.pt"))
        ids = None
        for c in cands:
            t = torch.load(c, map_location="cpu", weights_only=True)
            if t.shape[0] == n_rows:
                ids = t.numpy()
                print(f"Orbit-aware splits: using {c.name}")
                break
        if ids is None:
            print("Orbit-aware splits: no matching orbit_ids-*.pt for "
                  f"N={n_rows}; falling back to row-wise splits. Generate with "
                  "scripts/compute_orbit_ids.py.", file=sys.stderr)
            return None
    if ids.shape[0] != n_rows:
        print(f"Warning: orbit ids N={ids.shape[0]} != codes N={n_rows}; ignoring.",
              file=sys.stderr)
        return None
    return ids


def run(run_id, game, bsps, categories, random_run_id, cfg, quiet, orbit_spec="auto"):
    cache_dir = Path(f"saes/{game}/cache")
    data_dir = Path(f"data/{game}")

    h = load_h(cache_dir, run_id)
    N, d_dict = h.shape

    labels_path = resolve_labels(data_dir, bsps)
    y_all = torch.load(labels_path, map_location="cpu", weights_only=False)
    if y_all.shape[0] != N:
        print(f"Error: row mismatch h N={N} vs labels N={y_all.shape[0]} "
              f"({labels_path.name}). Codes and labels are not aligned.",
              file=sys.stderr)
        sys.exit(1)

    schema_path = resolve_schema_path(data_dir, bsps)
    if schema_path is None:
        print(f"Error: no BSP schema for '{bsps}' in {data_dir}", file=sys.stderr)
        sys.exit(1)
    with open(schema_path) as f:
        schema = json.load(f)
    bsp_defs = schema["bsps"]
    if len(bsp_defs) != y_all.shape[1]:
        print(f"Error: schema has {len(bsp_defs)} bsps but labels have "
              f"{y_all.shape[1]} columns.", file=sys.stderr)
        sys.exit(1)

    h_random = None
    if random_run_id and random_run_id != "none":
        rpath = cache_dir / f"{random_run_id}_h.pt"
        if rpath.exists():
            h_random = torch.load(rpath, map_location="cpu", weights_only=True)
            if h_random.shape[0] != N:
                print(f"Warning: random control N mismatch; ignoring it.",
                      file=sys.stderr)
                h_random = None
        else:
            print(f"Warning: random control cache not found: {rpath}; "
                  f"using permutation null only.", file=sys.stderr)

    if categories == "auto":
        selected = [i for i, b in enumerate(bsp_defs) if is_auto_category(b["category"])]
    else:
        wanted = {c.strip() for c in categories.split(",") if c.strip()}
        selected = [i for i, b in enumerate(bsp_defs) if b["category"] in wanted]
    if not selected:
        print(f"Error: no BSPs matched categories='{categories}'.", file=sys.stderr)
        sys.exit(1)

    # Stage-A state depends only on the codes, so build it once for the whole
    # run rather than once per concept (an ~18x speedup on a 296k x 4096 cache).
    cache = RankingCache(h, cfg)
    cache_random = RankingCache(h_random, cfg) if h_random is not None else None
    orbit = resolve_orbit_ids(data_dir, orbit_spec, N)

    concepts = []
    for n, idx in enumerate(selected, start=1):
        b = bsp_defs[idx]
        y = y_all[:, idx]
        if float(y.float().mean()) == 0.0 or float(y.float().mean()) == 1.0:
            m = {"verdict": "absent", "base_rate": round(float(y.float().mean()), 4),
                 "degenerate": True}
        else:
            m = diagnose_concept(h, y, cfg, h_random=h_random,
                                 cache=cache, cache_random=cache_random,
                                 orbit_ids=orbit)
        if not quiet and (n % 25 == 0 or n == len(selected)):
            print(f"  ... {n}/{len(selected)} concepts", flush=True)
        concepts.append({"bsp_index": idx, "bsp_id": b["id"],
                         "category": b["category"], **m})

    cat_fam = {k: v["concept_family"]
               for k, v in category_families(schema).items()
               if v.get("concept_family")}
    cat_summary, family_summary, gate = summarize(concepts, cat_fam, cfg)

    result = {
        "summary": {
            "run_id": run_id, "game": game, "bsp_set": bsps,
            "n_samples": int(N), "d_dict": int(d_dict),
            "categories": sorted({c["category"] for c in concepts}),
            "random_control": (random_run_id if h_random is not None else None),
            "orbit_aware_split": orbit is not None,
            "config": cfg.to_dict(),
        },
        "glossary": {"metrics": GLOSSARY, "verdicts": VERDICT_GLOSSARY},
        "gate_g3a": gate,
        "category_summary": cat_summary,
        "family_summary": family_summary,
        "concepts": concepts,
    }

    if not quiet:
        _print_summary(result)
    return result


def _print_summary(result):
    s, g = result["summary"], result["gate_g3a"]
    print("=" * 78)
    print(f"3A dilution diagnostic: {s['run_id']}  bsps={s['bsp_set']}")
    print(f"N={s['n_samples']}  d_dict={s['d_dict']}  "
          f"random_control={s['random_control']}")
    print("=" * 78)
    print(f"{'category':<32}{'n':>3}{'capt':>7}{'spread':>7}"
          f"{'abs':>5}{'meanR2':>8}{'solo':>7}{'|phi|':>7}{'sd':>6}")
    print("-" * 78)
    for cat in sorted(result["category_summary"]):
        d = result["category_summary"][cat]
        spread = (d.get('spread', 0) + d.get('diluted', 0) + d.get('tiled', 0))
        print(f"{cat:<32}{d['n']:>3}{d['captured']:>7}{spread:>7}"
              f"{d['absent']:>5}{d['mean_asymptote_r2']:>8.3f}"
              f"{d.get('mean_solo_frac', 0.0):>7.2f}"
              f"{d.get('mean_top_phi', 0.0):>7.3f}{d.get('sd_top_phi', 0.0):>6.3f}")
    print("-" * 78)
    for cat in sorted(result["category_summary"]):
        d = result["category_summary"][cat]
        if d.get("heterogeneous"):
            lo, hi = d["min_top_phi"], d["max_top_phi"]
            print(f"HETEROGENEOUS {cat}: |phi| {lo['top_phi']:.2f} ({lo['bsp_id']})"
                  f" .. {hi['top_phi']:.2f} ({hi['bsp_id']}) -- the category mean is"
                  f" not representative; read per-BSP.")
    print(f"Gate G-3A: {g.get('n_spread', 0)} spread of {g['n_threat_bsps']} "
          f"threat BSPs -> geometric_frac={g['geometric_frac']:.2f} -> {g['verdict']}")
    if "n_undecided" in g:
        b = g["band"]
        print(f"  band (rule 3A.4, +/-{b['sds']:g} sd): "
              f"geometric_frac in [{g['geometric_frac_lo']:.2f}, "
              f"{g['geometric_frac_hi']:.2f}]  "
              f"{g['n_undecided']}/{g['n_threat_bsps']} undecided "
              f"({g['undecided_frac']*100:.1f}%)"
              + (f"  resting on {g['undecided_flips_on']}"
                 if g["undecided_flips_on"] else ""))
        if not g["gate_verdict_is_stable"]:
            print("  WARNING: the band STRADDLES 0.50 -- this run does not "
                  "determine the gate. Do not quote the verdict.")
    cov = g.get("random_control_coverage")
    if g.get("gate_is_provisional"):
        print(f"!! PROVISIONAL: only {g.get('n_random_control_applied', 0)}/"
              f"{g['n_threat_bsps']} concepts ({cov:.0%}) were tested against a "
              f"random-model control. Without it `absent` only means 'not real', "
              f"never 'not learned' -- an untrained network's dictionary scores "
              f"R2 0.025-0.034 on these concepts.")
    print("meanR2 = mean asymptote_r2 (total recoverable signal); "
          "solo = mean solo_frac")
    print("(full metric glossary is embedded in the output JSON under 'glossary')")


def _config_from_args(args) -> DilutionConfig:
    return DilutionConfig(
        top_k=int(args["--top-k"]),
        captured_solo_frac=float(args["--captured-solo-frac"]),
        captured_idim=float(args["--captured-idim"]),
        band_sds=float(args["--band-sds"]),
        solo_frac_seed_sd=float(args["--solo-frac-seed-sd"]),
    )


def _do_reclassify(args) -> None:
    cfg = _config_from_args(args)
    dry = args["--dry-run"]
    print(f"Reclassify with rule {cfg.rule_version}: "
          f"captured_solo_frac={cfg.captured_solo_frac} "
          f"captured_idim={cfg.captured_idim}"
          f"{'  [DRY RUN -- nothing written]' if dry else ''}")
    print("=" * 78)
    for raw in args["<report>"]:
        path = Path(raw)
        if not path.exists():
            print(f"  [SKIP] not found: {path}", file=sys.stderr)
            continue
        r = reclassify_report(path, cfg, dry)
        before = r["geometric_frac_before"]
        print(f"{r['run_id'][:46]:<46}{r['bsp_set']:<11}"
              f"geom {before if before is None else f'{before:.2f}'}"
              f" -> {r['geometric_frac_after']:.2f}"
              f"  ({r['n_changed']} verdict change(s))"
              f"  {r['verdict_after']}")
        if r["verdict_before"] and r["verdict_before"] != r["verdict_after"]:
            print(f"    !! GATE FLIPPED: {r['verdict_before']} -> {r['verdict_after']}")
    print("=" * 78)
    if dry:
        print("Dry run -- rerun without --dry-run to write the reports.")


def main():
    args = docopt(__doc__)
    game = args["--game"]

    if args["reclassify"]:
        _do_reclassify(args)
        return

    if args["<checkpoint>"]:
        run_id = Path(args["<checkpoint>"]).stem
    else:
        run_id = args["--run-id"]
    if not run_id:
        print("Error: provide <checkpoint> or --run-id.", file=sys.stderr)
        sys.exit(1)
    if not args["--bsps"]:
        print("Error: --bsps=<name> is required.", file=sys.stderr)
        sys.exit(1)

    result = run(
        run_id=run_id, game=game, bsps=args["--bsps"],
        categories=args["--categories"], random_run_id=args["--random-run-id"],
        cfg=_config_from_args(args), quiet=args["--quiet"],
        orbit_spec=args["--orbit-ids"],
    )

    out_dir = Path(f"saes/{game}/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args["--output"] == "auto":
        out_path = out_dir / f"{run_id}_dilution-{args['--bsps']}.json"
    else:
        out_path = Path(args["--output"])
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    if not args["--quiet"]:
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
