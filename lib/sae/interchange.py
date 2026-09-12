"""Interchange interventions at a hook -- the game-agnostic core of Phase 3B-causal.

An interchange intervention runs the network on a *base* input but overwrites
one representation of a concept C with the value it takes on a *source* input.
If that representation IS the network's variable for C, the network now acts as
if C had the source's value (causal abstraction; Geiger et al.). The patched
value comes from a real input, so there is no steering dose to choose.

Everything here operates on plain tensors. Game knowledge -- which inputs are
paired, which actions are legal, which actions are the target -- enters only as
tensors built by a game module, exactly as ``lib/sae/eval.py`` and
``lib/sae/dilution.py`` do. The frozen design this implements:
``docs/diary/2026-09-12_3B-causal-preregistration.md`` as amended, pre-data, by
``docs/diary/2026-09-12_3B-causal-amendment-1.md`` (pre-ReLU hook, Tier-A
controls A1a/A1b/A2/A3, the ordered verdict rule).

Two things about the hook matter and are handled here, not left to callers:

* **Hooks are PRE-activations.** Every model in this repo applies ReLU
  functionally after the hooked module, so a hook value ``z`` is pre-ReLU and
  the next layer reads ``relu(z)``. A readout therefore always applies the
  network's own nonlinearity to a patched ``z``; there is no "clipping" choice.
* **Two readout paths, one interface.** :class:`LinearReluReadout` is the exact
  closed form when the network downstream of the hook is ``relu`` then one
  linear head (the S4 ``fc1`` case). :class:`ForwardHookReadout` works for ANY
  architecture and hook: it replaces the hooked module's output and runs the
  real model. Where both exist they must agree (control A3) -- which is also
  what makes the pipeline portable to a new champion whose downstream map is
  not linear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

# Version of the verdict rule defined in the pre-registration (S8). Stored in
# every report so a verdict can never be read against the wrong rule.
RULE_VERSION = "3B.C1"

# Pre-registered constants (pre-registration S8). Frozen: changing any of them
# needs a dated amendment entry, not an edit here.
IIA_STAR_FLOOR = 0.20
BH_Q = 0.05
N_MIN_SWITCH_ON = 100
N_MIN_SPECIFICITY = 100
N_MIN_SWITCH_OFF = 50

# Machine-readable definitions travel with every report (reporting standard 6).
GLOSSARY: dict[str, dict[str, str]] = {
    "IIA": {"range": "[0, 1], ideal 1",
            "meaning": "share of pairs whose patched legal decision lands in the target set"},
    "r0": {"range": "[0, 1]",
           "meaning": "share of pairs whose UNPATCHED base decision already lands in the target set"},
    "IIA_star": {"range": "(-inf, 1], ideal 1, 0 = no effect",
                 "meaning": "chance-corrected IIA = (IIA - r0) / (1 - r0)"},
    "flip_rate": {"range": "[0, 1]",
                  "meaning": "share of pairs whose legal decision changes at all under the patch"},
    "target_margin": {"range": "logit units, ideal > 0",
                      "meaning": "change in the best target logit minus the mean change of the other legal actions"},
    "verdict": {"range": "see VERDICTS",
                "meaning": f"rule {RULE_VERSION}; pre-registration S8"},
}

VERDICTS = (
    "concept-consistent",           # installs AND removes C, specifically
    "concept-consistent (on-only)",  # installs C specifically; removal underpowered
    "install-only",                  # installs C specifically; removal measured and fails
    "remove-only",                   # removal works; installation does not
    "context-blind",                 # installs C, but also where C is absent
    "anti-consistent",               # pushes AWAY from the target
    "off-target",                    # changes decisions, not toward the target
    "inert",                         # none of the above
    "underpowered",                  # too few pairs for any verdict
)


# ---------------------------------------------------------------------------
# Readouts: hook value z -> action logits
# ---------------------------------------------------------------------------


class Readout:
    """Maps a hook value ``z`` (B, d) to action logits (B, A).

    ``inputs`` are the base inputs that produced ``z``. The closed form ignores
    them; the forward-hook path needs them to drive the real model.
    """

    kind: str = "abstract"

    def __call__(self, z: torch.Tensor, inputs: Sequence[torch.Tensor] | None = None
                 ) -> torch.Tensor:
        raise NotImplementedError


class LinearReluReadout(Readout):
    """Exact closed form for ``logits = relu(z) @ W.T + b``.

    This is the whole network downstream of the S4 ``fc1`` hook for one head
    (``fc2_place`` or ``fc2_select``); the output tanh is monotone, so it
    changes no decision and is left out. Differentiable (used by DAS).
    """

    kind = "closed_form"

    def __init__(self, W: torch.Tensor, b: torch.Tensor):
        self.W = W.detach()
        self.b = b.detach()

    def head(self, h: torch.Tensor) -> torch.Tensor:
        """The linear head alone, on a POST-ReLU value (control A1a)."""
        return h @ self.W.T + self.b

    def __call__(self, z, inputs=None):
        return self.head(torch.relu(z))


class ForwardHookReadout(Readout):
    """Generic path: replace a module's output with ``z``, run the real model.

    Works for any architecture and any hooked module. ``run`` maps the base
    inputs to the model call; the logits are read from ``logit_module``'s
    output (e.g. the ``fc2_place`` Linear -- its output is pre-tanh).

    The hooked module must be the one the activations were collected from, so
    that "replace its output" means exactly "set the hook value to ``z``"; the
    model then applies whatever follows it (the functional ReLU, dropout in
    eval mode, the heads) itself.
    """

    kind = "forward_hook"

    def __init__(self, model: torch.nn.Module, hook_module: torch.nn.Module,
                 logit_module: torch.nn.Module,
                 run: Callable[[Sequence[torch.Tensor]], object]):
        self.model = model
        self.hook_module = hook_module
        self.logit_module = logit_module
        self.run = run

    def __call__(self, z, inputs=None):
        if inputs is None:
            raise ValueError("ForwardHookReadout needs the base inputs")
        captured: dict[str, torch.Tensor] = {}

        def replace(_m, _inp, out):
            if out.shape != z.shape:
                raise ValueError(
                    f"hook output {tuple(out.shape)} != patched z {tuple(z.shape)}; "
                    "flattened (conv) hooks need a readout that reshapes")
            return z.to(out.dtype)

        def grab(_m, _inp, out):
            captured["logits"] = out

        h1 = self.hook_module.register_forward_hook(replace)
        h2 = self.logit_module.register_forward_hook(grab)
        try:
            self.run(inputs)
        finally:
            h1.remove()
            h2.remove()
        return captured["logits"]


def resolve_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """``dict(model.named_modules())[name]`` with a helpful error."""
    mods = dict(model.named_modules())
    if name not in mods:
        raise KeyError(f"no module {name!r}; have e.g. {sorted(mods)[:12]}")
    return mods[name]


# ---------------------------------------------------------------------------
# Patch operations -- all on the PRE-activation hook value z
# ---------------------------------------------------------------------------


def patch_full(z_b: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
    """Ceiling: the whole hook value from the source (control A2)."""
    return z_s.clone()


def patch_subspace(z_b: torch.Tensor, z_s: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Replace the component of ``z_b`` in span(U) with the source's.

    ``U`` is (d, k); it is orthonormalised here, so callers may pass any basis.
    """
    Q, _ = torch.linalg.qr(U)
    return z_b + ((z_s - z_b) @ Q) @ Q.T


def patch_direction(z_b: torch.Tensor, z_s: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """1-D interchange along ``w`` (LP direction, DAS-1, random null)."""
    w = w / w.norm().clamp(min=1e-12)
    return z_b + ((z_s - z_b) @ w).unsqueeze(-1) * w


def patch_latents(z_b: torch.Tensor, z_s: torch.Tensor,
                  encode: Callable[[torch.Tensor], torch.Tensor],
                  W_dec: torch.Tensor, J: Sequence[int] | torch.Tensor) -> torch.Tensor:
    """Error-preserving SAE interchange of latent set ``J``.

    ``z_b + sum_{j in J} (a_j(s) - a_j(b)) * W_dec[j]``: only the chosen
    latents' contribution to the reconstruction changes. The rest of the
    reconstruction AND the SAE error term are untouched, so reconstruction error
    cannot masquerade as an effect. ``W_dec`` is (d_dict, d).
    """
    J = torch.as_tensor(J, dtype=torch.long, device=z_b.device)
    if J.numel() == 0:
        return z_b.clone()
    a_b = encode(z_b)[:, J]
    a_s = encode(z_s)[:, J]
    return z_b + (a_s - a_b) @ W_dec[J]


def check_encoder_is_per_sample(encode: Callable[[torch.Tensor], torch.Tensor],
                                z: torch.Tensor, atol: float = 1e-5) -> None:
    """Raise if ``encode`` depends on batch composition.

    Base and source are encoded in different batches, so a batch-level encoder
    (BatchTopK in train mode) would make a latent's value depend on its
    neighbours and the interchange meaningless. JumpReLU / TopK are per-sample.
    """
    whole = encode(z)
    singles = torch.cat([encode(z[i:i + 1]) for i in range(min(len(z), 16))])
    if not torch.allclose(whole[: len(singles)], singles, atol=atol):
        raise ValueError("SAE encoder is batch-dependent (BatchTopK in train "
                         "mode?); interchange needs a per-sample encoder")


# ---------------------------------------------------------------------------
# Decisions and scores -- always over the LEGAL set
# ---------------------------------------------------------------------------


def legal_argmax(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Argmax over legal actions only. ``legal`` is (B, A) bool, >=1 True per row.

    Q on illegal actions is never a training target (pre-registration S2), so
    it is excluded before the argmax, never interpreted.
    """
    if not bool(legal.any(dim=1).all()):
        raise ValueError("every pair needs at least one legal action")
    return logits.masked_fill(~legal, float("-inf")).argmax(dim=1)


def hits(decisions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(B,) bool: decision lands in the per-pair target set ``target`` (B, A)."""
    return target[torch.arange(len(decisions), device=target.device), decisions]


def chance_corrected(iia: float, r0: float) -> float | None:
    """IIA* = (IIA - r0) / (1 - r0); None when r0 == 1 (nothing to gain)."""
    if r0 >= 1.0:
        return None
    return (iia - r0) / (1.0 - r0)


def target_margin(logits_base: torch.Tensor, logits_patched: torch.Tensor,
                  target: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Per pair: change of the best target logit minus the mean change of the
    other legal actions. Continuous companion to IIA; shows PARTIAL effects."""
    d = logits_patched - logits_base
    neg = float("-inf")
    tgt = (d.masked_fill(~(target & legal), neg)).max(dim=1).values
    other = legal & ~target
    n_other = other.sum(dim=1).clamp(min=1)
    rest = (d * other).sum(dim=1) / n_other
    return tgt - rest


@dataclass
class PairScores:
    """Scores of one representation on one pair set."""

    n: int
    iia: float
    r0: float
    iia_star: float | None
    flip_rate: float
    mean_target_margin: float
    hit_patched: torch.Tensor    # (n,) bool, kept for bootstrap / per-pair records
    hit_base: torch.Tensor       # (n,) bool
    flipped: torch.Tensor        # (n,) bool


def score_pairs(logits_base: torch.Tensor, logits_patched: torch.Tensor,
                target: torch.Tensor, legal: torch.Tensor) -> PairScores:
    d_base = legal_argmax(logits_base, legal)
    d_pat = legal_argmax(logits_patched, legal)
    hb, hp = hits(d_base, target), hits(d_pat, target)
    n = len(d_base)
    iia = float(hp.float().mean()) if n else float("nan")
    r0 = float(hb.float().mean()) if n else float("nan")
    return PairScores(
        n=n, iia=iia, r0=r0,
        iia_star=chance_corrected(iia, r0) if n else None,
        flip_rate=float((d_pat != d_base).float().mean()) if n else float("nan"),
        mean_target_margin=float(target_margin(logits_base, logits_patched,
                                               target, legal).mean()) if n else float("nan"),
        hit_patched=hp, hit_base=hb, flipped=d_pat != d_base,
    )


# ---------------------------------------------------------------------------
# Null distributions and statistics
# ---------------------------------------------------------------------------


def random_unit_directions(n: int, d: int, generator: torch.Generator) -> torch.Tensor:
    v = torch.randn(n, d, generator=generator)
    return v / v.norm(dim=1, keepdim=True)


def frequency_matched_sets(J: Sequence[int], freq: torch.Tensor, alive: torch.Tensor,
                           n_draws: int, generator: torch.Generator,
                           tol: float = 0.20) -> list[torch.Tensor]:
    """Random latent sets the size of ``J``, each member matched in firing
    frequency to one member of ``J`` within ``±tol`` (relative), drawn from
    alive latents outside ``J``. Falls back to the nearest-frequency alive
    latents when a tolerance band is empty, so a draw always exists.
    """
    J = [int(j) for j in J]
    pool = torch.nonzero(alive, as_tuple=False).flatten()
    pool = pool[~torch.isin(pool, torch.tensor(J, dtype=torch.long))]
    if pool.numel() == 0:
        raise ValueError("no alive latents outside J")
    candidates = []
    for j in J:
        f = float(freq[j])
        band = pool[(freq[pool] >= f * (1 - tol)) & (freq[pool] <= f * (1 + tol))]
        if band.numel() == 0:
            band = pool[(freq[pool] - f).abs().argsort()[:8]]
        candidates.append(band)
    draws = []
    for _ in range(n_draws):
        pick = [c[torch.randint(len(c), (1,), generator=generator)].item()
                for c in candidates]
        draws.append(torch.tensor(pick, dtype=torch.long))
    return draws


def empirical_p(observed: float, null: torch.Tensor, tail: str = "greater"
                ) -> tuple[float, bool]:
    """(p, parametric_tail). Empirical ``(k + 1) / (n + 1)``; when the observed
    value lies beyond every draw, a Gaussian fit to the null supplies the tail
    (pre-registration S8) and the flag is set."""
    null = null[torch.isfinite(null)]
    n = null.numel()
    if n == 0 or observed is None or not math.isfinite(observed):
        return float("nan"), False
    if tail == "greater":
        k = int((null >= observed).sum())
    elif tail == "less":
        k = int((null <= observed).sum())
    else:
        raise ValueError(tail)
    if k > 0:
        return (k + 1) / (n + 1), False
    mu, sd = float(null.mean()), float(null.std())
    if sd == 0.0:
        return 1.0 / (n + 1), True
    zscore = (observed - mu) / sd if tail == "greater" else (mu - observed) / sd
    return 0.5 * math.erfc(zscore / math.sqrt(2.0)), True


def benjamini_hochberg(pvals: Sequence[float], q: float = BH_Q) -> list[bool]:
    """BH step-up. NaN p-values are never rejected and do not count toward m."""
    idx = [i for i, p in enumerate(pvals) if p is not None and math.isfinite(p)]
    m = len(idx)
    reject = [False] * len(pvals)
    if m == 0:
        return reject
    order = sorted(idx, key=lambda i: pvals[i])
    k_max = 0
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= rank * q / m:
            k_max = rank
    for i in order[:k_max]:
        reject[i] = True
    return reject


def orbit_folds(groups: torch.Tensor, n_folds: int, seed: int) -> torch.Tensor:
    """Fold id per pair, grouped so no group (board orbit) spans two folds."""
    uniq = torch.unique(groups)
    g = torch.Generator().manual_seed(seed)
    perm = uniq[torch.randperm(len(uniq), generator=g)]
    fold_of = {int(u): i % n_folds for i, u in enumerate(perm)}
    return torch.tensor([fold_of[int(x)] for x in groups], dtype=torch.long)


def bootstrap_iia_star(hit_patched: torch.Tensor, hit_base: torch.Tensor,
                       groups: torch.Tensor, n_boot: int = 1000, seed: int = 0,
                       level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of IIA*, resampling GROUPS (board orbits), not pairs."""
    g = torch.Generator().manual_seed(seed)
    uniq, inv = torch.unique(groups, return_inverse=True)
    G = len(uniq)
    hp = torch.zeros(G).index_add_(0, inv, hit_patched.float())
    hb = torch.zeros(G).index_add_(0, inv, hit_base.float())
    cnt = torch.zeros(G).index_add_(0, inv, torch.ones(len(inv)))
    stats = []
    for _ in range(n_boot):
        pick = torch.randint(G, (G,), generator=g)
        n = cnt[pick].sum()
        iia, r0 = float(hp[pick].sum() / n), float(hb[pick].sum() / n)
        s = chance_corrected(iia, r0)
        if s is not None:
            stats.append(s)
    if not stats:
        return float("nan"), float("nan")
    t = torch.tensor(stats)
    a = (1 - level) / 2
    return float(t.quantile(a)), float(t.quantile(1 - a))


# ---------------------------------------------------------------------------
# DAS-1: the single causal direction, learned (positive control + ceiling)
# ---------------------------------------------------------------------------


def train_das_direction(z_b: torch.Tensor, z_s: torch.Tensor, readout: Readout,
                        target: torch.Tensor, legal: torch.Tensor,
                        inputs: Sequence[torch.Tensor] | None = None,
                        steps: int = 300, lr: float = 0.05, seed: int = 0,
                        init: torch.Tensor | None = None) -> torch.Tensor:
    """Unit vector ``w`` maximising the probability that the patched decision
    lands in the target set: cross-entropy over the legal set, toward the
    target set (log-sum-exp over targets minus over legal actions).

    Train it on training folds only and score it held out -- it is selected on
    the outcome, so an in-sample score is meaningless.
    """
    g = torch.Generator().manual_seed(seed)
    v = (init.clone() if init is not None
         else torch.randn(z_b.shape[1], generator=g)).to(z_b.dtype).requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)
    neg = torch.finfo(z_b.dtype).min / 4
    for _ in range(steps):
        w = v / v.norm().clamp(min=1e-12)
        logits = readout(patch_direction(z_b, z_s, w), inputs)
        legal_l = logits.masked_fill(~legal, neg)
        tgt_l = logits.masked_fill(~(target & legal), neg)
        loss = (torch.logsumexp(legal_l, 1) - torch.logsumexp(tgt_l, 1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (v / v.norm()).detach()


# ---------------------------------------------------------------------------
# Verdict rule 3B.C1
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    """One pair type (switch-on / specificity / switch-off) for one
    (representation, concept). ``significant`` is the BH decision, made by the
    caller across concepts within (wave, representation)."""

    n: int
    iia_star: float | None
    significant: bool = False
    below_null_p5: bool = False       # switch-on only: anti-consistent test
    flip_significant: bool = False    # switch-on only: off-target test


def classify(on: ArmResult, spec: ArmResult, off: ArmResult | None) -> str:
    """Verdict rule 3B.C1: pre-registration S8 as amended (pre-data) by
    amendment 1, section A3, which names the install-only / remove-only
    asymmetries and fixes the order in which the rules apply."""
    if on.n < N_MIN_SWITCH_ON or spec.n < N_MIN_SPECIFICITY:
        return "underpowered"
    off_powered = off is not None and off.n >= N_MIN_SWITCH_OFF
    installs = (on.iia_star is not None and on.iia_star >= IIA_STAR_FLOOR
                and on.significant)
    removes = (off_powered and off.iia_star is not None
               and off.iia_star >= IIA_STAR_FLOOR and off.significant)
    if installs and spec.significant:
        return "context-blind"
    if installs:
        if not off_powered:
            return "concept-consistent (on-only)"
        return "concept-consistent" if removes else "install-only"
    if removes:
        return "remove-only"
    if on.below_null_p5:
        return "anti-consistent"
    if on.flip_significant:
        return "off-target"
    return "inert"
