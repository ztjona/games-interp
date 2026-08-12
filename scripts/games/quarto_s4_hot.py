"""Quarto game module for the hot-piece-shaped S4 champion (Yb).

This is a thin specialization of :mod:`scripts.games.quarto_s4`. The Yb
champion (``Yb_hotChamp``) uses class
:class:`models.quarto.CNN_autoreg_sa.QuartoCNNAutoregUnifiedS4Hot`, which is a
subclass of ``QuartoCNNAutoregUnifiedS4`` with one extra auxiliary head
(``fc_hot``, depth-1 hot-piece BCE scaffold). That head is a training-time
device only -- it is never read at inference, and the shared trunk (``conv1``,
``conv2``, ``fc1``=512, ``fc2_place``, ``fc2_select``) is byte-identical to the
S4 family. Consequently:

* All hook points are the same as champS4/Ta/Ve: ``s4.conv1``, ``s4.conv2``,
  ``s4.fc1``, ``s4.fc2_place``, ``s4.fc2_select`` (the wrapper stores the model
  as the child attribute ``s4``).
* The 32-d aux contract, the four opponent modes, position generation, and BSP
  delegation are all inherited unchanged from :mod:`scripts.games.quarto_s4`.

The ONLY reason a separate module exists is that the checkpoint carries the
extra ``fc_hot.{weight,bias}`` tensors, so ``load_model`` must instantiate the
Hot subclass; loading into the plain S4 class would raise on the unexpected
keys. Everything else is re-exported from ``quarto_s4`` so behaviour stays
identical and there is a single source of truth for the trunk logic.

See the champYb onboarding diary 2026-06-18.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Re-export the shared machinery from the S4 module so callers (and the
# pipeline) see an identical surface. Bots, the wrapper, and BSP delegation are
# all reused verbatim. NOTE: generate_positions is deliberately NOT re-exported
# here -- it is wrapped below so it loads the Hot checkpoint (see the wrapper).
from . import quarto_s4
from .quarto_s4 import (  # noqa: F401  (re-exported on purpose)
    OPPONENT_MODES,
    RandomBot,
    S4ModelBot,
    S4Wrapper,
    _build_aux32,
    compute_bsp_vector,
    get_all_bsp_definitions,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent


def load_model(model_path: str | Path, device: str = "cpu") -> nn.Module:
    """Load the Yb hot-piece champion and wrap it for the pipeline.

    Identical to :func:`scripts.games.quarto_s4.load_model` except that it
    instantiates :class:`QuartoCNNAutoregUnifiedS4Hot` (which carries the
    auxiliary ``fc_hot`` head present in the Yb checkpoint).
    """
    sys.path.insert(0, str(PROJECT_DIR))
    from models.quarto.CNN_autoreg_sa import QuartoCNNAutoregUnifiedS4Hot

    inner = QuartoCNNAutoregUnifiedS4Hot.from_file(str(model_path))
    inner.device = torch.device(device)  # sync attribute; NN_abstract sets cuda on init
    inner.eval()
    inner.to(device)
    wrapper = S4Wrapper(inner)
    wrapper.eval()
    wrapper.to(device)
    return wrapper


@contextlib.contextmanager
def _use_hot_loader():
    """Temporarily route quarto_s4's module-level ``load_model`` through the Hot
    loader.

    ``quarto_s4.generate_positions`` -> ``_load_shared_model`` -> ``load_model``
    all resolve the bare name ``load_model`` in the *quarto_s4* module globals,
    so simply re-exporting ``generate_positions`` here would still build the
    plain ``QuartoCNNAutoregUnifiedS4`` -- which raises on the Yb checkpoint's
    extra ``fc_hot.*`` keys (strict ``load_state_dict``). Rebinding the global
    for the duration of the call makes the reused generation logic instantiate
    the Hot subclass instead. Restored in ``finally`` so other games are
    unaffected.
    """
    original = quarto_s4.load_model
    quarto_s4.load_model = load_model
    try:
        yield
    finally:
        quarto_s4.load_model = original


def generate_positions(*args, **kwargs):
    """Generate self-play positions for the Yb champion.

    Thin wrapper over :func:`scripts.games.quarto_s4.generate_positions` that
    loads the model via the Hot subclass (see :func:`_use_hot_loader`). Signature
    and output layout are identical to the S4 version.
    """
    with _use_hot_loader():
        return quarto_s4.generate_positions(*args, **kwargs)


# Mirror BSP_SETS and the concept-family map (delegated to quarto).
try:
    from .quarto_s4 import (  # type: ignore  # noqa: F401
        BSP_SETS,
        CONCEPT_FAMILIES,
        FAMILY_ROLE_ORDER,
        concept_family_of,
        concept_triads,
    )
except Exception:  # pragma: no cover
    pass
