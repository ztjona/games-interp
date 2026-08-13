# legacy_wrong_distribution/

`positions-amalgam_ta_unique.pt` — champTa's **original, defective** position
dataset: 88,524 positions deduplicated from `positions-random_v_random_raw.pt`
alone, i.e. pure random-vs-random play with zero model-generated positions.
Kept so the champTa eval-registry rows computed on it stay reproducible.
See [`../DATA-STATUS.md`](../DATA-STATUS.md) and
[`../../../docs/diary/2026-08-11_position-dataset-integrity-audit.md`](../../../docs/diary/2026-08-11_position-dataset-integrity-audit.md).

**This file is a RECONSTRUCTION (2026-08-12).** The original was overwritten by
`Move-Item -Force` during the champTa-rebuild retries on 2026-08-12 — each
failed attempt moved that run's *correct* rebuild on top of it. Regenerated
with:

```bash
python scripts/deduplicate_positions.py data/quarto/positions-random_v_random_raw.pt \
    --output data/quarto/legacy_wrong_distribution/positions-amalgam_ta_unique.pt
```

Dedup is deterministic and the raw input is unchanged since May, so the row
count reproduces exactly (88,524). Only `provenance.deduplication_date` differs
from the original. `champTa-rebuild.ps1` now refuses to overwrite an existing
backup.
