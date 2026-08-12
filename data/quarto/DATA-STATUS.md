# Quarto dataset status

**Authoritative, human-readable status of every position dataset and everything
derived from it.** Machine-readable companion: `_dataset_status.json`.
Enforced by `python scripts/validate_datasets.py` (exit 1 on any undeclared
problem — runners gate on it).

Last reviewed: **2026-08-11**.
Audit that produced this file:
[`docs/diary/2026-08-11_position-dataset-integrity-audit.md`](../../docs/diary/2026-08-11_position-dataset-integrity-audit.md).

## Status legend

| status | meaning |
|---|---|
| **OK** | Correctly built. Safe for new work. |
| **QUARANTINED** | Known wrong. **Do not use for new work or new claims.** Kept only so existing results remain reproducible. |
| **RETIRED** | Champion dropped from the programme. Not wrong per se, but unsupported. |
| **SUPERSEDED** | A corrected rebuild exists or is planned; prefer that. |

## Position datasets

| file | champion | N | status | why |
|---|---|---:|---|---|
| `positions-amalgam_unique.pt` | Aa | 275,916 | **OK** | Correct: all four `Aa_replay` opponent modes. |
| `positions-amalgam_s4_unique.pt` | S4 | 275,916 | **RETIRED** | **Byte-identical to champAa's file.** champS4 was never measured on its own distribution. Its `Sa_archScan` raws exist, so rebuildable — but champS4 is retired. |
| `positions-amalgam_ta_unique.pt` | Ta | 88,524 | **QUARANTINED** | **Built from `random_v_random` alone — zero model-generated positions.** Pure random play, under champAa's game module. |
| `positions-amalgam_ve_unique.pt` | Ve | 289,795 | **OK** | Correct: four `Ve_oracleAblation` modes, `quarto_s4`. |
| `positions-amalgam_yb_unique.pt` | Yb | 296,045 | **OK** | Correct: four `Yb_hotChamp` modes, `quarto_s4_hot`. |
| `positions-amalgam_all_unique.pt` | unified | 677,744 | **SUPERSEDED** | Pool itself is sound, but its *composition* inherits the above: 2 of 5 inputs are champAa's positions and 1 is random-play-only. Rebuild after the champTa fix. |

## Derived artefacts inherit the status of their source

Everything below is computed from the **QUARANTINED** champTa dataset and is
equally unusable for new claims:

- **Labels** — `bsp_labels-{gorillaTa_164, hawkTa_173, tigerTa_36}.pt`
- **Activations** — `s4.{fc1,conv2}_amalgam_ta{,_random}_activations.pt`
- **SAEs** — `saes/quarto/*champTa*.pt`, and their `eval_registry.json` rows
  (keys ending `:gorillaTa`, `:hawkTa`, `:tigerTa`)
- **3A reports** — `saes/quarto/analysis/*champTa*_dilution-*.json`

From the **SUPERSEDED** unified pool:
`bsp_labels-{gorilla677k_164, hawk677k_173, tiger677k_36}.pt`.

## What is safe right now

**champVe and champYb, and comparisons between them.** Those datasets are
correct, and every number resting only on Ve and Yb stands.

## Fixing

```powershell
pwsh -File runners\champTa-rebuild.ps1 -DryRun     # inspect first
pwsh -File runners\champTa-rebuild.ps1 -SkipTrain  # cheap data stages only
pwsh -File runners\launch.ps1 champTa-rebuild      # full, incl. sweep retrain
pwsh -File runners\launch.ps1 unified-pool         # AFTER the above
```

The rebuild moves the quarantined file to `legacy_wrong_distribution/` rather
than deleting it — the existing champTa SAEs and registry rows are only
reproducible against it.

## Other legacy data in this directory

- `legacy_mode2x2_false/` — pre-2026-03-27 data generated with `mode_2x2=False`.
  Provenance only; **never mix with current data**.
- `legacy_wrong_distribution/` — created by `champTa-rebuild.ps1` to hold the
  quarantined champTa dataset once it is replaced.
