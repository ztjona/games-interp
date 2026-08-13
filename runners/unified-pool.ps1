#Requires -Version 7
<#
  runners/unified-pool.ps1 -- build the cross-champion unified position pool and
  re-evaluate the headline SAEs against it, so cross-champion comparisons are at
  MATCHED BASE RATES.

  Why: every cross-champion number to date is computed on each champion's OWN
  self-play distribution, and those distributions differ in base rate -- champTa
  shows 0.045 on tiger conjunctions vs champVe/champYb's 0.023, largely because
  champTa's dataset is random-play only (see DATA-STATUS.md). MCC, F1 and R2 all
  fall with base rate, so cross-champion numbers are not like-for-like without
  either this shared pool or `coverage_mcc_at_pref` (p_ref = 0.025). The unified pool merges + dedups all champions' positions, then
  recomputes labels and collects each champion's activations on that one pool.
  BSP sets get a numeric suffix encoding the pool size (e.g. gorilla677k);
  the suffix is auto-detected below, never hard-coded.

  Side benefit: the same pass also collects the RANDOM-model activations on the
  unified pool, which is the missing input for the 3A random-network control
  (see runners/3A-dilution.ps1 $RANDOM_CONTROLS) and for the reporting standard's
  learned-gap fraction.

  Cost: GPU-bound. 4 champions x 2 hooks x {trained, random} activation
  collections over the merged pool, then one eval per SAE per BSP set.

  RUN THIS LAST, AND DECIDE WHETHER IT IS STILL WORTH RUNNING AT ALL.

  (a) It DESTROYS the per-champion code caches. `sae_eval` names the cache
      `{run_id}_h.pt` with NO dataset in the name, and stage 3 below re-evaluates
      the 9 panel SAEs with --force against the unified activations. That
      overwrites ~36 GB of per-champion codes that 3A-dilution, basis_comparison
      and backfill_eval_metrics all read -- and 3A-dilution checks only whether
      the file EXISTS, so it would silently consume unified-pool codes against
      per-champion labels. Run 3A-prep, 3A-dilution, the registry backfill and
      basis-verdict FIRST.

  (b) Its original justification is largely gone. The pool existed because
      champTa's base rates were ~2x the other champions'; after the 2026-08-12
      rebuild champTa sits at 0.0255 on tiger conjunctions vs Ve 0.0239 and
      Yb 0.0223, with matched N (290k/290k/296k). What remains is byte-identical
      positions across champions -- but `coverage_mcc_at_pref` already
      standardises prevalence analytically, with no rows discarded and no
      sampling noise.

  (c) It costs ~2.5 h and ~160 GB (9 unified _h caches at ~14 GB each, 16
      activation files, the pool itself). Check free space before starting.

  Launch:   pwsh -File runners\launch.ps1 unified-pool
  Dry-run:  pwsh -File runners\unified-pool.ps1 -DryRun
  Stage 1 only (cheap, no GPU): pwsh -File runners\unified-pool.ps1 -LabelsOnly
#>
param([switch]$DryRun, [switch]$LabelsOnly)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Set-Location (Join-Path $PSScriptRoot '..')
$env:PYTHONUTF8 = '1'
$activate = '.\.venv\Scripts\Activate.ps1'
if (Test-Path $activate) { & $activate } else { throw ".venv not found at $activate" }
New-Item -ItemType Directory -Force -Path logs | Out-Null
Start-Transcript -Path 'logs/unified-pool.transcript.log' -Append | Out-Null

try {
    Write-Host '=================================================================='
    Write-Host 'Unified cross-champion pool: build + matched-base-rate re-eval'
    Write-Host '=================================================================='


    # Dataset integrity gate. champTa's positions were random-play-only and
    # champS4's were champAa's; both went unnoticed for months. Fail fast rather
    # than produce results on a quarantined dataset.
    Write-Host "`nValidating dataset provenance..."
    python scripts/validate_datasets.py
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset validation failed. See data/quarto/DATA-STATUS.md."
    }

    if ($DryRun) {
        python scripts/unify_positions.py --dry-run
        Write-Host "`nDryRun -> stopping before any write."
        return
    }

    # --- Stage 1: merge positions + recompute BSP labels (CPU, minutes) -------
    Write-Host "`n[1/3] Merging positions and recomputing BSP labels..."
    python scripts/unify_positions.py --skip-activations

    # Symmetry-orbit IDs for the merged pool. 3A resolves orbit files by
    # MATCHING N, so the pool needs its own; without it 3A silently falls back to
    # row-wise splits.
    Write-Host "`n[1b/3] Computing symmetry-orbit IDs for the pool..."
    python scripts/compute_orbit_ids.py data/quarto/positions-amalgam_all_unique.pt

    if ($LabelsOnly) {
        Write-Host "`n-LabelsOnly -> stopping before activation collection."
        return
    }

    # --- Stage 2: collect activations on the unified pool (GPU, slow) --------
    # Trained AND random model, both hooks, all champions. Idempotent: the
    # script skips outputs that already exist unless --force is passed.
    Write-Host "`n[2/3] Collecting activations on the unified pool (GPU)..."
    python scripts/unify_positions.py --skip-labels --device cuda

    # Resolve the pool-size suffix from the POOL ITSELF, not by globbing label
    # files. A glob matches every unified label set ever built -- the 677k sets
    # from the pre-rebuild pool are still on disk and are SUPERSEDED, not
    # deleted -- so it would find two and throw here, after all the GPU work in
    # stage 2 was already spent.
    $suffixProbe = @'
import torch
d = torch.load("data/quarto/positions-amalgam_all_unique.pt",
               map_location="cpu", weights_only=False)
print(f"{d['boards'].shape[0] // 1000}k")
'@
    $suffix = (($suffixProbe | python -) -join '').Trim()
    if ($suffix -notmatch '^\d+k$') {
        throw "Could not resolve the unified pool suffix (got '$suffix')."
    }
    Write-Host "  Unified BSP suffix: $suffix"

    # The labels must belong to THIS pool. If the merge was skipped the suffix
    # resolves to the old pool's and everything below would silently score the
    # superseded 677k labels.
    $lblCheck = @(Get-ChildItem "data/quarto/bsp_labels-gorilla${suffix}_*.pt" -ErrorAction SilentlyContinue)
    if (-not $lblCheck) {
        throw "No bsp_labels-gorilla$suffix on disk: stage 1 did not produce labels for this pool."
    }

    # --- Stage 3: re-evaluate the headline SAEs against the unified pool -----
    # "run_id|champTag|hook" -- the same panel 3A diagnoses, so the coverage
    # numbers and the dilution verdicts describe the same positions.
    $RUNS = @(
        'F04-champTa-s42-jumprelu-t64-exp8-s4.fc1|ta|s4.fc1',
        'E05-champTa-s42-batchtopk-k32-exp8-s4.conv2|ta|s4.conv2',
        'I04-champTa-lh100-s43-anchored-jumprelu-t64-exp8-s4.fc1|ta|s4.fc1',
        'F04-champVe-s42-jumprelu-t64-exp8-s4.fc1|ve|s4.fc1',
        'E05-champVe-s42-batchtopk-k32-exp8-s4.conv2|ve|s4.conv2',
        'I04-champVe-lh100-s42-anchored-jumprelu-t64-exp8-s4.fc1|ve|s4.fc1',
        'F04-champYb-s42-jumprelu-t64-exp8-s4.fc1|yb|s4.fc1',
        'E05-champYb-s42-batchtopk-k32-exp8-s4.conv2|yb|s4.conv2',
        'I04-champYb-lh100-s42-anchored-jumprelu-t64-exp8-s4.fc1|yb|s4.fc1'
    )
    $BASES = @('gorilla', 'tiger', 'hawk')

    Write-Host "`n[3/3] Re-evaluating $($RUNS.Count) SAEs x $($BASES.Count) BSP bases on the unified pool..."
    # Sequential on purpose: sae_eval writes the shared eval registry and the _h
    # cache, so parallel runs corrupt them (see CLAUDE.md).
    #
    # ONE call per checkpoint with all bases comma-separated. sae_eval encodes
    # the codes once and reuses them across every BSP set in the same call; a
    # per-basis loop with --force re-encodes the whole pool each time and
    # rewrites the same multi-GB _h file. On this panel that is 27 encodes and
    # ~80 GB of redundant writes instead of 9 encodes (a single conv2 _h on the
    # unified pool is ~5 GB).
    $bspList = ($BASES | ForEach-Object { "$_$suffix" }) -join ','
    foreach ($entry in $RUNS) {
        $rid, $tag, $hook = $entry.Split('|')
        $data = "data/quarto/${hook}_amalgam_all_${tag}_activations.pt"
        if (-not (Test-Path "saes/quarto/$rid.pt")) {
            Write-Host "  [SKIP] no checkpoint: $rid"; continue
        }
        if (-not (Test-Path $data)) {
            Write-Host "  [SKIP] no unified activations: $data"; continue
        }
        Write-Host "`n>> $rid  bsps=$bspList"
        python sae_eval.py evaluate "saes/quarto/$rid.pt" --bsps=$bspList --data=$data --force
    }

    Write-Host "`nEmitting stage plan..."
    python scripts/emit_stage.py --slug unified-pool saes/quarto/eval_registry.json

    Write-Host "`nDone. Compare champions at matched base rates with:"
    Write-Host "  python scripts/registry_query.py top --bsps=tiger$suffix"
    Write-Host "  python scripts/registry_query.py compare <run_a> <run_b> --bsps=tiger$suffix"
}
finally {
    Stop-Transcript | Out-Null
}
