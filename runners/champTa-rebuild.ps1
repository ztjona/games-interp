#Requires -Version 7
<#
  runners/champTa-rebuild.ps1 -- rebuild champTa's position dataset on its OWN
  distribution, then everything downstream.

  WHY: positions-amalgam_ta_unique.pt (88,524) was aggregated from
  positions-random_v_random_raw.pt ALONE. It contains zero model-generated
  positions -- it is pure random-vs-random play, generated under champAa's game
  module. Every champTa SAE was therefore trained on random-play activations,
  and every *Ta BSP label tensor scores that distribution. champVe and champYb
  are correct. Audit: docs/diary/2026-08-11_position-dataset-integrity-audit.md

  The raw data was never missing -- champTa's three self-play files exist and
  were simply never aggregated (263,393 raw positions, 10k games each). So this
  runner regenerates NOTHING; it re-runs the aggregation and everything below it.

  champS4 has the same defect (its file is byte-identical to champAa's) and is
  RETIRED rather than rebuilt -- see the audit note.

  Stages: 1 aggregate+dedup, 2 orbit ids, 3 BSP labels, 4 activations (GPU),
          5 retrain the champTa SAE sweep (GPU, the expensive step), 6 evaluate,
          7 clear the QUARANTINE.

  RUN THIS IN FULL. champTa's 43 sweep configs are recipe-identical to champVe's
  and champYb's, so a full retrain is what puts all three champions on equal
  footing -- which is the entire point of the rebuild. -SkipTrain is a
  validation escape hatch only: it leaves the champTa SAEs trained on the OLD
  random-play activations while the labels and activations around them are new,
  so every champTa number it produces describes a dictionary learned from random
  play. Stage 7 will not clear the quarantine on a -SkipTrain run.

  Launch:   pwsh -File runners\launch.ps1 champTa-rebuild
  Dry-run:  pwsh -File runners\champTa-rebuild.ps1 -DryRun
  Data only (validation escape hatch, see above): -SkipTrain
#>
param([switch]$DryRun, [switch]$SkipTrain)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Set-Location (Join-Path $PSScriptRoot '..')
$env:PYTHONUTF8 = '1'
$activate = '.\.venv\Scripts\Activate.ps1'
if (Test-Path $activate) { & $activate } else { throw ".venv not found at $activate" }
New-Item -ItemType Directory -Force -Path logs | Out-Null
Start-Transcript -Path 'logs/champTa-rebuild.transcript.log' -Append | Out-Null

try {
    $MODEL = 'models/quarto/20260517_1620-Ta_minimaxSelect(3)0515_MINIMAX_SELECT_E_10000.pt'
    $RANDOM_MODEL = ''   # resolved from configs/models/champTa.yaml below
    $GAME = 'quarto_s4'
    $OUT = 'data/quarto/positions-amalgam_ta_unique.pt'
    $BACKUP = 'data/quarto/legacy_wrong_distribution/positions-amalgam_ta_unique.pt'

    $RAWS = @(
        'data/quarto/positions-random_v_random_raw.pt',
        'data/quarto_s4/positions-model_v_random-Ta_minimaxSelect_raw.pt',
        'data/quarto_s4/positions-random_v_model-Ta_minimaxSelect_raw.pt',
        'data/quarto_s4/positions-model_v_model-Ta_minimaxSelect_raw.pt'
    )

    Write-Host '=================================================================='
    Write-Host 'champTa rebuild: correct the position distribution'
    Write-Host '=================================================================='

    Write-Host "`nPre-flight:"
    $ok = $true
    foreach ($r in $RAWS) {
        if (Test-Path $r) { Write-Host "  [ OK ] $r" }
        else { Write-Host "  [MISS] $r"; $ok = $false }
    }
    if (-not (Test-Path "configs/models/champTa.yaml")) {
        Write-Host '  [MISS] configs/models/champTa.yaml'; $ok = $false
    }
    if (-not $ok) { throw 'Missing inputs; nothing done.' }

    if ($DryRun) {
        Write-Host "`nDryRun -> would aggregate $($RAWS.Count) raw files into $OUT,"
        Write-Host 'then orbit ids -> BSP labels -> activations -> retrain -> eval.'
        return
    }

    # --- Stage 1: aggregate + dedup ---------------------------------------
    # Preserve the wrong dataset rather than deleting it: the existing champTa
    # SAEs and eval-registry rows were computed on it, so it is the only way to
    # reproduce those (now provisional) numbers.
    Write-Host "`n[1/6] Aggregating + deduplicating (dedup AFTER aggregation)..."
    if (Test-Path $OUT) {
        New-Item -ItemType Directory -Force -Path (Split-Path $BACKUP) | Out-Null
        Write-Host "  Preserving the old (random-play) dataset -> $BACKUP"
        Move-Item -Path $OUT -Destination $BACKUP -Force
    }
    python scripts/deduplicate_positions.py @RAWS --output $OUT

    # --- Stage 2: orbit ids ------------------------------------------------
    Write-Host "`n[2/6] Computing symmetry-orbit IDs (for orbit-aware 3A splits)..."
    python scripts/compute_orbit_ids.py $OUT

    # Confirm the rebuild actually corrected the provenance before spending any
    # GPU time. Deliberately NOT run at the start: champTa fails validation by
    # construction until this runner replaces its dataset.
    Write-Host "`n[2b/6] Re-validating dataset provenance..."
    python scripts/validate_datasets.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'NOTE: champTa should now pass. If it still fails, remove its'
        Write-Host 'QUARANTINED entry from data/quarto/_dataset_status.json only'
        Write-Host 'once the provenance check is genuinely clean.'
    }

    # --- Stage 3: BSP labels ----------------------------------------------
    Write-Host "`n[3/6] Recomputing BSP labels on the corrected distribution..."
    foreach ($name in @('gorilla', 'hawk', 'tiger')) {
        Write-Host "  -> $name"
        python scripts/compute_bsp_labels.py $OUT --game $GAME --name $name
    }

    # --- Stage 4: activations ---------------------------------------------
    Write-Host "`n[4/6] Collecting activations (trained + random model, both hooks)..."
    $resolve = @'
import sys, yaml
cfg = yaml.safe_load(open("configs/models/champTa.yaml"))
print(cfg.get("model_path", ""))
print(cfg.get("random_model_path", "") or cfg.get("random_model", ""))
'@
    $paths = @($resolve | python -)
    $MODEL = $paths[0]
    $RANDOM_MODEL = $paths[1]
    Write-Host "  trained: $MODEL"
    Write-Host "  random : $RANDOM_MODEL"

    foreach ($hook in @('s4.fc1', 's4.conv2')) {
        # conv2 is (B,C,H,W) and must be flattened for SAE training; fc1 is not.
        $flat = if ($hook -eq 's4.conv2') { '--flatten-position' } else { '' }
        foreach ($pair in @(@($MODEL, ''), @($RANDOM_MODEL, '_random'))) {
            $mdl, $sfx = $pair
            if (-not $mdl) { Write-Host "  [SKIP] no model for suffix '$sfx'"; continue }
            $out = "data/quarto/${hook}_amalgam_ta${sfx}_activations.pt"
            Write-Host "`n  >> $hook$sfx -> $out"
            python scripts/collect_activations.py $mdl --hook $hook --game $GAME `
                --positions-file $OUT --output $out --device cuda $flat
        }
    }

    if ($SkipTrain) {
        Write-Host "`n-SkipTrain -> stopping before the sweep retrain."
        Write-Host 'The champTa SAEs on disk are still trained on the OLD (random-play)'
        Write-Host 'activations and must NOT be compared against the new labels.'
        Write-Host 'champTa stays QUARANTINED, so every downstream runner will keep'
        Write-Host 'excluding it. Re-run WITHOUT -SkipTrain to finish the rebuild.'
        return
    }

    # --- Stage 5: retrain the champTa sweep -------------------------------
    Write-Host "`n[5/6] Retraining the champTa SAE sweep (GPU fan-out)..."
    if (-not (Test-Path 'configs/champTa')) { throw 'configs/champTa not found' }
    $nGpu = 3
    $jobs = 0..($nGpu - 1) | ForEach-Object {
        $gpu = $_
        Start-Job -ScriptBlock {
            param($root, $gpu, $n)
            Set-Location $root
            $env:PYTHONUTF8 = '1'
            & '.\.venv\Scripts\python.exe' run_sweep.py --configs=configs/champTa `
                --gpu=$gpu --split="$($gpu + 1)/$n" 2>&1
        } -ArgumentList (Get-Location).Path, $gpu, $nGpu
    }
    $jobs | Receive-Job -Wait -AutoRemoveJob | ForEach-Object { Write-Host $_ }

    # --- Stage 6: evaluate -------------------------------------------------
    # Sequential: sae_eval writes the shared registry and the _h cache.
    #
    # ONE call per checkpoint with all three bases comma-separated: sae_eval
    # encodes the codes once and reuses them across every BSP set in the call.
    # A per-basis loop with --force re-encodes the full dataset three times and
    # rewrites the same multi-GB _h file (1.4-4.8 GB each here), i.e. 132
    # encodes across this sweep instead of 44.
    Write-Host "`n[6/6] Evaluating champTa against all three bases..."
    $ckpts = @(Get-ChildItem 'saes/quarto/*champTa*.pt' | Where-Object { $_.Name -notlike '*_metrics*' })
    foreach ($ck in $ckpts) {
        Write-Host "`n  >> $($ck.BaseName)  bsps=gorillaTa,hawkTa,tigerTa"
        python sae_eval.py evaluate $ck.FullName --bsps=gorillaTa,hawkTa,tigerTa --force
    }

    # --- Stage 7: clear the quarantine ------------------------------------
    # Every downstream runner (3A-prep, basis-verdict, unified-pool) reads
    # _dataset_status.json and SKIPS quarantined champions. If this is left to a
    # human, the rest of the chain silently drops champTa and the whole rebuild
    # looks like it did nothing. Only done on a full run, and only after the
    # provenance check passes cleanly -- the flip is a claim about the data, so
    # it is gated on the evidence rather than on this script having reached the
    # end.
    Write-Host "`n[7/7] Clearing the champTa quarantine..."
    python scripts/validate_datasets.py
    if ($LASTEXITCODE -ne 0) {
        throw 'Provenance still fails after the rebuild; quarantine NOT cleared.'
    }
    $clear = @'
import json, pathlib
p = pathlib.Path("data/quarto/_dataset_status.json")
doc = json.loads(p.read_text(encoding="utf-8"))
e = doc["datasets"]["positions-amalgam_ta_unique.pt"]
if e.get("status") == "OK":
    print("  already OK; nothing to do.")
else:
    e.clear()
    e.update({
        "champion": "Ta",
        "status": "OK",
        "note": ("Rebuilt 2026-08-12 by runners/champTa-rebuild.ps1 from all four "
                 "Ta_minimaxSelect opponent modes under quarto_s4; SAE sweep "
                 "retrained on the corrected activations. Prior random-play-only "
                 "dataset preserved at legacy_wrong_distribution/. Audit: "
                 "docs/diary/2026-08-11_position-dataset-integrity-audit.md"),
    })
    doc["derived_artifacts_quarantined"].pop("labels", None)
    doc["derived_artifacts_quarantined"].pop("activations", None)
    doc["derived_artifacts_quarantined"].pop("saes", None)
    doc["derived_artifacts_quarantined"].pop("analysis", None)
    doc["updated"] = "2026-08-12"
    p.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print("  champTa -> OK. Downstream runners will now include it.")
'@
    $clear | python -

    Write-Host "`nEmitting stage plan..."
    python scripts/emit_stage.py --slug champTa-rebuild `
        saes/quarto/eval_registry.json saes/quarto/training_registry.json `
        data/quarto/_dataset_status.json

    Write-Host "`nDone. champTa is now on its own distribution, retrained and un-quarantined."
    Write-Host 'Next: rebuild the unified pool (runners/unified-pool.ps1), then 3A-prep.'
}
finally {
    Stop-Transcript | Out-Null
}
