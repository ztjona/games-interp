#Requires -Version 7
<#
  runners/3A-dilution.ps1 -- Phase 3A dilution diagnostic over the champTa/champVe
  winner SAEs (NOT champYb; those caches live on Deep Brain). CPU analysis on
  cached SAE codes, so it is sequential (no GPU fan-out needed). Verifies each
  run's inputs are present before invoking, writes one JSON per run plus a
  combined 3A_gate_summary.json, and emits stage_3A-dilution.md.

  Method + thresholds: docs/diary/2026-07-21_3A-dilution-diagnostic.md
  Launch:        runners\launch.ps1 3A-dilution
  Dry-run:       pwsh -File runners\3A-dilution.ps1 -DryRun
#>
param([switch]$DryRun)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Set-Location (Join-Path $PSScriptRoot '..')
$env:PYTHONUTF8 = '1'
New-Item -ItemType Directory -Force -Path logs | Out-Null
Start-Transcript -Path 'logs/3A-dilution.transcript.log' -Append | Out-Null

try {
    $GAME = 'quarto'
    $CACHE = "saes/$GAME/cache"
    $DATA = "data/$GAME"
    $ANALYSIS = "saes/$GAME/analysis"
    $TOPK = 64

    # "run_id|bsps|random_run_id"  (random 'none' -> permutation null)
    $RUNS = @(
        'F04-champTa-s42-jumprelu-t64-exp8-s4.fc1|tigerTa|none',
        'F04-champTa-s42-jumprelu-t64-exp8-s4.fc1|gorillaTa|none',
        'E05-champTa-s42-batchtopk-k32-exp8-s4.conv2|tigerTa|none',
        'I04-champTa-lh100-s43-anchored-jumprelu-t64-exp8-s4.fc1|tigerTa|none',
        'E01-champVe-s42-topk-k32-exp8-s4.conv2|gorillaVe|none',
        'E05-champVe-s42-batchtopk-k32-exp8-s4.conv2|tigerVe|none',
        'I04-champVe-lh100-s42-anchored-jumprelu-t64-exp8-s4.fc1|tigerVe|none',
        'I03-champVe-lh030-s43-anchored-jumprelu-t64-exp8-s4.fc1|gorillaVe|none'
    )

    Write-Host '=================================================================='
    Write-Host 'Phase 3A dilution diagnostic runner'
    Write-Host '=================================================================='

    if (-not (Test-Path lib/sae/dilution.py)) { throw 'lib/sae/dilution.py missing (wrong cwd?)' }
    New-Item -ItemType Directory -Force -Path $ANALYSIS | Out-Null

    function Test-Labels([string]$bsps) {
        @(Get-ChildItem "$DATA/bsp_labels-$($bsps)_*.pt" -ErrorAction SilentlyContinue).Count -gt 0
    }

    Write-Host "`nPre-flight (input presence):"
    $runnable = @()
    foreach ($entry in $RUNS) {
        $rid, $bsps, $rand = $entry.Split('|')
        $why = ''
        if (-not (Test-Path "$CACHE/$($rid)_h.pt")) { $why += ' no _h.pt;' }
        if (-not (Test-Labels $bsps)) { $why += " no bsp_labels-$bsps;" }
        if ($why) { Write-Host "  [SKIP] $rid  bsps=$bsps --$why" }
        else { Write-Host "  [ OK ] $rid  bsps=$bsps"; $runnable += $entry }
    }

    if ($runnable.Count -eq 0) {
        Write-Host "`nNothing runnable here (expected on Speed Mind; caches live on Deep Brain)."
        return
    }
    if ($DryRun) { Write-Host "`nDryRun -> not executing. $($runnable.Count) run(s) would execute."; return }

    Write-Host "`nExecuting $($runnable.Count) run(s)..."
    foreach ($entry in $runnable) {
        $rid, $bsps, $rand = $entry.Split('|')
        Write-Host "`n>> $rid  bsps=$bsps  random=$rand"
        python scripts/dilution_diagnostic.py --run-id=$rid --bsps=$bsps --random-run-id=$rand --top-k=$TOPK
    }

    Write-Host "`nWriting combined gate summary..."
    $agg = @'
import json, sys, glob
out = "saes/quarto/analysis/3A_gate_summary.json"
rows = []
for p in sorted(glob.glob("saes/quarto/analysis/*_dilution-*.json")):
    with open(p) as f: d = json.load(f)
    s, g = d["summary"], d["gate_g3a"]
    rows.append({"run_id": s["run_id"], "bsp_set": s["bsp_set"],
                 "n_threat_bsps": g["n_threat_bsps"], "n_diluted": g["n_diluted"],
                 "n_tiled": g["n_tiled"], "n_captured": g["n_captured"],
                 "n_absent": g["n_absent"], "geometric_frac": g["geometric_frac"],
                 "verdict": g["verdict"]})
json.dump({"runs": rows}, open(out, "w"), indent=2, sort_keys=True)
for r in rows:
    print(f"{r['run_id']:<52}{r['bsp_set']:<11}{r['geometric_frac']:>6.2f}  {r['verdict']}")
print("Saved:", out)
'@
    $agg | python -

    Write-Host "`nEmitting stage plan..."
    $files = @(Get-ChildItem "$ANALYSIS/*_dilution-*.json" | ForEach-Object { $_.FullName })
    $files += "$ANALYSIS/3A_gate_summary.json"
    python scripts/emit_stage.py --slug 3A-dilution @files

    Write-Host "`nDone. Per-run JSON in $ANALYSIS/; commit plan in stage_3A-dilution.md"
}
finally {
    Stop-Transcript | Out-Null
}
