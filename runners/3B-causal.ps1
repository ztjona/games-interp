<#
.SYNOPSIS
  Phase 3B-causal, Wave 1: interchange interventions on one champion.

.DESCRIPTION
  Portable by construction: -Champ selects configs/3B-causal/champ<Tag>.yaml,
  and nothing else in the pipeline names a champion. Stages:

    1. prerequisites  -- interchange_3b.py --prereqs. Missing probe
                         directions are produced here (the lines it prints
                         with "RUN: "); anything else missing is an error with
                         the command that produces it.
    2. dry run        -- freeze stamps, generator guard (vectorised concepts ==
                         stored labels), Tier A on the real model, power table.
                         Computes NO interchange score.
    3. the run        -- only with every design file committed and unmodified
                         (--require-frozen): the pre-registration must be frozen
                         before any score exists.
    4. stage plan     -- stage_3B-causal.md, the git-add list.

  Design: docs/diary/2026-09-12_3B-causal-preregistration.md (+ amendments).
  Runtime on champYb (measured on its untrained twin): ~15 s per concept,
  176 concepts, plus replicate dictionaries -- about 55 min on one GPU.

.EXAMPLE
  Dry-run:     pwsh -File runners\3B-causal.ps1 -Champ Yb -DryRun
  Smoke test:  pwsh -File runners\3B-causal.ps1 -Champ Yb -Smoke
  Real run:    pwsh -File runners\launch.ps1 3B-causal -Champ Yb
  Follow:      Get-Content logs\3B-causal.transcript.log -Wait -Tail 40
#>
param(
    [string]$Champ = 'Yb',
    # Stop after the power table; no interchange score is computed.
    [switch]$DryRun,
    # Untrained twin, five representative concepts, output under logs/: proves
    # the pipeline runs and times it without computing a real outcome.
    [switch]$Smoke
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Set-Location (Join-Path $PSScriptRoot '..')
$env:PYTHONUTF8 = '1'
New-Item -ItemType Directory -Force -Path logs | Out-Null
Start-Transcript -Path 'logs/3B-causal.transcript.log' -Append | Out-Null
$runStart = Get-Date
try {
    $cfg = "configs/3B-causal/champ$Champ.yaml"
    if (-not (Test-Path $cfg)) {
        throw "No config $cfg. Copy configs/3B-causal/champYb.yaml and change the names."
    }
    Write-Host "== 3B-causal Wave 1 on champ$Champ ($cfg) =="

    # 1. prerequisites -- the check itself is Python (a tested entry point)
    Write-Host "`n[1] prerequisites"
    $PSNativeCommandUseErrorActionPreference = $false
    $pre = python scripts/interchange_3b.py --config=$cfg --prereqs 2>&1
    $preExit = $LASTEXITCODE
    $PSNativeCommandUseErrorActionPreference = $true
    $pre | ForEach-Object { Write-Host $_ }
    if ($preExit -eq 3) {
        $runs = @($pre | Where-Object { "$_" -like 'RUN: *' } | ForEach-Object { "$_".Substring(5) })
        $other = @($pre | Where-Object { "$_" -like 'MISSING *' -and "$_" -notlike '*probe directions*' })
        if ($other.Count -gt 0) { throw "Missing inputs that this runner cannot produce (see above)." }
        foreach ($cmd in $runs) {
            Write-Host "  producing: $cmd"
            $parts = $cmd -split ' '
            & $parts[0] $parts[1..($parts.Count - 1)]
        }
        python scripts/interchange_3b.py --config=$cfg --prereqs
    } elseif ($preExit -ne 0) {
        throw "Prerequisite check failed (exit $preExit)."
    }

    if ($Smoke) {
        Write-Host "`n[smoke] untrained twin, five concepts, output in logs/3B-causal-smoke"
        python scripts/interchange_3b.py --config=$cfg --untrained --no-replicates `
            --concepts="row_0_completable_tall,row_0_completable_little,tiger_line_row_0_winnable,tiger_offered_completes_tall,tiger_win_now_exists" `
            --out-dir=logs/3B-causal-smoke
        Write-Host ("`nSmoke test done in {0:hh\:mm\:ss}. Its numbers are meaningless by construction." -f ((Get-Date) - $runStart))
        return
    }

    # 2. dry run: guard + Tier A + power, no score
    Write-Host "`n[2] dry run"
    python scripts/interchange_3b.py --config=$cfg --dry-run
    if ($DryRun) {
        Write-Host "`nDryRun -> stopping before any interchange score."
        return
    }

    # 3. the run -- refuses unless every design file is committed and clean
    Write-Host "`n[3] Wave 1 (this is the long part)"
    python scripts/interchange_3b.py --config=$cfg --require-frozen
    Write-Host ("`nWave 1 done in {0:hh\:mm\:ss}." -f ((Get-Date) - $runStart))

    # 4. stage plan
    Write-Host "`n[4] stage plan"
    $champName = "champ$Champ"
    $files = @(Get-ChildItem "saes/quarto/analysis/3B-causal_${champName}_wave1*" | ForEach-Object { $_.FullName })
    $files += @(Get-ChildItem "saes/quarto/analysis/*_topk-*.json" | Where-Object { $_.LastWriteTime -ge $runStart } |
            ForEach-Object { $_.FullName })
    python scripts/emit_stage.py --slug 3B-causal @files
    Write-Host "`nDone. Results in saes/quarto/analysis/3B-causal_${champName}_wave1.json; commit plan in stage_3B-causal.md"
}
finally {
    Stop-Transcript | Out-Null
}
