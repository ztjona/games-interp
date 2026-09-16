<#
.SYNOPSIS
  Phase 3B-causal: interchange interventions on one champion -- Wave 1 on its
  amalgam, or a later wave (1b, 1c) on fresh gold position sets.

.DESCRIPTION
  Portable by construction: -Champ selects configs/3B-causal/champ<Tag>.yaml,
  -Set selects position-set configs champ<Tag>-<set>.yaml (their `rule:` picks
  the wave: 3B.C2 = 1b, 3B.C3 = 1c), and nothing else in the pipeline names a
  champion or a set. Per config:

    1. prerequisites  -- interchange_3b.py --prereqs. Inputs it can produce are
                         produced here (the lines it prints with "RUN: ": probe
                         directions; for a gold set, the whole position set via
                         scripts/build_gold_sets.py -- ~10 min of CPU if absent);
                         anything else missing is an error with the command
                         that produces it.
    2. dry run        -- freeze stamps, generator guard, (Wave 1b: freshness
                         re-check), Tier A on the real model, power table
                         (Wave 1b: feasibility). Computes NO interchange score.
    3. the run        -- only with every design file committed and unmodified
                         (--require-frozen). Wave 1b sets run IN PARALLEL, one
                         per GPU, each logging to logs/3B-causal-<set>.log.
    4. stage plan     -- emit_stage: stage_3B-causal.md only if an output needs git add -f.

  Design: Wave 1  docs/diary/2026-09-12_3B-causal-preregistration.md (+ amendments);
          Wave 1b docs/diary/2026-09-14_3B-causal-wave1b-preregistration.md;
          Wave 1c docs/diary/2026-09-15_3B-causal-wave1c-preregistration.md.
  Runtime: Wave 1 on champYb ~55 min on one GPU; a Wave-1b set ~1 h; a Wave-1c
  set longer (DAS-k and R7-off), ~1.5-2 h -- sets run in parallel.

.EXAMPLE
  Wave 1c dry run:  pwsh -File runners\launch.ps1 3B-causal -Set gold3r2,gold5r2 -DryRun
  Wave 1c run:      pwsh -File runners\launch.ps1 3B-causal -Set gold3r2,gold5r2
  Wave 1b dry run:  pwsh -File runners\launch.ps1 3B-causal -Set gold3,gold5 -DryRun
  Wave 1b smoke:    pwsh -File runners\3B-causal.ps1 -Set gold3 -Smoke
  Wave 1b run:      pwsh -File runners\launch.ps1 3B-causal -Set gold3,gold5
  Wave 1 (pilot):   pwsh -File runners\launch.ps1 3B-causal -Champ Yb
  Follow:           Get-Content logs\3B-causal.transcript.log -Wait -Tail 40
                    Get-Content logs\3B-causal-gold3.log -Wait -Tail 20
#>
param(
    [string]$Champ = 'Yb',
    # Position sets, comma-separated (gold3,gold5 | gold3r2,gold5r2). Empty = Wave 1.
    [string]$Set = '',
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
# Activate the venv explicitly (a detached launch does not inherit it). Without
# this, `python` under launch.ps1's WMI spawn is the system interpreter, which
# has torch but no numpy -- the first 3B-causal launch died on exactly that.
$activate = '.\.venv\Scripts\Activate.ps1'
if (Test-Path $activate) { & $activate } else { throw ".venv not found at $activate" }
New-Item -ItemType Directory -Force -Path logs | Out-Null
Start-Transcript -Path 'logs/3B-causal.transcript.log' -Append | Out-Null
$runStart = Get-Date
try {
    $sets = @($Set -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    $names = if ($sets.Count) { @($sets | ForEach-Object { "champ$Champ-$_" }) } else { @("champ$Champ") }
    foreach ($n in $names) {
        if (-not (Test-Path "configs/3B-causal/$n.yaml")) {
            throw "No config configs/3B-causal/$n.yaml. Copy champYb.yaml (or champYb-gold3.yaml) and change the names."
        }
    }
    $wave = if ($sets.Count) { "sets $($sets -join ',')" } else { 'Wave 1' }
    Write-Host "== 3B-causal $wave on $($names -join ', ') =="

    # 1. prerequisites -- the check itself is Python (a tested entry point)
    foreach ($n in $names) {
        $cfg = "configs/3B-causal/$n.yaml"
        Write-Host "`n[1] prerequisites: $cfg"
        $PSNativeCommandUseErrorActionPreference = $false
        $pre = python scripts/interchange_3b.py --config=$cfg --prereqs 2>&1
        $preExit = $LASTEXITCODE
        $PSNativeCommandUseErrorActionPreference = $true
        $pre | ForEach-Object { Write-Host $_ }
        if ($preExit -eq 4) {
            throw "Missing inputs that this runner cannot produce (see above)."
        } elseif ($preExit -eq 3) {
            $runs = @($pre | Where-Object { "$_" -like 'RUN: *' } | ForEach-Object { "$_".Substring(5) } |
                    Select-Object -Unique)
            foreach ($cmd in $runs) {
                Write-Host "  producing: $cmd"
                $parts = $cmd -split ' '
                & $parts[0] $parts[1..($parts.Count - 1)]
            }
            python scripts/interchange_3b.py --config=$cfg --prereqs
        } elseif ($preExit -ne 0) {
            throw "Prerequisite check failed (exit $preExit)."
        }
    }

    if ($Smoke) {
        foreach ($n in $names) {
            Write-Host "`n[smoke] $n : untrained twin, five concepts, output in logs/3B-causal-smoke"
            python scripts/interchange_3b.py --config="configs/3B-causal/$n.yaml" --untrained --no-replicates `
                --concepts="row_0_completable_tall,row_0_completable_little,tiger_line_row_0_winnable,tiger_offered_completes_tall,tiger_win_now_exists" `
                --out-dir=logs/3B-causal-smoke
        }
        Write-Host ("`nSmoke test done in {0:hh\:mm\:ss}. Its numbers are meaningless by construction." -f ((Get-Date) - $runStart))
        return
    }

    # 2. dry run: guard (+ freshness) + Tier A + power (+ feasibility), no score
    foreach ($n in $names) {
        Write-Host "`n[2] dry run: $n"
        python scripts/interchange_3b.py --config="configs/3B-causal/$n.yaml" --dry-run
    }
    if ($DryRun) {
        Write-Host "`nDryRun -> stopping before any interchange score."
        return
    }

    # 3. the run -- refuses unless every design file is committed and clean
    if ($sets.Count) {
        Write-Host "`n[3] $wave, $($names.Count) set(s) in parallel, one GPU each (the long part)"
        $procs = @()
        for ($i = 0; $i -lt $names.Count; $i++) {
            $n = $names[$i]
            $env:CUDA_VISIBLE_DEVICES = "$($i % 3)"
            Write-Host "  $n on GPU $($i % 3) -> logs/3B-causal-$($sets[$i]).log"
            $procs += Start-Process -FilePath python -PassThru -WindowStyle Hidden `
                -ArgumentList @('scripts/interchange_3b.py', "--config=configs/3B-causal/$n.yaml", '--require-frozen') `
                -RedirectStandardOutput "logs/3B-causal-$($sets[$i]).log" `
                -RedirectStandardError "logs/3B-causal-$($sets[$i]).err"
        }
        Remove-Item Env:CUDA_VISIBLE_DEVICES
        foreach ($p in $procs) { $p.WaitForExit() }
        $bad = @($procs | Where-Object { $_.ExitCode -ne 0 })
        if ($bad.Count -gt 0) { throw "$($bad.Count) set(s) failed; see logs/3B-causal-<set>.log and .err" }
    } else {
        Write-Host "`n[3] $wave (this is the long part)"
        python scripts/interchange_3b.py --config="configs/3B-causal/$($names[0]).yaml" --require-frozen
    }
    Write-Host ("`n$wave done in {0:hh\:mm\:ss}." -f ((Get-Date) - $runStart))

    # 4. stage plan
    Write-Host "`n[4] stage plan"
    $files = @()
    foreach ($n in $names) {
        $files += @(Get-ChildItem "saes/quarto/analysis/3B-causal_${n}_wave1*" | ForEach-Object { $_.FullName })
    }
    $files += @(Get-ChildItem "saes/quarto/analysis/*_topk-*.json" | Where-Object { $_.LastWriteTime -ge $runStart } |
            ForEach-Object { $_.FullName })
    python scripts/emit_stage.py --slug 3B-causal @files
    Write-Host "`nDone. Results in saes/quarto/analysis/3B-causal_<name>_wave1*.json; commit what git status lists"
}
finally {
    Stop-Transcript | Out-Null
}
