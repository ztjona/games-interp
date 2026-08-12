#Requires -Version 7
<#
  runners/3A-prep.ps1 -- make the 3A panel COMPLETE and CONTROLLED before the
  diagnostic runs. Three independent stages, each skippable.

  Stage 1 -- tiger backfill (43 evals).
      The tiger basis was added on 2026-05-22, after champS4/Ta/Ve had already
      been swept, and was only partly backfilled. gorilla and hawk are complete
      on every champion; champYb is complete on all three. What is missing is
      exactly tiger on S4 (14), Ta (11) and Ve (18) -- including
      F04-champVe-s4.fc1, whose absence is why the first cross-champion read
      could only contrast Ta vs Yb. Manifest: runners/_missing_evals.txt
      (regenerate it with -Rescan).

  Stage 2 -- random-model SAE controls (6 trainings).
      configs/controls/{R1,R2}-champ{Ta,Ve,Yb}random-*.yaml. Each is recipe-
      identical to its unsupervised panel member (F04 fc1 jumprelu t=64 /
      E05 conv2 batchtopk k=32) and differs ONLY in that the activations come
      from the randomly-initialised network. Needed for the reporting standard's
      learned-gap fraction and for a trustworthy 3A 'absent' verdict -- a random
      CNN still yields SAE structure, so the permutation null understates it.

  Stage 3 -- evaluate the controls, which also writes their _h caches so
      3A-dilution.ps1 can use them as the random control.

  Launch:   pwsh -File runners\launch.ps1 3A-prep
  Dry-run:  pwsh -File runners\3A-prep.ps1 -DryRun
  Subset:   pwsh -File runners\3A-prep.ps1 -SkipBackfill
#>
param(
    [switch]$DryRun,
    [switch]$Rescan,
    [switch]$SkipBackfill,
    [switch]$SkipControls
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Set-Location (Join-Path $PSScriptRoot '..')
$env:PYTHONUTF8 = '1'
$activate = '.\.venv\Scripts\Activate.ps1'
if (Test-Path $activate) { & $activate } else { throw ".venv not found at $activate" }
New-Item -ItemType Directory -Force -Path logs | Out-Null
Start-Transcript -Path 'logs/3A-prep.transcript.log' -Append | Out-Null

try {
    $MANIFEST = 'runners/_missing_evals.txt'

    # Dataset integrity gate. This runner re-evaluates 43 (checkpoint x bsp_set)
    # cells; without the gate it would cheerfully produce fresh registry rows for
    # champTa, whose dataset is QUARANTINED (random-play only). Fail fast.
    Write-Host "`nValidating dataset provenance..."
    python scripts/validate_datasets.py
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset validation failed. See data/quarto/DATA-STATUS.md."
    }

    Write-Host '=================================================================='
    Write-Host '3A prep: tiger backfill + random-model controls'
    Write-Host '=================================================================='

    if ($Rescan) {
        Write-Host "`nRescanning the registry for missing (checkpoint x bsp_set) cells..."
        $scan = @'
import json, glob, os, collections
d = json.load(open("saes/quarto/eval_registry.json"))
have = collections.defaultdict(set)
for k, v in d.items():
    if not isinstance(v, dict): continue
    rid = v.get("run_id") or k.split(":")[0]
    have[rid].add(v.get("bsp_set") or (k.split(":")[1] if ":" in k else "gorilla"))
rows = []
for p in sorted(glob.glob("saes/quarto/*.pt")):
    c = os.path.basename(p)[:-3]
    if "champ" not in c: continue
    champ = c.split("champ")[1][:2]
    if champ not in ("S4", "Ta", "Ve", "Yb"): continue
    for basis in ("gorilla", "hawk", "tiger"):
        if f"{basis}{champ}" not in have.get(c, set()):
            rows.append(f"{c}|{basis}{champ}")
open("runners/_missing_evals.txt", "w", newline="").write("\n".join(rows) + "\n")
print(f"  {len(rows)} missing cell(s) written to runners/_missing_evals.txt")
'@
        $scan | python -
    }

    # Champions whose dataset is QUARANTINED or RETIRED. validate_datasets.py
    # alone is NOT enough: it passes when a problem is *declared*, so the gate
    # above would happily let champTa through and mint fresh registry rows from
    # random-play activations.
    #
    # Resolved ONCE, up front, and applied to every stage. It used to gate only
    # the backfill, so stages 2-3 still trained and evaluated
    # R{1,2}-champTarandom-* on champTa's quarantined random-model activations
    # -- two SAE trainings and six evals whose output could not be used, and
    # which would sit in the registry looking usable.
    $excluded = @'
import json
s = json.load(open("data/quarto/_dataset_status.json"))["datasets"]
bad = set()
for v in s.values():
    if v.get("status") in ("QUARANTINED", "RETIRED") and v.get("champion"):
        bad.add(v["champion"])
print(",".join(sorted(bad)))
'@
    $badChamps = (($excluded | python -) -join '').Trim()
    $bad = @($badChamps.Split(',') | Where-Object { $_ })
    if ($bad) {
        Write-Host "`nExcluded champion(s): $($bad -join ', ') -- dataset QUARANTINED/RETIRED."
        Write-Host '  See data/quarto/DATA-STATUS.md. Re-run this runner after the'
        Write-Host '  champion is rebuilt AND retrained, and its status flipped to OK.'
    }

    # --- Stage 1: tiger backfill ------------------------------------------
    if (-not $SkipBackfill) {
        if (-not (Test-Path $MANIFEST)) { throw "$MANIFEST missing (run with -Rescan)" }
        $pairs = @(Get-Content $MANIFEST | Where-Object { $_.Trim() })

        if ($bad) {
            $before = $pairs.Count
            $pairs = @($pairs | Where-Object {
                $c = ($_ -split '\|')[0]
                -not ($bad | Where-Object { $c -like "*champ$_-*" })
            })
            Write-Host "  Backfill: $($before - $pairs.Count) of $before cell(s) skipped (excluded champions)."
        }

        Write-Host "`n[1/3] Tiger backfill: $($pairs.Count) eval(s)."
        # Sequential: sae_eval writes the shared eval registry and the _h cache.
        # Parallel runs corrupt both (see CLAUDE.md).
        $i = 0
        foreach ($pair in $pairs) {
            $i++
            $rid, $bsps = $pair.Split('|')
            $ckpt = "saes/quarto/$rid.pt"
            if (-not (Test-Path $ckpt)) { Write-Host "  [SKIP] no checkpoint: $rid"; continue }
            Write-Host "`n  ($i/$($pairs.Count)) $rid  bsps=$bsps"
            if ($DryRun) { Write-Host "    [DRY] python sae_eval.py evaluate $ckpt --bsps=$bsps"; continue }
            # No --force: these are new (run_id:bsp_set) keys, so nothing is
            # being overwritten and an already-done cell is skipped on a rerun.
            python sae_eval.py evaluate $ckpt --bsps=$bsps
        }
    }
    else { Write-Host "`n[1/3] Tiger backfill SKIPPED." }

    # --- Stage 2: train the random-model controls -------------------------
    if (-not $SkipControls) {
        $cfgs = @(Get-ChildItem 'configs/controls/*.yaml')
        if ($bad) {
            # Same exclusion as stage 1. A control trained on a quarantined
            # champion's random-model activations is not a usable control: it
            # is the null for a distribution we have declared unfit for new
            # work, and it lands in the registry looking like every other row.
            $before = $cfgs.Count
            $cfgs = @($cfgs | Where-Object {
                $n = $_.Name
                -not ($bad | Where-Object { $n -like "*champ$($_)random*" })
            })
            Write-Host "  Controls: $($before - $cfgs.Count) of $before config(s) skipped (excluded champions)."
        }
        if (-not $cfgs) { throw 'No control configs left after exclusions.' }
        Write-Host "`n[2/3] Training $($cfgs.Count) random-model control SAE(s)."
        if ($DryRun) {
            $cfgs | ForEach-Object { Write-Host "    [DRY] python sae_train.py --config=$($_.FullName)" }
        }
        else {
            # Drive the FILTERED config list directly rather than through
            # run_sweep.py --configs=<dir>: run_sweep globs the whole directory,
            # so it would happily train the excluded champion's controls no
            # matter what this runner filtered out. The control set is small
            # (6 configs), so the round-robin here replaces run_sweep entirely.
            # Training is file-disjoint per config, so fan out across all GPUs;
            # only the eval stage below must stay sequential.
            $nGpu = 3
            $root = (Get-Location).Path
            $jobs = 0..($nGpu - 1) | ForEach-Object {
                $gpu = $_
                $mine = @($cfgs | Where-Object { $cfgs.IndexOf($_) % $nGpu -eq $gpu } |
                    ForEach-Object { $_.FullName })
                if (-not $mine) { return }
                Start-Job -ScriptBlock {
                    param($root, $gpu, $mine)
                    Set-Location $root
                    $env:PYTHONUTF8 = '1'
                    $env:CUDA_VISIBLE_DEVICES = "$gpu"
                    foreach ($c in $mine) {
                        Write-Output ">> [gpu $gpu] $(Split-Path $c -Leaf)"
                        & '.\.venv\Scripts\python.exe' sae_train.py --config=$c 2>&1
                    }
                } -ArgumentList $root, $gpu, $mine
            }
            $jobs | Receive-Job -Wait -AutoRemoveJob | ForEach-Object { Write-Host $_ }
        }

        # --- Stage 3: evaluate the controls (writes their _h caches) -------
        Write-Host "`n[3/3] Evaluating the controls (sequential; also writes _h)."
        foreach ($cfg in $cfgs) {
            # champTa/Ve/Yb -> the champion's own BSP label sets
            if ($cfg.Name -notmatch 'champ(\w\w)random') { continue }
            $champ = $Matches[1]
            $stem = @(Get-ChildItem "saes/quarto/*champ$($champ)random*.pt" -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -notlike '*_metrics*' })
            # ONE call, all three bases: sae_eval encodes the codes once and
            # reuses them across every BSP set in the call. A per-basis loop
            # re-encodes and rewrites the same multi-GB _h file each time.
            $bspList = (@('gorilla', 'hawk', 'tiger') | ForEach-Object { "$_$champ" }) -join ','
            foreach ($ck in $stem) {
                Write-Host "`n  >> $($ck.BaseName)  bsps=$bspList"
                if ($DryRun) { Write-Host "    [DRY] sae_eval evaluate --bsps=$bspList"; continue }
                python sae_eval.py evaluate $ck.FullName --bsps=$bspList
            }
        }
    }
    else { Write-Host "`n[2-3/3] Random-model controls SKIPPED." }

    if ($DryRun) { Write-Host "`nDryRun -> nothing executed."; return }

    Write-Host "`nEmitting stage plan..."
    python scripts/emit_stage.py --slug 3A-prep saes/quarto/eval_registry.json saes/quarto/training_registry.json

    Write-Host "`nDone. Next:"
    Write-Host "  1. populate `$RANDOM_CONTROLS in runners/3A-dilution.ps1 with the R1/R2 run_ids"
    Write-Host "  2. pwsh -File runners\launch.ps1 3A-dilution"
}
finally {
    Stop-Transcript | Out-Null
}
