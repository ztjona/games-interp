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
        # Capture then Write-Host -- stdout from `python -` is not transcribed.
        ($scan | python -) | ForEach-Object { Write-Host $_ }
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

    # --- Disk pre-flight ---------------------------------------------------
    # Every eval writes an (N x d_dict) float32 _h cache. At champVe's 289,795
    # positions that is 4.4 GB per exp8 run and 35 GB per exp64 run, so this
    # stage can want ~170 GB. Running out mid-way leaves a truncated cache that
    # later loads as a corrupt tensor, so size it up front and refuse early.
    $needGb = @'
import glob, json, os, sys, torch
pairs = [l.strip().split("|") for l in open("runners/_missing_evals.txt") if l.strip()]
status = json.load(open("data/quarto/_dataset_status.json"))["datasets"]
bad = {v["champion"] for v in status.values()
       if v.get("status") in ("QUARANTINED", "RETIRED") and v.get("champion")}
need = 0.0
for rid, bsps in pairs:
    if any(f"champ{c}-" in rid for c in bad):
        continue
    if os.path.exists(f"saes/quarto/cache/{rid}_h.pt"):
        continue          # reused, not rewritten
    ck = f"saes/quarto/{rid}.pt"
    lbl = sorted(glob.glob(f"data/quarto/bsp_labels-{bsps}_[0-9]*.pt"))
    if not os.path.exists(ck) or not lbl:
        continue
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict") or sd
    d = sd["W_enc"].shape[1]
    n = torch.load(lbl[0], map_location="cpu", weights_only=True).shape[0]
    need += n * d * 4 / 2**30
print(f"{need:.1f}")
'@
    $needStage1 = [double](($needGb | python -) -join '').Trim()
    $freeGb = [math]::Round((Get-PSDrive (Get-Location).Drive.Name).Free / 1GB, 1)
    # +25 GB headroom for the six control _h caches written by stage 3.
    $wantGb = [math]::Round($needStage1 + 25, 1)
    Write-Host "`nDisk pre-flight: ~$wantGb GB needed (stage 1 _h $needStage1 GB + ~25 GB controls); $freeGb GB free."
    if ($freeGb -lt $wantGb) {
        Write-Host ''
        Write-Host 'NOT ENOUGH DISK. _h is a regenerable cache, so the cheapest'
        Write-Host 'reclaim is the champTa caches nothing downstream reads --'
        Write-Host 'champTa registry rows already carry J and MCC@pref, so the'
        Write-Host 'backfill never re-reads them, and 3A/basis-verdict need only'
        Write-Host 'F04 / E05 / I04. See the 2026-08-13 diary entry for the'
        Write-Host 'exact prune command (frees ~274 GB).'
        throw "Insufficient disk: need ~$wantGb GB, have $freeGb GB."
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
        $bfStart = Get-Date
        foreach ($pair in $pairs) {
            $i++
            $rid, $bsps = $pair.Split('|')
            $ckpt = "saes/quarto/$rid.pt"
            if (-not (Test-Path $ckpt)) { Write-Host "  [SKIP] no checkpoint: $rid"; continue }
            $el = (Get-Date) - $bfStart
            $eta = if ($i -gt 1) {
                '~{0:hh\:mm}' -f [TimeSpan]::FromSeconds(
                    ($el.TotalSeconds / ($i - 1)) * ($pairs.Count - $i + 1))
            }
            else { 'unknown' }
            Write-Host ("`n  ({0}/{1}) {2}  bsps={3} | elapsed {4:hh\:mm} | ETA {5}" -f `
                    $i, $pairs.Count, $rid, $bsps, $el, $eta)
            if ($DryRun) { Write-Host "    [DRY] python sae_eval.py evaluate $ckpt --bsps=$bsps"; continue }
            # No --force: these are new (run_id:bsp_set) keys, so nothing is
            # being overwritten and an already-done cell is skipped on a rerun.
            python sae_eval.py evaluate $ckpt --bsps=$bsps
        }
        if (-not $DryRun) {
            Write-Host ("  Backfill done in {0:hh\:mm\:ss}." -f ((Get-Date) - $bfStart))
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
                Start-Job -Name "gpu$gpu" -ScriptBlock {
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

            # Drain incrementally, tagged by GPU. `Receive-Job -Wait` buffers
            # each job's whole output and flushes it only when that job ends, so
            # the transcript replays each GPU's timestamps from the start after
            # the previous one finishes -- time appears to run backwards and
            # every block looks like the end of the run.
            $trStart = Get-Date
            while (@($jobs | Where-Object { $_.State -eq 'Running' }).Count -gt 0) {
                foreach ($jb in $jobs) {
                    Receive-Job -Job $jb | ForEach-Object {
                        if ("$_".Trim()) { Write-Host "  [$($jb.Name)] $_" }
                    }
                }
                Start-Sleep -Seconds 5
            }
            foreach ($jb in $jobs) {
                Receive-Job -Job $jb | ForEach-Object {
                    if ("$_".Trim()) { Write-Host "  [$($jb.Name)] $_" }
                }
            }
            $failed = @($jobs | Where-Object { $_.State -eq 'Failed' } | ForEach-Object Name)
            $jobs | Remove-Job -Force
            if ($failed) { throw "Control training job(s) FAILED: $($failed -join ', ')" }
            Write-Host ("  Training done in {0:hh\:mm\:ss}." -f ((Get-Date) - $trStart))
        }

        # --- Stage 3: evaluate the controls (writes their _h caches) -------
        # Iterate CHAMPIONS, not configs. Each champion has two configs (R1 fc1
        # + R2 conv2) but the glob below already returns both of that champion's
        # checkpoints, so looping over configs evaluated every champion twice.
        $champs = @($cfgs | ForEach-Object {
                if ($_.Name -match 'champ(\w\w)random') { $Matches[1] } } | Select-Object -Unique)
        Write-Host "`n[3/3] Evaluating controls for $($champs.Count) champion(s) (sequential; also writes _h)."
        foreach ($champ in $champs) {
            $stem = @(Get-ChildItem "saes/quarto/*champ$($champ)random*.pt" -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -notlike '*_metrics*' })
            if (-not $stem) { Write-Host "  [SKIP] champ$champ -- no control checkpoint on disk"; continue }
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
    # Verify rather than instruct: the run_ids are already in 3A-dilution.ps1,
    # so what matters is whether each one now has a checkpoint AND an _h cache.
    # A control whose _h is missing does not fail -- the CLI silently falls back
    # to the permutation null, which UNDERSTATES what "absent" should mean.
    $ctlCheck = @'
import os, re
txt = open("runners/3A-dilution.ps1", encoding="utf-8").read()
ids = re.findall(r"=\s*'(R[12]-champ\w+random-[\w.\-]+)'", txt)
missing = [r for r in ids if not os.path.exists(f"saes/quarto/cache/{r}_h.pt")]
print(f"  {len(ids) - len(missing)}/{len(ids)} random-model controls have an _h cache.")
for r in missing:
    print(f"    NO _h (will fall back to the permutation null): {r}")
'@
    # Capture then Write-Host -- stdout from `python -` is not transcribed.
    ($ctlCheck | python -) | ForEach-Object { Write-Host $_ }
    Write-Host "  1. review the control coverage line above"
    Write-Host "  2. pwsh -File runners\launch.ps1 3A-dilution"
}
finally {
    Stop-Transcript | Out-Null
}
