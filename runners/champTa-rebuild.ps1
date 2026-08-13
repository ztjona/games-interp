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
    $StartedAt = Get-Date
    $MODEL = ''          # both resolved from configs/models/champTa.yaml below
    $RANDOM_MODEL = ''
    $GAME = 'quarto_s4'
    $CHAMP = 'Ta'        # champion suffix for the PER-DISTRIBUTION label sets
    $TAG = 'ta'          # lower-case tag used in activation filenames
    $CONFIG = 'configs/models/champTa.yaml'
    $OUT = 'data/quarto/positions-amalgam_ta_unique.pt'
    $BACKUP = 'data/quarto/legacy_wrong_distribution/positions-amalgam_ta_unique.pt'
    $N_POSITIONS = 0     # filled in after stage 1; every later stage checks it

    $RAWS = @(
        'data/quarto/positions-random_v_random_raw.pt',
        'data/quarto_s4/positions-model_v_random-Ta_minimaxSelect_raw.pt',
        'data/quarto_s4/positions-random_v_model-Ta_minimaxSelect_raw.pt',
        'data/quarto_s4/positions-model_v_model-Ta_minimaxSelect_raw.pt'
    )

    # One banner per stage, with wall-clock and elapsed. The transcript is read
    # top-to-bottom hours later (and tailed live), so every line that could be
    # mistaken for "finished" must say which stage it belongs to and what is
    # still ahead.
    $STAGES = @(
        'aggregate + dedup', 'orbit ids', 'BSP labels', 'activations (GPU)',
        'retrain sweep (GPU, ~2.5 h)', 'evaluate', 'clear quarantine')
    function Write-Stage([int]$n) {
        $el = (Get-Date) - $script:StartedAt
        Write-Host ''
        Write-Host ('=' * 70)
        Write-Host ("[{0}/{1}] {2}   (started {3:HH:mm:ss}, elapsed {4:hh\:mm\:ss})" -f `
                $n, $STAGES.Count, $STAGES[$n - 1], (Get-Date), $el)
        # Guard the range: PowerShell REVERSES $a[7..6] rather than returning
        # empty, so the last stage used to advertise itself as still remaining.
        if ($n -lt $STAGES.Count) {
            Write-Host ("        remaining: " + ($STAGES[$n..($STAGES.Count - 1)] -join ' -> '))
        }
        else { Write-Host '        remaining: (none -- final stage)' }
        Write-Host ('=' * 70)
    }

    Write-Host '=================================================================='
    Write-Host 'champTa rebuild: correct the position distribution'
    Write-Host "  $($STAGES.Count) stages: $($STAGES -join ' -> ')"
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
    Write-Stage 1
    if (Test-Path $OUT) {
        New-Item -ItemType Directory -Force -Path (Split-Path $BACKUP) | Out-Null
        if (Test-Path $BACKUP) {
            # NEVER overwrite an existing backup. The point of it is to preserve
            # the ORIGINAL random-play dataset so the old (now provisional)
            # champTa registry rows stay reproducible. On the 2026-08-12 retries
            # `Move-Item -Force` overwrote it with each successive REBUILD, so
            # after two failed attempts the "backup of the wrong dataset" was
            # itself a correct dataset and the original was gone.
            Write-Host "  Backup already exists ($BACKUP) -- keeping it."
            Write-Host '  Discarding the current file instead.'
            Remove-Item -Path $OUT -Force
        }
        else {
            Write-Host "  Preserving the old (random-play) dataset -> $BACKUP"
            Move-Item -Path $OUT -Destination $BACKUP
        }
    }
    # --game is EXPLICIT. deduplicate_positions.py otherwise auto-detects it from
    # the FIRST source file's provenance -- which here is champAa's shared
    # random_v_random file (champTa has none of its own; only Ve and Yb do), so
    # the merged dataset would claim game='quarto' and fail validate_datasets
    # check 3. The positions themselves are fine: random-vs-random involves no
    # model, so that file is legitimately reusable across champions.
    python scripts/deduplicate_positions.py @RAWS --output $OUT --game $GAME

    # The row count every later stage is checked against. Read from the file
    # rather than parsed from stdout, so it cannot drift from what is on disk.
    $probe = @"
import torch
d = torch.load(r"$OUT", map_location="cpu", weights_only=False)
print(d["boards"].shape[0])
"@
    $N_POSITIONS = [int](($probe | python -) -join '').Trim()
    Write-Host "  Rebuilt dataset: $N_POSITIONS positions."

    # --- Stage 2: orbit ids ------------------------------------------------
    Write-Stage 2
    python scripts/compute_orbit_ids.py $OUT

    # Confirm the rebuild actually corrected the provenance before spending any
    # GPU time. Deliberately NOT run at the start: champTa fails validation by
    # construction until this runner replaces its dataset.
    Write-Host "`n  Re-validating dataset provenance..."
    python scripts/validate_datasets.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'NOTE: champTa should now pass. If it still fails, remove its'
        Write-Host 'QUARANTINED entry from data/quarto/_dataset_status.json only'
        Write-Host 'once the provenance check is genuinely clean.'
    }

    # --- Stage 3: BSP labels ----------------------------------------------
    # --name MUST carry the champion suffix. Labels are PER-DISTRIBUTION and are
    # keyed by the suffixed animal (bsp_labels-gorillaTa_164.pt); only the
    # SCHEMA is basis-keyed. Passing the bare basis here wrote champTa's labels
    # over champAa's un-suffixed bsp_labels-gorilla_164.pt / -hawk_173.pt (the
    # baseline champion is the un-tagged one), destroyed them, and left
    # champTa's own *Ta files untouched and stale. See the 2026-08-12 incident
    # note in docs/diary/.
    $bases = @('gorilla', 'hawk', 'tiger')
    Write-Stage 3
    Write-Host "  $($bases.Count) bases on the corrected distribution."
    $i = 0
    foreach ($basis in $bases) {
        $i++
        $animal = "$basis$CHAMP"
        Write-Host "  ($i/$($bases.Count)) -> $animal"
        python scripts/compute_bsp_labels.py $OUT --game $GAME --name $animal

        # Verify, do not assume: the label file must exist AND have one row per
        # position. A silent mismatch here is what made every champTa number
        # meaningless for months.
        $lbl = @(Get-ChildItem "data/quarto/bsp_labels-${animal}_[0-9]*.pt" -ErrorAction SilentlyContinue)
        if (-not $lbl) { throw "Stage 3 produced no bsp_labels-$animal file." }
        $rows = @"
import sys, torch
t = torch.load(r"$($lbl[0].FullName)", map_location="cpu", weights_only=True)
sys.exit(0 if t.shape[0] == $N_POSITIONS else 1)
"@
        $rows | python -
        if ($LASTEXITCODE -ne 0) {
            throw "bsp_labels-$animal row count != $N_POSITIONS positions."
        }
        Write-Host "      OK: $($lbl[0].Name) matches $N_POSITIONS positions."
    }

    # --- Stage 4: activations ---------------------------------------------
    # The champion YAML keys are `path` and `random_path` (see any champ*.yaml,
    # and scripts/unify_positions.py which reads them correctly). This block used
    # to ask for `model_path` / `random_model_path`, got two empty strings, and
    # SKIPPED all four collections -- non-fatally. Stage 5 then retrained the
    # whole sweep on the OLD activations while the run reported success.
    $hooks = @('s4.fc1', 's4.conv2')
    Write-Stage 4
    Write-Host "  $($hooks.Count) hook(s) x {trained, random}."
    $resolve = @"
import sys, yaml
cfg = yaml.safe_load(open(r"$CONFIG"))
for key in ("path", "random_path"):
    v = cfg.get(key)
    if not v:
        sys.exit(f"champTa.yaml has no '{key}'")
    print(v)
"@
    $paths = @($resolve | python -)
    if ($paths.Count -lt 2) { throw "Could not resolve model paths from $CONFIG." }
    $MODEL, $RANDOM_MODEL = $paths[0], $paths[1]
    foreach ($m in @($MODEL, $RANDOM_MODEL)) {
        if (-not (Test-Path $m)) { throw "Model file not found: $m" }
    }
    Write-Host "  trained: $MODEL"
    Write-Host "  random : $RANDOM_MODEL"

    $j = 0; $nJobs = $hooks.Count * 2
    foreach ($hook in $hooks) {
        foreach ($pair in @(@($MODEL, ''), @($RANDOM_MODEL, '_random'))) {
            $j++
            $mdl, $sfx = $pair
            # NOT $out. PowerShell variable names are CASE-INSENSITIVE, so `$out`
            # and `$OUT` are one variable: assigning here silently overwrote
            # $OUT (the positions file) with the activations path, and
            # collect_activations.py was then handed its own output file as
            # --positions-file ("too many indices for tensor of dimension 2").
            $actOut = "data/quarto/${hook}_amalgam_${TAG}${sfx}_activations.pt"
            Write-Host "`n  ($j/$nJobs) $hook$sfx -> $actOut"

            # Build the argument list and SPLAT it. A conditional
            # `$flat = ... else { '' }` interpolated into the command line does
            # NOT vanish the way an unquoted empty variable does in bash --
            # PowerShell passes '' as a real, empty argument, docopt reads it as
            # a stray positional, and collect_activations.py exits 1 with its
            # usage block. (That was invisible until stage 4 started running at
            # all: the earlier empty-model bug skipped this line entirely.)
            $actArgs = @(
                'scripts/collect_activations.py', $mdl,
                '--hook', $hook, '--game', $GAME,
                '--positions-file', $OUT, '--output', $actOut, '--device', 'cuda')
            # conv2 is (B,C,H,W) and must be flattened for SAE training; fc1 is not.
            if ($hook -eq 's4.conv2') { $actArgs += '--flatten-position' }
            python @actArgs

            # Same rule as stage 3: verify, do not assume. Training on an
            # activation file whose row count does not match the positions is
            # the failure this whole rebuild exists to undo.
            if (-not (Test-Path $actOut)) { throw "Stage 4 produced no $actOut" }
            $chk = @"
import sys, torch
o = torch.load(r"$actOut", map_location="cpu", weights_only=False)
t = o if torch.is_tensor(o) else o.get("activations")
sys.exit(0 if t is not None and t.shape[0] == $N_POSITIONS else 1)
"@
            $chk | python -
            if ($LASTEXITCODE -ne 0) {
                throw "$actOut row count != $N_POSITIONS positions."
            }
            Write-Host "      OK: matches $N_POSITIONS positions."
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
    $nCfg = @(Get-ChildItem 'configs/champTa/*.yaml').Count
    Write-Stage 5
    Write-Host "  $nCfg config(s) across 3 GPUs."
    Write-Host '      Expect ~2.5 h. Progress is printed below every 60s as'
    Write-Host '      checkpoints land; per-GPU detail is in logs/sweep_gpu*.log.'
    if (-not (Test-Path 'configs/champTa')) { throw 'configs/champTa not found' }
    $nGpu = 3
    $jobs = 0..($nGpu - 1) | ForEach-Object {
        $gpu = $_
        Start-Job -Name "gpu$gpu" -ScriptBlock {
            param($root, $gpu, $n)
            Set-Location $root
            $env:PYTHONUTF8 = '1'
            & '.\.venv\Scripts\python.exe' run_sweep.py --configs=configs/champTa `
                --gpu=$gpu --split="$($gpu + 1)/$n" 2>&1
        } -ArgumentList (Get-Location).Path, $gpu, $nGpu
    }

    # Drain the jobs INCREMENTALLY. `Receive-Job -Wait` buffers each job's whole
    # output and flushes it only when that job ends, so the transcript showed
    # GPU1's full 13:57->15:25 block, then jumped BACK to 13:57 for GPU2, then
    # again for GPU0. Timestamps ran backwards and each block looked like the
    # run had finished while two GPUs were still working -- which is exactly the
    # "it said it finished but kept going" confusion. Tagging each line with its
    # GPU and draining on a timer keeps the log chronological and honest.
    $sweepStart = Get-Date
    $lastBeat = Get-Date
    while (@($jobs | Where-Object { $_.State -eq 'Running' }).Count -gt 0) {
        foreach ($jb in $jobs) {
            Receive-Job -Job $jb | ForEach-Object {
                if ("$_".Trim()) { Write-Host "  [$($jb.Name)] $_" }
            }
        }
        if (((Get-Date) - $lastBeat).TotalSeconds -ge 60) {
            $lastBeat = Get-Date
            $done = @(Get-ChildItem 'saes/quarto/*champTa*.pt' -ErrorAction SilentlyContinue |
                Where-Object { $_.LastWriteTime -gt $sweepStart -and $_.Name -notlike '*_metrics*' }).Count
            $el = (Get-Date) - $sweepStart
            $eta = if ($done -gt 0) {
                $per = $el.TotalSeconds / $done
                '~{0:hh\:mm}' -f [TimeSpan]::FromSeconds($per * ($nCfg - $done))
            }
            else { 'unknown' }
            $running = @($jobs | Where-Object { $_.State -eq 'Running' } | ForEach-Object Name) -join ','
            Write-Host ("  ... {0}/{1} trained | elapsed {2:hh\:mm} | ETA {3} | active: {4}" -f `
                    $done, $nCfg, $el, $eta, $running)
        }
        Start-Sleep -Seconds 5
    }
    # Final drain: output produced between the last poll and the job ending.
    foreach ($jb in $jobs) {
        Receive-Job -Job $jb | ForEach-Object {
            if ("$_".Trim()) { Write-Host "  [$($jb.Name)] $_" }
        }
    }
    $failed = @($jobs | Where-Object { $_.State -eq 'Failed' } | ForEach-Object Name)
    $jobs | Remove-Job -Force
    if ($failed) { throw "Sweep job(s) FAILED: $($failed -join ', ')" }
    Write-Host ("  Sweep done in {0:hh\:mm\:ss}." -f ((Get-Date) - $sweepStart))

    # --- Stage 6: evaluate -------------------------------------------------
    # Sequential: sae_eval writes the shared registry and the _h cache.
    #
    # ONE call per checkpoint with all three bases comma-separated: sae_eval
    # encodes the codes once and reuses them across every BSP set in the call.
    # A per-basis loop with --force re-encodes the full dataset three times and
    # rewrites the same multi-GB _h file (1.4-4.8 GB each here), i.e. 132
    # encodes across this sweep instead of 44.
    $ckpts = @(Get-ChildItem 'saes/quarto/*champTa*.pt' | Where-Object { $_.Name -notlike '*_metrics*' })
    $bspList = "gorilla$CHAMP,hawk$CHAMP,tiger$CHAMP"
    Write-Stage 6
    Write-Host "  $($ckpts.Count) checkpoint(s) against $bspList."
    $evalStart = Get-Date
    $k = 0
    foreach ($ck in $ckpts) {
        $k++
        $el = (Get-Date) - $evalStart
        $eta = if ($k -gt 1) {
            '~{0:hh\:mm}' -f [TimeSpan]::FromSeconds(
                ($el.TotalSeconds / ($k - 1)) * ($ckpts.Count - $k + 1))
        }
        else { 'unknown' }
        Write-Host ("`n  ({0}/{1}) {2}  | elapsed {3:hh\:mm} | ETA {4}" -f `
                $k, $ckpts.Count, $ck.BaseName, $el, $eta)
        python sae_eval.py evaluate $ck.FullName --bsps=$bspList --force
    }
    Write-Host ("  Eval done in {0:hh\:mm\:ss}." -f ((Get-Date) - $evalStart))

    # --- Stage 7: clear the quarantine ------------------------------------
    # Every downstream runner (3A-prep, basis-verdict, unified-pool) reads
    # _dataset_status.json and SKIPS quarantined champions. If this is left to a
    # human, the rest of the chain silently drops champTa and the whole rebuild
    # looks like it did nothing. Only done on a full run, and only after the
    # provenance check passes cleanly -- the flip is a claim about the data, so
    # it is gated on the evidence rather than on this script having reached the
    # end.
    # ORDER IS LOAD-BEARING: flip FIRST, validate AFTER, roll back on failure.
    #
    # The previous version validated before flipping, which could never fail:
    # while champTa is QUARANTINED, validate_datasets.py treats its problems as
    # DECLARED and exits 0. So the gate passed on the strength of the very
    # quarantine it was about to lift, and a dataset that fails validation was
    # marked OK. That is the "gate whose other arm never fires" failure mode
    # already recorded in CLAUDE.md -- reintroduced. Validating the POST-flip
    # state is the only version of this check that can actually fail.
    Write-Stage 7
    $status = 'data/quarto/_dataset_status.json'
    $rollback = "$status.pre-clear"
    Copy-Item $status $rollback -Force
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
    for k in ("labels", "activations", "saes", "analysis"):
        doc["derived_artifacts_quarantined"].pop(k, None)
    doc["updated"] = "2026-08-12"
    p.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print("  champTa -> OK (provisional; validated below).")
'@
    $clear | python -

    Write-Host '  Validating the POST-flip state...'
    $prev = $PSNativeCommandUseErrorActionPreference
    $PSNativeCommandUseErrorActionPreference = $false
    python scripts/validate_datasets.py
    $clean = ($LASTEXITCODE -eq 0)
    $PSNativeCommandUseErrorActionPreference = $prev
    if (-not $clean) {
        Move-Item $rollback $status -Force
        throw ('Provenance FAILS with champTa marked OK -- quarantine restored. ' +
               'Fix the dataset (see the table above), then re-run.')
    }
    Remove-Item $rollback -Force
    Write-Host '  Validation clean with champTa OK.'

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
