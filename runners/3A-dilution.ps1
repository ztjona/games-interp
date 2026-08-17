#Requires -Version 7
<#
  runners/3A-dilution.ps1 -- Phase 3A dilution diagnostic over the champTa /
  champVe / champYb panel SAEs. CPU analysis on cached SAE codes, so it is
  sequential (no GPU fan-out needed). Verifies each run's inputs are present
  before invoking, writes one JSON per run plus a combined
  3A_gate_summary.json, and emits stage_3A-dilution.md.

  Nulls: every panel run is paired with a recipe-matched RANDOM-MODEL SAE
  control (trained by runners/3A-prep.ps1). If a control's _h cache is absent,
  dilution_diagnostic.py only WARNS on stderr and quietly drops to the
  permutation null, which understates what "absent" should mean -- so the
  pre-flight below fails fast on a missing control instead.

  Read the verdicts against the 2026-08-13 learned-gap measurement: on conv2 the
  random-model control already reaches 45-92% of the trained SAE's score
  (champYb conv2: only 7-16% of the score is learned), so a conv2 dilution
  verdict says much less about the TRAINED model than an fc1 one does.

  Method + thresholds: docs/diary/2026-07-21_3A-dilution-diagnostic.md
  Launch:        runners\launch.ps1 3A-dilution
  Dry-run:       pwsh -File runners\3A-dilution.ps1 -DryRun
#>
param(
    [switch]$DryRun,
    # Substring filter over the "run_id|bsps" entries below. Partial re-runs
    # are first-class because every `_h` is now cached, so re-running a few
    # cells is minutes of CPU rather than the 2h12m a full pass costs. Doing it
    # by hand instead means retyping the random-control mapping, which is
    # exactly how a cell ends up silently uncontrolled.
    #   -Only champVe        re-run every champVe cell
    #   -Only K04            re-run only the new conv2 panel member
    [string]$Only = ''
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Set-Location (Join-Path $PSScriptRoot '..')
$env:PYTHONUTF8 = '1'
# Activate the venv explicitly (a detached launch does not inherit it).
$activate = '.\.venv\Scripts\Activate.ps1'
if (Test-Path $activate) { & $activate } else { throw ".venv not found at $activate" }
New-Item -ItemType Directory -Force -Path logs | Out-Null
Start-Transcript -Path 'logs/3A-dilution.transcript.log' -Append | Out-Null

try {
    $GAME = 'quarto'
    $CACHE = "saes/$GAME/cache"
    $DATA = "data/$GAME"
    $ANALYSIS = "saes/$GAME/analysis"
    $TOPK = 64

    # "run_id|bsps|random_run_id"  (random 'none' -> permutation null)
    #
    # Design of the panel (2026-07-28 revision):
    #   * three champions (Ta, Ve, Yb) x two hooks (fc1 F04/F01, conv2 E05/E01)
    #     x the anchored positive control (I03/I04), so every champion is
    #     comparable on the same axes -- champVe previously had no fc1
    #     unsupervised row, which is why the first cross-champion read could
    #     only contrast Ta vs Yb.
    #   * all three BSP bases (gorilla / hawk / tiger) on the fc1 winners. The
    #     first run covered gorilla+tiger only; hawk was dropped by the
    #     2026-05-22 reframing audit for *supervision targeting*, but that says
    #     nothing about whether dilution is basis-dependent, which is exactly
    #     what 3A can answer for free (the codes are already cached).
    $RUNS = @(
        # --- champTa ---
        'F04-champTa-s42-jumprelu-t64-exp8-s4.fc1|tigerTa|none',
        'F04-champTa-s42-jumprelu-t64-exp8-s4.fc1|gorillaTa|none',
        'F04-champTa-s42-jumprelu-t64-exp8-s4.fc1|hawkTa|none',
        'E05-champTa-s42-batchtopk-k32-exp8-s4.conv2|tigerTa|none',
        'I04-champTa-lh100-s43-anchored-jumprelu-t64-exp8-s4.fc1|tigerTa|none',
        # --- champVe ---
        'F04-champVe-s42-jumprelu-t64-exp8-s4.fc1|tigerVe|none',
        'F04-champVe-s42-jumprelu-t64-exp8-s4.fc1|gorillaVe|none',
        'F04-champVe-s42-jumprelu-t64-exp8-s4.fc1|hawkVe|none',
        'E01-champVe-s42-topk-k32-exp8-s4.conv2|gorillaVe|none',
        'E05-champVe-s42-batchtopk-k32-exp8-s4.conv2|tigerVe|none',
        'I04-champVe-lh100-s42-anchored-jumprelu-t64-exp8-s4.fc1|tigerVe|none',
        'I03-champVe-lh030-s43-anchored-jumprelu-t64-exp8-s4.fc1|gorillaVe|none',
        # --- champYb: unsupervised fc1 (F04) + conv2 (E05) are the focus; the
        # anchored I04 is the positive control (info IS extractable with
        # supervision -> the diagnostic must place it at 'captured'; under the
        # 3A.1 rule it did not, which is what forced the 3A.2 recalibration).
        'F04-champYb-s42-jumprelu-t64-exp8-s4.fc1|tigerYb|none',
        'F04-champYb-s42-jumprelu-t64-exp8-s4.fc1|gorillaYb|none',
        'F04-champYb-s42-jumprelu-t64-exp8-s4.fc1|hawkYb|none',
        # E05 -> K04 (2026-08-17). E05 is a COLLAPSED dictionary: 99.0% dead,
        # 41 alive latents (fewer than TOPK=64, so it could not even fill the
        # candidate list), FVU 0.110 against 0.043/0.051 for the same recipe on
        # champTa/champVe, and ranked 10th of 11 champYb conv2 runs. Its 3A
        # verdict of 0.17 "(absent)" described the broken dictionary, not
        # champYb's conv2 representation. K04 is the same recipe trained in
        # canonical form: FVU 0.0066, 174 alive, and it beats every other
        # canonical conv2 run on the threat families (seed-verified, n=3).
        'K04-champYb-s42-batchtopk-k32-exp8-s4.conv2|tigerYb|none',
        'K04-champYb-s42-batchtopk-k32-exp8-s4.conv2|gorillaYb|none',
        'I04-champYb-lh100-s42-anchored-jumprelu-t64-exp8-s4.fc1|tigerYb|none'
    )

    # Random-model SAE controls, keyed by "<champTag>|<hook>" -> run_id, e.g.
    #   'Yb|s4.fc1' = 'A03-champYb-random-s42-jumprelu-t64-exp8-s4.fc1'
    # Populating an entry is all that is needed to upgrade those runs from the
    # permutation null to a proper random-network control.
    # The reporting standard (RESEARCH-STATUS.md clause 2) and the 3A method
    # spec both want a random-network control rather than only the permutation
    # null: a randomly-initialised CNN still yields SAE structure, so the
    # permutation null UNDERSTATES what "absent" should mean.
    # Trained by runners/3A-prep.ps1 (configs/controls/*.yaml), recipe-matched
    # to the unsupervised panel members. If a control's _h cache is absent the
    # CLI warns and falls back to the permutation null, recording
    # random_control: null -- the weaker null is never silently assumed.
    $RANDOM_CONTROLS = @{
        # R1 -> R3 (2026-08-16). R1 was DEGENERATE: 4032/4096 latents never
        # fired and the 64 survivors fired on >99.98% of rows, so ZERO latents
        # were alive by the diagnostic's own definition and the control
        # produced no number -- 13 of 17 cells were named as controlled while
        # never being tested. Root cause was the missing tied init, not
        # JumpReLU: R3 is the SAME recipe plus W_enc = W_dec^T and has 190
        # alive latents. So the control stays recipe-matched to the fc1 panel
        # member and differs only in that the network is untrained.
        'Ta|s4.fc1'   = 'R3-champTarandom-s42-jumprelu-t64-exp8-s4.fc1'
        'Ve|s4.fc1'   = 'R3-champVerandom-s42-jumprelu-t64-exp8-s4.fc1'
        'Yb|s4.fc1'   = 'R3-champYbrandom-s42-jumprelu-t64-exp8-s4.fc1'
        'Ta|s4.conv2' = 'R2-champTarandom-s42-batchtopk-k32-exp8-s4.conv2'
        'Ve|s4.conv2' = 'R2-champVerandom-s42-batchtopk-k32-exp8-s4.conv2'
        'Yb|s4.conv2' = 'R2-champYbrandom-s42-batchtopk-k32-exp8-s4.conv2'
    }

    Write-Host '=================================================================='
    Write-Host 'Phase 3A dilution diagnostic runner'
    Write-Host '=================================================================='


    # Dataset integrity gate. champTa's positions were random-play-only and
    # champS4's were champAa's; both went unnoticed for months. Fail fast rather
    # than produce results on a quarantined dataset.
    Write-Host "`nValidating dataset provenance..."
    python scripts/validate_datasets.py
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset validation failed. See data/quarto/DATA-STATUS.md."
    }

    if (-not (Test-Path lib/sae/dilution.py)) { throw 'lib/sae/dilution.py missing (wrong cwd?)' }
    New-Item -ItemType Directory -Force -Path $ANALYSIS | Out-Null

    function Test-Labels([string]$bsps) {
        @(Get-ChildItem "$DATA/bsp_labels-$($bsps)_*.pt" -ErrorAction SilentlyContinue).Count -gt 0
    }

    # Need the checkpoint + BSP labels. The _h code cache is NOT required up front:
    # if a prior cleanup deleted it, we regenerate it from the checkpoint below.
    # --- Control usability gate --------------------------------------------
    # A control that TRAINS is not a control that WORKS. R1 trained fine and
    # contributed nothing, and nothing in this runner noticed, so 13 of 17
    # cells reported n_absent = 0 -- which read as "no concept failed" when it
    # meant "the test never ran". Fail loudly here instead: `spread` and
    # `captured` are only meaningful once `absent` can fire.
    Write-Host "`n[gate] Checking every random-model control is usable..."
    $ctlCkpts = @($RANDOM_CONTROLS.Values | Sort-Object -Unique |
        ForEach-Object { "saes/$GAME/$_.pt" } | Where-Object { Test-Path $_ })
    if ($ctlCkpts) {
        python scripts/check_control_usable.py @ctlCkpts
        if ($LASTEXITCODE -ne 0) {
            throw ("One or more random-model controls are UNUSABLE (fewer alive " +
                   "latents than top_k). Rule 3A.3's learned-signal floor cannot " +
                   "fire for those cells, so the gate would be provisional. " +
                   "Retrain the control before running 3A.")
        }
    }
    else { throw "No random-model control checkpoints found on disk." }

    if ($Only) {
        $before = $RUNS.Count
        $RUNS = @($RUNS | Where-Object { $_ -like "*$Only*" })
        if (-not $RUNS) { throw "-Only '$Only' matched none of the $before panel entries." }
        Write-Host "  -Only '$Only': $($RUNS.Count) of $before panel entries selected."
    }

    # Resolve each entry's random-model control the same way the exec loop does,
    # so the pre-flight reports the SAME pairing that will actually be used.
    function Resolve-Control([string]$rid, [string]$fallback) {
        if ($rid -match 'champ(\w\w)-.*-(s4\.\w+|fc1|conv2)$') {
            $k = "$($Matches[1])|$($Matches[2])"
            if ($RANDOM_CONTROLS.ContainsKey($k)) { return $RANDOM_CONTROLS[$k] }
        }
        return $fallback
    }

    Write-Host "`nPre-flight (input presence):"
    $runnable = @()
    $noControl = @()
    foreach ($entry in $RUNS) {
        $rid, $bsps, $rand = $entry.Split('|')
        $rand = Resolve-Control $rid $rand
        $why = ''
        if (-not (Test-Path "saes/quarto/$rid.pt")) { $why += ' no checkpoint;' }
        if (-not (Test-Labels $bsps)) { $why += " no bsp_labels-$bsps;" }
        $hasH = Test-Path "$CACHE/$($rid)_h.pt"
        # A control with no _h is NOT a skip -- the run would still succeed, but
        # silently on the weaker permutation null. Collect and fail below.
        if ($rand -ne 'none' -and -not (Test-Path "$CACHE/$($rand)_h.pt")) {
            $noControl += "$rid  ->  $rand"
        }
        if ($why) { Write-Host "  [SKIP] $rid  bsps=$bsps --$why" }
        else {
            Write-Host ("  [ OK ] {0}  bsps={1}  null={2}{3}" -f $rid, $bsps,
                $(if ($rand -eq 'none') { 'permutation' } else { 'random-model' }),
                $(if (-not $hasH) { '  (will regen _h)' } else { '' }))
            $runnable += $entry
        }
    }

    if ($noControl) {
        Write-Host "`nMissing random-model control _h cache for:"
        $noControl | ForEach-Object { Write-Host "  $_" }
        throw ('Refusing to run: these would silently fall back to the ' +
               'permutation null, which UNDERSTATES "absent". Run ' +
               'runners/3A-prep.ps1 to train and evaluate the controls first.')
    }

    if ($runnable.Count -eq 0) {
        Write-Host "`nNothing runnable (need the panel checkpoints + BSP labels on this box)."
        return
    }
    if ($DryRun) { Write-Host "`nDryRun -> not executing. $($runnable.Count) run(s) would execute."; return }

    Write-Host "`nExecuting $($runnable.Count) run(s)..."
    $runStart = Get-Date
    $n = 0
    foreach ($entry in $runnable) {
        $n++
        $rid, $bsps, $rand = $entry.Split('|')
        # Same resolver the pre-flight used, so the pairing reported above is
        # the pairing actually run -- two copies of this regex would be free to
        # disagree.
        $rand = Resolve-Control $rid $rand
        $el = (Get-Date) - $runStart
        $eta = if ($n -gt 1) {
            '~{0:hh\:mm}' -f [TimeSpan]::FromSeconds(
                ($el.TotalSeconds / ($n - 1)) * ($runnable.Count - $n + 1))
        }
        else { 'unknown' }
        Write-Host ("`n>> ({0}/{1}) {2}  bsps={3}  random={4} | elapsed {5:hh\:mm} | ETA {6}" -f `
                $n, $runnable.Count, $rid, $bsps, $rand, $el, $eta)
        # 3A reads the SAE code cache (_h). Regenerate it via a normal eval if a
        # prior disk cleanup removed it (writes {rid}_h.pt + matching + registry).
        if (-not (Test-Path "$CACHE/$($rid)_h.pt")) {
            Write-Host '   _h cache missing -> regenerating via sae_eval --force...'
            # --force is required: these champTa/Ve/Yb runs are already in the
            # eval registry, so without it sae_eval skips ("Found cached") and
            # never writes the _h cache that dilution_diagnostic needs.
            python sae_eval.py evaluate "saes/quarto/$rid.pt" --bsps=$bsps --force
        }
        python scripts/dilution_diagnostic.py --run-id=$rid --bsps=$bsps --random-run-id=$rand --top-k=$TOPK
    }
    Write-Host ("`nAll {0} run(s) done in {1:hh\:mm\:ss}." -f `
            $runnable.Count, ((Get-Date) - $runStart))

    Write-Host "`nWriting combined gate summary..."
    # Was an inline heredoc until 2026-08-17, when it still read the
    # `n_diluted`/`n_tiled` gate keys that rule 3A.3 removed -- so the runner
    # completed all 17 cells in 2h12m and then died on the last line. Logic that
    # can break a run belongs in a tested entry point (lessons.md 3.1); the
    # runner only sequences and checks the exit code.
    $aggOut = python scripts/summarize_3a_gate.py --game=$GAME --expect-rule=3A.3 2>&1
    $aggOut | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) { throw "Gate summary failed (exit $LASTEXITCODE)." }

    Write-Host "`nEmitting stage plan..."
    $files = @(Get-ChildItem "$ANALYSIS/*_dilution-*.json" | ForEach-Object { $_.FullName })
    $files += "$ANALYSIS/3A_gate_summary.json"
    python scripts/emit_stage.py --slug 3A-dilution @files

    Write-Host "`nDone. Per-run JSON in $ANALYSIS/; commit plan in stage_3A-dilution.md"
}
finally {
    Stop-Transcript | Out-Null
}
