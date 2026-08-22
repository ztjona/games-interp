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
    # -Panel: run the TOP-3-CONDITIONS panel from analysis/3A_panel.json
    # instead of the hand-pinned single member per cell below. 18 cells x 3
    # conditions + the anchored positive controls; with -WithHen, hawk's
    # selections are mirrored onto the `hen` basis as well. The entry list is
    # expanded by scripts/build_3a_panel_runlist.py -- the runner only
    # sequences it.
    [switch]$Panel,
    [switch]$WithHen,
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

    # --- -Panel: replace the hand-pinned list with the selected panel -------
    # The single-member panel above answered "is the wall real?"; it cannot
    # answer "does the conclusion depend on WHICH SAE we picked?", which is the
    # residual 2026-08-17 left open. Selection (threat-family MCC, deduped by
    # condition) is scripts/select_3a_panel.py; expansion is
    # scripts/build_3a_panel_runlist.py.
    if ($Panel) {
        Write-Host "`n[panel] Expanding analysis/3A_panel.json into a runlist..."
        $henArg = if ($WithHen) { '--with-hen' } else { '' }
        python scripts/build_3a_panel_runlist.py --game=$GAME @($henArg | Where-Object { $_ })
        if ($LASTEXITCODE -ne 0) {
            throw ("Runlist build failed -- BSP labels are missing for at least " +
                   "one basis. Run scripts/compute_bsp_labels.py first.")
        }
        $runlist = Get-Content "$ANALYSIS/3A_panel_runlist.json" -Raw | ConvertFrom-Json
        # Same "rid|bsps|random" shape the hand-pinned list uses, so everything
        # downstream is identical for both modes.
        $RUNS = @($runlist.entries | ForEach-Object {
                "$($_.run_id)|$($_.bsps)|$($_.random_run_id)" })
        Write-Host ("  {0} entries over {1} checkpoints." -f $RUNS.Count,
            @($runlist.entries.run_id | Sort-Object -Unique).Count)
    }


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
    if ($Only) {
        $before = $RUNS.Count
        $RUNS = @($RUNS | Where-Object { $_ -like "*$Only*" })
        if (-not $RUNS) { throw "-Only '$Only' matched none of the $before panel entries." }
        Write-Host "  -Only '$Only': $($RUNS.Count) of $before panel entries selected."
    }

    # --- STAGE 0: encode any missing _h, BEFORE the usability gate ----------
    # Ordering is load-bearing, not cosmetic. check_sae_usable.py exits 2 when
    # an _h cache is absent and this runner throws on a non-zero exit, so
    # gating first makes the FIRST run after a cache prune impossible -- and
    # the 2026-08-21 prune deliberately dropped 15 panel members and all four
    # anchored controls, on the mistaken belief they had already been encoded.
    # One encode per CHECKPOINT with a comma list of bases: _h does not depend
    # on the BSP set, so per-basis encoding rewrites a 1.4-4.8 GB tensor for
    # nothing. Sequential: sae_eval writes the shared eval registry.
    $needEncode = @{}
    foreach ($entry in $RUNS) {
        $rid, $bsps, $null = $entry.Split('|')
        if (-not (Test-Path "$CACHE/$($rid)_h.pt")) {
            if (-not $needEncode.ContainsKey($rid)) { $needEncode[$rid] = @() }
            if ($needEncode[$rid] -notcontains $bsps) { $needEncode[$rid] += $bsps }
        }
    }
    if ($needEncode.Count -gt 0) {
        Write-Host ("`n[stage 0] {0} checkpoint(s) need an _h encode (GPU)..." -f $needEncode.Count)
        if ($DryRun) {
            $needEncode.GetEnumerator() | Sort-Object Name | ForEach-Object {
                Write-Host ("  would encode {0}  --bsps={1}" -f $_.Key, ($_.Value -join ','))
            }
        }
        else {
            $encN = 0
            foreach ($kv in ($needEncode.GetEnumerator() | Sort-Object Name)) {
                $encN++
                $ck = "saes/$GAME/$($kv.Key).pt"
                if (-not (Test-Path $ck)) { throw "Checkpoint missing: $ck" }
                Write-Host ("  ({0}/{1}) {2}  --bsps={3}" -f $encN, $needEncode.Count,
                    $kv.Key, ($kv.Value -join ','))
                # --force: these runs are already in the eval registry, so
                # without it sae_eval reports "Found cached" and never writes
                # the _h the diagnostic needs.
                #
                # Build the argument as a STRING first. PowerShell splits
                # `--bsps=($x -join ',')` into TWO arguments -- `--bsps=` and
                # the joined value -- so the inline form silently passes an
                # empty --bsps plus a stray positional.
                $bspArg = '--bsps=' + ($kv.Value -join ',')
                python sae_eval.py evaluate $ck $bspArg --force
                if ($LASTEXITCODE -ne 0) { throw "Encode failed for $($kv.Key)." }
            }
        }
    }
    else { Write-Host "`n[stage 0] every _h cache is present -- nothing to encode." }

    Write-Host "`n[gate] Checking every SAE is usable (panel members AND controls)..."
    # Applies to BOTH roles because both have failed. E05-champYb was a PANEL
    # MEMBER with 41 alive latents -- fewer than TOPK -- so its candidate list
    # could not be filled and its verdict described the collapsed dictionary
    # rather than the champion; nothing in this runner noticed for weeks.
    # Gating only the controls would leave that hole open.
    $panelIds = @($RUNS | ForEach-Object { ($_ -split '\|')[0] } | Sort-Object -Unique)
    # Gate only what HAS an _h. After stage 0 that is everything; under -DryRun
    # it is everything stage 0 has not been asked to create yet. Handing the
    # gate a checkpoint whose cache stage 0 is about to write makes it exit 2
    # ("NO _h CACHE") and kills the run -- which would reintroduce, inside
    # -DryRun, the exact ordering bug stage 0 exists to fix.
    $gateIds = @($panelIds + @($RANDOM_CONTROLS.Values) | Sort-Object -Unique)
    $pending = @($gateIds | Where-Object { -not (Test-Path "$CACHE/$($_)_h.pt") })
    if ($pending) {
        Write-Host ("  {0} checkpoint(s) not yet encoded -- gated after stage 0:" -f $pending.Count)
        $pending | ForEach-Object { Write-Host "    $_" }
    }
    $ckpts = @($gateIds | Where-Object { Test-Path "$CACHE/$($_)_h.pt" } |
        ForEach-Object { "saes/$GAME/$_.pt" } | Where-Object { Test-Path $_ })
    if ($ckpts) {
        python scripts/check_sae_usable.py --top-k=$TOPK @ckpts
        if ($LASTEXITCODE -ne 0) {
            throw ("One or more SAEs are UNUSABLE (fewer alive latents than " +
                   "top_k=$TOPK). A control that cannot fill the candidate list " +
                   "makes rule 3A.3's learned-signal floor unfirable; a PANEL " +
                   "MEMBER that cannot fill it is measured at a smaller support " +
                   "than every other cell. Fix the dictionary before running 3A." +
                   "`n" +
                   "In -Panel mode this should be unreachable: select_3a_panel.py " +
                   "measures alive with the SAME RankingCache and promotes a " +
                   "reserve condition when one fails. If it fires anyway, the " +
                   "panel JSON is stale -- re-run:`n" +
                   "  python scripts/select_3a_panel.py --output=saes/$GAME/analysis/3A_panel.json`n" +
                   "On 2026-08-21 this gate killed a 76-entry run after 40 min " +
                   "having produced NOTHING, because selection estimated alive " +
                   "from the registry's dead_features_pct (which counts " +
                   "always-on latents as alive) while this gate measures it: " +
                   "E06-champYb estimated 85, measured 38.")
        }
    }
    else { throw "No SAE checkpoints found on disk for this panel." }

    # Resolve each entry's random-model control the same way the exec loop does,
    # so the pre-flight reports the SAME pairing that will actually be used.
    #
    # In -Panel mode the pairing was ALREADY resolved, by
    # scripts/build_3a_panel_runlist.py, and arrives in the third field -- so
    # this returns it untouched rather than re-deriving it. Two copies of one
    # mapping are free to disagree, and the copy that loses is the one that
    # silently drops a control (the failure that left 13 of 17 cells reporting
    # n_absent = 0 because the test never ran).
    function Resolve-Control([string]$rid, [string]$fallback) {
        if ($Panel) { return $fallback }
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
    $aggOut = python scripts/summarize_3a_gate.py --game=$GAME --expect-rule=3A.4 2>&1
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
