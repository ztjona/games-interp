#Requires -Version 7
<#
  runners/basis-verdict.ps1 -- settle hawk vs tiger as the supervision target,
  on prevalence-fair metrics, per CONCEPT FAMILY, against the linear-probe
  upper bound.

  WHY THIS RE-RUN EXISTS
  ----------------------
  The 2026-05-22 reframing audit chose tiger over hawk and redirected the
  programme. That comparison has three defects, all of which this runner fixes:

    1. It compared WHOLE-BASIS averages. hawk's 173 BSPs are ~76 line + ~81
       square + 2 global; tiger's 36 are 10 line + 9 square + 5 global + 12
       pool. Those averages are different partitions of the concept menu, so
       the difference between them is not a framing effect. Fixed by grouping
       on the schema's `concept_family` stamp and comparing family by family.
    2. It used PREVALENCE-DEPENDENT metrics across bases whose base rates
       differ by ~7x (hawk completable ~0.003 vs tiger conjunctions ~0.022).
       MCC, F1 and F1-lift all fall with base rate for a detector of fixed
       quality. Fixed by reporting `mcc_at_pref` (p_ref = 0.025) with Youden's
       J as the prevalence-invariant cross-check.
    3. It conflated AVAILABILITY with EFFICIENCY. A basis can win on raw SAE
       score simply because its concepts are easier to decode -- which says
       nothing about whether an unsupervised dictionary finds them. Fixed by
       measuring the linear probe (the upper bound) alongside the SAE and
       reporting the ratio.

  ORDER MATTERS. Each stage is a hard input to the next, and every one of them
  has already been observed to fail silently rather than loudly:
      0. dataset provenance gate      -- else the whole verdict rests on a
                                         quarantined distribution
      1. stamp concept families       -- else there are no family rollups and
                                         both sides fall back to whole-basis
      2. LP baselines (SLOW, CPU)     -- else efficiency has no denominator;
                                         reports written before mcc_at_pref
                                         existed roll up fine and silently
                                         yield "n/a" for every ratio
      3. SAE evals carry mcc_at_pref  -- checked, not assumed
      4. basis_comparison (per champ) -- one SAE, one code set, three framings
      5. sae_lp_efficiency            -- the verdict

  Champions: Ta, Ve, Yb. champTa was rebuilt and retrained on 2026-08-12
  (runners/champTa-rebuild.ps1), which brought its tiger-conjunction base rate
  from 0.045 to 0.0255 -- in line with Ve 0.0239 and Yb 0.0223 -- so the
  distribution confound this runner exists to remove is gone. Any champion
  still QUARANTINED/RETIRED is dropped automatically at stage 0.

  COST. The LP stage dominates and scales with the training split: ~32 s per
  BSP fit at the full 232k rows, i.e. 373 BSPs x 6 champion-hook combos ~= 21 h
  CPU (~7 h wall across 3 parallel champions). Measured on champTa/fc1/tiger:

      max_train    s/BSP    MCC (base 0.28)   MCC (base 0.12)
        232,117     31.8         0.6365            0.5754
        100,000     11.2         0.6350            0.5699
         50,000      4.2         0.6326            0.5643
         25,000      1.8         0.6296            0.5632

  So `-MaxTrain 50000` costs ~1-2% of MCC on common concepts and turns ~7 h of
  wall clock into ~50 min. The probe is an UPPER BOUND, not a precision
  estimate, and 512 features do not need 232k rows. Caveat: rare concepts
  degrade first (hawk `completable` sits near base rate 0.003, so 50k leaves
  ~150 positives) -- linear_probe_baseline.py names any BSP left with <50
  positives, so read that warning before trusting a rare-family ratio.

  Launch:   pwsh -File runners\launch.ps1 basis-verdict
  Dry-run:  pwsh -File runners\basis-verdict.ps1 -DryRun
  Skip the slow LP stage if the reports are already current: -SkipProbes
  Cap the LP training split (big time saver, see below): -MaxTrain 50000
#>
param([switch]$DryRun, [switch]$SkipProbes, [int]$MaxTrain = 0,
      [string[]]$Champions = @('Ta', 'Ve', 'Yb'))

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Set-Location (Join-Path $PSScriptRoot '..')
$env:PYTHONUTF8 = '1'
$activate = '.\.venv\Scripts\Activate.ps1'
if (Test-Path $activate) { & $activate } else { throw ".venv not found at $activate" }
New-Item -ItemType Directory -Force -Path logs | Out-Null
Start-Transcript -Path 'logs/basis-verdict.transcript.log' -Append | Out-Null

try {
    $GAME = 'quarto'
    $DATA = "data/$GAME"
    $HOOKS = @('s4.fc1', 's4.conv2')
    $BASES = @('gorilla', 'hawk', 'tiger')

    Write-Host '=================================================================='
    Write-Host 'Basis verdict: hawk vs tiger, per family, SAE against the LP bound'
    Write-Host '=================================================================='

    # --- Stage 0: dataset provenance gate ---------------------------------
    Write-Host "`n[0/5] Validating dataset provenance..."
    python scripts/validate_datasets.py
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset validation failed. See data/quarto/DATA-STATUS.md."
    }

    # Refuse to include a champion whose dataset is QUARANTINED or RETIRED.
    # validate_datasets.py passes when a problem is DECLARED, so it alone would
    # let champTa through and mint a verdict from random-play activations.
    $badChamps = @'
import json
s = json.load(open("data/quarto/_dataset_status.json"))["datasets"]
print(",".join(sorted({v["champion"] for v in s.values()
                       if v.get("status") in ("QUARANTINED", "RETIRED")
                       and v.get("champion")})))
'@
    $bad = ((($badChamps | python -) -join '').Trim() -split ',' | Where-Object { $_ })
    if ($bad) {
        $before = $Champions
        $Champions = @($Champions | Where-Object { $_ -notin $bad })
        $dropped = @($before | Where-Object { $_ -in $bad })
        if ($dropped) {
            Write-Host "  Excluding [$($dropped -join ', ')] -- dataset QUARANTINED/RETIRED."
        }
    }
    if (-not $Champions) { throw 'No champion has a usable dataset; nothing to do.' }
    Write-Host "  Champions in this verdict: $($Champions -join ', ')"

    # --- Stage 1: stamp concept families onto the schemas ------------------
    # Cheap and idempotent. Both the LP side and the SAE side group by the
    # schema's stamp, so this must happen BEFORE either is computed.
    Write-Host "`n[1/5] Stamping concept families onto the BSP schemas..."
    if ($DryRun) { python scripts/stamp_concept_families.py --dry-run }
    else { python scripts/stamp_concept_families.py }

    # --- Stage 2: linear-probe baselines (the denominator) -----------------
    # SLOW: one logistic regression per BSP per (champion, hook, basis).
    # CPU-bound and file-disjoint, so the champions fan out; within a champion
    # it stays sequential to keep memory bounded (each probe holds the full
    # activation matrix).
    if (-not $SkipProbes) {
        Write-Host "`n[2/5] Linear-probe baselines: $($Champions.Count) champion(s) x $($HOOKS.Count) hook(s) x $($BASES.Count) bases."
        Write-Host '      This is the slow stage. It is the UPPER BOUND every efficiency'
        Write-Host '      ratio is divided by, so it cannot be skipped on a first run.'

        $plan = foreach ($champ in $Champions) {
            $tag = $champ.ToLower()
            foreach ($hook in $HOOKS) {
                $act = "$DATA/${hook}_amalgam_${tag}_activations.pt"
                foreach ($basis in $BASES) {
                    $lbl = @(Get-ChildItem "$DATA/bsp_labels-$basis$champ`_[0-9]*.pt" -ErrorAction SilentlyContinue)
                    $sch = @(Get-ChildItem "$DATA/bsp_schema-$basis`_[0-9]*.json" -ErrorAction SilentlyContinue)
                    if (-not (Test-Path $act)) { Write-Host "  [SKIP] no activations: $act"; continue }
                    if (-not $lbl) { Write-Host "  [SKIP] no labels: bsp_labels-$basis$champ"; continue }
                    if (-not $sch) { Write-Host "  [SKIP] no schema: bsp_schema-$basis"; continue }
                    [pscustomobject]@{
                        Champ = $champ; Hook = $hook; Basis = $basis
                        Act = $act; Labels = $lbl[0].FullName; Schema = $sch[0].FullName
                    }
                }
            }
        }

        Write-Host "  $($plan.Count) probe job(s) planned."
        if ($DryRun) {
            $plan | ForEach-Object {
                Write-Host "    [DRY] linear_probe_baseline.py $($_.Act) $(Split-Path $_.Labels -Leaf) $(Split-Path $_.Schema -Leaf)"
            }
        }
        else {
            $root = (Get-Location).Path
            $jobs = $Champions | ForEach-Object {
                $champ = $_
                $mine = @($plan | Where-Object { $_.Champ -eq $champ })
                Start-Job -Name "champ$champ" -ScriptBlock {
                    param($root, $mine, $maxTrain)
                    Set-Location $root
                    $env:PYTHONUTF8 = '1'
                    foreach ($j in $mine) {
                        Write-Output ">> LP $($j.Champ) $($j.Hook) $($j.Basis)"
                        $a = @('scripts/linear_probe_baseline.py',
                            $j.Act, $j.Labels, $j.Schema)
                        if ($maxTrain -gt 0) { $a += "--max-train=$maxTrain" }
                        & '.\.venv\Scripts\python.exe' @a 2>&1
                    }
                } -ArgumentList $root, $mine, $MaxTrain
            }

            # Incremental drain, tagged per champion -- `Receive-Job -Wait`
            # buffers each job whole and replays its timestamps from the start
            # once the previous one ends, which makes the log read as though the
            # run finished several times over.
            $lpStart = Get-Date
            $lastBeat = Get-Date
            while (@($jobs | Where-Object { $_.State -eq 'Running' }).Count -gt 0) {
                foreach ($jb in $jobs) {
                    Receive-Job -Job $jb | ForEach-Object {
                        if ("$_".Trim()) { Write-Host "  [$($jb.Name)] $_" }
                    }
                }
                if (((Get-Date) - $lastBeat).TotalSeconds -ge 120) {
                    $lastBeat = Get-Date
                    $reports = @(Get-ChildItem "$DATA/linear_probe_*_results.json" -ErrorAction SilentlyContinue |
                        Where-Object { $_.LastWriteTime -gt $lpStart }).Count
                    $active = @($jobs | Where-Object { $_.State -eq 'Running' } | ForEach-Object Name) -join ','
                    Write-Host ("  ... {0}/{1} LP reports written | elapsed {2:hh\:mm} | active: {3}" -f `
                            $reports, $plan.Count, ((Get-Date) - $lpStart), $active)
                }
                Start-Sleep -Seconds 5
            }
            foreach ($jb in $jobs) {
                Receive-Job -Job $jb | ForEach-Object {
                    if ("$_".Trim()) { Write-Host "  [$($jb.Name)] $_" }
                }
            }
            $lpFailed = @($jobs | Where-Object { $_.State -eq 'Failed' } | ForEach-Object Name)
            $jobs | Remove-Job -Force
            if ($lpFailed) { throw "LP job(s) FAILED: $($lpFailed -join ', ')" }
            Write-Host ("  Probes done in {0:hh\:mm\:ss}." -f ((Get-Date) - $lpStart))
        }
    }
    else { Write-Host "`n[2/5] Linear-probe baselines SKIPPED (-SkipProbes)." }

    # --- Stage 3: confirm the SAE side carries the standardised metric -----
    # Checked rather than assumed: a registry row written before mcc_at_pref
    # existed still rolls up into families, so a missing metric would surface
    # as a page of "n/a" efficiencies with no explanation.
    Write-Host "`n[3/5] Checking the SAE registry carries mcc_at_pref..."
    $check = @'
import json, sys
reg = json.load(open("saes/quarto/eval_registry.json"))
champs = sys.argv[1].split(",")
scope = {k: v for k, v in reg.items() if any(f"champ{c}" in k for c in champs)}
stale = [k for k, v in scope.items()
         if "coverage_mcc_at_pref" not in (v.get("metrics") or {})]
print(f"  in scope ({'/'.join(champs)}): {len(scope) - len(stale)}/{len(scope)} "
      f"row(s) carry coverage_mcc_at_pref.")
if stale:
    for k in sorted(stale)[:8]:
        print(f"    stale: {k}")
    if len(stale) > 8:
        print(f"    ... and {len(stale) - 8} more")
    print("  -> run: python scripts/backfill_eval_metrics.py --game=quarto")
    print("     (or re-evaluate those checkpoints) BEFORE trusting stage 5.")
sys.exit(1 if stale else 0)
'@
    # Advisory, not fatal: a stale registry is worth finishing the LP work for,
    # and stage 5 excludes any basis whose row is stale rather than guessing.
    $prev = $PSNativeCommandUseErrorActionPreference
    $PSNativeCommandUseErrorActionPreference = $false
    $check | python - ($Champions -join ',')
    $saeReady = ($LASTEXITCODE -eq 0)
    $PSNativeCommandUseErrorActionPreference = $prev
    if (-not $saeReady) {
        Write-Host '  WARNING: continuing so the LP work is not wasted, but the'
        Write-Host '  stage-5 verdict will exclude any basis whose row is stale.'
    }

    # --- Stage 4: per-champion basis comparison ----------------------------
    # ONE SAE and ONE set of codes across all three framings, so the model, the
    # positions and the dictionary are held fixed and only the CONCEPT FRAMING
    # varies. This is the controlled version of the 2026-05-22 comparison.
    Write-Host "`n[4/5] Basis comparison (one SAE, one code set, three framings)..."
    foreach ($champ in $Champions) {
        foreach ($hook in $HOOKS) {
            $cached = @(Get-ChildItem "saes/$GAME/cache/*champ$champ*$hook`_h.pt" -ErrorAction SilentlyContinue |
                ForEach-Object { $_.Name -replace '_h\.pt$', '' })
            # Preference order, NOT filesystem order. F04/E05 are the
            # UNSUPERVISED panel members and are what the verdict is about;
            # I04 is the anchored positive control and only stands in when no
            # unsupervised run has a cache, in which case say so.
            $rid = $null
            foreach ($prefix in @('F04', 'E05', 'I04')) {
                $rid = @($cached | Where-Object { $_ -like "$prefix-*" } | Sort-Object | Select-Object -First 1)[0]
                if ($rid) { break }
            }
            if (-not $rid) {
                Write-Host "  [SKIP] champ$champ/$hook -- no _h cache for a panel run."
                continue
            }
            if ($rid -like 'I04-*') {
                Write-Host "  NOTE: champ$champ/$hook falls back to the ANCHORED control"
                Write-Host "        ($rid) -- no unsupervised _h cache. Its numbers"
                Write-Host '        describe what supervision achieves, not what the'
                Write-Host '        unsupervised dictionary finds.'
            }
            Write-Host "`n  >> $rid"
            if ($DryRun) { Write-Host "    [DRY] basis_comparison.py --run-id=$rid --champ=$champ"; continue }
            python scripts/basis_comparison.py --run-id=$rid --champ=$champ
        }
    }

    # --- Stage 5: the verdict ----------------------------------------------
    Write-Host "`n[5/5] SAE-vs-LP efficiency and the hawk-vs-tiger verdict..."
    foreach ($champ in $Champions) {
        foreach ($hook in $HOOKS) {
            Write-Host "`n  >> champ$champ / $hook"
            if ($DryRun) { Write-Host "    [DRY] sae_lp_efficiency.py --champ=$champ --hook=$hook"; continue }
            # Non-fatal: a champion/hook with incomplete inputs reports exactly
            # what is missing and exits 1; the remaining cells still run.
            $prev = $PSNativeCommandUseErrorActionPreference
            $PSNativeCommandUseErrorActionPreference = $false
            python scripts/sae_lp_efficiency.py --champ=$champ --hook=$hook
            $PSNativeCommandUseErrorActionPreference = $prev
        }
    }

    if ($DryRun) { Write-Host "`nDryRun -> nothing executed."; return }

    Write-Host "`nEmitting stage plan..."
    $files = @(Get-ChildItem "saes/$GAME/analysis/*_sae-lp-efficiency.json",
        "saes/$GAME/analysis/*_basis-comparison.json" -ErrorAction SilentlyContinue |
        ForEach-Object { $_.FullName })
    $files += @(Get-ChildItem "$DATA/linear_probe_*_results.json" | ForEach-Object { $_.FullName })
    $files += @(Get-ChildItem "$DATA/bsp_schema-*.json" | ForEach-Object { $_.FullName })
    python scripts/emit_stage.py --slug basis-verdict @files

    Write-Host "`nDone. Read the verdict with:"
    Write-Host "  python scripts/registry_query.py triads"
    Write-Host "  python scripts/registry_query.py family <run_id> --bsps=tiger<Champ>"
    Write-Host "  saes/$GAME/analysis/*_sae-lp-efficiency.json"
    if (-not $saeReady) {
        Write-Host "`nNOTE: some registry rows lacked mcc_at_pref (stage 3). Run"
        Write-Host "  python scripts/backfill_eval_metrics.py --game=quarto"
        Write-Host "then re-run this runner with -SkipProbes to refresh the verdict."
    }
}
finally {
    Stop-Transcript | Out-Null
}
