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
        'E05-champYb-s42-batchtopk-k32-exp8-s4.conv2|tigerYb|none',
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
        'Ta|s4.fc1'   = 'R1-champTarandom-s42-jumprelu-t64-exp8-s4.fc1'
        'Ve|s4.fc1'   = 'R1-champVerandom-s42-jumprelu-t64-exp8-s4.fc1'
        'Yb|s4.fc1'   = 'R1-champYbrandom-s42-jumprelu-t64-exp8-s4.fc1'
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
    Write-Host "`nPre-flight (input presence):"
    $runnable = @()
    foreach ($entry in $RUNS) {
        $rid, $bsps, $rand = $entry.Split('|')
        $why = ''
        if (-not (Test-Path "saes/quarto/$rid.pt")) { $why += ' no checkpoint;' }
        if (-not (Test-Labels $bsps)) { $why += " no bsp_labels-$bsps;" }
        $hasH = Test-Path "$CACHE/$($rid)_h.pt"
        if ($why) { Write-Host "  [SKIP] $rid  bsps=$bsps --$why" }
        else { Write-Host "  [ OK ] $rid  bsps=$bsps$(if(-not $hasH){'  (will regen _h)'})"; $runnable += $entry }
    }

    if ($runnable.Count -eq 0) {
        Write-Host "`nNothing runnable (need the champTa/champVe checkpoints + BSP labels on this box)."
        return
    }
    if ($DryRun) { Write-Host "`nDryRun -> not executing. $($runnable.Count) run(s) would execute."; return }

    Write-Host "`nExecuting $($runnable.Count) run(s)..."
    foreach ($entry in $runnable) {
        $rid, $bsps, $rand = $entry.Split('|')
        # Upgrade to the random-model control when one is registered for this
        # champion+hook; otherwise the entry's own value ('none') stands.
        if ($rid -match 'champ(\w\w)-.*-(s4\.\w+|fc1|conv2)$') {
            $key = "$($Matches[1])|$($Matches[2])"
            if ($RANDOM_CONTROLS.ContainsKey($key)) { $rand = $RANDOM_CONTROLS[$key] }
        }
        Write-Host "`n>> $rid  bsps=$bsps  random=$rand"
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

    Write-Host "`nWriting combined gate summary..."
    $agg = @'
import json, glob
from statistics import median

out = "saes/quarto/analysis/3A_gate_summary.json"
rows = []
for p in sorted(glob.glob("saes/quarto/analysis/*_dilution-*.json")):
    with open(p) as f: d = json.load(f)
    s, g = d["summary"], d["gate_g3a"]
    cs = [c for c in d["concepts"] if "asymptote_r2" in c]
    # Continuous companions to the 4-way verdict. The verdict is a thresholded
    # view; these are the quantities to report and plot, because they separate
    # unsupervised from anchored without depending on any threshold.
    rows.append({"run_id": s["run_id"], "bsp_set": s["bsp_set"],
                 "rule_version": s["config"].get("rule_version", "3A.1"),
                 "random_control": s.get("random_control"),
                 "n_threat_bsps": g["n_threat_bsps"], "n_diluted": g["n_diluted"],
                 "n_tiled": g["n_tiled"], "n_captured": g["n_captured"],
                 "n_absent": g["n_absent"], "geometric_frac": g["geometric_frac"],
                 "median_solo_frac": round(median([c.get("solo_frac", 0.0) for c in cs]), 4) if cs else 0.0,
                 "median_intrinsic_dim": round(median([c["intrinsic_dim"] for c in cs]), 4) if cs else 0.0,
                 "median_top_phi": round(median([abs(c["top_phi"]) for c in cs]), 4) if cs else 0.0,
                 "mean_asymptote_r2": round(sum(c["asymptote_r2"] for c in cs) / len(cs), 4) if cs else 0.0,
                 "verdict": g["verdict"]})
json.dump({"runs": rows}, open(out, "w"), indent=2, sort_keys=True)

hdr = f"{'run_id':<50}{'bsps':<11}{'geom':>6}{'solo':>6}{'idim':>6}{'|phi|':>7}{'R2':>7}  verdict"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['run_id'][:49]:<50}{r['bsp_set']:<11}{r['geometric_frac']:>6.2f}"
          f"{r['median_solo_frac']:>6.2f}{r['median_intrinsic_dim']:>6.2f}"
          f"{r['median_top_phi']:>7.3f}{r['mean_asymptote_r2']:>7.3f}  {r['verdict']}")
stale = {r["rule_version"] for r in rows}
if stale != {"3A.2"}:
    print(f"\nWARNING: mixed verdict rules across reports {sorted(stale)}. "
          f"Run: python scripts/dilution_diagnostic.py reclassify "
          f"saes/quarto/analysis/*_dilution-*.json")
if any(r["random_control"] is None for r in rows):
    print("\nNOTE: some runs used the permutation null, not a random-model SAE "
          "control. 'absent' verdicts from those runs are provisional "
          "(see runners/3A-dilution.ps1 $RANDOM_CONTROLS).")
print("\nSaved:", out)
print("geom = (diluted+tiled)/threat BSPs; solo = median share of recoverable")
print("signal in ONE latent; idim = median effective dims; R2 = mean total")
print("recoverable signal. Full glossary: 'glossary' key in each report JSON.")
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
