#Requires -Version 7
<#
  runners/champYb.ps1 -- champYb end-to-end on Deep Brain (Windows PowerShell 7).
  data pipeline-to-parity -> LP -> SAE sweep (train) -> eval -> Atlas export ->
  stage plan. Self-contained, committed (see CLAUDE.md "runners/").

  Correctness-first after the 2026-07 incident (a bash runner that parallelized
  eval corrupted 30/43 _h caches and lost 30/43 registry entries to concurrent
  writes, yet printed "complete"). Rules baked in here:
    * Fail-fast: $PSNativeCommandUseErrorActionPreference makes any python
      non-zero exit throw. The runner cannot lie about finishing.
    * Parallel ONLY where each process writes its OWN files (position-gen,
      activation collection, SAE *training*). All GPUs are used for those.
    * Eval is SEQUENTIAL by design: sae_eval writes the shared eval_registry.json
      and a per-run _h cache, so running it in parallel races and corrupts. One
      GPU here is deliberate, not an oversight.
    * Corrupt _h caches are deleted before eval so they regenerate.
    * Re-runnable: every slow step skips when its output already exists, so a
      re-run resumes at eval+export without retraining.

  No tunable flags. Launch detached with:  runners\launch.ps1 champYb
#>

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true   # native non-zero exit -> throw

Set-Location (Join-Path $PSScriptRoot '..')
$env:PYTHONUTF8 = '1'
New-Item -ItemType Directory -Force -Path logs | Out-Null
Start-Transcript -Path 'logs/champYb.transcript.log' -Append | Out-Null

try {
    $CHAMP_YB        = 'models/quarto/20260614_2031-Yb_hotChamp(3)0612_HOT_1.0_seedB_E_10000.pt'
    $CHAMP_YB_RANDOM = 'models/quarto/20260612_1414-Yb_hotChamp(3)0612_HOT_1.0_seedB_E_0000.pt'
    $YB_STEM         = 'Yb_hotChamp'
    $POS             = 'data/quarto/positions-amalgam_yb_unique.pt'
    $BSPS            = @('gorillaYb', 'hawkYb', 'tigerYb')

    # --- GPUs: use all of them for the parallel-safe stages -------------------
    $NGPU = @(nvidia-smi -L).Count
    if ($NGPU -lt 1) { throw 'No GPU detected (nvidia-smi -L). This runner is for Deep Brain.' }
    Write-Host "Using $NGPU GPU(s)."

    # Launch a python process pinned to a GPU, logging to logs/<tag>.{out,err}.
    function Start-Py([int]$Gpu, [string[]]$PyArgs, [string]$Tag) {
        $env:CUDA_VISIBLE_DEVICES = "$Gpu"
        Start-Process -FilePath python -ArgumentList $PyArgs -PassThru -NoNewWindow `
            -RedirectStandardOutput "logs/$Tag.out" -RedirectStandardError "logs/$Tag.err"
    }
    # Wait for a batch of processes; throw if any exited non-zero.
    function Wait-Checked([System.Diagnostics.Process[]]$Procs, [string]$What) {
        foreach ($p in $Procs) { $p.WaitForExit() }
        $bad = @($Procs | Where-Object { $_.ExitCode -ne 0 })
        if ($bad.Count -gt 0) { throw "${What}: $($bad.Count) job(s) failed; see logs/*.err" }
    }

    # --- 0. Competence audit -------------------------------------------------
    if (Test-Path data/quarto/audit-champYb.json) {
        Write-Host '=== Step 0: audit exists, skipping ==='
    } else {
        Write-Host '=== Step 0: Competence audit ==='
        python scripts/model_competence_audit.py --model-config=configs/models/champYb.yaml `
            --num-positions=5000 --device=cuda --output=data/quarto/audit-champYb.json
    }

    # --- 1-2. Positions (4 modes across GPUs) + dedup ------------------------
    if (Test-Path $POS) {
        Write-Host "=== Step 1-2: $POS exists, skipping position gen + dedup ==="
    } else {
        Write-Host '=== Step 1: Generate raw positions (4 modes across GPUs) ==='
        $modes = @('random_v_random', 'model_v_random', 'random_v_model', 'model_v_model')
        $procs = @()
        for ($i = 0; $i -lt $modes.Count; $i++) {
            $m = $modes[$i]
            $procs += Start-Py ($i % $NGPU) @(
                'scripts/generate_positions.py', '--game', 'quarto_s4_hot',
                '--opponents', $m, '--model', $CHAMP_YB, '--num-games', '10000',
                '--seed', '42', '--device', 'cuda', '--output-dir', 'data/quarto_Yb'
            ) "pos_$m"
        }
        Wait-Checked $procs 'position generation'

        Write-Host '=== Step 2: Deduplicate positions ==='
        python scripts/deduplicate_positions.py `
            data/quarto_Yb/positions-random_v_random_raw.pt `
            "data/quarto_Yb/positions-model_v_random-$($YB_STEM)_raw.pt" `
            "data/quarto_Yb/positions-random_v_model-$($YB_STEM)_raw.pt" `
            "data/quarto_Yb/positions-model_v_model-$($YB_STEM)_raw.pt" `
            --output $POS
    }

    # --- 3. BSP labels (skip if present) ------------------------------------
    Write-Host '=== Step 3: Compute BSP labels ==='
    $labelSpecs = @(
        @('gorilla', 'data/quarto/bsp_labels-gorillaYb_164.pt', 'data/quarto/bsp_schema-gorillaYb_164.json'),
        @('hawk',    'data/quarto/bsp_labels-hawkYb_173.pt',    'data/quarto/bsp_schema-hawkYb_173.json'),
        @('tiger',   'data/quarto/bsp_labels-tigerYb_36.pt',    'data/quarto/bsp_schema-tiger_36.json')
    )
    foreach ($s in $labelSpecs) {
        if (Test-Path $s[1]) { Write-Host "  $($s[1]) exists, skipping"; continue }
        python scripts/compute_bsp_labels.py $POS --game quarto_s4_hot --name $s[0] `
            --output $s[1] --schema-out $s[2]
    }

    # --- 4. Activations (4 jobs across GPUs, skip if present) ----------------
    Write-Host '=== Step 4: Collect activations (across GPUs) ==='
    # each: @(model, hook, outfile, flattenBool)
    $actJobs = @(
        @($CHAMP_YB,        's4.fc1',   'data/quarto/s4.fc1_amalgam_yb_activations.pt',          $false),
        @($CHAMP_YB_RANDOM, 's4.fc1',   'data/quarto/s4.fc1_amalgam_yb_random_activations.pt',   $false),
        @($CHAMP_YB,        's4.conv2', 'data/quarto/s4.conv2_amalgam_yb_activations.pt',        $true),
        @($CHAMP_YB_RANDOM, 's4.conv2', 'data/quarto/s4.conv2_amalgam_yb_random_activations.pt', $true)
    )
    $procs = @(); $i = 0
    foreach ($j in $actJobs) {
        if (Test-Path $j[2]) { Write-Host "  $($j[2]) exists, skipping"; continue }
        $a = @('scripts/collect_activations.py', $j[0], '--hook', $j[1], '--game', 'quarto_s4_hot',
               '--positions-file', $POS, '--output', $j[2], '--device', 'cuda')
        if ($j[3]) { $a += '--flatten-position' }
        $procs += Start-Py ($i % $NGPU) $a ("act_" + ($j[2] -replace '[^\w]', '_'))
        $i++
    }
    if ($procs.Count -gt 0) { Wait-Checked $procs 'activation collection' }

    # --- 5. Linear-probe baselines (skip if result present) -----------------
    Write-Host '=== Step 5: Linear-probe baselines ==='
    foreach ($hook in @('s4.fc1', 's4.conv2')) {
        $lp = @(
            @("data/quarto/${hook}_amalgam_yb_activations.pt", 'data/quarto/bsp_labels-gorillaYb_164.pt', 'data/quarto/bsp_schema-gorillaYb_164.json', "gorilla_164_${hook}"),
            @("data/quarto/${hook}_amalgam_yb_activations.pt", 'data/quarto/bsp_labels-hawkYb_173.pt',    'data/quarto/bsp_schema-hawkYb_173.json',    "hawk_173_${hook}"),
            @("data/quarto/${hook}_amalgam_yb_activations.pt", 'data/quarto/bsp_labels-tigerYb_36.pt',    'data/quarto/bsp_schema-tiger_36.json',      "tiger_36_${hook}")
        )
        foreach ($p in $lp) {
            $res = "data/quarto/linear_probe_$($p[3])_amalgam_yb_activations_results.json"
            if (Test-Path $res) { Write-Host "  $res exists, skipping"; continue }
            python scripts/linear_probe_baseline.py $p[0] $p[1] $p[2]
        }
    }

    # --- 6. SAE sweep: TRAIN ONLY, sharded across all GPUs -------------------
    # No --eval here: parallel shards must not write the shared registry.
    Write-Host "=== Step 6: SAE sweep TRAIN across $NGPU GPU(s) (no eval) ==="
    Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    $procs = @()
    for ($g = 0; $g -lt $NGPU; $g++) {
        $procs += Start-Process -FilePath python -PassThru -NoNewWindow `
            -RedirectStandardOutput "logs/sweep_gpu$g.out" -RedirectStandardError "logs/sweep_gpu$g.err" `
            -ArgumentList @('run_sweep.py', '--configs=configs/champYb', "--gpu=$g",
                            "--split=$($g + 1)/$NGPU", '--skip-existing')
    }
    Wait-Checked $procs 'SAE training sweep'

    # --- 6b. Repair: delete any corrupt _h caches so eval regenerates them ---
    Write-Host '=== Step 6b: Clean corrupt caches ==='
    python scripts/clean_corrupt_caches.py 'saes/quarto/cache/*champYb*_h.pt'

    # --- 7. Eval: SEQUENTIAL over all champYb SAEs x 3 BSP sets --------------
    # Sequential on purpose (registry + _h are shared/serial). sae_eval skips
    # already-evaluated run_id:bsp_set, so a resume only fills the gaps.
    Write-Host '=== Step 7: Eval (sequential, all BSP sets) ==='
    $env:CUDA_VISIBLE_DEVICES = '0'
    $ckpts = @(Get-ChildItem 'saes/quarto/*champYb*.pt' | Select-Object -ExpandProperty FullName)
    $failed = @()
    foreach ($ck in $ckpts) {
        foreach ($b in $BSPS) {
            try { python sae_eval.py evaluate $ck --bsps=$b }
            catch { $failed += "$(Split-Path $ck -Leaf):$b"; Write-Warning "eval failed: $_" }
        }
    }
    python scripts/registry_query.py top --bsps=gorillaYb --limit=20
    if ($failed.Count -gt 0) { throw "Eval: $($failed.Count) (ckpt:bsp) failed: $($failed -join ', ')" }

    # --- 8. Atlas export, scoped to champYb (correct positions pool) ---------
    Write-Host '=== Step 8: Atlas viz export (champYb only) ==='
    python scripts/export_viz_data.py --game quarto --bsps gorillaYb `
        --champion Yb --positions $POS

    # --- 9. Stage plan (stage_champYb.md at repo root) ----------------------
    Write-Host '=== Step 9: Stage plan ==='
    python scripts/emit_stage.py --slug champYb `
        --shipped-jsonl saes/quarto/shipped_saes.jsonl `
        data/quarto/audit-champYb.json `
        data/quarto/bsp_schema-gorillaYb_164.json `
        data/quarto/bsp_schema-hawkYb_173.json `
        data/quarto/bsp_schema-tiger_36.json `
        (Get-ChildItem 'data/quarto/linear_probe_*_yb_activations_results.json' | ForEach-Object { $_.FullName }) `
        saes/quarto/training_registry.json `
        saes/quarto/eval_registry.json `
        saes/quarto/shipped_saes.jsonl

    Write-Host ''
    Write-Host '=== champYb runner COMPLETE; commit plan in stage_champYb.md ==='
}
finally {
    Stop-Transcript | Out-Null
}
