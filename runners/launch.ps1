#Requires -Version 7
<#
  runners/launch.ps1 -- fire-and-forget launcher that SURVIVES SSH disconnect.

  Why not Start-Process: Windows OpenSSH puts the SSH session in a job object and
  kills the whole process tree on disconnect. A plain Start-Process child stays
  in that job and dies with the session -- same failure as `nohup`. Instead we
  ask the WMI service (Win32_Process.Create) to spawn the runner: the new process
  is a child of WmiPrvSE, NOT of the sshd session, so it keeps running after you
  disconnect. The runner logs itself via Start-Transcript, so no stdio wiring is
  needed here.

  Usage:
    pwsh -File runners\launch.ps1 champYb
    pwsh -File runners\launch.ps1 3A-dilution

  (For survival across a full user LOGOFF, not just SSH disconnect, use Task
  Scheduler instead -- ask and I'll add a schtasks variant.)
#>
param([Parameter(Mandatory)][string]$Name)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runner = Join-Path $repo "runners\$Name.ps1"
if (-not (Test-Path $runner)) { throw "No such runner: $runner" }
New-Item -ItemType Directory -Force -Path (Join-Path $repo 'logs') | Out-Null

$cmd = "pwsh -NoProfile -File `"$runner`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine      = $cmd
    CurrentDirectory = $repo
}
if ($r.ReturnValue -ne 0) {
    throw "Win32_Process.Create failed (return code $($r.ReturnValue))"
}

$transcript = Join-Path $repo "logs\$Name.transcript.log"
Write-Host "Launched '$Name' detached via WMI (PID $($r.ProcessId))."
Write-Host "  It survives SSH disconnect (spawned outside the sshd job)."
Write-Host "  follow:  Get-Content '$transcript' -Wait -Tail 40"
Write-Host "  stop:    Stop-Process -Id $($r.ProcessId)   # (children may need separate kill)"
