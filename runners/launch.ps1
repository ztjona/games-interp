#Requires -Version 7
<#
  runners/launch.ps1 -- fire-and-forget launcher for a runner on Deep Brain.

  Starts runners/<Name>.ps1 as a DETACHED process (survives SSH logout -- the
  Windows replacement for the unreliable `nohup`), with stdout/stderr going to
  logs/<Name>.out / logs/<Name>.err. Returns immediately with the PID.

  Usage:
    pwsh -File runners\launch.ps1 champYb
    pwsh -File runners\launch.ps1 3A-dilution

  Watch progress:  Get-Content logs\champYb.out -Wait
#>
param([Parameter(Mandatory)][string]$Name)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

$runner = "runners/$Name.ps1"
if (-not (Test-Path $runner)) { throw "No such runner: $runner" }
New-Item -ItemType Directory -Force -Path logs | Out-Null

$p = Start-Process -FilePath 'pwsh' `
    -ArgumentList @('-NoProfile', '-File', $runner) `
    -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "logs/$Name.out" `
    -RedirectStandardError  "logs/$Name.err"

Write-Host "Launched $runner detached (PID $($p.Id))."
Write-Host "  logs:   logs/$Name.out  logs/$Name.err"
Write-Host "  follow: Get-Content logs/$Name.out -Wait"
