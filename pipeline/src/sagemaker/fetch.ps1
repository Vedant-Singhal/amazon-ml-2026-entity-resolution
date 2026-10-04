# Download a run's submission files and small reports (not the large intermediate parquet/npy).
#   powershell -ExecutionPolicy Bypass -File src\sagemaker\fetch.ps1 -RunName v5 [-Dest output]
param([Parameter(Mandatory)][string]$RunName, [string]$Dest = "")
$ErrorActionPreference = "Stop"
if (-not (Get-Command aws -ErrorAction SilentlyContinue)) { $env:Path += ";C:\Program Files\Amazon\AWSCLIV2" }
. (Join-Path $PSScriptRoot "account.ps1")
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $Dest) { $Dest = Join-Path $root "runs\$RunName" }
$s3 = "s3://$Bucket/$Prefix/ber-runs/$RunName"
aws s3 sync "$s3/output/" "$Dest\output" --region $Region --only-show-errors
aws s3 sync "$s3/work/" "$Dest\work" --region $Region --only-show-errors --exclude "*" `
    --include "*.json" --include "model/*.txt" --include "model/*.pkl"
Get-ChildItem $Dest -Recurse -File | Select-Object @{n = "File"; e = { $_.FullName.Substring($Dest.Length + 1) } }, Length
