# Download an EDA job's reports into eda/runs/<job>/ - works mid-run for jobs whose reports
# stream to S3 (eda-live/), and for older jobs from the final model.tar.gz.
param([Parameter(Mandatory)][string]$Job)
$ErrorActionPreference = "Stop"
if (-not (Get-Command aws -ErrorAction SilentlyContinue)) { $env:Path += ";C:\Program Files\Amazon\AWSCLIV2" }

$Region = "us-east-1"
$Bucket = "<your-bucket>"
$status = aws sagemaker describe-training-job --training-job-name $Job --region $Region --query "[TrainingJobStatus,FailureReason]" --output text
Write-Host "Job status: $status"

$dest = Join-Path $PSScriptRoot "runs\$Job"
New-Item -ItemType Directory -Force $dest | Out-Null
$live = aws s3 ls "s3://$Bucket/MLChallenge/eda-live/$Job/reports/" --region $Region
if ($live) {
    aws s3 sync "s3://$Bucket/MLChallenge/eda-live/$Job/reports/" "$dest\reports" --region $Region --only-show-errors
} else {
    aws s3 cp "s3://$Bucket/MLChallenge/eda-runs/$Job/output/model.tar.gz" "$dest\model.tar.gz" --region $Region --only-show-errors
    if ($LASTEXITCODE -ne 0) { throw "no output yet (job still running, or failed before writing output)" }
    & "$env:SystemRoot\System32\tar.exe" -xzf "$dest\model.tar.gz" -C $dest
    Remove-Item "$dest\model.tar.gz"
}
Get-ChildItem "$dest\reports" -Recurse -File | Select-Object Name, Length
