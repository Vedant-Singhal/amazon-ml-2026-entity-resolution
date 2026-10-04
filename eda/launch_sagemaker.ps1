# Run the EDA suite as a SageMaker training job on one GPU; data is read from S3.
#   powershell -ExecutionPolicy Bypass -File eda\launch_sagemaker.ps1 [-Smoke] [-Steps "02 04"] [-InstanceType ml.g5.8xlarge]
param(
    [switch]$Smoke,
    [string]$Steps = "",
    [string]$InstanceType = "ml.g5.8xlarge",   # 1x A10G 24GB, 32 vCPU, 128GB RAM
    [int]$MaxHours = 4
)
$ErrorActionPreference = "Stop"
if (-not (Get-Command aws -ErrorAction SilentlyContinue)) { $env:Path += ";C:\Program Files\Amazon\AWSCLIV2" }

$Region = "us-east-1"
$Bucket = "<your-bucket>"
$Role   = "<your-sagemaker-execution-role-arn>"
$Image  = "763104351884.dkr.ecr.$Region.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker"
$Job    = "eda-" + $(if ($Smoke) { "smoke-" } else { "" }) + (Get-Date -Format "yyyyMMdd-HHmmss")

$here = $PSScriptRoot
$tgz = Join-Path $env:TEMP "$Job-sourcedir.tar.gz"
& "$env:SystemRoot\System32\tar.exe" -czf $tgz -C $here --exclude reports --exclude cache --exclude runs --exclude __pycache__ .
if ($LASTEXITCODE -ne 0) { throw "tar failed" }
$code = "s3://$Bucket/MLChallenge/eda-code/$Job/sourcedir.tar.gz"
aws s3 cp $tgz $code --region $Region --only-show-errors
if ($LASTEXITCODE -ne 0) { throw "upload failed" }
Remove-Item $tgz

# Reports go to the checkpoint dir, which SageMaker syncs to S3 continuously (readable mid-run).
$envVars = @{ EDA_CACHE = "/tmp/eda_cache"; HF_HOME = "/tmp/hf"; EDA_REPORTS = "/opt/ml/checkpoints/reports" }
if ($Smoke) { $envVars.EDA_SMOKE = "1" }
if ($Steps) { $envVars.EDA_STEPS = $Steps }

$spec = [ordered]@{
    TrainingJobName = $Job
    RoleArn = $Role
    AlgorithmSpecification = @{ TrainingImage = $Image; TrainingInputMode = "File" }
    # Values are JSON-encoded, as the SageMaker SDK does for script mode.
    HyperParameters = @{ sagemaker_program = '"run_all.py"'; sagemaker_submit_directory = "`"$code`"" }
    InputDataConfig = @(@{
        ChannelName = "dataset"
        DataSource = @{ S3DataSource = @{ S3DataType = "S3Prefix"; S3Uri = "s3://$Bucket/MLChallenge/dataset/"; S3DataDistributionType = "FullyReplicated" } }
    })
    OutputDataConfig = @{ S3OutputPath = "s3://$Bucket/MLChallenge/eda-runs/" }
    CheckpointConfig = @{ S3Uri = "s3://$Bucket/MLChallenge/eda-live/$Job/"; LocalPath = "/opt/ml/checkpoints" }
    ResourceConfig = @{ InstanceType = $InstanceType; InstanceCount = 1; VolumeSizeInGB = 100 }
    StoppingCondition = @{ MaxRuntimeInSeconds = $MaxHours * 3600 }
    Environment = $envVars
}
$specFile = Join-Path $env:TEMP "$Job.json"
[IO.File]::WriteAllText($specFile, ($spec | ConvertTo-Json -Depth 8))
aws sagemaker create-training-job --cli-input-json "file://$specFile" --region $Region --query TrainingJobArn --output text
if ($LASTEXITCODE -ne 0) { throw "create-training-job failed" }
Remove-Item $specFile
Write-Host "Started $Job on $InstanceType"
Write-Host "Logs:   aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix $Job --follow --region $Region"
Write-Host "Fetch:  powershell -ExecutionPolicy Bypass -File eda\fetch_results.ps1 -Job $Job"
