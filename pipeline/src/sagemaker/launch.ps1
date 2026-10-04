# Run a pipeline program as a SageMaker training job (v5: -Program run_v5.py). Stage outputs live in a per-run S3 checkpoint,
# so relaunching with the same -RunName resumes from cached stages.
#   powershell -ExecutionPolicy Bypass -File src\sagemaker\launch.ps1 -RunName v5 -Program run_v5.py [-SmokeS1 10000] [-V2Inputs]
param(
    [string]$RunName = "v3",
    [string]$Stages = "",
    [int]$SmokeS1 = 0,                           # >0: sample this many train S1 and estimate the full run
    [string]$EmbModel = "",
    [string]$InstanceType = "ml.g6e.8xlarge",   # 1x L40S 48GB, 32 vCPU (16 used: 50% rule), 256GB RAM
    [int]$MaxHours = 10,
    [string]$Program = "run_pipeline.py",
    [string]$ExtraEnv = "",                     # "KEY=VALUE;KEY2=VALUE2"
    [switch]$V2Inputs,                          # attach v2 OOF / test scores (s3://<bucket>/<prefix>/ber-runs/<run>/v2in/{oof,test}/)
    [string]$V2Run = ""                         # run whose v2in/ holds them (default: this run)
)
$ErrorActionPreference = "Stop"
if (-not (Get-Command aws -ErrorAction SilentlyContinue)) { $env:Path += ";C:\Program Files\Amazon\AWSCLIV2" }
. (Join-Path $PSScriptRoot "account.ps1")

$Image  = "763104351884.dkr.ecr.$Region.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker"
if ($SmokeS1 -gt 0 -and $RunName -notlike "smoke*") { $RunName = "smoke-$RunName" }
$Job = "ber-$RunName-" + (Get-Date -Format "MMdd-HHmmss")

$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$tgz = Join-Path $env:TEMP "$Job.tar.gz"
& "$env:SystemRoot\System32\tar.exe" -czf $tgz -C (Join-Path $root "src") --exclude __pycache__ . -C $root requirements.txt
if ($LASTEXITCODE -ne 0) { throw "tar failed" }
$code = "s3://$Bucket/$Prefix/ber-code/$Job/sourcedir.tar.gz"
aws s3 cp $tgz $code --region $Region --only-show-errors
if ($LASTEXITCODE -ne 0) { throw "upload failed" }
Remove-Item $tgz

$envVars = @{ BER_WORK = "/opt/ml/checkpoints/work"; BER_OUT = "/opt/ml/checkpoints/output"; HF_HOME = "/tmp/hf";
              BER_INSTANCE = $InstanceType; TOKENIZERS_PARALLELISM = "true"; BER_BUCKET = $Bucket; BER_PREFIX = $Prefix }
if ($SmokeS1 -gt 0) { $envVars.BER_SMOKE_S1 = "$SmokeS1" }
if ($Stages) { $envVars.BER_STAGES = $Stages }
if ($EmbModel) { $envVars.BER_EMB_MODEL = $EmbModel }
foreach ($kv in ($ExtraEnv -split ";" | Where-Object { $_ })) { $k, $v = $kv -split "=", 2; $envVars[$k] = $v }

$spec = [ordered]@{
    TrainingJobName = $Job
    RoleArn = $Role
    AlgorithmSpecification = @{ TrainingImage = $Image; TrainingInputMode = "File" }
    HyperParameters = @{ sagemaker_program = "`"$Program`""; sagemaker_submit_directory = "`"$code`"" }
    InputDataConfig = @(@{
        ChannelName = "dataset"
        DataSource = @{ S3DataSource = @{ S3DataType = "S3Prefix"; S3Uri = $(if ($DataUri) { $DataUri } else { "s3://$Bucket/$Prefix/dataset/" }); S3DataDistributionType = "FullyReplicated" } }
    })
    OutputDataConfig = @{ S3OutputPath = "s3://$Bucket/$Prefix/ber-jobs/" }
    CheckpointConfig = @{ S3Uri = "s3://$Bucket/$Prefix/ber-runs/$RunName/"; LocalPath = "/opt/ml/checkpoints" }
    ResourceConfig = @{ InstanceType = $InstanceType; InstanceCount = 1; VolumeSizeInGB = 400 }
    StoppingCondition = @{ MaxRuntimeInSeconds = $MaxHours * 3600 }
    Environment = $envVars
}
if ($V2Inputs) {
    foreach ($ch in @(@("v2oof", "oof"), @("v2test", "test"))) {
        $spec.InputDataConfig += @{ ChannelName = $ch[0]; DataSource = @{ S3DataSource = @{ S3DataType = "S3Prefix";
            S3Uri = "s3://$Bucket/$Prefix/ber-runs/$(if ($V2Run) { $V2Run } else { $RunName })/v2in/$($ch[1])/"; S3DataDistributionType = "FullyReplicated" } } }
    }
}
$specFile = Join-Path $env:TEMP "$Job.json"
[IO.File]::WriteAllText($specFile, ($spec | ConvertTo-Json -Depth 8))
aws sagemaker create-training-job --cli-input-json "file://$specFile" --region $Region --query TrainingJobArn --output text
if ($LASTEXITCODE -ne 0) { throw "create-training-job failed" }
Remove-Item $specFile
Write-Host "Started $Job ($InstanceType) run=$RunName"
Write-Host "Logs:  aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix $Job --follow --region $Region"
