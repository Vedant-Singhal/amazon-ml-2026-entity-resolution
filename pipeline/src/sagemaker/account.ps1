# The only account-specific settings - fill in your own.
# Bucket: any S3 bucket the SageMaker execution role can read and write (e.g. sagemaker-<region>-<account-id>).
# Role: the SageMaker execution role ARN (needs S3 access to the bucket).
$Region  = "us-east-1"
$Bucket  = "<your-bucket>"
$Role    = "arn:aws:iam::<account-id>:role/<your-sagemaker-execution-role>"
$Prefix  = "ber"                                  # top-level S3 folder for code, runs, outputs
$DataUri = "s3://$Bucket/$Prefix/dataset/"        # folder holding train/ and test/ TSVs
# optional: $env:AWS_PROFILE = "<cli-profile>"
