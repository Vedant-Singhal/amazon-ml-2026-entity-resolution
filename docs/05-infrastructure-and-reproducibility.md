# 5. Infrastructure and reproducibility

There are three ways to run the project, from cheapest to complete:

| option | needs | runs | time |
|---|---|---|---|
| **A. Final blend only** | any computer, Python with `numpy` + `pyarrow` | `pipeline/src/run_final.py` on the two models' test-score files | ≈ 3 min |
| **B. Full pipeline on your own GPU machine** | Linux (or WSL2), 1 NVIDIA GPU ≥ 24 GB, 256 GB RAM, 400 GB disk | `pipeline/src/run.sh`: auxiliary matcher → Model A → Model B → blend | ≈ 13–15 h |
| **C. Full pipeline on AWS SageMaker** | an AWS account with GPU training quota, an S3 bucket, an execution role, the AWS CLI | the same stages as training jobs | ≈ 13–15 h of GPU time |

The competition data is **not in this repository** (it belongs to the organizers and is about 2.5 GB). Put it in
`dataset/` at the repository root, where `.gitignore` keeps it out of git:

```
dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
dataset/test/{test_source1,test_source2,test_source3}.tsv
```

## A. Reproduce the submitted file from the two models' scores

`run_final.py` needs only `numpy` and `pyarrow`. Given Model A's and Model B's test scores
(`work_*/test/scored3.parquet`, written by the full pipeline) it rebuilds `matching_results.tsv` and
`candidate_pairs.tsv`. We checked it on this repository's code: the output is **byte-identical** to the uploaded
0.984403 submission, and with `FINAL_CHECK=1` it reprints the training check that chose the weights (Model A 0.99021,
blend 0.99031).

```bash
V4_WORK=path/to/work_v4 V5_WORK=path/to/work_v5a DATA=dataset FINAL_OUT=submission FINAL_CHECK=1 \
    python pipeline/src/run_final.py
```

`FINAL_CHECK=1` also needs each run's `model3/oof.parquet` and `dataset/train/train_ground_truth.tsv`.

## B. Full pipeline on your own GPU machine

1. **Environment** (Python 3.12):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   # install the PyTorch build that matches your CUDA driver (see pytorch.org), then:
   pip install -r pipeline/requirements.txt
   ```
2. **Data:** place it in `dataset/` at the repository root, or point `DATA` at any folder holding `train/` and
   `test/`.
3. **Smoke test** on a 10k-S1 sample (about 30 minutes), then the **full run**:
   ```bash
   SMOKE=1 bash pipeline/src/run.sh        # writes *_smoke folders
   bash pipeline/src/run.sh                # writes pipeline/work_*/, output_*/ and pipeline/submission/
   ```
   Every stage caches its outputs, so an interrupted run resumes at the first missing stage. Models download from
   Hugging Face on first use.
4. **Validate** with the organizers' `validate_submission.py` (shipped with the challenge data):
   `python validate_submission.py -m pipeline/submission/matching_results.tsv -c pipeline/submission/candidate_pairs.tsv -t dataset/test`

Resources: peak ≈ 140 GB RAM during stage-1 training; ≈ 400 GB of cached stages for both models. The code uses 50%
of the CPU cores (`BER_CPU_FRAC=0.7` for 70%).

## C. Full pipeline on AWS SageMaker with your own account

The helpers are PowerShell scripts in [`pipeline/src/sagemaker/`](../pipeline/src/sagemaker/) (Windows PowerShell, or
`pwsh` on Linux/macOS). They package `pipeline/src` + `requirements.txt`, start a training job in the
`pytorch-training:2.9.0-gpu-py312-cu130` container, and keep every stage in an S3 checkpoint per run name, so a
relaunch with the same `-RunName` resumes.

**Prerequisites**
- **SageMaker training-job quota** for a GPU instance. We used `ml.g6e.16xlarge` (1× L40S, 64 vCPU, 512 GB);
  `ml.g6.16xlarge` (1× L4) also works.
- **An S3 bucket** and **a SageMaker execution role** that can read and write it (e.g. `AmazonSageMakerFullAccess`).
- **AWS CLI v2** configured with credentials (`aws configure`, or a named profile) allowed to call
  `sagemaker:CreateTrainingJob`, `iam:PassRole` on that role, and S3 on the bucket.

**Steps** (from the `pipeline/` folder)

1. Fill in `src/sagemaker/account.ps1`: `$Region`, `$Bucket`, `$Role`, `$Prefix`, and optionally
   `$env:AWS_PROFILE`. No credentials go in any file; the AWS CLI provides them.
2. Upload the data: `aws s3 sync ../dataset s3://<bucket>/<prefix>/dataset/`
3. **Auxiliary matcher (v2)**. It reads its data path from `BER_DATA`:
   ```powershell
   .\src\sagemaker\launch.ps1 -RunName v2 -Program v2/run_pipeline.py -InstanceType ml.g6e.16xlarge -MaxHours 8 `
       -ExtraEnv "BER_DATA=/opt/ml/input/data/dataset"
   ```
   Then stage its scores where Model A and Model B read them:
   ```bash
   aws s3 cp s3://<bucket>/<prefix>/ber-runs/v2/work/model/oof.parquet    s3://<bucket>/<prefix>/ber-runs/v4/v2in/oof/oof.parquet
   aws s3 cp s3://<bucket>/<prefix>/ber-runs/v2/work/test/scored.parquet  s3://<bucket>/<prefix>/ber-runs/v4/v2in/test/scored.parquet
   ```
4. **Model A (v4)**:
   ```powershell
   .\src\sagemaker\launch.ps1 -RunName v4 -Program run_v4.py -InstanceType ml.g6e.16xlarge -MaxHours 9 -V2Inputs `
       -ExtraEnv "BER_EMB_MODELS=intfloat/multilingual-e5-small,intfloat/multilingual-e5-large;BER_ANN_MIN=1000000000;BER_SL_KEEP=0.999;BER_V4_RETRIEVER=1;BER_V4_EXTRA_T=0.95;PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
   ```
5. **Model B (v5a)**, reusing Model A's candidate pool and v2 scores:
   ```powershell
   .\src\sagemaker\launch.ps1 -RunName v5a -Program run_v5.py -InstanceType ml.g6e.16xlarge -MaxHours 6 -V2Inputs -V2Run v4 `
       -ExtraEnv "BER_SEED_FROM=s3://<bucket>/<prefix>/ber-runs/v4/work;BER_SEED_SET=reprep;BER_V4_RETRIEVER=0"
   ```
6. **Blend** (option A) after downloading the two score files:
   ```bash
   for r in v4 v5a; do
     aws s3 cp s3://<bucket>/<prefix>/ber-runs/$r/work/test/scored3.parquet  runs/$r/work/test/scored3.parquet
     aws s3 cp s3://<bucket>/<prefix>/ber-runs/$r/work/model3/oof.parquet    runs/$r/work/model3/oof.parquet
   done
   V4_WORK=runs/v4/work V5_WORK=runs/v5a/work DATA=../dataset FINAL_CHECK=1 python src/run_final.py
   ```
   `src/sagemaker/fetch.ps1 -RunName <run>` downloads a run's submission files and small reports.

Add `-SmokeS1 10000` to any launch to run that stage chain on a sample first. Logs:
`aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix ber-<run> --follow`.

## Compute we used

| instance | GPU | vCPU / RAM | used for | on-demand price |
|---|---|---|---|---|
| ml.g6.16xlarge | 1× L4 24 GB | 64 / 256 GB | v2 full run, smoke tests, v5c | ≈ $4.25/h |
| ml.g6e.8xlarge | 1× L40S 48 GB | 32 / 256 GB | v3 full run, embedder ablations | ≈ $5.66/h |
| ml.g6e.16xlarge | 1× L40S 48 GB | 64 / 512 GB | v4, v5a, retriever, blend checks | ≈ $9.47/h |
| ml.g5.16xlarge | 1× A10G 24 GB | 64 / 256 GB | fallback when L40S capacity was short | |
| ml.g4dn.2xlarge | 1× T4 16 GB | 8 / 32 GB | small analysis jobs | ≈ $1/h |

Typical cost per full run: v2 ≈ $22 (5.2 h), v4 ≈ $20 (2.1 h billed on top of cached stages).

**Measured stage times**

| stage | v2 (L4, 32 cores) | v4 (L40S, 32 cores) |
|---|---|---|
| prepare | 34 min | cached from v3 |
| embed | 44 min | cached (e5-small + e5-large ≈ 35–40 min on L40S) |
| search | 57 min | cached (≈ 40 min); retriever merge 1.4 min |
| stage 1 (shortlist) | — | 48 min |
| features | 61 min | 10 min |
| stage 2 | 102 min (LightGBM, CPU) | 20 min (XGBoost, GPU) |
| cross-encoder / graph / stage 3 | — | 25 / 1 / 11 min |

**How the leaderboard runs were executed.** The pipeline was developed in versions, so the submitted Model A run reused
cached stages instead of starting from raw files. Its normalization, embeddings and candidate search came from the v3
run (same code), and its retriever routes from a separate job. Model A then merged them and computed stage 1 → stage
3, and Model B reused Model A's pool. `run.sh` performs the same sequence in one go, and the whole chain was tested end
to end on the 10k-S1 sample. Seeds are fixed (13), but GPU training (XGBoost `hist`, transformer fine-tuning, fp16
embeddings) is not bit-exact. A fresh end-to-end run therefore lands very close to the submitted scores rather than
identical; the blend step alone (option A) is exact.

## Compliance

- **Data:** only the provided files. Supervised learning uses the train ground truth only. Unsupervised statistics
  (IDF, generic words, uniqueness counts, abbreviation mining) use the provided records without labels. No external
  data, APIs, geocoding or hand-written lists.
- **Models:** `multilingual-e5-small` (118M) and `multilingual-e5-large` (560M), both MIT, plus our fine-tuned
  retriever and cross-encoder derived from e5-small. All far below the 8B limit.
- **Software:** MIT, Apache-2.0, BSD or ISC licences only. Transliteration uses `anyascii` (ISC).
- **`country`** only partitions the search. It is never a feature and never hard-coded.
- **No credentials in the repository:** the SageMaker helpers read account settings from `account.ps1`
  (placeholders) and credentials from the AWS CLI.
