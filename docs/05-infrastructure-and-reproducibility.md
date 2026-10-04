# 5. Infrastructure and reproducibility

## Compute

Every full run was an **AWS SageMaker training job** (us-east-1) using the PyTorch 2.9 / Python 3.12 / CUDA 13 training
container. The launcher is [`pipeline/src/sagemaker/launch.ps1`](../pipeline/src/sagemaker/launch.ps1); fill in
`account.ps1` (region, bucket, execution role) first.

| instance | GPU | vCPU / RAM | used for | on-demand price |
|---|---|---|---|---|
| ml.g6.16xlarge | 1× L4 24 GB | 64 / 256 GB | v2 full run, smoke tests, v5c | ≈ $4.25/h |
| ml.g6e.8xlarge | 1× L40S 48 GB | 32 / 256 GB | v3 full run, embedder ablations | ≈ $5.66/h |
| ml.g6e.16xlarge | 1× L40S 48 GB | 64 / 512 GB | v4, v5a, retriever, ensemble checks | ≈ $9.47/h |
| ml.g5.16xlarge | 1× A10G 24 GB | 64 / 256 GB | fallback when L40S capacity was short | |
| ml.g4dn.2xlarge | 1× T4 16 GB | 8 / 32 GB | small analysis jobs | ≈ $1/h |

Typical cost per full run: v2 ≈ $22 (5.2 h), v4 ≈ $20 (2.1 h billed on top of cached stages).

**How jobs were run:**
- **Smoke test first.** Every configuration ran on a 10k-S1 sample (`BER_SMOKE_S1=10000`) before the full run, and
  the smoke run also estimated the full runtime ([`versions/v3/reports/smoke_runtime_estimate.md`](../versions/v3/reports/smoke_runtime_estimate.md)).
- **Checkpoint and resume.** Each stage writes its outputs to a work folder synced to an S3 checkpoint per run name,
  so a restarted job resumes at the first missing stage.
- **Seeding.** A new run can copy finished stages from an earlier run inside AWS (`BER_SEED_FROM`, with
  `BER_SEED_SET=pool` or `reprep`), e.g. Model B reused Model A's candidate pool.
- **Half the machine.** Threads and worker pools use 50% of the cores by default (`BER_CPU_FRAC`, 0.7 for the deadline
  runs), so memory or CPU spikes cannot take a job down.

## Runtimes

| stage | v2 (L4, 32 cores) | v4 (L40S, 32 cores) |
|---|---|---|
| prepare | 34 min | cached from v3 |
| embed | 44 min | cached (e5-small + e5-large ≈ 35–40 min on L40S) |
| search | 57 min | cached (≈ 40 min); retriever merge 1.4 min |
| stage 1 (shortlist) | — | 48 min |
| features | 61 min | 10 min |
| stage 2 | 102 min (LightGBM, CPU) | 20 min (XGBoost, GPU) |
| cross-encoder / graph / stage 3 | — | 25 / 1 / 11 min |

From raw files, the full chain takes roughly: v2 4–5 h, v4 6–7 h, v5a ≈ 3 h (it reuses v4's pool), blend 2 min.

## Reproducing the submission

Everything needed is in [`pipeline/`](../pipeline/) ([`pipeline/README.md`](../pipeline/README.md) has the
details):

```bash
cd pipeline
pip install -r requirements.txt
SMOKE=1 DATA=/path/to/dataset bash src/run.sh   # 10k-S1 sample of the whole chain
DATA=/path/to/dataset bash src/run.sh           # full run -> submission/
```

`src/run.sh` runs the auxiliary matcher (v2), Model A (`run_v4.py`), Model B (`run_v5.py`, reusing Model A's pool) and
the blend (`run_final.py`). The blend step reproduces the submitted `matching_results.tsv` byte for byte from the two
models' test scores. Seeds are fixed (13), but GPU training (XGBoost `hist`, transformer fine-tuning, fp16
embeddings) is not bit-exact, so a fresh end-to-end run lands very close to the submitted scores rather than
identical.

**How the leaderboard runs were actually executed.** The pipeline was developed in versions, so the submitted Model A
run reused cached stages instead of starting from raw files. Its normalization, embeddings and candidate search came
from the v3 run (same code), and its retriever routes from a separate job. Model A then merged them and computed
stage 1 → stage 3, and Model B reused Model A's pool. `src/run.sh` performs the same sequence in one go. The whole
chain was tested end to end on the 10k-S1 sample.

## Data

The competition data is **not included** (it belongs to the organizers). Place it as:

```
dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
dataset/test/{test_source1,test_source2,test_source3}.tsv
```

## Compliance

- **Data:** only the provided files. Supervised learning uses the train ground truth only. Unsupervised statistics
  (IDF, generic words, uniqueness counts, abbreviation mining) use the provided records without labels. No external
  data, APIs, geocoding or hand-written lists.
- **Models:** `multilingual-e5-small` (118M) and `multilingual-e5-large` (560M), both MIT, plus our fine-tuned
  retriever and cross-encoder derived from e5-small. All far below the 8B limit.
- **Software:** MIT, Apache-2.0, BSD or ISC licences only. Transliteration uses `anyascii` (ISC).
- **`country`** only partitions the search. It is never a feature and never hard-coded.
