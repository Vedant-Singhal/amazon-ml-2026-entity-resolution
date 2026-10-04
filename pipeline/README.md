# Business entity resolution: how to reproduce the submission

For every Source-1 (S1) business in test, the pipeline finds all matching Source-2 / Source-3 (S2/S3) records. It
writes two files:
- `matching_results.tsv`: the final matches;
- `candidate_pairs.tsv`: the exact set of pairs the models scored.

Visual guides are in [`../visuals/`](../visuals/):
- `pipeline-flow.html`: stage-by-stage flow of the main matcher;
- `inside-v4-v5.html`: plain-language and technical views of v4 and the v5a changes;
- `pipeline-math.html`: the formulas behind each stage.

**The submitted files are a blend of two runs of the same pipeline: 0.7 × v4 + 0.3 × v5a.** Public leaderboard
**0.984403**; v4 alone scored 0.984312 and v5a alone 0.98376.

**Names.** The methodology document (`Documentation_template.md`) uses descriptive names; the code keeps version
names:

| methodology document | code |
|---|---|
| auxiliary matcher (step 0) | v2: `src/v2/` |
| Model A, the main pipeline (steps 1–11) | v4: `src/run_v4.py` |
| Model B, Model A plus three training changes | v5a: `src/run_v5.py` |
| final blend | `src/run_final.py` |

Only the provided data is used: no external data, APIs, gazetteers or hand-written word lists. Models and software use
MIT, Apache-2.0, BSD or ISC licenses; the largest pretrained model has 0.56B parameters. `country` is never a model
feature and is not hard-coded: candidate search stays within each record's own country label, so France is handled like
any other value.

## 1. Requirements

| | |
|---|---|
| **Hardware** | Linux, 1 NVIDIA GPU with ≥ 24 GB (we used an L40S 48 GB), 32–64 vCPU, 256–512 GB RAM (peak ≈ 140 GB), ≈ 400 GB free disk for the cached stages of both runs |
| **Runtime** | v2 ≈ 4–5 h, v4 ≈ 6–7 h, v5a ≈ 3 h (reuses v4's candidate pool), final blend ≈ 2 min |
| **Software** | Python 3.12 + CUDA; `pip install -r requirements.txt` (pinned versions from the submitted runs; we ran in the AWS SageMaker container `pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04`) |
| **Data layout** | `dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv`, `dataset/test/{test_source1,test_source2,test_source3}.tsv` |
| **Models** | Downloaded from Hugging Face on first use: `intfloat/multilingual-e5-small`, `intfloat/multilingual-e5-large` (MIT) |

The code uses 50% of the CPU cores by default. `BER_CPU_FRAC=0.7` raises that to 70%.

## 2. Run

```bash
pip install -r requirements.txt
SMOKE=1 DATA=/path/to/dataset bash src/run.sh  # optional: 10k-S1 sample of the whole chain (separate *_smoke folders)
DATA=/path/to/dataset bash src/run.sh          # full run -> submission/
```

`src/run.sh` runs four steps. Each step caches its stages, so a restart resumes where it stopped.

| step | command | writes | used for |
|---|---|---|---|
| 1. v2 | `src/v2/run_pipeline.py` | `work_v2/` | its out-of-fold train score and test score are one stage-3 feature (`p_v2`) of v4 and v5a |
| 2. v4 | `src/run_v4.py` | `work_v4/`, `output_v4/` | test probabilities `work_v4/test/scored3.parquet`; the candidate set |
| 3. v5a | `src/run_v5.py` | `work_v5a/`, `output_v5a/` | test probabilities `work_v5a/test/scored3.parquet` |
| 4. final | `src/run_final.py` | `submission/` | the submitted files |

**Step 3 reuses v4's candidate pool.** It copies four files from `work_v4/`: `pool.parquet`, `route_ft.parquet`,
`block_stats.json` and `emb_oth.npy`. It then rebuilds everything else with three changes:
- 19% of the train S1s are dropped, and their S2/S3 records are kept as unmatched records. This gives training data
  the test set's density of unmatched records.
- Abbreviations are mined per country from the records themselves.
- Stage 1 gets a 3,000-tree cap.

**Step 4, the final blend:**
1. p = 0.7 · p_v4 + 0.3 · p_v5a. A pair a model did not score counts as 0 for that model.
2. Each S2/S3 record goes to its highest-p S1.
3. The cut keeps exactly as many pairs as v4 alone at p ≥ 0.95, which was the leaderboard-validated setting. In the
   submitted run that cut is t = 0.9157, and it keeps 5,836,213 pairs.
4. `candidate_pairs.tsv` is v4's stage-1 shortlist: 11,171,750 pairs, 6.45 per S1.

`FINAL_CHECK=1` makes step 4 also print the train check that chose the weights, on v5a's S1s:
- v4 alone 0.99021;
- 0.7/0.3 blend 0.99031.

We checked step 4 on the two submitted score files: it reproduces the uploaded `matching_results.tsv` byte for byte.

Validate with the organizers' checker (`validate_submission.py`, distributed with the challenge data, not included here):
`python validate_submission.py -m submission/matching_results.tsv -c submission/candidate_pairs.tsv -t /path/to/dataset/test`

**Determinism.** Seeds are fixed (13 everywhere), but GPU training (XGBoost `hist`, transformer fine-tuning,
fp16 embeddings) is not bit-exact. The submitted TSVs (97–166 MB, not stored in this repository) are the leaderboard-tested 0.7/0.3 blend. Step 4 reproduces the
matching file exactly when given the same v4/v5a test-score files.

**How the submitted runs were executed.** We developed the pipeline in versions, so the submitted v4 run reused
cached stages instead of starting from the raw files:
- prepare, embed and block (the candidate pool) came from an earlier run of the same stages, whose code for these
  stages is unchanged;
- the retriever routes came from a separate retriever job;
- the v4 run merged them and computed stage 1, features, stage 2, cross-encoder, graph and stage 3;
- v5a then reused v4's pool and rebuilt everything else.

`src/run.sh` performs the same stage sequence from the provided raw files. A 10k-S1 sample of the chain was tested end to
end with `SMOKE=1`; the full-size leaderboard runs reused the cached stages described above.

**SageMaker (optional).** Use `src/sagemaker/launch.ps1` and `src/sagemaker/fetch.ps1` after filling in `src/sagemaker/account.ps1` (region,
bucket, execution role). Run v4 with `-Program run_v4.py` and v5a with `-Program run_v5.py`, each with `-V2Inputs`
(v2's scores uploaded to `<prefix>/ber-runs/<run>/v2in/{oof,test}/`). v5a can seed v4's pool with
`BER_SEED_FROM=s3://<bucket>/<prefix>/ber-runs/v4/work` and `BER_SEED_SET=reprep`.

## 3. Pipeline (shared by v4 and v5a)

| stage | code | what it does | result on train (v4) |
|---|---|---|---|
| prepare | `ber/prepare.py`, `text.py`, `lexicon.py` (v5a also `mining.py`) | Normalize names and addresses (see list 1 below) | |
| embed | `ber/embed.py` | e5-small + e5-large on `name \| address`, fp16, max 64 tokens | |
| block | `ber/block.py`, `sketch.py` | Candidates within each country, from several routes (see list 2 below) | pool recall 0.9908 |
| retriever | `ber/retriever.py` | e5-small fine-tuned on train pairs (contrastive, 2-fold cross-fitted); adds its top 10 per S1 and top 3 S1 per record | **pool recall 0.9964**, 110 pairs per S1 |
| s1drop (v5a) | `ber/s1drop.py` | drops 19% of train S1 and keeps their records as orphans | |
| stage 1 | `ber/stage1.py` | 51 cheap features, XGBoost 5-fold OOF; keeps rank ≤ 10 and p1 ≥ 0.003 per S1, holding 99.9% of pool recall | **shortlist recall 0.9955 at 5.85 pairs per S1** |
| features | `ber/features.py` | 101 pair features (name/address similarities, shared-word rarity, typed numbers, phonetic keys, typos, ranks and gaps within each S1 and each record) | |
| stage 2 | `ber/train.py`, `learner.py`, `decide.py` | XGBoost, 5-fold GroupKFold by S1, isotonic calibration, "has any match" model | OOF 0.9888 |
| cross-encoder | `ber/crossenc.py` | e5-small + 1-logit head on the uncertain band (0.02 ≤ p2 ≤ 0.98), 2-fold cross-fitted | |
| graph | `ber/graph.py` | Consistency features: the record's best other S1, the S1's other records, similarity to its strong siblings | |
| stage 3 | `ber/stage3.py` | XGBoost stack on stage-2 features + p2 + cross-encoder + graph + v2 score, same folds (leak-free) | **OOF 0.9913** (US 0.9908, India 0.9921) |

1. **Prepare** covers:
   - mojibake repair, zero-width characters removed, dotted abbreviations merged;
   - a lexicon of token equivalences learned from the train ground-truth pairs;
   - per-country generic words (≥ 1% of a country's names) removed to form the core name;
   - compact, skeleton, metaphone and typed-number keys.
2. **Block** routes:
   - exact GPU kNN in both directions for each embedder: 25 records per S1 and 5 S1 per record;
   - character-3-gram count-sketch routes on name and on address;
   - an IDF shared-key route (top 20);
   - 6 exact-key routes, each with bucket caps (≤ 10 S1 and ≤ 50 records per key).

Folds are `hash(S1 id, seed 13) % 5` in every stage, so every stacked score is out-of-fold.

## 4. Code layout

```
src/run.sh               the whole chain (v2 -> v4 -> v5a -> final)
src/run_v4.py            v4 entry point (all stages; cached in $BER_WORK)
src/run_v5.py            v5 entry point = run_v4.py with v5's settings (s1drop 0.19, mined abbreviations, 3,000 trees)
src/run_final.py         final blend 0.7 / 0.3 -> submission files (numpy + pyarrow only)
src/run_pipeline.py      prepare … stage 2 + predict, called by run_v4.py
src/run_v4_retriever.py  retriever alone (optional; run_v4.py calls it)
src/run_probes.py        extra fixed-threshold files from a finished run (analysis)
src/ber/                 the pipeline package; diagnose*.py / ablate*.py / estimate.py are analysis tools
src/v2/                  the v2 pipeline (its own `ber` package; entry src/v2/run_pipeline.py)
src/sagemaker/           optional AWS SageMaker launch / fetch helpers (fill in account.ps1)
```

## 5. Licences

| component | licence |
|---|---|
| `intfloat/multilingual-e5-small` (118M), `multilingual-e5-large` (560M), and the fine-tuned retriever and cross-encoder derived from e5-small | MIT |
| XGBoost | Apache-2.0 |
| LightGBM, polars, RapidFuzz, sentence-transformers, jellyfish | MIT |
| transformers | Apache-2.0 |
| PyTorch, numpy, scipy, scikit-learn | BSD |
| anyascii | ISC |

All pipeline stages, including the auxiliary v2 matcher, use `anyascii` (ISC) to romanize remaining non-Latin tokens.
