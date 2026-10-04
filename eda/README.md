# Exploratory data analysis

Four scripts, run once on the full data before any modelling. The findings are summarized in
[`docs/01-problem-and-data.md`](../docs/01-problem-and-data.md); the full reports are in [`reports/`](reports/).

| script | report | what it measures |
|---|---|---|
| `01_profile.py` | `reports/01_profile.md` | field quality per source and country (missing addresses, scripts, all-caps, web names, `NULL` tokens, landmarks), duplicates, matches per S1, cross-country pairs, records with several owners |
| `02_pairs.py` | `reports/02_pairs.md` | what true pairs look like vs name-similar hard negatives: per-feature AUC, unrelated-name pairs, transliteration gain, a LightGBM baseline |
| `03_blocking.py` | `reports/03_blocking.md` | recall vs candidate volume for exact shared-key blocking (name / address / number keys, bucket caps), and what the keys miss |
| `04_embeddings.py` | `reports/04_embeddings.md` | off-the-shelf `multilingual-e5-small` retrieval: recall@k for name-only vs name + address, rank of the true pair, encoding speed |

Run locally with the data in `../dataset/{train,test}/`:

```bash
pip install -r eda/requirements.txt
python eda/run_all.py            # all steps; or e.g. `python eda/run_all.py 01 03`
EDA_SMOKE=1 python eda/run_all.py  # quick run on a sample
```

`launch_sagemaker.ps1` runs the same steps as a SageMaker job; fill in the bucket and execution role first.
