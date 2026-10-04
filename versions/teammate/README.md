# Teammate versions (public LB 0.904, later 0.964)

Built by a member of the team in parallel with v0–v2. It brought ideas that later went into v3.

## `char3gram_mlp_notebook.ipynb`: LB 0.904

A Kaggle notebook (outputs kept, so its validation logs are visible):

| stage | what it does |
|---|---|
| normalize | `anyascii` transliteration; small legal-form and street-type maps; core name = name minus legal words |
| sample | trains on a 30% sample of train S1 (all their matches + 30% of decoys); 60/40 train/validation split |
| search | per country, 3 views (core name, address, name + address): char-3-gram TF-IDF, count-sketched to 256 dims on the GPU, FAISS IVF top 30 per view + reverse top 3; ≈ 85 candidates per S1, pair recall 0.972 on the sample |
| features | 33: the three view similarities, gap to best, rank within S1, RapidFuzz scores on name / core / address, word and house-number Jaccard, script and source flags |
| model | PyTorch MLP (BatchNorm → 256 → 256, GELU, dropout 0.1), AdamW + OneCycle, 5 epochs |
| decision | one global threshold (0.70) maximizing macro F0.5 on the hold-out |

Its own hold-out scored 0.955, against 0.904 on the leaderboard. The 30% sample has about three times fewer look-alikes
per S1 than the real data, and test is denser still: an early sign of the density effect that later explained our own
train → test gap ([`docs/01`](../../docs/01-problem-and-data.md#the-train--test-shift)). The address-only and
typo-robust sketch views were adopted in v3.

## A later version: LB 0.964

The teammate's later submission scored 0.964. Its code is not part of this archive.

## `entity_resolution_single_file.py`: single-file draft

A self-contained CPU pipeline draft (polars, anyascii, RapidFuzz, sparse top-n TF-IDF, FAISS, LightGBM / XGBoost). It
includes exact-key "rescue" candidates, cross-source agreement features and two-stage stacking. We did not run it.
Note that its stacking trains the second stage on in-sample first-stage scores; use out-of-fold scores, as the main
pipeline does, to avoid leakage. Set `ER_DATA_DIR` to the dataset folder to run it.
