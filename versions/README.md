# Versions

Each version kept what worked in the previous one and targeted its measured errors. Code for v2 → v5 lives in
[`../pipeline/`](../pipeline/) (one code base; later versions are switches on top of earlier stages). Each folder
here holds that version's notes, reports and metrics.

| folder | version | main idea | train OOF | public LB | code |
|---|---|---|---|---|---|
| [`v0-baseline/`](v0-baseline/) | v0 | name-only TF-IDF search, 9 features, XGBoost | — | 0.593 | `v0-baseline/code/` |
| [`teammate/`](teammate/) | teammate v1 | char-3-gram sketch views + FAISS, 33 features, MLP | 0.955 (own hold-out) | 0.904 | `teammate/` |
| | teammate, later | (code not archived) | | 0.964 | |
| [`v2/`](v2/) | v2 | learned lexicon, dense search both ways, 44 features, LightGBM | 0.9781 | 0.9700 | `pipeline/src/v2/` |
| [`v3/`](v3/) | v3 | 13 search routes, learned shortlist, ~100 features, XGBoost | 0.9881 | 0.978 (t 0.95) | `pipeline/src/run_pipeline.py` |
| [`v4/`](v4/) | v4 = Model A | fine-tuned retriever, cross-encoder, graph, stage-3 stack | 0.9913 | 0.984312 (t 0.95) | `pipeline/src/run_v4.py` |
| [`v5/`](v5/) | v5a = Model B | v4 trained at test orphan density, mined abbreviations | 0.9910 (harder) | 0.98376 | `pipeline/src/run_v5.py` |
| | **final** | **0.7 · v4 + 0.3 · v5a at v4's pair count** | +0.0001 | **0.984403** | `pipeline/src/run_final.py` |
| [`v6/`](v6/) | v6 | third embedder view (e5-large-instruct) | not run | | `v6/code/` |

Reports inside these folders were written during the challenge and are kept as they were. Some quote v4's t 0.95
score as 0.98414; the exact leaderboard value is **0.984312**.
