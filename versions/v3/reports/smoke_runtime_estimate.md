# v3 full-run estimate (from smoke run, ml.g6e.8xlarge)

Smoke: 9,921 train S1 (x222 to full). FAISS and XGBoost benchmarked at full size on this machine; other stages scaled linearly.

| stage | minutes |
|---|---|
| prepare | 103 |
| embed | 29 |
| block | 775 |
| stage1 features | 10 |
| stage1 model (5 folds, <=600 trees) | 16 |
| features | 15 |
| train (5 folds, <= 5000 trees) | 13 |
| train final model | 4 |
| any-match model + decide | 13 |
| predict | 0 |
| ablations (A3 9 variants x 3 folds + A4 LightGBM + A6) | 24 |
| **total** | **1001** (16.7 h) |

Cost ≈ **$94** at $5.66/h. Pool 87 pairs/S1; shortlist 3.5 pairs/S1. Training rows use the cap (5000 trees): an upper bound.

FAISS bench: {"256": {"d": 256, "n": 6186873, "build_s": 56.09254312515259, "search_s_per_query": 0.0005313988041877746}, "768": {"d": 768, "n": 6186873, "build_s": 109.93229341506958, "search_s_per_query": 0.0015853422927856445}}; XGBoost 5.0 ms/tree/1M rows.
