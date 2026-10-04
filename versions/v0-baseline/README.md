# v0: baseline (public LB 0.593)

The first submitted pipeline (`code/src/run_pipeline.py`).

1. **Preprocess:** lower-case and strip names and addresses.
2. **Search** per country and per source: TF-IDF character 2–3-grams on the **name only** (50k features) →
   TruncatedSVD to 128 dims → FAISS GPU inner-product search → top 10 per source, 20 candidates per S1.
3. **Features (9):** FAISS score; name token-set / token-sort / partial / ratio; address partial / token-set; country
   match; has-address.
4. **Model:** XGBoost (GPU), 400 rounds, depth 7, lr 0.05, random 90/10 pair split.
5. **Decision:** one global threshold maximizing *pair-level* F0.5.

**Why it scored 0.593**, and what every later version fixed:
- The name-only search misses copies whose names differ.
- Records can be given to several S1 (108,297 test records had more than one owner): there is no one-owner rule.
- It was tuned on pair-level F0.5, not the per-S1 metric, so the empty answer was never chosen well. It predicted empty
  for 38.6% of India S1, against a true no-match rate of 5.6%.

Run: `pip install -r code/requirements.txt`, put the data in `./mlchallenge_data/{train,test}/`, then
`python code/src/run_pipeline.py` (needs CUDA and FAISS-GPU).
