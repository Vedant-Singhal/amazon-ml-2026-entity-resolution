# v3: scalable search, learned shortlist, targeted features (OOF 0.9881, LB 0.978)

Code: the v3 stages are `prepare → embed → block → stage1 → features → train → predict` in
[`pipeline/src/run_pipeline.py`](../../pipeline/src/run_pipeline.py), with modules in `pipeline/src/ber/`.

**What changed from v2**
- **Two dense views** (e5-small + e5-large), chosen by a 14-model embedder ablation, plus name and address
  char-3-gram count sketches and 6 capped exact-key routes. Pool recall 0.9908 at 92 per S1 (v2: 0.981 at 56).
- **Exact GPU search** instead of approximate IVF, which lost 2–18 recall points at practical settings.
- **Learned shortlist:** a stage-1 XGBoost keeps the smallest candidate set holding 99.9% of the pool's recall:
  5.37 per S1 at recall 0.9902, 11× fewer than v2.
- **About 50 new features** aimed at v2's error types (uniqueness, house-number relations, twins, trade names,
  phonetic, typos, listwise rank/gap), **XGBoost on GPU** (lr 0.05, ≤ 5,000 trees), and a "has any match" model.

**Results**: OOF 0.9881 (US 0.9881, India 0.9881). On the leaderboard the train-tuned rule gave 0.97493, fold-consistent
scoring 0.974654, t 0.30 0.958 and **t 0.95 0.978**. Test is precision-limited.

**Reports** (`reports/`)
- `ablate_A0_embedders.md`, `ablate_A0r2.md` … `ablate_A0r4.md`: embedder ablation rounds (recall@10 by country,
  no-address, native script, trade names; speed).
- `ablate_v3_full.md`: A1–A6 on the full train set (feature groups, learners, decision rules, leave-one-country-out).
- `gapcheck.md`: train vs test score distributions (test S1 have more plausible look-alikes).
- `smoke_runtime_estimate.md`: full-run time estimated from the 10k-S1 smoke run.
