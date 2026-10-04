# v4 = Model A (OOF 0.9913, LB 0.984312)

Code: [`pipeline/src/run_v4.py`](../../pipeline/src/run_v4.py). Full description:
[`docs/02-solution.md`](../../docs/02-solution.md).

**What changed from v3**
- **Fine-tuned retriever** (`ber/retriever.py`): e5-small trained on our own pairs (InfoNCE, romanized copies of
  native-script records). Pool recall 0.9908 → 0.9964; shortlist recall 0.9902 → 0.9955 at 5.85 per S1.
- **Cross-encoder** on the uncertain band (`ber/crossenc.py`), **graph-consistency features** (`ber/graph.py`), and
  a **stage-3 stack** with v2's score (`ber/stage3.py`). Count features that depend on candidate density are dropped.
- Stage-3 OOF **0.9913** (US 0.9908, India 0.9921), +0.0032 over v3.

**Leaderboard**: train-tuned rule 0.983 · **t 0.95 0.984312** · t 0.99 0.980 · band probes 0.984 (number conflicts
dropped) and 0.983 (no-address pairs dropped).

**Reports** (`reports/`; written during the challenge, which quote t 0.95 as 0.98414, an earlier reading of
0.984312)
- `train-vs-test-shift.md`: the orphan-density shift (26% → ≈ 42% of S2/S3 records), the main finding.
- `leaderboard-diagnosis.md`: precision of submission differences, measured from leaderboard scores.
- `test-diagnosis.md`: label-free test analysis (loss split, France profile, what the uncertain pairs are).
- `oof-errors-stage2.md`, `oof-errors-stage3.md`: where train true pairs are lost, by stage and error type.

**Metrics** (`metrics/`): `block_stats_train.json` (recall per search route), `stage1_shortlist.json` (shortlist
curve and rule), `stage2_report.json` / `stage2_decision.json`, `stage3_report.json` (OOF, feature gain) /
`stage3_decision.json`, `timing.json`.
