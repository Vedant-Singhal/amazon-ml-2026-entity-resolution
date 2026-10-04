# Figures

Drawn by [`../scripts/make_figures.py`](../scripts/make_figures.py) from the numbers in the run reports; rerun it
to regenerate them.

| figure | shows | used in |
|---|---|---|
| `score_progression.png` | train OOF and public leaderboard per version | README, docs/03 |
| `funnel.png` | candidates per S1 and recall at each stage | README, docs/02 |
| `threshold_probes.png` | leaderboard vs decision threshold for v3 and v4 | docs/03 |
| `train_test_shift.png` | share of orphan records in train, in train after S1 dropout, and in test | docs/01 |
| `stage3_feature_gain.png` | what the final stack relies on | docs/02 |
| `loss_budget.png` | where v4 loses true pairs on train | docs/03 |
| `blend_variants.png` | blends and votes at v4's pair count | docs/03 |
| `embedder_ablation.png` | recall@10 of 12 off-the-shelf embedders | docs/03 |
| `feature_group_ablation.png` | effect of removing each stage-2 feature group | docs/03 |
