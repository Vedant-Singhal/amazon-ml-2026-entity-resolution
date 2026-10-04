# Results

[`scores.csv`](scores.csv) lists every score in one table, one row per submission or variant.

| column | meaning |
|---|---|
| `version` | v0, teammate_v1, teammate_v2, v2, v3, v4 (Model A), v5a (Model B), final |
| `variant` | what was submitted (main run, threshold probe, band probe, blend) |
| `decision_rule` | how pairs were selected from the scores |
| `train_oof_macro_f05` | out-of-fold macro F0.5 on train (5 folds grouped by S1); empty when not measured |
| `public_lb` | public leaderboard macro F0.5 |
| `notes` | per-country scores and caveats |

The final row's train score is measured on the 1.78M train S1 that Model B kept (test-like orphan density), where
Model A alone scores 0.99021. The other train scores use all 2.2M train S1.
