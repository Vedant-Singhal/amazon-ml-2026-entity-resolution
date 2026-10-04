# Analysis scripts

Post-hoc analyses behind [`docs/03-results-and-analysis.md`](../docs/03-results-and-analysis.md). They read
submission files and run outputs that are not stored in this repository (too large), so they document *how* each
result was obtained. Set `BER_ROOT` to the folder holding `dataset/` and the run outputs.

| script | what it does | result |
|---|---|---|
| `lb_difference_precision.py` | For two nested submissions, solves for the precision `q` of the pairs that differ from their leaderboard difference | band precision table in docs/03 |
| `band_probes.py` | Builds probe submissions that drop one suspect group of 0.95–0.99 band pairs (house-number conflicts, no address) | 0.984 / 0.983: no rule-based filter helps |
| `country_probes.py` | Builds per-country probe submissions (France strict, France empty, US empty) to split the score by country | France holds most uncertainty |
| `train_test_profile.py` | Profiles S1 and S2/S3 records per split, country and source (empty addresses, numbers, web names, scripts) | the orphan-density shift |
| `orphan_share.py` | Orphan share per split: from ground truth in train, estimated from predictions in test | 26% train vs ≈ 42% test |
| `run_ens.py` | Weighted blends of v4 / v5a / v3 on train out-of-fold predictions, judged at v4's pair count; writes the test files | 0.7 · v4 + 0.3 · v5a chosen |
| `run_vote.py` | Majority, unanimous and equal-average votes, same protocol | all at or below the weighted blend |
| `run_twins.py` | Cluster completion: adds unmatched exact normalized copies of a predicted cluster member | −0.0019 on train: rejected |

`run_ens.py`, `run_vote.py` and `run_twins.py` import the pipeline package from `../pipeline/src` and were run as
SageMaker jobs (`ENS_SRC` / `TWIN_SRC` point at the S3 folder with the runs' out-of-fold and test scores).
