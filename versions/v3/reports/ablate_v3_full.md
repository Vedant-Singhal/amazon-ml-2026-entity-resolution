# v3 ablations

## A1 · Retrieval routes (train)
Pool: **92.3 candidates/S1**, union recall **0.9908**, S1 with every match found 0.9680.

| route | recall | only_this_route | pairs_per_s1 |
|---|---|---|---|
| dense_fwd | 0.9610 | 0.0009 | 25.0000 |
| dense2_fwd | 0.9627 | 0.0003 | 25.0000 |
| dense_rev | 0.9755 | 0.0008 | 23.3825 |
| dense2_rev | 0.9798 | 0.0016 | 23.3825 |
| name_c3 | 0.6264 | 0.0009 | 19.5741 |
| addr_c3 | 0.8090 | 0.0008 | 6.6100 |
| sparse | 0.4852 | 0.0002 | 15.6135 |
| k_core | 0.4411 | 0.0000 | 4.1192 |
| k_compact | 0.4645 | 0.0000 | 4.2767 |
| k_sorted | 0.4576 | 0.0000 | 4.3559 |
| k_skel | 0.4495 | 0.0001 | 5.0855 |
| k_akey | 0.3762 | 0.0000 | 1.7082 |
| k_numset | 0.1542 | 0.0000 | 1.6314 |

## A2 · Candidate size (stage-1 shortlist)
Chosen rule **K=10, tau=0.003**: 5.37 candidates/S1, recall 0.9902 (pool 92.3/S1, recall 0.9908; target 0.9898).

| K | tau | recall | per_s1 |
|---|---|---|---|
| 3 | 0.0000 | 0.7289 | 3.0000 |
| 5 | 0.0000 | 0.9392 | 5.0000 |
| 8 | 0.0000 | 0.9892 | 8.0000 |
| 10 | 0.0000 | 0.9905 | 10.0000 |
| 12 | 0.0000 | 0.9907 | 12.0000 |
| 15 | 0.0000 | 0.9908 | 15.0000 |
| 20 | 0.0000 | 0.9908 | 20.0000 |
| 30 | 0.0000 | 0.9908 | 30.0000 |
| 3 | 0.0100 | 0.7288 | 2.8543 |
| 5 | 0.0100 | 0.9387 | 4.1543 |
| 8 | 0.0100 | 0.9881 | 4.7992 |
| 10 | 0.0100 | 0.9893 | 4.8678 |
| 12 | 0.0100 | 0.9894 | 4.8835 |
| 15 | 0.0100 | 0.9894 | 4.8886 |
| 20 | 0.0100 | 0.9894 | 4.8897 |
| 30 | 0.0100 | 0.9894 | 4.8900 |
| 3 | 0.0500 | 0.7279 | 2.7227 |
| 5 | 0.0500 | 0.9359 | 3.7439 |
| 8 | 0.0500 | 0.9835 | 4.0930 |
| 10 | 0.0500 | 0.9844 | 4.1077 |
| 12 | 0.0500 | 0.9844 | 4.1087 |
| 15 | 0.0500 | 0.9844 | 4.1088 |
| 20 | 0.0500 | 0.9844 | 4.1089 |
| 30 | 0.0500 | 0.9844 | 4.1089 |

## A3 · Feature groups (100% of S1, 3-fold OOF, lr 0.1)
delta < 0 = removing the group hurts (the group helps).

| variant | n_features | macro_f05 | rule | trees | minutes | delta |
|---|---|---|---|---|---|---|
| all features | 99 | 0.9877 | expf 0.0 | 1465 | 4.2000 | 0.0000 |
| - uniq | 92 | 0.9867 | expf 0.0 | 1458 | 3.8000 | -0.0010 |
| - numbers | 91 | 0.9866 | expf 0.0 | 1735 | 4.2000 | -0.0011 |
| - trade_addr | 91 | 0.9876 | expf 0.0 | 1432 | 3.8000 | -0.0002 |
| - name_c3 | 92 | 0.9878 | expf 0.0 | 1546 | 3.9000 | 0.0000 |
| - phonetic | 93 | 0.9876 | expf 0.0 | 1469 | 3.9000 | -0.0001 |
| - typo | 96 | 0.9876 | expf 0.0 | 1253 | 3.5000 | -0.0001 |
| - dense2 | 94 | 0.9877 | expf 0.0 | 1358 | 3.7000 | 0.0000 |
| - listwise | 89 | 0.9870 | thresh 0.7 | 1607 | 3.9000 | -0.0007 |
| v2 features only | 45 | 0.9803 | thresh 0.65 | 2247 | 3.4000 | -0.0074 |

## A4 · Learner (same sample, features, folds)

| learner | macro_f05 | minutes |
|---|---|---|
| XGBoost (GPU) | 0.9877 | 4.2000 |
| LightGBM (CPU, v2 params) | 0.9876 | 17.7000 |
| mean of both | 0.9878 | 0.0000 |

## A6 · Leave-one-country-out (proxy for unseen France)

| train on | score on | transfer (src model+rule) | in-country OOF | gap |
|---|---|---|---|---|
| US | India | 0.9548 | 0.9876 | -0.0328 |
| India | US | 0.9736 | 0.9878 | -0.0142 |

## A5 · Decision rule (full train OOF)

| kind | macro_f05 | params |
|---|---|---|
| expf_any | 0.9881 | {"m": 1.25, "miss": 0.0, "floor": 0.02, "one_owner": true} |
| expf | 0.9880 | {"miss": 0.0, "floor": 0.2, "one_owner": true} |
| thresh+any_min | 0.9879 | {"t": 0.7, "floor": 0.0, "one_owner": true, "any_min": 0.35} |
| thresh | 0.9879 | {"t": 0.7, "floor": 0.0, "one_owner": true} |

Full OOF macro F0.5 **0.9881** (folds 0.9881 ± 0.0000); by country {"US": 0.9881, "India": 0.9881}.


### Feature gain (top 30)

| feature | gain |
|---|---|
| p1 | 0.6432 |
| p1_gap_a | 0.0824 |
| p1_gap2_a | 0.0720 |
| p1_sum_a | 0.0245 |
| n_idf_jacc | 0.0129 |
| pun_max_b | 0.0115 |
| n_tsort | 0.0097 |
| len_name_b | 0.0092 |
| n_sl_b | 0.0089 |
| n_partial | 0.0079 |
| num_extra_b | 0.0067 |
| num_jacc | 0.0061 |
| num_a_in_b | 0.0051 |
| stop_jacc | 0.0051 |
| num_rel | 0.0048 |
| n_tset | 0.0043 |
| n_jw | 0.0043 |
| first_num_eq | 0.0041 |
| len_name_a | 0.0039 |
| pun_max_a | 0.0037 |
| p1_gap_b | 0.0035 |
| n_ratio | 0.0034 |
| lu_s1_name_a | 0.0033 |
| n_sl_a | 0.0032 |
| lu_oth_numst_b | 0.0028 |
| num_main_cov | 0.0023 |
| core_idf_sum_a | 0.0022 |
| a_idf_jacc | 0.0021 |
| core_idf_min_a | 0.0019 |
| a_tsort | 0.0018 |