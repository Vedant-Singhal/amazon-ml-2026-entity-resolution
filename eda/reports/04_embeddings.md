# 04 · Dense retrieval with `intfloat/multilingual-e5-small`


## Pair recall@k (share of GT pairs retrieved in each S1's top-k, within country)

| variant | country | src | pairs | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | R@200 |
|---|---|---|---|---|---|---|---|---|---|---|
| name | India | S2 | 34107 | 0.1249 | 0.3376 | 0.4169 | 0.4772 | 0.5427 | 0.5814 | 0.6156 |
| name | India | S3 | 36328 | 0.1217 | 0.3574 | 0.4390 | 0.5089 | 0.5838 | 0.6284 | 0.6663 |
| name | India | all | 70435 | 0.1232 | 0.3478 | 0.4283 | 0.4936 | 0.5639 | 0.6056 | 0.6418 |
| name | US | S2 | 50036 | 0.1505 | 0.4670 | 0.5806 | 0.6601 | 0.7354 | 0.7823 | 0.8269 |
| name | US | S3 | 53285 | 0.1504 | 0.4759 | 0.5826 | 0.6588 | 0.7361 | 0.7855 | 0.8315 |
| name | US | all | 103321 | 0.1504 | 0.4716 | 0.5816 | 0.6595 | 0.7358 | 0.7839 | 0.8293 |
| name_addr | India | S2 | 34107 | 0.1727 | 0.8194 | 0.9116 | 0.9306 | 0.9470 | 0.9555 | 0.9636 |
| name_addr | India | S3 | 36328 | 0.3064 | 0.8389 | 0.9196 | 0.9381 | 0.9525 | 0.9598 | 0.9667 |
| name_addr | India | all | 70435 | 0.2416 | 0.8295 | 0.9157 | 0.9345 | 0.9499 | 0.9577 | 0.9652 |
| name_addr | US | S2 | 50036 | 0.1574 | 0.8474 | 0.9621 | 0.9722 | 0.9794 | 0.9843 | 0.9873 |
| name_addr | US | S3 | 53285 | 0.3185 | 0.8963 | 0.9707 | 0.9785 | 0.9841 | 0.9873 | 0.9903 |
| name_addr | US | all | 103321 | 0.2405 | 0.8726 | 0.9665 | 0.9755 | 0.9819 | 0.9858 | 0.9888 |


## Entity recall@k (S1 entities whose ALL matches are in top-k; non-singletons)

| variant | country | all@1 | all@5 | all@10 | all@20 | all@50 | all@100 | all@200 |
|---|---|---|---|---|---|---|---|---|
| name | India | 0.0125 | 0.1086 | 0.1716 | 0.2324 | 0.3049 | 0.3519 | 0.3938 |
| name | US | 0.0172 | 0.1919 | 0.3062 | 0.3945 | 0.4901 | 0.5508 | 0.6132 |
| name_addr | India | 0.0342 | 0.5754 | 0.7595 | 0.8057 | 0.8457 | 0.8668 | 0.8886 |
| name_addr | US | 0.0349 | 0.6689 | 0.8862 | 0.9155 | 0.9365 | 0.9498 | 0.9602 |


## Cosine calibration

`true_*` = cosine of GT pairs; `wrong_top1` = best non-matching neighbour of matched S1s; `singleton_top1` = best neighbour of singletons; `true_gt_best_wrong` = share of GT pairs scoring above that S1's best wrong neighbour (ranking quality).

| variant | country | true_p10 | true_p50 | wrong_top1_p50 | wrong_top1_p90 | singleton_top1_p50 | singleton_top1_p90 | true_gt_best_wrong |
|---|---|---|---|---|---|---|---|---|
| name | India | 0.8865 | 0.9518 | 0.9849 | 1.0010 | 0.9858 | 1.0010 | 0.2312 |
| name | US | 0.9125 | 0.9615 | 0.9790 | 1.0010 | 0.9805 | 1.0010 | 0.3170 |
| name_addr | India | 0.9066 | 0.9554 | 0.9414 | 0.9673 | 0.9487 | 0.9708 | 0.6460 |
| name_addr | US | 0.9208 | 0.9624 | 0.9521 | 0.9746 | 0.9595 | 0.9766 | 0.6359 |


## Train → test drift: neighbour similarity by country (name variant)

Compare France against US/India: a shifted distribution means thresholds tuned on train won't transfer as-is.

| country | corpus | top1_p10 | top1_p25 | top1_p50 | top1_p75 | top1_p90 | top1_minus_top5_p50 | share_top1_ge_0.9 |
|---|---|---|---|---|---|---|---|---|
| France | 1434993 | 0.9771 | 0.9902 | 1.0000 | 1.0010 | 1.0010 | 0.0195 | 1.0000 |
| India | 4717565 | 0.9751 | 0.9863 | 0.9995 | 1.0000 | 1.0010 | 0.0171 | 1.0000 |
| US | 3817031 | 0.9712 | 0.9824 | 0.9961 | 1.0000 | 1.0010 | 0.0273 | 1.0000 |
