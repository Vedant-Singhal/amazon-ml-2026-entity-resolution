# Error diagnosis — v4 (train OOF, rule expf)

## 1. Where the true pairs go

| stage | pairs | share |
|---|---|---|
| 1 not found by search | 27,488 | 0.0036 |
| 2 dropped by shortlist | 6,773 | 0.0009 |
| 3 rejected by model | 151,656 | 0.0199 |
| found | 7,452,448 | 0.9757 |

## 2. By address

| address | true_pairs | missed | false_pairs | miss_rate |
|---|---|---|---|---|
| has address | 7,295,758 | 32,111 | 6,264 | 0.0044 |
| no address | 308,346 | 119,545 | 3,216 | 0.3877 |

## 2. By shared

| shared | true_pairs | missed | false_pairs | miss_rate |
|---|---|---|---|---|
| 2-3 S1 | 4,256,791 | 47,671 | 5,246 | 0.0112 |
| 4+ S1 | 3,347,313 | 103,985 | 4,234 | 0.0311 |

## 2. By script

| script | true_pairs | missed | false_pairs | miss_rate |
|---|---|---|---|---|
| Latin | 7,053,334 | 150,140 | 9,061 | 0.0213 |
| native script | 550,770 | 1,516 | 419 | 0.0028 |

## 2. By num

| num | true_pairs | missed | false_pairs | miss_rate |
|---|---|---|---|---|
| different | 502,134 | 6,796 | 1,229 | 0.0135 |
| equal | 5,615,883 | 12,779 | 2,501 | 0.0023 |
| missing | 961,230 | 123,551 | 3,686 | 0.1285 |
| near (<=10) | 27,779 | 1,261 | 232 | 0.0454 |
| one-digit edit | 158,399 | 4,508 | 863 | 0.0285 |
| truncated | 338,679 | 2,761 | 969 | 0.0082 |

## 2. By source

| source | true_pairs | missed | false_pairs | miss_rate |
|---|---|---|---|---|
| S2 | 3,677,249 | 72,658 | 4,565 | 0.0198 |
| S3 | 3,926,855 | 78,998 | 4,915 | 0.0201 |

## 2. By country

| country | true_pairs | missed | false_pairs | miss_rate |
|---|---|---|---|---|
| India | 3,046,791 | 54,753 | 3,987 | 0.0180 |
| US | 4,557,313 | 96,903 | 5,493 | 0.0213 |

## 3. Test profile per country (vs train OOF)

| country | with_match | matches_per_s1 | candidates_per_s1 | p_max_median | train_with_match | train_matches_per_s1 |
|---|---|---|---|---|---|---|
| France | 0.9463 | 3.3989 | 6.4220 | 1.0000 | None | None |
| India | 0.9430 | 3.3941 | 6.4484 | 1.0000 | 0.9432 | 3.3923 |
| US | 0.9438 | 3.4339 | 6.4688 | 1.0000 | 0.9428 | 3.3740 |
