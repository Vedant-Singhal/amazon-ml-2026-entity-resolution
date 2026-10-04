# Error diagnosis — v3 (train OOF, rule expf_any)

## 1. Where the true pairs go

| stage | pairs | share |
|---|---|---|
| 1 not found by search | 27,488 | 0.0036 |
| 2 dropped by shortlist | 6,773 | 0.0009 |
| 3 rejected by model | 184,512 | 0.0242 |
| found | 7,419,592 | 0.9714 |

## 2. By address

| address | true_pairs | missed | false_pairs | miss_rate |
|---|---|---|---|---|
| has address | 7,295,758 | 59,898 | 13,300 | 0.0082 |
| no address | 308,346 | 124,614 | 4,526 | 0.4041 |

## 2. By shared

| shared | true_pairs | missed | false_pairs | miss_rate |
|---|---|---|---|---|
| 2-3 S1 | 4,256,791 | 65,280 | 9,815 | 0.0153 |
| 4+ S1 | 3,347,313 | 119,232 | 8,011 | 0.0356 |

## 2. By script

| script | true_pairs | missed | false_pairs | miss_rate |
|---|---|---|---|---|
| Latin | 7,053,334 | 181,995 | 17,119 | 0.0258 |
| native script | 550,770 | 2,517 | 707 | 0.0046 |

## 2. By num

| num | true_pairs | missed | false_pairs | miss_rate |
|---|---|---|---|---|
| different | 502,134 | 15,263 | 2,761 | 0.0304 |
| equal | 5,615,883 | 17,952 | 5,185 | 0.0032 |
| missing | 961,230 | 130,630 | 5,614 | 0.1359 |
| near (<=10) | 27,779 | 3,308 | 514 | 0.1191 |
| one-digit edit | 158,399 | 11,537 | 2,005 | 0.0728 |
| truncated | 338,679 | 5,822 | 1,747 | 0.0172 |

## 2. By source

| source | true_pairs | missed | false_pairs | miss_rate |
|---|---|---|---|---|
| S2 | 3,677,249 | 88,352 | 8,435 | 0.0240 |
| S3 | 3,926,855 | 96,160 | 9,391 | 0.0245 |

## 2. By country

| country | true_pairs | missed | false_pairs | miss_rate |
|---|---|---|---|---|
| India | 3,046,791 | 69,200 | 7,232 | 0.0227 |
| US | 4,557,313 | 115,312 | 10,594 | 0.0253 |

## 3. Test profile per country (vs train OOF)

| country | with_match | matches_per_s1 | candidates_per_s1 | p_max_median | train_with_match | train_matches_per_s1 |
|---|---|---|---|---|---|---|
| France | 0.9479 | 3.4382 | 6.4220 | 1.0000 | None | None |
| India | 0.9433 | 3.3897 | 6.4484 | 1.0000 | 0.9432 | 3.3796 |
| US | 0.9447 | 3.4605 | 6.4688 | 1.0000 | 0.9428 | 3.3639 |
