# Deep diagnosis of run `v1b`

Signals are scored by AUC between true matches and the look-alike non-matches of each pool. **AUC_confused_band** is measured only where v2 was unsure (0.02 ≤ p ≤ 0.95): a signal that is high there is new information for v3.


## E1 · S2/S3 record without address

true matches 60,000 vs look-alike non-matches 60,000; confused band (v2 p in [0.02, 0.95]): 109,759 rows

| signal | AUC_all | AUC_confused_band | coverage | mean_true | mean_false |
|---|---|---|---|---|---|
| name uniqueness (1 / #S1 with same core) | 0.8495 | 0.8471 | 1.0000 | 0.7039 | 0.1926 |
| name core Jaro-Winkler | 0.5439 | 0.5467 | 1.0000 | 0.9770 | 0.9640 |
| name core ratio | 0.5437 | 0.5463 | 1.0000 | 94.5097 | 91.5745 |
| name core Levenshtein | 0.5430 | 0.5457 | 1.0000 | 0.9162 | 0.8750 |
| name Monge-Elkan (JW) | 0.5435 | 0.5451 | 1.0000 | 0.9760 | 0.9602 |
| name core WRatio | 0.5406 | 0.5432 | 1.0000 | 97.0394 | 95.4335 |
| name metaphone overlap | 0.5391 | 0.5410 | 1.0000 | 0.8986 | 0.8601 |
| name SoftTF-IDF (JW>=0.9) | 0.5383 | 0.5403 | 1.0000 | 0.9432 | 0.9244 |
| name char-3gram TF-IDF cos | 0.5355 | 0.5377 | 1.0000 | 0.9306 | 0.9123 |
| name romanize: anyascii token_set | 0.5208 | 0.5247 | 1.0000 | 97.3710 | 96.3012 |
| name romanize: best of three | 0.5195 | 0.5235 | 1.0000 | 97.7387 | 96.8100 |
| trade: b name is invented (words unseen elsewhere) | 0.5251 | 0.5235 | 1.0000 | 0.0336 | 0.0149 |
| name romanize: learned lexicon token_set (v2) | 0.5188 | 0.5229 | 1.0000 | 97.6997 | 96.7943 |
| name core token_set (v2) | 0.5089 | 0.5105 | 1.0000 | 97.9901 | 96.8549 |
| name core partial | 0.5060 | 0.5091 | 1.0000 | 97.5499 | 96.8847 |
| trade: domain / compact containment | 0.5032 | 0.5047 | 0.9996 | 0.9477 | 0.9410 |
| trade: initials match | 0.5000 | 0.5000 | 1.0000 | 0.0000 | 0.0000 |
| addr token_set (v2) | None | None | 0.0000 | None | None |
| addr ratio | None | None | 0.0000 | None | None |
| addr exact | None | None | 0.0000 | None | None |
| addr char-3gram TF-IDF cos | None | None | 0.0000 | None | None |
| addr IDF-weighted agreement (-disagreement) | None | None | 0.0000 | None | None |
| num relation (equal>trunc>edit1>near>diff) | None | None | 0.0000 | None | None |
| num all S1 numbers contained in b | None | None | 0.0000 | None | None |
| num extra numbers in b (negated) | None | None | 0.0000 | None | None |

**Number relation (first house number) — true vs look-alike**

| rel | true | false |
|---|---|---|
| missing | 60000 | 60000 |

**True matches of this pool by outcome in v2**

| cause | len |
|---|---|
| found | 34590 |
| below_threshold | 24669 |
| lost_to_one_owner | 741 |

## E2 · house number differs

true matches 60,000 vs look-alike non-matches 60,000; confused band (v2 p in [0.02, 0.95]): 69,310 rows

| signal | AUC_all | AUC_confused_band | coverage | mean_true | mean_false |
|---|---|---|---|---|---|
| num relation (equal>trunc>edit1>near>diff) | 0.5218 | 0.6195 | 1.0000 | 1.2325 | 1.0782 |
| trade: b name is invented (words unseen elsewhere) | 0.5544 | 0.5828 | 1.0000 | 0.0925 | 0.0162 |
| addr token_set (v2) | 0.7156 | 0.5727 | 1.0000 | 94.8117 | 90.3635 |
| addr char-3gram TF-IDF cos | 0.6818 | 0.5638 | 1.0000 | 0.8394 | 0.7539 |
| num all S1 numbers contained in b | 0.6883 | 0.5521 | 1.0000 | 0.4149 | 0.0383 |
| num extra numbers in b (negated) | 0.6233 | 0.5350 | 1.0000 | -0.8333 | -1.1246 |
| name uniqueness (1 / #S1 with same core) | 0.5092 | 0.5153 | 1.0000 | 0.5480 | 0.5361 |
| addr ratio | 0.5028 | 0.5069 | 1.0000 | 78.2685 | 78.2084 |
| trade: initials match | 0.5003 | 0.5001 | 1.0000 | 0.0006 | 0.0001 |
| addr exact | 0.5000 | 0.5000 | 1.0000 | 0.0000 | 0.0000 |
| addr IDF-weighted agreement (-disagreement) | 0.5165 | 0.4931 | 1.0000 | -0.1963 | -0.2382 |
| name romanize: best of three | 0.4792 | 0.3747 | 1.0000 | 93.8117 | 96.5149 |
| name romanize: anyascii token_set | 0.4310 | 0.3736 | 1.0000 | 90.7712 | 95.2858 |
| name romanize: learned lexicon token_set (v2) | 0.4793 | 0.3734 | 1.0000 | 93.6962 | 96.4846 |
| name core Levenshtein | 0.4755 | 0.3643 | 1.0000 | 0.8719 | 0.9041 |
| name core ratio | 0.4743 | 0.3588 | 1.0000 | 91.1434 | 93.9174 |
| name core Jaro-Winkler | 0.4748 | 0.3574 | 1.0000 | 0.9530 | 0.9733 |
| name Monge-Elkan (JW) | 0.4653 | 0.3557 | 1.0000 | 0.9471 | 0.9739 |
| name metaphone overlap | 0.4536 | 0.3551 | 1.0000 | 0.7981 | 0.8873 |
| name SoftTF-IDF (JW>=0.9) | 0.4620 | 0.3544 | 1.0000 | 0.8566 | 0.9364 |
| trade: domain / compact containment | 0.4516 | 0.3535 | 0.9994 | 0.9025 | 0.9600 |
| name core WRatio | 0.4659 | 0.3439 | 1.0000 | 94.3181 | 97.3218 |
| name core partial | 0.4339 | 0.3430 | 1.0000 | 94.9022 | 98.1005 |
| name core token_set (v2) | 0.4278 | 0.3392 | 1.0000 | 93.7431 | 98.9939 |
| name char-3gram TF-IDF cos | 0.4608 | 0.3383 | 1.0000 | 0.8737 | 0.9312 |

**Number relation (first house number) — true vs look-alike**

| rel | true | false |
|---|---|---|
| different | 31425 | 22798 |
| truncated | 18368 | 2938 |
| one_digit_edit | 8637 | 21611 |
| near_twin | 1570 | 12653 |

**True matches of this pool by outcome in v2**

| cause | len |
|---|---|
| found | 55996 |
| below_threshold | 3996 |
| lost_to_one_owner | 8 |

## E3 · trade names (name unrelated, not native script)

true matches 60,000 vs look-alike non-matches 60,000; confused band (v2 p in [0.02, 0.95]): 77,163 rows

| signal | AUC_all | AUC_confused_band | coverage | mean_true | mean_false |
|---|---|---|---|---|---|
| addr char-3gram TF-IDF cos | 0.6818 | 0.6458 | 0.9992 | 0.9229 | 0.8456 |
| num all S1 numbers contained in b | 0.6575 | 0.6294 | 0.8992 | 0.8751 | 0.5602 |
| num relation (equal>trunc>edit1>near>diff) | 0.6180 | 0.6027 | 0.8992 | 3.6220 | 2.8310 |
| addr token_set (v2) | 0.6219 | 0.6003 | 0.9992 | 97.8809 | 96.1750 |
| num extra numbers in b (negated) | 0.6168 | 0.5910 | 0.9083 | -0.1671 | -0.4344 |
| addr ratio | 0.6147 | 0.5904 | 0.9992 | 86.0630 | 80.3132 |
| addr IDF-weighted agreement (-disagreement) | 0.6061 | 0.5819 | 0.9992 | -0.1296 | -0.2182 |
| addr exact | 0.5938 | 0.5706 | 0.9992 | 0.3365 | 0.1489 |
| trade: domain / compact containment | 0.6624 | 0.5081 | 0.9999 | 0.2908 | 0.0200 |
| trade: b name is invented (words unseen elsewhere) | 0.5526 | 0.5073 | 1.0000 | 0.4435 | 0.3461 |
| trade: initials match | 0.5268 | 0.5021 | 1.0000 | 0.0541 | 0.0004 |
| name SoftTF-IDF (JW>=0.9) | 0.5722 | 0.4971 | 1.0000 | 0.1280 | 0.0421 |
| name metaphone overlap | 0.5229 | 0.4927 | 1.0000 | 0.0339 | 0.0191 |
| name char-3gram TF-IDF cos | 0.6678 | 0.4924 | 1.0000 | 0.2237 | 0.0336 |
| name core partial | 0.6822 | 0.4914 | 1.0000 | 55.5737 | 39.5110 |
| name uniqueness (1 / #S1 with same core) | 0.5054 | 0.4886 | 1.0000 | 0.5847 | 0.5765 |
| name Monge-Elkan (JW) | 0.6668 | 0.4884 | 1.0000 | 0.5801 | 0.4871 |
| name core WRatio | 0.6783 | 0.4874 | 1.0000 | 50.9530 | 35.0739 |
| name core Jaro-Winkler | 0.6639 | 0.4827 | 1.0000 | 0.6104 | 0.4881 |
| name core ratio | 0.6401 | 0.4818 | 1.0000 | 41.0039 | 27.2773 |
| name core token_set (v2) | 0.6285 | 0.4817 | 1.0000 | 32.0349 | 26.7631 |
| name core Levenshtein | 0.6352 | 0.4796 | 1.0000 | 0.2997 | 0.1646 |
| name romanize: learned lexicon token_set (v2) | 0.6038 | 0.4725 | 1.0000 | 35.9286 | 28.2372 |
| name romanize: anyascii token_set | 0.5988 | 0.4520 | 1.0000 | 36.9320 | 28.7489 |
| name romanize: best of three | 0.5955 | 0.4463 | 1.0000 | 37.2141 | 29.2381 |

**Number relation (first house number) — true vs look-alike**

| rel | true | false |
|---|---|---|
| equal | 47476 | 33572 |
| missing | 4951 | 7148 |
| different | 3923 | 12088 |
| truncated | 2372 | 2167 |
| one_digit_edit | 1087 | 3808 |
| near_twin | 191 | 1217 |

**True matches of this pool by outcome in v2**

| cause | len |
|---|---|
| found | 53618 |
| below_threshold | 6020 |
| lost_to_one_owner | 362 |

## E4 · native-script names

true matches 60,000 vs look-alike non-matches 29,481; confused band (v2 p in [0.02, 0.95]): 30,436 rows

| signal | AUC_all | AUC_confused_band | coverage | mean_true | mean_false |
|---|---|---|---|---|---|
| name metaphone overlap | 0.7678 | 0.7086 | 1.0000 | 0.9778 | 0.6218 |
| name core Levenshtein | 0.7716 | 0.6711 | 1.0000 | 0.9814 | 0.7201 |
| name core ratio | 0.7718 | 0.6651 | 1.0000 | 98.7092 | 78.8898 |
| name romanize: anyascii token_set | 0.6778 | 0.6401 | 1.0000 | 73.5315 | 63.0000 |
| name core WRatio | 0.7683 | 0.6388 | 1.0000 | 98.9512 | 83.6113 |
| name core Jaro-Winkler | 0.7658 | 0.6309 | 1.0000 | 0.9896 | 0.8731 |
| name Monge-Elkan (JW) | 0.7657 | 0.6271 | 1.0000 | 0.9893 | 0.8656 |
| name char-3gram TF-IDF cos | 0.7659 | 0.6210 | 1.0000 | 0.9721 | 0.6858 |
| name core token_set (v2) | 0.7210 | 0.6071 | 1.0000 | 98.9957 | 83.2165 |
| name core partial | 0.7188 | 0.5995 | 1.0000 | 99.0252 | 86.0634 |
| num relation (equal>trunc>edit1>near>diff) | 0.7352 | 0.5923 | 0.8773 | 3.2752 | 1.8343 |
| name romanize: best of three | 0.7221 | 0.5866 | 1.0000 | 98.9699 | 86.8978 |
| name romanize: learned lexicon token_set (v2) | 0.7222 | 0.5864 | 1.0000 | 98.9676 | 86.6354 |
| trade: domain / compact containment | 0.6922 | 0.5832 | 0.9991 | 0.9749 | 0.7338 |
| name SoftTF-IDF (JW>=0.9) | 0.7580 | 0.5670 | 1.0000 | 0.9644 | 0.6928 |
| num extra numbers in b (negated) | 0.7819 | 0.5659 | 0.9083 | -0.2405 | -0.9367 |
| num all S1 numbers contained in b | 0.7921 | 0.5386 | 0.8773 | 0.8214 | 0.2371 |
| addr exact | 0.5779 | 0.5105 | 0.9843 | 0.1645 | 0.0087 |
| trade: initials match | 0.5000 | 0.5000 | 1.0000 | 0.0000 | 0.0000 |
| trade: b name is invented (words unseen elsewhere) | 0.5000 | 0.5000 | 1.0000 | 0.0000 | 0.0000 |
| addr char-3gram TF-IDF cos | 0.7804 | 0.4725 | 0.9843 | 0.8566 | 0.6205 |
| name uniqueness (1 / #S1 with same core) | 0.4768 | 0.4693 | 1.0000 | 0.1092 | 0.1533 |
| addr token_set (v2) | 0.7928 | 0.4671 | 0.9843 | 97.5950 | 88.7958 |
| addr ratio | 0.7021 | 0.4388 | 0.9843 | 78.3708 | 63.8765 |
| addr IDF-weighted agreement (-disagreement) | 0.7106 | 0.4309 | 0.9843 | -0.2489 | -0.5105 |

**Number relation (first house number) — true vs look-alike**

| rel | true | false |
|---|---|---|
| equal | 40652 | 7221 |
| different | 7868 | 9618 |
| missing | 6604 | 4371 |
| truncated | 2814 | 2299 |
| one_digit_edit | 1769 | 4306 |
| near_twin | 293 | 1666 |

**True matches of this pool by outcome in v2**

| cause | len |
|---|---|
| found | 59640 |
| below_threshold | 351 |
| lost_to_one_owner | 9 |

## E5 · typos / clean pairs near the threshold

true matches 60,000 vs look-alike non-matches 60,000; confused band (v2 p in [0.02, 0.95]): 119,427 rows

| signal | AUC_all | AUC_confused_band | coverage | mean_true | mean_false |
|---|---|---|---|---|---|
| num relation (equal>trunc>edit1>near>diff) | 0.9059 | 0.9066 | 0.8804 | 4.0000 | 1.6466 |
| num extra numbers in b (negated) | 0.7645 | 0.7657 | 0.9527 | -0.3734 | -0.9822 |
| num all S1 numbers contained in b | 0.7585 | 0.7595 | 0.8804 | 0.7041 | 0.1871 |
| trade: b name is invented (words unseen elsewhere) | 0.6907 | 0.6911 | 1.0000 | 0.2186 | 0.0317 |
| addr token_set (v2) | 0.6601 | 0.6616 | 1.0000 | 93.8371 | 90.9701 |
| addr char-3gram TF-IDF cos | 0.6125 | 0.6132 | 1.0000 | 0.8098 | 0.7562 |
| addr exact | 0.5551 | 0.5554 | 1.0000 | 0.1364 | 0.0261 |
| name uniqueness (1 / #S1 with same core) | 0.5329 | 0.5324 | 1.0000 | 0.5963 | 0.5480 |
| addr ratio | 0.5281 | 0.5282 | 1.0000 | 77.1481 | 76.1306 |
| trade: initials match | 0.5000 | 0.4999 | 1.0000 | 0.0000 | 0.0001 |
| addr IDF-weighted agreement (-disagreement) | 0.4863 | 0.4862 | 1.0000 | -0.3129 | -0.2938 |
| name romanize: anyascii token_set | 0.4107 | 0.4103 | 1.0000 | 86.4057 | 88.9318 |
| name core ratio | 0.3818 | 0.3823 | 1.0000 | 79.7641 | 85.3100 |
| name core Levenshtein | 0.3812 | 0.3817 | 1.0000 | 0.7174 | 0.7954 |
| name Monge-Elkan (JW) | 0.3813 | 0.3817 | 1.0000 | 0.8814 | 0.9142 |
| name core Jaro-Winkler | 0.3802 | 0.3807 | 1.0000 | 0.8888 | 0.9202 |
| name romanize: best of three | 0.3745 | 0.3748 | 1.0000 | 86.6969 | 90.8803 |
| name romanize: learned lexicon token_set (v2) | 0.3670 | 0.3674 | 1.0000 | 86.3256 | 90.8324 |
| name core WRatio | 0.3637 | 0.3641 | 1.0000 | 84.9835 | 89.8109 |
| name core partial | 0.3603 | 0.3607 | 1.0000 | 86.5473 | 91.4235 |
| name core token_set (v2) | 0.3574 | 0.3576 | 1.0000 | 84.9870 | 90.3631 |
| trade: domain / compact containment | 0.3476 | 0.3475 | 0.9995 | 0.6535 | 0.8153 |
| name metaphone overlap | 0.3446 | 0.3445 | 1.0000 | 0.5173 | 0.7080 |
| name SoftTF-IDF (JW>=0.9) | 0.3292 | 0.3293 | 1.0000 | 0.6662 | 0.8160 |
| name char-3gram TF-IDF cos | 0.3212 | 0.3213 | 1.0000 | 0.6559 | 0.8018 |

**Number relation (first house number) — true vs look-alike**

| rel | true | false |
|---|---|---|
| equal | 48905 | 10684 |
| missing | 11095 | 3256 |
| different | 0 | 18468 |
| one_digit_edit | 0 | 15584 |
| truncated | 0 | 3762 |
| near_twin | 0 | 8246 |

**True matches of this pool by outcome in v2**

| cause | len |
|---|---|
| found | 46809 |
| below_threshold | 13007 |
| lost_to_one_owner | 184 |

## E6 · same name, different address (false merges)

true matches 60,000 vs look-alike non-matches 60,000; confused band (v2 p in [0.02, 0.95]): 103,018 rows

| signal | AUC_all | AUC_confused_band | coverage | mean_true | mean_false |
|---|---|---|---|---|---|
| name uniqueness (1 / #S1 with same core) | 0.9333 | 0.9319 | 1.0000 | 0.8668 | 0.1678 |
| num extra numbers in b (negated) | 0.7427 | 0.6321 | 0.0269 | -0.3643 | -1.0249 |
| addr char-3gram TF-IDF cos | 0.6791 | 0.6121 | 0.0321 | 0.3159 | 0.2470 |
| num relation (equal>trunc>edit1>near>diff) | 0.6737 | 0.5805 | 0.0268 | 2.6471 | 1.4802 |
| num all S1 numbers contained in b | 0.6912 | 0.5582 | 0.0268 | 0.5112 | 0.1288 |
| addr token_set (v2) | 0.6156 | 0.5399 | 0.0321 | 54.6386 | 51.7720 |
| name core Jaro-Winkler | 0.5303 | 0.5366 | 1.0000 | 0.9876 | 0.9808 |
| name core ratio | 0.5307 | 0.5366 | 1.0000 | 96.7351 | 94.7218 |
| name core Levenshtein | 0.5303 | 0.5363 | 1.0000 | 0.9462 | 0.9156 |
| name Monge-Elkan (JW) | 0.5316 | 0.5363 | 1.0000 | 0.9891 | 0.9819 |
| name core WRatio | 0.5294 | 0.5354 | 1.0000 | 98.6261 | 98.0285 |
| name SoftTF-IDF (JW>=0.9) | 0.5296 | 0.5344 | 1.0000 | 0.9687 | 0.9576 |
| name metaphone overlap | 0.5272 | 0.5323 | 1.0000 | 0.9413 | 0.9232 |
| name char-3gram TF-IDF cos | 0.5265 | 0.5319 | 1.0000 | 0.9613 | 0.9541 |
| trade: b name is invented (words unseen elsewhere) | 0.5204 | 0.5180 | 1.0000 | 0.0235 | 0.0090 |
| addr ratio | 0.5846 | 0.5164 | 0.0321 | 48.3283 | 45.3137 |
| name romanize: anyascii token_set | 0.5074 | 0.5143 | 1.0000 | 98.2024 | 97.6820 |
| addr IDF-weighted agreement (-disagreement) | 0.5924 | 0.5133 | 0.0321 | -0.8417 | -0.8718 |
| name romanize: best of three | 0.5066 | 0.5126 | 1.0000 | 98.6314 | 98.2139 |
| name romanize: learned lexicon token_set (v2) | 0.5063 | 0.5124 | 1.0000 | 98.6143 | 98.2020 |
| trade: initials match | 0.5000 | 0.5000 | 1.0000 | 0.0000 | 0.0000 |
| addr exact | 0.5000 | 0.5000 | 0.0321 | 0.0000 | 0.0000 |
| name core token_set (v2) | 0.4824 | 0.4846 | 1.0000 | 99.5483 | 99.7615 |
| name core partial | 0.4790 | 0.4836 | 1.0000 | 98.7426 | 99.0847 |
| trade: domain / compact containment | 0.4808 | 0.4825 | 0.9996 | 0.9735 | 0.9871 |

**Number relation (first house number) — true vs look-alike**

| rel | true | false |
|---|---|---|
| missing | 57461 | 59317 |
| equal | 1505 | 149 |
| different | 756 | 319 |
| truncated | 169 | 42 |
| one_digit_edit | 85 | 116 |
| near_twin | 24 | 57 |

**True matches of this pool by outcome in v2**

| cause | len |
|---|---|
| found | 60000 |

## Blocking misses — which extra route would have found them


**no address**

| country | missed | exact_core | exact_compact | exact_addr | exact_numbers | name_tset_ge_90 | addr_tset_ge_90 | anyascii_name_ge_80 | any_route |
|---|---|---|---|---|---|---|---|---|---|
| India | 46592 | 0.5464 | 0.5464 | 0.0000 | 0.0000 | 0.6892 | 0.0000 | 0.9135 | 0.9306 |
| US | 41819 | 0.3240 | 0.3240 | 0.0000 | 0.0000 | 0.6774 | 0.0000 | 0.8679 | 0.8886 |

**trade names**

| country | missed | exact_core | exact_compact | exact_addr | exact_numbers | name_tset_ge_90 | addr_tset_ge_90 | anyascii_name_ge_80 | any_route |
|---|---|---|---|---|---|---|---|---|---|
| India | 8972 | 0.0000 | 0.0042 | 0.0114 | 0.3135 | 0.0000 | 0.6487 | 0.0422 | 0.7548 |
| US | 13633 | 0.0000 | 0.0002 | 0.1422 | 0.4524 | 0.0000 | 0.7013 | 0.0053 | 0.8248 |

**native script**

| country | missed | exact_core | exact_compact | exact_addr | exact_numbers | name_tset_ge_90 | addr_tset_ge_90 | anyascii_name_ge_80 | any_route |
|---|---|---|---|---|---|---|---|---|---|
| India | 22373 | 0.9226 | 0.9226 | 0.0007 | 0.2086 | 0.9341 | 0.5730 | 0.3025 | 0.9860 |

**all missed**

| country | missed | exact_core | exact_compact | exact_addr | exact_numbers | name_tset_ge_90 | addr_tset_ge_90 | anyascii_name_ge_80 | any_route |
|---|---|---|---|---|---|---|---|---|---|
| India | 86519 | 0.5516 | 0.5599 | 0.0014 | 0.1033 | 0.6548 | 0.2734 | 0.6432 | 0.9262 |
| US | 56154 | 0.2421 | 0.2435 | 0.0347 | 0.1142 | 0.5083 | 0.1734 | 0.6528 | 0.8733 |

## Ceiling check — no-address true pairs: is the name enough?

| country_a | name_bucket | cause | len |
|---|---|---|---|
| India | 2-3 S1 share it | missed_by_blocking | 6864 |
| India | 2-3 S1 share it | below_threshold | 6317 |
| India | 2-3 S1 share it | found | 6203 |
| India | 2-3 S1 share it | lost_to_one_owner | 1075 |
| India | 4+ S1 share it | missed_by_blocking | 26199 |
| India | 4+ S1 share it | below_threshold | 8563 |
| India | 4+ S1 share it | found | 1897 |
| India | 4+ S1 share it | lost_to_one_owner | 482 |
| India | unique name | found | 37965 |
| India | unique name | missed_by_blocking | 13529 |
| India | unique name | below_threshold | 10898 |
| India | unique name | lost_to_one_owner | 95 |
| US | 2-3 S1 share it | below_threshold | 18898 |
| US | 2-3 S1 share it | found | 12289 |
| US | 2-3 S1 share it | missed_by_blocking | 4583 |
| US | 2-3 S1 share it | lost_to_one_owner | 986 |
| US | 4+ S1 share it | below_threshold | 37027 |
| US | 4+ S1 share it | missed_by_blocking | 28962 |
| US | 4+ S1 share it | found | 6634 |
| US | 4+ S1 share it | lost_to_one_owner | 424 |
| US | unique name | found | 78391 |
| US | unique name | below_threshold | 20416 |
| US | unique name | missed_by_blocking | 8274 |
| US | unique name | lost_to_one_owner | 47 |

| country_a | name_bucket | pairs | found | name_sim_ge_80 |
|---|---|---|---|---|
| India | 2-3 S1 share it | 20459 | 0.3032 | 0.8748 |
| India | 4+ S1 share it | 37141 | 0.0511 | 0.8523 |
| India | unique name | 62487 | 0.6076 | 0.8967 |
| US | 2-3 S1 share it | 36756 | 0.3343 | 0.9554 |
| US | 4+ S1 share it | 73047 | 0.0908 | 0.9195 |
| US | unique name | 107128 | 0.7318 | 0.9630 |

## Noise mining over the ground truth (what the sources change)

**name_n · India** — inserted in S2/S3: ltd 6.8%, center 3.2%, limited 2.4%, services 2.2%, shri 1.4%, smt 1.3%, mr 1.3%, sri 1.3%, dr 1.3%, ms 1.3%, service 1.1%, partners 0.9%, private 0.8%, co 0.7%, company 0.6%, corp 0.6%, corporation 0.6%, com 0.6%, dba 0.6%, llp 0.5%, as 0.3%, shrii 0.2%, id 0.2%, formerly 0.2%, aka 0.2%

**name_n · India** — dropped from S1: limited 20.0%, private 8.9%, ltd 6.1%, pvt 2.5%, india 1.4%, llp 1.1%, co 0.7%, brothers 0.6%, services 0.5%, solutions 0.5%, trading 0.5%, company 0.4%, corp 0.3%, corporation 0.3%, technologies 0.3%, traders 0.3%, group 0.3%, sons 0.3%, foundation 0.3%, industries 0.3%, shree 0.3%, consultancy 0.2%, trust 0.2%, enterprises 0.2%, consultants 0.2%

**name_n · US** — inserted in S2/S3: center 3.0%, inc 2.2%, services 2.2%, the 1.4%, llc 1.3%, co 1.1%, corporation 1.1%, service 1.0%, corp 1.0%, lp 1.0%, ltd 0.9%, partners 0.8%, dba 0.8%, formerly 0.5%, as 0.4%, fka 0.4%, id 0.3%, aka 0.3%, and 0.2%, trading 0.2%, com 0.2%, business 0.2%, doing 0.2%, ta 0.1%, nee 0.1%

**name_n · US** — dropped from S1: llc 8.9%, inc 5.7%, pc 1.3%, group 0.9%, corp 0.8%, and 0.8%, associates 0.7%, care 0.7%, center 0.6%, partners 0.6%, lp 0.5%, pllc 0.5%, clinic 0.4%, health 0.4%, of 0.4%, s 0.3%, medicine 0.3%, specialists 0.2%, co 0.2%, corporation 0.2%, physicians 0.2%, company 0.2%, holdings 0.2%, global 0.2%, dds 0.2%

**addr_n · India** — inserted in S2/S3: no 7.0%, dl 4.7%, door 4.3%, h 4.2%, dillii 3.2%, tn 2.5%, a 1.8%, hn 1.7%, block 1.6%, plot 1.5%, g 1.3%, d 1.3%, c 1.2%, pradesh 1.2%, andhra 1.2%, b3 1.1%, n 0.7%, bombay 0.7%, region 0.6%, bengaluru 0.5%, calcutta 0.3%, e 0.2%, f 0.2%, b 0.2%, urban 0.2%

**addr_n · India** — dropped from S1: road 6.3%, no 6.2%, nagar 4.5%, floor 4.3%, plot 3.1%, tamil 2.7%, nadu 2.7%, west 2.7%, city 2.6%, new 2.1%, south 2.0%, flat 1.8%, near 1.6%, h 1.5%, sector 1.5%, telangana 1.4%, north 1.4%, mumbai 1.4%, howrah 1.4%, colony 1.3%, east 1.3%, hyderabad 1.2%, b 1.2%, street 1.1%, a 1.1%

**addr_n · US** — inserted in S2/S3: city 2.6%, ct 2.1%, virginia 1.9%, po 1.6%, box 1.6%, pmb 1.5%, washington 1.5%, a 1.3%, cdp 1.1%, saint 1.1%, township 1.1%, n 1.0%, pl 0.9%, kansas 0.5%, d 0.5%, c 0.5%, b 0.4%, twp 0.3%, hwy 0.3%, columbia 0.2%, floor 0.2%, street 0.1%, ft 0.1%, east 0.1%, west 0.1%

**addr_n · US** — dropped from S1: unit 10.7%, court 2.3%, va 2.3%, street 2.1%, city 1.7%, wa 1.6%, county 1.5%, fl 1.3%, of 1.1%, place 1.1%, road 0.9%, apartment 0.8%, drive 0.8%, apt 0.8%, avenue 0.7%, ks 0.6%, town 0.5%, tx 0.5%, lane 0.4%, ny 0.4%, a 0.3%, nc 0.3%, b 0.3%, oh 0.3%, bldg 0.3%

**Words right before numbers that exist only in S2/S3 (inserted-number prefixes):** no 23,944, 1 12,436, box 9,429, pmb 9,139, hn 6,361, plot 6,182, a 2,434, c 2,051, d 1,961, b 1,745, g 1,692, f 1,175, e 1,148, 2 983, 3 796, 4 741, b3 630, 5 558, 6 521, 7 439, 8 420, 9 411, tx 293, maharashtra 243, nc 242, city 197, 10 195, delhi 184, ny 180, oh 180

**Repeated consecutive words in S2/S3:** name 2.44%, address 5.01%
