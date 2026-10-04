# v3 ablation A0 — embedders and approximate search

40,000 S1 queries; corpus = all their true matches + 20% of other S2/S3 records of the same country. Recall@k = share of true pairs in the query's top-k.

- `Snowflake/snowflake-arctic-embed-m-v2.0`: failed (AssertionError: please install xformers)

## Step 1 — embedders (exact search)


**all**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| multilingual-e5-large | India | 0.2640 | 0.8893 | 0.9517 | 0.9610 | 0.9695 | 55327 |
| multilingual-e5-small | India | 0.2629 | 0.8791 | 0.9429 | 0.9559 | 0.9667 | 55327 |
| snowflake-arctic-embed-l-v2.0 | India | 0.2619 | 0.8773 | 0.9410 | 0.9545 | 0.9665 | 55327 |
| multilingual-e5-large-instruct | India | 0.2538 | 0.8061 | 0.8784 | 0.9027 | 0.9263 | 55327 |
| bhasha-embed-v0 | India | 0.2566 | 0.8026 | 0.8697 | 0.8944 | 0.9175 | 55327 |
| granite-embedding-311m-multilingual-r2 | India | 0.2337 | 0.6674 | 0.7356 | 0.7721 | 0.8116 | 55327 |
| granite-embedding-97m-multilingual-r2 | India | 0.2283 | 0.6283 | 0.6868 | 0.7180 | 0.7516 | 55327 |
| IndicBERTv2-SS | India | 0.1561 | 0.2644 | 0.2745 | 0.2840 | 0.2946 | 55327 |
| multilingual-e5-small | US | 0.2665 | 0.9174 | 0.9774 | 0.9832 | 0.9881 | 82857 |
| multilingual-e5-large | US | 0.2660 | 0.9142 | 0.9743 | 0.9808 | 0.9860 | 82857 |
| snowflake-arctic-embed-l-v2.0 | US | 0.2656 | 0.9126 | 0.9732 | 0.9804 | 0.9870 | 82857 |
| multilingual-e5-large-instruct | US | 0.2634 | 0.9084 | 0.9708 | 0.9772 | 0.9836 | 82857 |
| granite-embedding-97m-multilingual-r2 | US | 0.2554 | 0.8095 | 0.8813 | 0.9052 | 0.9272 | 82857 |
| granite-embedding-311m-multilingual-r2 | US | 0.2514 | 0.7940 | 0.8711 | 0.8990 | 0.9245 | 82857 |
| bhasha-embed-v0 | US | 0.2555 | 0.7673 | 0.8315 | 0.8576 | 0.8844 | 82857 |
| IndicBERTv2-SS | US | 0.1559 | 0.2737 | 0.2923 | 0.3089 | 0.3290 | 82857 |

**no address**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| multilingual-e5-large-instruct | India | 0.0092 | 0.2271 | 0.3582 | 0.4326 | 0.5182 | 2175 |
| granite-embedding-97m-multilingual-r2 | India | 0.0317 | 0.2786 | 0.3568 | 0.4156 | 0.4759 | 2175 |
| multilingual-e5-small | India | 0.0032 | 0.2046 | 0.3398 | 0.4294 | 0.5223 | 2175 |
| multilingual-e5-large | India | 0.0032 | 0.1641 | 0.3057 | 0.3848 | 0.4837 | 2175 |
| granite-embedding-311m-multilingual-r2 | India | 0.0170 | 0.2147 | 0.2929 | 0.3568 | 0.4331 | 2175 |
| snowflake-arctic-embed-l-v2.0 | India | 0.0018 | 0.1540 | 0.2910 | 0.4060 | 0.5182 | 2175 |
| bhasha-embed-v0 | India | 0.0074 | 0.1448 | 0.2326 | 0.2874 | 0.3591 | 2175 |
| IndicBERTv2-SS | India | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2175 |
| multilingual-e5-large-instruct | US | 0.0125 | 0.5236 | 0.7586 | 0.8163 | 0.8722 | 3919 |
| multilingual-e5-small | US | 0.0117 | 0.5017 | 0.7356 | 0.8028 | 0.8602 | 3919 |
| multilingual-e5-large | US | 0.0084 | 0.4218 | 0.6550 | 0.7346 | 0.8033 | 3919 |
| snowflake-arctic-embed-l-v2.0 | US | 0.0064 | 0.3600 | 0.6012 | 0.6997 | 0.7936 | 3919 |
| granite-embedding-311m-multilingual-r2 | US | 0.0309 | 0.4182 | 0.5512 | 0.6178 | 0.6864 | 3919 |
| granite-embedding-97m-multilingual-r2 | US | 0.0296 | 0.3881 | 0.5195 | 0.5790 | 0.6402 | 3919 |
| bhasha-embed-v0 | US | 0.0084 | 0.2230 | 0.3409 | 0.3970 | 0.4596 | 3919 |
| IndicBERTv2-SS | US | 0.0000 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 3919 |

**trade name (name sim < 50)**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| multilingual-e5-small | India | 0.0943 | 0.6981 | 0.8404 | 0.8620 | 0.8830 | 1855 |
| multilingual-e5-large | India | 0.0663 | 0.5849 | 0.7358 | 0.7790 | 0.8189 | 1855 |
| snowflake-arctic-embed-l-v2.0 | India | 0.0658 | 0.5628 | 0.7170 | 0.7617 | 0.8124 | 1855 |
| multilingual-e5-large-instruct | India | 0.0674 | 0.4485 | 0.5682 | 0.6178 | 0.6836 | 1855 |
| IndicBERTv2-SS | India | 0.2199 | 0.4205 | 0.4399 | 0.4582 | 0.4744 | 1855 |
| bhasha-embed-v0 | India | 0.0410 | 0.3272 | 0.4059 | 0.4534 | 0.5013 | 1855 |
| granite-embedding-97m-multilingual-r2 | India | 0.0474 | 0.2340 | 0.2900 | 0.3208 | 0.3601 | 1855 |
| granite-embedding-311m-multilingual-r2 | India | 0.0377 | 0.2216 | 0.2857 | 0.3197 | 0.3580 | 1855 |
| snowflake-arctic-embed-l-v2.0 | US | 0.0497 | 0.5555 | 0.7536 | 0.8219 | 0.8890 | 2252 |
| multilingual-e5-large | US | 0.0369 | 0.5249 | 0.7114 | 0.7798 | 0.8393 | 2252 |
| multilingual-e5-small | US | 0.0320 | 0.5067 | 0.6918 | 0.7540 | 0.8215 | 2252 |
| multilingual-e5-large-instruct | US | 0.0324 | 0.4067 | 0.5409 | 0.5981 | 0.6798 | 2252 |
| granite-embedding-97m-multilingual-r2 | US | 0.0266 | 0.2469 | 0.3259 | 0.3632 | 0.4165 | 2252 |
| granite-embedding-311m-multilingual-r2 | US | 0.0182 | 0.2052 | 0.2762 | 0.3113 | 0.3539 | 2252 |
| bhasha-embed-v0 | US | 0.0244 | 0.1936 | 0.2629 | 0.3042 | 0.3544 | 2252 |
| IndicBERTv2-SS | US | 0.0302 | 0.0662 | 0.0795 | 0.0888 | 0.1008 | 2252 |

**native script**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| multilingual-e5-large | India | 0.1361 | 0.8572 | 0.9548 | 0.9718 | 0.9847 | 9855 |
| snowflake-arctic-embed-l-v2.0 | India | 0.1354 | 0.8056 | 0.9057 | 0.9363 | 0.9625 | 9855 |
| multilingual-e5-small | India | 0.1103 | 0.7819 | 0.8908 | 0.9248 | 0.9528 | 9855 |
| bhasha-embed-v0 | India | 0.1958 | 0.7992 | 0.8723 | 0.9070 | 0.9404 | 9855 |
| multilingual-e5-large-instruct | India | 0.0619 | 0.4956 | 0.6339 | 0.7115 | 0.7896 | 9855 |
| granite-embedding-311m-multilingual-r2 | India | 0.0510 | 0.2577 | 0.3337 | 0.4013 | 0.4856 | 9855 |
| granite-embedding-97m-multilingual-r2 | India | 0.0213 | 0.0983 | 0.1368 | 0.1687 | 0.2149 | 9855 |
| IndicBERTv2-SS | India | 0.0196 | 0.0492 | 0.0550 | 0.0611 | 0.0686 | 9855 |

**Speed on this GPU (fp16, max 64 tokens) and projected time for all 24.2M records**

| model | docs_per_s | load_s | full_embed_min |
|---|---|---|---|
| multilingual-e5-small | 15866 | 6.4000 | 25.5000 |
| multilingual-e5-large | 4170 | 9.4000 | 96.9000 |
| multilingual-e5-large-instruct | 2966 | 9.3000 | 136.1000 |
| snowflake-arctic-embed-l-v2.0 | 4488 | 9.8000 | 90.0000 |
| granite-embedding-311m-multilingual-r2 | 4194 | 7.0000 | 96.3000 |
| granite-embedding-97m-multilingual-r2 | 11065 | 4.1000 | 36.5000 |
| bhasha-embed-v0 | 8877 | 5.5000 | 45.5000 |
| IndicBERTv2-SS | 9302 | 6.7000 | 43.4000 |

**Best embedder by mean R@10: `intfloat/multilingual-e5-large`**

| model | R@10 | R@20 |
|---|---|---|
| multilingual-e5-large | 0.9630 | 0.9709 |
| multilingual-e5-small | 0.9601 | 0.9695 |
| snowflake-arctic-embed-l-v2.0 | 0.9571 | 0.9674 |
| multilingual-e5-large-instruct | 0.9246 | 0.9400 |
| bhasha-embed-v0 | 0.8506 | 0.8760 |
| granite-embedding-311m-multilingual-r2 | 0.8034 | 0.8356 |
| granite-embedding-97m-multilingual-r2 | 0.7840 | 0.8116 |
| IndicBERTv2-SS | 0.2834 | 0.2965 |

## Step 2 — approximate search on `intfloat/multilingual-e5-large` (corpus per country ~1-2M)

vs_exact@25 = share of the exact top-25 that the approximate top-25 also returns.

| country | method | seconds | vs_exact@25 | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|---|---|
| India | exact (GPU brute force) | nan | 1.0000 | 0.2640 | 0.8893 | 0.9517 | 0.9610 | 0.9695 | 55327 |
| India | GPU IVF nprobe=32 | 1.6000 | 0.7293 | 0.2567 | 0.7719 | 0.8060 | 0.8092 | 0.8117 | 55327 |
| India | GPU IVF nprobe=64 | 1.5000 | 0.7993 | 0.2599 | 0.8142 | 0.8552 | 0.8592 | 0.8622 | 55327 |
| India | GPU IVF nprobe=128 | 1.5000 | 0.8596 | 0.2621 | 0.8467 | 0.8946 | 0.8992 | 0.9031 | 55327 |
| India | GPU IVF nprobe=256 | 1.5000 | 0.9050 | 0.2630 | 0.8656 | 0.9182 | 0.9236 | 0.9282 | 55327 |
| India | FAISS IVF-Flat CPU nprobe=64 | 25.8000 | 0.7994 | 0.2599 | 0.8141 | 0.8553 | 0.8592 | 0.8623 | 55327 |
| US | exact (GPU brute force) | nan | 1.0000 | 0.2660 | 0.9142 | 0.9743 | 0.9808 | 0.9860 | 82857 |
| US | GPU IVF nprobe=32 | 2.0000 | 0.7286 | 0.2645 | 0.8719 | 0.9174 | 0.9206 | 0.9230 | 82857 |
| US | GPU IVF nprobe=64 | 1.8000 | 0.7688 | 0.2650 | 0.8828 | 0.9307 | 0.9341 | 0.9366 | 82857 |
| US | GPU IVF nprobe=128 | 1.8000 | 0.8089 | 0.2653 | 0.8892 | 0.9391 | 0.9428 | 0.9455 | 82857 |
| US | GPU IVF nprobe=256 | 1.8000 | 0.8489 | 0.2654 | 0.8947 | 0.9460 | 0.9500 | 0.9530 | 82857 |
| US | FAISS IVF-Flat CPU nprobe=64 | 44.7000 | 0.7684 | 0.2649 | 0.8826 | 0.9305 | 0.9339 | 0.9364 | 82857 |

## Step 3 — GPU IVF speed at full size (random vectors)

| base | queries | dim | k | nprobe | seconds |
|---|---|---|---|---|---|
| 6186873 | 1323633 | 384 | 25 | 64 | 8.2000 |
| 1323633 | 6186873 | 384 | 5 | 64 | 13.2000 |