# v3 ablation A0 — embedders and approximate search

40,000 S1 queries; corpus = all their true matches + 20% of other S2/S3 records of the same country. Recall@k = share of true pairs in the query's top-k.

- `Alibaba-NLP/gte-multilingual-base`: failed (AttributeError: 'NewModel' object has no attribute 'get_extended_attention_mask')
- `Qwen/Qwen3-Embedding-0.6B`: failed (AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more inform)

## Step 1 — embedders (exact search)


**all**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| multilingual-e5-large | India | 0.2640 | 0.8893 | 0.9517 | 0.9610 | 0.9695 | 55327 |
| multilingual-e5-small | India | 0.2629 | 0.8791 | 0.9429 | 0.9559 | 0.9667 | 55327 |
| bge-m3 | India | 0.2595 | 0.8562 | 0.9167 | 0.9302 | 0.9418 | 55327 |
| multilingual-e5-base | India | 0.2579 | 0.8432 | 0.9062 | 0.9219 | 0.9383 | 55327 |
| multilingual-e5-small | US | 0.2665 | 0.9174 | 0.9774 | 0.9832 | 0.9881 | 82857 |
| multilingual-e5-large | US | 0.2660 | 0.9142 | 0.9743 | 0.9808 | 0.9860 | 82857 |
| multilingual-e5-base | US | 0.2639 | 0.9053 | 0.9666 | 0.9736 | 0.9805 | 82857 |
| bge-m3 | US | 0.2652 | 0.9057 | 0.9665 | 0.9737 | 0.9801 | 82857 |

**no address**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| multilingual-e5-small | India | 0.0032 | 0.2046 | 0.3398 | 0.4294 | 0.5223 | 2175 |
| multilingual-e5-large | India | 0.0032 | 0.1641 | 0.3057 | 0.3848 | 0.4837 | 2175 |
| multilingual-e5-base | India | 0.0023 | 0.1085 | 0.1959 | 0.2556 | 0.3370 | 2175 |
| bge-m3 | India | 0.0005 | 0.0372 | 0.0828 | 0.1163 | 0.1651 | 2175 |
| multilingual-e5-small | US | 0.0117 | 0.5017 | 0.7356 | 0.8028 | 0.8602 | 3919 |
| multilingual-e5-large | US | 0.0084 | 0.4218 | 0.6550 | 0.7346 | 0.8033 | 3919 |
| multilingual-e5-base | US | 0.0048 | 0.3256 | 0.5310 | 0.6088 | 0.6974 | 3919 |
| bge-m3 | US | 0.0046 | 0.2779 | 0.4841 | 0.5808 | 0.6708 | 3919 |

**native script**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| multilingual-e5-large | India | 0.1361 | 0.8572 | 0.9548 | 0.9718 | 0.9847 | 9855 |
| multilingual-e5-small | India | 0.1103 | 0.7819 | 0.8908 | 0.9248 | 0.9528 | 9855 |
| bge-m3 | India | 0.1561 | 0.7927 | 0.8828 | 0.9127 | 0.9378 | 9855 |
| multilingual-e5-base | India | 0.0948 | 0.6523 | 0.7638 | 0.8125 | 0.8627 | 9855 |

**trade name (name sim < 50)**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| multilingual-e5-small | India | 0.0943 | 0.6981 | 0.8404 | 0.8620 | 0.8830 | 1855 |
| bge-m3 | India | 0.0782 | 0.5908 | 0.7380 | 0.7806 | 0.8146 | 1855 |
| multilingual-e5-large | India | 0.0663 | 0.5849 | 0.7358 | 0.7790 | 0.8189 | 1855 |
| multilingual-e5-base | India | 0.0733 | 0.5590 | 0.6911 | 0.7337 | 0.7784 | 1855 |
| bge-m3 | US | 0.0462 | 0.5688 | 0.7558 | 0.8095 | 0.8628 | 2252 |
| multilingual-e5-large | US | 0.0369 | 0.5249 | 0.7114 | 0.7798 | 0.8393 | 2252 |
| multilingual-e5-small | US | 0.0320 | 0.5067 | 0.6918 | 0.7540 | 0.8215 | 2252 |
| multilingual-e5-base | US | 0.0391 | 0.5013 | 0.6856 | 0.7638 | 0.8344 | 2252 |

**Speed on this GPU (fp16, max 64 tokens) and projected time for all 24.2M records**

| model | docs_per_s | load_s | full_embed_min |
|---|---|---|---|
| multilingual-e5-small | 15548 | 4.4000 | 26.0000 |
| multilingual-e5-base | 9636 | 4.3000 | 41.9000 |
| multilingual-e5-large | 4105 | 6.7000 | 98.4000 |
| bge-m3 | 4432 | 7.0000 | 91.1000 |