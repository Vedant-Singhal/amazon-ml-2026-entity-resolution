
# A0r3 run `ablate_retrieval_kalm15` (HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5) — exit code 0

# v3 ablation A0 — embedders and approximate search

40,000 S1 queries; corpus = all their true matches + 20% of other S2/S3 records of the same country. Recall@k = share of true pairs in the query's top-k.


## Step 1 — embedders (exact search)


**all**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| KaLM-embedding-multilingual-mini-instruct-v1.5 | India | 0.2401 | 0.6982 | 0.7494 | 0.7679 | 0.7852 | 55327 |
| KaLM-embedding-multilingual-mini-instruct-v1.5 | US | 0.2594 | 0.8709 | 0.9358 | 0.9498 | 0.9623 | 82857 |

**no address**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| KaLM-embedding-multilingual-mini-instruct-v1.5 | India | 0.0152 | 0.2345 | 0.3246 | 0.3839 | 0.4469 | 2175 |
| KaLM-embedding-multilingual-mini-instruct-v1.5 | US | 0.0059 | 0.1865 | 0.3075 | 0.3873 | 0.4871 | 3919 |

**native script**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| KaLM-embedding-multilingual-mini-instruct-v1.5 | India | 0.0068 | 0.0421 | 0.0573 | 0.0695 | 0.0901 | 9855 |

**trade name (name sim < 50)**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| KaLM-embedding-multilingual-mini-instruct-v1.5 | India | 0.0954 | 0.5094 | 0.6167 | 0.6663 | 0.7197 | 1855 |
| KaLM-embedding-multilingual-mini-instruct-v1.5 | US | 0.0448 | 0.4845 | 0.6385 | 0.7114 | 0.7718 | 2252 |

**Speed on this GPU (fp16, max 64 tokens) and projected time for all 24.2M records**

| model | docs_per_s | load_s | full_embed_min |
|---|---|---|---|
| KaLM-embedding-multilingual-mini-instruct-v1.5 | 578 | 8.8000 | 698.1000 |

**Best embedder by mean R@10: `HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5`**

| model | R@10 | R@20 |
|---|---|---|
| KaLM-embedding-multilingual-mini-instruct-v1.5 | 0.8426 | 0.8589 |