
# A0r3 run `ablate_retrieval_qwen_bf16` (Qwen/Qwen3-Embedding-0.6B) — exit code 0

# v3 ablation A0 — embedders and approximate search

40,000 S1 queries; corpus = all their true matches + 20% of other S2/S3 records of the same country. Recall@k = share of true pairs in the query's top-k.


## Step 1 — embedders (exact search)


**all**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | India | 0.2473 | 0.7668 | 0.8336 | 0.8587 | 0.8843 | 55327 |
| Qwen3-Embedding-0.6B | US | 0.2590 | 0.8792 | 0.9424 | 0.9537 | 0.9632 | 82857 |

**no address**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | India | 0.0106 | 0.2133 | 0.3370 | 0.4189 | 0.5214 | 2175 |
| Qwen3-Embedding-0.6B | US | 0.0059 | 0.2169 | 0.3649 | 0.4435 | 0.5333 | 3919 |

**trade name (name sim < 50)**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | India | 0.0447 | 0.3240 | 0.4000 | 0.4372 | 0.4846 | 1855 |
| Qwen3-Embedding-0.6B | US | 0.0506 | 0.3899 | 0.5058 | 0.5555 | 0.6146 | 2252 |

**native script**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | India | 0.0895 | 0.4295 | 0.5176 | 0.5751 | 0.6400 | 9855 |

**Speed on this GPU (fp16, max 64 tokens) and projected time for all 24.2M records**

| model | docs_per_s | load_s | full_embed_min |
|---|---|---|---|
| Qwen3-Embedding-0.6B | 391 | 4.8000 | 1032.9000 |

**Best embedder by mean R@10: `Qwen/Qwen3-Embedding-0.6B`**

| model | R@10 | R@20 |
|---|---|---|
| Qwen3-Embedding-0.6B | 0.8880 | 0.9062 |

# A0r3 run `ablate_retrieval_arctic_m` (Snowflake/snowflake-arctic-embed-m-v2.0) — exit code 1

(no report written)


# A0r3 run `ablate_retrieval_gte` (Alibaba-NLP/gte-multilingual-base) — exit code 1

(no report written)


# A0r3 run `ablate_retrieval_e5small_ivf` (intfloat/multilingual-e5-small) — exit code 0

# v3 ablation A0 — embedders and approximate search

40,000 S1 queries; corpus = all their true matches + 20% of other S2/S3 records of the same country. Recall@k = share of true pairs in the query's top-k.


## Step 1 — embedders (exact search)


**all**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| multilingual-e5-small | India | 0.2629 | 0.8791 | 0.9429 | 0.9559 | 0.9667 | 55327 |
| multilingual-e5-small | US | 0.2665 | 0.9174 | 0.9774 | 0.9832 | 0.9881 | 82857 |

**no address**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| multilingual-e5-small | India | 0.0032 | 0.2046 | 0.3402 | 0.4290 | 0.5223 | 2175 |
| multilingual-e5-small | US | 0.0117 | 0.5014 | 0.7356 | 0.8033 | 0.8599 | 3919 |

**native script**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| multilingual-e5-small | India | 0.1103 | 0.7819 | 0.8907 | 0.9248 | 0.9528 | 9855 |

**trade name (name sim < 50)**

| model | country | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|
| multilingual-e5-small | India | 0.0943 | 0.6981 | 0.8404 | 0.8620 | 0.8830 | 1855 |
| multilingual-e5-small | US | 0.0320 | 0.5067 | 0.6918 | 0.7540 | 0.8215 | 2252 |

**Speed on this GPU (fp16, max 64 tokens) and projected time for all 24.2M records**

| model | docs_per_s | load_s | full_embed_min |
|---|---|---|---|
| multilingual-e5-small | 9129 | 4.9000 | 44.2000 |

**Best embedder by mean R@10: `intfloat/multilingual-e5-small`**

| model | R@10 | R@20 |
|---|---|---|
| multilingual-e5-small | 0.9601 | 0.9695 |

## Step 2 — approximate search on `intfloat/multilingual-e5-small` (corpus per country ~1-2M)

vs_exact@25 = share of the exact top-25 that the approximate top-25 also returns.

| country | method | seconds | vs_exact@25 | R@1 | R@5 | R@10 | R@20 | R@50 | pairs |
|---|---|---|---|---|---|---|---|---|---|
| India | exact (GPU brute force) | nan | 1.0000 | 0.2629 | 0.8791 | 0.9429 | 0.9559 | 0.9667 | 55327 |
| India | GPU IVF nprobe=64 | 1.5000 | 0.7514 | 0.2518 | 0.7303 | 0.7609 | 0.7645 | 0.7669 | 55327 |
| India | GPU IVF nprobe=256 | 1.5000 | 0.8858 | 0.2592 | 0.8183 | 0.8647 | 0.8711 | 0.8757 | 55327 |
| India | GPU IVF nprobe=1024 | 3.0000 | 0.9569 | 0.2620 | 0.8629 | 0.9202 | 0.9295 | 0.9370 | 55327 |
| India | GPU IVF nprobe=64 centred | 2.3000 | 0.7671 | 0.2540 | 0.7501 | 0.7843 | 0.7887 | 0.7926 | 55327 |
| India | GPU IVF nprobe=256 centred | 2.3000 | 0.9000 | 0.2610 | 0.8392 | 0.8901 | 0.8984 | 0.9053 | 55327 |
| India | GPU IVF nprobe=1024 centred | 3.6000 | 0.9632 | 0.2627 | 0.8725 | 0.9335 | 0.9453 | 0.9550 | 55327 |
| India | FAISS IVF-Flat CPU nprobe=64 | 7.8000 | 0.7513 | 0.2519 | 0.7306 | 0.7611 | 0.7647 | 0.7671 | 55327 |
| US | exact (GPU brute force) | nan | 1.0000 | 0.2665 | 0.9174 | 0.9774 | 0.9832 | 0.9881 | 82857 |
| US | GPU IVF nprobe=64 | 1.9000 | 0.7647 | 0.2655 | 0.8788 | 0.9262 | 0.9293 | 0.9320 | 82857 |
| US | GPU IVF nprobe=256 | 2.0000 | 0.8414 | 0.2659 | 0.8931 | 0.9437 | 0.9472 | 0.9503 | 82857 |
| US | GPU IVF nprobe=1024 | 4.3000 | 0.9221 | 0.2663 | 0.9050 | 0.9595 | 0.9635 | 0.9671 | 82857 |
| US | GPU IVF nprobe=64 centred | 3.1000 | 0.7750 | 0.2654 | 0.8812 | 0.9294 | 0.9327 | 0.9356 | 82857 |
| US | GPU IVF nprobe=256 centred | 3.1000 | 0.8512 | 0.2660 | 0.8946 | 0.9459 | 0.9496 | 0.9528 | 82857 |
| US | GPU IVF nprobe=1024 centred | 5.2000 | 0.9284 | 0.2663 | 0.9052 | 0.9599 | 0.9640 | 0.9677 | 82857 |
| US | FAISS IVF-Flat CPU nprobe=64 | 14.5000 | 0.7638 | 0.2654 | 0.8789 | 0.9264 | 0.9294 | 0.9321 | 82857 |