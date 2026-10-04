"""Stage `embed`: multilingual sentence embeddings of 'name | address' on the GPU (fp16).

The model is chosen by the retrieval ablation (ablate_retrieval.py); all candidates are MIT or
Apache-2.0. Raw text is embedded (the model handles native scripts itself). Output:
work/<split>/emb_{s1,oth}.npy — float16, L2-normalized, row order = the prepared parquet.
"""
import time

import numpy as np
import polars as pl

from .config import P, log, record_time, work_dir

INSTRUCT = "Instruct: Given a business name and address, retrieve records of the same business\nQuery: "
PREFIX = {  # model -> (S1 prefix, S2/S3 prefix, extra kwargs)
    "intfloat/multilingual-e5-small": ("query: ", "query: ", {}),
    "intfloat/multilingual-e5-base": ("query: ", "query: ", {}),
    "intfloat/multilingual-e5-large": ("query: ", "query: ", {}),
    "BAAI/bge-m3": ("", "", {}),
    "Alibaba-NLP/gte-multilingual-base": ("", "", {"trust_remote_code": True}),
    "Qwen/Qwen3-Embedding-0.6B": (INSTRUCT, "", {}),
    # A0 round 2 (all MIT / Apache-2.0)
    "intfloat/multilingual-e5-large-instruct": (INSTRUCT, INSTRUCT, {}),       # symmetric task: both sides
    "Snowflake/snowflake-arctic-embed-l-v2.0": ("query: ", "", {}),
    "Snowflake/snowflake-arctic-embed-m-v2.0": ("query: ", "", {"trust_remote_code": True}),
    "ibm-granite/granite-embedding-311m-multilingual-r2": ("", "", {}),
    "ibm-granite/granite-embedding-97m-multilingual-r2": ("", "", {}),
    "AkshitaS/bhasha-embed-v0": ("", "", {}),
    "ai4bharat/IndicBERTv2-SS": ("", "", {}),      # plain transformers model: mean pooling added on load
    # A0 round 4: MIT (KaLM, Qwen2-0.5B based) and Apache-2.0 (F2LLM-v2, Sept 2026); decoder models -> bf16
    "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5": (INSTRUCT, INSTRUCT, {}),
    "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v2": (INSTRUCT, INSTRUCT, {}),
    "codefuse-ai/F2LLM-v2-0.6B": (INSTRUCT, "", {}),
}
BF16 = {"Qwen/Qwen3-Embedding-0.6B", "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
        "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v2", "codefuse-ai/F2LLM-v2-0.6B"}      # fp16 overflowed (CUDA device-side assert) in run smoke-v3b
# GTE-architecture remote code (gte-multilingual, arctic-m): plain attention instead of xformers
LOAD_KW = {m: {"config_kwargs": {"use_memory_efficient_attention": False, "unpad_inputs": False}}
           for m in ("Alibaba-NLP/gte-multilingual-base", "Snowflake/snowflake-arctic-embed-m-v2.0")}


def _patch_extended_mask():
    """transformers 5 dropped PreTrainedModel.get_extended_attention_mask, still used by GTE remote code."""
    import torch
    from transformers import PreTrainedModel
    if hasattr(PreTrainedModel, "get_extended_attention_mask"):
        return

    def get_extended_attention_mask(self, attention_mask, input_shape=None, device=None, dtype=None):
        dtype = dtype or self.dtype
        ext = attention_mask[:, None, None, :] if attention_mask.dim() == 2 else attention_mask[:, None, :, :]
        return (1.0 - ext.to(dtype)) * torch.finfo(dtype).min
    PreTrainedModel.get_extended_attention_mask = get_extended_attention_mask


def load_model(name: str):
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import os
    _patch_extended_mask()
    model = SentenceTransformer(name, device=dev, **PREFIX.get(name, ("", "", {}))[2], **LOAD_KW.get(name, {}))
    dtype = os.environ.get("BER_EMB_DTYPE") or ("bf16" if name in BF16 else "fp16")
    if dev == "cuda" and dtype != "fp32":
        model.bfloat16() if dtype == "bf16" else model.half()
    model.max_seq_length = P.emb_max_len
    return model


def suffix(i: int) -> str:
    """View i's file / column suffix: '' for the first embedder, '2' for the second, ..."""
    return "" if i == 0 else str(i + 1)


def run(split: str) -> None:
    out = work_dir(split)
    t0, t_load, n_done = time.time(), 0.0, 0
    for i, name in enumerate(P.emb_models):
        sfx = suffix(i)
        if (out / f"emb{sfx}_oth.npy").exists():
            log(f"embed[{split}] {name}: cached")
            continue
        t = time.time()
        model = load_model(name)
        t_load += time.time() - t
        for part, prefix in zip(("s1", "oth"), PREFIX.get(name, ("", "", {}))[:2]):
            df = pl.read_parquet(out / f"{part}.parquet", columns=["name", "addr"])
            texts = df.select(pl.lit(prefix) + pl.col("name") + " | " + pl.col("addr")).to_series().to_list()
            embs = []
            for j in range(0, len(texts), 500_000):
                e = model.encode(texts[j:j + 500_000], batch_size=P.emb_batch, normalize_embeddings=True,
                                 convert_to_tensor=True, show_progress_bar=False)
                embs.append(e.half().cpu().numpy())
                log(f"embed[{split}/{part}] {name}: {min(j + 500_000, len(texts)):,}/{len(texts):,}")
            np.save(out / f"emb{sfx}_{part}.npy", np.concatenate(embs))
            n_done += len(texts)
        del model
    secs = time.time() - t0
    if n_done:
        record_time(f"embed.{split}", secs, records=n_done, load_s=round(t_load, 1),
                    docs_per_s=round(n_done / max(secs - t_load, 1e-6), 1), models=len(P.emb_models))
        log(f"embed[{split}]: {n_done:,} texts x models, {n_done / max(secs - t_load, 1e-6):,.0f}/s")
