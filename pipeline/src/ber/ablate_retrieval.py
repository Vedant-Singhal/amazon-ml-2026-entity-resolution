"""Ablation A0 for v3 retrieval: which embedder, and how good the approximate search is.

Step 1 — embedders: v2's multilingual-e5-small vs larger MIT / Apache-2.0 models. Recall@k of
    S1 -> S2/S3 search within country (exact), overall and on hard subsets (no address, native
    script, trade names), encoding speed on this GPU, and the projected time to embed all 24.2M
    train+test records. A model that fails to load or encode is reported and skipped; the report
    is rewritten after every model (a late failure no longer loses earlier results).
Step 2 — search: GPU IVF (block.knn) at several nprobe vs exact search on the best embedder:
    recall vs exact and time; FAISS IVF-Flat on the CPU as the reference.
Sample: N_Q S1 queries; corpus = every true match of those queries + POOL_FRAC of all other
S2/S3 records of the same country (lower density than the real pool -> recall is optimistic;
comparisons between models are fair). Reads v2's prepared records from S3.
Output: output/ablate_retrieval.md
"""
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import polars as pl
import torch
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer

from .block import faiss_knn, gpu_exact, gpu_ivf
from .config import P, log
from .data import load_gt_pairs
from .diagnose import _get, _table
from .embed import PREFIX, load_model

OUT = Path(os.environ.get("BER_OUT", "/opt/ml/checkpoints/output"))
N_Q = int(os.environ.get("ABL_QUERIES", "40000"))
POOL_FRAC = float(os.environ.get("ABL_POOL_FRAC", "0.2"))
FULL_RECORDS = 24_229_173            # train + test, S1 + S2/S3
KS = [1, 5, 10, 20, 50]
NATIVE = r"[^\p{Latin}\p{Common}\p{Inherited}]"
MODELS = [  # run order: the report is rewritten after each model; the riskiest model runs last
    "intfloat/multilingual-e5-small",                       # v2 baseline, MIT, 118M
    "intfloat/multilingual-e5-large",                       # MIT, 560M (round-1 best)
    "intfloat/multilingual-e5-large-instruct",              # MIT, 560M
    "Snowflake/snowflake-arctic-embed-l-v2.0",              # Apache-2.0, 568M
    "Snowflake/snowflake-arctic-embed-m-v2.0",              # Apache-2.0, 305M (remote code)
    "ibm-granite/granite-embedding-311m-multilingual-r2",   # Apache-2.0, 311M (Apr 2026)
    "ibm-granite/granite-embedding-97m-multilingual-r2",    # Apache-2.0, 97M (Apr 2026)
    "AkshitaS/bhasha-embed-v0",                             # Apache-2.0, 238M: Hindi + English + Romanized Hindi
    "ai4bharat/IndicBERTv2-SS",                             # MIT, ~278M: 23 Indic languages + English
]
LAST = ["Qwen/Qwen3-Embedding-0.6B"]                        # Apache-2.0; bf16 retry, after steps 2-3
if os.environ.get("ABL_MODELS"):                          # isolated runs: one process per model
    MODELS, LAST = os.environ["ABL_MODELS"].split(","), []
STEPS = set(os.environ.get("ABL_STEPS", "1,2,3").split(","))
TAG = os.environ.get("ABL_TAG", "")
# Round 1: e5-base 0.936 and bge-m3 0.942 mean R@10 < e5-small 0.960 <
# e5-large 0.963; gte-multilingual-base fails under transformers 5 (remote code).


def _texts(df: pl.DataFrame, prefix: str) -> list:
    return df.select(pl.lit(prefix) + pl.col("name") + " | " + pl.col("addr")).to_series().to_list()


def _encode(model, texts: list) -> tuple:
    t0 = time.time()
    e = model.encode(texts, batch_size=512, normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=False).astype(np.float16)
    return e, len(texts) / max(time.time() - t0, 1e-6)


def _recall(idx: np.ndarray, qpos: np.ndarray, cpos: np.ndarray, masks: dict) -> dict:
    hit = idx[qpos] == cpos[:, None]
    rank = np.where(hit.any(1), hit.argmax(1) + 1, 10**6)
    out = {}
    for name, m in masks.items():
        if m.sum():
            out[name] = {f"R@{k}": float((rank[m] <= k).mean()) for k in KS} | {"pairs": int(m.sum())}
    return out


def _write(rep: list, res: list, speed: list, tail: list) -> None:
    lines = list(rep)
    if res:
        df = pl.DataFrame(res)
        lines.append("\n## Step 1 — embedders (exact search)\n")
        for sub in df["subset"].unique(maintain_order=True).to_list():
            lines.append(f"\n**{sub}**\n")
            lines.append(_table(df.filter(pl.col("subset") == sub).drop("subset")
                                .sort("country", "R@10", descending=[False, True])))
    if speed:
        lines.append("\n**Speed on this GPU (fp16, max 64 tokens) and projected time for all 24.2M records**\n")
        lines.append(_table(pl.DataFrame(speed)))
    lines += tail
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ablate_retrieval{TAG}.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    rep = ["# v3 ablation A0 — embedders and approximate search\n",
           f"{N_Q:,} S1 queries; corpus = all their true matches + {POOL_FRAC:.0%} of other S2/S3 "
           "records of the same country. Recall@k = share of true pairs in the query's top-k.\n"]
    s1 = pl.read_parquet(_get("v1b", "train/s1.parquet"), columns=["iid", "country", "name", "addr", "core"])
    oth = pl.read_parquet(_get("v1b", "train/oth.parquet"), columns=["iid", "country", "name", "addr", "core", "has_addr"])
    pairs, _ = load_gt_pairs()
    q = s1.sample(n=min(N_Q, len(s1)), seed=P.seed)
    gp = pairs.join(q.select(a="iid"), on="a")
    keep = pl.concat([gp.select(iid="b"), oth.filter(pl.col("iid").hash(P.seed) % 1000 < POOL_FRAC * 1000).select("iid")]).unique()
    corp = oth.join(keep, on="iid")
    log(f"A0: {len(q):,} queries, {len(corp):,} corpus records, {len(gp):,} true pairs")
    t = (gp.join(q.select(a="iid", core_a="core"), on="a")
         .join(corp.select(b="iid", core_b="core", name_b="name", has_b="has_addr"), on="b"))
    nsim = process.cpdist(t["core_a"].to_list(), t["core_b"].to_list(), scorer=fuzz.token_set_ratio, workers=P.workers)
    masks_base = {"all": np.ones(len(t), bool), "no address": ~t["has_b"].to_numpy(),
                  "native script": t["name_b"].str.contains(NATIVE).to_numpy(),
                  "trade name (name sim < 50)": nsim < 50}

    res, speed, embs, tail = [], [], {}, []

    def one(name):
        qp, dp, kw = PREFIX.get(name, ("", "", {}))
        short = name.split("/")[-1]
        try:
            t0 = time.time()
            model = load_model(name)
            load_s = time.time() - t0
            rates = []
            for (country,), qc in q.group_by("country"):
                cc = corp.filter(pl.col("country") == country)
                qe, _ = _encode(model, _texts(qc, qp))
                ce, r = _encode(model, _texts(cc, dp))
                rates.append(r)
                idx = gpu_exact(ce, qe, max(KS))
                qmap = dict(zip(qc["iid"].to_list(), range(len(qc))))
                cmap = dict(zip(cc["iid"].to_list(), range(len(cc))))
                tc = t.with_row_index("r").filter(pl.col("a").is_in(qc["iid"].implode()))
                rows = tc["r"].to_numpy()
                qpos = np.array([qmap[x] for x in tc["a"].to_list()])
                cpos = np.array([cmap[x] for x in tc["b"].to_list()])
                for sub, r_ in _recall(idx, qpos, cpos, {k: v[rows] for k, v in masks_base.items()}).items():
                    res.append({"model": short, "country": country, "subset": sub, **r_})
                embs[(name, country)] = (qe, ce, qpos, cpos)
            dps = float(np.mean(rates))
            speed.append({"model": short, "docs_per_s": round(dps), "load_s": round(load_s, 1),
                          "full_embed_min": round(FULL_RECORDS / dps / 60, 1)})
            log(f"A0: {name} done ({dps:,.0f} docs/s) " + str([r for r in res if r["model"] == short and r["subset"] == "all"]))
        except Exception as e:  # noqa: BLE001 - a failing model is reported, not fatal
            log(f"A0: {name} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            rep.append(f"- `{name}`: failed ({type(e).__name__}: {str(e)[:160]})")
            for key in [k for k in embs if k[0] == name]:
                del embs[key]
        finally:
            model = None
            torch.cuda.empty_cache()
        _write(rep, res, speed, tail)

    for name in MODELS:
        one(name)
    if not res:
        return
    agg = (pl.DataFrame(res).filter(pl.col("subset") == "all").group_by("model")
           .agg(pl.col("R@10").mean(), pl.col("R@20").mean()).sort("R@10", descending=True))
    top = agg["model"][0]
    top_full = next(n for n in MODELS if n.split("/")[-1] == top)
    tail.append(f"\n**Best embedder by mean R@10: `{top_full}`**\n")
    tail.append(_table(agg))

    if "2" not in STEPS:
        _write(rep, res, speed, tail)
        return
    # ---- Step 2: approximate search on the best embedder -------------------------------------
    rows = []
    for (name, country), (qe, ce, qpos, cpos) in embs.items():
        if name != top_full:
            continue
        k = max(KS)
        exact = gpu_exact(ce, qe, k)
        allm = {"all": np.ones(len(qpos), bool)}
        rows.append({"country": country, "method": "exact (GPU brute force)", "seconds": math.nan,
                     "vs_exact@25": 1.0, **_recall(exact, qpos, cpos, allm)["all"]})
        for npb, ctr in ((64, False), (256, False), (1024, False), (64, True), (256, True), (1024, True)):
            t0 = time.time()
            idx = gpu_ivf(ce, qe, k, npb, center=ctr)
            secs = time.time() - t0
            overlap = float(np.mean([len(set(a[:25]) & set(b[:25])) / 25 for a, b in zip(idx, exact)]))
            label = f"GPU IVF nprobe={npb}" + (" centred" if ctr else "")
            rows.append({"country": country, "method": label, "seconds": round(secs, 1),
                         "vs_exact@25": overlap, **_recall(idx, qpos, cpos, allm)["all"]})
        t0 = time.time()
        idx = faiss_knn(ce, qe, k)
        rows.append({"country": country, "method": f"FAISS IVF-Flat CPU nprobe={P.nprobe}",
                     "seconds": round(time.time() - t0, 1),
                     "vs_exact@25": float(np.mean([len(set(a[:25]) & set(b[:25])) / 25 for a, b in zip(idx, exact)])),
                     **_recall(idx, qpos, cpos, allm)["all"]})
        log(f"A0 step 2: {country} done")
    tail.append(f"\n## Step 2 — approximate search on `{top_full}` (corpus per country ~1-2M)\n")
    tail.append("vs_exact@25 = share of the exact top-25 that the approximate top-25 also returns.\n")
    tail.append(_table(pl.DataFrame(rows).sort("country")))
    _write(rep, res, speed, tail)
    if "3" not in STEPS:
        log(f"A0: wrote ablate_retrieval{TAG}.md")
        return
    # ---- Step 3: GPU IVF speed at full size (largest country: 6.19M records x 1.32M S1) ----------
    rows = []
    d = embs[next(iter(embs))][1].shape[1]
    embs.clear()
    for nb, nq, k in ((6_186_873, 1_323_633, 25), (1_323_633, 6_186_873, 5)):
        g = np.random.default_rng(0)
        xb = g.standard_normal((nb, d), dtype=np.float32)
        xb = (xb / np.linalg.norm(xb, axis=1, keepdims=True)).astype(np.float16)
        xq = g.standard_normal((nq, d), dtype=np.float32)
        xq = (xq / np.linalg.norm(xq, axis=1, keepdims=True)).astype(np.float16)
        t0 = time.time()
        gpu_ivf(xb, xq, k, P.nprobe)
        rows.append({"base": nb, "queries": nq, "dim": d, "k": k, "nprobe": P.nprobe,
                     "seconds": round(time.time() - t0, 1)})
        del xb, xq
        log(f"A0 step 3: {rows[-1]}")
    tail.append("\n## Step 3 — GPU IVF speed at full size (random vectors)\n")
    tail.append(_table(pl.DataFrame(rows)))
    _write(rep, res, speed, tail)
    for name in LAST:            # may break the CUDA context: everything above is already saved
        one(name)
    log(f"A0: wrote {OUT / 'ablate_retrieval.md'}")
