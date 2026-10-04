"""Stage `retriever` (v4, optional — BER_V4_RETRIEVER=1): a search route from multilingual-e5-small fine-tuned
on our train pairs (supervised contrastive, SC-Block — Brinkmann et al., ESWC'24), with romanized copies of
native-script records as extra positives (Lost in Transliteration, SIGIR'25).

  - Loss: in-batch negatives (InfoNCE, scale 20) on (S1 text, matching S2/S3 text); prefix "query: " both sides.
  - Leak-free on train: cross-fitted by S1 half (hash(a, seed + 11) % 2). Model h is trained on the pairs of
    half h and searches for the S1 of the other half. Test uses a model trained on all train pairs.
  - New pairs are merged into work/<split>/pool.parquet with their other route ranks / cosines; the stages after
    the pool (stage1, features, train) must then be re-run (run_v4.py does this when the flag is set).
Output: work/<split>/route_ft.parquet (a, b, ft_rank_fwd, ft_rank_rev)
"""
import math
import time

import numpy as np
import polars as pl

from .block import DENSE, KEY_ROUTES, RANKS, gpu_exact, pair_dot
from .config import P, log, record_time, work_dir
from .data import load_gt_pairs

BASE = "intfloat/multilingual-e5-small"
K_FWD, K_REV = 10, 3
CAP = 1_500_000        # training pairs per model
BATCH = 256


def _texts(split):
    wd = work_dir(split)
    raw = lambda: pl.lit("query: ") + pl.col("name") + pl.lit(" | ") + pl.col("addr")
    rom = lambda: pl.lit("query: ") + pl.col("name_n") + pl.lit(" | ") + pl.col("addr_n")
    s1 = pl.read_parquet(wd / "s1.parquet", columns=["iid", "country", "name", "addr"]).select("iid", "country", t=raw())
    oth = pl.read_parquet(wd / "oth.parquet", columns=["iid", "country", "name", "addr", "name_n", "addr_n", "native"])
    oth = oth.select("iid", "country", "native", t=raw(), t_rom=rom())
    return s1, oth


def _fit(q: list, d: list):
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BASE, device="cuda")
    model.max_seq_length = P.emb_max_len
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    perm = np.random.default_rng(P.seed).permutation(len(q))
    steps = math.ceil(len(q) / BATCH)
    t0 = time.time()
    for i in range(steps):
        idx = perm[i * BATCH:(i + 1) * BATCH]
        dev = lambda f: {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in f.items()}   # ST 6 adds non-tensor keys
        fq = dev(model.tokenize([q[j] for j in idx]))
        fd = dev(model.tokenize([d[j] for j in idx]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            eq = F.normalize(model(fq)["sentence_embedding"].float(), dim=1)
            ed = F.normalize(model(fd)["sentence_embedding"].float(), dim=1)
        logits = 20.0 * eq @ ed.T
        lab = torch.arange(len(idx), device="cuda")
        loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (i + 1) % 1000 == 0 or i + 1 == steps:
            log(f"retriever: step {i + 1}/{steps} loss {loss.item():.4f} ({(i + 1) * BATCH / (time.time() - t0):,.0f} pairs/s)")
    model.eval()
    model.half()
    return model


def _enc(model, texts: list) -> np.ndarray:
    return model.encode(texts, batch_size=P.emb_batch, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False).astype(np.float16)


def _train_pairs(s1, oth, pairs, s1_ids):
    """(query, doc) texts: S1 -> its S2/S3 records, plus romanized copies of native-script records."""
    p = (pairs.filter(pl.col("a").is_in(s1_ids.implode()))
         .join(s1.select(a="iid", q="t"), on="a").join(oth.select(b="iid", d="t", dr="t_rom", native="native"), on="b"))
    extra = p.filter(pl.col("native")).select("q", d="dr")
    p = pl.concat([p.select("q", "d"), extra])
    if len(p) > CAP:
        p = p.sample(n=CAP, seed=P.seed)
    return p["q"].to_list(), p["d"].to_list()


def _search(model, s1q, oth):
    """Both directions within country with the model's embeddings. s1q: the S1 rows to search for."""
    E1, E2 = _enc(model, s1q["t"].to_list()), _enc(model, oth["t"].to_list())
    fwd, rev = [], []
    for country in s1q["country"].unique().to_list():
        i1 = np.flatnonzero((s1q["country"] == country).to_numpy())
        i2 = np.flatnonzero((oth["country"] == country).to_numpy())
        if not len(i1) or not len(i2):
            continue
        a_ids, b_ids = s1q["iid"].to_numpy()[i1], oth["iid"].to_numpy()[i2]
        nn = gpu_exact(E2[i2], E1[i1], K_FWD)
        fwd.append(pl.DataFrame({"a": np.repeat(a_ids, nn.shape[1]), "b": b_ids[nn.ravel()],
                                 "ft_rank_fwd": np.tile(np.arange(1, nn.shape[1] + 1, dtype=np.int16), len(i1))}))
        nn = gpu_exact(E1[i1], E2[i2], K_REV)
        rev.append(pl.DataFrame({"b": np.repeat(b_ids, nn.shape[1]), "a": a_ids[nn.ravel()],
                                 "ft_rank_rev": np.tile(np.arange(1, nn.shape[1] + 1, dtype=np.int16), len(i2))}))
    return pl.concat(fwd).join(pl.concat(rev), on=["a", "b"], how="full", coalesce=True)


def _merge(split: str, route: pl.DataFrame) -> None:
    """Add the route's pairs to the pool; new pairs get the other routes' defaults and all cosines."""
    wd = work_dir(split)
    pool = pl.read_parquet(wd / "pool.parquet")
    new = route.join(pool.select("a", "b"), on=["a", "b"], how="anti")
    s1 = pl.read_parquet(wd / "s1.parquet", columns=["iid"])
    oth = pl.read_parquet(wd / "oth.parquet", columns=["iid"])
    ids1, ids2 = s1["iid"].to_numpy(), oth["iid"].to_numpy()
    new = new.with_columns(*[pl.lit(k + 1, pl.Int16).alias(c) for c, k in RANKS.items()],
                           *[pl.lit(0, pl.Int8).alias(c) for c in KEY_ROUTES],
                           sp_score=pl.lit(0.0, pl.Float32), sp_n=pl.lit(0, pl.Int16))
    for v in DENSE:
        new = pair_dot(new, ids1, ids2, np.load(wd / f"emb{v}_s1.npy"), np.load(wd / f"emb{v}_oth.npy"), f"cos{v}")
    for view in ("n3", "a3"):
        new = pair_dot(new, ids1, ids2, np.load(wd / f"sk_{view}_s1.npy"), np.load(wd / f"sk_{view}_oth.npy"), f"{view}cos")
    old = pool.join(route, on=["a", "b"], how="left")
    pool = pl.concat([old, new], how="diagonal_relaxed").with_columns(
        pl.col("ft_rank_fwd").fill_null(K_FWD + 1).cast(pl.Int16), pl.col("ft_rank_rev").fill_null(K_REV + 1).cast(pl.Int16))
    pool.write_parquet(wd / "pool.parquet")
    log(f"retriever[{split}]: pool {len(old):,} -> {len(pool):,} pairs (+{len(new):,} found only by the fine-tuned route)")


def routes() -> None:
    """Train the cross-fitted models and write route_ft for both splits (needs only prepared records)."""
    import torch
    if all((work_dir(sp) / "route_ft.parquet").exists() for sp in ("train", "test")):
        log("retriever: routes cached")
        return
    pairs, _ = load_gt_pairs()
    t0 = time.time()
    s1, oth = _texts("train")
    half = (s1["iid"].hash(P.seed + 11) % 2).to_numpy()
    parts = []
    for h in (0, 1):
        q, d = _train_pairs(s1, oth, pairs, s1.filter(pl.Series(half == h))["iid"])
        model = _fit(q, d)
        parts.append(_search(model, s1.filter(pl.Series(half != h)), oth))
        del model
        torch.cuda.empty_cache()
    route = pl.concat(parts)
    route.write_parquet(work_dir("train") / "route_ft.parquet")
    hit = pairs.join(s1.select(a="iid"), on="a").join(route.select("a", "b"), on=["a", "b"]).height
    n = pairs.join(s1.select(a="iid"), on="a").height
    log(f"retriever: train route recall {hit / max(n, 1):.4f} ({len(route) / len(s1):.1f} pairs/S1)")
    q, d = _train_pairs(s1, oth, pairs, s1["iid"])
    model = _fit(q, d)
    ts1, toth = _texts("test")
    _search(model, ts1, toth).write_parquet(work_dir("test") / "route_ft.parquet")
    record_time("retriever.routes", time.time() - t0)


def needs_merge() -> bool:
    return any((work_dir(sp) / "route_ft.parquet").exists()
               and "ft_rank_fwd" not in pl.read_parquet_schema(work_dir(sp) / "pool.parquet")
               for sp in ("train", "test"))


def merge() -> None:
    for split in ("train", "test"):
        if "ft_rank_fwd" not in pl.read_parquet_schema(work_dir(split) / "pool.parquet"):
            _merge(split, pl.read_parquet(work_dir(split) / "route_ft.parquet"))
