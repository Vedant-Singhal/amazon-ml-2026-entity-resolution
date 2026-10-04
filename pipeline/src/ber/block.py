"""Stage `block`: the candidate POOL, always within country. Every route is built to scale
(approximate nearest neighbours, capped hash joins); the stage-1 filter then cuts the pool to a
small shortlist per S1 (stage1.py).

  dense      e5/gte/... embeddings of 'name | address', FAISS IVF: S1 -> top-25, record -> top-5 S1
  name c3    char-3-gram TF-IDF of the core name (count sketch), FAISS: S1 -> top-10, record -> top-3
             S1 — the name-only route for records without address (E1)
  addr c3    same on the address core, number-compatible pairs only — the trade-name route (E3)
  sparse     IDF-weighted shared rare keys (name words, address words, numbers) top-20 (v2 route)
  keys       exact core / compact / sorted-core / consonant-skeleton / address+number / number-set
             keys, only for keys shared by <= 10 S1 and <= 50 records (native script E4, E1)
Output work/<split>/pool.parquet: one row per (a, b) with route ranks/flags and three cosines.
On train, recall per route, unique contribution per route and union recall -> block_stats.json.
"""
import json
import math
import os
import time

import faiss
import numpy as np
import polars as pl

from .config import P, log, record_time, work_dir

ANN_MIN = int(os.environ.get("BER_ANN_MIN", "100000"))   # smaller bases: exact search
from .sketch import sketch

faiss.omp_set_num_threads(P.workers)
REC = ["iid", "country", "core", "compact", "sorted_core", "skel", "akey", "numset", "alpha", "nums",
       "first_num", "has_addr", "addr_core"]
KEY_ROUTES = {"k_core": ("core", 3), "k_compact": ("compact", 5), "k_sorted": ("sorted_core", 3),
              "k_skel": ("skel", 4), "k_akey": ("akey", 3), "k_numset": ("numset", 3)}
DENSE = ["" if i == 0 else str(i + 1) for i in range(len(P.emb_models))]   # view suffixes
RANKS = {**{f"rank{v}_fwd": P.dense_k_fwd for v in DENSE}, **{f"rank{v}_rev": P.dense_k_rev for v in DENSE},
         "n3_rank_fwd": P.c3_k_fwd,
         "n3_rank_rev": P.c3_k_rev, "a3_rank_fwd": P.c3_k_fwd, "a3_rank_rev": P.c3_k_rev,
         "sp_rank": P.sparse_k}


_KEEP = []   # coarse quantizers must outlive the python wrapper of their IVF index


def _kmeans_gpu(x: np.ndarray, k: int, iters: int = 12):
    """Spherical k-means on the GPU (IVF training); None without a GPU. ~100x faster than CPU."""
    import torch
    if not torch.cuda.is_available():
        return None
    X = torch.from_numpy(x).cuda().half()
    g = torch.Generator(device="cuda").manual_seed(P.seed)
    C = X[torch.randperm(len(X), device="cuda", generator=g)[:k]].clone()
    assign = torch.empty(len(X), dtype=torch.long, device="cuda")
    for _ in range(iters):
        for s in range(0, len(X), 65536):
            assign[s:s + 65536] = (X[s:s + 65536] @ C.T).argmax(1)
        S = torch.zeros((k, X.shape[1]), dtype=torch.float32, device="cuda").index_add_(0, assign, X.float())
        cnt = torch.bincount(assign, minlength=k)
        empty = cnt == 0
        if empty.any():   # re-seed empty clusters with random points
            S[empty] = X[torch.randint(len(X), (int(empty.sum()),), device="cuda", generator=g)].float()
        C = torch.nn.functional.normalize(S, dim=1).half()
    out = C.float().cpu().numpy()
    del X, C, S
    torch.cuda.empty_cache()
    return out


def _index(base: np.ndarray):
    n, d = base.shape
    nlist = int(4 * math.sqrt(n))
    if n < 100_000:                                     # small: exact search is cheaper than training
        idx = faiss.IndexFlatIP(d)
    elif P.index == "IVF,Flat" and (C := _kmeans_gpu(
            base[np.random.default_rng(P.seed).choice(n, min(n, 64 * nlist), replace=False)], nlist)) is not None:
        q = faiss.IndexFlatIP(d)
        q.add(C)
        _KEEP.append(q)
        idx = faiss.IndexIVFFlat(q, d, nlist, faiss.METRIC_INNER_PRODUCT)   # trained: q holds nlist centroids
        idx.nprobe = P.nprobe
    else:
        spec = P.index.replace("IVF,", f"IVF{int(4 * math.sqrt(n))},")
        idx = faiss.index_factory(d, spec, faiss.METRIC_INNER_PRODUCT)
        if not idx.is_trained:
            rng = np.random.default_rng(P.seed)
            idx.train(base[rng.choice(n, min(n, 500_000), replace=False)])
        ps = faiss.ParameterSpace()
        if "IVF" in spec:
            ps.set_index_parameter(idx, "nprobe", P.nprobe)
        if "HNSW" in spec:
            ps.set_index_parameter(idx, "efSearch", max(64, 2 * P.dense_k_fwd))
    idx.add(base)
    return idx


def faiss_knn(base: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """CPU fallback (no GPU): FAISS IVF-Flat / flat. Top-k row positions into base; -1 = none."""
    base = np.ascontiguousarray(base, dtype=np.float32)
    k = min(k, len(base))
    idx = _index(base)
    out = [idx.search(np.ascontiguousarray(query[s:s + 200_000], dtype=np.float32), k)[1]
           for s in range(0, len(query), 200_000)]
    del idx
    _KEEP.clear()
    return np.concatenate(out) if out else np.zeros((0, k), np.int64)


def gpu_exact(base: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Exact top-k on the GPU (small bases, and the reference for the ANN recall check)."""
    import torch
    B = torch.from_numpy(np.ascontiguousarray(base)).cuda().half()
    rows = max(64, min(8192, int(2e9 / max(len(base), 1))))     # ~4 GB fp16 score block per step
    out = []
    for s in range(0, len(query), rows):
        q = torch.from_numpy(np.ascontiguousarray(query[s:s + rows])).cuda().half()
        out.append((q @ B.T).topk(k, dim=1).indices.cpu().numpy())
    del B
    torch.cuda.empty_cache()
    return np.concatenate(out) if out else np.zeros((0, k), np.int64)


def gpu_ivf(base: np.ndarray, query: np.ndarray, k: int, nprobe: int, center: bool = False) -> np.ndarray:
    """Inverted-file (IVF) search on the GPU, the same method as FAISS IVF-Flat: the base is split
    into nlist = 4*sqrt(n) lists (spherical k-means on the GPU); every query scans the nprobe lists
    whose centroids are nearest to it. Computed list by list: all queries that probe list l are scored
    against its members in one matrix product and merged into their running top-k.
    Work ~ nprobe/nlist of brute force (1% for 6M records): sublinear in n.
    center=True: lists are built on mean-centred vectors (embeddings share a large common direction,
    which makes lists unbalanced); scores still use the original vectors."""
    import torch
    n = len(base)
    nlist = max(1, int(4 * math.sqrt(n)))
    npb = min(nprobe, nlist)
    rng = np.random.default_rng(P.seed)
    mu = torch.from_numpy(base.mean(0, dtype=np.float32)).cuda() if center else None

    def coarse(x):          # vectors used only to choose lists
        if mu is None:
            return x
        return torch.nn.functional.normalize(x.float() - mu, dim=1).half()
    samp = base[rng.choice(n, min(n, 64 * nlist), replace=False)]
    if center:
        samp = coarse(torch.from_numpy(np.ascontiguousarray(samp)).cuda()).cpu().numpy()
    C = torch.from_numpy(_kmeans_gpu(samp, nlist)).cuda().half()
    B = torch.from_numpy(np.ascontiguousarray(base)).cuda().half()
    lab = torch.cat([(coarse(B[s:s + 262144]) @ C.T).argmax(1) for s in range(0, n, 262144)])
    order = torch.argsort(lab)
    B = B[order]                                                  # each list is a contiguous slice
    cnt = torch.bincount(lab, minlength=nlist).cpu().numpy()
    st = np.cumsum(cnt) - cnt
    del lab
    Q = torch.from_numpy(np.ascontiguousarray(query)).cuda().half()
    out = np.empty((len(Q), k), dtype=np.int64)
    for q0 in range(0, len(Q), 1_000_000):
        Qc = Q[q0:q0 + 1_000_000]
        m = len(Qc)
        probes = torch.cat([(coarse(Qc[s:s + 131072]) @ C.T).topk(npb, dim=1).indices
                            for s in range(0, m, 131072)])
        lists = probes.flatten()
        qids = torch.arange(m, device="cuda").repeat_interleave(npb)
        srt = torch.argsort(lists)
        lists, qids = lists[srt], qids[srt]
        lc = torch.bincount(lists, minlength=nlist).cpu().numpy()
        ls = np.cumsum(lc) - lc
        del probes, lists, srt
        best_s = torch.full((m, k), -1e4, dtype=torch.float32, device="cuda")
        best_i = torch.full((m, k), -1, dtype=torch.long, device="cuda")
        for l in np.flatnonzero((lc > 0) & (cnt > 0)):
            qi = qids[ls[l]:ls[l] + lc[l]]
            v, j = (Qc[qi] @ B[st[l]:st[l] + cnt[l]].T).float().topk(min(k, int(cnt[l])), dim=1)
            v2, j2 = torch.cat([best_s[qi], v], 1).topk(k, dim=1)
            best_s[qi] = v2
            best_i[qi] = torch.cat([best_i[qi], j + int(st[l])], 1).gather(1, j2)
        res = torch.where(best_i >= 0, order[best_i.clamp(min=0)], torch.full_like(best_i, -1))
        out[q0:q0 + m] = res.cpu().numpy()
        del best_s, best_i, res, qids
    del B, Q, C, order
    torch.cuda.empty_cache()
    return out


def knn(base: np.ndarray, query: np.ndarray, k: int, nprobe: int = None) -> np.ndarray:
    """Top-k inner-product neighbours (row positions into base); -1 = none. GPU IVF for large bases,
    exact GPU search for small ones, FAISS on CPU without a GPU."""
    import torch
    k = min(k, len(base))
    if not torch.cuda.is_available():
        return faiss_knn(base, query, k)
    if len(base) >= ANN_MIN:
        return gpu_ivf(base, query, k, nprobe or P.nprobe)
    return gpu_exact(base, query, k)


ANN_CHECK = {}


def _ann_check(base, query, nn, k, tag):
    """Recall of the approximate top-k against exact search on 5,000 sampled queries."""
    if len(base) < ANN_MIN or len(query) == 0:
        return
    import torch
    if not torch.cuda.is_available():
        return
    s = np.random.default_rng(P.seed).choice(len(query), min(5000, len(query)), replace=False)
    ex = gpu_exact(base, query[s], nn.shape[1])
    ANN_CHECK[tag] = float(np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(nn[s], ex)]))
    log(f"  ann recall@{nn.shape[1]} vs exact [{tag}]: {ANN_CHECK[tag]:.4f}")


def _vector_route(s1, oth, E1, E2, m1, m2, k_fwd, k_rev, fwd_name, rev_name):
    """ANN both ways within country on the rows where masks m1/m2 hold."""
    fwd, rev = [], []
    c1, c2 = s1["country"].to_numpy(), oth["country"].to_numpy()
    a_all, b_all = s1["iid"].to_numpy(), oth["iid"].to_numpy()
    for country in np.unique(c1):
        i1 = np.flatnonzero((c1 == country) & m1)
        i2 = np.flatnonzero((c2 == country) & m2)
        if not len(i1) or not len(i2):
            continue
        for q, base, ids_q, ids_b, k, name, out in [(i1, i2, a_all, b_all, k_fwd, fwd_name, fwd),
                                                    (i2, i1, b_all, a_all, k_rev, rev_name, rev)]:
            Eq, Eb = (E1, E2) if q is i1 else (E2, E1)
            nn = knn(Eb[base], Eq[q], k)
            _ann_check(Eb[base], Eq[q], nn, k, f"{name}/{country}")
            kk = nn.shape[1]
            qid = np.repeat(ids_q[q], kk)
            rank = np.tile(np.arange(1, kk + 1, dtype=np.int16), len(q))
            ok = nn.ravel() >= 0
            bid = ids_b[base][np.where(ok, nn.ravel(), 0)]
            cols = ("a", "b") if q is i1 else ("b", "a")
            out.append(pl.DataFrame({cols[0]: qid[ok], cols[1]: bid[ok], name: rank[ok]}))
        log(f"  {fwd_name}/{rev_name} {country}: {len(i1):,} x {len(i2):,}")
    empty = lambda name: pl.DataFrame(schema={"a": pl.Int64, "b": pl.Int64, name: pl.Int16})
    return (pl.concat(fwd) if fwd else empty(fwd_name)), (pl.concat(rev) if rev else empty(rev_name))


def _keys(df: pl.DataFrame) -> pl.DataFrame:
    h = lambda kind, t: pl.concat_str([pl.col("country"), pl.lit(kind), t], separator="|").hash(seed=7)
    n = (df.select("iid", "country", t=pl.col("core").str.split(" ")).explode("t")
         .filter(pl.col("t").str.len_chars() >= 2).select("iid", k=h("n", pl.col("t"))))
    a = (df.select("iid", "country", t="alpha").explode("t")
         .filter(pl.col("t").str.len_chars() >= 3).select("iid", k=h("a", pl.col("t"))))
    m = df.select("iid", "country", t="nums").explode("t").drop_nulls("t").select("iid", k=h("#", pl.col("t")))
    return pl.concat([n, a, m]).unique()


def _sparse(s1, oth):
    ko = _keys(oth)
    df = ko.group_by("k").len("df").filter(pl.col("df") <= P.key_df_cap)
    n_oth = oth.group_by("country").len("n")
    kc = ko.join(oth.select("iid", "country"), on="iid").select("k", "country").unique("k")
    df = df.join(kc, on="k").join(n_oth, on="country").select(
        "k", w=(pl.col("n") / pl.col("df")).log().cast(pl.Float32))
    ko = ko.join(df.select("k"), on="k")
    ks = _keys(s1).join(df, on="k")
    ids = s1["iid"].to_numpy()
    out = []
    for s in range(0, len(ids), 20_000):
        chunk = ks.filter(pl.col("iid").is_in(pl.Series(ids[s:s + 20_000])))
        j = (chunk.rename({"iid": "a"}).join(ko.rename({"iid": "b"}), on="k")
             .group_by("a", "b").agg(sp_score=pl.col("w").sum(), sp_n=pl.len().cast(pl.Int16)))
        j = j.with_columns(sp_rank=pl.col("sp_score").rank("ordinal", descending=True).over("a"))
        out.append(j.filter(pl.col("sp_rank") <= P.sparse_k).with_columns(pl.col("sp_rank").cast(pl.Int16)))
    return pl.concat(out)


def _key_route(s1, oth, col, minlen, name):
    ka = s1.filter(pl.col(col).str.len_chars() >= minlen).select(a="iid", c="country", k=col)
    kb = oth.filter(pl.col(col).str.len_chars() >= minlen).select(b="iid", c="country", k=col)
    okA = ka.group_by("c", "k").len("n").filter(pl.col("n") <= P.key_cap_s1).select("c", "k")
    okB = kb.group_by("c", "k").len("n").filter(pl.col("n") <= P.key_cap_oth).select("c", "k")
    j = ka.join(okA, on=["c", "k"]).join(kb.join(okB, on=["c", "k"]), on=["c", "k"])
    return j.select("a", "b").unique().with_columns(pl.lit(1, pl.Int8).alias(name))


def pair_dot(pairs: pl.DataFrame, ids1, ids2, E1, E2, name: str) -> pl.DataFrame:
    """Inner product of E1[a] and E2[b] for every pair (GPU when available)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = (pairs.join(pl.DataFrame({"a": ids1, "_ia": np.arange(len(ids1))}), on="a")
         .join(pl.DataFrame({"b": ids2, "_ib": np.arange(len(ids2))}), on="b"))
    ia, ib = c["_ia"].to_numpy(), c["_ib"].to_numpy()
    need = (E1.nbytes + E2.nbytes) * 1.2
    on_gpu = dev == "cuda" and torch.cuda.mem_get_info()[0] > need
    if on_gpu:
        T1, T2 = torch.from_numpy(E1).to(dev), torch.from_numpy(E2).to(dev)
    out = np.empty(len(c), dtype=np.float32)
    step = max(65_536, int(2.5e8 / E1.shape[1]))   # 1024-d: 244k pairs (~1 GB fp32); 256-d: 977k
    for s in range(0, len(c), step):
        if on_gpu:
            x = T1[torch.from_numpy(ia[s:s + step]).to(dev)].float()
            y = T2[torch.from_numpy(ib[s:s + step]).to(dev)].float()
        else:
            x = torch.from_numpy(E1[ia[s:s + step]]).to(dev).float()
            y = torch.from_numpy(E2[ib[s:s + step]]).to(dev).float()
        out[s:s + step] = (x * y).sum(1).cpu().numpy()
    if on_gpu:
        del T1, T2
        torch.cuda.empty_cache()
    return c.drop("_ia", "_ib").with_columns(pl.Series(name, out))


def run(split: str) -> None:
    out = work_dir(split)
    if (out / "pool.parquet").exists():
        log(f"block[{split}]: cached")
        return
    t0 = time.time()
    s1 = pl.read_parquet(out / "s1.parquet", columns=REC)
    oth = pl.read_parquet(out / "oth.parquet", columns=REC)
    ids1, ids2 = s1["iid"].to_numpy(), oth["iid"].to_numpy()
    all1, all2 = np.ones(len(s1), bool), np.ones(len(oth), bool)
    tm = {}

    pre = out / "pool_routes.parquet"
    sk_files = {v: (out / f"sk_{v}_s1.npy", out / f"sk_{v}_oth.npy") for v in ("n3", "a3")}
    if pre.exists() and all(a.exists() and b.exists() for a, b in sk_files.values()):
        pool = pl.read_parquet(pre)
        views = {v: (np.load(a), np.load(b)) for v, (a, b) in sk_files.items()}
        log(f"block[{split}]: routes cached ({len(pool):,} pairs), computing cosines")
    else:
        pool, views = _routes(split, s1, oth, all1, all2, tm)
        pool.write_parquet(pre)
        for v, (a, b) in sk_files.items():
            np.save(a, views[v][0])
            np.save(b, views[v][1])
    t = time.time()
    for v in DENSE:
        E1, E2 = np.load(out / f"emb{v}_s1.npy"), np.load(out / f"emb{v}_oth.npy")
        pool = pair_dot(pool, ids1, ids2, E1, E2, f"cos{v}")
        del E1, E2
    for view, (S1v, S2v) in views.items():
        pool = pair_dot(pool, ids1, ids2, S1v, S2v, f"{view}cos")
    tm["cos_s"] = time.time() - t
    pool.write_parquet(out / "pool.parquet")
    secs = time.time() - t0
    record_time(f"block.{split}", secs, s1=len(s1), oth=len(oth), pairs=len(pool), **{k: round(v, 1) for k, v in tm.items()})
    log(f"block[{split}]: {len(pool):,} pool pairs, {len(pool) / len(s1):.1f} per S1 ({secs / 60:.1f} min; {tm})")
    _stats(split, s1, oth, pool)


def _routes(split, s1, oth, all1, all2, tm):
    """All retrieval routes merged into one row per (a, b) — before the cosine features."""
    out = work_dir(split)
    t = time.time()
    routes = []
    for v in DENSE:
        E1, E2 = np.load(out / f"emb{v}_s1.npy"), np.load(out / f"emb{v}_oth.npy")
        routes += list(_vector_route(s1, oth, E1, E2, all1, all2, P.dense_k_fwd, P.dense_k_rev,
                                     f"rank{v}_fwd", f"rank{v}_rev"))
        del E1, E2
    tm["dense_s"] = time.time() - t

    t = time.time()
    views = {}
    for view, col, has1, has2 in [("n3", "core", all1, all2),
                                  ("a3", "addr_core", s1["has_addr"].to_numpy(), oth["has_addr"].to_numpy())]:
        sk = sketch(s1[col].to_list() + oth[col].to_list(), P.sketch_dim)
        S1v, S2v = sk[:len(s1)], sk[len(s1):]
        views[view] = (S1v, S2v)
        m1 = has1 & (s1[col].str.len_chars() > 0).to_numpy()
        m2 = has2 & (oth[col].str.len_chars() > 0).to_numpy()
        f, r = _vector_route(s1, oth, S1v, S2v, m1, m2, P.c3_k_fwd, P.c3_k_rev, f"{view}_rank_fwd", f"{view}_rank_rev")
        if view == "a3":   # trade-name route: keep number-compatible pairs only
            numok = lambda d: (d.join(s1.select(a="iid", fa="first_num"), on="a")
                               .join(oth.select(b="iid", nb="nums"), on="b")
                               .filter((pl.col("fa") == "") | (pl.col("nb").list.len() == 0)
                                       | pl.col("nb").list.contains(pl.col("fa")))
                               .drop("fa", "nb"))
            f, r = numok(f), numok(r)
        routes += [f, r]
    tm["sketch_s"] = time.time() - t

    t = time.time()
    routes.append(_sparse(s1, oth))
    for name, (col, minlen) in KEY_ROUTES.items():
        routes.append(_key_route(s1, oth, col, minlen, name))
    tm["keys_s"] = time.time() - t

    pool = pl.concat(routes, how="diagonal_relaxed").group_by("a", "b").agg(pl.all().min())
    del routes
    pool = pool.with_columns(
        *[pl.col(c).fill_null(k + 1).cast(pl.Int16) for c, k in RANKS.items()],
        *[pl.col(c).fill_null(0).cast(pl.Int8) for c in KEY_ROUTES],
        sp_score=pl.col("sp_score").fill_null(0.0), sp_n=pl.col("sp_n").fill_null(0))
    return pool, views


def _stats(split, s1, oth, pool):
    out = work_dir(split)
    if split == "train":
        from .data import gt_for
        pairs, _ = gt_for(s1["iid"])
        pairs = pairs.join(oth.select(b="iid"), on="b").join(s1.select(a="iid", country="country"), on="a")
        flags = {**{f"dense{v}_fwd": pl.col(f"rank{v}_fwd") <= P.dense_k_fwd for v in DENSE},
                 **{f"dense{v}_rev": pl.col(f"rank{v}_rev") <= P.dense_k_rev for v in DENSE},
                 "name_c3": (pl.col("n3_rank_fwd") <= P.c3_k_fwd) | (pl.col("n3_rank_rev") <= P.c3_k_rev),
                 "addr_c3": (pl.col("a3_rank_fwd") <= P.c3_k_fwd) | (pl.col("a3_rank_rev") <= P.c3_k_rev),
                 "sparse": pl.col("sp_rank") <= P.sparse_k,
                 **{k: pl.col(k) == 1 for k in KEY_ROUTES}}
        fl = pool.select("a", "b", **flags)
        hit = pairs.join(fl, on=["a", "b"], how="left").with_columns(pl.col(list(flags)).fill_null(False))
        routes_n = pl.sum_horizontal([pl.col(k).cast(pl.Int8) for k in flags])
        hit = hit.with_columns(found=routes_n > 0, n_routes=routes_n)
        per_route = {}
        for k in flags:
            per_route[k] = {"recall": float(hit[k].mean()),
                            "only_this_route": float((hit[k] & (hit["n_routes"] == 1)).mean()),
                            "pairs_per_s1": float(fl[k].sum() / len(s1))}
        stats = {"pool_per_s1": len(pool) / len(s1), "recall_union": float(hit["found"].mean()),
                 "ann_recall_vs_exact": dict(ANN_CHECK),
                 "routes": per_route,
                 "by_country": {c: float(r) for c, r in hit.group_by("country").agg(r=pl.col("found").mean()).iter_rows()},
                 "entities_all_found": float(hit.group_by("a").agg(ok=pl.col("found").all())["ok"].mean())}
        (out / "block_stats.json").write_text(json.dumps(stats, indent=2))
        log(f"block[train] recall: {json.dumps(stats)}")
