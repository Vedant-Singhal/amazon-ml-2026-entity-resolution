"""v6: add a dense embedding view to a run whose candidate pool is already built (e.g. seeded from v4/v5).

BER_EMB_MODELS lists the views (v6: e5-small, e5-large, e5-large-instruct). For every view the pool does not know
yet (no rank{v}_fwd column): embed the records with that model only (embed.run skips views whose files exist), search
both ways within country exactly as `block` does (top dense_k_fwd per S1, top dense_k_rev S1 per record), add the pairs
only this view finds (all other routes at their "not found" defaults, every cosine / sketch similarity computed), and
add the view's rank columns and cosine to every existing pair. Stage 1 onward must be rebuilt afterwards.
Why e5-large-instruct: best of the ablated MIT/Apache embedders on records without an address (train's largest miss
group: R@10 India 0.358 / US 0.759 vs e5-small 0.340 / 0.736, e5-large 0.306 / 0.655 — v3/diagnosis/ablate_A0r2.md).
"""
import time

import numpy as np
import polars as pl

from . import embed
from .block import DENSE, KEY_ROUTES, RANKS, _vector_route, pair_dot
from .config import P, log, record_time, work_dir


def _missing(split: str) -> list:
    cols = pl.read_parquet_schema(work_dir(split) / "pool.parquet")
    return [v for v in DENSE if f"rank{v}_fwd" not in cols]


def needed() -> bool:
    return any(_missing(sp) for sp in ("train", "test"))


def _add(split: str, v: str) -> None:
    t0 = time.time()
    wd = work_dir(split)
    s1 = pl.read_parquet(wd / "s1.parquet", columns=["iid", "country"])
    oth = pl.read_parquet(wd / "oth.parquet", columns=["iid", "country"])
    ids1, ids2 = s1["iid"].to_numpy(), oth["iid"].to_numpy()
    E1, E2 = np.load(wd / f"emb{v}_s1.npy"), np.load(wd / f"emb{v}_oth.npy")
    fc, rc = f"rank{v}_fwd", f"rank{v}_rev"
    fwd, rev = _vector_route(s1, oth, E1, E2, np.ones(len(s1), bool), np.ones(len(oth), bool),
                             P.dense_k_fwd, P.dense_k_rev, fc, rc)
    route = fwd.join(rev, on=["a", "b"], how="full", coalesce=True)
    pool = pl.read_parquet(wd / "pool.parquet")
    n0 = len(pool)
    new = route.join(pool.select("a", "b"), on=["a", "b"], how="anti")
    defaults = [pl.lit(k + 1, pl.Int16).alias(c) for c, k in RANKS.items() if c in pool.columns and c not in (fc, rc)]
    defaults += [pl.lit(0, pl.Int8).alias(c) for c in KEY_ROUTES if c in pool.columns]
    if "sp_score" in pool.columns:
        defaults += [pl.lit(0.0, pl.Float32).alias("sp_score"), pl.lit(0, pl.Int16).alias("sp_n")]
    if "ft_rank_fwd" in pool.columns:                       # retriever route (v4+), K_FWD / K_REV = 10 / 3
        defaults += [pl.lit(11, pl.Int16).alias("ft_rank_fwd"), pl.lit(4, pl.Int16).alias("ft_rank_rev")]
    new = new.with_columns(defaults)
    for w in DENSE:                                          # all dense cosines for the new pairs
        new = pair_dot(new, ids1, ids2, np.load(wd / f"emb{w}_s1.npy"), np.load(wd / f"emb{w}_oth.npy"), f"cos{w}")
    for view in ("n3", "a3"):
        new = pair_dot(new, ids1, ids2, np.load(wd / f"sk_{view}_s1.npy"), np.load(wd / f"sk_{view}_oth.npy"), f"{view}cos")
    old = pair_dot(pool.join(route, on=["a", "b"], how="left"), ids1, ids2, E1, E2, f"cos{v}")
    del pool
    pool = pl.concat([old, new], how="diagonal_relaxed").with_columns(
        pl.col(fc).fill_null(P.dense_k_fwd + 1).cast(pl.Int16), pl.col(rc).fill_null(P.dense_k_rev + 1).cast(pl.Int16))
    pool.write_parquet(wd / "pool.parquet")
    record_time(f"addview{v}", time.time() - t0, pairs=len(pool))
    log(f"addview[{split}] view {v}: pool {n0:,} -> {len(pool):,} pairs (+{len(new):,} found only by this view) "
        f"in {(time.time() - t0) / 60:.1f} min")


def run() -> None:
    for split in ("train", "test"):
        miss = _missing(split)
        if not miss:
            continue
        embed.run(split)                                     # only the views without embedding files
        for v in miss:
            _add(split, v)
