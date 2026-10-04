"""Train at the test's orphan density (ISSUES.md).

Test has ~2x the orphan S2/S3 records per S1 of train (records that belong to no S1: 26% of train records vs ~42% of
test records) — the test looks like train with ~19% of its S1 removed and their records kept. Orphans are clean,
complete records and already cause 63% of v4's false pairs on train; on test the model meets twice as many.

This step removes a share BER_TRAIN_S1_DROP of the train S1 (by a hash of the id) once the pool is built:
their rows in s1.parquet and in every S1-aligned array (*_s1.npy), their pool pairs and retriever rows. Reverse-route
ranks (rank of an S1 among a record's nearest S1) are re-ranked among the S1 that remain, as a search without the
dropped S1 would give. Their S2/S3 records stay, as orphans. Stage 1 onward (b-side competition features, models,
calibration, decision rule) is then learned under the test situation, and OOF becomes an honest LB estimate. GT is
read through s1.parquet, so the dropped S1 are neither trained nor scored. Runs once (marker work/train/.s1drop).
"""
import json
import os
import time

import numpy as np
import polars as pl

from .block import RANKS
from .config import P, log, record_time, work_dir
from .retriever import K_REV

REV = {**{c: k for c, k in RANKS.items() if c.endswith("_rev")}, "ft_rank_rev": K_REV}


def share() -> float:
    return float(os.environ.get("BER_TRAIN_S1_DROP", "0") or 0)


def needed() -> bool:
    return share() > 0 and not (work_dir("train") / ".s1drop").exists()


def run() -> None:
    t0, d, wd = time.time(), share(), work_dir("train")
    s1 = pl.read_parquet(wd / "s1.parquet")
    keep = ((s1["iid"].hash(P.seed + 99) % 1000) >= int(round(d * 1000))).to_numpy()
    kept = pl.DataFrame({"a": s1["iid"].filter(pl.Series(keep))})
    for f in sorted(wd.glob("*_s1.npy")):                   # row-aligned with s1.parquet
        tmp = f.with_name(f.stem + ".tmp.npy")
        np.save(tmp, np.ascontiguousarray(np.load(f, mmap_mode="r")[keep]))
        os.replace(tmp, f)
    s1.filter(pl.Series(keep)).write_parquet(wd / "s1.parquet")
    del s1

    pool = pl.read_parquet(wd / "pool.parquet")
    n0 = len(pool)
    pool = pool.join(kept, on="a", how="semi")
    for c, k in REV.items():                                 # one column at a time: bounded peak memory
        if c in pool.columns:
            y = pl.when(pl.col(c) <= k).then(pl.col(c)).otherwise(32_000)
            pool = pool.with_columns(pl.when(pl.col(c) <= k).then(y.rank("min").over("b")).otherwise(pl.col(c))
                                     .cast(pool.schema[c]).alias(c))
    pool.write_parquet(wd / "pool.parquet")
    rf = wd / "route_ft.parquet"
    if rf.exists():
        pl.read_parquet(rf).join(kept, on="a", how="semi").write_parquet(rf)
    info = {"share": d, "s1_before": int(len(keep)), "s1_after": int(keep.sum()), "pool_before": n0,
            "pool_after": len(pool), "rev_ranks": [c for c in REV if c in pool.columns]}
    (wd / ".s1drop").write_text(json.dumps(info, indent=2))
    record_time("s1drop", time.time() - t0, pairs=len(pool))
    log(f"s1drop: train S1 {info['s1_before']:,} -> {info['s1_after']:,} (dropped {d:.0%}, their records stay as "
        f"orphans); pool {n0:,} -> {len(pool):,} pairs; re-ranked {info['rev_ranks']}")
