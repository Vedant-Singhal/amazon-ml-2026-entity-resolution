"""Stage `stage1`: cheap features on the whole pool -> fast OOF model p1 -> learned shortlist.

The portal rule: blocking "must cut the search space to a small candidate set per Source 1
entity", smaller ranks higher. The shortlist (= candidate_pairs.tsv, the exact set the final
model scores) keeps, per S1, candidates with rank(p1) <= K and p1 >= tau. (K, tau) is the
smallest set that keeps >= sl_keep (99.5%) of the pool's recall, chosen on train OOF p1.
The full recall-vs-size curve is written to shortlist.json (ablation A2).

Outputs work/<split>/s1feats/part-*.parquet, work/<split>/sl.parquet (shortlist + stage-1
features + p1), work/model1/ (fold models, shortlist.json).
"""
import gc
import json
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from . import learner
from .block import DENSE
from .config import P, SMOKE, log, record_time, work_dir
from .data import fold_of, gt_for

A_COLS = ["iid", "name_n", "core", "addr_n", "first_num", "has_addr", "native", "u_s1_core", "u_oth_core",
          "u_s1_addr", "u_oth_addr"]
ID_COLS = ["a", "b", "label"]
KS = [3, 5, 8, 10, 12, 15, 20, 30]
TAUS = [0.0, 0.001, 0.003, 0.01, 0.02, 0.05, 0.1]


def _competition(c: pl.DataFrame) -> pl.DataFrame:
    g = lambda x, by: pl.col(x) - pl.col(x).max().over(by)
    r = lambda x, by: pl.col(x).rank("min", descending=True).over(by).cast(pl.Int32)
    return c.with_columns(
        n_cand_a=pl.len().over("a").cast(pl.Int32), n_cand_b=pl.len().over("b").cast(pl.Int32),
        cos_gap_a=g("cos", "a"), cos_rank_a=r("cos", "a"), cos_gap_b=g("cos", "b"), cos_rank_b=r("cos", "b"),
        n3_gap_a=g("n3cos", "a"), n3_rank_b=r("n3cos", "b"), a3_gap_a=g("a3cos", "a"),
        sp_gap_a=g("sp_score", "a"),
        **{f"cos{v}_gap_a": g(f"cos{v}", "a") for v in DENSE[1:]},
        **{f"cos{v}_rank_b": r(f"cos{v}", "b") for v in DENSE[1:]},
        n_routes=pl.sum_horizontal(
            *[(pl.col(c) <= k).cast(pl.Int8) for c, k in [
                *[(f"rank{v}_fwd", P.dense_k_fwd) for v in DENSE], *[(f"rank{v}_rev", P.dense_k_rev) for v in DENSE],
                ("n3_rank_fwd", P.c3_k_fwd),
                ("n3_rank_rev", P.c3_k_rev), ("a3_rank_fwd", P.c3_k_fwd), ("a3_rank_rev", P.c3_k_rev),
                ("sp_rank", P.sparse_k)]],
            pl.col("^k_.*$").cast(pl.Int8)).cast(pl.Int8))


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=P.workers, dtype=np.float32)


def _cheap_chunk(c: pl.DataFrame) -> pl.DataFrame:
    g = lambda col: c[col].fill_null("").to_list()
    has = (c["has_addr"] & c["has_addr_b"]).to_numpy()
    lg = lambda col: np.log1p(c[col].cast(pl.Float32).to_numpy())
    f = {
        "n_tset": _cp(g("name_n"), g("name_n_b"), fuzz.token_set_ratio),
        "c_ratio": _cp(g("core"), g("core_b"), fuzz.ratio),
        "a_tset": np.where(has, _cp(g("addr_n"), g("addr_n_b"), fuzz.token_set_ratio), np.nan).astype(np.float32),
        "first_num_eq": np.where((c["first_num"] != "") & (c["first_num_b"] != ""),
                                 (c["first_num"] == c["first_num_b"]).cast(pl.Float32), np.nan).astype(np.float32),
        "core_eq": (c["core"] == c["core_b"]).cast(pl.Float32).to_numpy(),
        "has_addr_a": c["has_addr"].cast(pl.Float32).to_numpy(),
        "has_addr_b": c["has_addr_b"].cast(pl.Float32).to_numpy(),
        "native_b": c["native_b"].cast(pl.Float32).to_numpy(),
        "is_s3": (c["b"] >= 10**10).cast(pl.Float32).to_numpy(),
        "lu_s1_core_a": lg("u_s1_core"), "lu_oth_core_a": lg("u_oth_core"),
        "lu_s1_core_b": lg("u_s1_core_b"), "lu_oth_core_b": lg("u_oth_core_b"),
        "lu_s1_addr_a": lg("u_s1_addr"), "lu_oth_addr_b": lg("u_oth_addr_b"),
    }
    drop = [x for x in c.columns if x in A_COLS or x.endswith("_b") and x[:-2] in A_COLS]
    return c.drop(drop).with_columns(**{k: pl.Series(v) for k, v in f.items()})


def features(split: str) -> None:
    out = work_dir(split, "s1feats")
    if (out / "_DONE").exists():
        return
    t0 = time.time()
    wd = work_dir(split)
    pool = _competition(pl.read_parquet(wd / "pool.parquet"))
    s1 = pl.read_parquet(wd / "s1.parquet", columns=A_COLS)
    if split == "train":
        pairs, _ = gt_for(s1["iid"])
        pool = pool.join(pairs.with_columns(label=pl.lit(1, pl.Int8)), on=["a", "b"], how="left") \
                   .with_columns(pl.col("label").fill_null(0))
    oth = pl.read_parquet(wd / "oth.parquet", columns=A_COLS)
    oth = oth.rename({c: f"{c}_b" for c in oth.columns if c != "iid"})
    ids = s1["iid"].to_numpy()
    step = 200_000
    n = 0
    for i, s in enumerate(range(0, len(ids), step)):
        c = (pool.filter(pl.col("a").is_in(pl.Series(ids[s:s + step])))
             .join(s1, left_on="a", right_on="iid").join(oth, left_on="b", right_on="iid"))
        _cheap_chunk(c).write_parquet(out / f"part-{i:04d}.parquet")
        n += len(c)
        log(f"stage1 features[{split}]: chunk {i} ({len(c):,} pairs)")
    (out / "_DONE").touch()
    record_time(f"stage1.features.{split}", time.time() - t0, s1=len(s1), pairs=n)


def _load(split):
    df = pl.read_parquet(work_dir(split, "s1feats") / "part-*.parquet")
    feats = [c for c in df.columns if c not in ID_COLS]
    X = df.select(pl.col(feats).cast(pl.Float32)).to_numpy()
    return df, feats, X


def _curve(a: np.ndarray, p: np.ndarray, label: np.ndarray, n_gt: int, n_s1: int) -> list:
    """Recall and mean size of the shortlist for every (K, tau)."""
    order = np.lexsort((-p, a))
    a_s, p_s, l_s = a[order], p[order], label[order]
    start = np.r_[0, np.flatnonzero(a_s[1:] != a_s[:-1]) + 1]
    rank = np.arange(len(a_s)) - np.repeat(start, np.diff(np.r_[start, len(a_s)])) + 1
    rows = []
    for K in KS:
        for tau in TAUS:
            m = (rank <= K) & (p_s >= tau)
            rows.append({"K": K, "tau": tau, "recall": float(l_s[m].sum() / max(n_gt, 1)),
                         "per_s1": float(m.sum() / max(n_s1, 1))})
    return rows


def shortlist_mask(df: pl.DataFrame, p: np.ndarray, rule: dict) -> np.ndarray:
    r = (pl.DataFrame({"a": df["a"], "p": p})
         .with_columns(r=pl.col("p").rank("ordinal", descending=True).over("a"))["r"].to_numpy())
    return (r <= rule["K"]) & (p >= rule["tau"])


def _write_sl(split: str, df: pl.DataFrame, p: np.ndarray, m: np.ndarray) -> None:
    df.with_columns(p1=pl.Series(p)).filter(pl.Series(m)).write_parquet(work_dir(split) / "sl.parquet")


def run() -> None:
    md = work_dir("model1")
    if (md / "shortlist.json").exists() and (work_dir("test") / "sl.parquet").exists():
        log("stage1: cached")
        return
    for split in ("train", "test"):
        features(split)
    t0 = time.time()
    df, feats, X = _load("train")
    y = df["label"].to_numpy()
    fold = df.select(fold_of("a")).to_series().to_numpy()
    hard = ((df["label"] == 1) | (df["rank_fwd"] <= 3) | (df["cos_rank_b"] <= 1) | (df["n_routes"] >= 3)
            | (df.select(pl.max_horizontal(pl.col("^k_.*$")))[:, 0] == 1)).to_numpy()
    rng = np.random.default_rng(P.seed)
    keep = hard | (rng.random(len(df)) < P.sl_neg_rate)
    w = np.where(hard, 1.0, 1.0 / P.sl_neg_rate).astype(np.float32)
    log(f"stage1: {len(df):,} pool pairs, {int(y.sum()):,} positives, {len(feats)} features, "
        f"{int(keep.sum()):,} used for training")
    oof, models, iters, rate = learner.cv(X, y, w, fold, keep, P.lr1, P.rounds1, feats, "stage1")
    for k, m in enumerate(models):
        m.save_model(str(md / f"fold{k}.ubj"))

    s1 = pl.read_parquet(work_dir("train") / "s1.parquet", columns=["iid"])
    pairs, n_true = gt_for(s1["iid"])
    n_gt, n_s1 = len(pairs), len(s1)
    curve = _curve(df["a"].to_numpy(), oof, y, n_gt, n_s1)
    pool_recall = float(y.sum() / max(n_gt, 1))
    target = P.sl_keep * pool_recall
    ok = [r for r in curve if r["recall"] >= target]
    rule = min(ok, key=lambda r: r["per_s1"]) if ok else max(curve, key=lambda r: r["recall"])
    rep = {"rule": rule, "pool_recall": pool_recall, "pool_per_s1": len(df) / n_s1, "target_recall": target,
           "curve": curve, "trees": iters, "sec_per_round_per_1M": rate, "gain": learner.gain(models, feats)}
    (md / "shortlist.json").write_text(json.dumps(rep, indent=2))
    log(f"stage1: shortlist rule {rule} (pool {len(df) / n_s1:.1f}/S1 recall {pool_recall:.4f})")
    _write_sl("train", df, oof, shortlist_mask(df, oof, rule))
    record_time("stage1.train", time.time() - t0, rows=int(keep.sum()), pairs=len(df), trees=iters,
                sec_per_round_per_1M=rate)
    del df, X
    gc.collect()

    t0 = time.time()
    df, feats_t, X = _load("test")
    assert feats_t == feats
    p = np.mean([learner.predict(m, X, n, feats) for m, n in zip(models, iters)], axis=0)
    m = shortlist_mask(df, p, rule)
    _write_sl("test", df, p, m)
    s1t = df["a"].n_unique()
    log(f"stage1[test]: {int(m.sum()):,} shortlisted of {len(df):,} ({m.sum() / max(s1t, 1):.1f} per S1 with a pool)")
    record_time("stage1.test", time.time() - t0, pairs=len(df))
