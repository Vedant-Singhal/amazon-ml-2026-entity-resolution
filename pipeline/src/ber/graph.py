"""Stage `graph` (v4): consistency features across the records an S1 collects (S2 and S3 together).

A true match should look like the other records its S1 already strongly holds ("siblings"), and a record
claimed by several S1s is suspect. Ideas from multi-source ER cleanup: TransClean (2025, transitive false
positives) and CLIP clustering (Saeedi et al., 2018: strong links per source, weak "bridge" links removed).
Computed from v3's stage-2 probability p2 (OOF on train, so leak-free).

Per pair (a, b):
  g_n_hi_other     other records of a with p2 >= 0.5          g_sum_p_other    sum of p2 over a's other records
  g_best_other_a   best p2 among a's other records            g_best_other_b   best p2 of b with any other S1
  g_sib_cos_max    max cosine (embedding) between b and a's strong siblings (p2 >= 0.5), and mean / count
  g_sib_cos_xsrc   same, siblings from the other source only (S2 vs S3)
  g_sib_name_max   best name similarity between b and a's 3 strongest siblings
Output work/graph/{train,test}.parquet.
"""
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from .block import pair_dot
from .config import P, log, record_time, work_dir
from .data import S3_OFFSET
from .v4io import v3_scores

HI = 0.5


def _second_best(df: pl.DataFrame, by: str, name: str) -> pl.DataFrame:
    """Best p2 among the OTHER pairs of the same `by` key (top-2 trick)."""
    top1 = pl.col("p2").max().over(by)
    top2 = pl.col("p2").top_k(2).min().over(by)
    n = pl.len().over(by)
    return df.with_columns(pl.when(n == 1).then(pl.lit(None, dtype=pl.Float32))
                           .when(pl.col("p2") == top1).then(top2).otherwise(top1).alias(name))


def features(split: str) -> None:
    out = work_dir("graph")
    if (out / f"{split}.parquet").exists():
        return
    t0 = time.time()
    wd = work_dir(split)
    sc = v3_scores(split).select("a", "b", "p2")
    agg = sc.group_by("a").agg(sum_p=pl.col("p2").sum(), n_hi=(pl.col("p2") >= HI).sum())
    g = (sc.join(agg, on="a")
         .with_columns(g_n_hi_other=(pl.col("n_hi") - (pl.col("p2") >= HI).cast(pl.UInt32)).cast(pl.Float32),
                       g_sum_p_other=(pl.col("sum_p") - pl.col("p2")).cast(pl.Float32))
         .drop("sum_p", "n_hi"))
    g = _second_best(g, "a", "g_best_other_a")
    g = _second_best(g, "b", "g_best_other_b")

    # siblings: the strong records of the same S1, other than b itself
    hi = sc.filter(pl.col("p2") >= HI).select("a", sib="b", sib_p="p2")
    sibs = sc.select("a", "b").join(hi, on="a").filter(pl.col("sib") != pl.col("b"))
    sibs = sibs.with_columns(xsrc=(pl.col("b") >= S3_OFFSET) != (pl.col("sib") >= S3_OFFSET))
    log(f"graph[{split}]: {len(sc):,} pairs, {len(sibs):,} (pair, sibling) rows")
    oth = pl.read_parquet(wd / "oth.parquet", columns=["iid", "name_n"])
    E = np.load(wd / "emb_oth.npy")                                  # first embedder view
    ids = oth["iid"].to_numpy()
    cs = pair_dot(sibs.rename({"a": "s1", "b": "a", "sib": "b"}), ids, ids, E, E, "sib_cos")
    del E
    cs = cs.rename({"a": "_b", "b": "_sib", "s1": "_a"}).rename({"_a": "a", "_b": "b", "_sib": "sib"})
    cagg = cs.group_by("a", "b").agg(
        g_sib_cos_max=pl.col("sib_cos").max(), g_sib_cos_mean=pl.col("sib_cos").mean(),
        g_n_sib=pl.len().cast(pl.Float32),
        g_sib_cos_xsrc=pl.col("sib_cos").filter(pl.col("xsrc")).max())

    # name coherence with the 3 strongest siblings
    top3 = (sibs.with_columns(r=pl.col("sib_p").rank("ordinal", descending=True).over("a", "b"))
            .filter(pl.col("r") <= 3)
            .join(oth.select(b="iid", nb="name_n"), on="b")
            .join(oth.select(sib="iid", ns="name_n"), on="sib"))
    sim = process.cpdist(top3["nb"].fill_null("").to_list(), top3["ns"].fill_null("").to_list(),
                         scorer=fuzz.token_set_ratio, workers=P.workers, dtype=np.float32)
    nagg = top3.select("a", "b").with_columns(s=pl.Series(sim)).group_by("a", "b").agg(g_sib_name_max=pl.col("s").max())

    g = (g.join(cagg, on=["a", "b"], how="left").join(nagg, on=["a", "b"], how="left")
         .drop("p2").with_columns(pl.col("^g_.*$").cast(pl.Float32)))
    g.write_parquet(out / f"{split}.parquet")
    record_time(f"graph.{split}", time.time() - t0, pairs=len(g), sib_rows=len(sibs))
    log(f"graph[{split}]: done in {(time.time() - t0) / 60:.1f} min")


def run() -> None:
    for split in ("train", "test"):
        features(split)
