"""Stage `diagnose` (v4): where the remaining errors are, for a given model's OOF (v3 stage 2 or v4 stage 3).

  1. Where each missed true pair was lost: not in the pool (search) / in the pool but not shortlisted /
     shortlisted but rejected by the model.
  2. Missed and false pairs by type: record without address, native-script name, shared name (how many S1
     share the core name), house-number relation, source (S2 / S3), country.
  3. Test profile per country (France has no labels): share of S1 with a match, matches per S1, candidates per
     S1, score quantiles — compared with train OOF. A large gap for France flags a transfer problem.
Output: output/diagnose_{tag}.md
"""
import json

import numpy as np
import polars as pl

from . import decide
from .config import OUT, log, work_dir
from .data import S3_OFFSET, gt_for
from .train import load_features

NUM_REL = {0: "missing", 1: "equal", 2: "truncated", 3: "one-digit edit", 4: "near (<=10)", 5: "different"}


def _table(df: pl.DataFrame) -> str:
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.iter_rows()])


def run(tag: str = "v3") -> None:
    md = work_dir("model3" if tag == "v4" else "model")
    oof = pl.read_parquet(md / "oof.parquet")
    rule = json.loads((md / "decision.json").read_text())["best"]
    chosen = decide.select(oof.filter(pl.col("p") >= 0.02), rule).with_columns(pred=pl.lit(True))
    s1 = pl.read_parquet(work_dir("train") / "s1.parquet", columns=["iid", "country"])
    pairs, n_true = gt_for(s1["iid"])
    rep = [f"# Error diagnosis — {tag} (train OOF, rule {rule['kind']})\n"]

    # 1. where true pairs are lost
    pool = pl.read_parquet(work_dir("train") / "pool.parquet", columns=["a", "b"]).with_columns(in_pool=pl.lit(True))
    sl = pl.read_parquet(work_dir("train") / "sl.parquet", columns=["a", "b"]).with_columns(in_sl=pl.lit(True))
    t = (pairs.join(pool, on=["a", "b"], how="left").join(sl, on=["a", "b"], how="left")
         .join(chosen.select("a", "b", "pred"), on=["a", "b"], how="left").fill_null(False))
    stage = (pl.when(~pl.col("in_pool")).then(pl.lit("1 not found by search"))
             .when(~pl.col("in_sl")).then(pl.lit("2 dropped by shortlist"))
             .when(~pl.col("pred")).then(pl.lit("3 rejected by model")).otherwise(pl.lit("found")))
    loss = t.with_columns(stage=stage).group_by("stage").len("pairs").sort("stage")
    loss = loss.with_columns(share=(pl.col("pairs") / len(t)).round(4))
    rep += ["## 1. Where the true pairs go\n", _table(loss), ""]

    # 2. misses and false pairs by type (shortlisted pairs; features from v3 stage 2)
    f = load_features("train").select("a", "b", "label", "has_addr_b", "native_b", "num_rel", "lu_s1_core_a")
    f = f.join(oof.select("a", "b", "p"), on=["a", "b"]).join(chosen.select("a", "b", "pred"), on=["a", "b"], how="left")
    f = f.with_columns(pl.col("pred").fill_null(False)).join(s1.select(a="iid", country="country"), on="a")
    f = f.with_columns(
        miss=(pl.col("label") == 1) & ~pl.col("pred"), false=(pl.col("label") == 0) & pl.col("pred"),
        shared=pl.when(pl.col("lu_s1_core_a") <= np.log1p(1)).then(pl.lit("unique"))
        .when(pl.col("lu_s1_core_a") <= np.log1p(3)).then(pl.lit("2-3 S1")).otherwise(pl.lit("4+ S1")),
        num=pl.col("num_rel").cast(pl.Int8).replace_strict(NUM_REL, default="?", return_dtype=pl.String),
        source=pl.when(pl.col("b") >= S3_OFFSET).then(pl.lit("S3")).otherwise(pl.lit("S2")),
        address=pl.when(pl.col("has_addr_b") > 0).then(pl.lit("has address")).otherwise(pl.lit("no address")),
        script=pl.when(pl.col("native_b") > 0).then(pl.lit("native script")).otherwise(pl.lit("Latin")))
    for dim in ("address", "shared", "script", "num", "source", "country"):
        g = (f.group_by(dim).agg(true_pairs=pl.col("label").sum(), missed=pl.col("miss").sum(),
                                 false_pairs=pl.col("false").sum())
             .with_columns(miss_rate=(pl.col("missed") / pl.col("true_pairs")).round(4)).sort(dim))
        rep += [f"## 2. By {dim}\n", _table(g), ""]

    # 3. test profile per country vs train OOF
    tr = (chosen.group_by("a").len("k").join(s1.select(a="iid", country="country"), on="a", how="right")
          .fill_null(0).group_by("country").agg(with_match=(pl.col("k") > 0).mean(), matches_per_s1=pl.col("k").mean()))
    scored = work_dir("test") / ("scored3.parquet" if tag == "v4" else "scored.parquet")
    if scored.exists():
        sc = pl.read_parquet(scored)
        ts1 = pl.read_parquet(work_dir("test") / "s1.parquet", columns=["iid", "country"])
        ch = decide.select(sc.filter(pl.col("p") >= 0.02), rule)
        te = (ch.group_by("a").len("k").join(ts1.select(a="iid", country="country"), on="a", how="right").fill_null(0)
              .group_by("country").agg(with_match=(pl.col("k") > 0).mean(), matches_per_s1=pl.col("k").mean()))
        cand = (sc.group_by("a").agg(n=pl.len(), p_max=pl.col("p").max())
                .join(ts1.select(a="iid", country="country"), on="a")
                .group_by("country").agg(candidates_per_s1=pl.col("n").mean(),
                                         p_max_median=pl.col("p_max").median()))
        prof = (te.join(cand, on="country", how="left")
                .join(tr.rename({"with_match": "train_with_match", "matches_per_s1": "train_matches_per_s1"}),
                      on="country", how="left").sort("country"))
        rep += ["## 3. Test profile per country (vs train OOF)\n", _table(prof), ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"diagnose_{tag}.md").write_text("\n".join(rep), encoding="utf-8")
    log(f"diagnose[{tag}]: wrote {OUT / f'diagnose_{tag}.md'}\n" + "\n".join(rep[:4]))
