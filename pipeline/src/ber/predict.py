"""Stage `predict`: score the test shortlist, apply the tuned decision rule, write submission files.

  output/matching_results.tsv  source1_entity_id \t matched_entity_ids   (every test S1, empty = none)
  output/candidate_pairs.tsv   source1_entity_id \t candidate_entity_ids (the stage-1 shortlist =
                               exactly the pairs the final model scored)
Also saves work/test/scored.parquet (a, b, p1, p, p_any) for cross-version ensembling.
"""
import json
import pickle
import time

import numpy as np
import polars as pl
import xgboost as xgb

from . import decide, learner
from .config import OUT, log, record_time, work_dir
from .data import str_id
from .train import ANY_FEATS, S1_COLS, any_table, load_features


def _write_lists(s1: pl.DataFrame, pairs: pl.DataFrame, col: str, path) -> None:
    lists = (pairs.unique(["a", "b"]).sort(["a", "b"])
             .with_columns(bid=str_id("b")).group_by("a").agg(pl.col("bid").str.join(",")))
    out = (s1.select(a="iid").join(lists, on="a", how="left")
           .select(source1_entity_id=str_id("a", source1=True), **{col: pl.col("bid").fill_null("")}))
    assert out["source1_entity_id"].n_unique() == len(s1)
    out.write_csv(path, separator="\t", quote_style="never")


def run() -> None:
    t0 = time.time()
    md, wd = work_dir("model"), work_dir("test")
    OUT.mkdir(parents=True, exist_ok=True)
    rep = json.loads((md / "report.json").read_text())
    rule = json.loads((md / "decision.json").read_text())["best"]
    cals = pickle.load(open(md / "calibrators.pkl", "rb"))
    df = load_features("test")
    feats = rep["feats"]
    final = xgb.Booster(model_file=str(md / "final.ubj"))
    p_raw = learner.predict(final, df.select(pl.col(feats).cast(pl.Float32)).to_numpy(), rep["final_trees"], feats)
    scored = df.select("a", "b", "p1", "n_cand_a").with_columns(p=pl.Series(cals["pair"].predict(p_raw).astype(np.float32)))

    s1 = pl.read_parquet(wd / "s1.parquet", columns=S1_COLS)
    A = any_table(scored, s1)
    a_final = xgb.Booster(model_file=str(md / "any_final.ubj"))
    pa = learner.predict(a_final, A.select(ANY_FEATS).to_numpy(), rep["any_final_trees"], ANY_FEATS)
    A = A.with_columns(p_any=pl.Series(cals["any"].predict(pa).astype(np.float32)))
    scored = scored.join(A.select("a", "p_any"), on="a", how="left")
    scored.select("a", "b", "p1", "p", "p_any").write_parquet(wd / "scored.parquet")
    A.select("a", "p_any").write_parquet(wd / "any_scored.parquet")

    chosen = decide.select(scored.filter(pl.col("p") >= 0.02), rule)
    assert chosen.join(df.select("a", "b"), on=["a", "b"], how="anti").is_empty(), "match not in candidates"
    _write_lists(s1, chosen, "matched_entity_ids", OUT / "matching_results.tsv")
    _write_lists(s1, df.select("a", "b"), "candidate_entity_ids", OUT / "candidate_pairs.tsv")

    per = s1.join(chosen.group_by("a").len("k"), left_on="iid", right_on="a", how="left").fill_null(0)
    cand = s1.join(df.group_by("a").len("c"), left_on="iid", right_on="a", how="left").fill_null(0)
    stats = {c: {"s1": len(g), "empty_rate": float((g["k"] == 0).mean()), "mean_matches": float(g["k"].mean())}
             for (c,), g in per.group_by("country")}
    for (c,), g in cand.group_by("country"):
        stats[c]["candidates_per_s1"] = float(g["c"].mean())
    summary = {"rule": rule, "test_pairs_scored": len(df), "candidates_per_s1": len(df) / len(s1),
               "test_matches": len(chosen), "by_country": stats}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2))
    record_time("predict", time.time() - t0, pairs=len(df))
    log(f"predict: {len(chosen):,} matches, {len(df) / len(s1):.1f} candidates/S1; {json.dumps(stats)}")
