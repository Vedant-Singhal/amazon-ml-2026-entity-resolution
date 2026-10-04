"""Stage `stage3` (v4): the final matcher, stacked on v3.

Features = v3's stage-2 features + v3's stage-2 score p2 + cross-encoder score p_ce (uncertain band only) +
consistency features g_* + v2's score p_v2. Every added score is out-of-fold on train with the same folds
(hash(S1 id, 13) % 5) and is computed the same way on test, so stacking is leak-free (Wolpert 1992).
Then, as in v3: XGBoost (GPU) 5-fold OOF, isotonic calibration, "has any match" model, decision rule tuned
with the exact macro F0.5 on OOF, final models on all train, predictions for test. Candidates are unchanged:
candidate_pairs.tsv is v3's shortlist.
Outputs: output/{matching_results,candidate_pairs}.tsv, run_summary.json, work/model3/ (report.json, oof.parquet,
decision.json, models), work/test/scored3.parquet.
"""
import json
import os
import pickle
import time

import numpy as np
import polars as pl
import xgboost as xgb

from . import decide, learner
from .config import OUT, P, log, record_time, work_dir
from .data import fold_of, gt_for
from .predict import _write_lists
from .train import ANY_FEATS, ID_COLS, S1_COLS, any_table, load_features
from .v4io import v2_scores, v3_scores

NEW = ["p2", "p_ce", "p_v2"]
# Density-dependent sums / counts shift on test (test has +23% records per S1, 6.3 vs 5.4 candidates, sum of p1
# 3.9 vs 3.4 per S1); maxima, gaps and similarities are kept.
DROP = [c for c in os.environ.get(
    "BER_V4_DROP", "n_cand_a,n_cand_b,n_sl_a,n_sl_b,p1_sum_a,g_sum_p_other,g_n_hi_other,g_n_sib").split(",") if c]
ANY_ROBUST = ["p_max", "p_2nd", "p1_max", "has_addr", "lu_s1_core", "lu_oth_core", "lu_s1_addr"]


def _assemble(split: str) -> pl.DataFrame:
    df = load_features(split)
    s = v3_scores(split).select("a", "b", "p2")
    parts = [s]
    ce = work_dir("ce") / f"p_ce_{split}.parquet"
    if ce.exists():
        parts.append(pl.read_parquet(ce))
    g = work_dir("graph") / f"{split}.parquet"
    if g.exists():
        parts.append(pl.read_parquet(g))
    parts.append(v2_scores(split))
    for x in parts:
        df = df.join(x, on=["a", "b"], how="left")
    return df


def run() -> None:
    md = work_dir("model3")
    if (md / "report.json").exists():
        log("stage3: cached")
        return
    t0 = time.time()
    df = _assemble("train")
    feats = [c for c in df.columns if c not in ID_COLS and c not in DROP]
    added = [c for c in feats if c in NEW or c.startswith("g_")]
    log(f"stage3: {len(df):,} pairs, {len(feats)} features (added: {added}); "
        f"p_ce on {df['p_ce'].is_not_null().sum() if 'p_ce' in df.columns else 0:,} pairs, "
        f"p_v2 on {df['p_v2'].is_not_null().sum() if 'p_v2' in df.columns else 0:,}")
    X = df.select(pl.col(feats).cast(pl.Float32)).to_numpy()
    y = df["label"].to_numpy()
    fold = df.select(fold_of("a")).to_series().to_numpy()
    oof, models, iters, rate = learner.cv(X, y, None, fold, np.ones(len(df), bool), P.lr2, P.rounds2, feats, "stage3")
    cal = decide.fit_calibrator(oof, y)
    p = cal.predict(oof).astype(np.float32)

    s1 = pl.read_parquet(work_dir("train") / "s1.parquet", columns=S1_COLS)
    pairs, n_true = gt_for(s1["iid"])
    scored = df.select("a", "b", "p1", "n_cand_a").with_columns(p=pl.Series(p))
    A = any_table(scored, s1).join(n_true, on="a", how="left").with_columns(pl.col("n_true").fill_null(0))
    XA, yA = A.select(ANY_ROBUST).to_numpy(), (A["n_true"] > 0).cast(pl.Int8).to_numpy()
    foldA = A.select(fold_of("a")).to_series().to_numpy()
    a_oof, _, a_iters, _ = learner.cv(XA, yA, None, foldA, np.ones(len(A), bool), 0.05, 2000, ANY_ROBUST, "stage3-any")
    a_cal = decide.fit_calibrator(a_oof, yA)
    A = A.with_columns(p_any=pl.Series(a_cal.predict(a_oof).astype(np.float32)))

    o = scored.select("a", "b", "p").join(A.select("a", "p_any"), on="a", how="left")
    o.with_columns(p_raw=pl.Series(oof), label=df["label"]).write_parquet(md / "oof.parquet")
    tuned = decide.tune(o.filter(pl.col("p") >= 0.02), pairs, n_true)
    rule = tuned["best"]
    (md / "decision.json").write_text(json.dumps(tuned, indent=2))
    chosen = decide.select(o.filter(pl.col("p") >= 0.02), rule)
    by_country = {c: decide.macro_f05(chosen.join(g.select(a="iid"), on="a"), pairs,
                                      n_true.join(g.select(a="iid"), on="a"))
                  for (c,), g in s1.group_by("country")}
    v3_rep = json.loads((work_dir("model") / "report.json").read_text())
    report = {"oof_macro_f05": rule["macro_f05"], "v3_oof_macro_f05": v3_rep["oof_macro_f05"],
              "gain_vs_v3": rule["macro_f05"] - v3_rep["oof_macro_f05"], "by_country": by_country, "rule": rule,
              "trees": iters, "added_features": added, "feature_gain": learner.gain(models, feats)}
    log(f"stage3: OOF macro F0.5 {rule['macro_f05']:.4f} (v3 {v3_rep['oof_macro_f05']:.4f}) {by_country}")

    n_final = int(np.mean(iters) * 1.1)
    final, _, _ = learner.fit(X, y, None, None, None, P.lr2, n_final, feats)
    final.save_model(str(md / "final.ubj"))
    na_final = int(np.mean(a_iters) * 1.1)
    a_final, _, _ = learner.fit(XA, yA, None, None, None, 0.05, na_final, ANY_ROBUST)
    a_final.save_model(str(md / "any_final.ubj"))
    pickle.dump({"pair": cal, "any": a_cal}, open(md / "calibrators.pkl", "wb"))
    thr = max((r for r in tuned["all"] if r["kind"] == "thresh" and not r.get("any_min") and r.get("one_owner", True)),
              key=lambda r: r["macro_f05"])
    report.update(final_trees=n_final, any_final_trees=na_final, feats=feats, thresh_rule=thr, dropped=DROP)
    (md / "report.json").write_text(json.dumps(report, indent=2))
    record_time("stage3.train", time.time() - t0, rows=len(df))
    del X, df
    # the LB prefers stricter cuts than the OOF optimum (v3: t=0.30 0.958, OOF rule 0.975, t=0.95 0.978), so also
    # write fixed strict thresholds: output/t0950/, output/t0980/ ...
    extra = {f"t{int(round(t * 1000)):04d}": {"kind": "thresh", "t": t, "floor": 0.0, "one_owner": True}
             for t in (float(x) for x in os.environ.get("BER_V4_EXTRA_T", "0.95,0.98,0.99").split(",") if x)}
    _predict(md, {"": rule, "thresh": thr, **extra}, feats, final, n_final, cal, a_final, na_final, a_cal)


def _predict(md, rules, feats, final, n_final, cal, a_final, na_final, a_cal) -> None:
    t0 = time.time()
    df = _assemble("test")
    for c in feats:                      # a score missing on test (e.g. no v2 file) stays NaN
        if c not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float32).alias(c))
    p_raw = learner.predict(final, df.select(pl.col(feats).cast(pl.Float32)).to_numpy(), n_final, feats)
    scored = df.select("a", "b", "p1", "n_cand_a").with_columns(p=pl.Series(cal.predict(p_raw).astype(np.float32)))
    s1 = pl.read_parquet(work_dir("test") / "s1.parquet", columns=S1_COLS)
    A = any_table(scored, s1)
    pa = learner.predict(a_final, A.select(ANY_ROBUST).to_numpy(), na_final, ANY_ROBUST)
    A = A.with_columns(p_any=pl.Series(a_cal.predict(pa).astype(np.float32)))
    scored = scored.join(A.select("a", "p_any"), on="a", how="left")
    scored.select("a", "b", "p1", "p", "p_any").write_parquet(work_dir("test") / "scored3.parquet")
    for sub, rule in rules.items():
        out = OUT / sub if sub else OUT
        out.mkdir(parents=True, exist_ok=True)
        chosen = decide.select(scored.filter(pl.col("p") >= 0.02), rule)
        _write_lists(s1, chosen, "matched_entity_ids", out / "matching_results.tsv")
        _write_lists(s1, df.select("a", "b"), "candidate_entity_ids", out / "candidate_pairs.tsv")
        per = s1.join(chosen.group_by("a").len("k"), left_on="iid", right_on="a", how="left").fill_null(0)
        stats = {c: {"s1": len(g), "empty_rate": float((g["k"] == 0).mean()), "mean_matches": float(g["k"].mean())}
                 for (c,), g in per.group_by("country")}
        summary = {"version": "v4" + (f"-{sub}" if sub else ""), "rule": rule, "test_pairs_scored": len(df),
                   "candidates_per_s1": len(df) / len(s1), "test_matches": len(chosen), "by_country": stats}
        (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
        log(f"stage3: predict [{sub or 'best rule'}] {len(chosen):,} matches, {len(df) / len(s1):.1f} candidates/S1; "
            f"{json.dumps(stats)}")
    record_time("stage3.predict", time.time() - t0, pairs=len(df))
