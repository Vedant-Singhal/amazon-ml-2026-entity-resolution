"""Stage `train`: the stage-2 matcher on the shortlist, leakage-free, then final models on ALL data.

  1. Folds = hash(S1 id) % 5 (GroupKFold by entity; same folds as v2 and stage 1, so p1 is
     out-of-fold everywhere it is used as a feature — leak-free stacking).
  2. XGBoost (GPU), lr 0.05, up to 5,000 trees with early stopping (v2 hit its 3,000 cap).
  3. OOF -> isotonic calibration.
  4. "Has any match" model per S1 (aggregates of its candidates' OOF scores, name uniqueness,
     address presence), also OOF + calibrated: P(the empty prediction is right).
  5. Decision rule tuned on OOF with the exact competition metric (decide.tune).
  6. Final models on 100% of train with mean best trees x 1.1.
Outputs work/model/: final.ubj, any_final.ubj, calibrators, decision.json, report.json, oof.parquet.
"""
import json
import pickle
import time

import numpy as np
import polars as pl

from . import decide, learner
from .config import P, log, record_time, work_dir
from .data import fold_of, gt_for

ID_COLS = ["a", "b", "label"]
ANY_FEATS = ["p_max", "p_2nd", "p_sum", "n_sl", "n_p50", "p1_max", "n_pool", "has_addr", "lu_s1_core",
             "lu_oth_core", "lu_s1_addr"]
S1_COLS = ["iid", "country", "has_addr", "u_s1_core", "u_oth_core", "u_s1_addr"]


def load_features(split: str) -> pl.DataFrame:
    return pl.read_parquet(work_dir(split, "feats") / "part-*.parquet")


def feature_names(df: pl.DataFrame, drop=()) -> list:
    return [c for c in df.columns if c not in ID_COLS and c not in drop]


def any_table(scored: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    """Per-S1 features for the 'has any match' model. scored: (a, b, p, p1, n_cand_a)."""
    agg = (scored.group_by("a")
           .agg(p_max=pl.col("p").max(), p_2nd=pl.col("p").sort(descending=True).slice(1, 1).first(),
                p_sum=pl.col("p").sum(), n_sl=pl.len(), n_p50=(pl.col("p") >= 0.5).sum(),
                p1_max=pl.col("p1").max(), n_pool=pl.col("n_cand_a").first()))
    t = s1.select(a="iid", has_addr=pl.col("has_addr").cast(pl.Float32),
                  lu_s1_core=pl.col("u_s1_core").log1p(), lu_oth_core=pl.col("u_oth_core").log1p(),
                  lu_s1_addr=pl.col("u_s1_addr").log1p())
    return t.join(agg, on="a", how="left").with_columns(pl.col(ANY_FEATS).cast(pl.Float32).fill_null(0.0))


def run() -> None:
    md = work_dir("model")
    if (md / "report.json").exists():
        log("train: cached")
        return
    t0 = time.time()
    df = load_features("train")
    feats = feature_names(df)
    log(f"train: {len(df):,} shortlist pairs, {df['label'].sum():,} positives, {len(feats)} features")
    X = df.select(pl.col(feats).cast(pl.Float32)).to_numpy()
    y = df["label"].to_numpy()
    fold = df.select(fold_of("a")).to_series().to_numpy()
    oof, models, iters, rate = learner.cv(X, y, None, fold, np.ones(len(df), bool), P.lr2, P.rounds2,
                                          feats, "stage2")
    cal = decide.fit_calibrator(oof, y)
    p = cal.predict(oof).astype(np.float32)
    record_time("train.cv", time.time() - t0, rows=len(df), trees=iters, sec_per_round_per_1M=rate)

    # "has any match" model per S1
    t1 = time.time()
    s1 = pl.read_parquet(work_dir("train") / "s1.parquet", columns=S1_COLS)
    pairs, n_true = gt_for(s1["iid"])
    scored = df.select("a", "b", "p1", "n_cand_a").with_columns(p=pl.Series(p))
    A = any_table(scored, s1).join(n_true, on="a", how="left").with_columns(pl.col("n_true").fill_null(0))
    XA = A.select(ANY_FEATS).to_numpy()
    yA = (A["n_true"] > 0).cast(pl.Int8).to_numpy()
    foldA = A.select(fold_of("a")).to_series().to_numpy()
    a_oof, a_models, a_iters, _ = learner.cv(XA, yA, None, foldA, np.ones(len(A), bool), 0.05, 2000,
                                             ANY_FEATS, "any")
    a_cal = decide.fit_calibrator(a_oof, yA)
    A = A.with_columns(p_any=pl.Series(a_cal.predict(a_oof).astype(np.float32)))
    record_time("train.any", time.time() - t1, rows=len(A))

    # calibration + decision rule on OOF only
    t1 = time.time()
    o = scored.select("a", "b", "p").join(A.select("a", "p_any"), on="a", how="left")
    o.with_columns(p_raw=pl.Series(oof), label=df["label"]).write_parquet(md / "oof.parquet")
    A.select("a", "p_any", "n_true").write_parquet(md / "any_oof.parquet")
    tuned = decide.tune(o.filter(pl.col("p") >= 0.02), pairs, n_true)
    rule = tuned["best"]
    (md / "decision.json").write_text(json.dumps(tuned, indent=2))
    chosen = decide.select(o.filter(pl.col("p") >= 0.02), rule)
    by_country = {}
    for (c,), g in s1.group_by("country"):
        nt = n_true.join(g.select(a="iid"), on="a")
        by_country[c] = decide.macro_f05(chosen.join(nt.select("a"), on="a"), pairs, nt)
    s1_fold = n_true.with_columns(fold=fold_of("a"))
    per_fold = []
    for k in range(P.cv_folds):
        nt = s1_fold.filter(pl.col("fold") == k).select("a", "n_true")
        per_fold.append({"fold": k, "s1": len(nt),
                         "macro_f05": decide.macro_f05(chosen.join(nt.select("a"), on="a"), pairs, nt)})
    scores = np.array([r["macro_f05"] for r in per_fold])
    reach = pairs.join(df.select("a", "b"), on=["a", "b"]).height / max(len(pairs), 1)
    best_by_kind = {}
    for r in tuned["all"]:
        best_by_kind.setdefault(r["kind"] + ("+any_min" if r.get("any_min") else ""), r)
    report = {"oof_macro_f05": rule["macro_f05"], "fold_mean": float(scores.mean()), "fold_std": float(scores.std()),
              "per_fold": per_fold, "by_country": by_country, "rule": rule, "best_by_rule_kind": best_by_kind,
              "shortlist_recall": reach, "shortlist_per_s1": len(df) / len(s1), "n_pairs": len(df),
              "positives": int(y.sum()), "trees": iters, "any_trees": a_iters,
              "feature_gain": learner.gain(models, feats)}
    log(f"train: OOF macro F0.5 = {rule['macro_f05']:.4f} folds {np.round(scores, 4).tolist()} {by_country}")
    record_time("train.decide", time.time() - t1)

    # final models on all training data
    t1 = time.time()
    n_final = int(np.mean(iters) * 1.1)
    final, _, _ = learner.fit(X, y, None, None, None, P.lr2, n_final, feats)
    final.save_model(str(md / "final.ubj"))
    na_final = int(np.mean(a_iters) * 1.1)
    a_final, _, _ = learner.fit(XA, yA, None, None, None, 0.05, na_final, ANY_FEATS)
    a_final.save_model(str(md / "any_final.ubj"))
    pickle.dump({"pair": cal, "any": a_cal}, open(md / "calibrators.pkl", "wb"))
    report.update(final_trees=n_final, any_final_trees=na_final, feats=feats)
    (md / "report.json").write_text(json.dumps(report, indent=2))
    record_time("train.final", time.time() - t1, rows=len(df), trees=n_final)
    log(f"train: final models saved ({n_final} / {na_final} trees)")
