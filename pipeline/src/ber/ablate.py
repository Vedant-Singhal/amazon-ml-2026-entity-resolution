"""Stage `ablate`: the v3 ablation report (output/ablate.md), budgeted to ~1 hour.

  A0  embedder + index choice      separate job (ablate_retrieval.py), linked in the report
  A1  routes                        recall of each retrieval route, pairs it adds, pairs only it finds
  A2  candidate size                shortlist recall vs mean candidates per S1 for K x tau
  A3  feature groups                stage-2 retrained without each group (and v2 features only) on an
                                    abl_frac sample of S1 (3-fold OOF, lr 0.1): macro F0.5 delta
  A4  learner                       XGBoost-GPU vs LightGBM-CPU (v2's learner) on the same sample
  A5  decision rule                 best rule of each kind on the full OOF
  A6  leave-one-country-out         train on one country, score the other with that country's
                                    calibration + rule: how well rules transfer to unseen France
"""
import json
import time

import numpy as np
import polars as pl

from . import decide, learner
from .config import OUT, P, log, record_time, work_dir
from .data import gt_for
from .features import FEATURE_GROUPS
from .train import feature_names, load_features

FAST_RULES = ([{"kind": "thresh", "t": t, "floor": 0.0, "one_owner": True} for t in np.arange(0.4, 0.91, 0.05).round(2).tolist()]
              + [{"kind": "expf", "miss": m, "floor": 0.05, "one_owner": True} for m in (0.0, 0.1)])


def _table(df: pl.DataFrame) -> str:
    cols = df.columns
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.iter_rows()]
    return "\n".join(lines)


def _score(oof, y, df, pairs, n_true, rules=FAST_RULES):
    cal = decide.fit_calibrator(oof, y)
    o = df.select("a", "b").with_columns(p=pl.Series(cal.predict(oof).astype(np.float32)))
    best = decide.tune(o.filter(pl.col("p") >= 0.02), pairs, n_true, rules)["best"]
    return best, cal


def _lgb_cv(X, y, fold, feats, n_folds):
    import lightgbm as lgb
    prm = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=500,
               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
               seed=P.seed, num_threads=P.workers)
    oof = np.zeros(len(X), dtype=np.float32)
    for k in range(n_folds):
        tr, va = fold != k, fold == k
        m = lgb.train(prm, lgb.Dataset(X[tr], y[tr], feature_name=feats), num_boost_round=1500,
                      valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
    return oof


def run() -> None:
    t0 = time.time()
    rep = ["# v3 ablations\n"]
    wd = work_dir("train")

    # A1 routes
    bs = json.loads((wd / "block_stats.json").read_text())
    rep.append(f"## A1 · Retrieval routes (train)\nPool: **{bs['pool_per_s1']:.1f} candidates/S1**, union recall "
               f"**{bs['recall_union']:.4f}**, S1 with every match found {bs['entities_all_found']:.4f}.\n")
    rep.append(_table(pl.DataFrame([{"route": k, **v} for k, v in bs["routes"].items()])))

    # A2 candidate size
    sl = json.loads((work_dir("model1") / "shortlist.json").read_text())
    rep.append(f"\n## A2 · Candidate size (stage-1 shortlist)\nChosen rule **K={sl['rule']['K']}, tau={sl['rule']['tau']}**: "
               f"{sl['rule']['per_s1']:.2f} candidates/S1, recall {sl['rule']['recall']:.4f} "
               f"(pool {sl['pool_per_s1']:.1f}/S1, recall {sl['pool_recall']:.4f}; target {sl['target_recall']:.4f}).\n")
    curve = pl.DataFrame(sl["curve"])
    rep.append(_table(curve.filter(pl.col("tau").is_in([0.0, 0.01, 0.05])).sort("tau", "K")))

    # A3 / A4 / A6 on a sample of S1
    df = load_features("train")
    df = df.filter((pl.col("a").hash(P.seed + 1) % 1000) < int(P.abl_frac * 1000))
    s1 = pl.read_parquet(wd / "s1.parquet", columns=["iid", "country"])
    s1 = s1.filter((pl.col("iid").hash(P.seed + 1) % 1000) < int(P.abl_frac * 1000))
    pairs, n_true = gt_for(s1["iid"])
    y = df["label"].to_numpy()
    fold = (df.select((pl.col("a").hash(P.seed) % 3).cast(pl.Int32)).to_series().to_numpy())
    all_feats = feature_names(df)
    new = sorted({f for g in FEATURE_GROUPS.values() for f in g})
    variants = {"all features": []} | {f"- {g}": fs for g, fs in FEATURE_GROUPS.items()} | {"v2 features only": new}
    rows, oof_all = [], None
    log(f"ablate: {len(df):,} pairs from {len(s1):,} S1 ({P.abl_frac:.0%} sample)")
    for name, drop in variants.items():
        if drop and not set(drop) & set(all_feats):
            continue                                    # group absent in this run (e.g. one embedder)
        t = time.time()
        feats = [f for f in all_feats if f not in drop]
        X = df.select(pl.col(feats).cast(pl.Float32)).to_numpy()
        oof, _, iters, _ = learner.cv(X, y, None, fold, np.ones(len(df), bool), 0.1, 3000, feats, f"A3 {name}",
                                      n_folds=3, es_max=500_000)
        best, _ = _score(oof, y, df, pairs, n_true)
        rows.append({"variant": name, "n_features": len(feats), "macro_f05": best["macro_f05"],
                     "rule": f"{best['kind']} {best.get('t', best.get('miss'))}", "trees": int(np.mean(iters)),
                     "minutes": round((time.time() - t) / 60, 1)})
        if name == "all features":
            oof_all, X_all, feats_all, iters_all = oof, X, feats, iters
        log(f"ablate A3: {rows[-1]}")
    base = rows[0]["macro_f05"]
    a3 = pl.DataFrame(rows).with_columns(delta=pl.col("macro_f05") - base)
    rep.append(f"\n## A3 · Feature groups ({P.abl_frac:.0%} of S1, 3-fold OOF, lr 0.1)\n"
               "delta < 0 = removing the group hurts (the group helps).\n")
    rep.append(_table(a3))

    # A4 learner
    t = time.time()
    try:
        oof_l = _lgb_cv(X_all, y, fold, feats_all, 3)
        best_l, _ = _score(oof_l, y, df, pairs, n_true)
        a4 = [{"learner": "XGBoost (GPU)", "macro_f05": base, "minutes": rows[0]["minutes"]},
              {"learner": "LightGBM (CPU, v2 params)", "macro_f05": best_l["macro_f05"],
               "minutes": round((time.time() - t) / 60, 1)},
              {"learner": "mean of both", "macro_f05": _score((oof_all + oof_l) / 2, y, df, pairs, n_true)[0]["macro_f05"],
               "minutes": 0.0}]
        rep.append("\n## A4 · Learner (same sample, features, folds)\n")
        rep.append(_table(pl.DataFrame(a4)))
    except Exception as e:  # noqa: BLE001 - optional comparison
        rep.append(f"\n## A4 · Learner\nskipped: {type(e).__name__}: {e}\n")

    # A6 leave-one-country-out
    rows = []
    cmap = s1.select(a="iid", country="country")
    ctry = df.select("a").join(cmap, on="a", how="left")["country"].to_numpy()
    for src in ("US", "India"):
        for dst in ("US", "India"):
            if src == dst:
                continue
            ms, mt = ctry == src, ctry == dst
            if not ms.any() or not mt.any():
                continue
            nt_s = n_true.join(cmap.filter(pl.col("country") == src).select("a"), on="a")
            nt_t = n_true.join(cmap.filter(pl.col("country") == dst).select("a"), on="a")
            rule_s, cal_s = _score(oof_all[ms], y[ms], df.filter(pl.Series(ms)), pairs, nt_s)
            n = int(np.mean(iters_all))
            bst, _, _ = learner.fit(X_all[ms], y[ms], None, None, None, 0.1, n, feats_all)
            p = cal_s.predict(learner.predict(bst, X_all[mt], n, feats_all)).astype(np.float32)
            o = df.filter(pl.Series(mt)).select("a", "b").with_columns(p=pl.Series(p))
            transfer = decide.macro_f05(decide.select(o.filter(pl.col("p") >= 0.02), rule_s), pairs, nt_t)
            own, _ = _score(oof_all[mt], y[mt], df.filter(pl.Series(mt)), pairs, nt_t)
            rows.append({"train on": src, "score on": dst, "transfer (src model+rule)": transfer,
                         "in-country OOF": own["macro_f05"], "gap": transfer - own["macro_f05"]})
    if rows:
        rep.append("\n## A6 · Leave-one-country-out (proxy for unseen France)\n")
        rep.append(_table(pl.DataFrame(rows)))

    # A5 decision rules (full OOF)
    md = work_dir("model")
    if (md / "report.json").exists():
        r = json.loads((md / "report.json").read_text())
        rep.append("\n## A5 · Decision rule (full train OOF)\n")
        rep.append(_table(pl.DataFrame([{"kind": k, "macro_f05": v["macro_f05"],
                                         "params": json.dumps({x: v[x] for x in v if x not in ("kind", "macro_f05")})}
                                        for k, v in r["best_by_rule_kind"].items()])))
        rep.append(f"\nFull OOF macro F0.5 **{r['oof_macro_f05']:.4f}** (folds {r['fold_mean']:.4f} ± {r['fold_std']:.4f}); "
                   f"by country {json.dumps({k: round(v, 4) for k, v in r['by_country'].items()})}.\n")
        rep.append("\n### Feature gain (top 30)\n")
        rep.append(_table(pl.DataFrame([{"feature": k, "gain": v} for k, v in list(r["feature_gain"].items())[:30]])))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ablate.md").write_text("\n".join(rep), encoding="utf-8")
    record_time("ablate", time.time() - t0, pairs=len(df))
    log(f"ablate: wrote {OUT / 'ablate.md'} in {(time.time() - t0) / 60:.1f} min")
