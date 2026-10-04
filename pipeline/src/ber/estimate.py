"""Stage `estimate` (smoke runs): project the full run's time and cost from the smoke timings.

Linear stages (prepare, features, decide, predict) scale with records or pairs. Two parts do not
scale linearly and are benchmarked here at FULL size on this machine:
  - FAISS IVF build + search on random unit vectors of the largest country's size (US train, 6.19M
    records) for the embedding and sketch dimensions;
  - XGBoost GPU seconds per tree per million rows on a random 4M-row table.
Embedding speed comes from the smoke run itself (docs/s on real texts, model load excluded).
Writes output/estimate.md + estimate.json.
"""
import json
import time

import numpy as np

from .config import FULL, OUT, P, WORK, log

PRICE = {"ml.g6e.8xlarge": 5.66, "ml.g6.16xlarge": 4.246}   # SageMaker training, us-east-1, $/h


def _bench_ann(d: int, n_base: int, n_query: int, k: int) -> dict:
    """Time block.knn (GPU IVF) at full size on random unit vectors: build + search."""
    from .block import knn
    rng = np.random.default_rng(0)

    def unit(n):
        x = rng.standard_normal((n, d), dtype=np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x.astype(np.float16)
    base, query = unit(n_base), unit(n_query)
    t = time.time()
    knn(base, query, k)
    return {"d": d, "n_base": n_base, "n_query": n_query, "k": k, "seconds": time.time() - t}


def _bench_xgb(rows: int = 4_000_000, cols: int = 110, rounds: int = 60) -> float:
    import xgboost as xgb
    from . import learner
    rng = np.random.default_rng(0)
    X = rng.random((rows, cols), dtype=np.float32)
    y = (X[:, 0] + 0.3 * rng.random(rows) > 0.8).astype(np.float32)
    d = xgb.QuantileDMatrix(X, y, max_bin=256)
    xgb.train(learner.params(0.05), d, num_boost_round=5)          # warm-up
    t = time.time()
    xgb.train(learner.params(0.05), d, num_boost_round=rounds)
    return (time.time() - t) / rounds / (rows / 1e6)               # s per tree per 1M rows


def run(instance: str = "ml.g6e.8xlarge") -> None:
    rows = json.loads((WORK / "timing.json").read_text())
    T = {}
    for r in rows:
        T.setdefault(r["stage"], r)
    smoke_recs = {s: T[f"block.{s}"]["s1"] + T[f"block.{s}"]["oth"] for s in ("train", "test")}
    full_recs = {s: FULL[s]["s1"] + FULL[s]["oth"] for s in ("train", "test")}
    r_rec = {s: full_recs[s] / smoke_recs[s] for s in smoke_recs}
    r_s1 = {s: FULL[s]["s1"] / T[f"block.{s}"]["s1"] for s in smoke_recs}
    log(f"estimate: scale records {r_rec}, S1 {r_s1}")

    est = {}
    prep = [r for r in rows if r["stage"].startswith("prepare")]
    est["prepare"] = sum(r["seconds"] * r_rec[r["stage"].split(".")[1]] for r in prep)
    nm = len(P.emb_models)                      # docs_per_s counts texts x models, all models together
    est["embed"] = sum(full_recs[s] * nm / T[f"embed.{s}"]["docs_per_s"] + T[f"embed.{s}"]["load_s"] for s in full_recs)

    emb_dim = int(np.load(WORK / "train" / "emb_s1.npy", mmap_mode="r").shape[1])
    # largest train country (US): forward = 1.32M S1 vs 6.19M records, reverse = 6.19M vs 1.32M
    bench = [_bench_ann(emb_dim, 6_186_873, 1_323_633, P.dense_k_fwd),
             _bench_ann(emb_dim, 1_323_633, 6_186_873, P.dense_k_rev),
             _bench_ann(P.sketch_dim, 6_186_873, 1_323_633, P.c3_k_fwd),
             _bench_ann(P.sketch_dim, 1_323_633, 6_186_873, P.c3_k_rev)]
    fb = bench
    log(f"estimate: GPU IVF bench {bench}")
    # US train is 7.5M of 24.2M records; all country-splits together ~3.2x the US-train work
    us = FULL["train"]["s1"] * 0.6 + FULL["train"]["oth"] * 0.6
    scale = (sum(full_recs.values())) / us
    dims = [int(np.load(WORK / "train" / f"emb{'' if i == 0 else i + 1}_s1.npy", mmap_mode="r").shape[1])
            for i in range(nm)]
    dense = (bench[0]["seconds"] + bench[1]["seconds"]) * sum(d / emb_dim for d in dims)   # cost ~ dimension
    ann_s = scale * (dense + 2 * (bench[2]["seconds"] + bench[3]["seconds"]))
    lin = sum((T[f"block.{s}"]["keys_s"] + T[f"block.{s}"]["cos_s"] + T[f"block.{s}"]["sketch_s"] * 0.5) * r_rec[s]
              for s in r_rec)
    est["block"] = ann_s + lin

    rate = _bench_xgb()
    log(f"estimate: XGBoost {rate * 1000:.1f} ms per tree per 1M rows")
    pool_pairs = {s: T[f"block.{s}"]["pairs"] * r_s1[s] for s in r_s1}
    est["stage1 features"] = sum(T[f"stage1.features.{s}"]["seconds"] * r_s1[s] for s in r_s1)
    s1_rows = T["stage1.train"]["rows"] * r_s1["train"]
    est["stage1 model (5 folds, <=600 trees)"] = 5 * 0.8 * s1_rows / 1e6 * rate * 600 + T["stage1.test"]["seconds"] * r_s1["test"]
    per_pair = T["features.train"]["seconds"] / max(T["features.train"]["pairs"], 1)
    sl_per_s1 = T["features.train"]["pairs"] / T["features.train"]["s1"]
    est["features"] = per_pair * sl_per_s1 * (FULL["train"]["s1"] + FULL["test"]["s1"])
    sl_rows = T["train.cv"]["rows"] * r_s1["train"]
    trees = P.rounds2
    est[f"train (5 folds, <= {trees} trees)"] = 5 * 0.8 * sl_rows / 1e6 * rate * trees
    est["train final model"] = sl_rows / 1e6 * rate * trees * 1.1
    est["any-match model + decide"] = (T["train.any"]["seconds"] + T["train.decide"]["seconds"]) * r_s1["train"]
    est["predict"] = 120 + sl_per_s1 * FULL["test"]["s1"] / 1e6 * rate * P.rounds2 * 0.2   # GPU predict
    abl_rows = sl_rows * P.abl_frac
    est["ablations (A3 9 variants x 3 folds + A4 LightGBM + A6)"] = (9 * 3 * 0.67 * abl_rows / 1e6 * rate * 1500
                                                                    + 3 * 0.67 * abl_rows / 1e6 * 0.018 * 1500 + 600)

    total = sum(est.values())
    lines = [f"# v3 full-run estimate (from smoke run, {instance})\n",
             f"Smoke: {T['block.train']['s1']:,} train S1 (x{r_s1['train']:.0f} to full). FAISS and XGBoost "
             "benchmarked at full size on this machine; other stages scaled linearly.\n",
             "| stage | minutes |", "|---|---|"]
    lines += [f"| {k} | {v / 60:.0f} |" for k, v in est.items()]
    lines += [f"| **total** | **{total / 60:.0f}** ({total / 3600:.1f} h) |",
              f"\nCost ≈ **${total / 3600 * PRICE.get(instance, 5.66):.0f}** at ${PRICE.get(instance, 5.66)}/h. "
              f"Pool {pool_pairs['train'] / FULL['train']['s1']:.0f} pairs/S1; shortlist "
              f"{sl_rows / FULL['train']['s1']:.1f} pairs/S1. Training rows use the cap ({trees} trees): an upper bound.\n",
              f"FAISS bench: {json.dumps(fb)}; XGBoost {rate * 1000:.1f} ms/tree/1M rows.\n"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "estimate.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "estimate.json").write_text(json.dumps({"minutes": {k: v / 60 for k, v in est.items()},
                                                   "total_h": total / 3600, "ann": fb, "xgb_rate": rate}, indent=2))
    log("\n".join(lines))
