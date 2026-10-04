"""Gradient-boosted trees for both stages: XGBoost (Apache-2.0) on the GPU, CPU fallback.
v2 trained LightGBM on the CPU (102 min); the v3 machine has half the CPU cores and a 3x
faster GPU, so trees move to the GPU. `cv` gives leakage-free out-of-fold predictions with the
same GroupKFold-by-S1 folds as v2 (fold = hash(a, seed 13) % 5)."""
import time

import numpy as np
import xgboost as xgb

from .config import P, log

_DEVICE = None


def device() -> str:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = "cpu"
        try:
            import torch
            if torch.cuda.is_available() and xgb.build_info().get("USE_CUDA"):
                xgb.train({"device": "cuda", "tree_method": "hist"},
                          xgb.DMatrix(np.random.rand(64, 3), label=np.random.rand(64) > 0.5), 2)
                _DEVICE = "cuda"
        except Exception as e:  # noqa: BLE001 - any GPU problem -> CPU
            log(f"learner: GPU unavailable for XGBoost ({type(e).__name__}: {e}); using CPU")
    return _DEVICE


def params(lr: float) -> dict:
    return {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
            "device": device(), "learning_rate": lr, "max_depth": 9, "min_child_weight": 5,
            "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0, "max_bin": 256,
            "seed": P.seed, "nthread": P.workers}


def fit(X, y, w, X_es, y_es, lr: float, rounds: int, feats: list, early_stop: int = None):
    t0 = time.time()
    dtr = xgb.QuantileDMatrix(X, y, weight=w, feature_names=feats, max_bin=256)
    evals = []
    if X_es is not None:
        evals = [(xgb.QuantileDMatrix(X_es, y_es, ref=dtr, feature_names=feats), "es")]
    bst = xgb.train(params(lr), dtr, num_boost_round=rounds, evals=evals,
                    early_stopping_rounds=(early_stop or P.early_stop) if evals else None,
                    verbose_eval=500)
    n = (bst.best_iteration + 1) if evals else rounds
    rate = (time.time() - t0) / max(n, 1) / max(len(X) / 1e6, 1e-6)
    log(f"learner: {len(X):,} rows, {n} trees, {time.time() - t0:.0f}s ({rate * 1000:.1f} ms/round/1M rows)")
    return bst, n, rate


def predict(bst, X, n_trees: int, feats: list) -> np.ndarray:
    out = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), 5_000_000):
        d = xgb.DMatrix(X[s:s + 5_000_000], feature_names=feats)
        out[s:s + 5_000_000] = bst.predict(d, iteration_range=(0, n_trees))
    return out


def cv(X, y, w, fold, keep, lr, rounds, feats, tag: str, n_folds: int = None, es_max: int = 2_000_000):
    """-> (oof, models, n_trees per fold, sec/round/1M rows). Fold k: train on keep & fold != k,
    early-stop on a slice of fold k, predict all of fold k."""
    n_folds = n_folds or P.cv_folds
    rng = np.random.default_rng(P.seed)
    oof = np.zeros(len(X), dtype=np.float32)
    models, iters, rates = [], [], []
    for k in range(n_folds):
        tr = np.flatnonzero((fold != k) & keep)
        va = np.flatnonzero(fold == k)
        es = rng.choice(va, size=min(len(va), es_max), replace=False)
        bst, n, rate = fit(X[tr], y[tr], None if w is None else w[tr], X[es], y[es], lr, rounds, feats)
        oof[va] = predict(bst, X[va], n, feats)
        models.append(bst)
        iters.append(n)
        rates.append(rate)
        log(f"{tag}: fold {k} trees={n}")
    return oof, models, iters, float(np.mean(rates))


def gain(models: list, feats: list) -> dict:
    tot = {f: 0.0 for f in feats}
    for m in models:
        for f, g in m.get_score(importance_type="total_gain").items():
            tot[f] = tot.get(f, 0.0) + g
    s = sum(tot.values()) or 1.0
    return dict(sorted(((f, round(g / s, 4)) for f, g in tot.items()), key=lambda kv: -kv[1]))
