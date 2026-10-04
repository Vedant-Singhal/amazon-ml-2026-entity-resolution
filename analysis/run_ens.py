"""Ensemble v4 (stage 3) + v5a (stage 3) + v3 (stage 2) probabilities, weights and threshold checked on train OOF.

Train check: the S1s in v5a's OOF (v5a dropped 19% of train S1 -> test-like orphan density) that v4 also scored.
Per combo p = w4*p_v4 + w5*p_v5a + w3*p_v3 (unscored pair = 0), each S2/S3 record to its best S1, then threshold:
  * best threshold on the subset (exact macro F0.5, competition formula);
  * "matched": the threshold giving as many pairs as v4 alone at 0.95 (the LB-best operating point, 0.984312) —
    isolates better ranking from a different strictness, since OOF thresholds do not transfer to test.
Test: top combos written at the test threshold giving v4 t0.95's pair count (and at 0.95) -> output/ens/<name>/.
Env: ENS_SRC = s3://bucket/prefix/ber-runs"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "src"))
_HALF = str(max(1, int((os.cpu_count() or 2) * float(os.environ.get("BER_CPU_FRAC", "0.5")))))
for _v in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, _HALF)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from ber.config import OUT, WORK, log  # noqa: E402
from ber.data import int_id, load_gt_pairs, read_tsv  # noqa: E402
from ber.predict import _write_lists  # noqa: E402

FILES = {"oof_v4": "v4/work/model3/oof.parquet", "oof_v5a": "v5a/work/model3/oof.parquet",
         "oof_v3": "v3/work/model/oof.parquet", "test_v4": "v4/work/test/scored3.parquet",
         "test_v5a": "v5a/work/test/scored3.parquet", "test_v3": "v3/work/test/scored.parquet",
         "ref_m": "v4probe/output/probe_v4_t0950/matching_results.tsv",
         "ref_c": "v4probe/output/probe_v4_t0950/candidate_pairs.tsv"}
MODELS = ["v4", "v5a", "v3"]
COMBOS = [(1.0, 0.0, 0.0), (0.8, 0.2, 0.0), (0.7, 0.3, 0.0), (0.6, 0.4, 0.0), (0.5, 0.5, 0.0),
          (0.7, 0.2, 0.1), (0.6, 0.3, 0.1), (0.6, 0.2, 0.2), (0.5, 0.3, 0.2), (0.5, 0.4, 0.1),
          (0.8, 0.1, 0.1), (0.7, 0.15, 0.15), (0.9, 0.0, 0.1), (0.8, 0.0, 0.2)]
GRID = [round(x, 3) for x in np.arange(0.50, 0.991, 0.01)]
TOP = int(os.environ.get("ENS_TOP", "4"))


def fetch(src):
    import boto3
    bucket, prefix = src.removeprefix("s3://").split("/", 1)
    s3 = boto3.client("s3")
    d = WORK / "ens"
    d.mkdir(parents=True, exist_ok=True)
    for k, key in FILES.items():
        dst = d / (k + Path(key).suffix)
        if not dst.exists():
            s3.download_file(bucket, f"{prefix.rstrip('/')}/{key}", str(dst))
    return d


def wide(frames):
    """[(a, b, p)] per model -> one frame a, b, p_v4, p_v5a, p_v3 (0 where a model did not score the pair)."""
    out = None
    for m, f in zip(MODELS, frames):
        f = f.select("a", "b", pl.col("p").cast(pl.Float64).alias(f"p_{m}"))
        out = f if out is None else out.join(f, on=["a", "b"], how="full", coalesce=True)
    return out.with_columns(pl.col([f"p_{m}" for m in MODELS]).fill_null(0.0))


def winners(df, w):
    """Each S2/S3 record to its highest-scoring S1 -> (a, b, p[, label]) sorted by p desc."""
    p = sum(wi * pl.col(f"p_{m}") for wi, m in zip(w, MODELS))
    return (df.with_columns(p=p).sort("p", descending=True).unique("b", keep="first", maintain_order=True))


def f05_curve(win, idx, n_true, thresholds):
    """Exact macro F0.5 over the subset's S1 for each threshold (win: a, p, label; idx: a -> 0..n-1)."""
    ai = idx[win["a"].to_numpy()]
    p, lab = win["p"].to_numpy(), win["label"].to_numpy().astype(bool)
    n = n_true.astype(np.float64)
    res = []
    for t in thresholds:
        m = p >= t
        k = np.bincount(ai[m], minlength=len(n)).astype(np.float64)
        tp = np.bincount(ai[m & lab], minlength=len(n)).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(tp > 0, 1.25 * tp / (1.25 * tp + 0.25 * (n - tp) + (k - tp)), 0.0)
        f[(n == 0) & (k == 0)] = 1.0
        res.append((float(t), float(f.mean()), int(m.sum())))
    return res


def name_of(w):
    return "e_" + "_".join(f"{int(round(x * 100)):02d}" for x in w)


if __name__ == "__main__":
    t0 = time.time()
    d = fetch(os.environ["ENS_SRC"])
    rep = ["# Ensemble v4 + v5a + v3 (weights checked on train OOF)\n"]

    # ---- train: S1 subset scored by v5a (test-like orphan density) and v4
    oofs = [pl.read_parquet(d / f"oof_{m}.parquet", columns=["a", "b", "p", "label"]) for m in MODELS]
    keep = oofs[1].select("a").unique().join(oofs[0].select("a").unique(), on="a")
    gt_pairs, n_true_all = load_gt_pairs()
    tr = wide([o.join(keep, on="a") for o in oofs]).join(gt_pairs.with_columns(label=pl.lit(1, pl.Int8)), on=["a", "b"],
                                                        how="left").with_columns(pl.col("label").fill_null(0))
    nt = n_true_all.join(keep, on="a").sort("a")
    idx = np.full(int(nt["a"].max()) + 1, -1, np.int64)
    idx[nt["a"].to_numpy()] = np.arange(len(nt))
    n_true = nt["n_true"].to_numpy()
    log(f"ens: train subset {len(nt):,} S1, {len(tr):,} pairs ({(time.time() - t0) / 60:.1f} min)")

    v4w = winners(tr, COMBOS[0])
    n_ref_tr = int((v4w["p"] >= 0.95).sum())                   # v4 at 0.95 on the subset
    results = []
    for w in COMBOS:
        win = winners(tr, w)
        curve = f05_curve(win, idx, n_true, GRID)
        t_best, f_best, _ = max(curve, key=lambda r: r[1])
        t_match = float(win["p"][n_ref_tr - 1])               # same pair count as v4 @ 0.95
        _, f_match, _ = f05_curve(win, idx, n_true, [t_match])[0]
        _, f_095, k_095 = f05_curve(win, idx, n_true, [0.95])[0]
        results.append({"w": w, "t_best": t_best, "f_best": f_best, "t_match": t_match, "f_match": f_match,
                        "f_095": f_095, "k_095": k_095})
        log(f"ens: {name_of(w)} best {f_best:.5f} @ {t_best:.2f} | matched {f_match:.5f} @ {t_match:.4f} | "
            f"@0.95 {f_095:.5f} ({k_095:,} pairs)")
    base = results[0]
    rep += [f"Train subset: {len(nt):,} S1 (v5a's, test-like orphans), {len(tr):,} pairs; v4 @0.95 = {n_ref_tr:,} pairs.\n",
            "| combo (v4, v5a, v3) | best F0.5 (t) | matched-count F0.5 (t) | Δ vs v4 matched | F0.5 @0.95 |",
            "|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: -r["f_match"]):
        rep.append(f"| {r['w']} | {r['f_best']:.5f} ({r['t_best']:.2f}) | {r['f_match']:.5f} ({r['t_match']:.4f}) | "
                   f"{r['f_match'] - base['f_match']:+.5f} | {r['f_095']:.5f} |")
    del tr, oofs

    # ---- test: best combos at v4 t0.95's pair count, and at 0.95
    te = wide([pl.read_parquet(d / f"test_{m}.parquet", columns=["a", "b", "p"]) for m in MODELS])
    ref = read_tsv(d / "ref_m.tsv")
    s1 = ref.select(iid=int_id("source1_entity_id"))
    ref_pairs = (ref.select(s="source1_entity_id", ids=pl.col("matched_entity_ids").str.split(",")).explode("ids")
                 .filter(pl.col("ids").is_not_null() & (pl.col("ids") != "")).select(a=int_id("s"), b=int_id("ids")))
    rc = read_tsv(d / "ref_c.tsv")
    ref_cand = (rc.select(s="source1_entity_id", ids=pl.col("candidate_entity_ids").str.split(",")).explode("ids")
                .filter(pl.col("ids").is_not_null() & (pl.col("ids") != "")).select(a=int_id("s"), b=int_id("ids")))
    n_ref = len(ref_pairs)
    chk = winners(te, COMBOS[0]).filter(pl.col("p") >= 0.95)
    rep.append(f"\nTest: v4 alone @0.95 -> {len(chk):,} pairs (reference file {n_ref:,}; "
               f"common {len(chk.join(ref_pairs, on=['a', 'b'])):,})")
    ranked = sorted(results[1:], key=lambda r: -r["f_match"])[:TOP]
    for r in ranked:
        win = winners(te, r["w"])
        for tag, t in (("match", float(win["p"][n_ref - 1])), ("t095", 0.95)):
            pred = win.filter(pl.col("p") >= t).select("a", "b")
            out = OUT / "ens" / f"{name_of(r['w'])}_{tag}"
            out.mkdir(parents=True, exist_ok=True)
            _write_lists(s1, pred, "matched_entity_ids", out / "matching_results.tsv")
            _write_lists(s1, pl.concat([ref_cand, pred]), "candidate_entity_ids", out / "candidate_pairs.tsv")
            common = len(pred.join(ref_pairs, on=["a", "b"]))
            rep.append(f"- {out.name}: t={t:.4f}, {len(pred):,} pairs, vs v4 t0.95: dropped {n_ref - common:,}, "
                       f"added {len(pred) - common:,}")
    (OUT / "ens").mkdir(parents=True, exist_ok=True)
    (OUT / "ens" / "ensemble.md").write_text("\n".join(rep), encoding="utf-8")
    log("\n".join(rep))
    log(f"ens: done in {(time.time() - t0) / 60:.1f} min")
