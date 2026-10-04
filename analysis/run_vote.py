"""Equal-vote ensembles of v4, v5a, v3 (all models weighted the same), checked on train OOF like run_ens.py.

  * majority: a model votes for a pair when its p >= t (same t for all); a pair is accepted with >= 2 of 3 votes;
    each S2/S3 record then goes to the accepted S1 with the highest mean p. t on a grid.
  * unanimous (3 of 3) and any (1 of 3) for reference.
  * mean: p = (p_v4 + p_v5a + p_v3) / 3, best threshold and v4@0.95's pair count ("matched").
Train subset and metric as run_ens.py (v5a's S1, exact macro F0.5 vs ground truth). Test files -> output/vote/<name>/.
Env: ENS_SRC = s3://bucket/prefix/ber-runs"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "src"))
import run_ens  # noqa: E402  (sets thread env)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from ber.config import OUT, log  # noqa: E402
from ber.data import int_id, load_gt_pairs, read_tsv  # noqa: E402
from ber.predict import _write_lists  # noqa: E402

M = run_ens.MODELS
VGRID = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]


def voted(df, t, k):
    """Pairs with >= k of 3 votes at threshold t, then one owner per record by mean p."""
    votes = sum((pl.col(f"p_{m}") >= t).cast(pl.Int8) for m in M)
    mean = sum(pl.col(f"p_{m}") for m in M) / len(M)
    return (df.filter(votes >= k).with_columns(p=mean).sort("p", descending=True)
            .unique("b", keep="first", maintain_order=True))


if __name__ == "__main__":
    t0 = time.time()
    d = run_ens.fetch(os.environ["ENS_SRC"])
    oofs = [pl.read_parquet(d / f"oof_{m}.parquet", columns=["a", "b", "p"]) for m in M]
    keep = oofs[1].select("a").unique().join(oofs[0].select("a").unique(), on="a")
    gt_pairs, n_true_all = load_gt_pairs()
    tr = run_ens.wide([o.join(keep, on="a") for o in oofs]).join(
        gt_pairs.with_columns(label=pl.lit(1, pl.Int8)), on=["a", "b"], how="left").with_columns(
        pl.col("label").fill_null(0))
    nt = n_true_all.join(keep, on="a").sort("a")
    idx = np.full(int(nt["a"].max()) + 1, -1, np.int64)
    idx[nt["a"].to_numpy()] = np.arange(len(nt))
    n_true = nt["n_true"].to_numpy()

    def f(win, t=-1.0):
        return run_ens.f05_curve(win, idx, n_true, [t])[0][1]

    v4w = run_ens.winners(tr, (1.0, 0.0, 0.0))
    n_ref_tr = int((v4w["p"] >= 0.95).sum())
    f_v4 = f(v4w, 0.95)
    rep = ["# Equal-vote ensembles of v4, v5a, v3 (train OOF, exact macro F0.5 vs ground truth)\n",
           f"Train subset {len(nt):,} S1. v4 alone @0.95: {f_v4:.5f} ({n_ref_tr:,} pairs).\n",
           "| rule | t | train F0.5 | Δ vs v4@0.95 | pairs (train) |", "|---|---|---|---|---|"]
    rows = []
    for k, kname in ((2, "majority 2/3"), (3, "unanimous 3/3"), (1, "any 1/3")):
        for t in VGRID:
            win = voted(tr, t, k)
            rows.append((kname, t, f(win), len(win)))
            log(f"vote: {kname} t={t}: {rows[-1][2]:.5f} ({len(win):,} pairs)")
    mw = run_ens.winners(tr, (1 / 3, 1 / 3, 1 / 3))
    curve = run_ens.f05_curve(mw, idx, n_true, run_ens.GRID)
    tb, fb, nb = max(curve, key=lambda r: r[1])
    t_match = float(mw["p"][n_ref_tr - 1])
    rows += [("mean 1/3 each (best t)", tb, fb, nb), ("mean 1/3 each (v4 count)", t_match, f(mw, t_match), n_ref_tr),
             ("mean 1/3 each", 0.95, f(mw, 0.95), int((mw["p"] >= 0.95).sum()))]
    for r in sorted(rows, key=lambda r: -r[2]):
        rep.append(f"| {r[0]} | {r[1]:.4f} | {r[2]:.5f} | {r[2] - f_v4:+.5f} | {r[3]:,} |")
    maj = [r for r in rows if r[0] == "majority 2/3"]
    t_maj_best = max(maj, key=lambda r: r[2])[1]
    t_maj_cnt = min(maj, key=lambda r: abs(r[3] - n_ref_tr))[1]           # closest to v4's pair count on train
    del tr, oofs

    te = run_ens.wide([pl.read_parquet(d / f"test_{m}.parquet", columns=["a", "b", "p"]) for m in M])
    ref = read_tsv(d / "ref_m.tsv")
    s1 = ref.select(iid=int_id("source1_entity_id"))
    ref_pairs = (ref.select(s="source1_entity_id", ids=pl.col("matched_entity_ids").str.split(",")).explode("ids")
                 .filter(pl.col("ids").is_not_null() & (pl.col("ids") != "")).select(a=int_id("s"), b=int_id("ids")))
    rc = read_tsv(d / "ref_c.tsv")
    ref_cand = (rc.select(s="source1_entity_id", ids=pl.col("candidate_entity_ids").str.split(",")).explode("ids")
                .filter(pl.col("ids").is_not_null() & (pl.col("ids") != "")).select(a=int_id("s"), b=int_id("ids")))
    n_ref = len(ref_pairs)
    mw_te = run_ens.winners(te, (1 / 3, 1 / 3, 1 / 3))
    outs = {f"majority_t{int(t_maj_best * 100):02d}": voted(te, t_maj_best, 2),
            f"majority_t{int(t_maj_cnt * 100):02d}": voted(te, t_maj_cnt, 2),
            "majority_t95": voted(te, 0.95, 2),
            "mean33_match": mw_te.filter(pl.col("p") >= float(mw_te["p"][n_ref - 1])),
            "mean33_t95": mw_te.filter(pl.col("p") >= 0.95)}
    rep.append(f"\nTest (v4 t0.95 = {n_ref:,} pairs):")
    for name, pred in outs.items():
        pred = pred.select("a", "b")
        out = OUT / "vote" / name
        out.mkdir(parents=True, exist_ok=True)
        _write_lists(s1, pred, "matched_entity_ids", out / "matching_results.tsv")
        _write_lists(s1, pl.concat([ref_cand, pred]), "candidate_entity_ids", out / "candidate_pairs.tsv")
        common = len(pred.join(ref_pairs, on=["a", "b"]))
        rep.append(f"- {name}: {len(pred):,} pairs; vs v4 t0.95 dropped {n_ref - common:,}, added {len(pred) - common:,}")
    (OUT / "vote" / "vote.md").write_text("\n".join(rep), encoding="utf-8")
    log("\n".join(rep))
    log(f"vote: done in {(time.time() - t0) / 60:.1f} min")
