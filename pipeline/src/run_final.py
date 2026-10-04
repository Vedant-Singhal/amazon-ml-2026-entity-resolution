"""Final step: blend the stage-3 probabilities of v4 and v5a (0.7 / 0.3) and write the submission.

  p = 0.7 * p_v4 + 0.3 * p_v5a          (a pair one model did not score counts 0 for that model)
  each S2/S3 record goes to its highest-p S1 (ties: lower S1 id)
  threshold t = the value that keeps exactly as many pairs as v4 alone at p >= 0.95 (the leaderboard-validated
                operating point), so the blend swaps v4's weakest pairs for better ones without becoming stricter
                or looser. On the submitted run t = 0.9157.
  candidate_pairs.tsv = v4's stage-1 shortlist (every pair the models scored that can reach t).

Inputs (from run.sh): $V4_WORK/test/scored3.parquet, $V5_WORK/test/scored3.parquet, $DATA/test/test_source1.tsv.
Optional check on train (FINAL_CHECK=1): out-of-fold macro F0.5 of the blend vs v4 alone at the same pair count,
on the S1 that v5a kept (its train set has test-like orphan density); needs $V4_WORK/model3/oof.parquet,
$V5_WORK/model3/oof.parquet and $DATA/train/train_ground_truth.tsv.
Only numpy + pyarrow."""
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

V4 = Path(os.environ.get("V4_WORK", "work_v4"))
V5 = Path(os.environ.get("V5_WORK", "work_v5a"))
DATA = Path(os.environ.get("DATA", "dataset"))
OUT = Path(os.environ.get("FINAL_OUT", "submission"))
W = (0.7, 0.3)
T_REF = 0.95
S3_OFFSET = 10**10
MASK = np.uint64(2**34 - 1)


def load(path, cols=("a", "b", "p")):
    t = pq.read_table(path, columns=list(cols))
    a, b = t["a"].to_numpy(), t["b"].to_numpy()
    assert a.max() < 2**30 and b.max() < 2**34
    keys = (a.astype(np.uint64) << np.uint64(34)) | b.astype(np.uint64)
    return keys, {c: t[c].to_numpy() for c in cols if c not in ("a", "b")}


def blend(parts, w):
    """parts: [(keys, p)] -> union keys (sorted) and the weighted sum (missing = 0)."""
    keys, inv = np.unique(np.concatenate([k for k, _ in parts]), return_inverse=True)
    p = np.zeros(len(keys))
    start = 0
    for (k, pi), wi in zip(parts, w):
        np.add.at(p, inv[start:start + len(k)], wi * pi.astype(np.float64))
        start += len(k)
    return keys, p


def winners(keys, p):
    """One S1 per S2/S3 record: its highest p (ties -> lower S1 id). Returns kept keys and p, sorted by key."""
    a, b = (keys >> np.uint64(34)).astype(np.int64), (keys & MASK).astype(np.int64)
    o = np.lexsort((a, -p, b))
    first = np.ones(len(o), bool)
    first[1:] = b[o][1:] != b[o][:-1]
    idx = np.sort(o[first])
    return keys[idx], p[idx]


def cut(p_win, n):
    """Threshold keeping the n highest pairs (ties at the cut are kept)."""
    return float(np.sort(p_win)[::-1][n - 1])


def sid(b):
    return f"S3-{b - S3_OFFSET}" if b >= S3_OFFSET else f"S2-{b}"


def write(path, s1_order, keys, col):
    a = (keys >> np.uint64(34)).astype(np.int64)
    b = (keys & MASK).astype(np.int64)
    lists = {}
    for ai, bi in zip(a.tolist(), b.tolist()):          # keys are sorted: a, then b (S2 before S3)
        lists.setdefault(ai, []).append(sid(bi))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s in s1_order:
            f.write(f"S1-{s}\t{','.join(lists.get(s, []))}\n")


def macro_f05(sel_keys, label_of, s1_ids, n_true):
    """Exact competition metric over s1_ids (sel_keys: chosen pairs; label_of: bool per chosen pair)."""
    pos = {s: i for i, s in enumerate(s1_ids)}
    a = (sel_keys >> np.uint64(34)).astype(np.int64)
    ai = np.array([pos[x] for x in a.tolist()], dtype=np.int64)
    k = np.bincount(ai, minlength=len(s1_ids)).astype(float)
    tp = np.bincount(ai[label_of], minlength=len(s1_ids)).astype(float)
    n = n_true.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(tp > 0, 1.25 * tp / (1.25 * tp + 0.25 * (n - tp) + (k - tp)), 0.0)
    f[(n == 0) & (k == 0)] = 1.0
    return float(f.mean())


def train_check():
    k4, d4 = load(V4 / "model3" / "oof.parquet", ("a", "b", "p", "label"))
    k5, d5 = load(V5 / "model3" / "oof.parquet", ("a", "b", "p", "label"))
    keep = np.intersect1d(np.unique(k5 >> np.uint64(34)), np.unique(k4 >> np.uint64(34)))
    m4, m5 = np.isin(k4 >> np.uint64(34), keep), np.isin(k5 >> np.uint64(34), keep)
    n_true = {}
    with open(DATA / "train" / "train_ground_truth.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s, _, ids = line.rstrip("\r\n").partition("\t")
            n_true[int(s[3:])] = sum(1 for x in ids.split(",") if x)
    s1_ids = keep.astype(np.int64).tolist()
    nt = np.array([n_true.get(s, 0) for s in s1_ids])
    lk, lab = np.concatenate([k4[m4], k5[m5]]), np.concatenate([d4["label"][m4], d5["label"][m5]])
    lk, first = np.unique(lk, return_index=True)
    lab = lab[first].astype(bool)
    res = {}
    for name, w in (("v4 alone", (1.0, 0.0)), ("0.7 v4 + 0.3 v5a", W)):
        keys, p = blend([(k4[m4], d4["p"][m4]), (k5[m5], d5["p"][m5])], w)
        wk, wp = winners(keys, p)
        res[name] = (wk, wp)
    n_ref = int((res["v4 alone"][1] >= T_REF).sum())
    for name, (wk, wp) in res.items():
        t = cut(wp, n_ref)
        sel = wk[wp >= t]
        f = macro_f05(sel, lab[np.searchsorted(lk, sel)], s1_ids, nt)
        print(f"train check [{len(s1_ids):,} S1]: {name:18s} t={t:.4f} pairs={len(sel):,} macro F0.5={f:.5f}")


if __name__ == "__main__":
    t0 = time.time()
    if os.environ.get("FINAL_CHECK") == "1":
        train_check()
    k4, d4 = load(V4 / "test" / "scored3.parquet")
    k5, d5 = load(V5 / "test" / "scored3.parquet")
    n_ref = int((winners(k4, d4["p"])[1] >= T_REF).sum())
    keys, p = blend([(k4, d4["p"]), (k5, d5["p"])], W)
    wk, wp = winners(keys, p)
    t = cut(wp, n_ref)
    sel = wk[wp >= t]
    s1_order = []
    with open(DATA / "test" / "test_source1.tsv", encoding="utf-8") as f:
        next(f)
        s1_order = [int(line.split("\t", 1)[0][3:]) for line in f]
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "matching_results.tsv", s1_order, sel, "matched_entity_ids")
    write(OUT / "candidate_pairs.tsv", s1_order, np.union1d(k4, sel), "candidate_entity_ids")
    print(f"final: v4 alone at {T_REF} keeps {n_ref:,} pairs; blend {W[0]}/{W[1]} threshold {t:.4f} keeps {len(sel):,}; "
          f"candidates {len(np.union1d(k4, sel)):,} ({len(np.union1d(k4, sel)) / len(s1_order):.2f} per S1) "
          f"-> {OUT}/ in {time.time() - t0:.0f}s")
