"""Cluster completion ("twins"), label-free on test, measured exactly on train with v4's out-of-fold predictions.

A record left unmatched whose normalized name + address equals that of a record already matched to S1 s (and to no
other S1) is added to s — e.g. the S3 copy of a matched S2 record. Normalization per country, learned from each split's
own records: fold accents/case, mined abbreviations (ber/mining.py), drop the country's 40 most frequent name words and
60 most frequent address words; key = name words | first house number | address words.
Train: v4 OOF at threshold 0.95 (one owner per record) → add twins → exact macro F0.5 before/after, precision of adds.
Test: v4 t0.95 file (LB 0.984312) → add twins → output/twins/ (matching_results.tsv + candidate_pairs.tsv).
  TWIN_SRC = s3://bucket/prefix/ber-runs   (reads v4/work/model3/oof.parquet and v4probe/output/probe_v4_t0950/*)"""
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "src"))
_HALF = max(1, (os.cpu_count() or 2) // 2)          # 50% of the machine
for _v in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, str(_HALF))

import polars as pl  # noqa: E402

from ber import decide, mining  # noqa: E402
from ber.config import OUT, WORK, log  # noqa: E402
from ber.data import int_id, load_gt_pairs, load_split, read_tsv  # noqa: E402
from ber.predict import _write_lists  # noqa: E402

TOK = re.compile(r"\w+", re.UNICODE)
MAPS = {}


def fold(s):
    s = unicodedata.normalize("NFKD", (s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def toks_worker(rows):
    return [(TOK.findall(fold(n)), [TOK.findall(fold(p)) for p in (a or "").split(",")]) for n, a in rows]


def key_worker(rows):
    out = []
    for c, n, parts in rows:
        mn, ma, gn, ga = MAPS[c]
        name = sorted({mn.get(t, t) for t in n} - gn)
        num = next((t.lstrip("0") or "0" for p in parts for t in p if t.isdigit()), None)
        words = sorted({ma.get(t, t) for p in parts for t in p if t.isalpha()} - ga)
        out.append(" ".join(name) + "|" + num + "|" + " ".join(words) if (name and num and words) else None)
    return out


def pmap(fn, items, chunk=200_000):
    chunks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    with Pool(_HALF) as p:
        parts = p.map(fn, chunks)
    return [x for part in parts for x in part]


def keys_for(split):
    """-> (keys of S2/S3 records: iid, country, key), S1 frame (iid, country)."""
    t0 = time.time()
    s1, oth = load_split(split)
    oth = oth.select("iid", "name", "addr", "country")
    toks = pmap(toks_worker, list(zip(oth["name"].to_list(), oth["addr"].to_list())))
    countries = oth["country"].to_list()
    by_c = {}
    for i, c in enumerate(countries):
        by_c.setdefault(c, []).append(i)
    for c, idx in by_c.items():
        mined = mining.mine_by_country([(toks[i][0], []) for i in idx], [toks[i][1] for i in idx], [c] * len(idx),
                                       log=log)[c]
        mn, ma = mined["name"], mined["addr"]
        nf, af = Counter(), Counter()
        for i in idx:
            nf.update({mn.get(t, t) for t in toks[i][0]})
            af.update({ma.get(t, t) for p in toks[i][1] for t in p if t.isalpha()})
        MAPS[c] = (mn, ma, {w for w, _ in nf.most_common(40)}, {w for w, _ in af.most_common(60)})
    keys = pmap(key_worker, [(countries[i], toks[i][0], toks[i][1]) for i in range(len(countries))])
    del toks
    log(f"twins[{split}]: keys for {len(keys):,} records in {(time.time() - t0) / 60:.1f} min")
    return (oth.select("iid", "country").with_columns(key=pl.Series(keys, dtype=pl.String)),
            s1.select("iid", "country"))


def twins(pred, keys):
    """pred (a, b) -> unmatched records whose key equals a member's key of exactly one S1 → (a, b)."""
    k = keys.filter(pl.col("key").is_not_null()).rename({"iid": "b"})
    owner = (pred.join(k, on="b").group_by("key").agg(pl.col("a").unique().alias("owners"))
             .filter(pl.col("owners").list.len() == 1).select("key", a=pl.col("owners").list.first()))
    return (k.join(pred.select("b").unique(), on="b", how="anti").join(owner, on="key")
            .select("a", "b", "country").unique(["b"]))


def fetch(src):
    import boto3
    bucket, prefix = src.removeprefix("s3://").split("/", 1)
    files = {"oof.parquet": "v4/work/model3/oof.parquet",
             "best_matching.tsv": "v4probe/output/probe_v4_t0950/matching_results.tsv",
             "best_candidates.tsv": "v4probe/output/probe_v4_t0950/candidate_pairs.tsv"}
    WORK.mkdir(parents=True, exist_ok=True)
    for local, key in files.items():
        if not (WORK / local).exists():
            boto3.client("s3").download_file(bucket, f"{prefix.rstrip('/')}/{key}", str(WORK / local))


def as_pairs(path, col):
    m = read_tsv(path)
    return (m.select(s1="source1_entity_id", ids=pl.col(col).str.split(",")).explode("ids")
            .filter(pl.col("ids").is_not_null() & (pl.col("ids") != "")).select(a=int_id("s1"), b=int_id("ids")))


if __name__ == "__main__":
    fetch(os.environ["TWIN_SRC"])
    rep = ["# Cluster completion (twins)\n"]

    # ---- train: exact effect on v4 OOF at threshold 0.95
    keys_tr, s1_tr = keys_for("train")
    oof = pl.read_parquet(WORK / "oof.parquet", columns=["a", "b", "p"])
    pred = decide.one_owner(oof.filter(pl.col("p") >= 0.95)).select("a", "b")
    pairs, n_true = load_gt_pairs()
    add = twins(pred, keys_tr)
    lab = add.join(pairs.with_columns(t=pl.lit(1)), on=["a", "b"], how="left").with_columns(pl.col("t").fill_null(0))
    f0 = decide.macro_f05(pred, pairs, n_true)
    f1 = decide.macro_f05(pl.concat([pred, add.select("a", "b")]), pairs, n_true)
    rep.append(f"Train (v4 OOF, t0.95): {len(pred):,} predicted pairs; twins added {len(add):,}; precision "
               f"{lab['t'].mean():.4f}; macro F0.5 {f0:.5f} -> {f1:.5f} ({f1 - f0:+.5f})\n")
    for (c,), g in lab.group_by("country"):
        rep.append(f"  train {c}: {len(g):,} adds, precision {g['t'].mean():.4f}")
    log("\n".join(rep))
    del keys_tr, oof, pred, lab

    # ---- test: add twins to the best file
    keys_te, s1_te = keys_for("test")
    best = as_pairs(WORK / "best_matching.tsv", "matched_entity_ids")
    cand = as_pairs(WORK / "best_candidates.tsv", "candidate_entity_ids")
    add_te = twins(best, keys_te)
    rep.append(f"\nTest (v4 t0.95): {len(best):,} pairs; twins to add {len(add_te):,} on {add_te['a'].n_unique():,} S1")
    for (c,), g in add_te.group_by("country"):
        rep.append(f"  test {c}: {len(g):,} adds")
    out = OUT / "twins"
    out.mkdir(parents=True, exist_ok=True)
    s1_out = s1_te.select("iid", "country")
    _write_lists(s1_out, pl.concat([best, add_te.select("a", "b")]), "matched_entity_ids", out / "matching_results.tsv")
    _write_lists(s1_out, pl.concat([cand, add_te.select("a", "b")]), "candidate_entity_ids", out / "candidate_pairs.tsv")
    (OUT / "twins.md").write_text("\n".join(rep), encoding="utf-8")
    log("\n".join(rep))
