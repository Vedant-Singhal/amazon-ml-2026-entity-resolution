"""Stage `features`: full pairwise features on the SHORTLIST (stage-1 output), a = S1, b = S2/S3.

Country is not a feature (France is unseen); nothing is country-specific. Groups (FEATURE_GROUPS
names are used by the feature-group ablation A3):
  v2 base     fuzzy name/address ratios, IDF overlaps, aliases, initials, numbers (v2's 44 features)
  stage 1     retrieval ranks/cosines/route flags + cheap similarities + p1 (carried from stage 1)
  uniq        how many S1 / records share the core or full name, name information content  (E1, E6)
  numbers     main-number relation (equal / truncated / one-digit edit / near / different),
              coverage, extra numbers, number-street frequency, punishment tokens         (E2, E7)
  trade_addr  address char-3-gram route + cosine, co-location counts, address keys        (E3, E8)
  name_c3     name char-3-gram route + cosine, exact core keys                            (E1)
  phonetic    metaphone overlap, consonant skeleton, sorted-core key, native script       (E4)
  typo        invented (garbled) words and whether an edit explains them, Levenshtein     (E5)
  listwise    p1 rank / gap to best / gap to second / mutual best / shortlist sizes       (E6, E7)
Output work/<split>/feats/part-*.parquet (+ `label` on train).
"""
import gc
import math
import time
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .config import P, log, record_time, work_dir

FIELDS = ["iid", "name_n", "core", "compact", "ocr", "acr", "stop", "aliases", "addr_n", "addr_core",
          "alpha", "nums", "main_nums", "first_num", "has_addr", "country", "skel", "meta", "sorted_core",
          "u_s1_name", "u_oth_numst"]
FEATURE_GROUPS = {
    "uniq": ["lu_s1_core_a", "lu_oth_core_a", "lu_s1_core_b", "lu_oth_core_b", "lu_s1_name_a",
             "core_idf_sum_a", "core_idf_min_a"],
    "numbers": ["num_rel", "num_main_cov", "num_extra_a", "num_extra_b", "lu_oth_numst_b",
                "pun_max_a", "pun_max_b", "pun_n"],
    "trade_addr": ["a3cos", "a3_gap_a", "a3_rank_fwd", "a3_rank_rev", "lu_s1_addr_a", "lu_oth_addr_b",
                   "k_akey", "k_numset"],
    "name_c3": ["n3cos", "n3_gap_a", "n3_rank_b", "n3_rank_fwd", "n3_rank_rev", "k_core", "k_compact"],
    "phonetic": ["meta_jacc", "skel_ratio", "k_skel", "k_sorted", "sorted_eq", "native_b"],
    "typo": ["inv_b", "inv_fix", "n_lev"],
    "dense2": ["cos2", "rank2_fwd", "rank2_rev", "cos2_gap_a", "cos2_rank_b"],   # second embedder
    "listwise": ["p1", "p1_rank_a", "p1_gap_a", "p1_gap2_a", "p1_rank_b", "p1_gap_b", "mutual_best",
                 "p1_sum_a", "n_sl_a", "n_sl_b"],
}
_IDF_N, _IDF_A = {}, {}


def _listwise(c: pl.DataFrame) -> pl.DataFrame:
    top2 = pl.col("p1").top_k(2).min().over("a")
    c = c.with_columns(
        p1_rank_a=pl.col("p1").rank("min", descending=True).over("a").cast(pl.Int32),
        p1_gap_a=pl.col("p1") - pl.col("p1").max().over("a"),
        p1_rank_b=pl.col("p1").rank("min", descending=True).over("b").cast(pl.Int32),
        p1_gap_b=pl.col("p1") - pl.col("p1").max().over("b"),
        p1_sum_a=pl.col("p1").sum().over("a"),
        n_sl_a=pl.len().over("a").cast(pl.Int32), n_sl_b=pl.len().over("b").cast(pl.Int32))
    return c.with_columns(
        p1_gap2_a=pl.when(pl.col("n_sl_a") == 1).then(pl.col("p1"))
        .when(pl.col("p1_gap_a") == 0).then(pl.col("p1") - top2).otherwise(pl.col("p1_gap_a")),
        mutual_best=((pl.col("p1_rank_a") == 1) & (pl.col("p1_rank_b") == 1)).cast(pl.Int8))


def _wjacc(x, y, idf, default):
    if not x or not y:
        return math.nan
    sx, sy = set(x), set(y)
    w = lambda s: sum(idf.get(t, default) for t in s)
    union = w(sx | sy)
    return w(sx & sy) / union if union else math.nan


def _abbr_cov(x, y):
    if not x or not y:
        return math.nan
    sx = set(x)
    hit = sum(1 for t in set(y) if t in sx or any(u.startswith(t) or t.startswith(u) for u in sx))
    return hit / len(set(y))


def _num_rel(x, y):
    """0 missing, 1 equal, 2 truncated (9600~960: a match signal), 3 one-digit edit, 4 near (<=10),
    5 different — 3 and 4 are the near-twin signals (deep diagnosis E2/E7)."""
    if not x or not y:
        return 0
    if x == y:
        return 1
    if x.startswith(y) or y.startswith(x) or x.endswith(y) or y.endswith(x):
        return 2
    if len(x) == len(y) and sum(p != q for p, q in zip(x, y)) == 1:
        return 3
    if abs(int(x) - int(y)) <= 10:
        return 4
    return 5


def _contained(x, nums):
    return any(x == u or u.startswith(x) or x.startswith(u) for u in nums)


def _py_feats(rows):
    out = []
    for country, na, nb, ca, cb, aa, ab, ma, mb, alla, allb, sa, sb, fa, fb, meta_a, meta_b in rows:
        idf_n, idf_a = _IDF_N.get(country, {}), _IDF_A.get(country, {})
        dn, da = idf_n.get("\0max", 10.0), idf_a.get("\0max", 10.0)
        ma, mb, alla, allb = ma or [], mb or [], alla or [], allb or []
        sma, smb = set(alla), set(allb)
        tna, tnb = na.split(), nb.split()
        sna, snb = set(tna), set(tnb)
        tca, tcb = ca.split(), cb.split()
        stop_a, stop_b = set(sa.split()), set(sb.split())
        # punishment tokens: distinctive core words present on one side only (branch/twin names)
        un_a = [t for t in tca if t not in snb]
        un_b = [t for t in tcb if t not in sna]
        # invented words: b's core words that are (near) unique in the split = typo / garble noise
        inv = [t for t in un_b if idf_n.get(t, dn) >= dn - 0.7]
        fix = sum(1 for t in inv if any(Levenshtein.distance(t, u, score_cutoff=2) <= 2 for u in tca))
        ia = [idf_n.get(t, dn) for t in tca]
        smeta_a, smeta_b = set(meta_a or ()), set(meta_b or ())
        out.append((
            _wjacc(tna, tnb, idf_n, dn),
            _wjacc(aa, ab, idf_a, da),
            _abbr_cov(aa or (), ab or ()),
            len(sma & smb) / len(sma | smb) if (sma or smb) else math.nan,
            len(sma & smb) / len(sma) if sma else math.nan,
            len(stop_a & stop_b) / len(stop_a | stop_b) if (stop_a or stop_b) else math.nan,
            _num_rel(fa, fb),
            sum(_contained(x, allb) for x in ma) / len(ma) if (ma and allb) else math.nan,
            sum(not _contained(x, alla) for x in mb) if (mb and alla) else math.nan,
            sum(not _contained(x, allb) for x in ma) if (ma and allb) else math.nan,
            max((idf_n.get(t, dn) for t in un_a), default=0.0),
            max((idf_n.get(t, dn) for t in un_b), default=0.0),
            len(un_a) + len(un_b),
            len(inv),
            fix / len(inv) if inv else math.nan,
            len(smeta_a & smeta_b) / len(smeta_a | smeta_b) if (smeta_a or smeta_b) else math.nan,
            sum(ia), min(ia, default=0.0),
        ))
    return out


PY_COLS = ["n_idf_jacc", "a_idf_jacc", "a_abbr_cov", "num_jacc", "num_a_in_b", "stop_jacc",
           "num_rel", "num_main_cov", "num_extra_b", "num_extra_a", "pun_max_a", "pun_max_b", "pun_n",
           "inv_b", "inv_fix", "meta_jacc", "core_idf_sum_a", "core_idf_min_a"]


def _load_idf(split):
    out = {}
    for field in ("name", "addr"):
        df = pl.read_parquet(work_dir(split) / f"df_{field}.parquet").with_columns(
            idf=(pl.col("n") / pl.col("df")).log())
        d = {}
        for (c,), g in df.group_by("country"):
            m = dict(zip(g["t"].to_list(), g["idf"].to_list()))
            m["\0max"] = float(g["idf"].max())
            d[c] = m
        out[field] = d
    return out["name"], out["addr"]


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=P.workers, dtype=np.float32)


def _chunk_features(c: pl.DataFrame, pool, rec_cols: list) -> pl.DataFrame:
    g = lambda col: c[col].fill_null("").to_list()
    na, nb = g("name_n"), g("name_n_b")
    ca, cb = g("core"), g("core_b")
    ka, kb = g("compact"), g("compact_b")
    has_b = c["has_addr_b"].to_numpy() > 0           # stage-1 column (0/1)
    aa, ab = g("addr_n"), g("addr_n_b")
    nan_if_no_addr = lambda x: np.where(has_b, x, np.nan).astype(np.float32)
    f = {
        "n_ratio": _cp(na, nb, fuzz.ratio), "n_tsort": _cp(na, nb, fuzz.token_sort_ratio),
        "n_partial": _cp(na, nb, fuzz.partial_ratio),
        "n_jw": _cp(na, nb, JaroWinkler.normalized_similarity),
        "c_tset": _cp(ca, cb, fuzz.token_set_ratio),
        "n_lev": _cp(ca, cb, Levenshtein.normalized_similarity),
        "k_ratio": _cp(ka, kb, fuzz.ratio), "k_partial": _cp(ka, kb, fuzz.partial_ratio),
        "o_ratio": _cp(g("ocr"), g("ocr_b"), fuzz.ratio),
        "skel_ratio": _cp(g("skel"), g("skel_b"), fuzz.ratio),
        "a_ratio": nan_if_no_addr(_cp(aa, ab, fuzz.ratio)),
        "a_tsort": nan_if_no_addr(_cp(aa, ab, fuzz.token_sort_ratio)),
        "a_partial": nan_if_no_addr(_cp(aa, ab, fuzz.partial_ratio)),
    }
    best = c["n_tset"].to_numpy().copy()
    al_a, al_b = c["aliases"].to_list(), c["aliases_b"].to_list()
    for i in np.flatnonzero([bool(x) or bool(y) for x, y in zip(al_a, al_b)]):
        opts_a, opts_b = [na[i]] + (al_a[i] or []), [nb[i]] + (al_b[i] or [])
        best[i] = max(fuzz.token_set_ratio(x, y) for x in opts_a for y in opts_b)
    f["n_alias_best"] = best
    xa, xb = g("addr_core"), g("addr_core_b")
    f["a_core_tset"] = nan_if_no_addr(_cp(xa, xb, fuzz.token_set_ratio))
    f["a_core_ratio"] = nan_if_no_addr(_cp(xa, xb, fuzz.ratio))
    acr_a, acr_b = g("acr"), g("acr_b")
    f["acr_match"] = np.array([float(bool(x and x == kb_) or bool(y and y == ka_))
                               for x, y, ka_, kb_ in zip(acr_a, acr_b, ka, kb)], dtype=np.float32)
    f["sorted_eq"] = ((c["sorted_core"] == c["sorted_core_b"]) & (c["sorted_core"] != "")).cast(pl.Float32).to_numpy()
    rows = list(zip(c["country"].to_list(), na, nb, ca, cb, c["alpha"].to_list(), c["alpha_b"].to_list(),
                    c["main_nums"].to_list(), c["main_nums_b"].to_list(), c["nums"].to_list(),
                    c["nums_b"].to_list(), g("stop"), g("stop_b"), g("first_num"), g("first_num_b"),
                    c["meta"].to_list(), c["meta_b"].to_list()))
    parts = pool.map(_py_feats, [rows[i:i + 50_000] for i in range(0, len(rows), 50_000)])
    py = np.array([r for part in parts for r in part], dtype=np.float32).reshape(-1, len(PY_COLS))
    for i, col in enumerate(PY_COLS):
        f[col] = py[:, i]
    f["len_name_a"] = c["name_n"].str.len_chars().cast(pl.Float32).to_numpy()
    f["len_name_b"] = c["name_n_b"].str.len_chars().cast(pl.Float32).to_numpy()
    f["lu_s1_name_a"] = np.log1p(c["u_s1_name"].cast(pl.Float32).to_numpy())
    f["lu_oth_numst_b"] = np.log1p(c["u_oth_numst_b"].cast(pl.Float32).to_numpy())
    return c.drop(rec_cols).with_columns(**{k: pl.Series(v) for k, v in f.items()})


def run(split: str) -> None:
    global _IDF_N, _IDF_A
    out = work_dir(split, "feats")
    if (out / "_DONE").exists():
        log(f"features[{split}]: cached")
        return
    t0 = time.time()
    wd = work_dir(split)
    sl = _listwise(pl.read_parquet(wd / "sl.parquet"))
    s1 = pl.read_parquet(wd / "s1.parquet", columns=FIELDS)
    oth = pl.read_parquet(wd / "oth.parquet", columns=FIELDS).drop("country", "has_addr")  # has_addr_b: stage 1
    oth = oth.rename({c: f"{c}_b" for c in oth.columns if c != "iid"})
    rec_cols = [c for c in s1.columns + oth.columns if c != "iid"]
    _IDF_N, _IDF_A = _load_idf(split)
    ids = s1["iid"].to_numpy()
    step = 200_000
    n = 0
    gc.collect()
    gc.freeze()          # workers inherit the IDF dicts; keep their GC from copying them (OOM)
    with Pool(P.pool_workers) as pool:
        for i, s in enumerate(range(0, len(ids), step)):
            c = (sl.filter(pl.col("a").is_in(pl.Series(ids[s:s + step])))
                 .join(s1, left_on="a", right_on="iid").join(oth, left_on="b", right_on="iid"))
            if len(c):
                _chunk_features(c, pool, rec_cols).write_parquet(out / f"part-{i:04d}.parquet")
            n += len(c)
            log(f"features[{split}]: chunk {i} ({len(c):,} pairs)")
    gc.unfreeze()
    (out / "_DONE").touch()
    record_time(f"features.{split}", time.time() - t0, s1=len(s1), pairs=n)
