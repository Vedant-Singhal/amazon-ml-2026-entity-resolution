# =============================================================================
# Amazon / ML Challenge 2026 — Business Entity Resolution
# Single-file pipeline (CPU version, tested on Windows Python 3.12)
#
# Dependencies (pip install):
#   polars anyascii rapidfuzz sparse-dot-topn scikit-learn
#   faiss-cpu lightgbm xgboost joblib
#
# Set DATA_DIR below, or export ER_DATA_DIR environment variable.
# =============================================================================

import os, sys, re, json, time, math, random, shutil, zipfile
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import polars as pl
from anyascii import anyascii
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────────────── timing ──────────────────────────────────────────
RUN_T0 = time.perf_counter()

def STAGE(msg):
    print("\n" + "=" * 78, flush=True)
    print(f"[ER] {msg} | {(time.perf_counter()-RUN_T0)/60:.1f} min", flush=True)
    print("=" * 78, flush=True)

def timed(label):
    t = time.perf_counter()
    return lambda extra="": print(
        f"[TIME] {label}: {time.perf_counter()-t:.1f}s {extra}", flush=True
    )

# ─────────────────────────── config ──────────────────────────────────────────

# Point this at the folder containing train/ and test/
DATA_DIR = None   # override here, e.g. Path(r"C:\my\dataset")

def _is_root(p):
    return all((p / x).exists() for x in (
        "train/train_source1.tsv", "train/train_source2.tsv",
        "train/train_source3.tsv", "test/test_source1.tsv",
        "test/test_source2.tsv",  "test/test_source3.tsv",
    ))

def find_data_dir(explicit=None):
    cands = []
    if os.getenv("ER_DATA_DIR"):
        cands.append(Path(os.environ["ER_DATA_DIR"]).expanduser())
    if explicit:
        cands.append(Path(explicit).expanduser())
    cwd = Path.cwd().resolve()
    for base in [cwd, cwd.parent, cwd.parent.parent]:
        cands += [base, base / "data", base / "dataset",
                  base / "student_resource" / "dataset"]
        try:
            for p in base.rglob("train_source1.tsv"):
                if p.parent.name == "train" and _is_root(p.parent.parent):
                    cands.append(p.parent.parent)
        except (OSError, PermissionError):
            pass
    seen = set()
    for p in cands:
        try:
            p = p.resolve()
        except OSError:
            continue
        if p not in seen and _is_root(p):
            return p
        seen.add(p)
    raise FileNotFoundError(
        "Dataset not found. Set DATA_DIR or ER_DATA_DIR to the folder "
        "containing train/ and test/."
    )

DATA  = find_data_dir(DATA_DIR)
ROOT  = Path(os.getenv("ER_ROOT", Path.cwd())).resolve()
CACHE = Path(os.getenv("ER_CACHE_DIR", ROOT / "cache")).resolve()
OUT   = Path(os.getenv("ER_OUTPUT_DIR", ROOT / "output")).resolve()
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

print(f"[PATH] DATA  = {DATA}")
print(f"[PATH] CACHE = {CACHE}")
print(f"[PATH] OUT   = {OUT}")

# ─────────────────────────── switches ────────────────────────────────────────
DEVICE      = "cpu"                 # set "cuda" if you have a supported GPU
FULL_POOL   = True                  # test-like validation (use all S2/S3)
SAMPLE      = 0.30                  # fraction of S1 used for dev/validation
VAL_SHARE   = 0.40
SEED        = 0

TOP_PRE_RAPIDFUZZ = 80             # top-K candidates before RapidFuzz (was 40 — too low)
LOW_SIM_CUTOFF    = 0.12           # min ANN sim to keep (lowered from 0.20)

# Threshold tuning
GLOBAL_THRESH_GRID = np.arange(0.10, 0.97, 0.01).round(3)
MIN_GROUP_TRUE = 20
MIN_GROUP_S1   = 15

np.random.seed(SEED); random.seed(SEED)

# ─────────────────────────── booster detection ───────────────────────────────
def get_booster():
    try:
        import lightgbm as lgb
        return "lightgbm", lgb
    except ImportError:
        pass
    try:
        import xgboost as xgb
        return "xgboost", xgb
    except ImportError:
        raise ImportError("Install lightgbm or xgboost.")

BOOSTER_NAME, BOOSTER = get_booster()
print(f"[MODEL] booster: {BOOSTER_NAME}")

# ─────────────────────────── keys ────────────────────────────────────────────
S1   = "source1_entity_id"
CAND = "entity_id"

# =============================================================================
# STAGE 1 — Normalization
# =============================================================================
STAGE("STAGE 1/9 — Text normalisation")

PLACEHOLDER_RE = re.compile(r"<\s*null\s*>|\bn/?a\b|\bnull\b|\bnone\b", re.IGNORECASE)
NON_ALNUM_RE   = re.compile(r"[^a-z0-9]+")

TRANSLIT_FIXES = {
    "praibhet": "private", "praivet": "private", "piraivet": "private",
    "praivhet": "private", "praiveta": "private", "pryvett":  "private",
    "limitet":  "limited", "limitedd": "limited",  "limitadd": "limited",
    "elelpi":   "llp",     "elelpii":  "llp",
}
LEGAL_CANON = {
    "pvt": "private",    "pvtltd": "private limited",
    "ltd": "limited",    "ltda":   "limited",
    "corp": "corporation", "inc": "incorporated", "incorp": "incorporated",
    "co": "company",     "cos": "company",
    "llc": "llc", "pllc": "pllc", "llp": "llp", "lp": "lp",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu",
    "sa": "sa", "eurl": "eurl", "sci": "sci",
}
LEGAL_TOKENS = set(LEGAL_CANON.values()) | {
    "private", "limited", "company", "corporation", "incorporated", "ei", "snc"
}
ADDR_CANON = {
    "st": "street", "str": "street", "rd": "road",   "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "bd": "boulevard",
    "dr": "drive",  "ln": "lane",   "ct": "court",  "pl": "place",
    "pkwy": "parkway", "hwy": "highway", "cir": "circle",
    "ter": "terrace",  "sq": "square",   "apt": "apartment",
    "ste": "suite",    "fl": "floor",    "flr": "floor",
    "no": "number",    "nr": "near",     "opp": "opposite", "mkt": "market",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "r": "rue",   "ch": "chemin", "rte": "route", "imp": "impasse",
}

def _basic(text):
    if not text:
        return ""
    text = anyascii(str(text)).lower()
    text = text.replace("&", " and ").replace("+", " and ")
    return NON_ALNUM_RE.sub(" ", text).strip()

def norm_name(text):
    return " ".join(LEGAL_CANON.get(TRANSLIT_FIXES.get(t, t), TRANSLIT_FIXES.get(t, t))
                    for t in _basic(text).split())

def core_name(normed):
    kept = [t for t in normed.split() if t not in LEGAL_TOKENS]
    return " ".join(kept) if kept else normed

def norm_addr(text):
    if not text:
        return ""
    text = PLACEHOLDER_RE.sub(" ", str(text))
    return " ".join(ADDR_CANON.get(t, t) for t in _basic(text).split())

CHUNK = 100_000

def _norm_chunk(rows):
    names, addrs = rows
    n = [norm_name(x) for x in names]
    return n, [core_name(x) for x in n], [norm_addr(x) for x in addrs]

def prepare_file(src: Path, dst: Path, pool: Pool, source_no: int):
    df = pl.read_csv(src, separator="\t", quote_char=None, infer_schema=False)
    chunks = (
        (df["business_name"][i:i+CHUNK].to_list(),
         df["business_address"][i:i+CHUNK].to_list())
        for i in range(0, df.height, CHUNK)
    )
    parts = [
        pl.DataFrame({"name": n, "core": c, "addr": a},
                     schema={k: pl.String for k in ("name","core","addr")})
        for n, c, a in pool.imap(_norm_chunk, chunks)
    ]
    out = df.select(
        "entity_id", "country",
        pl.col("business_name").fill_null("").str.contains(
            r"[\p{L}&&[^\p{Latin}]]"
        ).alias("is_translit"),
        pl.lit(source_no, pl.Int8).alias("src"),
    ).hstack(pl.concat(parts))
    out.write_parquet(dst)
    print(f"  {src.name}: {out.height:,} rows -> {dst.name}", flush=True)

_done = timed("normalise")
with Pool(os.cpu_count() or 4) as pool:
    for split in ("train", "test"):
        for s in (1, 2, 3):
            name = f"{split}_source{s}"
            dst  = CACHE / f"{name}.parquet"
            src  = DATA / split / f"{name}.tsv"
            if not dst.exists() or "src" not in pl.read_parquet_schema(dst):
                prepare_file(src, dst, pool, s)
_done()

# =============================================================================
# STAGE 2 — Validation sample
# =============================================================================
STAGE("STAGE 2/9 — Building validation sample")
_done = timed("sample")

def _keep(col, frac, seed=SEED):
    return (pl.col(col).hash(seed) % 10_000) < int(round(frac * 10_000))

def _sample_tag(frac, full):
    return f"dev{round(frac*100)}" + ("f" if full else "")

def load_links(gt_path: Path) -> pl.DataFrame:
    return (
        pl.read_csv(gt_path, separator="\t", quote_char=None, infer_schema=False)
        .select("source1_entity_id", "matched_entity_ids")
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids", empty_as_null=True)
        .with_columns(pl.col("matched_entity_ids").str.strip_chars())
        .filter(pl.col("matched_entity_ids").is_not_null()
                & (pl.col("matched_entity_ids") != ""))
        .rename({"matched_entity_ids": CAND})
        .unique()
    )

TAG = _sample_tag(SAMPLE, FULL_POOL)
link_path = CACHE / f"{TAG}_links.parquet"

if not link_path.exists():
    all_links = load_links(DATA / "train" / "train_ground_truth.tsv")
    s1_all  = pl.read_parquet(CACHE / "train_source1.parquet")
    s1_dev  = s1_all.filter(_keep("entity_id", SAMPLE))
    pool_dev = pl.concat([
        pl.read_parquet(CACHE / "train_source2.parquet"),
        pl.read_parquet(CACHE / "train_source3.parquet"),
    ])
    links_dev = all_links.filter(
        pl.col(S1).is_in(s1_dev["entity_id"].to_list())
    )
    if not FULL_POOL:
        is_decoy = ~pl.col(CAND).is_in(all_links[CAND].to_list())
        pool_dev = pool_dev.filter(
            pl.col(CAND).is_in(links_dev[CAND].to_list())
            | (is_decoy & _keep(CAND, SAMPLE))
        )
    s1_dev.write_parquet(CACHE / f"{TAG}_source1.parquet")
    pool_dev.write_parquet(CACHE / f"{TAG}_pool.parquet")
    links_dev.write_parquet(link_path)
else:
    s1_dev   = pl.read_parquet(CACHE / f"{TAG}_source1.parquet")
    pool_dev = pl.read_parquet(CACHE / f"{TAG}_pool.parquet")
    links_dev = pl.read_parquet(link_path)

print(f"[VAL] S1={s1_dev.height:,}  pool={pool_dev.height:,}  "
      f"true_pairs={links_dev.height:,}  FULL_POOL={FULL_POOL}", flush=True)
_done()

# =============================================================================
# STAGE 3 — Candidate blocking (CPU TF-IDF + exact rescue)
# =============================================================================
STAGE("STAGE 3/9 — Candidate blocking")
_done = timed("blocking")

_NUM_RE = re.compile(r"\d+")

@dataclass
class View:
    name:      str
    top_k:     int
    reverse_k: int   = 0
    min_sim:   float = 0.08

VIEWS = [
    View("name", top_k=50, reverse_k=2,  min_sim=0.08),
    View("addr", top_k=50, reverse_k=2,  min_sim=0.08),
    View("full", top_k=75, reverse_k=8,  min_sim=0.08),
]

def view_text(df, view):
    if view == "name": return df["core"].to_list()
    if view == "addr": return df["addr"].to_list()
    return (df["core"] + " " + df["addr"]).to_list()

def exact_pairs(s1, pool, max_bucket=50):
    specs = [
        ("exact_core", "core", lambda x: x or None),
        ("exact_addr", "addr", lambda x: x or None),
        ("exact_nums", "addr", lambda x: tuple(_NUM_RE.findall(x)) or None),
    ]
    out = []
    for tag, col, key in specs:
        inv = {}
        for j, x in enumerate(pool[col].fill_null("").to_list()):
            k = key(x)
            if k is not None:
                inv.setdefault(k, []).append(j)
        rows, idx = [], []
        for i, x in enumerate(s1[col].fill_null("").to_list()):
            k = key(x)
            hits = inv.get(k) if k is not None else None
            if hits and len(hits) <= max_bucket:
                rows.extend([i]*len(hits)); idx.extend(hits)
        out.append((tag, np.asarray(rows, np.int64), np.asarray(idx, np.int64)))
    return out

def search_country(s1, pool, views, n_threads=None, chunk=20_000):
    from sparse_dot_topn import sp_matmul_topn
    n_threads = n_threads or max(1, (os.cpu_count() or 4))
    s1_ids, pool_ids = s1["entity_id"].to_numpy(), pool["entity_id"].to_numpy()
    found = []

    # Exact rescue
    for tag, qi, pi in exact_pairs(s1, pool):
        if len(qi):
            found.append(pl.DataFrame({
                S1: s1_ids[qi], CAND: pool_ids[pi],
                "view": tag, "sim": np.ones(len(qi), np.float32),
            }))

    # TF-IDF approximate
    for v in views:
        q_txt, p_txt = view_text(s1, v.name), view_text(pool, v.name)
        vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3,3), min_df=2,
            max_df=0.05, dtype=np.float32, sublinear_tf=True,
        ).fit(q_txt + p_txt)
        q, p = vec.transform(q_txt), vec.transform(p_txt)
        pt = p.T.tocsr()
        m = sp_matmul_topn(q, pt, top_n=v.top_k, threshold=v.min_sim,
                           n_threads=n_threads).tocoo()
        found.append(pl.DataFrame({
            S1: s1_ids[m.row], CAND: pool_ids[m.col],
            "view": v.name, "sim": m.data.astype(np.float32),
        }))
        if v.reverse_k:
            mr = sp_matmul_topn(p, q.T.tocsr(), top_n=v.reverse_k,
                                threshold=v.min_sim, n_threads=n_threads).tocoo()
            found.append(pl.DataFrame({
                S1: s1_ids[mr.col], CAND: pool_ids[mr.row],
                "view": v.name, "sim": mr.data.astype(np.float32),
            }))
        print(f"  view={v.name} pairs={m.nnz:,}", flush=True)

    return pl.concat(found)

def generate_candidates(s1, pool):
    parts = []
    for country in s1["country"].unique().sort().to_list():
        s1_c   = s1.filter(pl.col("country") == country)
        pool_c = pool.filter(pl.col("country") == country)
        if pool_c.height:
            print(f"  {country}: {s1_c.height:,} vs {pool_c.height:,}", flush=True)
            parts.append(search_country(s1_c, pool_c, VIEWS))
    long = pl.concat(parts)
    wide = long.pivot(on="view", index=[S1, CAND], values="sim",
                      aggregate_function="max")
    wide = wide.rename({c: f"sim_{c}" for c in wide.columns
                        if c not in (S1, CAND)})
    # Ensure all sim columns exist
    for col in ["sim_name","sim_addr","sim_full",
                "sim_exact_core","sim_exact_addr","sim_exact_nums"]:
        if col not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0, pl.Float32).alias(col))
        else:
            wide = wide.with_columns(pl.col(col).fill_null(0.0))
    return wide

cand_path = CACHE / f"{TAG}_cands.parquet"
if cand_path.exists():
    cands_dev = pl.read_parquet(cand_path)
else:
    cands_dev = generate_candidates(s1_dev, pool_dev)
    cands_dev.write_parquet(cand_path)

# Blocking diagnostics
hit  = cands_dev.select(S1, CAND).with_columns(hit=pl.lit(True))
lk   = (links_dev
        .join(hit, on=[S1, CAND], how="left")
        .with_columns(pl.col("hit").fill_null(False)))
print(f"[BLOCKING] pair_recall={lk['hit'].mean():.4%} "
      f"({lk['hit'].sum():,}/{lk.height:,})  "
      f"cands/S1={cands_dev.height/s1_dev.height:.1f}", flush=True)
_done(f"{cands_dev.height:,} raw candidates")

# =============================================================================
# STAGE 4 — Features
# =============================================================================
STAGE("STAGE 4/9 — Feature engineering")
_done = timed("features")

SIMS = ["sim_name","sim_addr","sim_full",
        "sim_exact_core","sim_exact_addr","sim_exact_nums"]

# Group-relative column names produced by group_features()
_GROUP_COLS = (set(SIMS)
               | {f"{s}_gap"  for s in SIMS}
               | {f"{s}_rank" for s in SIMS}
               | {"n_cands"})

FUZZ_FNS = {
    "ratio":   fuzz.ratio,
    "tsort":   fuzz.token_sort_ratio,
    "tset":    fuzz.token_set_ratio,
    "partial": fuzz.partial_ratio,
    "wratio":  fuzz.WRatio,
    "jw":      JaroWinkler.normalized_similarity,
}
SCORERS = {
    "name": ["ratio","tsort","tset","jw"],
    "core": ["ratio","tsort","tset","partial","wratio","jw"],
    "addr": ["ratio","tsort","tset","jw"],
}

def group_features(c: pl.DataFrame) -> pl.DataFrame:
    return c.with_columns(
        *[(pl.col(s) - pl.col(s).max().over(S1)).alias(f"{s}_gap") for s in SIMS],
        *[pl.col(s).rank("min", descending=True).over(S1).log1p().alias(f"{s}_rank")
          for s in SIMS],
        pl.len().over(S1).log1p().alias("n_cands"),
    )

def prefilter(cands: pl.DataFrame,
              top_k=TOP_PRE_RAPIDFUZZ,
              low_cutoff=LOW_SIM_CUTOFF) -> pl.DataFrame:
    best = pl.max_horizontal([pl.col(x) for x in
                               ["sim_name","sim_addr","sim_full"]])
    ex   = pl.max_horizontal([pl.col(x) for x in
                               ["sim_exact_core","sim_exact_addr","sim_exact_nums"]])
    return (
        cands.with_columns(best_view=best, _exact=ex)
        .filter((pl.col("best_view") >= low_cutoff) | (pl.col("_exact") > 0))
        .sort([S1, "best_view"], descending=[False, True])
        .with_columns(
            pl.col("best_view").rank("ordinal", descending=True)
            .over(S1).alias("_rk")
        )
        .filter((pl.col("_rk") <= top_k) | (pl.col("_exact") > 0))
        .drop(["best_view", "_exact", "_rk"])
    )

def _string_features(df: pl.DataFrame) -> pl.DataFrame:
    out = {}
    for fld, names in SCORERS.items():
        col_a = "name" if fld == "name" else fld
        a = df[col_a].fill_null("").to_list()
        b = df[f"{fld}_b"].fill_null("").to_list()
        for k in names:
            vals = process.cpdist(a, b, scorer=FUZZ_FNS[k], workers=-1)
            div  = 1 if k == "jw" else 100
            out[f"{fld}_{k}"] = np.asarray(vals, dtype=np.float32) / div
    return pl.DataFrame(out)

def exact_tfidf_cosine(s1_texts, pool_texts, s1_idx, pool_idx):
    if len(s1_idx) == 0:
        return np.zeros(0, np.float32)
    texts = s1_texts + pool_texts
    vec   = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,3),
                            min_df=1, dtype=np.float32,
                            sublinear_tf=True, norm="l2")
    X     = vec.fit_transform(texts)
    n     = len(s1_texts)
    a     = X[np.asarray(s1_idx)]
    b     = X[n + np.asarray(pool_idx)]
    return np.asarray(a.multiply(b).sum(axis=1)).ravel().astype(np.float32)

def _jaccard(a, b):
    return (a.list.set_intersection(b).list.len()
            / a.list.set_union(b).list.len()).fill_nan(0.0)

def add_cross_source_features(df: pl.DataFrame) -> pl.DataFrame:
    if df.height == 0:
        return df
    src = df.select(S1, CAND, "from_s3", "sim_full",
                    "sim_name", "sim_addr", "core_b", "addr_b")
    opp = (
        src.sort([S1, "from_s3", "sim_full"], descending=[False, False, True])
        .group_by([S1, "from_s3"], maintain_order=True)
        .agg([
            pl.col("core_b").first().alias("opp_core"),
            pl.col("addr_b").first().alias("opp_addr"),
            pl.col("sim_full").first().alias("opp_sim"),
        ])
    )
    # strongest opposite-source record for each S1
    s2_opp = (opp.filter(pl.col("from_s3") == True)
              .rename({"opp_core":"_os3_core","opp_addr":"_os3_addr",
                       "opp_sim":"_os3_sim"})
              .drop("from_s3"))
    s3_opp = (opp.filter(pl.col("from_s3") == False)
              .rename({"opp_core":"_os2_core","opp_addr":"_os2_addr",
                       "opp_sim":"_os2_sim"})
              .drop("from_s3"))

    out = (df.join(s2_opp, on=S1, how="left")
             .join(s3_opp, on=S1, how="left"))

    out = out.with_columns(
        pl.when(pl.col("from_s3") == True)
          .then(pl.col("_os2_sim").fill_null(0))
          .otherwise(pl.col("_os3_sim").fill_null(0))
          .alias("opp_best_sim"),
        pl.when(pl.col("from_s3") == True)
          .then((pl.col("core_b") == pl.col("_os2_core")).fill_null(False))
          .otherwise((pl.col("core_b") == pl.col("_os3_core")).fill_null(False))
          .cast(pl.Float32).alias("cross_core_agree"),
        pl.when(pl.col("from_s3") == True)
          .then((pl.col("addr_b") == pl.col("_os2_addr")).fill_null(False))
          .otherwise((pl.col("addr_b") == pl.col("_os3_addr")).fill_null(False))
          .cast(pl.Float32).alias("cross_addr_agree"),
    ).with_columns(
        (pl.col("cross_core_agree") * pl.col("opp_best_sim")).alias("cross_core_weighted"),
        (pl.col("cross_addr_agree") * pl.col("opp_best_sim")).alias("cross_addr_weighted"),
    ).drop(["_os3_core","_os3_addr","_os3_sim",
            "_os2_core","_os2_addr","_os2_sim"])
    return out

def featurize(cands, s1, pool, s1_per_chunk=10_000):
    if cands.height == 0:
        return
    filt = prefilter(cands)
    print(f"[FILTER] {cands.height:,} -> {filt.height:,} candidates "
          f"(top-K={TOP_PRE_RAPIDFUZZ})", flush=True)

    # Side tables
    s1_side = s1.select(
        pl.col("entity_id").alias(S1),
        pl.col("name"),
        pl.col("core"),
        pl.col("addr"),
        pl.col("is_translit"),
    )
    pool_side = pool.select(
        pl.col("entity_id").alias(CAND),
        pl.col("name").alias("name_b"),
        pl.col("core").alias("core_b"),
        pl.col("addr").alias("addr_b"),
        pl.col("is_translit").alias("is_translit_b"),
        (pl.col("src") == 3).alias("from_s3"),
    )

    s1_core_list  = s1["core"].fill_null("").to_list()
    pool_core_list = pool["core"].fill_null("").to_list()

    # Sort by s1_row for chunk splitting
    # filt may not have s1_row; we need to add ordinal S1 index
    s1_id_to_idx  = {eid: i for i, eid in enumerate(s1["entity_id"].to_list())}
    pool_id_to_idx = {eid: i for i, eid in enumerate(pool["entity_id"].to_list())}

    filt2 = (filt
             .with_columns(
                 pl.col(S1).map_elements(s1_id_to_idx.__getitem__, return_dtype=pl.Int32)
                 .alias("_s1_idx"),
                 pl.col(CAND).map_elements(pool_id_to_idx.__getitem__, return_dtype=pl.Int32)
                 .alias("_pool_idx"),
             )
             .sort("_s1_idx"))

    rows      = filt2["_s1_idx"].to_numpy()
    pool_rows = filt2["_pool_idx"].to_numpy()
    cuts      = np.searchsorted(rows, np.arange(0, rows[-1]+s1_per_chunk+1, s1_per_chunk))

    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if lo == hi:
            continue
        part     = group_features(filt2[int(lo):int(hi)])
        s1_idx   = part["_s1_idx"].to_numpy()
        pool_idx = part["_pool_idx"].to_numpy()

        joined = (
            part
            .join(s1_side,   on=S1,   how="inner")
            .join(pool_side, on=CAND,  how="inner")
        )

        # Exact sparse TF-IDF cosine
        etf = exact_tfidf_cosine(s1_core_list, pool_core_list, s1_idx, pool_idx)

        words  = lambda c: pl.col(c).str.extract_all(r"\w+")
        nums   = lambda c: pl.col(c).str.extract_all(r"\d+")
        pin_fn = lambda c: pl.col(c).str.extract(r"(\d{5,6})$", 1)

        tokens = joined.select(
            S1, CAND,
            *[c for c in joined.columns if c in _GROUP_COLS],
            exact_tfidf_cosine=pl.Series(etf),
            core_jac=_jaccard(words("core"), words("core_b")),
            num_jac=_jaccard(nums("addr"), nums("addr_b")),
            num_both=((nums("addr").list.len() > 0) & (nums("addr_b").list.len() > 0)),
            num_overlap=nums("addr").list.set_intersection(nums("addr_b")).list.len(),
            pin_exact=(pin_fn("addr") == pin_fn("addr_b")).fill_null(False),
            core_exact=(pl.col("core") == pl.col("core_b")),
            addr_exact=((pl.col("addr") == pl.col("addr_b")) & (pl.col("addr") != "")),
            name_exact=(pl.col("name") == pl.col("name_b")),
            addr_empty=((pl.col("addr") == "") | (pl.col("addr_b") == "")),
            len_ratio=(
                pl.min_horizontal(pl.col("core").str.len_chars(),
                                  pl.col("core_b").str.len_chars())
                / pl.max_horizontal(pl.col("core").str.len_chars(),
                                    pl.col("core_b").str.len_chars(), 1)
            ),
            translit=(pl.col("is_translit").cast(pl.Int8)
                      + pl.col("is_translit_b").cast(pl.Int8)),
            from_s3=pl.col("from_s3").cast(pl.Float32),
            # Keep for cross-source features
            core_b=pl.col("core_b"),
            addr_b=pl.col("addr_b"),
        )

        sf  = _string_features(joined)
        out = tokens.hstack(sf)
        out = add_cross_source_features(out)
        # Drop helper text cols not needed for model
        out = out.drop(["core_b", "addr_b"], strict=False)
        yield out.with_columns(pl.exclude(S1, CAND).cast(pl.Float32))

feats_path = CACHE / f"{TAG}_feats.parquet"
if feats_path.exists():
    feats = pl.read_parquet(feats_path)
else:
    feats = pl.concat(featurize(cands_dev, s1_dev, pool_dev))
    feats.write_parquet(feats_path)

FEATURES = [c for c in feats.columns if c not in (S1, CAND)]
print(f"[FEAT] {len(FEATURES)} features, {feats.height:,} pairs", flush=True)

labeled = (
    feats.join(links_dev.with_columns(label=pl.lit(1.0)), on=[S1, CAND], how="left")
    .with_columns(pl.col("label").fill_null(0.0))
)

is_val    = (pl.col(S1).hash(1) % 100) < round(VAL_SHARE * 100)
train_df  = labeled.filter(~is_val)
val_df    = labeled.filter(is_val)
val_s1_ids = (s1_dev
              .filter((pl.col("entity_id").hash(1) % 100) < round(VAL_SHARE * 100))
              ["entity_id"])
val_links = links_dev.filter(pl.col(S1).is_in(val_s1_ids.to_list()))

print(f"[SPLIT] train={train_df.height:,} val={val_df.height:,} "
      f"pos_rate={train_df['label'].mean():.4f}", flush=True)
_done()

# =============================================================================
# STAGE 5 — First-pass gradient boosting
# =============================================================================
STAGE("STAGE 5/9 — First-pass gradient booster")
_done = timed("model1")

def make_lgb(seed=SEED):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=600, learning_rate=0.04,
        num_leaves=63, min_child_samples=40, subsample=0.85,
        colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )

def make_xgb(seed=SEED):
    import xgboost as xgb
    return xgb.XGBClassifier(
        objective="binary:logistic", n_estimators=600,
        learning_rate=0.04, max_depth=8, min_child_weight=5,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.2, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, tree_method="hist",
    )

def fit_model(train_df, val_df, features, seed=SEED):
    X, y   = train_df.select(features).to_numpy(), train_df["label"].to_numpy()
    Xv, yv = val_df.select(features).to_numpy(),   val_df["label"].to_numpy()

    if BOOSTER_NAME == "lightgbm":
        m = make_lgb(seed)
        try:
            m.fit(X, y, eval_X=Xv, eval_y=yv)        # LightGBM 4.x API
        except TypeError:
            m.fit(X, y, eval_set=[(Xv, yv)])          # fallback
    else:
        m = make_xgb(seed)
        m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    return m

def predict(model, df, features):
    return np.asarray(model.predict_proba(df.select(features).to_numpy())[:, 1],
                      dtype=np.float32)

def feature_importance(model, features, top=30):
    if not hasattr(model, "feature_importances_"):
        return None
    return (pl.DataFrame({"feature": features,
                           "importance": model.feature_importances_.astype(float)})
            .sort("importance", descending=True).head(top))

model1 = fit_model(train_df, val_df, FEATURES)
imp = feature_importance(model1, FEATURES)
if imp is not None:
    print("[MODEL1] Top features:"); print(imp.head(20))

p1_train = predict(model1, train_df, FEATURES)
p1_val   = predict(model1, val_df,   FEATURES)
_done()

# =============================================================================
# STAGE 6 — Second-pass reranker
# =============================================================================
STAGE("STAGE 6/9 — Second-pass reranker")
_done = timed("model2")

def add_second_pass(df, scores):
    out = df.with_columns(pl.Series("_p1", scores))
    return out.with_columns(
        pl.col("_p1").rank("ordinal", descending=True).over(S1).alias("p1_rank"),
        (pl.col("_p1") - pl.col("_p1").max().over(S1)).alias("p1_gap_best"),
        (pl.col("_p1") >= 0.50).cast(pl.Int32).sum().over(S1).alias("n_p1_above_050"),
        (pl.col("_p1") >= 0.35).cast(pl.Int32).sum().over(S1).alias("n_p1_above_035"),
        pl.col("_p1").max().over(S1).alias("p1_best"),
        pl.col("_p1").mean().over(S1).alias("p1_mean"),
    ).with_columns(
        (1.0 / pl.col("p1_rank").cast(pl.Float32)).alias("p1_recip_rank"),
        (pl.col("p1_best") - pl.col("_p1")).alias("p1_margin"),
    )

SECOND_FEATURES = FEATURES + [
    "p1_rank","p1_gap_best","n_p1_above_050","n_p1_above_035",
    "p1_best","p1_mean","p1_recip_rank","p1_margin",
]

train2 = add_second_pass(train_df, p1_train)
val2   = add_second_pass(val_df,   p1_val)

model2 = fit_model(train2, val2, SECOND_FEATURES, seed=SEED+1)
imp2 = feature_importance(model2, SECOND_FEATURES)
if imp2 is not None:
    print("[MODEL2] Top features:"); print(imp2.head(20))

p2_val = predict(model2, val2, SECOND_FEATURES)
_done()

# =============================================================================
# STAGE 7 — Threshold tuning
# =============================================================================
STAGE("STAGE 7/9 — Threshold tuning")
_done = timed("threshold")

# val_scored keeps the full feature frame for audit later
val_full_scored = val2.with_columns(
    pl.Series("p", p2_val),
    pl.Series("_p2", p2_val),
)

def macro_f05(preds, links, s1_ids):
    per     = pl.DataFrame({S1: s1_ids})
    n_pred  = preds.group_by(S1).len("n_pred")
    n_true  = links.group_by(S1).len("n_true")
    tp_cnt  = preds.join(links, on=[S1, CAND], how="semi").group_by(S1).len("tp")
    per = (per
           .join(n_pred,  on=S1, how="left")
           .join(n_true,  on=S1, how="left")
           .join(tp_cnt,  on=S1, how="left")
           .fill_null(0))
    f = (pl.when((pl.col("n_true") + pl.col("n_pred")) == 0).then(1.0)
           .otherwise(1.25 * pl.col("tp")
                      / (0.25 * pl.col("n_true") + pl.col("n_pred"))))
    return float(per.select(f.mean()).item())

def threshold_search(scored, links, s1_ids, grid=GLOBAL_THRESH_GRID):
    vals = {float(t): macro_f05(scored.filter(pl.col("p") >= t), links, s1_ids)
            for t in grid}
    best = max(vals, key=vals.get)
    return float(best), vals

val_scored_slim = val_full_scored.select(S1, CAND, "p")
global_thresh, global_vals = threshold_search(val_scored_slim, val_links, val_s1_ids)
print(f"[THRESH] global={global_thresh:.3f}  "
      f"val_macro_F0.5={global_vals[global_thresh]:.5f}", flush=True)

# Country + source thresholds
pool_meta = pool_dev.select(pl.col("entity_id").alias(CAND),
                             pl.col("src").alias("cand_src"))
s1_meta   = s1_dev.select(pl.col("entity_id").alias(S1), "country")
val_aug   = (val_scored_slim
             .join(s1_meta,   on=S1,   how="left")
             .join(pool_meta, on=CAND,  how="left"))

def group_thresholds(scored, links, s1_ids, col):
    result = {}
    for grp in scored[col].drop_nulls().unique().to_list():
        g_s1    = scored.filter(pl.col(col) == grp)[S1].unique()
        g_links = links.filter(pl.col(S1).is_in(g_s1.to_list()))
        if len(g_s1) < MIN_GROUP_S1 or g_links.height < MIN_GROUP_TRUE:
            continue
        t, v = threshold_search(scored.filter(pl.col(col) == grp), g_links, g_s1)
        result[str(grp)] = {"threshold": float(t), "f05": float(v[t]),
                             "n_s1": int(len(g_s1)), "n_true": int(g_links.height)}
    return result

country_thresh = group_thresholds(val_aug, val_links, val_s1_ids, "country")
source_thresh  = group_thresholds(val_aug, val_links, val_s1_ids, "cand_src")

print("[THRESH] country thresholds:")
for k, v in sorted(country_thresh.items()):
    print(f"  {k}: {v['threshold']:.3f} | S1={v['n_s1']} true={v['n_true']}")
print("[THRESH] source thresholds:")
for k, v in sorted(source_thresh.items()):
    print(f"  src={k}: {v['threshold']:.3f} | S1={v['n_s1']} true={v['n_true']}")

thresh_config = {
    "global":  global_thresh,
    "country": country_thresh,
    "source":  source_thresh,
    "full_pool": FULL_POOL,
    "sample":  SAMPLE,
}
(CACHE / "thresholds.json").write_text(json.dumps(thresh_config, indent=2))
_done()

# =============================================================================
# STAGE 8 — Validation error audit
# =============================================================================
STAGE("STAGE 8/9 — Error audit")
_done = timed("audit")

def choose_threshold(country, src):
    c = country_thresh.get(str(country))
    if c: return float(c["threshold"])
    s = source_thresh.get(str(src))
    if s: return float(s["threshold"])
    return global_thresh

# Join features back for audit
audit = (
    val_full_scored
    .join(s1_meta,   on=S1,   how="left")
    .join(pool_meta, on=CAND,  how="left")
    .join(val_links.with_columns(label=pl.lit(1)), on=[S1, CAND], how="left")
    .with_columns(pl.col("label").fill_null(0))
    .with_columns(pl.col("p").rank("ordinal", descending=True).over(S1).alias("rank"))
)

fp = audit.filter((pl.col("p") >= global_thresh) & (pl.col("label") == 0))
fn = audit.filter((pl.col("p") <  global_thresh) & (pl.col("label") == 1))
fp = fp.sample(n=min(50, fp.height), seed=SEED)
fn = fn.sample(n=min(50, fn.height), seed=SEED)

audit_cols = [S1, CAND, "country", "cand_src", "p", "rank",
              "sim_name", "sim_addr", "sim_full",
              "exact_tfidf_cosine", "core_jac", "num_jac", "translit"]
audit_cols = [c for c in audit_cols if c in audit.columns]

fp.select(audit_cols).write_csv(OUT / "val_false_positives.csv")
fn.select(audit_cols).write_csv(OUT / "val_false_negatives.csv")
print(f"[AUDIT] FP={fp.height}  FN={fn.height}")
if "translit" in fp.columns:
    print(f"[AUDIT] Translit FP={fp.filter(pl.col('translit') > 0).height}  "
          f"FN={fn.filter(pl.col('translit') > 0).height}")
_done()

# =============================================================================
# STAGE 9 — Test inference
# =============================================================================
STAGE("STAGE 9/9 — Test inference")
_done = timed("inference")

def score_batch(cands, s1, pool):
    parts = []
    for f in featurize(cands, s1, pool):
        p1 = predict(model1, f, FEATURES)
        f2 = add_second_pass(f, p1)
        p2 = predict(model2, f2, SECOND_FEATURES)
        parts.append(f2.with_columns(pl.Series("p", p2)))
    return pl.concat(parts) if parts else pl.DataFrame()

s1_test   = pl.read_parquet(CACHE / "test_source1.parquet")
pool_test = pl.concat([
    pl.read_parquet(CACHE / "test_source2.parquet"),
    pl.read_parquet(CACHE / "test_source3.parquet"),
])

all_matches, all_cand_lists = [], []

for country in s1_test["country"].unique().sort().to_list():
    t0    = time.perf_counter()
    s1_c  = s1_test.filter(pl.col("country") == country)
    pool_c = pool_test.filter(pl.col("country") == country)
    if pool_c.height == 0:
        continue
    print(f"[TEST] {country}: S1={s1_c.height:,} pool={pool_c.height:,}", flush=True)

    cands  = generate_candidates(s1_c, pool_c)
    scored = score_batch(cands, s1_c, pool_c)
    if scored.height == 0:
        continue

    scored = scored.with_columns(
        pl.when(pl.col("from_s3") == 1.0).then(pl.lit(3)).otherwise(pl.lit(2))
        .alias("cand_src")
    )
    thresholds = [choose_threshold(country, int(x))
                  for x in scored["cand_src"].to_list()]
    scored = scored.with_columns(pl.Series("_thresh", thresholds))

    matches = scored.filter(pl.col("p") >= pl.col("_thresh")).select(S1, CAND)
    all_matches.append(matches)
    all_cand_lists.append(
        scored.group_by(S1).agg(
            pl.col(CAND).unique().sort().str.join(",").alias("candidate_entity_ids")
        )
    )
    print(f"[TEST] {country}: {cands.height:,} cands -> "
          f"{matches.height:,} matches  [{time.perf_counter()-t0:.0f}s]", flush=True)

# ─────────────────────────── write outputs ────────────────────────────────────
test_ids = s1_test["entity_id"]

def write_tsv(df, ids, col, path):
    base = pl.DataFrame({S1: ids})
    if df is not None and df.height:
        base = base.join(df, on=S1, how="left")
    else:
        base = base.with_columns(pl.lit(None, pl.String).alias(col))
    base.with_columns(pl.col(col).fill_null("")).write_csv(
        path, separator="\t", quote_style="never"
    )

cand_out  = OUT / "candidate_pairs.tsv"
match_out = OUT / "matching_results.tsv"

cand_lists = pl.concat(all_cand_lists) if all_cand_lists else None
write_tsv(cand_lists, test_ids, "candidate_entity_ids", cand_out)

matches = (pl.concat(all_matches).unique() if all_matches
           else pl.DataFrame(schema={S1: pl.String, CAND: pl.String}))
match_agg = (matches.group_by(S1)
             .agg(pl.col(CAND).unique().sort().str.join(",").alias("matched_entity_ids"))
             if matches.height else None)
write_tsv(match_agg, test_ids, "matched_entity_ids", match_out)

# Save artefacts
try:
    import joblib
    joblib.dump(model1, CACHE / "model1.joblib")
    joblib.dump(model2, CACHE / "model2.joblib")
except Exception as e:
    print(f"[MODEL] save skipped: {e}")

(CACHE / "features.json").write_text(json.dumps({
    "features": FEATURES,
    "second_features": SECOND_FEATURES,
    "booster": BOOSTER_NAME,
    "top_pre_rapidfuzz": TOP_PRE_RAPIDFUZZ,
    "low_sim_cutoff": LOW_SIM_CUTOFF,
    "global_threshold": global_thresh,
}, indent=2))

try:
    shutil.copy2(match_out, ROOT / "matching_results.tsv")
except Exception:
    pass

try:
    with zipfile.ZipFile(ROOT / "results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in [match_out, cand_out,
                  CACHE / "model1.joblib", CACHE / "model2.joblib",
                  CACHE / "thresholds.json",
                  OUT / "val_false_positives.csv",
                  OUT / "val_false_negatives.csv"]:
            if p.exists():
                z.write(p, p.relative_to(ROOT))
    print(f"[RESULT] package: {ROOT/'results.zip'}")
except Exception as e:
    print(f"[RESULT] zip skipped: {e}")

_done()

# ─────────────────────────── summary ─────────────────────────────────────────
print("\n" + "=" * 78)
print("DONE")
print(f"DATA            : {DATA}")
print(f"CACHE           : {CACHE}")
print(f"OUT             : {OUT}")
print(f"Global threshold: {global_thresh:.3f}")
print(f"Val Macro F0.5  : {global_vals[global_thresh]:.5f}")
print(f"Total S1 matched: {matches[S1].n_unique():,}/{len(test_ids):,} "
      f"({matches[S1].n_unique()/max(len(test_ids),1):.1%})")
print(f"Booster         : {BOOSTER_NAME}")
print(f"Top-K filter    : {TOP_PRE_RAPIDFUZZ}")
print(f"FULL_POOL       : {FULL_POOL}")
print("=" * 78)
