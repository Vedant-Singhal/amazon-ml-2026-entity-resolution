"""02 - What true matches look like vs. hard negatives, and how separable they are.

Samples S1 entities, builds positives (GT) + hard negatives (share a rare name token, highest
name similarity, not a match) + random same-country negatives, computes string-similarity
features, then reports: noise taxonomy, per-feature AUC, worst examples, a LightGBM baseline
with entity-level macro F0.5 (as scored), and singleton behaviour.
Output: eda/reports/02_pairs.md
"""
import re

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from common import (GENERIC, LEGAL, REPORTS, SMOKE, Report, addr_clean, addr_numbers, fold, load_gt,
                    load_split, log, name_core, pl_fold, translit)

N_S1 = 4000 if SMOKE else 40000
HARD_PER_S1, RAND_PER_S1, MAX_TOKEN_DF, MAX_CANDS = 5, 2, 1000, 100
rng = np.random.default_rng(0)
rep = Report("02_pairs", "02 · Pair analysis: matches vs hard negatives")

tr1, oth = load_split("train")
gt, pairs = load_gt()
s1 = tr1.rename({"entity_id": "s1_id"}).sample(n=min(N_S1, len(tr1)), seed=0)
s1 = s1.join(gt, on="s1_id")
log(f"sampled {len(s1):,} S1 entities")

# ---- positives -------------------------------------------------------------------
pos = pairs.join(s1.select("s1_id"), on="s1_id").select("s1_id", "id").with_columns(label=pl.lit(1))

# ---- hard negatives via rare shared name tokens ------------------------------------
stop = sorted(LEGAL | GENERIC)


def name_tokens(df, id_col):
    return (df.select(id_col, "country", tok=pl_fold("business_name").str.split(" "))
            .explode("tok")
            .filter((pl.col("tok").str.len_chars() >= 3) & ~pl.col("tok").is_in(stop))
            .unique())


o_tok = name_tokens(oth, "entity_id").rename({"entity_id": "id"})
df_tok = o_tok.group_by("country", "tok").len().rename({"len": "df"})
s_tok = (name_tokens(s1, "s1_id").join(df_tok, on=["country", "tok"])
         .filter(pl.col("df") <= MAX_TOKEN_DF)
         .sort("df").group_by("s1_id").head(2))
cands = (s_tok.join(o_tok, on=["country", "tok"]).select("s1_id", "id").unique()
         .join(pos.select("s1_id", "id"), on=["s1_id", "id"], how="anti")
         .with_columns(r=pl.int_range(pl.len()).shuffle(seed=0).over("s1_id"))
         .filter(pl.col("r") < MAX_CANDS).drop("r"))
log(f"hard-negative candidates: {len(cands):,}")

o_small = oth.rename({"entity_id": "id"}).select("id", "business_name", "business_address", "src")
s_small = s1.select("s1_id", "business_name", "business_address", "country")
c = cands.join(s_small, on="s1_id").join(o_small, on="id", suffix="_o")
c = c.with_columns(sc=pl.Series(process.cpdist(
    [translit(x) for x in c["business_name"]], [translit(x) for x in c["business_name_o"]],
    scorer=fuzz.token_set_ratio, workers=-1)))
hard = (c.sort("sc", descending=True).group_by("s1_id").head(HARD_PER_S1)
        .select("s1_id", "id").with_columns(label=pl.lit(0)))

# ---- random same-country negatives ------------------------------------------------
rand = []
for (country,), g in s1.group_by("country"):
    pool = oth.filter(pl.col("country") == country)["entity_id"].to_numpy()
    ids = rng.choice(pool, size=len(g) * RAND_PER_S1)
    rand.append(pl.DataFrame({"s1_id": np.repeat(g["s1_id"].to_numpy(), RAND_PER_S1), "id": ids}))
rand = (pl.concat(rand).join(pos.select("s1_id", "id"), on=["s1_id", "id"], how="anti")
        .with_columns(label=pl.lit(0)))

P = pl.concat([pos.with_columns(kind=pl.lit("pos")), hard.with_columns(kind=pl.lit("hard")),
               rand.with_columns(kind=pl.lit("random"))])
P = P.join(s_small, on="s1_id").join(o_small, on="id", suffix="_o")
log(f"pairs: {P.group_by('kind').len().rows()}")

# ---- features --------------------------------------------------------------------
_TLD = re.compile(r"\b(www|com|net|org|co|in|fr|biz|info)\b")


def compact(s):
    return _TLD.sub("", s).replace(" ", "")


n1, n2 = [translit(x) for x in P["business_name"]], [translit(x) for x in P["business_name_o"]]
r1, r2 = [fold(x) for x in P["business_name"]], [fold(x) for x in P["business_name_o"]]
c1, c2 = [name_core(x) for x in P["business_name"]], [name_core(x) for x in P["business_name_o"]]
k1, k2 = [compact(x) for x in c1], [compact(x) for x in c2]
a1, a2 = [addr_clean(x) for x in P["business_address"]], [addr_clean(x) for x in P["business_address_o"]]
num1, num2 = [addr_numbers(x) for x in a1], [addr_numbers(x) for x in a2]
first_num = lambda s: (re.search(r"\d+", s) or [None])[0]


def cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)


log("computing features")
tfv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
tfv.fit(list(set(c1) | set(c2)))
T1, T2 = tfv.transform(c1), tfv.transform(c2)
F = {
    "n_ratio": cp(n1, n2, fuzz.ratio),
    "n_ratio_raw": cp(r1, r2, fuzz.ratio),        # without transliteration
    "n_partial": cp(n1, n2, fuzz.partial_ratio),
    "n_tsort": cp(n1, n2, fuzz.token_sort_ratio),
    "n_tset": cp(n1, n2, fuzz.token_set_ratio),
    "n_jw": cp(n1, n2, JaroWinkler.normalized_similarity) * 100,
    "core_tset": cp(c1, c2, fuzz.token_set_ratio),
    "core_compact_ratio": cp(k1, k2, fuzz.ratio),
    "core_compact_partial": cp(k1, k2, fuzz.partial_ratio),
    "core_tfidf_cos": np.asarray(T1.multiply(T2).sum(1)).ravel().astype(np.float32) * 100,
    "a_ratio": cp(a1, a2, fuzz.ratio),
    "a_tset": cp(a1, a2, fuzz.token_set_ratio),
    "a_partial": cp(a1, a2, fuzz.partial_ratio),
    "a_num_jacc": np.array([len(x & y) / len(x | y) if x | y else np.nan
                            for x, y in zip(num1, num2)], dtype=np.float32),
    "a_first_num_eq": np.array([float(first_num(x) == first_num(y)) if first_num(x) and first_num(y)
                                else np.nan for x, y in zip(a1, a2)], dtype=np.float32),
    "a_other_empty": np.array([float(not y) for y in a2], dtype=np.float32),
    "len_diff": np.abs(np.array([len(x) - len(y) for x, y in zip(n1, n2)], dtype=np.float32)),
    "is_s3": (P["src"] == "S3").cast(pl.Float32).to_numpy(),
}
# Similarity is meaningless when the other address is empty.
for f in ("a_ratio", "a_tset", "a_partial"):
    F[f] = np.where(F["a_other_empty"] == 1, np.nan, F[f])
P = P.with_columns(**{k: pl.Series(v) for k, v in F.items()})
P = P.with_columns(
    o_domain=pl.col("business_name_o").str.contains(r"(?i)\.(com|net|org|in|fr|biz)\b"),
    o_non_latin=pl.col("business_name_o").str.contains(r"[^\p{Latin}\p{Common}\p{Inherited}]"),
)
P.write_parquet(REPORTS / "02_pairs_sample.parquet")

# ---- noise taxonomy of true matches --------------------------------------------------
rep.h("What true matches look like (positives only)")
pp = P.filter(pl.col("kind") == "pos")
rep.table(pp.group_by("country", "src").agg(
    pairs=pl.len(),
    core_name_exact=(pl.col("core_tset") == 100).mean(),
    name_tset_ge90=(pl.col("n_tset") >= 90).mean(),
    name_tset_lt50=(pl.col("n_tset") < 50).mean(),
    other_domain=pl.col("o_domain").mean(),
    other_non_latin=pl.col("o_non_latin").mean(),
    other_addr_empty=(pl.col("a_other_empty") == 1).mean(),
    first_num_eq=pl.col("a_first_num_eq").fill_nan(None).mean(),
    addr_tset_ge80=(pl.col("a_tset") >= 80).mean(),
).sort("country", "src"))
rep.p("**Name-dissimilar matches (n_tset < 50):** how often does the address rescue them?")
weak = pp.filter(pl.col("n_tset") < 50)
rep.table(weak.group_by("country").agg(
    n=pl.len(), addr_empty=(pl.col("a_other_empty") == 1).mean(),
    addr_tset_ge80=(pl.col("a_tset") >= 80).mean(), first_num_eq=pl.col("a_first_num_eq").fill_nan(None).mean(),
    domain=pl.col("o_domain").mean(), non_latin=pl.col("o_non_latin").mean()).sort("country"))

rep.h("Transliteration effect (positives whose other name is non-Latin)")
nl = pp.filter(pl.col("o_non_latin"))
if len(nl):
    rep.p(f"{len(nl):,} pairs. Mean fuzz.ratio raw = {nl['n_ratio_raw'].mean():.1f}, "
          f"after transliteration = {nl['n_ratio'].mean():.1f}, core tf-idf cos = "
          f"{nl['core_tfidf_cos'].mean():.1f}")

# ---- per-feature separability ----------------------------------------------------------
rep.h("Per-feature separability (AUC; NaN rows dropped)")
rows = []
for f in F:
    for country in ["ALL"] + sorted(P["country"].unique().to_list()):
        d = P if country == "ALL" else P.filter(pl.col("country") == country)
        r = {"feature": f, "country": country}
        for neg in ("hard", "random"):
            x = d.filter(pl.col("kind").is_in(["pos", neg])).drop_nulls(f).filter(pl.col(f).is_not_nan())
            r[f"auc_vs_{neg}"] = (roc_auc_score(x["label"], x[f]) if x["label"].n_unique() == 2
                                  else float("nan"))
        q = d.filter(pl.col("kind") == "pos")[f].drop_nans().drop_nulls()
        h = d.filter(pl.col("kind") == "hard")[f].drop_nans().drop_nulls()
        qf = lambda s, x: float(s.quantile(x)) if len(s) else float("nan")
        r.update(pos_p10=qf(q, 0.1), pos_p50=qf(q, 0.5), hard_p50=qf(h, 0.5), hard_p90=qf(h, 0.9))
        rows.append(r)
auc = pl.DataFrame(rows)
rep.table(auc.filter(pl.col("country") == "ALL").sort("auc_vs_hard", descending=True))
rep.table(auc.filter(pl.col("country") != "ALL")
          .pivot(on="country", index="feature", values="auc_vs_hard"))

# ---- examples ----------------------------------------------------------------------
cols = ["country", "src", "business_name", "business_address", "business_name_o",
        "business_address_o", "n_tset", "a_tset"]
rep.h("Hardest positives (lowest name similarity)")
rep.table(pp.sort("n_tset").head(30).select(cols), floatfmt="{:.0f}")
rep.h("Most confusable hard negatives (highest name similarity)")
rep.table(P.filter(pl.col("kind") == "hard").sort("n_tset", descending=True).head(30).select(cols),
          floatfmt="{:.0f}")

# ---- LightGBM baseline + entity-level F0.5 ---------------------------------------------
rep.h("LightGBM baseline on these features")
feat = list(F)
X, y = P.select(feat).to_numpy(), P["label"].to_numpy()
groups = P["s1_id"].to_numpy()
oof = np.zeros(len(P))
imp = np.zeros(len(feat))
for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(X[tr], y[tr])
    oof[va] = m.predict_proba(X[va])[:, 1]
    imp += m.booster_.feature_importance("gain")
P = P.with_columns(p=pl.Series(oof))
hard_mask = P["kind"].is_in(["pos", "hard"]).to_numpy()
rep.p(f"OOF AUC all = {roc_auc_score(y, oof):.4f}; pos vs hard = "
      f"{roc_auc_score(y[hard_mask], oof[hard_mask]):.4f}")
rep.table(pl.DataFrame({"feature": feat, "gain_share": imp / imp.sum()})
          .sort("gain_share", descending=True))


def macro_f05(df: pl.DataFrame, t: float, ents: pl.DataFrame = None) -> float:
    """Per-S1 F0.5 exactly as scored: singleton with no prediction = 1, else 0 when empty.
    `ents` = the S1 entities being averaged over (defaults to the whole sample)."""
    ents = s1 if ents is None else ents
    e = (df.with_columns(pred=pl.col("p") >= t)
         .group_by("s1_id").agg(
             tp=(pl.col("pred") & (pl.col("label") == 1)).sum(),
             fp=(pl.col("pred") & (pl.col("label") == 0)).sum()))
    e = e.join(ents.select("s1_id", "n_matches"), on="s1_id", how="right").fill_null(0)
    tp, fp, fn = e["tp"].to_numpy(), e["fp"].to_numpy(), e["n_matches"].to_numpy() - e["tp"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        pr, rc = tp / (tp + fp), tp / (tp + fn)
        f = 1.25 * pr * rc / (0.25 * pr + rc)
    f = np.nan_to_num(f)
    f[(e["n_matches"].to_numpy() == 0) & (fp == 0)] = 1.0
    return float(f.mean())


rep.p("Macro F0.5 over the sampled S1 entities, treating pos+hard+random as each entity's "
      "candidate set. **Optimistic**: assumes blocking finds every true match and the only "
      "confusers are these hard negatives.")
sweep = pl.DataFrame({"threshold": np.round(np.arange(0.1, 0.96, 0.05), 2)})
sweep = sweep.with_columns(macro_f05=pl.Series([macro_f05(P, t) for t in sweep["threshold"]]))
by_c = {cn: [macro_f05(P.filter(pl.col("country") == cn), t, s1.filter(pl.col("country") == cn))
             for t in sweep["threshold"]]
        for cn in sorted(P["country"].unique().to_list())}
rep.table(sweep.with_columns(**{k: pl.Series(v) for k, v in by_c.items()}), floatfmt="{:.4f}")

# ---- singletons ------------------------------------------------------------------------
rep.h("Singletons: how confident is the best (wrong) candidate?")
best = P.group_by("s1_id").agg(best_neg=pl.col("p").filter(pl.col("label") == 0).max(),
                               best_pos=pl.col("p").filter(pl.col("label") == 1).max())
best = best.join(s1.select("s1_id", "country", "n_matches"), on="s1_id")
best = best.with_columns(singleton=pl.col("n_matches") == 0)
rep.table(best.group_by("country", "singleton").agg(
    n=pl.len(), best_neg_p50=pl.col("best_neg").median(), best_neg_p90=pl.col("best_neg").quantile(0.9),
    best_pos_p10=pl.col("best_pos").quantile(0.1), best_pos_p50=pl.col("best_pos").median(),
).sort("country", "singleton"))

rep.save()
