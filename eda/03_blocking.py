"""03 - Blocking study on the FULL training set.

For each blocking key family, records sharing a key (within the same country) become
candidates. For every GT pair we find the smallest block (on the S2/S3 side) it shares, so one
pass gives recall at every block-size cap. Candidate volume is reported as sum(n_s1 * n_other)
over kept blocks (an upper bound: pairs sharing several keys are counted more than once).
Output: eda/reports/03_blocking.md
"""
import jellyfish
import polars as pl

from common import GENERIC, LEGAL, Report, load_gt, load_split, log, pl_fold, translit

CAPS = [None, 20000, 5000, 1000, 300, 100]
UNION_CAP = 1000
ADDR_STOP = {
    "road", "street", "avenue", "drive", "lane", "floor", "near", "null", "plot", "house",
    "block", "sector", "india", "main", "cross", "nagar", "colony", "building", "shop", "unit",
    "suite", "court", "place", "boulevard", "circle", "highway", "north", "south", "east", "west",
    "rue", "avenue", "chemin", "allee", "impasse", "route", "etage", "office", "first", "second",
}
NAME_STOP = sorted(LEGAL | GENERIC | {"www", "com", "net", "org"})
rep = Report("03_blocking", "03 · Blocking recall vs candidate volume (full train)")


def prep(df: pl.DataFrame) -> pl.DataFrame:
    """Integer ids + transliterated, folded name/address columns."""
    uniq = df.filter(~pl.col("business_name").str.contains(r"^[\x00-\x7F]*$"))["business_name"].unique()
    tl = {x: translit(x) for x in uniq.to_list()}   # only non-ASCII names need transliteration
    return df.select(
        iid=(pl.when(pl.col("entity_id").str.starts_with("S3")).then(10**10).otherwise(0)
             + pl.col("entity_id").str.slice(3).cast(pl.Int64)),
        country="country",
        name=pl.col("business_name").replace(tl),
        addr=pl_fold("business_address").str.replace_all(r"\b(null|n a)\b", " "),
    ).with_columns(name=pl_fold("name"))


def core_tokens(df):
    return (df.select("iid", "country", tok=pl.col("name").str.split(" ")).explode("tok")
            .filter((pl.col("tok").str.len_chars() >= 3) & ~pl.col("tok").is_in(NAME_STOP)))


def key(*parts):
    return pl.concat_str(list(parts), separator="|").hash()


def k_name_tok(df):
    return core_tokens(df).select("iid", k=key(pl.col("country"), pl.col("tok")))


def k_name_pair(df):
    t = (core_tokens(df).with_columns(r=pl.int_range(pl.len()).over("iid"))
         .filter(pl.col("r") < 4))
    j = t.join(t, on=["iid", "country"], suffix="_b").filter(pl.col("tok") < pl.col("tok_b"))
    return j.select("iid", k=key(pl.col("country"), pl.col("tok"), pl.col("tok_b")))


def k_name_prefix(n):
    def f(df):
        core = (pl.col("name").str.split(" ")
                .list.eval(pl.element().filter(~pl.element().is_in(NAME_STOP))).list.join(""))
        return (df.select("iid", "country", p=core.str.slice(0, n))
                .filter(pl.col("p").str.len_chars() >= n)
                .select("iid", k=key(pl.col("country"), pl.col("p"))))
    return f


_phon_cache = {}


def k_name_phonetic(df):
    t = core_tokens(df).filter(pl.col("tok").str.contains(r"^[a-z]+$"))
    vocab = t["tok"].unique().to_list()
    new = [v for v in vocab if v not in _phon_cache]
    _phon_cache.update(zip(new, [jellyfish.metaphone(v) for v in new]))
    return (t.with_columns(ph=pl.col("tok").replace_strict(_phon_cache, return_dtype=pl.String))
            .filter(pl.col("ph").str.len_chars() >= 2)
            .select("iid", k=key(pl.col("country"), pl.lit("ph"), pl.col("ph"))))


def k_addr_num_tok(df):
    a = df.select("iid", "country", num=pl.col("addr").str.extract(r"\b(\d+)"),
                  tok=pl.col("addr").str.split(" ")).drop_nulls("num").explode("tok")
    a = a.filter((pl.col("tok").str.len_chars() >= 4) & pl.col("tok").str.contains(r"^\p{L}+$")
                 & ~pl.col("tok").is_in(sorted(ADDR_STOP)))
    return a.select("iid", k=key(pl.col("country"), pl.col("num"), pl.col("tok"))).unique()


def k_addr_tok_pair(df):
    """Two rarest-looking alphabetic address tokens (street + locality), order-free."""
    a = (df.select("iid", "country", tok=pl.col("addr").str.split(" ")).explode("tok")
         .filter((pl.col("tok").str.len_chars() >= 4) & pl.col("tok").str.contains(r"^\p{L}+$")
                 & ~pl.col("tok").is_in(sorted(ADDR_STOP))).unique()
         .with_columns(r=pl.int_range(pl.len()).over("iid")).filter(pl.col("r") < 5))
    j = a.join(a, on=["iid", "country"], suffix="_b").filter(pl.col("tok") < pl.col("tok_b"))
    return j.select("iid", k=key(pl.col("country"), pl.col("tok"), pl.col("tok_b")))


KEYS = {
    "name_tok": k_name_tok,
    "name_tok_pair": k_name_pair,
    "name_prefix4": k_name_prefix(4),
    "name_prefix6": k_name_prefix(6),
    "name_metaphone": k_name_phonetic,
    "addr_num+tok": k_addr_num_tok,
    "addr_tok_pair": k_addr_tok_pair,
}

# ---- data --------------------------------------------------------------------------
log("loading")
tr1, oth = load_split("train")
gt, pairs = load_gt()
S1, O = prep(tr1), prep(oth)
iid = lambda c: (pl.when(pl.col(c).str.starts_with("S3")).then(10**10).otherwise(0)
                 + pl.col(c).str.slice(3).cast(pl.Int64))
GP = pairs.select(a=iid("s1_id"), b=iid("id"), src="src").join(
    S1.select(a="iid", country="country"), on="a")
n_s1 = S1.group_by("country").len().rename({"len": "n_s1"})
log(f"S1 {len(S1):,}  others {len(O):,}  GT pairs {len(GP):,}")

summary, cover = [], GP.select("a", "b")
for name, fn in KEYS.items():
    log(f"key {name}")
    ks, ko = fn(S1).unique(), fn(O).unique()
    n1 = ks.group_by("k").len().rename({"len": "n1"})
    n2 = ko.group_by("k").len().rename({"len": "n2"})
    blocks = n1.join(n2, on="k")
    # smallest shared block per GT pair
    shared = (GP.select("a", "b").join(ks.rename({"iid": "a"}), on="a")
              .join(ko.rename({"iid": "b"}), on=["b", "k"]).join(n2, on="k")
              .group_by("a", "b").agg(min_n2=pl.col("n2").min()))
    cover = cover.join(shared.rename({"min_n2": name}), on=["a", "b"], how="left")
    for cap in CAPS:
        kept = blocks if cap is None else blocks.filter(pl.col("n2") <= cap)
        vol = int((kept["n1"].cast(pl.Float64) * kept["n2"]).sum())
        rec = (len(shared) if cap is None else (shared["min_n2"] <= cap).sum()) / len(GP)
        summary.append({"key": name, "cap": str(cap or "none"), "pair_recall": rec,
                        "cand_pairs": vol, "cands_per_s1": vol / len(S1),
                        "s1_keyed": ks["iid"].n_unique() / len(S1),
                        "other_keyed": ko["iid"].n_unique() / len(O)})
    del ks, ko, shared

rep.h("Recall and volume per key family and block-size cap")
rep.p("`cap` = drop blocks with more than this many S2/S3 records. `pair_recall` = share of GT "
      "pairs sharing at least one kept block. `cands_per_s1` = candidate volume / #S1 (upper bound).")
rep.table(pl.DataFrame(summary), floatfmt="{:.4f}")

# ---- union ----------------------------------------------------------------------------
rep.h(f"Union of all key families at cap={UNION_CAP}")
hit = {k: (pl.col(k) <= UNION_CAP).fill_null(False) for k in KEYS}
cover = cover.with_columns(**hit).with_columns(any=pl.any_horizontal(list(KEYS)))
cover = cover.join(GP, on=["a", "b"])
rep.p(f"Pair recall (union): **{cover['any'].mean():.4f}**")
rep.table(cover.group_by("country", "src").agg(
    pairs=pl.len(), recall=pl.col("any").mean(),
    **{f"only_{k}": (pl.col(k) & (pl.sum_horizontal(list(KEYS)) == 1)).mean() for k in KEYS},
).sort("country", "src"), floatfmt="{:.4f}")

rep.p("**Greedy key selection** (each step adds the family with the largest recall gain):")
chosen, cur, rows = [], pl.lit(False), []
remaining = list(KEYS)
while remaining:
    gains = {k: cover.select((cur | pl.col(k)).mean()).item() for k in remaining}
    best = max(gains, key=gains.get)
    chosen.append(best)
    remaining.remove(best)
    cur = cur | pl.col(best)
    vol = sum(r["cand_pairs"] for r in summary if r["key"] in chosen and r["cap"] == str(UNION_CAP))
    rows.append({"step": len(chosen), "add": best, "recall": gains[best], "cands_per_s1_ub": vol / len(S1)})
rep.table(pl.DataFrame(rows), floatfmt="{:.4f}")

# entity-level: all matches of an S1 recovered
ent = cover.group_by("a").agg(all_found=pl.col("any").all(), frac=pl.col("any").mean())
rep.p(f"Non-singleton S1 entities with ALL matches recoverable: {ent['all_found'].mean():.4f}; "
      f"mean fraction recoverable: {ent['frac'].mean():.4f}")

rep.h("Sample of GT pairs missed by every key family")
missed = cover.filter(~pl.col("any")).sample(n=min(40, cover.filter(~pl.col("any")).height), seed=0)
ex = (missed.select("a", "b")
      .join(S1.select(a="iid", s1_name="name", s1_addr="addr"), on="a")
      .join(O.select(b="iid", o_name="name", o_addr="addr"), on="b"))
rep.table(ex.select("s1_name", "s1_addr", "o_name", "o_addr"))

# ---- test-set volume (incl. France) ---------------------------------------------------------
rep.h(f"Test set: candidate volume per S1 by country (cap={UNION_CAP})")
te1, te_o = load_split("test")
T1, TO = prep(te1), prep(te_o)
rows = []
for name, fn in KEYS.items():
    ks, ko = fn(T1).unique(), fn(TO).unique()
    b = (ks.group_by("k").len().rename({"len": "n1"})
         .join(ko.group_by("k").len().rename({"len": "n2"}), on="k")
         .filter(pl.col("n2") <= UNION_CAP))
    # attribute each block to a country via its S1 members
    kc = ks.join(T1.select("iid", "country"), on="iid").select("k", "country").unique("k")
    v = b.join(kc, on="k").group_by("country").agg(vol=(pl.col("n1").cast(pl.Float64) * pl.col("n2")).sum())
    cnt = T1.group_by("country").len()
    for c_, vol, n in v.join(cnt, on="country").select("country", "vol", "len").iter_rows():
        rows.append({"key": name, "country": c_, "cands_per_s1": vol / n,
                     "s1_keyed": ks.join(T1.select("iid", "country"), on="iid")
                     .filter(pl.col("country") == c_)["iid"].n_unique() / n})
rep.table(pl.DataFrame(rows).pivot(on="country", index="key", values="cands_per_s1"), floatfmt="{:.1f}")
rep.table(pl.DataFrame(rows).pivot(on="country", index="key", values="s1_keyed"), floatfmt="{:.3f}")

rep.save()
