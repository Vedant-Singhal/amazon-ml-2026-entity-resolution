"""01 - Data profile: field quality, scripts, noise markers, GT structure, train vs test drift.

Full-data scan with polars. Output: eda/reports/01_profile.md
"""
import polars as pl

from common import SCRIPTS, Report, load_gt, load_split, log, pl_fold

rep = Report("01_profile", "01 · Data profile")


def field_stats(df: pl.DataFrame, split: str, source: str) -> pl.DataFrame:
    n, a = pl.col("business_name"), pl.col("business_address")
    return (
        df.group_by("country")
        .agg(
            rows=pl.len(),
            name_empty=(n == "").mean(),
            addr_empty=(a == "").mean(),
            name_len_p50=n.str.len_chars().median(),
            name_tokens_mean=n.str.split(" ").list.len().mean(),
            addr_len_p50=a.str.len_chars().median(),
            addr_parts_mean=a.str.split(",").list.len().mean(),
            name_non_ascii=(~n.str.contains(r"^[\x00-\x7F]*$")).mean(),
            name_non_latin=n.str.contains(r"[^\p{Latin}\p{Common}\p{Inherited}]").mean(),
            addr_non_latin=a.str.contains(r"[^\p{Latin}\p{Common}\p{Inherited}]").mean(),
            name_all_caps=(n.str.contains(r"\p{Lu}") & ~n.str.contains(r"\p{Ll}")).mean(),
            name_domain=n.str.contains(r"(?i)\.(com|net|org|in|fr|co|biz|info)\b").mean(),
            name_compact_no_space=(~n.str.contains(" ") & (n.str.len_chars() > 12)).mean(),
            name_has_digit=n.str.contains(r"\d").mean(),
            name_paren=n.str.contains(r"[\(\)\[\]]").mean(),
            name_dba=n.str.contains(r"(?i)\b(dba|d/b/a|aka|formerly|fka|t/a)\b").mean(),
            name_junk_prefix=n.str.contains(r"^[^\p{L}\p{N}]").mean(),
            addr_null_token=a.str.contains(r"(?i)\bnull\b").mean(),
            addr_5digit=a.str.contains(r"\b\d{5}\b").mean(),
            addr_6digit=a.str.contains(r"\b\d{6}\b").mean(),
            addr_has_number=a.str.contains(r"\d").mean(),
            addr_po_box=a.str.contains(r"(?i)\bp\.?\s?o\.?\s?box\b|\bpmb\b").mean(),
            addr_near=a.str.contains(r"(?i)\b(near|opp|opposite|behind|beside|nr)\b").mean(),
        )
        .with_columns(split=pl.lit(split), source=pl.lit(source))
        .sort("country")
    )


def script_mix(df: pl.DataFrame, col: str, split: str, source: str) -> pl.DataFrame:
    return (
        df.group_by("country")
        .agg(**{s: pl.col(col).str.contains(rf"\p{{{s}}}").mean() for s in SCRIPTS})
        .with_columns(split=pl.lit(split), source=pl.lit(source), field=pl.lit(col))
    )


def top_tokens(df: pl.DataFrame, col: str, k=25) -> pl.DataFrame:
    toks = df.select("country", tok=pl_fold(col).str.split(" ")).explode("tok")
    toks = toks.filter(pl.col("tok").str.len_chars() > 0)
    out = []
    for (c,), g in toks.group_by("country"):
        vc = g["tok"].value_counts(sort=True).head(k)
        out.append(pl.DataFrame({"country": c, "tokens": ", ".join(
            f"{t}({n / len(df.filter(pl.col('country') == c)):.1%})" for t, n in vc.iter_rows())}))
    return pl.concat(out).sort("country")


# ---- load --------------------------------------------------------------------
log("loading train")
tr1, tr_o = load_split("train")
log("loading test")
te1, te_o = load_split("test")
gt, pairs = load_gt()

frames = {("train", "S1"): tr1, ("train", "S2"): tr_o.filter(pl.col("src") == "S2"),
          ("train", "S3"): tr_o.filter(pl.col("src") == "S3"), ("test", "S1"): te1,
          ("test", "S2"): te_o.filter(pl.col("src") == "S2"),
          ("test", "S3"): te_o.filter(pl.col("src") == "S3")}

# ---- field quality -------------------------------------------------------------
rep.h("Field quality and noise markers (fraction of rows)")
rep.p("Per split/source/country. `name_non_latin` = contains a non-Latin script letter; "
      "`name_domain` = looks like a web domain; `addr_5digit`/`addr_6digit` = US ZIP / India PIN "
      "candidates; `addr_near` = landmark-style references.")
stats = pl.concat([field_stats(df, s, src) for (s, src), df in frames.items()])
lead = ["split", "source", "country"]
stats = stats.select(lead + [c for c in stats.columns if c not in lead])
rep.table(stats.select(lead + stats.columns[3:14]))
rep.table(stats.select(lead + stats.columns[14:]))

rep.h("Script mix (fraction of rows containing each script)")
mix = pl.concat([script_mix(df, col, s, src) for (s, src), df in frames.items()
                 for col in ("business_name", "business_address")])
mix = mix.select(["split", "source", "field", "country"] + SCRIPTS)
rep.table(mix.filter(pl.col("country") != "US"))

rep.h("Most frequent name tokens (train S1 vs S2 vs S3, and test France)")
for key in [("train", "S1"), ("train", "S2"), ("train", "S3"), ("test", "S2")]:
    rep.p(f"**{key[0]} {key[1]}**")
    rep.table(top_tokens(frames[key], "business_name"))

rep.h("Most frequent address tokens")
for key in [("train", "S1"), ("train", "S2"), ("train", "S3")]:
    rep.p(f"**{key[0]} {key[1]}**")
    rep.table(top_tokens(frames[key], "business_address"))

# ---- ground truth structure ------------------------------------------------------
rep.h("Ground truth: matches per S1 entity")
g = gt.join(tr1.select(s1_id="entity_id", country="country"), on="s1_id")
per = pairs.group_by("s1_id", "src").len().pivot(on="src", index="s1_id", values="len")
g = g.join(per, on="s1_id", how="left").fill_null(0)
rep.table(
    g.group_by("country").agg(
        entities=pl.len(), singleton_rate=(pl.col("n_matches") == 0).mean(),
        mean_matches=pl.col("n_matches").mean(), p90=pl.col("n_matches").quantile(0.9),
        max=pl.col("n_matches").max(), mean_S2=pl.col("S2").mean(), mean_S3=pl.col("S3").mean(),
        no_S2=(pl.col("S2") == 0).mean(), no_S3=(pl.col("S3") == 0).mean(),
    ).sort("country"))
hist = g.with_columns(k=pl.col("n_matches").clip(0, 12)).group_by("country", "k").len()
rep.table(hist.pivot(on="country", index="k", values="len").sort("k"))

rep.h("Ground truth: consistency checks")
multi = pairs.group_by("id").len().filter(pl.col("len") > 1)
known = set(tr_o["entity_id"].to_list())
pc = pairs.join(tr1.select(s1_id="entity_id", c1="country"), on="s1_id").join(
    tr_o.select(id="entity_id", c2="country", o_addr="business_address"), on="id", how="left")
rep.p(f"- GT pairs: {len(pairs):,}")
rep.p(f"- S2/S3 ids matched to more than one S1: {len(multi):,} (0 ⇒ clusters are disjoint)")
rep.p(f"- GT ids missing from source files: {pairs.filter(~pl.col('id').is_in(known)).height:,}")
rep.p(f"- GT pairs with country mismatch: {(pc['c1'] != pc['c2']).sum():,} "
      f"(0 ⇒ country can be a hard blocking constraint)")
rep.p(f"- GT pairs whose S2/S3 record has an empty address: {(pc['o_addr'] == '').mean():.1%}")

rep.h("Distractors: S2/S3 records that match nothing")
matched = pairs.select("id").unique().with_columns(m=pl.lit(True))
d = tr_o.join(matched, left_on="entity_id", right_on="id", how="left")
rep.table(d.group_by("src", "country").agg(
    records=pl.len(), unmatched_rate=pl.col("m").is_null().mean()).sort("src", "country"))

rep.h("Exact duplicates")
for (s, src), df in frames.items():
    nm = df.select(k=pl_fold("business_name"))
    na = df.select(k=pl_fold("business_name") + "|" + pl_fold("business_address"))
    rep.p(f"- {s} {src}: duplicate folded names {1 - nm['k'].n_unique() / len(df):.2%}, "
          f"duplicate name+address {1 - na['k'].n_unique() / len(df):.2%}")

rep.h("Train → test drift")
rep.p("Country mix of S1 and the S2+S3 : S1 ratio (candidate pool density).")
dens = []
for s, one, oth in [("train", tr1, tr_o), ("test", te1, te_o)]:
    a = one.group_by("country").len().rename({"len": "s1"})
    b = oth.group_by("country").len().rename({"len": "others"})
    dens.append(a.join(b, on="country").with_columns(
        split=pl.lit(s), ratio=pl.col("others") / pl.col("s1")))
rep.table(pl.concat(dens).select("split", "country", "s1", "others", "ratio").sort("split", "country"))

rep.save()
