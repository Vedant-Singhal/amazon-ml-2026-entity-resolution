"""Diagnosis of a finished (or partly finished) run: where do we lose score?

Reads a run's stage outputs from S3 (only the files needed) and writes diagnosis.md:
  A. candidate misses — GT pairs the blocking never produced, by country x source x noise type
  B. (when OOF predictions exist) false negatives / false positives of the tuned decision rule:
     per-entity F0.5 by country, precision/recall by source (S2/S3) and noise type,
     where FNs come from (blocking / below threshold / one-owner), FP patterns, examples.
Noise type per pair (first that applies): native_script, empty_addr, name_unrelated,
house_no_diff, name_typo, clean.
"""
import json
import os
from pathlib import Path

import boto3
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from .config import P, log

BUCKET = os.environ.get("BER_BUCKET", "")          # set by launch.ps1 from sagemaker/account.ps1
LOCAL = Path("/tmp/diag")
OUT = Path(os.environ.get("BER_OUT", "/opt/ml/checkpoints/output"))
FIELDS = ["iid", "country", "name", "addr", "core", "addr_n", "first_num", "has_addr"]
NATIVE = r"[^\p{Latin}\p{Common}\p{Inherited}]"


def _get(run: str, key: str):
    dst = LOCAL / key
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        boto3.client("s3").download_file(BUCKET, f"{os.environ.get('BER_PREFIX', 'ber')}/ber-runs/{run}/work/{key}", str(dst))
        return dst
    except Exception as e:  # noqa: BLE001 - optional files (OOF) may not exist yet
        log(f"diagnose: {key} not available ({type(e).__name__})")
        return None


def _typed(pairs: pl.DataFrame, s1: pl.DataFrame, oth: pl.DataFrame) -> pl.DataFrame:
    """pairs(a, b, ...) -> + country, src, noise type, name/address similarity, raw text."""
    d = (pairs.join(s1.rename({c: f"{c}_a" for c in FIELDS if c != "iid"}), left_on="a", right_on="iid")
         .join(oth.rename({c: f"{c}_b" for c in FIELDS if c != "iid"}), left_on="b", right_on="iid"))
    d = d.with_columns(country=pl.col("country_a"),
                       src=pl.when(pl.col("b") >= 10**10).then(pl.lit("S3")).otherwise(pl.lit("S2")))
    cp = lambda x, y, f: process.cpdist(d[x].fill_null("").to_list(), d[y].fill_null("").to_list(),
                                        scorer=f, workers=P.workers)
    d = d.with_columns(name_sim=pl.Series(cp("core_a", "core_b", fuzz.token_set_ratio)),
                       addr_sim=pl.Series(cp("addr_n_a", "addr_n_b", fuzz.token_set_ratio)))
    return d.with_columns(type=pl.when(pl.col("name_b").str.contains(NATIVE)).then(pl.lit("native_script"))
                          .when(~pl.col("has_addr_b")).then(pl.lit("empty_addr"))
                          .when(pl.col("name_sim") < 50).then(pl.lit("name_unrelated"))
                          .when((pl.col("first_num_a") != "") & (pl.col("first_num_b") != "")
                                & (pl.col("first_num_a") != pl.col("first_num_b"))).then(pl.lit("house_no_diff"))
                          .when(pl.col("name_sim") < 85).then(pl.lit("name_typo"))
                          .otherwise(pl.lit("clean")))


def _table(df: pl.DataFrame) -> str:
    cols = df.columns
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows():
        rows.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(rows)


def _examples(d: pl.DataFrame, n=12) -> str:
    cols = ["country", "src", "type", "name_a", "addr_a", "name_b", "addr_b"]
    return _table(d.sample(n=min(n, len(d)), seed=0).select(cols)) if len(d) else "_none_"


def run(src_run: str) -> None:
    from .data import load_gt_pairs
    rep = [f"# Diagnosis of run `{src_run}`\n"]
    s1 = pl.read_parquet(_get(src_run, "train/s1.parquet"), columns=FIELDS)
    oth = pl.read_parquet(_get(src_run, "train/oth.parquet"), columns=FIELDS)
    cands = pl.read_parquet(_get(src_run, "train/cands.parquet"), columns=["a", "b"])
    pairs, n_true = load_gt_pairs()
    log("diagnose: loaded")

    # ---- A. candidate misses ------------------------------------------------------------
    gt = _typed(pairs, s1, oth).join(cands.with_columns(in_cands=pl.lit(True)), on=["a", "b"], how="left") \
        .with_columns(pl.col("in_cands").fill_null(False))
    rep.append("## A. Candidate recall (blocking) by country × source × noise type\n")
    rep.append(_table(gt.group_by("country", "src").agg(
        pairs=pl.len(), recall=pl.col("in_cands").mean()).sort("country", "src")))
    by_type = gt.group_by("country", "type").agg(pairs=pl.len(), share=pl.len(), recall=pl.col("in_cands").mean(),
                                                 misses=(~pl.col("in_cands")).sum())
    by_type = by_type.with_columns(share=pl.col("pairs") / pl.col("pairs").sum().over("country"),
                                   miss_share=pl.col("misses") / pl.col("misses").sum().over("country"))
    rep.append("\n" + _table(by_type.sort("country", "misses", descending=[False, True])))
    rep.append("\n**Examples of missed pairs**\n\n" + _examples(gt.filter(~pl.col("in_cands")), 25))

    # ---- B. model errors (needs OOF) ----------------------------------------------------------
    oof_p, dec_p = _get(src_run, "model/oof.parquet"), _get(src_run, "model/decision.json")
    if oof_p and dec_p:
        from . import decide
        rule = json.loads(Path(dec_p).read_text())["best"]
        oof = pl.read_parquet(oof_p, columns=["a", "b", "p"]).filter(pl.col("p") >= 0.02)
        chosen = decide.select(oof, rule).with_columns(pred=pl.lit(True))
        nt = n_true.join(s1.select(a="iid"), on="a")
        rep.append(f"\n## B. Model errors with the tuned rule `{json.dumps(rule)}`\n")
        cty = s1.select(a="iid", country="country")
        rows = []
        for (c,), g in cty.group_by("country"):
            ntc = nt.join(g.select("a"), on="a")
            rows.append({"country": c, "macro_f05": decide.macro_f05(chosen.join(ntc.select("a"), on="a").select("a", "b"),
                                                                     pairs, ntc), "entities": len(ntc)})
        rep.append(_table(pl.DataFrame(rows)))

        # False negatives: GT pairs not predicted — why?
        above = oof.filter(pl.col("p") >= rule.get("t", rule.get("floor", 0.0))).select("a", "b").with_columns(above=pl.lit(True))
        fn = (gt.join(chosen.select("a", "b", "pred"), on=["a", "b"], how="left")
              .join(above, on=["a", "b"], how="left")
              .with_columns(pl.col("pred").fill_null(False), pl.col("above").fill_null(False)))
        fn = fn.with_columns(cause=pl.when(pl.col("pred")).then(pl.lit("found"))
                             .when(~pl.col("in_cands")).then(pl.lit("missed_by_blocking"))
                             .when(pl.col("above")).then(pl.lit("lost_to_one_owner_or_set_rule"))
                             .otherwise(pl.lit("below_threshold")))
        rep.append("\n### Recall by country × source, and why true pairs are lost\n")
        rep.append(_table(fn.group_by("country", "src").agg(pairs=pl.len(), recall=pl.col("pred").mean())
                          .sort("country", "src")))
        rep.append("\n" + _table(fn.group_by("country", "type", "cause").len().sort("country", "len", descending=[False, True])))
        rep.append("\n**Examples of true pairs predicted as non-match (candidates, below threshold)**\n\n"
                   + _examples(fn.filter(pl.col("cause") == "below_threshold"), 20))

        # False positives: predicted pairs that are not true — patterns.
        fp = chosen.select("a", "b").join(pairs, on=["a", "b"], how="anti")
        fp = _typed(fp, s1, oth).join(nt, on="a")
        fp = fp.with_columns(pattern=pl.when(pl.col("n_true") == 0).then(pl.lit("singleton_false_merge"))
                             .when((pl.col("name_sim") >= 90) & (pl.col("addr_sim") < 60)).then(pl.lit("same_name_diff_addr"))
                             .when((pl.col("addr_sim") >= 90) & (pl.col("name_sim") < 60)).then(pl.lit("same_addr_diff_name"))
                             .when(pl.col("type") == "house_no_diff").then(pl.lit("near_twin_number"))
                             .otherwise(pl.lit("other")))
        prec = (chosen.select("a", "b").join(pairs.with_columns(t=pl.lit(True)), on=["a", "b"], how="left")
                .with_columns(pl.col("t").fill_null(False)).join(cty, on="a")
                .with_columns(src=pl.when(pl.col("b") >= 10**10).then(pl.lit("S3")).otherwise(pl.lit("S2"))))
        rep.append("\n### Precision by country × source\n")
        rep.append(_table(prec.group_by("country", "src").agg(predicted=pl.len(), precision=pl.col("t").mean())
                          .sort("country", "src")))
        rep.append("\n### False-positive patterns\n")
        rep.append(_table(fp.group_by("country", "pattern").len().sort("country", "len", descending=[False, True])))
        rep.append("\n**Examples of false positives**\n\n" + _examples(fp, 20))
    else:
        rep.append("\n_Part B skipped: OOF predictions not available yet._")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnosis.md").write_text("\n".join(rep), encoding="utf-8")
    log(f"diagnose: wrote {OUT / 'diagnosis.md'}")
