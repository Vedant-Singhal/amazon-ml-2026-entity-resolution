"""Shared loading + normalization for the EDA scripts."""
import os
import re
import time
import unicodedata
from pathlib import Path

import polars as pl
from anyascii import anyascii

EDA = Path(__file__).resolve().parent
# Locally: ../dataset and eda/reports. On SageMaker: the input channel and the model dir
# (uploaded to S3 when the job ends).
DATA = Path(os.environ.get("SM_CHANNEL_DATASET", EDA.parent / "dataset"))
REPORTS = Path(os.environ.get("EDA_REPORTS") or Path(os.environ.get("SM_MODEL_DIR", EDA)) / "reports")
CACHE = Path(os.environ.get("EDA_CACHE", EDA / "cache"))
REPORTS.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
# EDA_SMOKE=1: tiny subsample, to shake out bugs before a full run.
SMOKE = os.environ.get("EDA_SMOKE") == "1"

# Legal-form tokens across US / India / France. Stripped for "core name" comparisons.
LEGAL = {
    "inc", "incorporated", "llc", "l", "c", "ltd", "limited", "pvt", "private", "corp",
    "corporation", "co", "company", "pllc", "llp", "lp", "plc", "pc", "sarl", "sas", "sasu",
    "sa", "eurl", "sci", "snc", "gmbh", "opc", "the", "and", "et", "of", "de", "du", "la", "le",
    "les", "des",
}
# Words so common they make useless blocking keys (still kept for similarity).
GENERIC = {
    "services", "service", "solutions", "group", "center", "centre", "enterprises", "enterprise",
    "international", "india", "france", "us", "usa", "america", "american", "global", "trading",
    "industries", "technologies", "technology", "consultants", "consulting", "associates",
    "holdings", "partners", "management", "company", "business", "systems", "infra", "groupe",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_tsv(path) -> pl.DataFrame:
    # quote_char=None: fields contain literal '"' characters; files are plain tab-split.
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                     n_rows=300_000 if SMOKE else None)
    return df.with_columns(pl.all().fill_null(""))


def load_split(split: str):
    """Return (s1, others) where others = S2 + S3 with a `src` column."""
    s1 = read_tsv(DATA / split / f"{split}_source1.tsv")
    s2 = read_tsv(DATA / split / f"{split}_source2.tsv").with_columns(src=pl.lit("S2"))
    s3 = read_tsv(DATA / split / f"{split}_source3.tsv").with_columns(src=pl.lit("S3"))
    return s1, pl.concat([s2, s3])


def load_gt():
    """Return (gt, pairs): gt has n_matches per S1; pairs has one row per (s1_id, id)."""
    gt = read_tsv(DATA / "train" / "train_ground_truth.tsv")
    gt = gt.with_columns(
        ids=pl.when(pl.col("matched_entity_ids") == "")
        .then(pl.lit([], dtype=pl.List(pl.String)))
        .otherwise(pl.col("matched_entity_ids").str.split(","))
    ).with_columns(n_matches=pl.col("ids").list.len())
    pairs = (
        gt.select(s1_id="source1_entity_id", id="ids")
        .explode("id")
        .drop_nulls("id")
        .with_columns(src=pl.col("id").str.slice(0, 2))
    )
    return gt.select(s1_id="source1_entity_id", n_matches="n_matches"), pairs


# ---- normalization -----------------------------------------------------------
_COMBINING = re.compile(r"[̀-ͯ]")   # Latin diacritics only; Indic vowel signs are kept
_NON_WORD = re.compile(r"[^\wऀ-෿]+")   # \w excludes Indic vowel signs; keep them
_NULLISH = re.compile(r"<?\bnull\b>?|\bn/?a\b", re.I)


def fold(s: str) -> str:
    """Lowercase, strip Latin accents, punctuation -> space. Keeps non-Latin scripts."""
    s = _COMBINING.sub("", unicodedata.normalize("NFKD", s or ""))
    return " ".join(_NON_WORD.sub(" ", s.lower()).replace("_", " ").split())


def translit(s: str) -> str:
    """fold() after transliterating any non-Latin script to ASCII (anyascii)."""
    return fold(anyascii(s or "")) if s and not s.isascii() else fold(s)


def name_core(s: str) -> str:
    """Transliterated name without legal forms / filler words."""
    return " ".join(t for t in translit(s).split() if t not in LEGAL)


def addr_clean(s: str) -> str:
    return translit(_NULLISH.sub(" ", s or ""))


_NUM = re.compile(r"\d+")


def addr_numbers(s: str) -> frozenset:
    return frozenset(_NUM.findall(s or ""))


# Polars expression versions (fast, for full-data scans; no transliteration).
def pl_fold(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .str.normalize("NFKD")
        .str.replace_all(r"[̀-ͯ]", "")
        .str.to_lowercase()
        .str.replace_all(r"[^\p{L}\p{N}\p{M}]+", " ")  # \p{M}: keep Indic vowel signs
        .str.strip_chars()
    )


SCRIPTS = ["Latin", "Devanagari", "Bengali", "Gujarati", "Gurmukhi", "Tamil", "Telugu",
           "Kannada", "Malayalam", "Oriya", "Arabic"]


def md_table(df: pl.DataFrame, floatfmt="{:.3f}") -> str:
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


class Report:
    """Accumulates markdown sections and writes them to eda/reports/<name>.md."""

    def __init__(self, name: str, title: str):
        self.path = REPORTS / f"{name}.md"
        self.parts = [f"# {title}\n"]

    def h(self, text):
        self.parts.append(f"\n## {text}\n")
        log(text)

    def p(self, text):
        self.parts.append(text + "\n")

    def table(self, df: pl.DataFrame, **kw):
        self.parts.append(md_table(df, **kw) + "\n")

    def save(self):
        self.path.write_text("\n".join(self.parts), encoding="utf-8")
        log(f"wrote {self.path}")
