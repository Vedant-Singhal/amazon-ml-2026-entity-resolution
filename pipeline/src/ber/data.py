"""Loading the provided TSVs. Integer ids keep joins fast: S1-123 -> 123, S2-123 -> 123,
S3-123 -> 10**10 + 123 (S2/S3 share one id space; S1 has its own)."""
import polars as pl

from .config import DATA, P, SMOKE, SMOKE_S1

S3_OFFSET = 10**10


def read_tsv(path) -> pl.DataFrame:
    # Fields contain literal '"' characters: disable quoting; files are plain tab-split.
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return df.with_columns(pl.all().fill_null(""))


def int_id(col: str) -> pl.Expr:
    num = pl.col(col).str.slice(3).cast(pl.Int64)
    return pl.when(pl.col(col).str.starts_with("S3")).then(num + S3_OFFSET).otherwise(num)


def str_id(col: str, source1: bool = False) -> pl.Expr:
    if source1:
        return pl.lit("S1-") + pl.col(col).cast(pl.String)
    return (pl.when(pl.col(col) >= S3_OFFSET)
            .then(pl.lit("S3-") + (pl.col(col) - S3_OFFSET).cast(pl.String))
            .otherwise(pl.lit("S2-") + pl.col(col).cast(pl.String)))


def smoke_frac() -> float:
    """Smoke runs keep the same share f of every split (S1 and S2/S3), so densities stay realistic."""
    return min(1.0, SMOKE_S1 / 2_206_821)


def _keep(col: str, f: float) -> pl.Expr:
    return (pl.col(col).hash(P.seed) % 1_000_000) < int(f * 1_000_000)


def load_split(split: str):
    """Return (s1, oth): columns iid, entity_id, name, addr, country (+ src on oth)."""
    def load(n):
        df = read_tsv(DATA / split / f"{split}_source{n}.tsv")
        return df.select(iid=int_id("entity_id"), entity_id="entity_id", name="business_name",
                         addr="business_address", country="country")
    s1 = load(1)
    oth = pl.concat([load(2).with_columns(src=pl.lit("S2")), load(3).with_columns(src=pl.lit("S3"))])
    if SMOKE:
        f = smoke_frac()
        s1 = s1.filter(_keep("iid", f))
        keep = oth.filter(_keep("iid", f))["iid"]
        if split == "train":                     # plus every true match of the sampled S1
            pairs, _ = load_gt_pairs()
            keep = pl.concat([keep, pairs.filter(pl.col("a").is_in(s1["iid"].implode()))["b"]])
        oth = oth.filter(pl.col("iid").is_in(keep.implode()))
    return s1, oth


def load_gt_pairs():
    """Ground-truth pairs (a = S1 iid, b = S2/S3 iid) plus n_true per S1 (incl. zeros).
    Not subsampled in smoke mode: callers restrict to the S1s they loaded."""
    gt = read_tsv(DATA / "train" / "train_ground_truth.tsv")
    pairs = (gt.select(s1="source1_entity_id", ids=pl.col("matched_entity_ids").str.split(","))
             .explode("ids").filter(pl.col("ids").is_not_null() & (pl.col("ids") != ""))
             .select(a=int_id("s1"), b=int_id("ids")))
    n_true = gt.select(a=int_id("source1_entity_id")).join(
        pairs.group_by("a").len().rename({"len": "n_true"}), on="a", how="left").fill_null(0)
    return pairs, n_true


def gt_for(s1_ids: pl.Series):
    """GT restricted to the given S1 ids (smoke / ablation samples)."""
    pairs, n_true = load_gt_pairs()
    n_true = n_true.filter(pl.col("a").is_in(s1_ids.implode()))
    return pairs.join(n_true.select("a"), on="a"), n_true


def fold_of(col: str = "a") -> pl.Expr:
    """GroupKFold by S1: the same fold assignment as v2 (ensembling across versions needs it)."""
    return (pl.col(col).hash(seed=P.seed) % P.cv_folds).cast(pl.Int32)


def load_pseudo_pairs(src: str) -> pl.DataFrame:
    """(a, b) test pairs from a previous run's high-confidence submission file (source1_entity_id, matched_entity_ids),
    e.g. v4's threshold-0.99 file (~99% right). Used only to learn test-side normalization maps (French abbreviations,
    region/department names, legal forms) — no labels. `src` is a local path or s3://bucket/key."""
    from pathlib import Path
    from .config import WORK
    path = Path(src)
    if src.startswith("s3://"):
        import boto3
        bucket, key = src[5:].split("/", 1)
        path = WORK / "pseudo_pairs.tsv"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            boto3.client("s3").download_file(bucket, key, str(path))
    m = read_tsv(path)
    return (m.select(s1="source1_entity_id", ids=pl.col("matched_entity_ids").str.split(","))
            .explode("ids").filter(pl.col("ids").is_not_null() & (pl.col("ids") != ""))
            .select(a=int_id("s1"), b=int_id("ids")))
