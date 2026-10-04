"""Inputs v4 stacks on. All scores are out-of-fold on train with the same folds (hash(S1 id, 13) % 5),
so adding them as features is leak-free.

  v3 stage-2   work/model/oof.parquet (a, b, p, p_raw, label, p_any)   work/test/scored.parquet (a, b, p1, p, p_any)
  v2           channel `v2oof` -> oof.parquet (a, b, p, p_raw, label)    channel `v2test` -> scored.parquet (a, b, p)
               (optional: missing files -> the v2 score is simply absent)
"""
import os
from pathlib import Path

import polars as pl

from .config import log, work_dir

V2_OOF = Path(os.environ.get("SM_CHANNEL_V2OOF", os.environ.get("BER_V2_OOF", "/nonexistent"))) / "oof.parquet"
V2_TEST = Path(os.environ.get("SM_CHANNEL_V2TEST", os.environ.get("BER_V2_TEST", "/nonexistent"))) / "scored.parquet"


def v3_scores(split: str) -> pl.DataFrame:
    """(a, b, p2[, label]) — v3 stage-2 calibrated probability (OOF on train)."""
    if split == "train":
        return pl.read_parquet(work_dir("model") / "oof.parquet", columns=["a", "b", "p", "label"]).rename({"p": "p2"})
    return pl.read_parquet(work_dir("test") / "scored.parquet", columns=["a", "b", "p"]).rename({"p": "p2"})


def v2_scores(split: str) -> pl.DataFrame:
    """(a, b, p_v2) or an empty frame when the v2 files are not attached."""
    path = V2_OOF if split == "train" else V2_TEST
    if not path.exists():
        log(f"v4io: no v2 scores for {split} ({path})")
        return pl.DataFrame(schema={"a": pl.Int64, "b": pl.Int64, "p_v2": pl.Float32})
    return pl.read_parquet(path, columns=["a", "b", "p"]).select("a", "b", p_v2=pl.col("p").cast(pl.Float32))


def texts(split: str) -> tuple:
    """(s1, oth) frames with iid and the serialized record text used by the cross-encoder."""
    wd = work_dir(split)
    ser = lambda: (pl.lit("name: ") + pl.col("name") + pl.lit(" | address: ") + pl.col("addr"))
    s1 = pl.read_parquet(wd / "s1.parquet", columns=["iid", "name", "addr"]).select(a="iid", ta=ser())
    oth = pl.read_parquet(wd / "oth.parquet", columns=["iid", "name", "addr"]).select(b="iid", tb=ser())
    return s1, oth
