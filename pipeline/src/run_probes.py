"""Strict-threshold submissions from a finished run's test scores (the LB says test is precision-limited: v3 at
threshold 0.30 scored 0.958, v3's rule 0.975, threshold 0.95 scored 0.978). One owner per record, then p >= t.
  PROBE_T       thresholds, e.g. "0.95,0.97,0.98"
  PROBE_SCORES  score file under work/test (default scored3.parquet = v4 stage 3; scored.parquet = v3 stage 2)
  PROBE_SRC     optional s3://.../work of another run: only test/s1.parquet and the score file are downloaded
Candidates = every scored pair (the run's shortlist). Writes output/probe_{tag}_tNNNN/."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_HALF = str(max(1, (os.cpu_count() or 2) // 2))
for _v in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, _HALF)

import polars as pl  # noqa: E402

from ber import decide  # noqa: E402
from ber.config import OUT, WORK, log  # noqa: E402
from ber.predict import _write_lists  # noqa: E402

if __name__ == "__main__":
    name = os.environ.get("PROBE_SCORES", "scored3.parquet")
    tag = os.environ.get("PROBE_TAG", "v4")
    src = os.environ.get("PROBE_SRC")
    if src:
        import boto3
        bucket, prefix = src.removeprefix("s3://").split("/", 1)
        for f in ("test/s1.parquet", f"test/{name}"):
            (WORK / f).parent.mkdir(parents=True, exist_ok=True)
            boto3.client("s3").download_file(bucket, f"{prefix.rstrip('/')}/{f}", str(WORK / f))
    s1 = pl.read_parquet(WORK / "test" / "s1.parquet", columns=["iid", "country"])
    sc = pl.read_parquet(WORK / "test" / name)
    for t in [float(x) for x in os.environ.get("PROBE_T", "0.95,0.97,0.98").split(",")]:
        rule = {"kind": "thresh", "t": t, "floor": 0.0, "one_owner": True}
        chosen = decide.select(sc, rule)
        out = OUT / f"probe_{tag}_t{int(round(t * 1000)):04d}"
        out.mkdir(parents=True, exist_ok=True)
        _write_lists(s1, chosen, "matched_entity_ids", out / "matching_results.tsv")
        _write_lists(s1, sc.select("a", "b"), "candidate_entity_ids", out / "candidate_pairs.tsv")
        per = s1.join(chosen.group_by("a").len("k"), left_on="iid", right_on="a", how="left").fill_null(0)
        stats = {c: {"empty_rate": round(float((g["k"] == 0).mean()), 4), "matches_per_s1": round(float(g["k"].mean()), 4)}
                 for (c,), g in per.group_by("country")}
        (out / "run_summary.json").write_text(json.dumps({"scores": name, "rule": rule, "matches": len(chosen),
                                                          "by_country": stats}, indent=2))
        log(f"probe {tag} t={t}: {len(chosen):,} matches ({len(chosen) / len(s1):.3f}/S1) {json.dumps(stats)}")
