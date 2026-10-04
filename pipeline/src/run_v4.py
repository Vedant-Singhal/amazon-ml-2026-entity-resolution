"""v4 entry point. v4 = v3 + the best next steps, stacked on v3's cached outputs:

  [v3 stages, cached]  prepare embed block stage1 features train predict   (skipped when their outputs exist)
  retriever            optional (BER_V4_RETRIEVER=1): fine-tuned search route -> re-runs stage1/features/train
  diagnose (v3)        where v3 still loses (pool / shortlist / model; by record type; test profile per country)
  crossenc             Ditto-style fine-tuned cross-encoder on the uncertain band (2-fold cross-fitted)
  graph                S2/S3 cluster-consistency features
  stage3               final stacked matcher (+ v2 score) -> output/*.tsv
  diagnose (v4)        the same diagnosis for the final model

Smoke test: BER_SMOKE_S1=10000 runs everything on a 10k-S1 sample. A full run starts from the raw TSVs; every stage
is cached in work/, so a restart resumes where it stopped. v5 settings: run_v5.py.
"""
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_HALF = str(max(1, int((os.cpu_count() or 2) * float(os.environ.get("BER_CPU_FRAC", "0.5")))))  # share of the machine (default 50%)
for _v in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, _HALF)

import run_pipeline  # noqa: E402
from ber import crossenc, diagnose3, graph, s1drop, stage3  # noqa: E402
from ber.config import WORK, log  # noqa: E402

V3_STAGES = ["prepare", "embed", "block", "stage1", "features", "train"]


def _step(name, fn):
    t0 = time.time()
    fn()
    log(f"== v4 {name} done in {(time.time() - t0) / 60:.1f} min")


# optional: seed work/ from another finished run in S3 (BER_SEED_FROM=s3://bucket/prefix/work)
SEED = ["timing.json", "*/s1.parquet", "*/oth.parquet", "*/df_*.parquet", "*/generic_words.json", "*/lexicon.json",
        "*/emb_s1.npy", "*/emb_oth.npy", "*/pool.parquet", "*/block_stats.json", "*/sl.parquet", "*/s1feats/*",
        "*/feats/*", "test/scored.parquet", "test/any_scored.parquet", "model1/*", "model/*",
        "*/emb2_s1.npy", "*/emb2_oth.npy", "*/sk_*.npy"]
# BER_SEED_SET=pool: only what a rebuild from stage 1 needs, seeded from a run whose pool is final (e.g. v4 with the
# retriever merged): records, pools, the embedding of S2/S3 (graph features) and test's stage-1 features (test pool
# unchanged); everything from stage 1 on is recomputed
SEED_POOL = ["timing.json", "*/s1.parquet", "*/oth.parquet", "*/df_*.parquet", "*/generic_words.json", "*/lexicon.json",
             "*/emb_oth.npy", "*/pool.parquet", "*/block_stats.json", "*/route_ft.parquet", "test/s1feats/*"]
# BER_SEED_SET=reprep (v5): only the pools (+ record embeddings for graph features) — `prepare` runs again with the new
# vocabulary (pseudo-pair lexicon), so records, features and models are all rebuilt; search is not redone
SEED_REPREP = ["timing.json", "*/emb_oth.npy", "*/pool.parquet", "*/block_stats.json", "*/route_ft.parquet"]


def _seed(src: str) -> None:
    """Download v3's cached stage outputs (s3://bucket/prefix/work) into WORK inside AWS — a server-side copy from a
    slow home connection took hours. Files already present with the same size are skipped."""
    import fnmatch
    from concurrent.futures import ThreadPoolExecutor

    import boto3
    from boto3.s3.transfer import TransferConfig

    bucket, prefix = src.removeprefix("s3://").split("/", 1)
    prefix = prefix.rstrip("/") + "/"
    s3 = boto3.client("s3")
    pats = {"pool": SEED_POOL, "reprep": SEED_REPREP}.get(os.environ.get("BER_SEED_SET", ""), SEED)
    todo = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            rel = o["Key"][len(prefix):]
            dst = WORK / rel
            if any(fnmatch.fnmatch(rel, p) for p in pats) and not (dst.exists() and dst.stat().st_size == o["Size"]):
                todo.append((o["Key"], dst, o["Size"]))
    cfg = TransferConfig(multipart_chunksize=64 << 20, max_concurrency=8)
    t0, total = time.time(), sum(t[2] for t in todo)

    def get(t):
        t[1].parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, t[0], str(t[1]), Config=cfg)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(get, todo))
    log(f"seed: {len(todo)} files, {total / 1e9:.1f} GB from {src} in {(time.time() - t0) / 60:.1f} min")


def _clear_after_pool(changed=("train", "test")):
    """A pool changed: remove everything computed from it (stage 1 onward) so it is rebuilt. Stage-1 features of an
    unchanged split stay (they depend only on its own pool); its shortlist / features / scores depend on the models."""
    for split in ("train", "test"):
        if split in changed:
            shutil.rmtree(WORK / split / "s1feats", ignore_errors=True)
        shutil.rmtree(WORK / split / "feats", ignore_errors=True)
        (WORK / split / "sl.parquet").unlink(missing_ok=True)
        (WORK / split / "scored.parquet").unlink(missing_ok=True)
    for d in ("model1", "model", "ce", "graph", "model3"):
        shutil.rmtree(WORK / d, ignore_errors=True)


def main():
    wanted = os.environ.get("BER_V4_STAGES", "diagnose3 crossenc graph stage3 diagnose4").split()
    if os.environ.get("BER_SEED_FROM") and not (WORK / ".seeded").exists():   # once: later restarts keep the merged pool
        _step("seed", lambda: _seed(os.environ["BER_SEED_FROM"]))
        (WORK / ".seeded").write_text(os.environ["BER_SEED_FROM"])
    pooled = all((WORK / sp / "pool.parquet").exists() for sp in ("train", "test"))
    retr = os.environ.get("BER_V4_RETRIEVER") == "1"
    if retr:
        # the retriever needs only the records and the pool, so on a fresh run it goes right after `block` and
        # stage 1 -> stage 2 run once (seeded from v3: the v3 stages exist and are rebuilt after the merge)
        from ber import retriever
        run_pipeline.main(V3_STAGES[:3])                      # prepare embed block (cached when present)
        _step("retriever routes", retriever.routes)          # cached when built by run_v4_retriever.py
        if retriever.needs_merge():
            _step("retriever merge", retriever.merge)
            _clear_after_pool()
    else:
        # seeded: the pool exists, so embeddings / search are not needed (and not copied)
        run_pipeline.main([s for s in V3_STAGES[:3] if not (pooled and s in ("embed", "block"))])
    if s1drop.needed():                                      # BER_TRAIN_S1_DROP: train at test-like orphan density
        _step("train S1 drop", s1drop.run)
        _clear_after_pool(("train",))
    run_pipeline.main(V3_STAGES[3:])                         # stage1 features train (cached stages are skipped)
    if not (WORK / "test" / "scored.parquet").exists():
        run_pipeline.main(["predict"])
    steps = {"diagnose3": lambda: diagnose3.run("v3"), "crossenc": crossenc.run, "graph": graph.run,
             "stage3": stage3.run, "diagnose4": lambda: diagnose3.run("v4")}
    for name in steps:
        if name in wanted:
            _step(name, steps[name])


if __name__ == "__main__":
    main()
