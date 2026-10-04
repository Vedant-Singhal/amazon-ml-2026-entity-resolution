"""v3 end-to-end entry point:
  prepare -> embed -> block (pool) -> stage1 (shortlist) -> features -> train -> predict -> ablate [-> estimate]

  python src/run_pipeline.py                     # all stages
  python src/run_pipeline.py block stage1        # selected stages (earlier outputs are cached)
Env: BER_STAGES (same as args), BER_SMOKE_S1=10000 (subsample + full-run estimate), BER_WORK /
BER_OUT (paths), BER_EMB_MODEL, SM_CHANNEL_DATASET. Every stage skips work whose output exists.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_HALF = str(max(1, int((os.cpu_count() or 2) * float(os.environ.get("BER_CPU_FRAC", "0.5")))))  # share of the machine (default 50%)
for _v in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, _HALF)

from ber import ablate, block, embed, features, predict, prepare, stage1, train  # noqa: E402
from ber.config import DATA, OUT, P, SMOKE, SMOKE_S1, WORK, log, record_time  # noqa: E402


def _prepare():
    for split in ("train", "test"):
        t0 = time.time()
        cached = (WORK / split / "oth.parquet").exists()
        prepare.run(split)
        if not cached:
            import polars as pl
            n1 = pl.scan_parquet(WORK / split / "s1.parquet").select(pl.len()).collect().item()
            n2 = pl.scan_parquet(WORK / split / "oth.parquet").select(pl.len()).collect().item()
            record_time(f"prepare.{split}", time.time() - t0, s1=n1, oth=n2)


STAGES = {
    "prepare": _prepare,
    "embed": lambda: (embed.run("train"), embed.run("test")),
    "block": lambda: (block.run("train"), block.run("test")),
    "stage1": stage1.run,
    "features": lambda: (features.run("train"), features.run("test")),
    "train": train.run,
    "predict": predict.run,
    "ablate": ablate.run,
}


def _env_report():
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (OUT / "pip_freeze.txt").write_text(freeze)
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:  # noqa: BLE001
        gpu = "unknown"
    log(f"env: gpu={gpu} cpus={os.cpu_count()} workers={P.workers} emb={",".join(P.emb_models)} index={P.index} "
        f"nprobe={P.nprobe}")


def main(argv):
    wanted = [a for a in argv if not a.startswith("-")] or os.environ.get("BER_STAGES", "").split()
    wanted = wanted or list(STAGES) + (["estimate"] if SMOKE else [])
    log(f"data={DATA} work={WORK} out={OUT} smoke_s1={SMOKE_S1} stages={wanted}")
    _env_report()
    t_all = time.time()
    for name in STAGES:
        if name in wanted:
            t0 = time.time()
            STAGES[name]()
            log(f"== {name} done in {(time.time() - t0) / 60:.1f} min (total {(time.time() - t_all) / 3600:.2f} h)")
    if "estimate" in wanted:
        from ber import estimate
        estimate.run(os.environ.get("BER_INSTANCE", "ml.g6e.8xlarge"))


if __name__ == "__main__":
    main(sys.argv[1:])
