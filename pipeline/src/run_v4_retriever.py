"""v4 retriever routes only (can run while v3 is still training): needs just the prepared records.
Needs only the prepared records (runs `prepare` first when missing). Smoke: BER_SMOKE_S1 prepares a sample first."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_HALF = str(max(1, int((os.cpu_count() or 2) * float(os.environ.get("BER_CPU_FRAC", "0.5")))))  # share of the machine (default 50%)
for _v in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, _HALF)

import run_pipeline  # noqa: E402
from ber import retriever  # noqa: E402

if __name__ == "__main__":
    run_pipeline.main(["prepare"])
    retriever.routes()
