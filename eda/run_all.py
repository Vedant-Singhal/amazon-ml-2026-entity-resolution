"""Run every EDA step in order; one failing step doesn't stop the rest.

Locally:     python eda/run_all.py [01 03 ...]
SageMaker:   entry point of the training job (see launch_sagemaker.ps1).
"""
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, REPORTS, SMOKE  # noqa: E402

STEPS = ["01_profile.py", "02_pairs.py", "03_blocking.py", "04_embeddings.py"]
only = [a for a in sys.argv[1:] if not a.startswith("-")] or os.environ.get("EDA_STEPS", "").split()
steps = [s for s in STEPS if not only or s[:2] in only]

logs = REPORTS / "logs"
logs.mkdir(exist_ok=True)
status = [f"smoke={SMOKE} data={DATA} python={platform.python_version()} cpus={os.cpu_count()}"]
try:
    import torch
    status.append(f"cuda={torch.cuda.is_available()} "
                  f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
except ImportError:
    pass
print("\n".join(status), flush=True)

for step in steps:
    t0 = time.time()
    with open(logs / f"{step[:-3]}.log", "w", encoding="utf-8") as fh:
        proc = subprocess.Popen([sys.executable, "-u", str(HERE / step)], cwd=HERE,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:           # tee: CloudWatch + log file
            print(f"[{step[:2]}] {line}", end="", flush=True)
            fh.write(line)
        rc = proc.wait()
    msg = f"{step}: {'OK' if rc == 0 else f'FAILED rc={rc}'} in {(time.time() - t0) / 60:.1f} min"
    status.append(msg)
    print(msg, flush=True)

(REPORTS / "status.txt").write_text("\n".join(status) + "\n", encoding="utf-8")
