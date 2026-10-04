"""Paths, hyper-parameters, logging and the stage timer (v3). Paths follow SageMaker env vars."""
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]                     # .../business_entity_resolution/src
_REPO = next((p for p in _SRC.parents if (p / "dataset").is_dir()), _SRC.parent)

DATA = Path(os.environ.get("SM_CHANNEL_DATASET", _REPO / "dataset"))
WORK = Path(os.environ.get("BER_WORK", _SRC.parent / "work"))  # cached stage outputs
OUT = Path(os.environ.get("BER_OUT", _SRC.parent / "output"))  # submission files + reports
SMOKE_S1 = int(os.environ.get("BER_SMOKE_S1", "0"))            # >0: sample this many train S1
SMOKE = SMOKE_S1 > 0

# Full data sizes (v2 logs) — used to extrapolate smoke timings to the full run.
FULL = {"train": {"s1": 2_206_821, "oth": 10_320_219}, "test": {"s1": 1_732_544, "oth": 9_969_589}}


def _f(name, default):
    return float(os.environ.get(name, default))


def _i(name, default):
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Params:
    # embedder + dense retrieval (FAISS approximate index; brute force does not scale)
    # two dense views (ablation A0): e5-large best overall + native script, e5-small best on
    # no-address records and trade names; each gets its own route and cosine feature
    emb_models: tuple = tuple(os.environ.get(
        "BER_EMB_MODELS", "intfloat/multilingual-e5-large,intfloat/multilingual-e5-small").split(","))
    emb_batch: int = _i("BER_EMB_BATCH", 512)
    emb_max_len: int = 64
    index: str = os.environ.get("BER_INDEX", "IVF,Flat")      # nlist = 4*sqrt(n) is filled in
    nprobe: int = _i("BER_NPROBE", 64)
    dense_k_fwd: int = 25         # per S1: nearest S2/S3 records
    dense_k_rev: int = 5          # per S2/S3 record: nearest S1 entities
    # char-3-gram routes (count-sketch vectors of TF-IDF, searched with the same FAISS index)
    sketch_dim: int = 256
    c3_k_fwd: int = 10
    c3_k_rev: int = 3
    # lexical routes
    sparse_k: int = 20            # IDF-weighted shared keys (v2 route)
    key_df_cap: int = 500           # v2: 2000 (join size ~ df^2; ~40 min at 16 cores)
    key_cap_s1: int = 10          # exact-key routes: skip keys shared by more S1 than this
    key_cap_oth: int = 50         # ... or by more S2/S3 records than this
    # normalization (all learned from the provided data)
    stop_df: float = 0.01         # words in >= 1% of a country's names = generic
    landmark: bool = os.environ.get("BER_LANDMARK", "1") != "0"
    lex_min_count: int = 20
    lex_min_ratio: float = 0.5
    # stage-1 shortlist (= candidate_pairs.tsv)
    sl_keep: float = _f("BER_SL_KEEP", 0.995)   # keep >= this share of the pool's recall
    sl_neg_rate: float = 0.2                   # stage-1 training: easy negatives sampled at this rate
    # learners (XGBoost on GPU; CPU fallback)
    cv_folds: int = 5
    lr1: float = 0.1
    rounds1: int = _i("BER_ROUNDS1", 1500)
    lr2: float = _f("BER_LR2", 0.05)
    rounds2: int = _i("BER_ROUNDS2", 5000)
    early_stop: int = 100
    seed: int = 13
    # ablations: share of S1 used, learning rate
    abl_frac: float = _f("BER_ABL_FRAC", 1.0)
    # share of the machine (BER_CPU_FRAC, default 50%): memory/CPU spikes can't crash a run
    workers: int = max(1, int((os.cpu_count() or 2) * _f("BER_CPU_FRAC", 0.5)))
    pool_workers: int = max(1, int((os.cpu_count() or 2) * _f("BER_CPU_FRAC", 0.5)))


P = Params()


def _mem() -> str:
    try:
        info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
        total = int(info["MemTotal"].split()[0]) / 2**20
        avail = int(info["MemAvailable"].split()[0]) / 2**20
        return f" [mem {total - avail:.0f}/{total:.0f} GB]"
    except (OSError, KeyError, ValueError):
        return ""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]{_mem()} {msg}", flush=True)


def work_dir(*parts) -> Path:
    p = WORK.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def record_time(stage: str, seconds: float, **units) -> None:
    """Append a timing row to work/timing.json (smoke runs extrapolate from these)."""
    path = WORK / "timing.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    rows.append({"stage": stage, "seconds": round(seconds, 2), **units})
    WORK.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1))
