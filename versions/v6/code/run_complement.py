"""Which third embedder complements e5-small + e5-large best? (v6 choice; GPU job)
Sample N_Q train S1 (queries); corpus = all their true S2/S3 matches + POOL_FRAC of the other records of the same
country. Each model: exact top-K per query (K = 25 = the pipeline's dense_k_fwd). A true pair is "found" by a model when
it is in that query's top-K. Report per subset (all / no address / trade name / native script) and country:
  base    = found by e5-small OR e5-large (what the pipeline's two dense views already give)
  own     = found by the candidate
  added   = found by the candidate AND missed by both base models  ← the complementary value
  union   = base OR candidate
Writes output/complement.md.   Env: CMP_MODELS (candidates, comma list), CMP_NQ, CMP_FRAC."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "pipeline" / "src"))
_HALF = str(max(1, int((os.cpu_count() or 2) * float(os.environ.get("BER_CPU_FRAC", "0.5")))))
for _v in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, _HALF)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from rapidfuzz import fuzz, process  # noqa: E402

from ber.block import gpu_exact  # noqa: E402
from ber.config import OUT, P, log  # noqa: E402
from ber.data import load_gt_pairs, load_split  # noqa: E402
from ber.embed import PREFIX, load_model  # noqa: E402
from ber.text import NATIVE  # noqa: E402

BASE = ["intfloat/multilingual-e5-small", "intfloat/multilingual-e5-large"]
CANDS = os.environ.get("CMP_MODELS", ",".join([
    "intfloat/multilingual-e5-large-instruct", "Snowflake/snowflake-arctic-embed-l-v2.0", "BAAI/bge-m3",
    "Qwen/Qwen3-Embedding-0.6B", "intfloat/multilingual-e5-base",
    "ibm-granite/granite-embedding-311m-multilingual-r2"])).split(",")
N_Q = int(os.environ.get("CMP_NQ", "30000"))
FRAC = float(os.environ.get("CMP_FRAC", "0.10"))
K = 25


def texts(df, prefix):
    return df.select(pl.lit(prefix) + pl.col("name") + " | " + pl.col("addr")).to_series().to_list()


def found_by(name, q, corp, t):
    """-> bool array over the true pairs t (in t's row order): pair in the query's top-K; docs/s."""
    import torch
    qp, dp, _ = PREFIX.get(name, ("", "", {}))
    model = load_model(name)
    hit = np.zeros(len(t), bool)
    rates = []
    for (country,), qc in q.group_by("country"):
        cc = corp.filter(pl.col("country") == country)
        qe = model.encode(texts(qc, qp), batch_size=512, normalize_embeddings=True, convert_to_numpy=True,
                          show_progress_bar=False).astype(np.float16)
        t0 = time.time()
        ce = model.encode(texts(cc, dp), batch_size=512, normalize_embeddings=True, convert_to_numpy=True,
                          show_progress_bar=False).astype(np.float16)
        rates.append(len(cc) / max(time.time() - t0, 1e-6))
        idx = gpu_exact(ce, qe, K)
        qmap = dict(zip(qc["iid"].to_list(), range(len(qc))))
        cmap = dict(zip(cc["iid"].to_list(), range(len(cc))))
        tc = t.with_row_index("r").filter(pl.col("a").is_in(qc["iid"].implode()))
        qpos = np.array([qmap[x] for x in tc["a"].to_list()])
        cpos = np.array([cmap[x] for x in tc["b"].to_list()])
        hit[tc["r"].to_numpy()] = (idx[qpos] == cpos[:, None]).any(1)
    del model
    torch.cuda.empty_cache()
    return hit, float(np.mean(rates))


if __name__ == "__main__":
    s1, oth = load_split("train")
    pairs, _ = load_gt_pairs()
    q = s1.select("iid", "country", "name", "addr").sample(n=min(N_Q, len(s1)), seed=P.seed)
    gp = pairs.join(q.select(a="iid"), on="a")
    keep = pl.concat([gp.select(iid="b"), oth.filter(pl.col("iid").hash(P.seed) % 1000 < FRAC * 1000).select("iid")]).unique()
    corp = oth.select("iid", "country", "name", "addr").join(keep, on="iid")
    t = (gp.join(q.select(a="iid", name_a="name", country="country"), on="a")
         .join(corp.select(b="iid", name_b="name", addr_b="addr"), on="b"))
    nsim = process.cpdist([x.lower() for x in t["name_a"].to_list()], [x.lower() for x in t["name_b"].to_list()],
                          scorer=fuzz.token_set_ratio, workers=P.workers)
    subsets = {"all": np.ones(len(t), bool), "no address": (t["addr_b"].str.strip_chars() == "").to_numpy(),
               "trade name (name sim < 50)": nsim < 50, "native script": t["name_b"].str.contains(NATIVE).to_numpy()}
    ctry = t["country"].to_numpy()
    log(f"complement: {len(q):,} queries, {len(corp):,} corpus records, {len(t):,} true pairs")

    hits, speed = {}, {}
    for name in BASE + CANDS:
        try:
            hits[name], speed[name] = found_by(name, q, corp, t)
            log(f"complement: {name} done ({speed[name]:,.0f} docs/s)")
        except Exception as e:  # one model failing must not stop the others
            log(f"complement: {name} FAILED: {type(e).__name__}: {str(e)[:200]}")
    base = hits[BASE[0]] | hits[BASE[1]]

    rep = ["# Third embedder: complement to e5-small + e5-large (exact top-25 per S1)\n",
           f"{len(q):,} train S1 queries; corpus = their {len(t):,} true pairs + {FRAC:.0%} of other same-country records "
           f"({len(corp):,}).\n", "| subset | country | pairs | base (small OR large) | candidate | own | added (base missed) | union |",
           "|---|---|---|---|---|---|---|---|"]
    for sub, m in subsets.items():
        for c in ("US", "India"):
            mm = m & (ctry == c)
            n = int(mm.sum())
            if not n:
                continue
            for name in CANDS:
                if name not in hits:
                    continue
                h = hits[name]
                rep.append(f"| {sub} | {c} | {n:,} | {base[mm].mean():.4f} | {name.split('/')[-1]} | {h[mm].mean():.4f} | "
                           f"{(h & ~base)[mm].sum():,} ({(h & ~base)[mm].mean():.4f}) | {(base | h)[mm].mean():.4f} |")
    rep += ["\n**Speed (docs/s, fp16, this GPU):** " + ", ".join(f"{k.split('/')[-1]} {v:,.0f}" for k, v in speed.items())]
    best = max((n for n in CANDS if n in hits), key=lambda n: (hits[n] & ~base).sum(), default=None)
    if best:
        rep.append(f"\n**Most complementary (most true pairs added over both base models, all subsets): `{best}`**")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "complement.md").write_text("\n".join(rep), encoding="utf-8")
    log("\n".join(rep))
