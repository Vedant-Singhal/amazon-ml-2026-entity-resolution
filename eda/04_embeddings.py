"""04 - Dense retrieval study (GPU): multilingual sentence embeddings as a blocking signal.

Encodes sampled S1 queries and the FULL S2+S3 corpus, does exact top-k cosine search within
country on the GPU, and reports recall@k, similarity calibration (true pairs vs best wrong
neighbour vs singletons), and train→test drift incl. France (unseen in training).
Default model: intfloat/multilingual-e5-small (MIT). Override with EMB_MODEL.
Output: eda/reports/04_embeddings.md
"""
import os

import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

from common import SMOKE, Report, load_gt, load_split, log

MODEL = os.environ.get("EMB_MODEL", "intfloat/multilingual-e5-small")
N_Q = 2000 if SMOKE else 50000
N_Q_TEST = 1000 if SMOKE else 20000
KS = [1, 5, 10, 20, 50, 100, 200]
KMAX = max(KS)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PREFIX = "query: " if "e5" in MODEL else ""
rep = Report("04_embeddings", f"04 · Dense retrieval with `{MODEL}`")

log(f"device={DEV} {torch.cuda.get_device_name(0) if DEV == 'cuda' else ''}")
model = SentenceTransformer(MODEL, device=DEV)
if DEV == "cuda":
    model.half()
model.max_seq_length = 64


def texts(df: pl.DataFrame, variant: str) -> list:
    if variant == "name":
        return df.select(pl.lit(PREFIX) + pl.col("business_name")).to_series().to_list()
    return df.select(pl.lit(PREFIX) + pl.col("business_name") + " | "
                     + pl.col("business_address")).to_series().to_list()


def encode(txts: list, chunk=500_000) -> np.ndarray:
    out = []
    for i in range(0, len(txts), chunk):
        e = model.encode(txts[i:i + chunk], batch_size=1024, normalize_embeddings=True,
                         convert_to_tensor=True, show_progress_bar=False)
        out.append(e.half().cpu().numpy())
        log(f"  encoded {min(i + chunk, len(txts)):,}/{len(txts):,}")
    return np.concatenate(out)


@torch.no_grad()
def search(q: np.ndarray, c: np.ndarray, k=KMAX, bs=256):
    C = torch.from_numpy(c).to(DEV)
    k = min(k, len(c))
    sims, idx = [], []
    for i in range(0, len(q), bs):
        s = torch.from_numpy(q[i:i + bs]).to(DEV) @ C.T
        v, j = s.float().topk(k, dim=1)
        sims.append(v.cpu().numpy())
        idx.append(j.cpu().numpy())
    del C
    torch.cuda.empty_cache()
    return np.concatenate(sims), np.concatenate(idx)


# ---- train: recall + calibration ------------------------------------------------------
tr1, oth = load_split("train")
gt, pairs = load_gt()
Q = tr1.sample(n=min(N_Q, len(tr1)), seed=1).join(gt, left_on="entity_id", right_on="s1_id")
recall_rows, calib_rows, ent_rows = [], [], []
for variant in ["name", "name_addr"]:
    log(f"variant {variant}")
    for (country,), qc in Q.group_by("country"):
        cc = oth.filter(pl.col("country") == country)
        log(f" {country}: {len(qc):,} queries vs {len(cc):,} corpus")
        qe, ce = encode(texts(qc, variant)), encode(texts(cc, variant))
        sims, idx = search(qe, ce)
        pos = {e: i for i, e in enumerate(cc["entity_id"].to_list())}
        qpos = {e: i for i, e in enumerate(qc["entity_id"].to_list())}
        gp = pairs.join(qc.select(s1_id="entity_id"), on="s1_id").filter(
            pl.col("id").is_in(list(pos)))
        qi = np.array([qpos[x] for x in gp["s1_id"]])
        ci = np.array([pos[x] for x in gp["id"]])
        hit = idx[qi] == ci[:, None]
        rank = np.where(hit.any(1), hit.argmax(1) + 1, 10**6)
        true_cos = (qe[qi].astype(np.float32) * ce[ci].astype(np.float32)).sum(1)
        src = gp["src"].to_numpy()
        for s in ["S2", "S3", "all"]:
            m = np.ones(len(rank), bool) if s == "all" else src == s
            recall_rows.append({"variant": variant, "country": country, "src": s, "pairs": int(m.sum()),
                                **{f"R@{k}": float((rank[m] <= k).mean()) for k in KS}})
        # entity-level: all of an S1's matches within top-k
        e = pl.DataFrame({"q": qi, "rank": rank}).group_by("q").agg(worst=pl.col("rank").max())
        ent_rows.append({"variant": variant, "country": country,
                         **{f"all@{k}": float((e["worst"] <= k).mean()) for k in KS}})
        # best wrong neighbour per query
        gtset = [set() for _ in range(len(qc))]
        for a, b in zip(qi, ci):
            gtset[a].add(b)
        best_wrong = np.array([next((s for s, j in zip(sims[r], idx[r]) if j not in gtset[r]), np.nan)
                               for r in range(len(qc))])
        single = (qc["n_matches"] == 0).to_numpy()
        pct = lambda x, p: float(np.nanpercentile(x, p)) if len(x) else float("nan")
        calib_rows.append({
            "variant": variant, "country": country,
            "true_p10": pct(true_cos, 10), "true_p50": pct(true_cos, 50),
            "wrong_top1_p50": pct(best_wrong[~single], 50), "wrong_top1_p90": pct(best_wrong[~single], 90),
            "singleton_top1_p50": pct(sims[single, 0], 50), "singleton_top1_p90": pct(sims[single, 0], 90),
            "true_gt_best_wrong": float(np.mean(true_cos > best_wrong[qi])),
        })

rep.h("Pair recall@k (share of GT pairs retrieved in each S1's top-k, within country)")
rep.table(pl.DataFrame(recall_rows), floatfmt="{:.4f}")
rep.h("Entity recall@k (S1 entities whose ALL matches are in top-k; non-singletons)")
rep.table(pl.DataFrame(ent_rows), floatfmt="{:.4f}")
rep.h("Cosine calibration")
rep.p("`true_*` = cosine of GT pairs; `wrong_top1` = best non-matching neighbour of matched S1s; "
      "`singleton_top1` = best neighbour of singletons; `true_gt_best_wrong` = share of GT pairs "
      "scoring above that S1's best wrong neighbour (ranking quality).")
rep.table(pl.DataFrame(calib_rows), floatfmt="{:.4f}")
del oth, tr1

# ---- test drift, incl. France ------------------------------------------------------------
rep.h("Train → test drift: neighbour similarity by country (name variant)")
te1, te_o = load_split("test")
rows = []
for (country,), qc in te1.group_by("country"):
    qc = qc.sample(n=min(N_Q_TEST, len(qc)), seed=2)
    cc = te_o.filter(pl.col("country") == country)
    log(f" test {country}: {len(qc):,} queries vs {len(cc):,} corpus")
    sims, _ = search(encode(texts(qc, "name")), encode(texts(cc, "name")), k=10)
    rows.append({"country": country, "corpus": len(cc),
                 **{f"top1_p{p}": float(np.percentile(sims[:, 0], p)) for p in (10, 25, 50, 75, 90)},
                 "top1_minus_top5_p50": float(np.median(sims[:, 0] - sims[:, 4])),
                 "share_top1_ge_0.9": float((sims[:, 0] >= 0.9).mean())})
rep.p("Compare France against US/India: a shifted distribution means thresholds tuned on "
      "train won't transfer as-is.")
rep.table(pl.DataFrame(rows).sort("country"), floatfmt="{:.4f}")

rep.save()
