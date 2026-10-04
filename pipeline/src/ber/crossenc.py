"""Stage `crossenc` (v4): a Ditto-style cross-encoder (Li et al., PVLDB'21) fine-tuned on our train pairs and
used only where the v3 model is unsure.

  - Input: both records serialized ("name: … | address: …"), read jointly by one transformer, so it can learn
    abbreviations, word order, number meaning and branch/twin words that hand-made features miss.
  - Base model: multilingual-e5-small (MIT, 118M) with a 1-logit classification head (BER_CE_MODEL to change).
  - Pairs scored: the uncertain band LO <= p2 <= HI of v3's stage-2 probability (OOF on train). Training uses
    the band plus a small sample of confident pairs (keeps the extremes calibrated).
  - Leak-free: 2-fold cross-fitting by S1 (hash(a, seed + 7) % 2); every train pair is scored by the model
    that did not see its S1. Test pairs get the mean of both models.
Output: work/ce/p_ce_{train,test}.parquet (a, b, p_ce); pairs outside the band have no p_ce (feature = NaN).
"""
import math
import os
import time

import numpy as np
import polars as pl

from .config import P, log, record_time, work_dir
from .v4io import texts, v3_scores

BASE = os.environ.get("BER_CE_MODEL", "intfloat/multilingual-e5-small")
LO, HI = float(os.environ.get("BER_CE_LO", "0.02")), float(os.environ.get("BER_CE_HI", "0.98"))
MAX_LEN = 96
BATCH = 256
LR = 3e-5
EPOCHS = int(os.environ.get("BER_CE_EPOCHS", "1"))                 # v5: 2-3 with a larger base
TRAIN_CAP = int(os.environ.get("BER_CE_CAP", "3000000"))     # pairs per fold model
OUT_SAMPLE = 0.05                                              # confident pairs mixed into training


def _with_text(pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    s1, oth = texts(split)
    return (pairs.with_row_index("_r").join(s1, on="a", how="left").join(oth, on="b", how="left")
            .sort("_r").drop("_r").with_columns(pl.col("ta", "tb").fill_null("")))


def _load():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).cuda()
    torch.manual_seed(P.seed)
    return tok, model


def _train(ta: list, tb: list, y: np.ndarray):
    import torch
    import torch.nn.functional as F
    from transformers import get_linear_schedule_with_warmup
    tok, model = _load()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    per_epoch = math.ceil(len(y) / BATCH)
    steps = per_epoch * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    rng = np.random.default_rng(P.seed)
    perm = np.concatenate([rng.permutation(len(y)) for _ in range(EPOCHS)])   # a fresh shuffle per epoch
    t0, run_loss = time.time(), 0.0
    for i in range(steps):
        idx = perm[i * BATCH:(i + 1) * BATCH]
        enc = tok([ta[j] for j in idx], [tb[j] for j in idx], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to("cuda")
        yb = torch.from_numpy(y[idx].astype(np.float32)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(**enc).logits.squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logit.float(), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        run_loss += loss.item()
        if (i + 1) % 2000 == 0 or i + 1 == steps:
            log(f"crossenc: step {i + 1}/{steps} loss {run_loss / min(2000, i + 1):.4f} "
                f"({(i + 1) * BATCH / (time.time() - t0):,.0f} pairs/s)")
            run_loss = 0.0
    model.eval()
    return tok, model


def _predict(tok, model, ta: list, tb: list) -> np.ndarray:
    import torch
    order = np.argsort([len(x) + len(z) for x, z in zip(ta, tb)])     # length-sorted batches: little padding
    out = np.empty(len(ta), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(order), 1024):
            idx = order[s:s + 1024]
            enc = tok([ta[j] for j in idx], [tb[j] for j in idx], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[idx] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
    return out


def run() -> None:
    import torch
    md = work_dir("ce")
    if (md / "p_ce_test.parquet").exists():
        log("crossenc: cached")
        return
    t0 = time.time()
    tr = v3_scores("train")
    band = tr.filter(pl.col("p2").is_between(LO, HI))
    rest = tr.filter(~pl.col("p2").is_between(LO, HI)).sample(fraction=OUT_SAMPLE, seed=P.seed)
    half = lambda d: d.with_columns(h=(pl.col("a").hash(P.seed + 7) % 2).cast(pl.Int8))
    fit_set, band = half(pl.concat([band, rest])), half(band)
    te = v3_scores("test").filter(pl.col("p2").is_between(LO, HI))
    log(f"crossenc: band {LO}-{HI}: {len(band):,} train pairs ({band['label'].mean():.3f} positive), "
        f"{len(te):,} test pairs; fit set {len(fit_set):,}")
    fit_set, band, te = _with_text(fit_set, "train"), _with_text(band, "train"), _with_text(te, "test")

    parts, p_test = [], np.zeros(len(te), dtype=np.float32)
    for h in (0, 1):
        f = fit_set.filter(pl.col("h") != h)
        if len(f) > TRAIN_CAP:
            f = f.sample(n=TRAIN_CAP, seed=P.seed)
        tok, model = _train(f["ta"].to_list(), f["tb"].to_list(), f["label"].to_numpy())
        b = band.filter(pl.col("h") == h)
        parts.append(b.select("a", "b").with_columns(
            p_ce=pl.Series(_predict(tok, model, b["ta"].to_list(), b["tb"].to_list()))))
        p_test += _predict(tok, model, te["ta"].to_list(), te["tb"].to_list()) / 2
        model.save_pretrained(md / f"model{h}")
        tok.save_pretrained(md / f"model{h}")
        del model
        torch.cuda.empty_cache()
        log(f"crossenc: fold {h} done ({(time.time() - t0) / 60:.1f} min)")
    oof = pl.concat(parts)
    oof.write_parquet(md / "p_ce_train.parquet")
    te.select("a", "b").with_columns(p_ce=pl.Series(p_test)).write_parquet(md / "p_ce_test.parquet")
    # quick check: does the cross-encoder separate the band better than v3?
    chk = oof.join(band.select("a", "b", "p2", "label"), on=["a", "b"])
    from sklearn.metrics import roc_auc_score
    y = chk["label"].to_numpy()
    if 0 < y.sum() < len(y):
        log(f"crossenc: AUC on the band — v3 {roc_auc_score(y, chk['p2'].to_numpy()):.4f}, "
            f"cross-encoder {roc_auc_score(y, chk['p_ce'].to_numpy()):.4f}")
    record_time("crossenc", time.time() - t0, train_pairs=len(oof), test_pairs=len(te))
