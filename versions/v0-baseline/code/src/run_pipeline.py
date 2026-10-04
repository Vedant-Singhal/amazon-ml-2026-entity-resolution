#!/usr/bin/env python3
"""
MLChallenge Entity Resolution Pipeline — GPU Accelerated
4× NVIDIA L4 (23.7GB each)
TF-IDF + TruncatedSVD → FAISS GPU → RapidFuzz features → XGBoost GPU
"""
import os, gc, time, warnings
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as sk_norm
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import faiss
import xgboost as xgb
import torch

warnings.filterwarnings("ignore")

LOCAL_DIR  = "./mlchallenge_data"
OUTPUT_DIR = "./pipeline_output"
TOP_K      = 10
SVD_DIM    = 128
# Use all 4 GPUs for FAISS
N_GPUS     = faiss.get_num_gpus()
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log(f"GPUs available: {N_GPUS}")
log(f"GPU 0: {torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
log("Loading data...")
t0 = time.time()
train_s1     = pd.read_csv(f"{LOCAL_DIR}/train/train_source1.tsv", sep="\t", dtype=str).fillna("")
train_s2     = pd.read_csv(f"{LOCAL_DIR}/train/train_source2.tsv", sep="\t", dtype=str).fillna("")
train_s3     = pd.read_csv(f"{LOCAL_DIR}/train/train_source3.tsv", sep="\t", dtype=str).fillna("")
ground_truth = pd.read_csv(f"{LOCAL_DIR}/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
test_s1      = pd.read_csv(f"{LOCAL_DIR}/test/test_source1.tsv",  sep="\t", dtype=str).fillna("")
test_s2      = pd.read_csv(f"{LOCAL_DIR}/test/test_source2.tsv",  sep="\t", dtype=str).fillna("")
test_s3      = pd.read_csv(f"{LOCAL_DIR}/test/test_source3.tsv",  sep="\t", dtype=str).fillna("")
log(f"  Loaded in {time.time()-t0:.0f}s")
log(f"  train: s1={len(train_s1):,} s2={len(train_s2):,} s3={len(train_s3):,}")
log(f"  test:  s1={len(test_s1):,}  s2={len(test_s2):,}  s3={len(test_s3):,}")

# ── 2. PREPROCESS ─────────────────────────────────────────────────────────────
log("Preprocessing...")
for df in [train_s1, train_s2, train_s3, test_s1, test_s2, test_s3]:
    df["name_norm"]    = df["business_name"].str.lower().str.strip()
    df["address_norm"] = df["business_address"].str.lower().str.strip()
log("  Done")

# ── 3. GPU BLOCKING ───────────────────────────────────────────────────────────
def build_and_search_gpu(s1_df, s23_df, country, top_k=TOP_K, svd_dim=SVD_DIM):
    """
    TF-IDF + TruncatedSVD (CPU, sparse-friendly) + FAISS GPU IndexFlatIP.
    Uses all available GPUs via index_cpu_to_all_gpus.
    """
    s1_c  = s1_df[s1_df["country"] == country].reset_index(drop=True)
    s23_c = s23_df[s23_df["country"] == country].reset_index(drop=True)
    if len(s1_c) == 0 or len(s23_c) == 0:
        return pd.DataFrame(columns=["s1_id","s23_id","faiss_score"])
    log(f"    S1={len(s1_c):,}  DB={len(s23_c):,}")

    # Step 1: TF-IDF (sparse, CPU)
    t = time.time()
    vect = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 3), lowercase=True,
        min_df=3, max_features=50000, dtype=np.float32, sublinear_tf=True,
    )
    db_tfidf = vect.fit_transform(s23_c["name_norm"].tolist())
    log(f"    TF-IDF vocab={db_tfidf.shape[1]:,}  {time.time()-t:.1f}s")

    # Step 2: TruncatedSVD → dense 128-d (CPU, works on sparse)
    t = time.time()
    n_comp = min(svd_dim, db_tfidf.shape[1] - 1, db_tfidf.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42, n_iter=5)
    db_dense = svd.fit_transform(db_tfidf).astype("float32")
    db_norm  = sk_norm(db_dense, norm="l2")
    log(f"    SVD({n_comp}d)  {time.time()-t:.1f}s")
    del db_tfidf, db_dense; gc.collect()

    # Step 3: FAISS GPU index (all GPUs)
    t = time.time()
    cpu_index = faiss.IndexFlatIP(n_comp)
    if N_GPUS > 1:
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    else:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.add(db_norm)
    log(f"    FAISS GPU: {gpu_index.ntotal:,} vectors  {time.time()-t:.1f}s")
    del db_norm; gc.collect()

    # Step 4: Transform S1 and search
    t = time.time()
    q_tfidf  = vect.transform(s1_c["name_norm"].tolist())
    q_dense  = sk_norm(svd.transform(q_tfidf).astype("float32"), norm="l2")
    k = min(top_k, len(s23_c))
    D, I = gpu_index.search(q_dense, k)
    log(f"    Search {len(s1_c):,} queries  {time.time()-t:.1f}s")
    del q_tfidf, q_dense, gpu_index; gc.collect()
    torch.cuda.empty_cache()

    # Step 5: Flatten
    s23_ids = s23_c["entity_id"].values
    s1_ids  = s1_c["entity_id"].values
    idx_flat = I.flatten()
    s1_rep   = np.repeat(s1_ids, k)
    s23_rep  = s23_ids[idx_flat]
    scores   = D.flatten()
    return pd.DataFrame({"s1_id": s1_rep, "s23_id": s23_rep, "faiss_score": scores})

# ── TRAIN CANDIDATES ──────────────────────────────────────────────────────────
log("=== TRAIN BLOCKING ===")
train_cands = []
for country in ["US", "India"]:
    for s23_df, tag in [(train_s2, "S2"), (train_s3, "S3")]:
        log(f"  {tag} x {country}")
        cdf = build_and_search_gpu(train_s1, s23_df, country)
        train_cands.append(cdf)
        log(f"  → {len(cdf):,} pairs")

train_cands_df = pd.concat(train_cands, ignore_index=True).drop_duplicates(["s1_id","s23_id"])
log(f"Total train candidates: {len(train_cands_df):,}")
train_cands_df.to_parquet(f"{OUTPUT_DIR}/train_candidates.parquet", index=False)

# ── 4. FEATURES ───────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "faiss_score",
    "name_token_set", "name_token_sort", "name_partial", "name_ratio",
    "addr_partial", "addr_token_set",
    "country_match", "has_address",
]

def compute_features(cands_df, s1_idx, s23_idx):
    df = cands_df.copy()
    df = df.join(s1_idx[["name_norm","address_norm","country"]].rename(
        columns={"name_norm":"n1","address_norm":"a1","country":"c1"}), on="s1_id")
    df = df.join(s23_idx[["name_norm","address_norm","country"]].rename(
        columns={"name_norm":"n2","address_norm":"a2","country":"c2"}), on="s23_id")
    for col in ["n1","n2","a1","a2"]:
        df[col] = df[col].fillna("")
    n1 = df["n1"].tolist(); n2 = df["n2"].tolist()
    a1 = df["a1"].tolist(); a2 = df["a2"].tolist()
    df["name_token_set"]  = [fuzz.token_set_ratio(x,y)/100  for x,y in zip(n1,n2)]
    df["name_token_sort"] = [fuzz.token_sort_ratio(x,y)/100 for x,y in zip(n1,n2)]
    df["name_partial"]    = [fuzz.partial_ratio(x,y)/100    for x,y in zip(n1,n2)]
    df["name_ratio"]      = [fuzz.ratio(x,y)/100            for x,y in zip(n1,n2)]
    df["addr_partial"]    = [fuzz.partial_ratio(x,y)/100  if x and y else 0.0 for x,y in zip(a1,a2)]
    df["addr_token_set"]  = [fuzz.token_set_ratio(x,y)/100 if x and y else 0.0 for x,y in zip(a1,a2)]
    df["country_match"]   = (df["c1"]==df["c2"]).astype(float)
    df["has_address"]     = ((df["a1"]!="") & (df["a2"]!="")).astype(float)
    return df

log("=== TRAIN FEATURES ===")
s1_idx_train  = train_s1.set_index("entity_id")
s23_idx_train = pd.concat([
    train_s2[["entity_id","name_norm","address_norm","country"]],
    train_s3[["entity_id","name_norm","address_norm","country"]],
]).set_index("entity_id")
train_feat_df = compute_features(train_cands_df, s1_idx_train, s23_idx_train)
log(f"  Feature matrix: {train_feat_df.shape}")

# ── 5. LABELS ─────────────────────────────────────────────────────────────────
log("=== LABELS ===")
positive_pairs = set()
for _, row in ground_truth.iterrows():
    s1_id = row["source1_entity_id"]
    for m in str(row["matched_entity_ids"]).split(","):
        m = m.strip()
        if m: positive_pairs.add((s1_id, m))
log(f"  Positive pairs in GT: {len(positive_pairs):,}")

train_feat_df["label"] = train_feat_df.apply(
    lambda r: 1 if (r["s1_id"],r["s23_id"]) in positive_pairs else 0, axis=1
)
n_pos = int(train_feat_df["label"].sum())
n_neg = len(train_feat_df) - n_pos
log(f"  Pos={n_pos:,}  Neg={n_neg:,}  Blocking recall={100*n_pos/len(positive_pairs):.1f}%")

# ── 6. XGBOOST (GPU) ──────────────────────────────────────────────────────────
log("=== XGBOOST (GPU) ===")
X = train_feat_df[FEATURE_COLS].values.astype("float32")
y = train_feat_df["label"].values
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
scale_pw = (y_tr==0).sum() / max((y_tr==1).sum(), 1)
log(f"  scale_pos_weight={scale_pw:.1f}  train={len(X_tr):,}  val={len(X_val):,}")

model = xgb.XGBClassifier(
    n_estimators=400, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pw, eval_metric="logloss",
    device="cuda",          # GPU training
    random_state=42, tree_method="hist",
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)
model.save_model(f"{OUTPUT_DIR}/xgb_model.json")
log("  Model saved")

# Feature importances
imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
log("  Feature importances:\n" + imp.round(4).to_string())

# ── 7. THRESHOLD TUNING ───────────────────────────────────────────────────────
log("=== THRESHOLD TUNING (F0.5) ===")
val_probs = model.predict_proba(X_val)[:, 1]
best_f05, best_thresh = 0.0, 0.5
for thresh in np.arange(0.30, 0.95, 0.05):
    preds = (val_probs >= thresh).astype(int)
    if preds.sum() == 0: continue
    f = fbeta_score(y_val, preds, beta=0.5, zero_division=0)
    p = precision_score(y_val, preds, zero_division=0)
    r = recall_score(y_val, preds, zero_division=0)
    log(f"  t={thresh:.2f}  P={p:.4f}  R={r:.4f}  F0.5={f:.4f}  n={preds.sum():,}")
    if f > best_f05: best_f05, best_thresh = f, thresh
log(f"  ★ Best threshold={best_thresh:.2f}  F0.5={best_f05:.4f}")

# ── 8. TEST PREDICTIONS ───────────────────────────────────────────────────────
log("=== TEST BLOCKING ===")
test_cands = []
for country in ["US", "India", "France"]:
    for s23_df, tag in [(test_s2, "S2"), (test_s3, "S3")]:
        log(f"  {tag} x {country}")
        cdf = build_and_search_gpu(test_s1, s23_df, country)
        test_cands.append(cdf)
        log(f"  → {len(cdf):,} pairs")
test_cands_df = pd.concat(test_cands, ignore_index=True).drop_duplicates(["s1_id","s23_id"])
log(f"Total test candidates: {len(test_cands_df):,}")
test_cands_df.to_parquet(f"{OUTPUT_DIR}/test_candidates.parquet", index=False)

log("=== TEST FEATURES ===")
s1_idx_test  = test_s1.set_index("entity_id")
s23_idx_test = pd.concat([
    test_s2[["entity_id","name_norm","address_norm","country"]],
    test_s3[["entity_id","name_norm","address_norm","country"]],
]).set_index("entity_id")
test_feat_df = compute_features(test_cands_df, s1_idx_test, s23_idx_test)

X_test = test_feat_df[FEATURE_COLS].values.astype("float32")
test_probs = model.predict_proba(X_test)[:, 1]
test_feat_df["predicted"] = (test_probs >= best_thresh).astype(int)
log(f"  Predicted matches: {int(test_feat_df['predicted'].sum()):,} / {len(test_feat_df):,}")

# ── 9. SUBMISSION ─────────────────────────────────────────────────────────────
log("=== SUBMISSION ===")
matches_df = test_feat_df[test_feat_df["predicted"]==1][["s1_id","s23_id"]]
grouped = (
    matches_df.groupby("s1_id")["s23_id"]
    .apply(lambda x: ",".join(sorted(x.unique())))
    .reset_index()
    .rename(columns={"s1_id":"source1_entity_id","s23_id":"matched_entity_ids"})
)
all_s1 = pd.DataFrame({"source1_entity_id": test_s1["entity_id"].unique()})
submission = all_s1.merge(grouped, on="source1_entity_id", how="left")
submission["matched_entity_ids"] = submission["matched_entity_ids"].fillna("")

sub_path = f"{OUTPUT_DIR}/submission.tsv"
submission.to_csv(sub_path, sep="\t", index=False)
n_match = int((submission["matched_entity_ids"]!="").sum())
log(f"Saved → {sub_path}")
log(f"  S1 entities={len(submission):,}  with_match={n_match:,} ({100*n_match/len(submission):.1f}%)")
log("PIPELINE COMPLETE ✅")
