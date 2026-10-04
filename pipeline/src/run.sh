#!/usr/bin/env bash
# End to end on one Linux GPU machine: data -> v2 -> v4 -> v5a -> 0.7/0.3 blend -> submission/.
#   DATA=/path/to/dataset bash src/run.sh  folder holding train/ and test/ (default: pipeline/dataset or ../dataset)
#   SMOKE=1 bash src/run.sh                quick check on a 10k-S1 sample (separate *_smoke folders)
# Run from anywhere; work folders and submission/ are created inside pipeline/.
# Every stage caches its result in its work folder: a restart resumes where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${DATA:-}" ]; then
  for d in dataset ../dataset; do
    if [ -d "$d/train" ] && [ -d "$d/test" ]; then DATA=$d; break; fi
  done
fi
if [ -z "${DATA:-}" ] || [ ! -f "$DATA/test/test_source1.tsv" ]; then
  echo "Dataset not found. Set DATA to the folder holding train/ and test/ (e.g. DATA=/data/dataset)." >&2
  exit 1
fi
DATA=$(cd "$DATA" && pwd)
SFX=""
SMOKE_ENV=()
if [ "${SMOKE:-0}" = "1" ]; then
  SFX="_smoke"
  SMOKE_ENV=(BER_SMOKE=1 BER_SMOKE_S1=10000)
fi
W2=work_v2$SFX; W4=work_v4$SFX; W5=work_v5a$SFX

# settings shared by v4 and v5a (v2's scores are one stage-3 feature of both)
COMMON=(SM_CHANNEL_DATASET="$DATA" BER_V2_OOF="$W2/model" BER_V2_TEST="$W2/test"
        BER_EMB_MODELS=intfloat/multilingual-e5-small,intfloat/multilingual-e5-large
        BER_ANN_MIN=1000000000 BER_SL_KEEP=0.999 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        ${SMOKE_ENV[@]+"${SMOKE_ENV[@]}"})

# 1) v2: LightGBM on 44 features (its out-of-fold train score and test score feed v4 and v5a)
env ${SMOKE_ENV[@]+"${SMOKE_ENV[@]}"} BER_DATA="$DATA" BER_WORK="$W2" BER_OUT="output_v2$SFX" \
    python src/v2/run_pipeline.py

# 2) v4: search (2 embedders + fine-tuned retriever + sketch/key routes) -> shortlist -> 3-stage matcher
env "${COMMON[@]}" BER_WORK="$W4" BER_OUT="output_v4$SFX" BER_V4_RETRIEVER=1 BER_V4_EXTRA_T=0.95 \
    python src/run_v4.py

# 3) v5a: v4's candidate pool (copied, not searched again), then everything else rebuilt with 19% of train S1
#    dropped (test-like orphan density), abbreviations mined per country from the records, 3,000 stage-1 trees
for s in train test; do
  mkdir -p "$W5/$s"
  for f in pool.parquet route_ft.parquet block_stats.json emb_oth.npy; do   # block_stats.json exists for train only
    if [ -f "$W4/$s/$f" ] && [ ! -f "$W5/$s/$f" ]; then cp "$W4/$s/$f" "$W5/$s/$f"; fi
  done
done
if [ -f "$W4/timing.json" ] && [ ! -f "$W5/timing.json" ]; then cp "$W4/timing.json" "$W5/timing.json"; fi
env "${COMMON[@]}" BER_WORK="$W5" BER_OUT="output_v5a$SFX" BER_V4_RETRIEVER=0 python src/run_v5.py

# 4) final: p = 0.7 * p_v4 + 0.3 * p_v5a, one S1 per record, cut at v4@0.95's pair count -> submission/
env V4_WORK="$W4" V5_WORK="$W5" DATA="$DATA" FINAL_OUT="submission$SFX" FINAL_CHECK=1 python src/run_final.py
echo "done: submission$SFX/matching_results.tsv, submission$SFX/candidate_pairs.tsv"
