"""v5 entry point = v4 + three train-side changes, with every setting built in (any env var set before overrides it):
  BER_TRAIN_S1_DROP=0.19  train at the test's orphan density (ber/s1drop.py)
  BER_ROUNDS1=3000        stage-1 tree cap (v4 stopped at its 1,500 cap)
  BER_MINE_ABBR=1         abbreviations mined from the records themselves, per country (ber/mining.py), e.g. France
                          r->rue, bd->boulevard, st->saint; applied to both splits before the learned lexicon
Plus v4's settings: two embedders, exact GPU search, 99.9% shortlist, fine-tuned retriever, strict-threshold files.
Submission: output/t0950/ (threshold 0.95, the leaderboard-validated cut)."""
import os

V5 = {
    "BER_EMB_MODELS": "intfloat/multilingual-e5-small,intfloat/multilingual-e5-large",
    "BER_ANN_MIN": "1000000000",
    "BER_SL_KEEP": "0.999",
    "BER_V4_RETRIEVER": "1",
    "BER_TRAIN_S1_DROP": "0.19",
    "BER_ROUNDS1": "3000",
    "BER_MINE_ABBR": "1",
    "BER_V4_EXTRA_T": "0.93,0.95,0.97",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
for k, v in V5.items():
    os.environ.setdefault(k, v)

import run_v4  # noqa: E402  (reads the settings at import time)

if __name__ == "__main__":
    run_v4.main()
