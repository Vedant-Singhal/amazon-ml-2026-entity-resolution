# v2: learned normalization + dense search (OOF 0.9781, LB 0.9700)

Code: [`pipeline/src/v2/`](../../pipeline/src/v2/). In the final solution it is the **auxiliary matcher** (step 0),
whose out-of-fold score `p_v2` is one input of the stage-3 stack.

| | value |
|---|---|
| train OOF macro F0.5 (5 folds) | **0.9781 ± 0.0001** (folds 0.9780 / 0.9783 / 0.9781 / 0.9781 / 0.9782); US 0.9795, India 0.9760 |
| candidate recall | 0.981 at 56 candidates per S1 (India 0.972, US 0.987) |
| public LB | 0.9700 |

**Pipeline**
- **Prepare:** learned lexicon (2,034 name / 690 address mappings, including multi-word ones), mojibake fix,
  zero-width removal, dotted-abbreviation merge, URL strip, alias split, core name (≥ 2 words), learned landmark cues,
  `anyascii` for remaining non-Latin words.
- **Embed:** `multilingual-e5-small` on raw `name | address`.
- **Search (within country):** dense top 25 per S1 ∪ dense top 5 S1 per record ∪ IDF-weighted rare keys (top 25).
- **Features:** 44 (no country).
- **Train:** LightGBM (lr 0.1, 127 leaves), 5-fold GroupKFold by S1 (`fold = hash(S1 id, 13) % 5`), hard negatives
  kept + 15% of easy ones (reweighted).
- **Decide:** isotonic calibration → one owner per S2/S3 record → threshold 0.65.

**Reports**
- [`reports/error-diagnosis.md`](reports/error-diagnosis.md): precision ≈ 99.2% everywhere; the loss is recall (no
  address, house numbers, trade names, native script, typos).
- [`reports/deep-diagnosis.md`](reports/deep-diagnosis.md): what separates each error type (e.g. name uniqueness
  AUC 0.85 on no-address records, truncated house numbers are match evidence). This drove the v3 features.
- [`metrics/`](metrics/): `report.json` (scores, feature gain), `decision.json` (decision-rule grid).
