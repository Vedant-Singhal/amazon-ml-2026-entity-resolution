# LB diagnosis — where v4 (0.983) still loses (2026-09-27)

No test labels; the evidence is (1) train OOF errors, (2) submission diffs + their LB scores. Macro F0.5 is a mean
over test S1 (N = 1,732,544: France 259,452 / US 663,106 / India 809,986), so an LB difference × N = sum of per-S1
F0.5 changes on the S1 whose predictions differ. Script: lbdiff.py (pure Python, local).

## 1. Marginal pairs are ~75% precise on test

| comparison (looser vs stricter, nested) | extra pairs | LB diff × N | all extra correct / all wrong | implied precision |
|---|---|---|---|---|
| v4 best rule (0.983) vs v4 t=0.99 (0.980) | 202,656 | +5,198 | +23,263 / −51,749 | **0.75** (0.73–0.78) |
| v3 rule (0.97493) vs v3 t=0.95 (0.978) | 170,525 | −5,319 | +18,640 / −42,926 | 0.60 (0.58–0.63) |
| v3 t=0.30 (0.958) vs v3 t=0.95 (0.978) | 433,550 | −34,651 | +41,595 / −99,523 | 0.45 |

Break-even precision for adding a marginal pair is ≈ 0.69. v4's marginal band (p between its rule cut and 0.99)
is just above it: net positive, but it holds ~50k false matches. **If that band were judged perfectly, the gain would
be up to +23.3k − 5.2k ≈ 18k S1-units ≈ +0.010 LB — more than the whole gap to the LB top (0.992).** The remaining gap
lives in the uncertain band, not in search recall.

## 2. France concentrates the uncertainty

| | France | India | US |
|---|---|---|---|
| v4 marginal pairs (best rule − t0.99) per S1 | **0.231** | 0.087 | 0.109 |
| S1 where v4 best and v3 t0.95 differ | **18.2%** | 9.6% | 8.4% |
| S1 where v4 and v3 disagree non-nested | **4.0%** | 0.6% | 0.8% |

France (unseen in train) has 2–2.6× more uncertain pairs per S1 and 5–6× more model disagreement.

## 3. Train OOF (v4): what is missed

186k of 7.60M true pairs missed (2.4%): 27.5k not found by search, 6.8k dropped by shortlist, 151.7k rejected by the
model; only 9.5k false pairs. Misses by type: **no address 39% missed** (119.5k of the misses), **number missing
12.9%**, name shared by 4+ S1 3.1%, one-digit number edit 2.9%. On train the model is recall-limited; on test the
marginal band is precision-limited — the test-specific over-confidence (density + France) is the gap.

## 4. Per-country probes (local, validator PASS) — how to read them

| folder | content | read |
|---|---|---|
| `v4/output_diag/fr_strict/` | v4 best, France rows from t0.99 | LB − 0.983 = France's share of the strict effect; > 0 → France wants a stricter cut |
| `v4/output_diag/fr_empty/` | v4 best, France rows empty | France F0.5 ≈ 0.055 + 6.678 × (0.983 − LB) |
| `v4/output_diag/us_empty/` | v4 best, US rows empty | US F0.5 ≈ 0.057 + 2.613 × (0.983 − LB); India = (0.983·N − Fr − US)/N_India |

(0.055 / 0.057 = assumed singleton rate: v4's predicted empty rate on test and train truth; ±0.01 → ±0.01 on F.
Public LB is a subset of test, so these are estimates.)

## 5. Levers (in order)
1. Better judge of the uncertain band: stronger cross-encoder (larger base, 2–3 epochs, hard negatives; now band AUC
   0.894 and only 2.7% of stage-3 gain), trained on exactly the pairs in the band.
2. Country-specific decision (per-country threshold) if fr_strict says France wants a stricter cut.
3. No-address / missing-number records (the train recall misses).

## 6. From 0.98414 (t0.95) toward 0.99 — local, label-free (2026-09-27)
Gap: +0.0058613 × 1,732,544 ≈ 10,155 S1-units ≈ 51k fewer false pairs, or ~145k more found pairs, or ~10k fewer
fully wrong S1.

**Matches per S1 (train GT vs v4 t0.95 test):** mean 3.46 (train US/India) vs 3.39 US, 3.37 India, 3.30 France. Test has
too many S1 with 0–3 matches and too few with 5+ (excess 0–1 matches: France +5.1k, India +8.3k, US +5.9k S1; deficit
5+: France −8.0k, India −15.3k, US −8.4k). If test clusters are sized like train's, v4 splits big clusters (misses
~160k pairs net). Caveat: the record-mixture estimate (orphan share 41–42%) gives ~3.35–3.38 true matches/S1, i.e.
test clusters may be slightly smaller — then the count is about right. Source mix: "both S2 and S3" 76.9% France,
78.7–78.8% US/India vs 80.5% in train GT.

**What the uncertain band looks like (0.95–0.99, 130.8k pairs, ~0.85 precise):** ~24% of its S2/S3 records have no
address (true train pairs: 4–5%; ≥0.99 band: 1–2%) → name-only decisions; street numbers conflict in 51% (US), 17%
(India), 25% (France) of pairs where both have numbers (true pairs: 7% / 3%); exact-name rate lower than true pairs.
The ≥0.99 band looks exactly like true train pairs on every signal. Note: band true pairs are uncertain *because*
their evidence conflicts, so these rates do not give precision directly — LB probes do.

**Probes built (validator PASS):** v4/output_diag/t095_drop_numconf (−28,542 band pairs with conflicting numbers;
US 16.0k, India 6.3k, France 6.2k) and v4/output_diag/t095_drop_noaddr (−31,393 band pairs whose record has no
address). LB up → that group was < ~0.69 precise.

## 7. Probes and self-checks after the 0.98414 best (2026-09-27)
- **drop no-address band pairs (−31,393) → LB 0.983**: those name-only pairs are ~0.90 precise → keep them.
- **Self-consistency (30% of S1):** v4's predicted clusters have train-like structure — street-number outliers India
  1.05% / US 2.51% / France 1.22% of members (train true 1.01% / 2.24%); name outliers 9.0% / 6.85% / 8.4% (train 9.2% /
  7.0%). Unmatched exact copies of a member are mostly real decoys (same name, other street). France has ~10x more
  same-name look-alikes (generic names). No structural bug.
- **v4 vs v3 (both t0.95; +0.00614 = +10.6k units):** v4 added 126k pairs (partial-name/numbers-agree 53.7k — India 36.5k;
  number conflicts 25.5k — US 15.4k; no-address 18.4k; same-address renames 17.3k; native script 8.6k) and removed 36k;
  3,645 S1 empty in v3 got matches. The gain is recall on hard pair types — the same types hold the remaining errors.
- **Loss at 0.98414 ≈ 27.5k units:** band 0.95–0.99 false pairs ≈ 6.8k (all-correct vs actual, 14,281 − 7,450); the
  ~60%-right pairs cut at 0.95 hold up to ~5k if judged perfectly; the rest (~15k) = pairs never predicted + a few
  false pairs ≥ 0.99.
- New probe **v4/output_diag/t095_addback_safe**: +23,427 cut pairs of safe-looking types (numbers agree with a shared
  name word, native script, web names); left out 48,469 (number conflicts, no shared word, no address).
