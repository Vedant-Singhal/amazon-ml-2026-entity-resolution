# Test diagnosis report — v4 at threshold 0.95 (LB 0.98414) (2026-09-27)

Everything below is measured from the provided data, v4's train OOF, and the LB scores of the files we submitted.
Test has no labels, so test numbers are estimates; each says how it was obtained.

## 1. Pipeline on train (ground truth, v4 OOF)
| where the 7.60M true train pairs go | pairs | share |
|---|---|---|
| not found by search (not in the candidate pool) | 27,488 | 0.36% |
| dropped by the stage-1 shortlist | 6,773 | 0.09% |
| rejected by the model | 151,656 | 2.0% |
| **found** | 7,452,448 | **97.6%** |
False pairs on train: only 9,480. Misses by type: no address 39% of their true pairs, no street number 12.9%,
name shared by 4+ S1 3.1%.

Search recall per route (train, before the fine-tuned retriever): e5-small forward 0.961, e5-large forward 0.963,
reverse (record → S1) 0.976 / 0.980, name char-3-grams 0.626, address char-3-grams 0.809, IDF keys 0.485, exact keys
0.15–0.46; all routes together 0.9908 (92 candidates/S1). + fine-tuned retriever → **0.9964**; shortlist keeps 0.9955
at 5.85/S1.

## 2. Is the loss "hallucination" (false matches) or misses? — mostly misses
LB loss at 0.98414 = 0.0159 × 1,732,544 ≈ **27.5k S1-units**. Split (sources in brackets):
| error | amount | cost (S1-units) |
|---|---|---|
| false pairs among the 0.95–0.99 matches (~15% of 125k) [LB diffs] | ~18.8k pairs | ~3.8k |
| S1 whose whole answer is uncertain and wrong [S1 audit] | ~0.8k S1 | ~0.8k |
| false pairs among ≥ 0.99 matches [profile ≈ true pairs] | small | ~1–2k |
| **false matches total** | | **~6–7k (≈ 25%)** |
| S1 wrongly left empty (101.6k empty vs ~96.7k expected) [train no-match rate] | ~4.9k S1 | ~4.9k |
| true pairs cut at 0.95 (~60% of 71.9k) [LB diffs] | ~43k pairs | ~3.0k |
| pairs never predicted (search / model misses) [remainder; cluster-size deficit] | ~180k pairs | ~13k |
| **misses total** | | **~21k (≈ 75%)** |
Proof on the LB: stricter (0.99) removed 131k pairs that were ~85% right → 0.980; looser (v4 rule) added 72k pairs
~60% right → 0.983. At 0.95 the file is balanced — fewer false matches only by losing more true ones.

## 3. France (no labels)
| | France | India | US |
|---|---|---|---|
| test S1 | 259,452 | 809,986 | 663,106 |
| predicted matches / S1 (train truth 3.46) | **3.30** | 3.37 | 3.39 |
| estimated share of true pairs missed (if clusters are train-sized) | **~4–6%** | ~3% | ~2.5% |
| S1 predicted empty (train no-match 5.58%) | 5.99% | 5.84% | 5.85% |
| uncertain (0.95–0.99) pairs per S1 | **0.23** | 0.09 | 0.11 |
| S1 answered only with uncertain pairs | 1,302 | 2,173 | 1,809 |
| S1 possibly wrongly empty | 1,610 | 1,077 | 1,503 |
| S1 sharing their exact name with another S1 | **34%** | 44% | 29% |
Why France is hard: generic names (`bordeaux club sarl` ×205, ~10 cities) → the same name at another street of the same
city is common and looks like a match; the lexicon never learned French (`rue→r` in 25% of records); the model never
saw France.

## 4. What the test pairs look like (label-free checks, all done)
- Pairs kept at ≥ 0.99 look exactly like true train pairs on every signal (name/address agreement, numbers, cluster
  structure, blatant-mismatch rate 196 of 5.84M, below what true pairs alone would give).
- The uncertain pairs are: records without address (~24%), street-number conflicts, renames at the same building,
  generic-name look-alikes (France). No simple rule separates right from wrong inside them (LB probes: no-address
  ~0.90, number conflicts ~0.72, street identity ~0.80, classical Fellegi–Sunter disagrees with v4 where v4 is right).
- Exact normalized name+address matches: v4 agrees 97.8% (France) / 86.6% (India) / 99.35% (US); the "missed" ones are
  explained by test's 2× orphan records (identical-looking decoys).
- Train → test shift: test has ~2× the orphan records per S1 (42% vs 26%); S1 records are identical.

## 5. What can still raise the score
1. **Retrain (running now):** v5a/b/c/d — orphan-density training and the mined vocabulary; they target the ~21k-unit
   miss side and France. Submit each `output/t0950`.
2. **`v4/output_diag/t095_fill_empty`:** 4,190 likely wrongly-empty S1 (break-even 50%, ceiling ≈ 0.9866).
3. **Next analysis:** cluster completion — records identical (after normalization) to a *member* of a predicted
   cluster but left unmatched (twins across S2/S3), checked on train first.
4. Longer term: name-only (no-address) records and big clusters (the train miss types).
