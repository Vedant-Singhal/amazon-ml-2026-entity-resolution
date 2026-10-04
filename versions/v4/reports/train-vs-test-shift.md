# Train vs test — what changes and how much (2026-09-27)

Local pure-Python profiles of the provided files (dataset/train, dataset/test) + v4's test predictions, and a
train-OOF simulation job (`ber-orphansim-0927-011125`, output/orphansim.md). Scripts: shift.py, orphans.py, orphprof.py.

## 1. S1 records: identical
US and India S1 have the same profile in train and test (address length, numbers, name length, script, duplicates
up to split size). Test S1 were sampled like train S1.

## 2. S2/S3 records: only the orphan share changes
Orphan = S2/S3 record that belongs to no S1 of its split.

| | train US | test US | train India | test India | test France |
|---|---|---|---|---|---|
| S2+S3 records per S1 | 4.67 | 5.76 | 4.68 | 5.82 | 5.53 |
| orphan share | 26.0% (GT) | ~41–42% (est.) | 26.0% (GT) | ~42% (est.) | ~38% (v4 pred.) |
| orphan records per S1 | 1.22 | ~2.4 | 1.22 | ~2.4 | ~2.1 |

Train, by GT — orphans are *clean* records, matched records are the noisy copies:

| train | address empty | address has number | web-style name | name chars |
|---|---|---|---|---|
| US matched | 4.74% | 87.8% | 5.60% | 22.7 |
| US orphan | 0.32% | 99.1% | 0.64% | 26.9 |
| India matched | 3.92% | 87.8% | 4.59% | 25.9 |
| India orphan | 0.25% | 99.0% | 0.50% | 31.0 |

The test S2/S3 profile (address empty US 2.89% vs 3.59% train, web names 3.55% vs 4.31%, longer names) is exactly the
train mixture with ~42% orphans: solving empty-address and web-name mixtures independently gives orphan shares of
41.3–42.2% for both US and India. **Same generator, ~2x the orphan records per S1** (test ≈ train with ~19% of S1
removed and their S2/S3 records kept).

## 3. Why that hurts
- Orphans are the main false-match source already on train: **6,007 of 9,480 v4 false pairs (63%) go to orphan
  records.** They are clean, complete records — strong-looking candidates for a similar S1 (same chain, nearby address).
- On test v4's uncertain band (pairs between its rule cut and p 0.99) has **0.117 pairs/S1 vs 0.066 on train (1.8x;
  France 0.23 = 3.5x)**, and its precision is **~0.75 on test (LB-implied) vs 0.947 on train OOF**.
- The OOF-only simulation (drop 0–30% of train S1, keep OOF scores) leaves F0.5 at 0.9913 — the scores were computed
  with the true owner present (competition features: record's best other S1, ranks, graph), so the model still
  rejects the leftover records. The effect needs retraining with S1 actually removed.

## 4. How far from 0.990
- LB 0.983 → 0.990 = +0.007 × 1,732,544 S1 = **~12,100 S1-units of F0.5**.
- Cost per error for a typical S1 (3–4 true matches): one false pair ≈ −0.20, one missed pair ≈ −0.07, a wrong
  non-empty answer for a singleton / an empty answer for a matched S1 = −1.0.
- So 0.990 needs about **60k fewer false pairs**, or ~175k more found pairs, or ~12k fewer fully wrong S1 (or a mix).
- Loss budget: test loses 0.017 × N ≈ 29.5k units; train-level quality on the same N would lose 0.0087 × N ≈ 15.0k.
  The ~14.5k excess is the train→test shift. 0.990 = removing ~84% of that excess, i.e. making v4 behave on test as
  it does on train. The LB-implied false pairs in the band alone are ~50k.

## 5. Mitigations (ranked by evidence)
1. **Train at test-like orphan density** (S1-dropout): drop ~19% of train S1 at `prepare` (hash-based), keep all their
   S2/S3 records as orphans, run the whole pipeline. Every feature, the calibration and the rule then see the test
   situation, and the OOF becomes an honest LB estimate. (Optional: 2 replicas with different dropped sets.)
2. **Orphan-aware evidence**: record-quality features (clean record = likely orphan when unclaimed), S2↔S3 cluster
   evidence (an orphan S2 with a matching S3 sibling forms a cluster without this S1), trained at test density.
3. **Band judge** (cross-encoder) trained with orphans as hard negatives at test density.
4. **Per-country cut** (France has 3.5x the uncertain pairs; per-country probes in v4/output_diag/).
5. Recall on no-address / missing-number records (39% / 13% missed on train) — second order under F0.5.
