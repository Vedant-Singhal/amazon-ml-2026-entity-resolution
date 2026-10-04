# 4. Development journey

How the solution grew over four days, what each version changed and why, and what we learned. Per-version
details and reports: [`../versions/`](../versions/).

| date (2026) | version | key change | train OOF | public LB |
|---|---|---|---|---|
| 25 Sep | v0 | name-only TF-IDF search, 9 features, XGBoost | — | 0.593 |
| 25 Sep | teammate v1 | char-3-gram sketch views, 33 features, MLP | (0.955 own hold-out) | 0.904 |
| 25–26 Sep | v2 | learned vocabulary, dense search both ways, 44 features, LightGBM, one owner | 0.9781 | 0.9700 |
| 26 Sep | teammate, later | (code not archived) | | 0.964 |
| 26 Sep | v3 | 13 search routes, learned shortlist, ~100 features, GPU XGBoost, ablations | 0.9881 | 0.97493 → 0.978 at t 0.95 |
| 26–27 Sep | v4 | fine-tuned retriever, cross-encoder, graph features, stage-3 stack | 0.9913 | 0.983 → **0.984312** at t 0.95 |
| 27 Sep | v5a | v4 trained at test orphan density, mined abbreviations, 3,000 trees | 0.9910 (harder) | 0.98376 |
| 27 Sep | **final** | **blend 0.7 · v4 + 0.3 · v5a at v4's pair count** | +0.0001 | **0.984403** |
| 27 Sep | v6 | third embedder (e5-large-instruct) | written, not run | |

## Day 1: understand the data before modelling

- **Exploratory analysis** on the full data ([`../eda/`](../eda/)): no true pair crosses countries; no record belongs
  to two S1; address agreement is the strongest signal; names are ambiguous (31% duplicate S1 names); off-the-shelf
  e5 on `name | address` already retrieves 95–98% of true pairs in its top 50.
- **Literature review** ([`06-literature.md`](06-literature.md)) gave the blueprint: lexical + dense candidate search
  (Sparkly, SC-Block), a gradient-boosting core matcher (Papadakis et al.), a cross-encoder for hard cases (Ditto),
  one owner per record (MSCD-HAC), and an expected-F decision rule (Lipton et al.).
- **v0 (0.593)**: the first submission searched by name only (20 candidates per S1), used 9 features and allowed a
  record to have several owners. Its 0.593 shows how much the address, the one-owner rule and the empty answer matter.
- **Teammate v1 (0.904)** ([`../versions/teammate/`](../versions/teammate/)): character-3-gram TF-IDF views of name,
  address and both, count-sketched to 256 dimensions and searched with FAISS. Then 33 features, a GPU MLP and one
  global threshold. Its own validation (0.955) was on a 30% sample with three times fewer look-alikes per S1 than the
  real data, an early hint that density matters. Its address-only and typo-robust views were adopted in v3.

## v2: a clean, learned foundation (OOF 0.9781, LB 0.9700)

- **Learned lexicon**: from the true train pairs, count which word of an S2/S3 record replaces which word of its S1,
  and keep frequent, consistent mappings (≈ 2,000 name + 700 address maps, including scripts:
  `महाराष्ट्र → maharashtra`). Plus mojibake repair, zero-width removal, alias splitting and the core name.
- **Search**: e5-small top 25 per S1 **and** top 5 S1 per record (the reverse direction has higher recall, 0.975 vs
  0.961), plus IDF-weighted rare keys: recall 0.981 at 56 per S1.
- **Matcher**: 44 features, LightGBM with 5-fold GroupKFold by S1, isotonic calibration, one owner per record,
  threshold tuned with the exact metric.
- **Error diagnosis** found precision ≈ 99.2% everywhere and recall as the loss. It also produced a taxonomy (E1–E8:
  no address, house numbers, trade names, native script, typos, near-twins) that defined v3.

## v3: scale and targeted features (OOF 0.9881, LB 0.97493)

- **Search**: two dense views (e5-small + e5-large, chosen by a 14-model ablation), name and address character
  sketches, capped exact keys. The pool reached recall 0.9908 at 92 per S1. Exact GPU search beat approximate IVF,
  which lost recall.
- **Learned shortlist**: a fast stage-1 XGBoost keeps the smallest candidate set holding 99.9% of the pool's recall
  (5.4 per S1, recall 0.9902). This answered the "small candidate set" requirement: 11× fewer candidates than v2,
  with higher recall.
- **About 50 new features**, one group per error type, and **XGBoost on GPU** (equal to LightGBM, faster on our
  machine).
- **The leaderboard gap.** v3 gained +0.010 on train but only +0.005 on the LB (OOF 0.9881 vs LB 0.97493). We tested
  hypotheses one by one:
  - **Fold-consistent test scoring:** 0.974654, so not a scoring mismatch.
  - **Gap check:** test S1 have more plausible look-alikes.
  - **Threshold probes:** 0.958 at t 0.30 and **0.978 at t 0.95**. Test is **precision-limited**, and the cut must be
    far stricter than the train optimum.

## v4: text-reading models and a robust stack (OOF 0.9913, LB 0.984312)

- **Fine-tuned retriever** (SC-Block idea + romanized copies): e5-small trained on our own pairs. Alone it reached
  recall 0.9946 at 31 per S1; merged into the pool, recall rose to 0.9964 and shortlist recall to 0.9955.
- **Cross-encoder** reading both records together on the uncertain band, **graph-consistency features**, and a
  **stage-3 stack** that adds v2's score as an independent opinion.
- **Density-robust design.** Stage 3 drops count features whose meaning changes with how many candidates exist. This
  shrank the train → LB gap from 0.013 to 0.008.
- **Diagnosis of the remaining gap:** test has twice as many orphan records per S1 (≈ 42% vs 26%). These are clean,
  look-alike records, and they already cause 63% of the false matches on train. With the leaderboard-difference
  method we measured band precision on test (0.95–0.99 band ≈ 0.85; rule-to-0.95 pairs ≈ 0.60) and set the cut at
  **0.95: LB 0.984312**.
- **Post-processing exhausted.** Every rule-based filter or addition tested was at or below break-even (dropping
  number conflicts, no-address pairs or blatant mismatches; adding twins or exact-key pairs; Fellegi–Sunter).
  Further gains needed retraining.

## v5: train the way test looks (LB 0.98376 alone, 0.984403 blended)

- **S1 dropout**: remove 19% of train S1 after the search and keep their records as orphans. Train then has test's
  orphan density, and stages 1–3 learn in that setting.
- **Abbreviation mining** from the raw records, per country and without labels (Auto-FuzzyJoin idea). It found 88
  French address maps (`r → rue`, `bd → boulevard`) that the train-only lexicon could never learn.
- **3,000 stage-1 trees**, since v4 hit its 1,500 cap.
- Four variants were launched (a: all changes; b: no dropout; c: no mining; d: 30% dropout). Under time and budget
  limits only **v5a** finished in time to submit. Alone it was more conservative than v4 (0.98376).
- **Blend at equal pair count.** On test-like training data, 0.7 · v4 + 0.3 · v5a gained +0.0001 over v4 at the same
  number of predicted pairs, the best of 14 weightings and 3 voting rules. On the LB it gained +0.00009: **0.984403,
  the final submission**.

## v6 (not run): a third view

A third embedder, `multilingual-e5-large-instruct` (best of the ablated models on records without an address), would be
added to an existing pool as an extra dense route (`addview.py`), with a complementarity test against the two current
views (`run_complement.py`). It was written but not run before the deadline.

## After the challenge

- **Look-alike analysis** ([section 3](03-results-and-analysis.md#look-alike-patterns-same-address-one-name-word-changed)):
  a one-word name swap at the same address is usually a true match in train (92–97%). Our model accepts it at the
  right rate for US and India but rejects many French swaps (`groupe`, `cie`, `développement`), the clearest remaining
  France weakness.

## Lessons learned

1. **Find the train → test difference early.** The biggest single insight, twice as many orphan look-alikes on test,
   explained the whole train → LB gap and set both the threshold and Model B.
2. **The leaderboard is a measuring instrument.** Because the metric averages over S1, comparing two nested
   submissions measures the precision of the pairs that differ. That turned a handful of probes into hard evidence.
3. **Compare models at equal output size.** Train-optimal thresholds did not transfer, so blends and votes were judged
   at a fixed pair count, and the training estimate (+0.0001) matched the leaderboard (+0.00009).
4. **Fine-tune retrieval on your own pairs.** One supervised-contrastive retriever beat thirteen hand-designed search
   routes on recall per candidate.
5. **Learn the vocabulary from the data.** The learned lexicon and the label-free miner covered abbreviations,
   scripts and an unseen country without a single hand-written list.
6. **Make stacking leak-free and density-robust.** Every stacked score was out-of-fold with shared folds, and
   features that depend on candidate counts were dropped from the final stack.
7. **Smoke-test every full run.** Each configuration first ran on 10k S1. That caught crashes (a 16-bit rank overflow
   from a "hub" record, a GPU out-of-memory error) before they cost hours.

## What we would do next

- **France vocabulary for generic words**: treat mined French legal and generic words like `ltd` / `services`, the
  largest remaining France loss.
- **Look-alike modelling**: features that tell a generic-word swap from a distinguishing-word swap, trained at test
  orphan density.
- **A stronger judge for the uncertain band**: a multilingual reranker (Apache-2.0) trained with orphans as hard
  negatives.
- **Finish the v5 variants and v6**, and blend them on the same equal-pair-count protocol.
- **Cluster-level decisions**: use S2 ↔ S3 agreement to confirm or veto S1 links (TransClean idea).
