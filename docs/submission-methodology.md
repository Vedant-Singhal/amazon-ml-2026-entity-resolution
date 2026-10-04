# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** 4Miracles
**Team Members:** Shreyansh Gupta (Team Leader), Vedant Singhal, Kumaradithya Chadalavada, Jithin Shaji
**Submission Date:** September 2026

---

## 1. Executive Summary

We built one entity-resolution pipeline and trained it in two ways.

- **Model A, the main pipeline** (`run_v4.py` in the code), runs these steps in order:
  1. clean the text with a vocabulary learned from the training pairs;
  2. search each country for candidates in several complementary ways, including a retriever fine-tuned on the
     training pairs;
  3. shortlist about 6 candidates per S1 with a fast model;
  4. score them with a three-stage stacked matcher (XGBoost, a cross-encoder and graph-consistency features);
  5. give each S2/S3 record to at most one S1.
- **Model B** (`run_v5.py`) is the same pipeline trained after removing 19% of the training S1s, so the training data
  has as many unmatched records as the test set. It also mines each country's abbreviations from the records.
- **The submission blends them:** p = 0.7 · p_A + 0.3 · p_B, cut to Model A's leaderboard-validated pair count.

| | train, out-of-fold macro F0.5 | public leaderboard |
|---|---|---|
| Model A | 0.9913 | 0.984312 |
| Model B | 0.9910 (on its harder training set) | 0.98376 |
| **Submitted blend** | +0.0001 over Model A at equal pair count | **0.984403** |

The candidate search finds 99.64% of the true pairs, and the shortlist keeps 99.55% at 5.85 candidates per S1. Only
the provided data is used; model and software licenses are MIT, Apache-2.0, BSD or ISC, and the largest pretrained
model has 0.56B parameters.

---

## 2. Methodology

### 2.1 Problem Analysis

**Scale.**

| | S1 | S2 | S3 | true pairs |
|---|---|---|---|---|
| Train | 2,206,821 | 5,034,616 | 5,285,603 | 7.60M (3.46 per S1; 5.6% of S1 have none) |
| Test | 1,732,544 (US 663,106 · India 809,986 · **France 259,452, absent from train**) | 4,887,273 | 5,082,316 | hidden |

**How S2/S3 records differ from their S1** (measured on the train ground-truth pairs):

| noise | examples |
|---|---|
| abbreviations and expansions | `st`/`street`, `pvt`/`private`, `up`/`uttar pradesh`, `new york`/`ny`; in France `r`/`rue`, `bd`/`boulevard` |
| legal-form / generic words added, dropped or swapped | `llc`, `ltd`, `services`, `center`, `partners`: at the same house number, a one-word swap is still a true match 92–97% of the time |
| scripts | Hindi / Tamil / Bengali script against the Latin spelling of the S1 |
| encoding and OCR noise | mojibake (`Ã©` for `é`), zero-width characters that split Hindi words, digit-for-letter swaps (`8lack` for `black`) |
| extra parts | web-style names (`.com`, `www`), aliases (`dba`, `aka`, brackets), honorifics |
| missing or changed address parts | no address on 4.7% of matched US records; house numbers dropped or changed; city/state shortened |

**Findings that shaped the design:**
1. **Records without an address or a house number are the hardest.** Model A misses 39% and 12.9% of their true pairs,
   and they produce most look-alike false matches.
2. **Names are shared.** 29–44% of test S1s share their exact name with another S1 (chains, generic names). The address
   and the competition between candidates matter as much as name similarity.
3. **The test set has twice as many unmatched records.** An "orphan" is an S2/S3 record that belongs to no S1.
   - Train has 1.22 orphans per S1 (26% of records); test has about 2.4 (≈ 42%).
   - We measured this without labels: the test S2/S3 profile (share with empty address, share with web-style names)
     equals the train mixture at 41–42% orphans, for both US and India.
   - Orphans are clean, complete records that resemble a nearby S1, and they cause 63% of the false matches on train.
   - So test needs a stricter decision than train suggests (section 4, step 11), and a model trained at the test's
     orphan density helps (Model B).
4. **France is unseen.** Its abbreviations and legal words (`sarl`, `cie`, `groupe`) never appear in train. It has 2–3×
   more uncertain pairs per S1 than US/India, and 34% of French S1s share an exact name.

### 2.2 Solution Strategy

**Approach Type:** Hybrid: multi-route blocking → learned shortlist → stacked classifiers (gradient boosting +
cross-encoder + graph features) → one-to-one assignment, trained under two regimes and blended.

**The pipeline, first step to last:**

```
 raw S1 / S2 / S3 records
   │
   ├─ step 0  auxiliary matcher (LightGBM, 44 features) ──────────── its score becomes a stage-3 feature ─┐
   │                                                                                                      │
   ├─ step 1  normalization: learned lexicon, generic words, typed numbers, keys                          │
   ├─ step 2  embeddings: multilingual-e5-small + e5-large                                                │
   ├─ step 3  candidate search in each country: dense kNN, char sketches, rare keys, exact keys           │
   ├─ step 4  fine-tuned retriever route added  → pool: 110 per S1, recall 0.9964                         │
   ├─ step 5  stage 1: fast XGBoost → shortlist: 5.85 per S1, recall 0.9955  (= candidate_pairs.tsv)       │
   ├─ step 6  101 pair features                                                                           │
   ├─ step 7  stage 2: XGBoost → p2                                                                       │
   ├─ step 8  cross-encoder on uncertain pairs → p_ce                                                     │
   ├─ step 9  graph-consistency features                                                                  │
   ├─ step 10 stage 3: XGBoost stack (features + p2 + p_ce + graph + auxiliary score) ←───────────────────┘
   └─ step 11 decision: one S1 per record, p ≥ 0.95                               = Model A (LB 0.984312)

 Model B = the same steps with 3 training changes (19% of train S1 removed, mined
           abbreviations, 3,000 stage-1 trees)                                     (LB 0.98376)
 Submission = 0.7 · Model A + 0.3 · Model B, cut to Model A's pair count           (LB 0.984403)
```

**Core Innovations:**
1. **Vocabulary learned from the data, not written by hand.** A lexicon learned from the ground-truth pairs, and in
   Model B a label-free abbreviation miner that gives France its vocabulary.
2. **Complementary search routes plus a fine-tuned retriever.** The retriever removed 61% of the true pairs the other
   routes missed.
3. **Leak-free three-stage stacking.** The folds `hash(S1 id, 13) % 5` are the same in every stage, so every stacked
   score is out-of-fold.
4. **Training at the test's orphan density (Model B).**
5. **Measuring pair precision on test from leaderboard differences** to set the operating point.
6. **A blend compared at equal pair count.** It can only swap weaker pairs for stronger ones; it never becomes looser
   or stricter.

---

## 3. Candidate Generation (Blocking)

This section covers steps 1–5. All routes run within the record's own `country` label. The label is used only as a
partition, never as a feature, and no country is hard-coded.

**Step 1: Normalization** (`ber/prepare.py`, `text.py`, `lexicon.py`). Every record is cleaned the same way.
- **Text repairs:** mojibake repair; zero-width characters removed; URLs stripped; dotted abbreviations merged
  (`l.l.c` → `llc`); aliases split out (`dba`, `aka`, brackets); repeated words collapsed.
- **Learned lexicon.** For every train ground-truth pair, words or word pairs of the S2/S3 record that are missing from
  the S1 are counted against the S1 words missing from the record. A mapping is kept when seen ≥ 20 times and
  consistent in ≥ 50% of cases, and it always points to the S1 spelling. That gives about 2,000 name and 700 address
  mappings: `st → street`, `mh → maharashtra`, `new york → ny`, native-script words → Latin.
- **Romanization:** remaining non-Latin tokens go through `anyascii`.
- **Generic words:** words in ≥ 1% of a country's names (computed on each split's own records) are removed to form the
  core name (at least 2 words are kept).
- **Keys:** compact name, sorted core, consonant skeleton, metaphone codes, OCR-repaired name (`0→o`, `8→b`), initials.
- **Typed numbers:** house number, PIN/ZIP, unit; numbers after learned markers such as `box` are not treated as the
  main number.
- **Uniqueness counts:** how many S1s and records of the country share the core name, the address, and the
  number-plus-street key.

**Step 2: Embeddings** (`ber/embed.py`). `intfloat/multilingual-e5-small` and `multilingual-e5-large` embed
`"name | address"` (fp16, 64 tokens, L2-normalized). The two models have complementary strengths: e5-large is better
overall and on native script, e5-small on records without an address.

**Step 3: Candidate search** (`ber/block.py`, `sketch.py`). Each route adds pairs to one pool, which keeps every
route's rank for every pair.

| route | what it matches | per S1 | recall alone (train) |
|---|---|---|---|
| e5-small, S1 → records | 25 nearest records by cosine | 25 | 0.961 |
| e5-large, S1 → records | same with the large model | 25 | 0.963 |
| e5-small / e5-large, record → S1 | each S2/S3 record's 5 nearest S1 | 23 | 0.975 / 0.980 |
| name 3-gram sketch | TF-IDF of core-name character 3-grams, count-sketched to 256 dims | 10 + 3 | 0.626 |
| address 3-gram sketch | same on the address core, number-compatible pairs only | 10 + 3 | 0.809 |
| IDF shared keys | rare shared name words, address words and numbers, IDF-weighted | 20 | 0.485 |
| 6 exact keys | core name, compact name, sorted core, consonant skeleton, address+number, set of house numbers; keys shared by more than 10 S1 or 50 records are skipped | ≈ 1–5 each | 0.15–0.46 |

The union reaches **0.9908 recall at 92 pairs per S1**.

**Step 4: Fine-tuned retriever** (`ber/retriever.py`).
- **Model and training:** e5-small is fine-tuned on the train pairs with a contrastive loss (InfoNCE, in-batch
  negatives, scale 20, lr 2e-5, batch 256). Romanized copies of native-script records serve as extra positives.
- **Leak-free:** on train it is cross-fitted, so the model trained on one half of the S1s searches for the other half.
  Test uses a model trained on all pairs.
- **Search:** it adds its top 10 records per S1 and top 3 S1 per record.
- **Result:** the pool reaches **0.9964 recall at 110 pairs per S1**.

**Step 5: Stage-1 shortlist** (`ber/stage1.py`).
- **Model:** a fast XGBoost on 51 cheap features (route ranks, cosines, key flags, quick similarities), trained 5-fold
  out-of-fold on the whole pool (lr 0.1, ≤ 1,500 trees).
- **Cut:** per S1, keep the candidates with rank ≤ 10 and p1 ≥ 0.003. That is the smallest (rank, threshold) rule that
  holds 99.9% of the pool's recall, chosen on out-of-fold p1.
- **Result:** **recall 0.9955 at 5.85 pairs per S1** on train.
- **Most important features:** the e5-large reverse rank 52%, e5-large cosine rank within the record 19.5%, first house
  number equal 8.8%, retriever rank 4.3%.

- **Candidate pairs generated:**
  - Test pool ≈ 110 per S1.
  - `candidate_pairs.tsv` = the shortlist: **11,171,750 pairs (6.45 per S1)**. Both models and the blend score exactly
    these pairs, and every submitted match lies in it.
- **How we ensured true matches were not lost:**
  - The routes are complementary. Embeddings handle paraphrase and script; sketches handle typos and joined words;
    keys handle exact rare evidence; the reverse direction handles S1s with many candidates.
  - Recall was measured per route, per country and for the union at every change. Each route's unique contribution is
    logged (`block_stats.json`).
  - The shortlist is cut at a fixed share of recall, not a fixed size.
  - Losses on train: 0.36% of true pairs are never found and 0.09% are cut by the shortlist.
- **Scalability.**
  - Every route is sub-quadratic within a country. Nearest-neighbour search runs on the GPU in chunks; the code uses
    FAISS IVF (GPU k-means, nlist = 4√n) above a size threshold and logs an ANN-vs-exact recall check.
  - At this data size exact search fits on one GPU and IVF lost some recall, so the submitted run used exact search
    (`BER_ANN_MIN=10^9`). Lowering that threshold switches to IVF.
  - Key routes are capped hash joins, and nothing compares all pairs.

---

## 4. Matching Model

This section covers steps 0 and 6–11, then Model B and the blend.

**Step 0: Auxiliary matcher** (`src/v2/`). A simpler, independent matcher whose score becomes one stage-3 feature.
- **Candidates:** e5-small dense search (25 per S1 and 5 S1 per record) plus IDF shared keys.
- **Model:** 44 pair features and LightGBM, 5-fold out-of-fold, isotonic calibration.
- **Result:** out-of-fold macro F0.5 0.9781.

It gives stage 3 a second, differently built opinion: 17.5% of stage 3's gain.

**Step 6: Pair features** (`ber/features.py`). 101 features per shortlisted pair. Country is not a feature.
- **Name features:**
  - Levenshtein ratio, token-sort, token-set, partial ratio and Jaro-Winkler (RapidFuzz), on the full, core, compact
    and OCR-repaired names;
  - best alias match; initials/acronym match;
  - IDF-weighted word overlap; information content of the shared words; how many S1s and records share the core name;
  - metaphone overlap, consonant-skeleton equality, sorted-core equality, native-script flag;
  - invented (garbled) words and whether a one-edit typo explains them.
- **Address features:**
  - ratio, token-set, token-sort and partial ratio on the full address and the core address (landmark words removed);
  - IDF overlap; abbreviation coverage (`av` ~ `avenue`); address 3-gram cosine;
  - **typed numbers:** main-number relation (equal / truncated / one-digit edit / near / different), Jaccard,
    contained-in, extra numbers on either side, number-plus-street frequency;
  - co-location counts (how many businesses share the address).
- **Other:**
  - retrieval evidence: cosine for each embedder and each sketch, ranks in both directions, retriever ranks, number of
    routes that found the pair, exact-key flags;
  - **listwise competition:** p1's rank and gap to the best and second-best candidate of the same S1 and of the same
    record, and mutual best;
  - has-address flags; source (S2 or S3).

**Step 7: Stage 2** (`ber/train.py`).
- **Model:** XGBoost on the GPU (lr 0.05, ≤ 5,000 trees, early stopping 100, depth 9, min child weight 5, subsample
  0.8, colsample 0.8), 5-fold GroupKFold by S1, isotonic calibration → **p2**.
- **"Has any match" model:** a per-S1 model on aggregates of its candidates' scores.
- **Result:** out-of-fold macro F0.5 0.9888.

**Step 8: Cross-encoder** (`ber/crossenc.py`).
- **Model:** multilingual-e5-small with a 1-logit head reads both records serialized together (`name: … | address: …`,
  max 96 tokens).
- **Training:** fine-tuned on the uncertain band 0.02 ≤ p2 ≤ 0.98, plus a 5% sample of confident pairs (lr 3e-5,
  batch 256, 1 epoch), 2-fold cross-fitted by S1.
- **Output:** **p_ce**, used only inside the band.

**Step 9: Graph consistency** (`ber/graph.py`). A true match should look like the other records its S1 already holds
strongly, and a record wanted by several S1s is suspect. From p2 (out-of-fold):
- the record's best score with any other S1;
- the S1's best other record;
- cosine and name similarity between the record and the S1's strong siblings (p2 ≥ 0.5), including siblings from the
  other source.

**Step 10: Stage 3** (`ber/stage3.py`).
- **Inputs:** the step-6 features + p2 + p_ce + graph features + the auxiliary score. Counts that depend on how many
  candidates exist are dropped, because that density differs on test.
- **Model:** XGBoost with the same settings and the same folds (≈ 800 trees per fold, 886 in the final refit on all
  train), then isotonic calibration.
- **Result:** out-of-fold macro F0.5 **0.9913** (US 0.9908, India 0.9921).
- **Gain share:** p2 69%, auxiliary score 17.5%, p1 8.4%, p_ce 2.7%, graph ≈ 0.7%.

**Step 11: Decision.**
1. Each S2/S3 record is assigned to at most one S1, its highest-probability candidate.
2. Pairs with p ≥ t are kept.
3. On train the best cut is t = 0.70, but the leaderboard preferred 0.95:

   | Model A decision | public LB |
   |---|---|
   | best rule on train | 0.983 |
   | **p ≥ 0.95** | **0.984312** |
   | p ≥ 0.99 | 0.980 |

4. Leaderboard-difference analysis (Appendix B) shows the 0.95–0.99 band is ≈ 85% precise on test against ≈ 95% on
   train, which is the orphan shift. **Model A = p ≥ 0.95.**

**Model B: what we added** (`run_v5.py`: the same code with three switches on)
1. **Training at the test's orphan density** (`ber/s1drop.py`).
   - After the candidate search, 19% of the train S1s are removed (by a hash of the id). Their S2/S3 records stay as
     orphans, and reverse-search ranks are recomputed among the remaining S1s.
   - Stage 1 through stage 3 then learn in the test situation: about twice as many unmatched look-alikes.
2. **Abbreviations mined per country from the records** (`ber/mining.py`, label-free).
   - Inside small groups of records that share a house number and a rare word, a word missing from one record that
     abbreviates a word in the other is counted: same first letter, letters in order.
   - Frequent, consistent mappings are kept (≥ 30 occurrences, ≥ 50% consistency) and applied before the lexicon.
   - This finds e.g. `r → rue`, `bd → boulevard`, `ets → etablissements`: France's vocabulary, which has no training
     labels.
3. **3,000 stage-1 trees** (Model A stopped at its 1,500-tree cap).

Model B reuses Model A's candidate pool, so both score the same shortlist. Its train out-of-fold F0.5 is 0.9910 on
the harder training set. Alone it scores 0.98376 on the leaderboard: it is more conservative than Model A.

**Final blend** (`src/run_final.py`).
1. p = 0.7 · p_A + 0.3 · p_B.
2. Each S2/S3 record goes to its highest-p S1.
3. The cut keeps exactly as many pairs as Model A at 0.95, which gives t = 0.9157 and 5,836,213 pairs.
4. **How the weights were chosen:**
   - On the train S1s that Model B kept (1,782,911 S1s, test-like orphan density), we compared 14 weightings of Model
     A, Model B and an earlier pipeline version, plus majority and unanimous votes, each held at Model A's pair count.
   - 0.7/0.3 was best: 0.99031 against Model A's 0.99021.
5. On the leaderboard the blend scored **0.984403**, against Model A's 0.984312: +0.00009, as the train
   check predicted (+0.0001).

**Model type:** XGBoost (Apache-2.0) for stages 1–3, LightGBM (MIT) for the auxiliary matcher, and
multilingual-e5-small / -large (MIT) for embeddings, the retriever and the cross-encoder.

**Threshold selection method:** out-of-fold tuning with the exact metric, corrected by leaderboard-difference
precision estimates. The final cut is held at the validated pair count (steps 11 and the final blend).

---

## 5. Results & Error Analysis

- **F0.5 Score (macro):** Model A train out-of-fold **0.9913** (US 0.9908, India 0.9921); submitted blend public LB
  **0.984403**.

**Where Model A loses train true pairs** (out-of-fold, 7.60M true pairs):

| where | pairs | share |
|---|---|---|
| not found by the candidate search | 27,488 | 0.36% |
| cut by the shortlist | 6,773 | 0.09% |
| rejected by the model | 151,656 | 2.0% |
| **found** | 7,452,448 | **97.6%** |

Only 9,480 false pairs were predicted on train.

- **Common false positives (wrong merges):**
  - **Orphan look-alikes (63% of train false positives):** a clean record of the same chain, or of a neighbouring
    business at the same or a similar address, whose own S1 is absent. The name typically differs in its distinguishing
    word (`group`, `holdings`, `ventures`, `industries`), not in a generic word.
  - Same-name businesses whose addresses differ only in the house number or unit.
  - **France:** generic names (`boulangerie`, `pharmacie`) shared by many S1s.
- **Common false negatives (missed matches):**
  - **Records without an address:** 39% of their true pairs are missed. The only evidence is the name, often shared by
    many S1s.
  - Records without a house number (12.9% missed), and records whose house number changed.
  - Trade names that differ from the legal name.
  - Heavy transliteration differences.
  - **France:** French legal and generic words swapped in the name (`groupe`, `cie`, `sarl`) and unseen abbreviations
    with typos (`126 R PSCAL` for `126 Rue Pascal`) push true pairs below the cut.
  - On test we estimate that about 75% of the remaining loss is misses and 25% false matches.

---

## 6. Conclusion

High recall at blocking combined with a precision-first stacked matcher works well for this task:
- 99.6% of true pairs reach the matcher at 6.45 candidates per S1;
- the vocabulary is learned from the data, and every stacked score is leak-free.

The main lesson was the train→test shift in unmatched records. We handled it in three ways:
- we measured pair precision on test through leaderboard differences and chose a stricter cut;
- we trained a second model at the test's orphan density;
- we blended the two at a fixed pair count.

None of these uses test labels.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains the pipeline that generates both output files from the provided data.
The included `output/` files are the leaderboard-tested 0.7/0.3 blend. All pipeline stages use `anyascii` for
romanization. `README.md` has the commands. Supplementary guides in
`code/business_entity_resolution/documentation/` show the V4 flow and V4/V5 details.

**Names in the code:**

| in this document | in the code |
|---|---|
| auxiliary matcher | `src/v2/` (`src/v2/run_pipeline.py`) |
| Model A (main pipeline) | `src/run_v4.py` → `work_v4/` |
| Model B | `src/run_v5.py` → `work_v5a/` |
| final blend | `src/run_final.py` → `submission/` |

- `run.sh` runs all four in order, with Model B reusing Model A's candidate pool. `src/run_final.py` reproduces the
  uploaded `matching_results.tsv` byte for byte from the two models' test scores. With `FINAL_CHECK=1` it also prints
  the train check behind the weights.
- `src/ber/` holds the stages (prepare, embed, block, retriever, s1drop, stage1, features, train, crossenc, graph,
  stage3, mining). Every stage is cached and resumable.
- `requirements.txt` pins the versions of the submitted runs (Python 3.12, CUDA).
- **Hardware:** one L40S 48 GB GPU, 32–64 vCPU, up to 512 GB RAM.
- **Runtime:**
  - auxiliary matcher: 1–2 h;
  - Model A: 6–7 h. Measured stages: stage 1 48 min, features 10 min, stage 2 20 min, cross-encoder 25 min, graph
    1 min, stage 3 11 min;
  - Model B: about 3 h;
  - blend: 2 min.
- **Models:** `intfloat/multilingual-e5-small` (118M) and `intfloat/multilingual-e5-large` (560M), both MIT, plus our
  fine-tuned retriever and cross-encoder derived from e5-small. All are far below 8B parameters.
- **Data:** only the provided files.
  - Supervised learning uses the train ground truth only.
  - Unsupervised statistics (IDF, generic words, uniqueness counts, Model B's abbreviation mining) use the provided
    records without labels.
  - No external data, APIs, geocoding or hand-written lists; country is never a feature.
- **Romanization:** all stages, including the auxiliary matcher, use `anyascii` (ISC) for remaining non-Latin tokens.

### B. Additional Results

**Precision measured through leaderboard differences.** When two submissions differ only on a set of pairs, the score
difference × 1,732,544 S1 gives the net per-S1 F0.5 change, which implies the precision of those pairs:

| comparison (same model, two cuts) | extra pairs | implied precision on test |
|---|---|---|
| Model A best train rule vs p ≥ 0.99 | 202,656 | ≈ 0.75 |
| Model A p ≥ 0.95 vs p ≥ 0.99 (band 0.95–0.99) | ≈ 125k | ≈ 0.85 |
| earlier version, p ≥ 0.30 vs p ≥ 0.95 | 433,550 | ≈ 0.45 |

Adding a pair pays off only above ≈ 0.70 precision; filling an empty S1 pays off above 0.50.

**Blend and vote variants, on train out-of-fold at Model A's pair count** (1,782,911 S1s with test-like orphan
density; Model A alone = 0.99021). "Earlier version" is the pipeline without the retriever, cross-encoder and stage 3.

| variant | train F0.5 | vs Model A |
|---|---|---|
| **0.7 A + 0.3 B (submitted)** | **0.99031** | **+0.00010** |
| 0.8 A + 0.2 B | 0.99030 | +0.00009 |
| 0.6 A + 0.4 B | 0.99030 | +0.00009 |
| majority vote of A, B, earlier version | 0.99026 | +0.00005 |
| equal average of A, B, earlier version | 0.98999 | −0.00021 |
| unanimous vote | 0.98409–0.98980 | −0.0004 to −0.006 |

**How the method grew** (each step kept only if it improved out-of-fold F0.5):

| stage of development | train OOF | public LB |
|---|---|---|
| auxiliary matcher alone (step 0) | 0.9781 | 0.9700 |
| + two embedders, exact search, stage-1 shortlist, stage-2 XGBoost | 0.9881 | 0.97493 |
| + retriever, cross-encoder, graph features, stage 3, cut at 0.95 (Model A) | 0.9913 | 0.984312 |
| + Model B and the 0.7 / 0.3 blend (submitted) | — | **0.984403** |

**Tried and not kept:**

| attempt | result |
|---|---|
| rule-based post-processing (drop pairs with conflicting house numbers, no-address pairs, street conflicts; add exact-key "twins"; a classical Fellegi–Sunter model) | no clear gain (0.983–0.984 on the LB) |
| a higher shortlist target | the shortlist loses only 0.09% of true pairs, so little to gain |
| a third embedder (multilingual-e5-large-instruct) | written, not run in time |
