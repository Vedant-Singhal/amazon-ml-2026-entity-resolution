# 6. Literature: what we read and what we used

Papers reviewed on 25–26 September 2026, ranked by how much they shaped the solution. "Used" means the idea is in the
final pipeline; no third-party code or model other than the e5 embedders is used.

## Shaped the final pipeline

| paper | venue | idea | where it is in our pipeline |
|---|---|---|---|
| Sparkly: a TF/IDF blocker for entity matching (Paulsen et al.) | PVLDB 2023 | lexical top-k retrieval is a strong, scalable candidate generator | IDF-weighted rare-key route, char-3-gram routes |
| SC-Block: supervised contrastive blocking (Brinkmann et al.) | ESWC 2024 | fine-tune the dense retriever on cluster labels | the fine-tuned e5-small retriever, the biggest recall gain |
| Lost in Transliteration (Chari et al.) | SIGIR 2025 | retrievers fail between native script and romanized text; train on both | romanized copies of native-script records as extra positives |
| UniBlocker (Wang et al.) | arXiv 2024 | dense and sparse candidates are complementary: take the union | union of 13 routes + retriever |
| A critical re-evaluation of EM benchmarks (Papadakis et al.) | ICDE 2024 | most matching is "easy"; GBDT on good similarity features is a strong core | XGBoost stages 1–3 on 101 features |
| Ditto (Li et al.) | PVLDB 2021 | serialize both records and fine-tune a cross-encoder | cross-encoder on the uncertain band |
| Beyond Scale and Generation (Zhang et al.) | arXiv 2026 | cross-encoders beat bi-encoders for matching | why the judge is a cross-encoder |
| Thresholding classifiers to maximize F1 (Lipton et al.) | ECML-PKDD 2014 | per-entity expected-F decision from calibrated probabilities, incl. the empty set | expected-F0.5 rule and calibration (tested; the threshold rule won on test) |
| MSCD-HAC: multi-source clean–dirty clustering (Lerm et al.) | KEOD 2021 | at most one entity from a clean source per cluster | one owner per S2/S3 record |
| Match, Compare, or Select? (ComEM, Wang et al.) | COLING 2025 | choosing among candidates beats independent pairwise decisions | listwise rank/gap features, one-owner assignment |
| TransClean (de Meer Pardo et al.) | IEEE Access 2025 | transitive consistency removes false positives in multi-source matching | graph-consistency features (sibling similarity, competing owners) |
| Auto-FuzzyJoin (Li et al.) | SIGMOD 2021 | learn a fuzzy join's transformations from the data itself | label-free abbreviation mining (Model B) |
| Wolpert, stacked generalization | 1992 | train the stack on out-of-fold predictions | leak-free three-stage stacking with shared folds |
| Charikar et al., count sketch | 2002 | random signed hashing preserves inner products | 256-d sketches of char-3-gram TF-IDF vectors |

## Read and considered, not used

| paper | why not |
|---|---|
| Sudowoodo (ICDE 2023), DADER (SIGMOD 2022): self-supervised / domain adaptation for an unseen domain | France handled by abbreviation mining and a stricter cut; no time for adversarial adaptation |
| Entity Matching using LLMs (EDBT 2025), TailorMatch (2025): generative matchers | strongest under distribution shift, but too costly at ~11M candidate pairs; only viable on the uncertain band |
| Unicorn (SIGMOD 2023), MultiEM (ICDE 2024) | ideas overlap with ours; code has no licence |
| BGE-M3 (Findings ACL 2024) | tested in the embedder ablation: below e5-small and e5-large on these strings |
| AnyMatch (AAAI 2025 workshop) | hard-pair selection idea only |
| OpenSanctions Pairs, IndicXlit, libpostal | external data or models trained on external data, which our rules exclude |

## Models checked for the licence rule (MIT / Apache-2.0, ≤ 8B)

Allowed and tested: `multilingual-e5-small / base / large / large-instruct` (MIT), `bge-m3` (MIT),
`arctic-embed-l-v2.0` (Apache-2.0), `Qwen3-Embedding-0.6B` (Apache-2.0), `granite-embedding-*-multilingual-r2`
(Apache-2.0), `bhasha-embed-v0` (Apache-2.0), `KaLM-mini-instruct-v1.5` (MIT), `IndicBERTv2-SS` (MIT).
Excluded by licence: jina-embeddings-v3 (CC-BY-NC), EmbeddingGemma / Gemma-based models, Llama-based models,
L3Cube IndicSBERT (CC-BY-4.0). Excluded by size: Qwen3-8B (8.19B parameters).
