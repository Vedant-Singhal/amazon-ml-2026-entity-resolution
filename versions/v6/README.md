# v6: a third embedder view (written, not run)

Planned after v5, but the deadline arrived first.

- **Why:** `multilingual-e5-large-instruct` had the best recall of all ablated models on records **without an
  address** (recall@10 India 0.358, US 0.759 vs 0.340 / 0.736 for e5-small), the largest miss group in train.
- **`code/ber/addview.py`:** adds a new dense view to a run whose candidate pool already exists. It embeds only with
  the new model, searches both directions within country, adds the pairs only this view finds (other routes set to
  "not found", every cosine computed), and adds the view's ranks and cosine to every existing pair. Copy it to
  `pipeline/src/ber/` and run it between the search and stage 1; stage 1 onward must then be rebuilt.
- **`code/run_complement.py`:** decides which third embedder complements e5-small + e5-large best. For each candidate
  model it counts the true pairs found in its top 25 that **both** current views miss, overall and on no-address,
  trade-name and native-script subsets.
