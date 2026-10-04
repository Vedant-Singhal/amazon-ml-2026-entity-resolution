# v5 = Model B, and the final blend (LB 0.98376 alone, 0.984403 blended)

Code: [`pipeline/src/run_v5.py`](../../pipeline/src/run_v5.py) is `run_v4.py` with three switches on:

| switch | module | what it does |
|---|---|---|
| `BER_TRAIN_S1_DROP=0.19` | `ber/s1drop.py` | after the search, drop 19% of train S1 (2,206,821 → 1,786,233) and keep their records as orphans; pool 243.8M → 197.3M pairs; reverse ranks recomputed. Train then has test's orphan density |
| `BER_MINE_ABBR=1` | `ber/mining.py` | label-free abbreviation mining per country (88 French address maps such as `r → rue`, `bd → boulevard`) |
| `BER_ROUNDS1=3000` | `ber/config.py` | stage-1 tree cap 1,500 → 3,000 |

**Variants launched**: a (all three), b (no dropout), c (no mining), d (30% dropout). Only **v5a** finished in time to
submit: stage-3 OOF 0.9910 (US 0.9905, India 0.9917) on its harder training set; LB 0.98376 at t 0.95, ≈ 0.983 at
looser cuts. v5c reached stage-2 OOF 0.9882 (comparison only).

**Final blend** ([`pipeline/src/run_final.py`](../../pipeline/src/run_final.py)): `0.7 · p_v4 + 0.3 · p_v5a`, each
record to its best S1, cut at v4's pair count (t = 0.9157, 5,836,213 pairs). **LB 0.984403** (+0.00009 over v4).

**Reports** (`reports/`)
- `blend-weights-on-train.md`: 14 weightings of v4 / v5a / v3, judged on the train S1 that v5a kept, at v4's pair
  count, at their own best threshold and at 0.95.
- `voting-variants-on-train.md`: majority, unanimous and equal-average variants.

The scripts that produced them are [`analysis/run_ens.py`](../../analysis/run_ens.py) and
[`analysis/run_vote.py`](../../analysis/run_vote.py).
