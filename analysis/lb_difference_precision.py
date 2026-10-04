"""Where do submissions differ, and what does the LB difference imply about the differing pairs?
Nested pairs (Y = stricter subset of X per S1): assume Y's pairs are correct and each extra pair of X is correct with
probability q; expected sum of per-S1 F0.5 differences E[F_X - F_Y](q) is matched to (LB_X - LB_Y) * N_S1 -> q."""
import os
import csv
import sys
from collections import Counter, defaultdict
from math import comb

R = os.environ.get("BER_ROOT", "./")
F = {
    "v4_best": (R + "v4/output/matching_results.tsv", 0.983),
    "v4_t099": (R + "v4/output_probes/probe_v4_t0990/matching_results.tsv", 0.980),
    "v3_t095": (R + "v3/output_probes/probe_t095/matching_results.tsv", 0.978),
    "v3_rule": (R + "v3/output/matching_results.tsv", 0.97493),
    "v3_t030": (R + "v3/output_probes/probe_t030/matching_results.tsv", 0.958),
}
csv.field_size_limit(10**9)
country = {}
with open(R + "dataset/test/test_source1.tsv", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    h = next(r)
    ic, ii = h.index("country"), h.index("entity_id")
    for row in r:
        country[row[ii]] = row[ic]


def rows(path):
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s, _, m = line.rstrip("\n").partition("\t")
            yield s, frozenset(m.split(",")) if m else frozenset()


def f05(p, r):
    return 0.0 if p == 0 or r == 0 else 1.25 * p * r / (0.25 * p + r)


def exp_df(k, e, q):
    """E[F_X - F_Y] for one S1: Y has k (correct) pairs, X has k + e; each extra correct w.p. q."""
    tot = 0.0
    for c in range(e + 1):
        w = comb(e, c) * q ** c * (1 - q) ** (e - c)
        fx = f05((k + c) / (k + e), 1.0) if k + c > 0 else 0.0
        if k > 0:
            fy = f05(1.0, k / (k + c))
        else:
            fy = 1.0 if c == 0 else 0.0
        tot += w * (fx - fy)
    return tot


def compare(xn, yn):
    X, Y = F[xn], F[yn]
    st = defaultdict(Counter)
    nested_hist = Counter()      # (k, e) for S1s where Y ⊂ X
    n = 0
    for (s, a), (s2, b) in zip(rows(X[0]), rows(Y[0])):
        assert s == s2
        n += 1
        c = country.get(s, "?")
        st[c]["s1"] += 1
        if a == b:
            continue
        st[c]["differ"] += 1
        st[c]["only_x"] += len(a - b)
        st[c]["only_y"] += len(b - a)
        if a and not b:
            st[c]["x_only_nonempty"] += 1
        if b and not a:
            st[c]["y_only_nonempty"] += 1
        if b <= a:
            nested_hist[(len(b), len(a - b))] += 1
        else:
            st[c]["not_nested"] += 1
    dlb = (X[1] - Y[1]) * n
    print(f"\n### {xn} (LB {X[1]}) vs {yn} (LB {Y[1]}): N={n:,}  LB diff x N = {dlb:+,.0f} S1-units")
    print("| country | S1 | differ | pairs only in " + xn + " | pairs only in " + yn +
          " | S1 empty only in " + yn + " | S1 empty only in " + xn + " | not nested |")
    print("|---|---|---|---|---|---|---|---|")
    for c, d in sorted(st.items()):
        print(f"| {c} | {d['s1']:,} | {d['differ']:,} | {d['only_x']:,} | {d['only_y']:,} | {d['x_only_nonempty']:,} | "
              f"{d['y_only_nonempty']:,} | {d['not_nested']:,} |")
    tot_nn = sum(d["not_nested"] for d in st.values())
    if nested_hist and tot_nn < 0.05 * sum(d["differ"] for d in st.values()):
        grid = [i / 100 for i in range(101)]
        vals = [sum(cnt * exp_df(k, e, q) for (k, e), cnt in nested_hist.items()) for q in grid]
        def solve(target):
            best = min(range(101), key=lambda i: abs(vals[i] - target))
            return grid[best]
        lo, hi = (X[1] - Y[1] - 0.001) * n, (X[1] - Y[1] + 0.001) * n
        print(f"all extra pairs correct -> diff {vals[-1]:+,.0f}; all wrong -> {vals[0]:+,.0f}")
        print(f"implied precision of the {sum(e * c for (k, e), c in nested_hist.items()):,} extra pairs of {xn}: "
              f"q = {solve(dlb):.2f} (LB rounding ±0.001 -> {solve(lo):.2f}..{solve(hi):.2f})")
    sys.stdout.flush()


if __name__ == "__main__":
    for x, y in [("v4_best", "v4_t099"), ("v3_rule", "v3_t095"), ("v3_t030", "v3_t095"),
                 ("v4_best", "v3_t095"), ("v4_best", "v3_rule")]:
        compare(x, y)
