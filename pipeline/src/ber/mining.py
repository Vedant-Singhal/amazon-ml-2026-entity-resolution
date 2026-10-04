"""v5: unsupervised abbreviation mining from the records themselves (no labels, no model predictions).

Idea (as in Auto-FuzzyJoin, Li et al. SIGMOD'21: learn the transformations of a fuzzy join from the data): two records
of one country that share a house number AND a rare word (a street name or a name word used by few records) are very
likely the same place. Inside such small groups, a word present in one record but not the other, where the shorter
word is an abbreviation of the longer one (same first letter, its letters appear in order in the longer word), is a
candidate mapping short -> long: 'r' -> 'rue', 'bd' -> 'boulevard', 'ets' -> 'etablissements'. A mapping is kept when
it is frequent and consistent. This finds the local vocabulary of a country that has no training labels (France).
"""
from collections import Counter, defaultdict
from itertools import combinations


def is_abbrev(x: str, y: str) -> bool:
    """x is an abbreviation of y: shorter, same first letter, letters of x appear in order in y."""
    if not (x and y and len(x) < len(y) and x[0] == y[0] and x.isalpha() and y.isalpha()):
        return False
    if len(x) > 4 and len(x) > 0.6 * len(y):        # long near-copies are typos, not abbreviations
        return False
    it = iter(y)
    return all(ch in it for ch in x)


def mine(records, min_count: int = 30, min_ratio: float = 0.5, rare_max: int = 300, max_group: int = 30,
         max_pairs: int = 5_000_000, single_letters: bool = True, single_min_count: int = 200,
         single_min_ratio: float = 0.8) -> dict:
    """records: list of (country, anchor_number, tokens). Returns {short: long}."""
    df = defaultdict(Counter)
    for c, _, t in records:
        df[c].update(set(t))
    groups = defaultdict(list)
    for i, (c, num, t) in enumerate(records):
        if not num:
            continue
        for w in set(t):
            if len(w) >= 4 and w.isalpha() and 2 <= df[c][w] <= rare_max:
                groups[(c, num, w)].append(i)
    seen = set()
    pair_cnt, diff_cnt = Counter(), Counter()
    for ids in groups.values():
        if not 2 <= len(ids) <= max_group:
            continue
        for a, b in combinations(ids, 2):
            if (a, b) in seen:
                continue
            seen.add((a, b))
            sa, sb = set(records[a][2]), set(records[b][2])
            da, db = sa - sb, sb - sa
            if not da or not db or len(da) > 4 or len(db) > 4:
                continue
            for u in da | db:
                diff_cnt[u] += 1
            for u in da:
                for v in db:
                    if is_abbrev(u, v):
                        pair_cnt[(u, v)] += 1
                    elif is_abbrev(v, u):
                        pair_cnt[(v, u)] += 1
            if len(seen) >= max_pairs:
                break
        if len(seen) >= max_pairs:
            break
    best = {}
    for (x, y), n in pair_cnt.items():
        need_n, need_r = (single_min_count, single_min_ratio) if len(x) == 1 else (min_count, min_ratio)
        if len(x) == 1 and not single_letters:
            continue
        if n >= need_n and n / max(1, diff_cnt[x]) >= need_r and (x not in best or n > best[x][1]):
            best[x] = (y, n)
    return {x: y for x, (y, n) in sorted(best.items(), key=lambda kv: -kv[1][1])}


def _num(comps) -> str:
    return next((t.lstrip("0") or "0" for comp in comps for t in comp if t.isdigit()), "")


def mine_by_country(names, addrs, countries, cap: int = 2_000_000, log=print) -> dict:
    """names: [(main tokens, alias token lists)], addrs: [[part tokens]] (folded, before any lexicon), countries.
    -> {country: {"addr": {short: long}, "name": {short: long}}}; at most `cap` records per country (evenly spaced)."""
    idx = defaultdict(list)
    for i, c in enumerate(countries):
        idx[c].append(i)
    out = {}
    for c, ids in sorted(idx.items()):
        step = max(1, len(ids) // cap)
        ids = ids[::step]
        a_recs = [(c, _num(addrs[i]), [t for comp in addrs[i] for t in comp]) for i in ids]
        n_recs = [(c, a_recs[k][1], names[i][0]) for k, i in enumerate(ids)]
        ma = mine(a_recs)
        mn = mine(n_recs, single_letters=False)
        out[c] = {"addr": ma, "name": mn}
        log(f"mined[{c}] from {len(ids):,} records: {len(ma)} address maps ("
            + ", ".join(f"{k}->{v}" for k, v in list(ma.items())[:20]) + f"); {len(mn)} name maps ("
            + ", ".join(f"{k}->{v}" for k, v in list(mn.items())[:12]) + ")")
    return out


def apply(rows, mined: dict) -> list:
    """rows: [((main, alts), addr parts, country)] -> [((main, alts), addr parts)] with that country's mined maps."""
    out = []
    for (main, alts), comps, c in rows:
        m = mined.get(c)
        if not m:
            out.append(((main, alts), comps))
            continue
        mn, ma = m["name"], m["addr"]
        out.append(((
            [mn.get(t, t) for t in main], [[mn.get(t, t) for t in a] for a in alts]),
            [[ma.get(t, t) for t in comp] for comp in comps]))
    return out
