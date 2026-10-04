"""Stage `prepare`: normalize every record of a split.

Output work/<split>/{s1,oth}.parquet with, per record:
  name_n      romanized name tokens (learned lexicon + anyascii), repeated words collapsed
  core        name without generic words (per-country frequent words + learned noise words)
  compact / ocr / acr / stop / aliases      as in v2
  sorted_core core words sorted (word-order-free key)
  skel        consonant skeleton of the core (romanization-variant-free key, native script)
  meta        metaphone codes of the core words
  addr_n / addr_core / alpha                 as in v2 (landmark cues fixed: multi-word parts only)
  nums        all numbers;  main_nums  numbers not after a learned marker ('pmb', 'box', ...)
  first_num   first main number;  akey  address words + first number;  numset  >= 2 main numbers
  numst       first number + first street word (for number-street frequency)
  u_s1_* / u_oth_*   how many S1 / S2-S3 records of the country share the core, name, address,
              number-street key: name uniqueness (E1/E6) and co-location (E8) signals
Frequent-word lists, document frequencies and counts come from the split's own records.
"""
import gc
import json
import os
from multiprocessing import Pool

import polars as pl

from . import lexicon, mining
from .config import P, log, work_dir
from .data import load_gt_pairs, load_pseudo_pairs, load_split
from .text import (NATIVE, PLACEHOLDERS, acronym, apply_phrases, collapse_repeats, fold_tokens,
                   merge_letters, metaphone_codes, norm_number, number_parts, ocr_fix, romanize,
                   skeleton, split_aliases, strip_domain, strip_urls)

_LEX_N, _LEX_A, _STOP, _STOP_DF, _CUES, _MARKERS = {}, {}, {}, {}, frozenset(), frozenset()
_MINED = {}   # v5: {country: {"name": {short: long}, "addr": {short: long}}} mined from the records (BER_MINE_ABBR)
U_KEYS = [("core", "core"), ("name", "name_n"), ("addr", "addr_core"), ("numst", "numst")]


def _pmap(fn, items, chunk=100_000):
    chunks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    # Forked workers inherit the parent's heap; gc.freeze() keeps their GC from copying it (OOM).
    gc.collect()
    gc.freeze()
    try:
        with Pool(P.pool_workers) as pool:
            parts = pool.map(fn, chunks)
    finally:
        gc.unfreeze()
    return [x for part in parts for x in part]


def _name_tokens(s):
    return collapse_repeats(merge_letters(fold_tokens(strip_domain(s))))


def _fold_names(xs):
    out = []
    for x in xs:
        x = strip_urls(x)
        main = _name_tokens(x)
        alts = [a for a in (_name_tokens(s) for s in split_aliases(x)) if a and a != main]
        out.append((main, alts))
    return out


def _fold_addrs(xs):
    """Folded tokens per comma-separated part; repeated words and repeated parts collapsed."""
    out = []
    for x in xs:
        parts = []
        for c in x.split(","):
            p = collapse_repeats(fold_tokens(c))
            if p and (not parts or parts[-1] != p):
                parts.append(p)
        out.append(parts)
    return out


def _apply_mined(rows):
    return mining.apply(rows, _MINED)


def _rn(toks):
    return collapse_repeats([r for t in apply_phrases(toks, _LEX_N) for r in romanize(t, _LEX_N)])


def _roman_names(items):
    return [(_rn(main), [_rn(a) for a in alts]) for main, alts in items]


def _clean_part(comp):
    return apply_phrases([t for t in comp if t not in PLACEHOLDERS], _LEX_A)


def _lex_parts(comp_lists):
    out = []
    for comps in comp_lists:
        parts = [[w for t in _clean_part(comp) for w in _LEX_A.get(t, t).split()] for comp in comps]
        out.append([p for p in parts if p])
    return out


def _roman_addrs(comp_lists):
    """-> (all tokens, core tokens without parts that start with a learned landmark cue)."""
    out = []
    for comps in comp_lists:
        full, core = [], []
        for comp in comps:
            toks = _clean_part(comp)
            r = [w for t in toks for w in ([norm_number(t)] if t.isdigit() else romanize(t, _LEX_A))]
            full.extend(r)
            if not r or r[0] not in _CUES:
                core.extend(r)
        out.append((collapse_repeats(full), collapse_repeats(core)))
    return out


def _core(tokens, country):
    """Tokens minus generic words, never fewer than 2 words (rarest removed words come back)."""
    stop = _STOP.get(country, frozenset())
    core = [t for t in tokens if t not in stop]
    if len(core) >= 2:
        return core
    removed = sorted({t for t in tokens if t in stop}, key=lambda t: _STOP_DF[country].get(t, 0))
    keep = set(removed[:2 - len(core)])
    return [t for t in tokens if t not in stop or t in keep]


def _numbers(addr):
    """-> (all numbers, main numbers): numbers right after a learned marker are secondary."""
    nums, main = [], []
    for i, t in enumerate(addr):
        ds = number_parts(t)
        if not ds:
            continue
        nums.extend(ds)
        if not (i > 0 and addr[i - 1] in _MARKERS):
            main.extend(ds)
    return nums, main


def _alpha(toks):
    return [t for t in toks if not any(c.isdigit() for c in t)]


def _derive(rows):
    out = []
    for (name, alts), (addr, addr_core), country in rows:
        stop = _STOP.get(country, frozenset())
        core = _core(name, country)
        nums, main = _numbers(addr)
        main_u = list(dict.fromkeys(main))
        first = main_u[0] if main_u else ""
        alpha_core = sorted(set(_alpha(addr_core)))
        aliases = [" ".join(a) for a in alts if any(t not in stop for t in a)]
        out.append((
            " ".join(name), " ".join(core), "".join(core), "".join(ocr_fix(t) for t in core),
            acronym(core), " ".join(t for t in name if t in stop and t not in core), aliases,
            " ".join(sorted(core)), skeleton(core), metaphone_codes(core),
            " ".join(addr), " ".join(addr_core), _alpha(addr), sorted(set(nums)), sorted(set(main_u)), first,
            (" ".join(alpha_core) + "#" + first) if (alpha_core and first) else "",
            " ".join(sorted(main_u)) if len(main_u) >= 2 else "",
            (first + " " + alpha_core[0]) if (first and alpha_core) else "",
        ))
    return out


COLS = [("name_n", 0), ("core", 0), ("compact", 0), ("ocr", 0), ("acr", 0), ("stop", 0), ("aliases", 1),
        ("sorted_core", 0), ("skel", 0), ("meta", 1), ("addr_n", 0), ("addr_core", 0), ("alpha", 1),
        ("nums", 1), ("main_nums", 1), ("first_num", 0), ("akey", 0), ("numset", 0), ("numst", 0)]


def _tok_df(ids, toks):
    return pl.DataFrame({"iid": ids, "toks": toks}, schema={"iid": pl.Int64, "toks": pl.List(pl.String)})


def _pseudo_test(train_countries):
    """v5: test records of countries absent from train (France) + a previous run's confident pairs for them
    (BER_PSEUDO_PAIRS), folded like train, so their abbreviations / region names / legal forms get learned."""
    pp = load_pseudo_pairs(os.environ["BER_PSEUDO_PAIRS"])
    s1, oth = load_split("test")
    s1 = s1.filter(~pl.col("country").is_in(sorted(train_countries)))
    oth = oth.filter(~pl.col("country").is_in(sorted(train_countries)))
    pp = pp.filter(pl.col("a").is_in(s1["iid"].implode()) & pl.col("b").is_in(oth["iid"].implode()))
    log(f"pseudo pairs: {len(pp):,} for countries {sorted(set(s1['country'].to_list()))} "
        f"({len(s1):,} S1, {len(oth):,} S2/S3)")
    if len(pp) < 1000:
        return None
    recs = pl.concat([s1.with_columns(src=pl.lit("S1")), oth])
    return (pp, recs["iid"].to_list(), len(s1), _pmap(_fold_names, recs["name"].to_list()),
            _pmap(_fold_addrs, recs["addr"].to_list()))


def run(split: str) -> None:
    global _LEX_N, _LEX_A, _STOP, _STOP_DF, _CUES, _MARKERS, _MINED
    out = work_dir(split)
    if (out / "oth.parquet").exists():
        log(f"prepare[{split}]: cached")
        return
    s1, oth = load_split(split)
    log(f"prepare[{split}]: {len(s1):,} S1, {len(oth):,} S2/S3")
    recs = pl.concat([s1.with_columns(src=pl.lit("S1")), oth])
    names = _pmap(_fold_names, recs["name"].to_list())
    addrs = _pmap(_fold_addrs, recs["addr"].to_list())
    log("folded")
    if split == "train":
        _MINED = {}
        if os.environ.get("BER_MINE_ABBR") == "1":          # v5: abbreviations mined from the raw records (no labels)
            ts1, toth = load_split("test")
            trec = pl.concat([ts1.with_columns(src=pl.lit("S1")), toth])
            tn, ta = _pmap(_fold_names, trec["name"].to_list()), _pmap(_fold_addrs, trec["addr"].to_list())
            _MINED = mining.mine_by_country(names + tn, addrs + ta,
                                            recs["country"].to_list() + trec["country"].to_list(), log=log)
            del ts1, toth, trec, tn, ta
            both = _pmap(_apply_mined, list(zip(names, addrs, recs["country"].to_list())))
            names, addrs = [n for n, _ in both], [a for _, a in both]
            del both

    lex_path = work_dir("train") / "lexicon.json"
    ids = recs["iid"].to_list()
    n1 = len(s1)
    if split == "train":
        pairs, _ = load_gt_pairs()
        mains = [m for m, _ in names]
        _LEX_N = lexicon.learn(pairs, _tok_df(ids[:n1], mains[:n1]), _tok_df(ids[n1:], mains[n1:]), "name")
        del mains
        a_alpha = [_alpha([t for part in comps for t in part]) for comps in addrs]
        _LEX_A = lexicon.learn(pairs, _tok_df(ids[:n1], a_alpha[:n1]), _tok_df(ids[n1:], a_alpha[n1:]), "addr")
        del a_alpha
        mc = lambda ids_, comps: pl.DataFrame(
            {"iid": ids_, "comps": comps}, schema={"iid": pl.Int64, "comps": pl.List(pl.List(pl.String))})
        pseudo = _pseudo_test(set(recs["country"].unique().to_list())) if os.environ.get("BER_PSEUDO_PAIRS") else None
        if pseudo is not None:                                      # v5: learn the unseen country's vocabulary
            pp, tids, tn1, tnames, taddrs = pseudo
            tm = [m for m, _ in tnames]
            new_n = lexicon.learn(pp, _tok_df(tids[:tn1], tm[:tn1]), _tok_df(tids[tn1:], tm[tn1:]), "name")
            ta = [_alpha([t for part in comps for t in part]) for comps in taddrs]
            new_a = lexicon.learn(pp, _tok_df(tids[:tn1], ta[:tn1]), _tok_df(tids[tn1:], ta[tn1:]), "addr")
            del tm, ta
            add_n = {u: v for u, v in new_n.items() if u not in _LEX_N}     # train (ground truth) maps win
            add_a = {u: v for u, v in new_a.items() if u not in _LEX_A}
            _LEX_N.update(add_n)
            _LEX_A.update(add_a)
            log(f"pseudo lexicon: +{len(add_n):,} name maps, +{len(add_a):,} address maps; e.g. "
                + ", ".join(f"{u}->{v}" for u, v in list(add_a.items())[:25]))
        cues = []
        if P.landmark:
            mapped = _pmap(_lex_parts, addrs)
            cues = lexicon.learn_drop_cues(pairs, mc(ids[:n1], mapped[:n1]), mc(ids[n1:], mapped[n1:]))
            del mapped
            if pseudo is not None:
                mapped = _pmap(_lex_parts, taddrs)
                tcues = lexicon.learn_drop_cues(pp, mc(tids[:tn1], mapped[:tn1]), mc(tids[tn1:], mapped[tn1:]))
                del mapped
                log(f"pseudo landmark cues: {tcues}")
                cues = list(dict.fromkeys(cues + tcues))
            lex_words = set(_LEX_A) | {w for v in _LEX_A.values() for w in v.split()}
            cues = [c for c in cues if c not in lex_words]          # region names / their variants
        lex = {"name": _LEX_N, "addr": _LEX_A, "addr_cues": cues, "mined": _MINED}
    else:
        lex = json.loads(lex_path.read_text(encoding="utf-8"))
        _LEX_N, _LEX_A = lex["name"], lex["addr"]
        _MINED = lex.get("mined", {})
        if _MINED:                                          # v5: the same mined maps as train, per country
            both = _pmap(_apply_mined, list(zip(names, addrs, recs["country"].to_list())))
            names, addrs = [n for n, _ in both], [a for _, a in both]
            del both
    _CUES = frozenset(lex["addr_cues"]) if P.landmark else frozenset()
    log(f"landmark cues used: {sorted(_CUES)}")

    names = _pmap(_roman_names, names)
    addrs = _pmap(_roman_addrs, addrs)
    log("romanized")

    if split == "train":
        mains = [m for m, _ in names]
        lex["noise"] = lexicon.learn_noise_words(pairs, _tok_df(ids[:n1], mains[:n1]), _tok_df(ids[n1:], mains[n1:]))
        del mains
        fulls = [f for f, _ in addrs]
        lex["num_markers"] = lexicon.learn_number_markers(pairs, _tok_df(ids[:n1], fulls[:n1]),
                                                          _tok_df(ids[n1:], fulls[n1:]))
        del fulls
        if pseudo is not None:                                      # v5: the unseen country's noise words / markers
            rn = _pmap(_roman_names, tnames)
            tm = [m for m, _ in rn]
            t_noise = lexicon.learn_noise_words(pp, _tok_df(tids[:tn1], tm[:tn1]), _tok_df(tids[tn1:], tm[tn1:]))
            ra = _pmap(_roman_addrs, taddrs)
            tf = [f for f, _ in ra]
            t_mark = lexicon.learn_number_markers(pp, _tok_df(tids[:tn1], tf[:tn1]), _tok_df(tids[tn1:], tf[tn1:]))
            log(f"pseudo noise words: {sorted(set(t_noise) - set(lex['noise']))[:40]}; "
                f"number markers: {sorted(set(t_mark) - set(lex['num_markers']))}")
            lex["noise"] = sorted(set(lex["noise"]) | set(t_noise))
            lex["num_markers"] = sorted(set(lex["num_markers"]) | set(t_mark))
            del rn, tm, ra, tf, tnames, taddrs
        lex_path.write_text(json.dumps(lex, ensure_ascii=False), encoding="utf-8")
    noise = frozenset(lex["noise"])
    _MARKERS = frozenset(lex["num_markers"])

    # Generic words per country: in >= stop_df of that country's names (this split) + noise words.
    tok = (pl.DataFrame({"country": recs["country"], "t": [m for m, _ in names]},
                        schema={"country": pl.String, "t": pl.List(pl.String)})
           .with_row_index("r").explode("t"))
    df_t = tok.drop_nulls("t").unique(["r", "t"]).group_by("country", "t").len("df")
    del tok
    n_c = recs.group_by("country").len("n")
    stop = df_t.join(n_c, on="country").filter(pl.col("df") >= P.stop_df * pl.col("n"))
    _STOP = {c: frozenset(g["t"].to_list()) | noise for (c,), g in stop.group_by("country")}
    df_all = {c: dict(zip(g["t"].to_list(), g["df"].to_list())) for (c,), g in df_t.group_by("country")}
    _STOP_DF = {c: {t: df_all.get(c, {}).get(t, 0) for t in s} for c, s in _STOP.items()}
    del df_t, df_all
    log("generic words: " + "; ".join(f"{c}: {len(s)} ({' '.join(sorted(s)[:25])})" for c, s in _STOP.items()))
    (out / "generic_words.json").write_text(json.dumps(
        {c: sorted(s) for c, s in _STOP.items()}, ensure_ascii=False, indent=1), encoding="utf-8")

    derived = _pmap(_derive, list(zip(names, addrs, recs["country"].to_list())))
    del names, addrs
    d = pl.DataFrame(derived, schema=[(c, pl.List(pl.String) if is_list else pl.String) for c, is_list in COLS],
                     orient="row")
    del derived
    full = pl.concat([recs.select("iid", "country", "src", "name", "addr"), d], how="horizontal")
    full = full.with_columns(has_addr=pl.col("addr_n").str.len_chars() > 0,
                             native=pl.col("name").str.contains(NATIVE))

    # Uniqueness / co-location counts (per country, this split, unsupervised).
    for key, col in U_KEYS:
        g = (full.filter(pl.col(col) != "").group_by("country", col)
             .agg((pl.col("src") == "S1").sum().cast(pl.Int32).alias(f"u_s1_{key}"),
                  (pl.col("src") != "S1").sum().cast(pl.Int32).alias(f"u_oth_{key}")))
        full = full.join(g, on=["country", col], how="left")
    full = full.with_columns(pl.col("^u_.*$").fill_null(0))

    # Document frequencies for IDF weighting (per country, this split, unsupervised).
    for field, col in [("name", "name_n"), ("addr", "addr_n")]:
        (full.select("country", t=pl.col(col).str.split(" ")).with_row_index("r").explode("t")
         .filter(pl.col("t") != "").unique(["r", "t"]).group_by("country", "t").len("df")
         .join(full.group_by("country").len("n"), on="country")
         .write_parquet(out / f"df_{field}.parquet"))

    full.filter(pl.col("src") == "S1").drop("src").write_parquet(out / "s1.parquet")
    full.filter(pl.col("src") != "S1").write_parquet(out / "oth.parquet")
    log(f"prepare[{split}]: done; example: {full.head(3).select('name', 'core', 'skel', 'main_nums', 'u_s1_core').to_dicts()}")
