"""LB probes that remove one suspect group of v4's 0.95-0.99 band pairs from the best file (t0.95, LB 0.984312):
  drop_numconf  band pairs whose addresses both have numbers and share none (same name, other street number)
  drop_noaddr   band pairs whose S2/S3 record has no address (name-only decision)
The LB change tells the precision of the removed group (break-even ~0.69): up = the group was mostly wrong."""
import csv
import os
import re
import shutil
from collections import Counter

csv.field_size_limit(10**9)
R = os.environ.get("BER_ROOT", "./")
DIG = re.compile(r"\d+")
BASE = R + "v4/output_probes/probe_v4_t0950/"
STRICT = R + "v4/output_probes/probe_v4_t0990/matching_results.tsv"


def lines(path):
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s, _, m = line.rstrip("\n").partition("\t")
            yield s, (m.split(",") if m else [])


band = {}
for (s, m95), (s2, m99) in zip(lines(BASE + "matching_results.tsv"), lines(STRICT)):
    assert s == s2
    extra = set(m95) - set(m99)
    if extra:
        band[s] = extra
need = set(band) | {b for bs in band.values() for b in bs}
rec = {}
for src in ("source1", "source2", "source3"):
    with open(R + f"dataset/test/test_{src}.tsv", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        h = next(r)
        ii, ia, ic = h.index("entity_id"), h.index("business_address"), h.index("country")
        for row in r:
            if row[ii] in need:
                rec[row[ii]] = (row[ia], row[ic])


def nums(s):
    return {x.lstrip("0") or "0" for x in DIG.findall(s)}


drop = {"drop_numconf": set(), "drop_noaddr": set()}
cnt = Counter()
for s, bs in band.items():
    aa, c = rec[s]
    for b in bs:
        ab = rec[b][0]
        cnt[("band", c)] += 1
        xa, xb = nums(aa), nums(ab)
        if xa and xb and not (xa & xb):
            drop["drop_numconf"].add((s, b))
            cnt[("numconf", c)] += 1
        if not ab.strip():
            drop["drop_noaddr"].add((s, b))
            cnt[("noaddr", c)] += 1
for k in ("band", "numconf", "noaddr"):
    print(k, {c: cnt[(k, c)] for c in ("France", "India", "US")}, "total", sum(cnt[(k, c)] for c in ("France", "India", "US")))

for name, pairs in drop.items():
    d = R + f"v4/output_diag/t095_{name}"
    os.makedirs(d, exist_ok=True)
    changed = 0
    with open(d + "/matching_results.tsv", "w", encoding="utf-8", newline="") as o, \
         open(BASE + "matching_results.tsv", encoding="utf-8") as f:
        o.write(next(f))
        for line in f:
            s, _, m = line.rstrip("\n").partition("\t")
            ids = m.split(",") if m else []
            keep = [b for b in ids if (s, b) not in pairs]
            changed += len(keep) != len(ids)
            o.write(s + "\t" + ",".join(keep) + "\n")
    shutil.copyfile(BASE + "candidate_pairs.tsv", d + "/candidate_pairs.tsv")
    print(f"{name}: removed {len(pairs):,} pairs, {changed:,} S1 changed -> {d}")
