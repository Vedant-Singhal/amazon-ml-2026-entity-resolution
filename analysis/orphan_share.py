"""Orphan records: S2/S3 records that belong to no S1 in the split. Train: from GT. Test: estimated from v4's predictions
(records v4 matches to some S1) — and the ratio of records per S1, per country."""
import csv
from collections import Counter
csv.field_size_limit(10**9)
R = "dataset/"

def countries(path):
    c = Counter(); ids = {}
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE); h = next(r)
        ii, ic = h.index("entity_id"), h.index("country")
        for row in r:
            c[row[ic]] += 1; ids[row[ii]] = row[ic]
    return c, ids

def matched(path):
    s = set()
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            m = line.rstrip("\n").split("\t", 1)[1] if "\t" in line else ""
            if m: s.update(m.split(","))
    return s

for split, mfile in (("train", R + "train/train_ground_truth.tsv"), ("test", "v4/output/matching_results.tsv")):
    c1, id1 = countries(R + f"{split}/{split}_source1.tsv")
    c2, id2 = countries(R + f"{split}/{split}_source2.tsv")
    c3, id3 = countries(R + f"{split}/{split}_source3.tsv")
    m = matched(mfile)
    print(f"\n## {split} ({'ground truth' if split == 'train' else 'v4 predictions'})")
    print("| country | S1 | S2+S3 | S2+S3 per S1 | matched S2+S3 | orphan share |")
    print("|---|---|---|---|---|---|")
    for c in sorted(c1):
        tot = c2[c] + c3[c]
        mm = sum(1 for x in m if (id2.get(x) or id3.get(x)) == c)
        print(f"| {c} | {c1[c]:,} | {tot:,} | {tot / c1[c]:.3f} | {mm:,} | {1 - mm / tot:.4f} |")
