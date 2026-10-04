"""Train vs test data profile per country and source (pure Python, local)."""
import os
import csv
import re
from collections import Counter, defaultdict

csv.field_size_limit(10**9)
R = os.environ.get("BER_ROOT", "./") + "dataset/"
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
SP = re.compile(r"\s+")
DIGIT = re.compile(r"\d")


def norm(s):
    return SP.sub(" ", PUNCT.sub(" ", s.lower())).strip()


def latin(s):
    return all(ord(ch) < 0x250 for ch in s)


def read(path):
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        h = next(r)
        ii, iname, iaddr, ic = (h.index(k) for k in ("entity_id", "business_name", "business_address", "country"))
        for row in r:
            yield row[ii], row[iname], row[iaddr], row[ic]


stats = defaultdict(Counter)          # (split, country, src) -> counters
s1_names = {}                        # (split, country) -> Counter of normalized S1 names
oth_names = {}                       # (split, country) -> Counter of normalized S2+S3 names
for split in ("train", "test"):
    for src in ("source1", "source2", "source3"):
        for eid, name, addr, c in read(R + f"{split}/{split}_{src}.tsv"):
            k = (split, c, "S1" if src == "source1" else "S2+S3")
            st = stats[k]
            st["n"] += 1
            st["addr_empty"] += not addr.strip()
            st["addr_digit"] += bool(DIGIT.search(addr))
            st["name_digit"] += bool(DIGIT.search(name))
            st["native"] += not latin(name + addr)
            st["name_chars"] += len(name)
            st["name_words"] += len(name.split())
            st["addr_chars"] += len(addr)
            st["addr_commas"] += addr.count(",")
            st["web_name"] += (".com" in name.lower() or "www" in name.lower())
            nn = norm(name)
            d = s1_names if src == "source1" else oth_names
            d.setdefault((split, c), Counter())[nn] += 1

print("| split | country | source | records | per S1 | addr empty | addr has number | name has number | native script | name chars | name words | addr chars | addr commas | web-style name |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k in sorted(stats):
    split, c, src = k
    st = stats[k]
    n = st["n"]
    per = n / stats[(split, c, "S1")]["n"]
    print(f"| {split} | {c} | {src} | {n:,} | {per:.2f} | {st['addr_empty'] / n:.4f} | {st['addr_digit'] / n:.4f} | "
          f"{st['name_digit'] / n:.4f} | {st['native'] / n:.4f} | {st['name_chars'] / n:.1f} | {st['name_words'] / n:.2f} | "
          f"{st['addr_chars'] / n:.1f} | {st['addr_commas'] / n:.2f} | {st['web_name'] / n:.4f} |")

print("\n| split | country | S1 sharing exact name with 1+ other S1 | with 3+ other S1 | exact-name S2/S3 look-alikes per S1 (mean) | S1 with 10+ exact-name S2/S3 |")
print("|---|---|---|---|---|---|")
for (split, c), cnt in sorted(s1_names.items()):
    oth = oth_names.get((split, c), Counter())
    n = sum(cnt.values())
    dup1 = sum(v for v in cnt.values() if v >= 2) / n
    dup3 = sum(v for v in cnt.values() if v >= 4) / n
    look = sum(v * oth.get(nm, 0) for nm, v in cnt.items()) / n
    big = sum(v for nm, v in cnt.items() if oth.get(nm, 0) >= 10) / n
    print(f"| {split} | {c} | {dup1:.4f} | {dup3:.4f} | {look:.2f} | {big:.4f} |")

# train GT: matches per S1
gt = Counter()
with open(R + "train/train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        m = line.rstrip("\n").split("\t", 1)[1] if "\t" in line else ""
        gt[min(len(m.split(",")) if m else 0, 8)] += 1
tot = sum(gt.values())
print("\ntrain matches per S1 (GT): " + ", ".join(f"{k}{'+' if k == 8 else ''}: {v / tot:.4f}" for k, v in sorted(gt.items())))
pr = Counter()
with open(os.environ.get("BER_ROOT", "./") + "v4/output/matching_results.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        m = line.rstrip("\n").split("\t", 1)[1] if "\t" in line else ""
        pr[min(len(m.split(",")) if m else 0, 8)] += 1
tot = sum(pr.values())
print("test matches per S1 (v4 predicted): " + ", ".join(f"{k}{'+' if k == 8 else ''}: {v / tot:.4f}" for k, v in sorted(pr.items())))
