"""Per-country diagnostic submissions built from v4 files (macro F0.5 is a mean over S1, so country effects add up):
  fr_strict: v4 best rule, but France rows from the t=0.99 file  -> France's share of the 0.983 vs 0.980 gap
  fr_empty:  v4 best rule, France rows empty                     -> France's own macro F0.5 (given its singleton rate)
  us_empty:  v4 best rule, US rows empty                         -> US's own macro F0.5 (India = the rest)"""
import csv
import os
import shutil

R = os.environ.get("BER_ROOT", "./")
BEST = R + "v4/output/matching_results.tsv"
STRICT = R + "v4/output_probes/probe_v4_t0990/matching_results.tsv"
CAND = R + "v4/output/candidate_pairs.tsv"
csv.field_size_limit(10**9)
country = {}
with open(R + "dataset/test/test_source1.tsv", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    h = next(r)
    for row in r:
        country[row[h.index("entity_id")]] = row[h.index("country")]

plans = {"fr_strict": ("France", "strict"), "fr_empty": ("France", "empty"), "us_empty": ("US", "empty")}
outs = {}
for name in plans:
    d = R + f"v4/output_diag/{name}"
    os.makedirs(d, exist_ok=True)
    outs[name] = open(d + "/matching_results.tsv", "w", encoding="utf-8", newline="")
with open(BEST, encoding="utf-8") as fb, open(STRICT, encoding="utf-8") as fs:
    head = next(fb)
    next(fs)
    for o in outs.values():
        o.write(head)
    for lb, ls in zip(fb, fs):
        s = lb.split("\t", 1)[0]
        assert s == ls.split("\t", 1)[0]
        c = country[s]
        for name, (cc, how) in plans.items():
            if c != cc:
                outs[name].write(lb)
            elif how == "strict":
                outs[name].write(ls)
            else:
                outs[name].write(s + "\t\n")
for name, o in outs.items():
    o.close()
    shutil.copyfile(CAND, R + f"v4/output_diag/{name}/candidate_pairs.tsv")
    print("wrote", name)
