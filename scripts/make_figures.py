"""Draw the figures in assets/ from the numbers reported by the runs (see docs/03-results-and-analysis.md for sources).
Usage: python scripts/make_figures.py   (matplotlib only)"""
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "assets"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated categorical pair (light surface)
INK, INK2, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": NEUTRAL, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white",
})


def frame(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


NOTES = {}


def title(fig, ax, t, sub):
    """Title and subtitle anchored to the figure's left edge (not the plot area), so long y labels can't push them off."""
    fig.text(0.012, 0.985, t, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.985 - 0.36 / fig.get_figheight(), textwrap.fill(sub, 105), fontsize=10, color=INK2, va="top")


def note(fig, text):
    NOTES[fig] = textwrap.fill(text, 120)


def save(fig, name):
    h = fig.get_figheight()
    top = 1 - 0.85 / h
    bottom = (0.32 + 0.16 * NOTES[fig].count("\n")) / h if fig in NOTES else 0
    if fig in NOTES:
        fig.text(0.012, 0.012, NOTES[fig], fontsize=8.5, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print("wrote", OUT / name)


# 1. score progression -------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.6))
x = ["v2", "v3", "v4", "v5a", "Final blend\n0.7·v4 + 0.3·v5a"]
oof = [0.9781, 0.9881, 0.9913, 0.9910, None]
lb = [0.9700, 0.9780, 0.984312, 0.98376, 0.984403]
xi = range(len(x))
ax.plot([i for i, v in zip(xi, oof) if v], [v for v in oof if v], color=BLUE, lw=2, marker="o", ms=7,
        mec="white", mew=2, label="Train out-of-fold macro F0.5")
ax.plot(list(xi), lb, color=ORANGE, lw=2, marker="o", ms=7, mec="white", mew=2, label="Public leaderboard (best cut)")
for i, v in zip(xi, oof):
    if v:
        ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 9), ha="center", color=INK, fontsize=9.5)
for i, v in zip(xi, lb):
    ax.annotate(f"{v:.6f}".rstrip("0") if v > 0.98 else f"{v:.4f}", (i, v), textcoords="offset points",
                xytext=(0, -16), ha="center", color=INK, fontsize=9.5)
ax.set_xticks(list(xi), x)
ax.set_ylim(0.965, 0.996)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}"))
frame(ax)
ax.legend(loc="lower right", frameon=False)
note(fig, "v0 baseline (name-only TF-IDF + XGBoost) scored 0.593 on the leaderboard, below this scale. "
     "v5a's train score is measured on a harder, test-like training set.")
title(fig, ax, "Score progression across versions", "Macro F0.5 per S1 entity; leaderboard = public subset of test")
save(fig, "score_progression.png")

# 2. candidate funnel ---------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 3.9))
stages = ["All records of the S1's country", "13 search routes (union)", "+ fine-tuned retriever",
          "Stage-1 shortlist (= candidate_pairs.tsv)", "Final matches"]
vals = [3.9e6, 92.3, 110.5, 5.85, 3.37]
notes = ["nothing compared yet", "recall 0.9908", "recall 0.9964", "recall 0.9955 (test: 6.45 per S1)",
         "97.6% of true pairs found (train OOF)"]
y = list(range(len(stages)))[::-1]
ax.barh(y, vals, color=BLUE, height=0.5)
ax.set_xscale("log")
ax.set_xlim(1, 6e6)
for yi, v, n in zip(y, vals, notes):
    lab = f"{v / 1e6:.1f} million" if v >= 1e6 else (f"{v:,.0f}" if v >= 100 else f"{v:g}")
    if v >= 1e6:
        ax.text(1.4, yi, f"≈ {lab} per S1 · {n}", va="center", fontsize=9.5, color="white", fontweight="bold")
    else:
        ax.text(v * 1.25, yi, f"{lab} per S1 · {n}", va="center", fontsize=9.5, color=INK)
ax.set_yticks(y, stages)
ax.set_xlabel("records per S1 (log scale)")
frame(ax, "x")
title(fig, ax, "From millions of records to a handful per S1", "Candidates kept per S1 at each stage, and the share "
      "of true pairs still reachable (train)")
save(fig, "funnel.png")

# 3. threshold probes ---------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.4))
ax.plot([0.30, 0.70, 0.95], [0.958, 0.97493, 0.978], color=ORANGE, lw=2, marker="o", ms=7, mec="white", mew=2,
        label="v3")
ax.plot([0.70, 0.95, 0.99], [0.983, 0.984312, 0.980], color=BLUE, lw=2, marker="o", ms=7, mec="white", mew=2,
        label="v4")
for xv, yv in ((0.30, 0.958), (0.70, 0.97493), (0.95, 0.978), (0.70, 0.983), (0.95, 0.984312), (0.99, 0.980)):
    lab = f"{yv:.6f}".rstrip("0")
    lab = lab if len(lab.split(".")[1]) >= 3 else f"{yv:.3f}"
    ax.annotate(lab, (xv, yv), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9.5, color=INK)
ax.set_xticks([0.30, 0.50, 0.70, 0.95, 0.99], ["0.30", "0.50", "0.70\n(train-tuned rule)", "0.95", "0.99"])
ax.set_xlim(0.25, 1.02)
ax.set_ylim(0.955, 0.988)
ax.set_xlabel("decision threshold on the calibrated match probability")
frame(ax)
ax.legend(loc="lower right", frameon=False)
title(fig, ax, "Test rewards a stricter cut than training suggests", "Public leaderboard for the same scores at different "
      "thresholds (one S1 per record)")
save(fig, "threshold_probes.png")

# 4. train vs test orphan share -----------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 3.2))
rows = [("Train as given (v4)", 74, 26), ("Train after dropping 19% of S1 (v5a)", 60, 40), ("Test (estimated)", 58, 42)]
for i, (lab, m, o) in enumerate(rows[::-1]):
    ax.barh(i, m, color=BLUE, height=0.5)
    ax.barh(i, o, left=m + 0.4, color=ORANGE, height=0.5)
    ax.text(m / 2, i, f"{m}%", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.text(m + 0.4 + o / 2, i, f"{o}%", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
ax.set_yticks(range(3), [r[0] for r in rows[::-1]])
ax.set_xlim(0, 100.5)
ax.set_xticks([])
for s in ("top", "right", "bottom"):
    ax.spines[s].set_visible(False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
          labels=["S2/S3 records that belong to an S1", "orphans (no S1 in the data)"], loc="upper center",
          bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
title(fig, ax, "The train→test shift: twice as many unmatched look-alikes", "Share of S2/S3 records; 1.22 orphans per "
      "S1 in train vs ≈ 2.4 in test")
save(fig, "train_test_shift.png")

# 5. stage-3 feature gain -----------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 3.6))
feats = [("p2 (stage-2 matcher)", 69.3), ("p_v2 (auxiliary LightGBM matcher)", 17.5), ("p1 (stage-1 shortlist)", 8.4),
         ("p_ce (cross-encoder)", 2.7), ("pair features", 1.4), ("graph consistency", 0.7)]
y = list(range(len(feats)))[::-1]
ax.barh(y, [v for _, v in feats], color=BLUE, height=0.5)
for yi, (_, v) in zip(y, feats):
    ax.text(v + 0.8, yi, f"{v:g}%", va="center", fontsize=9.5, color=INK)
ax.set_yticks(y, [n for n, _ in feats])
ax.set_xlim(0, 80)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
frame(ax, "x")
title(fig, ax, "What the final model (stage 3) relies on", "Share of XGBoost gain by input, v4 stage 3")
save(fig, "stage3_feature_gain.png")

# 6. where true pairs are lost ------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 3.0))
loss = [("Not found by the candidate search", 27488, 0.36), ("Cut by the stage-1 shortlist", 6773, 0.09),
        ("Rejected by the matcher", 151656, 2.0)]
y = list(range(len(loss)))[::-1]
ax.barh(y, [v for _, v, _ in loss], color=ORANGE, height=0.5)
for yi, (_, v, s) in zip(y, loss):
    ax.text(v + 2500, yi, f"{v:,} pairs ({s}%)", va="center", fontsize=9.5, color=INK)
ax.set_yticks(y, [n for n, _, _ in loss])
ax.set_xlim(0, 200000)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
frame(ax, "x")
title(fig, ax, "Where v4 loses true pairs (train, out-of-fold)", "Of 7.60M true pairs, 97.6% are found; only 9,480 false "
      "pairs are predicted")
save(fig, "loss_budget.png")

# 7. blend and vote variants --------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.0))
var = [("0.7 v4 + 0.3 v5a (submitted)", 0.00010), ("0.8 v4 + 0.2 v5a", 0.00009), ("0.6 v4 + 0.4 v5a", 0.00009),
       ("0.5 v4 + 0.5 v5a", 0.00008), ("majority vote (v4, v5a, v3)", 0.00005), ("0.5/0.3/0.2 with v3", -0.00006),
       ("equal average of the three", -0.00021), ("unanimous vote (best cut)", -0.00040)]
y = list(range(len(var)))[::-1]
ax.barh(y, [v * 1e4 for _, v in var], color=[BLUE if v > 0 else ORANGE for _, v in var], height=0.5)
for yi, (_, v) in zip(y, var):
    ax.text(v * 1e4 + (0.08 if v > 0 else -0.08), yi, f"{v:+.5f}", va="center", ha="left" if v > 0 else "right",
            fontsize=9.5, color=INK)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(y, [n for n, _ in var])
ax.set_xlim(-5.5, 2)
ax.set_xlabel("change in train macro F0.5 vs v4 alone, at v4's pair count (×10⁻⁴)")
frame(ax, "x")
title(fig, ax, "Blends and votes, judged on test-like training data", "Same number of predicted pairs as v4 at 0.95; "
      "1.78M train S1 with test-like orphan density")
save(fig, "blend_variants.png")

# 8. embedder ablation --------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.8))
emb = [("multilingual-e5-large", 0.963), ("multilingual-e5-small", 0.960), ("arctic-embed-l-v2.0", 0.957),
       ("bge-m3", 0.942), ("multilingual-e5-base", 0.936), ("multilingual-e5-large-instruct", 0.925),
       ("Qwen3-Embedding-0.6B", 0.888), ("bhasha-embed-v0", 0.851), ("KaLM-mini-instruct-v1.5", 0.843),
       ("granite-embedding-311m-r2", 0.803), ("granite-embedding-97m-r2", 0.784), ("IndicBERTv2-SS", 0.283)]
y = list(range(len(emb)))[::-1]
chosen = {"multilingual-e5-large", "multilingual-e5-small"}
ax.barh(y, [v for _, v in emb], color=[BLUE if n in chosen else NEUTRAL for n, _ in emb], height=0.55)
for yi, (n, v) in zip(y, emb):
    ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=9.5, color=INK)
ax.set_yticks(y, [n for n, _ in emb])
ax.set_xlim(0, 1.08)
frame(ax, "x")
note(fig, "Blue = the two models used: complementary, e5-large is best on native script and e5-small on records "
     "without an address.")
title(fig, ax, "Choosing the embedders: recall@10 of off-the-shelf models", "Mean of US and India, exact search, 40k "
      "train S1 queries (MIT/Apache-2.0 models only)")
save(fig, "embedder_ablation.png")

# 9. feature-group ablation (v3) ----------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 3.9))
grp = [("house-number relations", -0.00111), ("name uniqueness", -0.00103), ("listwise rank / gap", -0.00072),
       ("trade-name / address route", -0.00016), ("phonetic", -0.00011), ("typo", -0.00010),
       ("second dense view", 0.0), ("name 3-gram route", 0.00003)]
y = list(range(len(grp)))[::-1]
ax.barh(y, [v * 1e4 for _, v in grp], color=[ORANGE if v < 0 else BLUE for _, v in grp], height=0.5)
for yi, (_, v) in zip(y, grp):
    ax.text(v * 1e4 - 0.15 if v < 0 else v * 1e4 + 0.15, yi, f"{v:+.5f}", va="center",
            ha="right" if v < 0 else "left", fontsize=9.5, color=INK)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(y, [n for n, _ in grp])
ax.set_xlim(-14, 3)
ax.set_xlabel("change in out-of-fold macro F0.5 when the group is removed (×10⁻⁴)")
frame(ax, "x")
note(fig, "All groups: 0.98772 (3-fold). Using only the 44 v2 features instead: 0.98032 (−0.0074).")
title(fig, ax, "Which stage-2 feature groups matter (v3 ablation A3)", "Removing one group at a time, full train set")
save(fig, "feature_group_ablation.png")
