# Interactive visual guides

Self-contained HTML pages: download and open in a browser. GitHub shows HTML files as source code, so use **Raw →
save**, or serve them with GitHub Pages. They follow the browser's light or dark theme.

| page | what it shows |
|---|---|
| `pipeline-flow.html` | the main pipeline step by step, with each step's code file and result |
| `inside-v4-v5.html` | every step in plain words with a running example, plus a technical panel (formula, settings, model, libraries) that can be toggled; the model inventory, libraries, what v5 adds, and the blend |
| `pipeline-math.html` | the formulas behind each stage (metric, lexicon learning, retrieval, contrastive loss, boosting, calibration, decision, the leaderboard-difference method, the blend); needs internet for the equation renderer |

Static charts for the docs are in [`../assets/`](../assets/), drawn by [`../scripts/make_figures.py`](../scripts/make_figures.py).
