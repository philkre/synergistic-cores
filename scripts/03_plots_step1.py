"""Fig 2c layer-profile overlay and per-model combined Fig 2a/2b. Writes to results/figures/."""
import json
from pathlib import Path

import numpy as np

from syncore import plots

out = Path("results/figures")
out.mkdir(parents=True, exist_ok=True)

runs = {}
for name, d in [("Gemma-3-4B-it (chat)", "gemma"), ("Gemma-3-4B-it (plain prompt)", "gemma_nochat"),
                ("Qwen3-4B-Base (plain prompt)", "qwen3base")]:
    p = Path("results") / d
    if (p / "rank.npy").exists():
        runs[name] = (np.load(p / "rank.npy"), json.load(open(p / "meta.json"))["n_layers"])
plots.fig2c(runs, out / "fig2c_profiles.png")
print("wrote", sorted(f.name for f in out.iterdir()))

for title, d in [("Gemma-3-4B-it (chat)", "gemma"), ("Gemma-3-4B-it (plain prompt)", "gemma_nochat"),
                 ("Qwen3-4B-Base (plain prompt)", "qwen3base")]:
    p = Path("results") / d
    if not (p / "rank.npy").exists():
        continue
    plots.fig2ab(np.nanmean(np.load(p / "S_p.npy"), 0), np.nanmean(np.load(p / "R_p.npy"), 0), np.load(p / "rank.npy"),
                 json.load(open(p / "meta.json"))["n_layers"], title, out / f"fig2ab_{d}.png")
    print("wrote", f"fig2ab_{d}.png")
