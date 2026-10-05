"""Fig 2a/2b for Gemma and Fig 2c overlay of all runs. Writes to results/figures/."""
import json
from pathlib import Path

import numpy as np

from syncore import plots

out = Path("results/figures")
out.mkdir(parents=True, exist_ok=True)
g = Path("results/gemma")
meta = json.load(open(g / "meta.json"))
plots.fig2a(np.nanmean(np.load(g / "S_p.npy"), 0), np.nanmean(np.load(g / "R_p.npy"), 0), out / "fig2a_gemma.png")
plots.fig2b(np.load(g / "rank.npy"), meta["n_layers"], out / "fig2b_gemma.png")

runs = {}
for name, d in [("Gemma-3-4B-it", "gemma"), ("Gemma-3-4B random init", "gemma_random"),
                ("Qwen2.5-Math-1.5B", "qwen")]:
    p = Path("results") / d
    if (p / "rank.npy").exists():
        runs[name] = (np.load(p / "rank.npy"), json.load(open(p / "meta.json"))["n_layers"])
plots.fig2c(runs, out / "fig2c_profiles.png")
print("wrote", sorted(f.name for f in out.iterdir()))
