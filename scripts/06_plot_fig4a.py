"""Paper-style combined Fig 4a from saved divergence.json files."""
import json
from pathlib import Path

import numpy as np

from syncore import plots

models = {}
for name, d in [("Gemma-3-4B-it", "gemma"), ("Qwen3-4B-Base", "qwen3base")]:
    f = Path("results") / d / "divergence.json"
    if f.exists():
        r = json.load(open(f))
        c = r["curves"]
        models[name] = dict(syn=np.array(c["synergistic"]), red=np.array(c["redundant"]),
                            rand=np.array([c[k] for k in c if k.startswith("random")]))
        fractions = np.array(r["fractions"])
plots.fig4a_combined(fractions, models, Path("results/figures/fig4a.png"))
print("wrote fig4a.png:", list(models))
