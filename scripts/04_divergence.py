"""Fig 4a: KL behaviour divergence vs fraction of heads zeroed, synergistic vs 5 random orders.

Usage: uv run python scripts/04_divergence.py --run results/gemma [--batch-size 5]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from syncore import plots
from syncore.divergence import divergence_curve
from syncore.model import load

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--batch-size", type=int, default=5)
args = ap.parse_args()

run = Path(args.run)
meta = json.load(open(run / "meta.json"))
rank = np.load(run / "rank.npy")
tokens = np.load(run / "tokens.npy")
N = len(rank)
fractions = np.round(np.arange(0, 0.401, 0.025), 3).tolist()
orders = {"synergistic": np.argsort(-rank).tolist(), "redundant": np.argsort(rank).tolist()}
for s in range(5):
    orders[f"random{s}"] = np.random.default_rng(s).permutation(N).tolist()

model, tok = load(meta["model"])
t0 = time.time()
curves = divergence_curve(model, tok, meta["prompts"], tokens, orders, fractions, batch_size=args.batch_size,
                          chat=meta.get("chat", True))
json.dump({"fractions": fractions, "curves": curves}, open(run / "divergence.json", "w"), indent=1)
rand = np.array([curves[f"random{s}"] for s in range(5)])
plots.fig4a(np.array(fractions), np.array(curves["synergistic"]), rand, Path("results/figures/fig4a_gemma.png"))
print(f"done in {(time.time() - t0) / 60:.1f} min")
for f, sv, rv, rm, rs in zip(fractions, curves["synergistic"], curves["redundant"], rand.mean(0), rand.std(0)):
    print(f"  f={f:.3f}  syn={sv:.3f}  red={rv:.3f}  rand={rm:.3f}±{rs:.3f}")
