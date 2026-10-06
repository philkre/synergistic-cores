"""Fig 4b: MATH accuracy under Gaussian weight noise on 25% of heads.

Calibrate: uv run python scripts/05_math.py --run results/gemma --calibrate
Evaluate:  uv run python scripts/05_math.py --run results/gemma --alpha 1.0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from syncore import plots
from syncore.ablate import noise_heads
from syncore.math_eval import evaluate, load_subset
from syncore.model import head_geometry, load

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--calibrate", action="store_true")
ap.add_argument("--alpha", type=float)
ap.add_argument("--frac", type=float, default=0.25)
ap.add_argument("--batch-size", type=int, default=10)
args = ap.parse_args()

run = Path(args.run)
meta = json.load(open(run / "meta.json"))
rank = np.load(run / "rank.npy")
model, tok = load(meta["model"])
h = head_geometry(model)
k = round(args.frac * h.n_total)
order = np.argsort(-rank)
conds = {"Synergistic core": order[:k].tolist(), "Redundant core": order[-k:].tolist()}
for s in range(3):
    conds[f"Random{s}"] = np.random.default_rng(100 + s).choice(h.n_total, k, replace=False).tolist()


def run_cond(problems, idx, alpha, seed=0):
    if idx is None:
        return evaluate(model, tok, problems, batch_size=args.batch_size)
    with noise_heads(h, idx, alpha, seed=seed):
        return evaluate(model, tok, problems, batch_size=args.batch_size)


t0 = time.time()
if args.calibrate:
    probs = load_subset(per_level=10, seed=1)  # 50 problems, disjoint seed from eval set
    res = {"baseline": run_cond(probs, None, 0)["accuracy"]}
    for alpha in [0.5, 1.0, 2.0]:
        res[f"random_a{alpha}"] = run_cond(probs, conds["Random0"], alpha)["accuracy"]
        print(res, f"({(time.time() - t0) / 60:.0f} min)", flush=True)
    json.dump(res, open(run / "math_calibration.json", "w"), indent=1)
    raise SystemExit

assert args.alpha is not None, "pass --alpha (from calibration) or --calibrate"
probs = load_subset(per_level=30, seed=0)
results = {"alpha": args.alpha, "baseline": run_cond(probs, None, 0)}
print(f"baseline {results['baseline']['accuracy']:.3f}", flush=True)
for name, idx in conds.items():
    results[name] = run_cond(probs, idx, args.alpha)
    print(f"{name} {results[name]['accuracy']:.3f}  ({(time.time() - t0) / 60:.0f} min)", flush=True)
    json.dump(results, open(run / "math.json", "w"), indent=1)  # checkpoint after each condition

acc = {"Baseline": [results["baseline"]["accuracy"]],
       "Redundant core": [results["Redundant core"]["accuracy"]],
       "Random": [results[f"Random{s}"]["accuracy"] for s in range(3)],
       "Synergistic core": [results["Synergistic core"]["accuracy"]]}
plots.fig4b(acc, Path("results/figures/fig4b_gemma.png"))
print({k: round(float(np.mean(v)), 3) for k, v in acc.items()})
