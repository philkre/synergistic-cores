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
from syncore.math_eval import load_subset
from syncore.math_mlx import evaluate_mlx, noise_heads_mlx

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--calibrate", action="store_true")
ap.add_argument("--alpha", type=float)
ap.add_argument("--frac", type=float, default=0.25)
ap.add_argument("--batch-size", type=int, default=10, help="concurrent sequences; KV cache grows with max-tokens")
ap.add_argument("--max-tokens", type=int, default=1024)
args = ap.parse_args()

run = Path(args.run)
meta = json.load(open(run / "meta.json"))
rank = np.load(run / "rank.npy")
import mlx.core as mx
from mlx_lm import load  # MLX backend: ~2x faster than PyTorch/MPS, continuous batching

mx.set_cache_limit(1 << 30)  # don't let MLX hoard freed buffers (16 GB machine)
model, tok = load(meta["model"])
n_heads, head_dim, n_total = meta["n_heads"], meta["head_dim"], meta["n_layers"] * meta["n_heads"]
chat = meta.get("chat", True)
k = round(args.frac * n_total)
order = np.argsort(-rank)
conds = {"Synergistic core": order[:k].tolist(), "Redundant core": order[-k:].tolist()}
for s in range(3):
    conds[f"Random{s}"] = np.random.default_rng(100 + s).choice(n_total, k, replace=False).tolist()


def run_cond(problems, idx, alpha, seed=0):
    kw = dict(max_tokens=args.max_tokens, batch_size=args.batch_size, chat=chat)
    if idx is None:
        return evaluate_mlx(model, tok, problems, **kw)
    with noise_heads_mlx(model, idx, alpha, n_heads, head_dim, seed=seed):
        return evaluate_mlx(model, tok, problems, **kw)


t0 = time.time()
if args.calibrate:
    probs = load_subset(per_level=10, seed=1)  # 50 problems, disjoint seed from eval set
    base = run_cond(probs, None, 0)
    lens = [len(tok.encode(o)) for o in base["outputs"]]
    res = {"baseline": base["accuracy"], "max_tokens": args.max_tokens,
           "answer_tokens": {"median": float(np.median(lens)), "p90": float(np.percentile(lens, 90)),
                             "hit_cap": int(sum(l >= args.max_tokens - 1 for l in lens)),
                             "no_boxed": int(sum("\\boxed{" not in o for o in base["outputs"]))}}
    print(res, flush=True)
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
