"""Generate 100 tokens per prompt and save per-head activation time series.

Usage: uv run python scripts/01_capture.py --model google/gemma-3-4b-it --out results/gemma [--random-init]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from syncore.capture import capture, natural_lengths
from syncore.model import head_geometry, load
from syncore.prompts import all_prompts

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--random-init", action="store_true")
ap.add_argument("--n-tokens", type=int, default=100)
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
model, tok = load(args.model, random_init=args.random_init)
h = head_geometry(model)
cats, prompts = zip(*all_prompts())

t0 = time.time()
res = capture(model, tok, list(prompts), n_tokens=args.n_tokens)
np.save(out / "acts.npy", res["acts"])
np.save(out / "tokens.npy", res["tokens"])
if not args.random_init:
    lens = natural_lengths(model, tok, list(prompts), n_tokens=args.n_tokens)
    json.dump(lens, open(out / "natural_len.json", "w"))
    print(f"natural length < {args.n_tokens}: {sum(n < args.n_tokens for n in lens)}/{len(lens)}")
json.dump({"model": args.model, "random_init": args.random_init,
           "n_layers": h.n_layers, "n_heads": h.n_heads, "head_dim": h.head_dim,
           "categories": list(cats), "prompts": list(prompts)}, open(out / "meta.json", "w"), indent=1)

a = res["acts"]
print(f"acts {a.shape} finite={np.isfinite(a).all()} min={a.min():.3f} max={a.max():.1f} "
      f"zero-var series={(a.std(-1) < 1e-6).sum()}  ({time.time() - t0:.0f}s)")
print("sample:", repr(tok.decode(res["tokens"][0][:60])))
