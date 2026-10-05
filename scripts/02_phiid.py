"""Compute per-prompt S/R matrices, the synergy-redundancy rank, and robustness checks.

Usage: uv run python scripts/02_phiid.py --run results/gemma [--n-jobs 8] [--bench]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from syncore.phiid import layer_profile, pair_syn_red, syn_red_matrices, syn_red_rank
from syncore.phiid_fast import syn_red_matrices_fast

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--n-jobs", type=int, default=8)
ap.add_argument("--bench", action="store_true", help="time 200 pairs and extrapolate, then exit")
ap.add_argument("--slow", action="store_true", help="use phyid per pair instead of the vectorised equivalent")
args = ap.parse_args()

run = Path(args.run)
acts = np.load(run / "acts.npy").astype(np.float64)
meta = json.load(open(run / "meta.json"))
P, N, T = acts.shape
n_pairs = N * (N - 1) // 2

if args.bench:
    t0 = time.time()
    for k in range(200):
        pair_syn_red(acts[0, k % N], acts[0, (k + 1) % N])
    per = (time.time() - t0) / 200
    print(f"{per * 1e3:.2f} ms/pair → est. {per * n_pairs * P / args.n_jobs / 60:.1f} min on {args.n_jobs} jobs")
    raise SystemExit

t0 = time.time()
S_p, R_p = syn_red_matrices(acts, n_jobs=args.n_jobs) if args.slow else syn_red_matrices_fast(acts)
np.save(run / "S_p.npy", S_p)
np.save(run / "R_p.npy", R_p)
S, R = np.nanmean(S_p, 0), np.nanmean(R_p, 0)
rank = syn_red_rank(S, R)
np.save(run / "rank.npy", rank)
prof = layer_profile(rank, meta["n_layers"])
print(f"done in {(time.time() - t0) / 60:.1f} min; NaN pairs: {np.isnan(S_p).sum() // 2}")
print("layer profile:", np.round(prof, 2).tolist())

robust = {}
lens_f = run / "natural_len.json"
if lens_f.exists():
    keep = np.array(json.load(open(lens_f))) >= T
    r_sub = syn_red_rank(np.nanmean(S_p[keep], 0), np.nanmean(R_p[keep], 0))
    robust["natural_long_prompts"] = int(keep.sum())
    robust["spearman_rank_vs_long_only"] = float(spearmanr(rank, r_sub)[0])
half = np.arange(P) % 2 == 0
r_a = syn_red_rank(np.nanmean(S_p[half], 0), np.nanmean(R_p[half], 0))
r_b = syn_red_rank(np.nanmean(S_p[~half], 0), np.nanmean(R_p[~half], 0))
robust["spearman_split_half"] = float(spearmanr(r_a, r_b)[0])
json.dump(robust, open(run / "robustness.json", "w"), indent=1)
print("robustness:", robust)
