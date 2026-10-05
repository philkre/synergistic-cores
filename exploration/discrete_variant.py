"""Discrete (mean-binarised) ΦID variant vs Gaussian, for chat/plain Gemma and Qwen3-4B-Base."""
import json, time
import numpy as np
from scipy.stats import pearsonr, spearmanr
from syncore.phiid import layer_profile, syn_red_rank
from syncore.phiid_fast import syn_red_matrices_fast

PAPER = np.array([0.0, 0.03, 0.09, 0.21, 0.73, 0.67, 0.33, 0.52, 0.27, 0.97, 0.49, 0.76, 0.79, 0.55, 0.82, 0.85, 0.94,
                  1.0, 0.91, 0.88, 0.70, 0.61, 0.64, 0.58, 0.39, 0.30, 0.43, 0.06, 0.12, 0.24, 0.37, 0.15, 0.18, 0.45])
sm = lambda v: np.convolve(np.r_[v[0], v, v[-1]], np.ones(3) / 3, "valid")


def mid_minus_ends(p):
    a, b = len(p) // 3, 2 * len(p) // 3
    return p[a:b].mean() - np.r_[p[:a], p[b:]].mean()


acts = np.load("results/gemma/acts.npy").astype(float)
S, _ = syn_red_matrices_fast(acts)
assert np.nanmax(np.abs(S - np.load("results/gemma/S_p.npy"))) < 1e-10, "gaussian path changed"
print("gaussian path still exact on Gemma")

rng = np.random.default_rng(0)
for run in ["gemma", "gemma_nochat", "qwen3base"]:
    acts = np.load(f"results/{run}/acts.npy").astype(float)
    L = json.load(open(f"results/{run}/meta.json"))["n_layers"]
    t0 = time.time()
    S_p, R_p = syn_red_matrices_fast(acts, kind="discrete")
    f = lambda idx: syn_red_rank(np.nanmean(S_p[idx], 0), np.nanmean(R_p[idx], 0))
    rank = f(np.arange(len(acts)))
    p = layer_profile(rank, L)
    ci = np.percentile([mid_minus_ends(layer_profile(f(rng.integers(0, 60, 60)), L)) for _ in range(100)], [2.5, 97.5])
    half = np.arange(60) % 2 == 0
    N = rank.size; off = ~np.eye(N, dtype=bool); Sm, Rm = np.nanmean(S_p, 0), np.nanmean(R_p, 0)
    c = np.corrcoef([np.nanmean(Sm[i, off[i]]) for i in range(N)], [np.nanmean(Rm[i, off[i]]) for i in range(N)])[0, 1]
    line = (f"{run:13s} discrete ({time.time()-t0:.0f}s): midMinusEnds={mid_minus_ends(p):+.2f} CI[{ci[0]:+.2f},{ci[1]:+.2f}]  "
            f"corr(syn,red)={c:+.2f}  split-half={spearmanr(f(half), f(~half))[0]:.2f}  "
            f"rank vs gaussian={spearmanr(rank, np.load(f'results/{run}/rank.npy'))[0]:+.2f}  ends=({p[0]:.2f},{p[-1]:.2f})")
    if L == 34:
        line += f"  r_paper={pearsonr(p, PAPER)[0]:+.2f} smooth={pearsonr(sm(p), sm(PAPER))[0]:+.2f}"
    print(line, flush=True)
    np.save(f"results/{run}/rank_discrete.npy", rank)
