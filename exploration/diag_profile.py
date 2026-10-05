"""Diagnose Fig 2c layer profile: per-layer syn/red decomposition + activation statistics."""
import json, sys
import numpy as np
from scipy.stats import rankdata
from syncore.phiid import layer_profile, syn_red_rank

run = sys.argv[1]
S_p, R_p = np.load(f"results/{run}/S_p.npy"), np.load(f"results/{run}/R_p.npy")
acts = np.load(f"results/{run}/acts.npy").astype(float)
L = json.load(open(f"results/{run}/meta.json"))["n_layers"]
P, N, T = acts.shape
S, R = np.nanmean(S_p, 0), np.nanmean(R_p, 0)
off = ~np.eye(N, dtype=bool)
syn = np.array([np.nanmean(S[i, off[i]]) for i in range(N)])
red = np.array([np.nanmean(R[i, off[i]]) for i in range(N)])
rank = syn_red_rank(S, R)

def per_layer(v): return np.asarray(v, float).reshape(L, -1).mean(1)
cv = acts.std(-1) / acts.mean(-1)                                   # (P, N)
x0, x1 = acts[..., :-1] - acts[..., :-1].mean(-1, keepdims=True), acts[..., 1:] - acts[..., 1:].mean(-1, keepdims=True)
ac1 = (x0 * x1).sum(-1) / np.sqrt((x0**2).sum(-1) * (x1**2).sum(-1))  # lag-1 autocorr (P, N)
z0 = (acts[..., 0] - acts[..., 1:].mean(-1)) / acts[..., 1:].std(-1)  # prefill step vs rest
nuniq = np.array([[len(np.unique(acts[p, i])) for i in range(N)] for p in range(P)])

print(f"{run}: rows = layers.  syn/red = mean pairwise atom;  rk = mean syn-red rank;  prof = Fig2c value")
print(" L    syn     red   rk_syn rk_red  prof |  CV    ac1   |z_t0|  uniq<50")
rs, rr, prof = per_layer(rankdata(syn)), per_layer(rankdata(red)), layer_profile(rank, L)
for l in range(L):
    sl = slice(l * (N // L), (l + 1) * (N // L))
    print(f"{l:2d}  {syn[sl].mean():.4f}  {red[sl].mean():.4f}  {rs[l]:5.0f}  {rr[l]:5.0f}  {prof[l]:.2f} | "
          f"{np.median(cv[:, sl]):.3f} {np.median(ac1[:, sl]):+.2f}  {np.median(np.abs(z0[:, sl])):5.2f}  {(nuniq[:, sl] < 50).mean():.2f}")
print(f"corr over heads: syn vs red {np.corrcoef(syn, red)[0,1]:+.2f};  rank vs CV {np.corrcoef(rank, cv.mean(0))[0,1]:+.2f};  "
      f"rank vs ac1 {np.corrcoef(rank, ac1.mean(0))[0,1]:+.2f}")
