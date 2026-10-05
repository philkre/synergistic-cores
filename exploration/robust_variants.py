"""Is the discrete improvement about binarising, or about robustness to heavy-tailed norms?
Gaussian ΦID on (a) rank-transformed normal scores (copula), (b) log norms; compare with Gaussian and discrete."""
import json
import numpy as np
from scipy.stats import norm, pearsonr, rankdata, spearmanr
from syncore.phiid import layer_profile, syn_red_rank
from syncore.phiid_fast import syn_red_matrices_fast

PAPER = np.array([0.0, 0.03, 0.09, 0.21, 0.73, 0.67, 0.33, 0.52, 0.27, 0.97, 0.49, 0.76, 0.79, 0.55, 0.82, 0.85, 0.94,
                  1.0, 0.91, 0.88, 0.70, 0.61, 0.64, 0.58, 0.39, 0.30, 0.43, 0.06, 0.12, 0.24, 0.37, 0.15, 0.18, 0.45])
sm = lambda v: np.convolve(np.r_[v[0], v, v[-1]], np.ones(3) / 3, "valid")
mme = lambda p: p[len(p) // 3:2 * len(p) // 3].mean() - np.r_[p[:len(p) // 3], p[2 * len(p) // 3:]].mean()
copula = lambda a: norm.ppf(rankdata(a, axis=-1) / (a.shape[-1] + 1))

for run in ["gemma", "gemma_nochat", "qwen3base"]:
    acts = np.load(f"results/{run}/acts.npy").astype(float)
    L = json.load(open(f"results/{run}/meta.json"))["n_layers"]
    print(f"skew of norms (median over series): {np.median(((acts - acts.mean(-1, keepdims=True))**3).mean(-1) / acts.std(-1)**3):.2f}" if run == "gemma" else "", end="")
    ranks = {"gaussian": np.load(f"results/{run}/rank.npy"), "discrete": np.load(f"results/{run}/rank_discrete.npy")}
    for name, x in [("copula", copula(acts)), ("log", np.log(acts))]:
        S, R = syn_red_matrices_fast(x)
        ranks[name] = syn_red_rank(np.nanmean(S, 0), np.nanmean(R, 0))
    print(f"\n{run}")
    for name, rk in ranks.items():
        p = layer_profile(rk, L)
        line = f"  {name:9s} midMinusEnds={mme(p):+.2f}  rho_vs_discrete={spearmanr(rk, ranks['discrete'])[0]:+.2f}  ends=({p[0]:.2f},{p[-1]:.2f})"
        if L == 34:
            line += f"  r_paper={pearsonr(p, PAPER)[0]:+.2f} smooth={pearsonr(sm(p), sm(PAPER))[0]:+.2f}"
        print(line)
