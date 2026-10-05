"""H1 test: pooled ΦID per head pair (past/future pairs taken within each prompt, pooled over prompts)
instead of averaging per-prompt estimates. Uses phyid internals so the maths is the library's."""
import json, sys, time
import numpy as np
from joblib import Parallel, delayed
from phyid.calculate import (_get_atoms_four_vec, _get_coinfo_four_vec, _get_double_redundancy_four_vec,
                             _get_entropy_four_vec, _get_redundancy_four_vec, calc_PhiID)
from syncore.phiid import layer_profile, syn_red_rank


def pooled_syn_red(x, y, tau=1, redundancy="MMI"):
    """x, y: (P, T). Returns mean Syn→Syn, Red→Red over all within-prompt transitions."""
    X = np.vstack([x[:, :-tau].ravel(), y[:, :-tau].ravel(), x[:, tau:].ravel(), y[:, tau:].ravel()])
    X = X / np.std(X, axis=1, ddof=1, keepdims=True)
    h = _get_entropy_four_vec(X, "gaussian")
    I = _get_coinfo_four_vec(h)
    calc = {"h_res": h, "I_res": I, "R_res": _get_redundancy_four_vec(redundancy, I)}
    calc["rtr"] = _get_double_redundancy_four_vec(redundancy, calc)
    a = _get_atoms_four_vec(calc)
    return float(np.mean(a["sts"])), float(np.mean(a["rtr"]))


# sanity: one prompt == library's calc_PhiID
r = np.random.default_rng(0).standard_normal((2, 1, 80))
a, _ = calc_PhiID(r[0, 0], r[1, 0], 1)
assert np.allclose(pooled_syn_red(r[0], r[1]), (a["sts"].mean(), a["rtr"].mean())), "pooled != library for P=1"

run = sys.argv[1]
acts = np.load(f"results/{run}/acts.npy").astype(float)  # (P, N, T)
L = json.load(open(f"results/{run}/meta.json"))["n_layers"]
P, N, T = acts.shape


def row(i):
    out = np.zeros((2, N))
    for j in range(i + 1, N):
        out[:, j] = pooled_syn_red(acts[:, i], acts[:, j])
    return out


t0 = time.time()
rows = Parallel(n_jobs=8)(delayed(row)(i) for i in range(N))
S = np.array([r_[0] for r_ in rows]); R = np.array([r_[1] for r_ in rows])
S, R = S + S.T, R + R.T
np.save(f"results/{run}/S_pooled.npy", S); np.save(f"results/{run}/R_pooled.npy", R)
rank = syn_red_rank(S, R)
off = ~np.eye(N, dtype=bool)
syn = np.array([S[i, off[i]].mean() for i in range(N)]); red = np.array([R[i, off[i]].mean() for i in range(N)])
print(f"{run} pooled ({(time.time()-t0)/60:.1f} min): corr(syn, red) over heads = {np.corrcoef(syn, red)[0,1]:+.2f}")
print("profile pooled   :", np.round(layer_profile(rank, L), 2).tolist())
print("profile per-prompt:", np.round(layer_profile(np.load(f"results/{run}/rank.npy"), L), 2).tolist())
