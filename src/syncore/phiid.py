"""Pairwise ΦID (Gaussian, MMI) between head activation series → synergy/redundancy matrices and ranking."""
import numpy as np
from joblib import Parallel, delayed
from phyid.calculate import calc_PhiID
from scipy.stats import rankdata


def pair_syn_red(x, y, tau=1, redundancy="MMI") -> tuple[float, float]:
    """Time-averaged Syn→Syn and Red→Red atoms. NaN if either series is constant or the joint
    past/future covariance is singular (e.g. one series is a lagged copy of the other)."""
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan, np.nan
    try:
        atoms, _ = calc_PhiID(np.asarray(x, float), np.asarray(y, float), tau, kind="gaussian", redundancy=redundancy)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    return float(np.mean(atoms["sts"])), float(np.mean(atoms["rtr"]))


def prompt_matrices(acts_p, tau=1, redundancy="MMI"):
    """acts_p: (N, T) → symmetric S, R of shape (N, N), zero diagonal."""
    n = acts_p.shape[0]
    S, R = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s, r = pair_syn_red(acts_p[i], acts_p[j], tau, redundancy)
            S[i, j] = S[j, i] = s
            R[i, j] = R[j, i] = r
    return S, R


def syn_red_matrices(acts, tau=1, redundancy="MMI", n_jobs=8):
    """acts: (P, N, T) → per-prompt S_p, R_p of shape (P, N, N)."""
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(prompt_matrices)(a, tau, redundancy) for a in acts)
    return np.stack([r[0] for r in res]), np.stack([r[1] for r in res])


def syn_red_rank(S, R):
    """S, R: (N, N) prompt-averaged. Per head: mean over its pairs; rank(rank(syn) − rank(red)) in 1..N.
    Higher = more synergistic. The outer re-rank matches the paper's Fig 2b scale (1..N, no negatives)
    and stops a few extreme heads dominating layer means; head order is unchanged."""
    n = S.shape[0]
    off = ~np.eye(n, dtype=bool)
    syn = np.array([np.nanmean(S[i, off[i]]) for i in range(n)])
    red = np.array([np.nanmean(R[i, off[i]]) for i in range(n)])
    return rankdata(rankdata(syn) - rankdata(red))


def layer_profile(rank, n_layers):
    """Mean rank per layer, min-max normalised to [0, 1] (paper Fig 2c)."""
    m = np.asarray(rank, float).reshape(n_layers, -1).mean(1)
    return (m - m.min()) / (m.max() - m.min())
