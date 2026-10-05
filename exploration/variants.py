"""Batch 1: does an alternative ΦID / ranking variant explain the Fig 2c gap?
Variants on existing captures (no new generation):
  base      : Gaussian, MMI, tau=1, average S/R over prompts then rank (current)
  rankfirst : rank heads per prompt, average ranks over prompts (no new ΦID)
  discrete  : phyid kind='discrete' (median binarisation)
  ccs       : redundancy='CCS'
  tau2      : tau=2
Criteria: (a) r vs paper Gemma curve (eyeballed), (b) inverted U in both models (mid-third minus outer thirds)."""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from syncore.phiid import layer_profile, syn_red_matrices, syn_red_rank

PAPER = np.array([0.0, 0.03, 0.09, 0.21, 0.73, 0.67, 0.33, 0.52, 0.27, 0.97, 0.49, 0.76, 0.79, 0.55, 0.82, 0.85, 0.94,
                  1.0, 0.91, 0.88, 0.70, 0.61, 0.64, 0.58, 0.39, 0.30, 0.43, 0.06, 0.12, 0.24, 0.37, 0.15, 0.18, 0.45])
VARIANTS = {"discrete": dict(kind="discrete"), "ccs": dict(redundancy="CCS"), "tau2": dict(tau=2)}
sm = lambda v: np.convolve(np.r_[v[0], v, v[-1]], np.ones(3) / 3, "valid")


def mid_minus_ends(p):
    a, b = len(p) // 3, 2 * len(p) // 3
    return p[a:b].mean() - np.r_[p[:a], p[b:]].mean()


def head_means(S, R):
    n = S.shape[-1]; off = ~np.eye(n, dtype=bool)
    return np.array([np.nanmean(S[i, off[i]]) for i in range(n)]), np.array([np.nanmean(R[i, off[i]]) for i in range(n)])


def summarize(run, name, rank, S, R, L):
    p = layer_profile(rank, L)
    syn, red = head_means(S, R)
    line = f"{run:6s} {name:10s} midMinusEnds={mid_minus_ends(p):+.2f}  corr(syn,red)={np.corrcoef(syn, red)[0,1]:+.2f}  ends=({p[0]:.2f},{p[-1]:.2f})"
    if L == 34:
        line += f"  r_paper={pearsonr(p, PAPER)[0]:+.2f}  r_paper_smooth={pearsonr(sm(p), sm(PAPER))[0]:+.2f}"
    print(line, flush=True)


for run in ["gemma", "qwen"]:
    d = Path("results") / run
    L = json.load(open(d / "meta.json"))["n_layers"]
    S_p, R_p = np.load(d / "S_p.npy"), np.load(d / "R_p.npy")
    S, R = np.nanmean(S_p, 0), np.nanmean(R_p, 0)
    summarize(run, "base", syn_red_rank(S, R), S, R, L)
    rf = np.mean([syn_red_rank(S_p[k], R_p[k]) for k in range(len(S_p))], 0)
    summarize(run, "rankfirst", rf, S, R, L)
    acts = np.load(d / "acts.npy").astype(float)
    for name, kw in VARIANTS.items():
        f = d / f"S_p_{name}.npy"
        if not f.exists():
            t0 = time.time()
            Sv, Rv = syn_red_matrices(acts, n_jobs=8, **kw)
            np.save(f, Sv); np.save(d / f"R_p_{name}.npy", Rv)
            print(f"  ({run} {name} ΦID {(time.time()-t0)/60:.1f} min)", flush=True)
        Sv, Rv = np.load(f), np.load(d / f"R_p_{name}.npy")
        Sm, Rm = np.nanmean(Sv, 0), np.nanmean(Rv, 0)
        summarize(run, name, syn_red_rank(Sm, Rm), Sm, Rm, L)
