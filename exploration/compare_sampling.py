"""Compare greedy vs sampled runs vs 5-run average: roughness, reliability, agreement with paper Fig 2c (Gemma)."""
import numpy as np
from scipy.stats import pearsonr
from syncore.phiid import layer_profile, syn_red_rank

# Paper Fig 2c Gemma-3-4B-it values read by eye from the rendered figure (approx. +-0.03)
PAPER = np.array([0.0, 0.03, 0.09, 0.21, 0.73, 0.67, 0.33, 0.52, 0.27, 0.97, 0.49, 0.76, 0.79, 0.55, 0.82, 0.85, 0.94,
                  1.0, 0.91, 0.88, 0.70, 0.61, 0.64, 0.58, 0.39, 0.30, 0.43, 0.06, 0.12, 0.24, 0.37, 0.15, 0.18, 0.45])
L = 34
sm = lambda v: np.convolve(np.r_[v[0], v, v[-1]], np.ones(3) / 3, "valid")
prof = lambda S, R: layer_profile(syn_red_rank(np.nanmean(S, 0), np.nanmean(R, 0)), L)


def report(name, S, R):
    p = prof(S, R)
    print(f"{name:22s} r_paper={pearsonr(p, PAPER)[0]:+.2f}  r_paper_smooth={pearsonr(sm(p), sm(PAPER))[0]:+.2f}  "
          f"roughness={np.abs(np.diff(p)).mean():.2f}  ends=({p[0]:.2f},{p[-1]:.2f})")
    return p


load = lambda d: (np.load(f"results/{d}/S_p.npy"), np.load(f"results/{d}/R_p.npy"))
print(f"{'paper (eyeballed)':22s} roughness={np.abs(np.diff(PAPER)).mean():.2f}")
greedy = report("greedy (1 run)", *load("gemma"))
runs = [load(f"gemma_sample{s}") for s in range(5)]
profs = [report(f"sampled seed {s}", *r) for s, r in enumerate(runs)]
avg = report("sampled, 5-run average", np.concatenate([r[0] for r in runs]), np.concatenate([r[1] for r in runs]))
rr = [pearsonr(profs[i], profs[j])[0] for i in range(5) for j in range(i + 1, 5)]
print(f"run-to-run profile agreement (sampled): r mean={np.mean(rr):.2f} range {min(rr):.2f}..{max(rr):.2f}")
print(f"greedy vs 5-run sampled average: r={pearsonr(greedy, avg)[0]:.2f}")
np.save("results/gemma_sample_avg_profile.npy", avg)
