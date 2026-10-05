import numpy as np

from syncore.phiid import syn_red_matrices
from syncore.phiid_fast import syn_red_matrices_fast


def _coupled(rng, n_heads, T):
    """Heads with shared drivers, autocorrelation and cross-lag coupling, so all atoms are non-trivial."""
    z = rng.standard_normal((3, T))
    for t in range(1, T):
        z[:, t] += 0.7 * z[:, t - 1]
    mix = rng.standard_normal((n_heads, 3))
    x = mix @ z + 0.5 * rng.standard_normal((n_heads, T))
    x[1:, 1:] += 0.4 * x[:-1, :-1]  # head i+1 at t driven by head i at t-1
    return np.abs(x) + 0.1  # positive like L2 norms


def test_matches_phyid_on_coupled_data():
    rng = np.random.default_rng(0)
    acts = np.stack([_coupled(rng, 7, 80) for _ in range(3)])  # (P=3, N=7, T=80)
    S_ref, R_ref = syn_red_matrices(acts, n_jobs=1)
    S, R = syn_red_matrices_fast(acts)
    np.testing.assert_allclose(S, S_ref, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(R, R_ref, rtol=1e-6, atol=1e-9)


def test_matches_phyid_tau2():
    rng = np.random.default_rng(1)
    acts = np.stack([_coupled(rng, 5, 60) for _ in range(2)])
    S_ref, R_ref = syn_red_matrices(acts, tau=2, n_jobs=1)
    S, R = syn_red_matrices_fast(acts, tau=2)
    np.testing.assert_allclose(S, S_ref, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(R, R_ref, rtol=1e-6, atol=1e-9)


def test_nan_for_constant_and_singular_pairs():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(50)
    acts = np.stack([x, np.roll(x, -1), np.ones(50), rng.standard_normal(50)])[None]  # (1, 4, 50)
    S, R = syn_red_matrices_fast(acts)
    assert np.isnan(S[0, 0, 1]) and np.isnan(R[0, 0, 1])  # lagged copy -> singular
    assert np.isnan(S[0, 2, 3]) and np.isnan(S[0, 3, 2])  # constant series
    assert np.isfinite(S[0, 0, 3]) and (np.diag(S[0]) == 0).all()
