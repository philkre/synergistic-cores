import numpy as np

from syncore.phiid import layer_profile, pair_syn_red, prompt_matrices, syn_red_matrices, syn_red_rank

T = 2000


def _common_driver(rng):
    z = np.zeros(T)
    for t in range(1, T):
        z[t] = 0.9 * z[t - 1] + rng.standard_normal()
    return z + 0.3 * rng.standard_normal(T), z + 0.3 * rng.standard_normal(T)


def _coupled_sum(rng):
    x, y = np.zeros(T), np.zeros(T)
    for t in range(1, T):
        s = 0.6 * (x[t - 1] + y[t - 1])
        x[t], y[t] = s + rng.standard_normal(), -s + rng.standard_normal()
    return x, y


def test_redundant_system():
    syn, red = pair_syn_red(*_common_driver(np.random.default_rng(0)))
    assert red > 0.5 and red > 10 * abs(syn)


def test_synergistic_system():
    syn, red = pair_syn_red(*_coupled_sum(np.random.default_rng(0)))
    assert syn > 0.1 and syn > red


def test_independent_system():
    rng = np.random.default_rng(0)
    syn, red = pair_syn_red(rng.standard_normal(T), rng.standard_normal(T))
    assert abs(syn) < 0.01 and abs(red) < 0.01


def test_constant_series_gives_nan():
    syn, red = pair_syn_red(np.ones(50), np.random.default_rng(0).standard_normal(50))
    assert np.isnan(syn) and np.isnan(red)


def test_prompt_matrices_symmetric_zero_diag():
    acts = np.random.default_rng(0).standard_normal((4, 60))
    S, R = prompt_matrices(acts)
    assert S.shape == (4, 4)
    np.testing.assert_allclose(S, S.T)
    np.testing.assert_allclose(R, R.T)
    assert (np.diag(S) == 0).all()


def test_syn_red_matrices_shapes():
    acts = np.random.default_rng(0).standard_normal((3, 5, 40))
    S_p, R_p = syn_red_matrices(acts, n_jobs=2)
    assert S_p.shape == R_p.shape == (3, 5, 5)


def test_rank_orders_synergistic_first():
    S = np.array([[0, 3, 3], [3, 0, 1], [3, 1, 0]], float)  # head 0 most synergistic
    R = np.array([[0, 1, 1], [1, 0, 3], [1, 3, 0]], float)  # heads 1, 2 most redundant
    r = syn_red_rank(S, R)
    assert r.argmax() == 0 and r[0] > r[1]


def test_layer_profile_minmax():
    rank = np.array([0, 0, 5, 5, 1, 1], float)  # 3 layers x 2 heads
    np.testing.assert_allclose(layer_profile(rank, n_layers=3), [0, 1, 0.2])
