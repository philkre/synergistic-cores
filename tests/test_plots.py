import numpy as np

from syncore import plots


def test_step1_figures_write_files(tmp_path):
    rng = np.random.default_rng(0)
    S, R = rng.random((6, 6)), rng.random((6, 6))
    rank = rng.permutation(6).astype(float)
    plots.fig2a(S, R, tmp_path / "a.png")
    plots.fig2b(rank, n_layers=3, path=tmp_path / "b.png")
    plots.fig2c({"m1": (rank, 3), "m2": (rank[::-1], 3)}, tmp_path / "c.png")
    assert all((tmp_path / f).stat().st_size > 1000 for f in ["a.png", "b.png", "c.png"])


def test_step2_figures_write_files(tmp_path):
    fr = np.linspace(0, 0.4, 5)
    plots.fig4a(fr, np.linspace(0, 2, 5), np.vstack([np.linspace(0, 1, 5)] * 3), tmp_path / "d.png")
    plots.fig4b({"Baseline": [0.5], "Redundant core": [0.45], "Random": [0.4, 0.42], "Synergistic core": [0.3]},
                tmp_path / "e.png")
    assert (tmp_path / "d.png").stat().st_size > 1000 and (tmp_path / "e.png").stat().st_size > 1000
