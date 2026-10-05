import numpy as np

from syncore.capture import capture, natural_lengths
from syncore.model import eos_ids

PROMPTS = ["Why do people wear sunglasses?", "Correct the error: The books is on the table.", "Hi"]


def test_capture_shapes_and_no_eos(tiny):
    model, tok = tiny
    out = capture(model, tok, PROMPTS, n_tokens=12, batch_size=2)
    assert out["acts"].shape == (3, 12, 12)  # (P, N=3*4, T)
    assert out["tokens"].shape == (3, 12)
    assert np.isfinite(out["acts"]).all()
    assert not np.isin(out["tokens"], eos_ids(model)).any()


def test_natural_lengths_in_range(tiny):
    model, tok = tiny
    lens = natural_lengths(model, tok, PROMPTS, n_tokens=12, batch_size=2)
    assert len(lens) == 3 and all(1 <= n <= 12 for n in lens)


def test_batched_equals_single(tiny):
    """Left padding must not change a prompt's activations (mask/position_ids correctness)."""
    model, tok = tiny
    short, long = "Hi", "Describe a world where water is scarce, and every drop counts."
    batched = capture(model, tok, [short, long], n_tokens=6, batch_size=2)
    single = capture(model, tok, [short], n_tokens=6, batch_size=1)
    np.testing.assert_array_equal(batched["tokens"][0], single["tokens"][0])
    np.testing.assert_allclose(batched["acts"][0], single["acts"][0], rtol=1e-4, atol=1e-4)


def test_teacher_forced_reproduces_free_generation(tiny):
    """Feeding a model its own greedy tokens step by step must give identical head norms."""
    from syncore.capture import capture_teacher_forced
    model, tok = tiny
    free = capture(model, tok, PROMPTS, n_tokens=6, batch_size=2)
    forced = capture_teacher_forced(model, tok, PROMPTS, free["tokens"], batch_size=2)
    assert forced.shape == free["acts"].shape
    np.testing.assert_allclose(forced, free["acts"], rtol=1e-4, atol=1e-4)


def test_sampled_capture_seeded(tiny):
    model, tok = tiny
    a = capture(model, tok, PROMPTS, n_tokens=8, batch_size=3, sample=True, seed=0)
    b = capture(model, tok, PROMPTS, n_tokens=8, batch_size=3, sample=True, seed=0)
    c = capture(model, tok, PROMPTS, n_tokens=8, batch_size=3, sample=True, seed=1)
    np.testing.assert_array_equal(a["tokens"], b["tokens"])
    assert not np.array_equal(a["tokens"], c["tokens"])
    assert not np.isin(c["tokens"], eos_ids(model)).any()
