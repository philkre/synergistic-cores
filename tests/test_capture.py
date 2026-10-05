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
