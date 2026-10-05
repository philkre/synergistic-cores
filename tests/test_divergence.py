import numpy as np
import torch

from syncore.capture import capture
from syncore.divergence import divergence_curve, teacher_forced_logprobs
from syncore.model import head_geometry

PROMPTS = ["Why do people wear sunglasses?", "Correct the error: The books is on the table.", "Hi there"]


def test_teacher_forcing_reproduces_greedy(tiny):
    model, tok = tiny
    gen = capture(model, tok, PROMPTS, n_tokens=8, batch_size=3)["tokens"]
    lp = teacher_forced_logprobs(model, tok, PROMPTS, torch.tensor(gen))
    assert lp.shape[:2] == (3, 8)
    agree = (lp.argmax(-1).numpy() == gen).mean()
    assert agree > 0.9


def test_curve_zero_at_zero_and_positive_after(tiny):
    model, tok = tiny
    h = head_geometry(model)
    gen = capture(model, tok, PROMPTS, n_tokens=8, batch_size=3)["tokens"]
    orders = {"syn": list(range(h.n_total)), "rand0": list(np.random.default_rng(0).permutation(h.n_total))}
    curves = divergence_curve(model, tok, PROMPTS, gen, orders, fractions=[0.0, 0.25, 0.5], batch_size=2)
    assert set(curves) == {"syn", "rand0"}
    for c in curves.values():
        assert len(c) == 3 and abs(c[0]) < 1e-5 and c[2] > 0
