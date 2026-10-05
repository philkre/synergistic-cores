import torch

from syncore.ablate import noise_heads, zero_heads
from syncore.model import head_geometry

IDS = torch.randint(0, 1000, (2, 9), generator=torch.Generator().manual_seed(0))


def _logits(model):
    with torch.no_grad():
        return model(input_ids=IDS).logits


def test_zero_nothing_is_identity(tiny):
    model, _ = tiny
    h = head_geometry(model)
    base = _logits(model)
    with zero_heads(h, []):
        torch.testing.assert_close(_logits(model), base, rtol=0, atol=0)


def test_zero_heads_equals_zeroing_o_proj_columns(tiny):
    model, _ = tiny
    h = head_geometry(model)
    idx = [1, 6]  # layer 0 head 1, layer 1 head 2
    with zero_heads(h, idx):
        masked = _logits(model)
    saved = [h.o_projs[0].weight[:, 16:32].clone(), h.o_projs[1].weight[:, 32:48].clone()]
    with torch.no_grad():
        h.o_projs[0].weight[:, 16:32] = 0
        h.o_projs[1].weight[:, 32:48] = 0
    try:
        ref = _logits(model)
    finally:
        with torch.no_grad():
            h.o_projs[0].weight[:, 16:32] = saved[0]
            h.o_projs[1].weight[:, 32:48] = saved[1]
    torch.testing.assert_close(masked, ref)
    assert not torch.allclose(masked, _logits(model))


def test_noise_restores_and_touches_only_targets(tiny):
    model, _ = tiny
    h = head_geometry(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    with noise_heads(h, [5], alpha=1.0, seed=0):  # layer 1, head 1 → rows/cols 16:32
        q, o = h.q_projs[1].weight, h.o_projs[1].weight
        q0 = before["model.layers.1.self_attn.q_proj.weight"]
        o0 = before["model.layers.1.self_attn.o_proj.weight"]
        assert not torch.equal(q[16:32], q0[16:32]) and torch.equal(q[:16], q0[:16]) and torch.equal(q[32:], q0[32:])
        assert not torch.equal(o[:, 16:32], o0[:, 16:32]) and torch.equal(o[:, :16], o0[:, :16])
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]), n
