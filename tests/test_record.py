import torch

from syncore.model import head_geometry
from syncore.record import HeadNormRecorder


def test_one_forward_matches_manual_norms(tiny):
    model, _ = tiny
    h = head_geometry(model)
    ids = torch.randint(0, 1000, (2, 7))
    raw = []
    hook = h.o_projs[1].register_forward_pre_hook(lambda m, a: raw.append(a[0].detach().clone()))
    with torch.no_grad(), HeadNormRecorder(model, h) as rec:
        model(input_ids=ids)
    hook.remove()
    out = rec.result()  # (B, N, T)
    assert out.shape == (2, h.n_total, 1)
    x_last = raw[0][:, -1]  # (B, H*D)
    manual = torch.stack([c.norm(dim=-1) for c in x_last.chunk(h.n_heads, dim=-1)], dim=1)  # (B, H)
    layer1 = out[:, h.n_heads:2 * h.n_heads, 0]
    torch.testing.assert_close(layer1, manual)


def test_one_column_per_forward_pass(tiny):
    model, _ = tiny
    h = head_geometry(model)
    with torch.no_grad(), HeadNormRecorder(model, h) as rec:
        for _ in range(3):
            model(input_ids=torch.randint(0, 1000, (1, 5)))
    assert rec.result().shape == (1, h.n_total, 3)


def test_hooks_removed_after_exit(tiny):
    model, _ = tiny
    h = head_geometry(model)
    with HeadNormRecorder(model, h):
        pass
    assert all(len(o._forward_pre_hooks) == 0 for o in h.o_projs)
    assert len(model._forward_pre_hooks) == 0 and len(model._forward_hooks) == 0
