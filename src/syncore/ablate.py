"""Head interventions. Head index convention: g = layer * n_heads + head."""
from contextlib import contextmanager

import torch


@contextmanager
def zero_heads(heads, idx):
    """Zero the selected heads' attention outputs by masking their slice of the o_proj input."""
    mask = torch.ones(heads.n_layers, heads.n_heads)
    for g in idx:
        mask[g // heads.n_heads, g % heads.n_heads] = 0
    handles = []
    for layer, o in enumerate(heads.o_projs):
        if bool(mask[layer].all()):
            continue
        m = mask[layer].repeat_interleave(heads.head_dim).to(o.weight.device, o.weight.dtype)
        handles.append(o.register_forward_pre_hook(lambda mod, args, m=m: (args[0] * m,) + tuple(args[1:])))
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


@contextmanager
def noise_heads(heads, idx, alpha, seed=0):
    """Add N(0, (alpha * std(W))^2) to each selected head's q_proj rows and o_proj columns; restore on exit.
    std is computed once per weight matrix before any perturbation."""
    gen = torch.Generator().manual_seed(seed)
    D = heads.head_dim
    saved = []
    with torch.no_grad():
        stds = {}
        for g in sorted(idx):
            layer, h = divmod(g, heads.n_heads)
            q, o = heads.q_projs[layer].weight, heads.o_projs[layer].weight
            if layer not in stds:
                stds[layer] = (q.float().std().item(), o.float().std().item())
            sl = slice(h * D, (h + 1) * D)
            saved.append((q, o, sl, q[sl].clone(), o[:, sl].clone()))
            q[sl] += (alpha * stds[layer][0] * torch.randn(q[sl].shape, generator=gen)).to(q.device, q.dtype)
            o[:, sl] += (alpha * stds[layer][1] * torch.randn(o[:, sl].shape, generator=gen)).to(o.device, o.dtype)
    try:
        yield
    finally:
        with torch.no_grad():
            for q, o, sl, q_old, o_old in reversed(saved):
                q[sl] = q_old
                o[:, sl] = o_old
