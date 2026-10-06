"""MATH evaluation on MLX (Apple's framework): ~2x faster generation than PyTorch/MPS on this Mac, plus continuous
batching (finished answers are replaced immediately instead of waiting for the slowest in a fixed batch).
Head noise mirrors syncore.ablate.noise_heads: same head slices, same torch-generated noise values."""
from contextlib import contextmanager

import mlx.core as mx
import numpy as np
import torch

from syncore.math_eval import SUFFIX, is_correct


def attention_layers(model):
    """Decoder layers with softmax attention (q_proj + o_proj), for Gemma 3 / Qwen 3 MLX layouts."""
    for path in [("layers",), ("model", "layers"), ("language_model", "model", "layers"), ("language_model", "layers")]:
        obj = model
        try:
            for p in path:
                obj = getattr(obj, p)
        except AttributeError:
            continue
        layers = [l for l in obj if hasattr(getattr(l, "self_attn", None), "q_proj")]
        if layers:
            return layers
    raise ValueError("no attention layers found")


def _np(w):
    return np.array(w.astype(mx.float32))


@contextmanager
def noise_heads_mlx(model, idx, alpha, n_heads, head_dim, seed=0):
    """Add N(0, (alpha * std(W))^2) to each selected head's q_proj rows and o_proj columns; restore on exit.
    Noise drawn from the same torch generator sequence as syncore.ablate.noise_heads."""
    layers = attention_layers(model)
    gen = torch.Generator().manual_seed(seed)
    D = head_dim
    saved, stds = {}, {}
    for g in sorted(idx):
        layer, h = divmod(g, n_heads)
        attn = layers[layer].self_attn
        if layer not in saved:
            saved[layer] = (attn.q_proj.weight, attn.o_proj.weight)
            stds[layer] = (float(np.std(_np(attn.q_proj.weight), ddof=1)), float(np.std(_np(attn.o_proj.weight), ddof=1)))
        sl = slice(h * D, (h + 1) * D)
        q, o = _np(attn.q_proj.weight), _np(attn.o_proj.weight)
        q[sl] += alpha * stds[layer][0] * torch.randn(q[sl].shape, generator=gen).numpy()
        o[:, sl] += alpha * stds[layer][1] * torch.randn(o[:, sl].shape, generator=gen).numpy()
        dtype = attn.q_proj.weight.dtype
        attn.q_proj.weight, attn.o_proj.weight = mx.array(q).astype(dtype), mx.array(o).astype(dtype)
    try:
        yield
    finally:
        for layer, (q0, o0) in saved.items():
            layers[layer].self_attn.q_proj.weight, layers[layer].self_attn.o_proj.weight = q0, o0


def evaluate_mlx(model, tok, problems, max_tokens=512, batch_size=20, chat=True) -> dict:
    """Greedy, continuously batched generation; scored by last \\boxed{} answer."""
    from mlx_lm import batch_generate
    texts = [p["problem"] + SUFFIX for p in problems]
    prompts = ([tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True) for t in texts]
               if chat else [tok.encode(t) for t in texts])
    outputs = batch_generate(model, tok, prompts, max_tokens=max_tokens, completion_batch_size=batch_size).texts
    correct = [is_correct(o, p["answer"]) for o, p in zip(outputs, problems)]
    return {"accuracy": float(np.mean(correct)), "correct": correct, "outputs": outputs}
