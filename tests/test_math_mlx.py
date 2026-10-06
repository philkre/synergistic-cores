import types

import mlx.core as mx
import numpy as np

from syncore.math_mlx import attention_layers, noise_heads_mlx


def _fake_model(n_layers=2, n_heads=4, head_dim=8, hidden=16, seed=0):
    rng = np.random.default_rng(seed)
    layers = []
    for _ in range(n_layers):
        attn = types.SimpleNamespace(
            q_proj=types.SimpleNamespace(weight=mx.array(rng.standard_normal((n_heads * head_dim, hidden)), dtype=mx.bfloat16)),
            o_proj=types.SimpleNamespace(weight=mx.array(rng.standard_normal((hidden, n_heads * head_dim)), dtype=mx.bfloat16)))
        layers.append(types.SimpleNamespace(self_attn=attn))
    return types.SimpleNamespace(layers=layers)


def test_attention_layers_found():
    assert len(attention_layers(_fake_model())) == 2


def test_noise_targets_only_selected_head_and_restores():
    m = _fake_model()
    D = 8
    before = [(np.array(l.self_attn.q_proj.weight.astype(mx.float32)), np.array(l.self_attn.o_proj.weight.astype(mx.float32)))
              for l in attention_layers(m)]
    with noise_heads_mlx(m, [5], alpha=1.0, n_heads=4, head_dim=D, seed=0):  # layer 1, head 1 -> 8:16
        q = np.array(attention_layers(m)[1].self_attn.q_proj.weight.astype(mx.float32))
        o = np.array(attention_layers(m)[1].self_attn.o_proj.weight.astype(mx.float32))
        q0, o0 = before[1]
        assert not np.array_equal(q[8:16], q0[8:16]) and np.array_equal(q[:8], q0[:8]) and np.array_equal(q[16:], q0[16:])
        assert not np.array_equal(o[:, 8:16], o0[:, 8:16]) and np.array_equal(o[:, :8], o0[:, :8])
        assert np.array_equal(np.array(attention_layers(m)[0].self_attn.q_proj.weight.astype(mx.float32)), before[0][0])
    for l, (q0, o0) in zip(attention_layers(m), before):
        assert np.array_equal(np.array(l.self_attn.q_proj.weight.astype(mx.float32)), q0)
        assert np.array_equal(np.array(l.self_attn.o_proj.weight.astype(mx.float32)), o0)
