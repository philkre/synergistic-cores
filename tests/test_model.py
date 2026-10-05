from syncore.model import eos_ids, head_geometry


def test_head_geometry_tiny(tiny):
    model, _ = tiny
    h = head_geometry(model)
    assert (h.n_layers, h.n_heads, h.head_dim, h.n_total) == (3, 4, 16, 12)
    assert len(h.o_projs) == 3 and len(h.q_projs) == 3
    assert h.o_projs[0].weight.shape == (64, 64)


def test_eos_ids_is_list(tiny):
    model, tok = tiny
    assert eos_ids(model) == [tok.eos_token_id]


def test_tokenizer_left_pads(tiny):
    _, tok = tiny
    assert tok.padding_side == "left"
