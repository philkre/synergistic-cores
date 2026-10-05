import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

TOKENIZER_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"  # cached locally; has a chat template


@pytest.fixture(scope="session")
def tiny_path(tmp_path_factory):
    """A 3-layer, 4-head random Qwen2 saved to disk with a real tokenizer."""
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    cfg = Qwen2Config(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=1024,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    d = tmp_path_factory.mktemp("tiny")
    model.save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)


@pytest.fixture(scope="session")
def tiny(tiny_path):
    from syncore.model import load
    return load(tiny_path, dtype=torch.float32, device="cpu")
