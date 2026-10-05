"""Model loading and attention-head geometry."""
from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class Heads:
    n_layers: int
    n_heads: int
    head_dim: int
    o_projs: list  # one nn.Linear per layer; input is concat of per-head attention outputs
    q_projs: list

    @property
    def n_total(self) -> int:
        return self.n_layers * self.n_heads


def head_geometry(model) -> Heads:
    """Language-model attention layers only (Gemma 3's vision tower uses out_proj, so it is skipped)."""
    cfg = model.config.get_text_config()
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    attns = [m for n, m in model.named_modules() if n.endswith("self_attn") and hasattr(m, "o_proj")]
    return Heads(len(attns), n_heads, head_dim, [a.o_proj for a in attns], [a.q_proj for a in attns])


def eos_ids(model) -> list[int]:
    e = model.generation_config.eos_token_id
    return list(e) if isinstance(e, (list, tuple)) else [e]


def load(model_id: str, dtype=torch.bfloat16, device: str = DEVICE, random_init: bool = False):
    """Return (model, tokenizer). random_init keeps architecture + generation config, re-initialises weights."""
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if random_init:
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_id), dtype=dtype)
        model.generation_config = GenerationConfig.from_pretrained(model_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    return model.to(device).eval(), tok
