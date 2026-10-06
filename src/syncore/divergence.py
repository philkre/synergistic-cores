"""Teacher-forced behaviour divergence: mean over prompts and tokens of KL(p_clean || p_ablated)."""
import numpy as np
import torch

from syncore.ablate import zero_heads
from syncore.capture import encode
from syncore.model import head_geometry


@torch.no_grad()
def teacher_forced_logits(model, tok, prompts, gen, chat=True):
    """gen: (B, T) clean generated ids. Returns logits (B, T, V) on the model's device, in model dtype,
    predicting each gen token. position_ids come from the attention mask so left padding matches generate()."""
    enc = encode(tok, prompts, model.device, chat)
    gen = gen.to(model.device)
    ids = torch.cat([enc["input_ids"], gen], 1)
    am = torch.cat([enc["attention_mask"], torch.ones_like(gen)], 1)
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    T = gen.shape[1]
    return model(input_ids=ids, attention_mask=am, position_ids=pos, logits_to_keep=T + 1).logits[:, :-1]


def teacher_forced_logprobs(model, tok, prompts, gen, chat=True):
    """float32 log-probs (B, T, V) on CPU. Convenience for inspection/tests; the curve stays on device."""
    return torch.log_softmax(teacher_forced_logits(model, tok, prompts, gen, chat).float(), -1).cpu()


def _kl_sum(lp_clean, logits_abl, chunk=16):
    """Sum over batch and tokens of KL(p_clean || p_abl), computed on device in token chunks so no full-size
    float32 temporaries are materialised. Returns (sum, count)."""
    total = 0.0
    for t in range(0, lp_clean.shape[1], chunk):
        lc = lp_clean[:, t:t + chunk]
        la = torch.log_softmax(logits_abl[:, t:t + chunk].float(), -1)
        total += (lc.exp() * (lc - la)).sum().item()
    return total, lp_clean.shape[0] * lp_clean.shape[1]


def divergence_curve(model, tok, prompts, gen_tokens, orders: dict, fractions, batch_size=5, chat=True) -> dict:
    """orders: name -> list of global head indices in ablation order.
    For each order and fraction f, zero the first round(f*N) heads; returns name -> [mean KL per fraction]."""
    h = head_geometry(model)
    gen_tokens = torch.as_tensor(np.asarray(gen_tokens))
    sums = {k: np.zeros(len(fractions)) for k in orders}
    count = 0
    for i in range(0, len(prompts), batch_size):
        ps, g = prompts[i:i + batch_size], gen_tokens[i:i + batch_size]
        clean = torch.log_softmax(teacher_forced_logits(model, tok, ps, g, chat).float(), -1)  # on device
        n = 0
        for name, order in orders.items():
            for fi, f in enumerate(fractions):
                with zero_heads(h, list(order)[: round(f * h.n_total)]):
                    s, n = _kl_sum(clean, teacher_forced_logits(model, tok, ps, g, chat))
                sums[name][fi] += s
        count += n
        del clean
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"  divergence batch {i // batch_size + 1}/{-(-len(prompts) // batch_size)}", flush=True)
    return {k: (v / count).tolist() for k, v in sums.items()}
