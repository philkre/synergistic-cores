"""Teacher-forced behaviour divergence: mean over prompts and tokens of KL(p_clean || p_ablated)."""
import numpy as np
import torch

from syncore.ablate import zero_heads
from syncore.capture import encode


@torch.no_grad()
def teacher_forced_logprobs(model, tok, prompts, gen):
    """gen: (B, T) clean generated ids. Returns float32 log-probs (B, T, V) predicting each gen token.
    position_ids are derived from the attention mask so left padding matches generate()."""
    enc = encode(tok, prompts, model.device)
    gen = gen.to(model.device)
    ids = torch.cat([enc["input_ids"], gen], 1)
    am = torch.cat([enc["attention_mask"], torch.ones_like(gen)], 1)
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    T = gen.shape[1]
    logits = model(input_ids=ids, attention_mask=am, position_ids=pos, logits_to_keep=T + 1).logits[:, :-1]
    return torch.log_softmax(logits.float(), -1).cpu()


def _kl(lp_clean, lp_abl):
    """Mean over batch and tokens of sum_v p_c (log p_c − log p_a). Returns (sum, count) for pooling."""
    kl = (lp_clean.exp() * (lp_clean - lp_abl)).sum(-1)  # (B, T)
    return kl.sum().item(), kl.numel()


def divergence_curve(model, tok, prompts, gen_tokens, orders: dict, fractions, batch_size=5) -> dict:
    """orders: name -> list of global head indices in ablation order.
    For each order and fraction f, zero the first round(f*N) heads; returns name -> [mean KL per fraction]."""
    from syncore.model import head_geometry
    h = head_geometry(model)
    gen_tokens = torch.as_tensor(np.asarray(gen_tokens))
    sums = {k: np.zeros(len(fractions)) for k in orders}
    count = 0
    for i in range(0, len(prompts), batch_size):
        ps, g = prompts[i:i + batch_size], gen_tokens[i:i + batch_size]
        clean = teacher_forced_logprobs(model, tok, ps, g)
        n = 0
        for name, order in orders.items():
            for fi, f in enumerate(fractions):
                with zero_heads(h, list(order)[: round(f * h.n_total)]):
                    s, n = _kl(clean, teacher_forced_logprobs(model, tok, ps, g))
                sums[name][fi] += s
        count += n
        print(f"  divergence batch {i // batch_size + 1}/{-(-len(prompts) // batch_size)}", flush=True)
    return {k: (v / count).tolist() for k, v in sums.items()}
