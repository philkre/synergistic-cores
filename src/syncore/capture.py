"""Batched greedy chat generation with head-norm capture."""
import numpy as np
import torch

from syncore.model import eos_ids, head_geometry
from syncore.record import HeadNormRecorder


def encode(tok, prompts, device):
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                   return_dict=True, padding=True).to(device)


@torch.no_grad()
def generate(model, tok, prompts, n_tokens, ban_eos=True):
    """Returns (encoding, generated ids (B, <=n_tokens)). ban_eos forces exactly n_tokens without EOS."""
    enc = encode(tok, prompts, model.device)
    kw = dict(max_new_tokens=n_tokens, do_sample=False)
    if ban_eos:
        kw.update(min_new_tokens=n_tokens, suppress_tokens=eos_ids(model))
    out = model.generate(**enc, **kw)
    return enc, out[:, enc["input_ids"].shape[1]:]


def capture(model, tok, prompts, n_tokens=100, batch_size=10) -> dict:
    """acts: (P, N, T) float32 head norms; tokens: (P, T) generated ids.
    Timestep 0 is the prefill pass's last position (it produces generated token 0)."""
    h = head_geometry(model)
    acts, toks = [], []
    for i in range(0, len(prompts), batch_size):
        with HeadNormRecorder(model, h) as rec:
            _, gen = generate(model, tok, prompts[i:i + batch_size], n_tokens, ban_eos=True)
        acts.append(rec.result()[..., :n_tokens])
        toks.append(gen.cpu())
        print(f"  captured {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return {"acts": torch.cat(acts).numpy(), "tokens": torch.cat(toks).numpy()}


def natural_lengths(model, tok, prompts, n_tokens=100, batch_size=10) -> list[int]:
    """Tokens generated before the first EOS (capped at n_tokens), EOS allowed."""
    eos = set(eos_ids(model))
    lens = []
    for i in range(0, len(prompts), batch_size):
        _, gen = generate(model, tok, prompts[i:i + batch_size], n_tokens, ban_eos=False)
        for row in gen.tolist():
            lens.append(next((j for j, t in enumerate(row) if t in eos), len(row)) or 1)
    return lens
