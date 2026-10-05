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
        r = rec.result()
        assert r.shape[-1] == n_tokens, f"{r.shape[-1]} forward passes for {n_tokens} tokens"
        acts.append(r)
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


@torch.no_grad()
def capture_teacher_forced(model, tok, prompts, gen_tokens, batch_size=10) -> np.ndarray:
    """Feed given token sequences (P, T) step by step with KV cache, mirroring generate()'s T forward
    passes (prefill, then tokens 0..T-2). Returns acts (P, N, T). Used to drive a random-init model
    with a trained model's outputs, so both see identical inputs."""
    h = head_geometry(model)
    gen_tokens = torch.as_tensor(np.asarray(gen_tokens))
    T = gen_tokens.shape[1]
    acts = []
    for i in range(0, len(prompts), batch_size):
        enc = encode(tok, prompts[i:i + batch_size], model.device)
        gen = gen_tokens[i:i + batch_size].to(model.device)
        am = enc["attention_mask"]
        with HeadNormRecorder(model, h) as rec:
            out = model(input_ids=enc["input_ids"], attention_mask=am,
                        position_ids=(am.cumsum(-1) - 1).clamp(min=0), use_cache=True)
            for t in range(T - 1):
                am = torch.cat([am, torch.ones_like(am[:, :1])], 1)
                out = model(input_ids=gen[:, t:t + 1], attention_mask=am, position_ids=am.sum(-1, keepdim=True) - 1,
                            past_key_values=out.past_key_values, use_cache=True)
        r = rec.result()
        assert r.shape[-1] == T, f"{r.shape[-1]} forward passes for {T} tokens"
        acts.append(r)
        print(f"  teacher-forced {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return torch.cat(acts).numpy()
