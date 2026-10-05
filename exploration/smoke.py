"""Smoke test: load model on MPS, generate 100 tokens, capture per-head o_proj-input norms."""
import sys, time
import torch, psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = sys.argv[1]
dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[sys.argv[2] if len(sys.argv) > 2 else "bf16"]
dev = "mps"

t0 = time.time()
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(dev).eval()
print(f"class={type(model).__name__} load={time.time()-t0:.1f}s mps_alloc={torch.mps.driver_allocated_memory()/1e9:.2f}GB rss={psutil.Process().memory_info().rss/1e9:.2f}GB")

cfg = model.config.get_text_config()
n_heads, head_dim = cfg.num_attention_heads, getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
o_projs = [m for n, m in model.named_modules() if n.endswith("self_attn.o_proj")]
print(f"layers={len(o_projs)} heads/layer={n_heads} head_dim={head_dim}")

norms = []  # per generated step: (layers, heads)
step = []
def hook(mod, args):
    x = args[0][:, -1]  # last position only
    step.append(x.view(n_heads, head_dim).float().norm(dim=-1).cpu())
hs = [m.register_forward_pre_hook(hook) for m in o_projs]

msgs = [{"role": "user", "content": "If you have 15 apples and you give away 5, how many do you have left?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(dev)

# flush per-forward-pass hook outputs (hooks, not forward patching: generate() inspects forward's signature)
hs.append(model.register_forward_pre_hook(lambda m, a: step.clear()))
hs.append(model.register_forward_hook(lambda m, a, o: norms.append(torch.stack(step))))

t0 = time.time()
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=100, min_new_tokens=100, do_sample=False)
dt = time.time() - t0
for h in hs: h.remove()

acts = torch.stack(norms[1:])  # drop prefill pass -> (T, layers, heads)
print(f"gen 100 tok: {dt:.1f}s = {100/dt:.1f} tok/s  peak mps_alloc={torch.mps.driver_allocated_memory()/1e9:.2f}GB rss={psutil.Process().memory_info().rss/1e9:.2f}GB")
print(f"acts shape={tuple(acts.shape)} finite={torch.isfinite(acts).all().item()} mean={acts.mean():.3f} min={acts.min():.3f} max={acts.max():.3f}")
print("frac heads with ~zero variance over time:", (acts.std(0) < 1e-6).float().mean().item())
print("---\n" + tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)[:400])
