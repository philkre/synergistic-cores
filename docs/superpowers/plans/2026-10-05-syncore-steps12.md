# Synergistic Core Reproduction (Steps 1 & 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **USER REQUIREMENT:** Stop and check in with the user at every `🛑 CHECK-IN` marker. Report the listed numbers/figures and wait for a go-ahead before continuing.

**Goal:** Reproduce the synergy map (Fig 2a/b/c) and ablation results (Fig 4a/b) of arXiv:2601.06851 for Gemma-3-4B-it on an M1 Pro 16 GB, plus a Qwen2.5-Math-1.5B head ranking.

**Architecture:** A small `syncore` package with one module per concern (prompts, model loading, head-norm recording, capture, ΦID, ablation, divergence, MATH eval, plots). Numbered scripts run each pipeline stage and cache results to `results/<run>/` so stages rerun independently. Unit tests use a tiny randomly-initialised Qwen2 model on CPU; real runs use MPS.

**Tech Stack:** Python 3.13, uv, PyTorch 2.14 (MPS), transformers 5.18, `phyid` (Imperial-MIND-lab ΦID), joblib, scipy, datasets, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-syncore-steps12-design.md`

**Conventions used everywhere:**
- Global head index `g = layer * n_heads + head`.
- `acts` array shape `(P, N, T)`: prompts × heads × timesteps (T = 100).
- `rank` array shape `(N,)`: `rankdata(mean_syn) − rankdata(mean_red)`; **higher = more synergistic**.
- Run dirs: `results/gemma` (Gemma-3-4B-it), `results/gemma_random` (random init), `results/qwen` (Qwen2.5-Math-1.5B-Instruct).
- Forced length = EOS suppressed. (HF's `min_new_tokens` itself works by suppressing EOS, so "forced" and "banned" are identical — no separate no-ban variant.)

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | deps, build system (so `syncore` is importable), pytest config |
| `src/syncore/__init__.py` | empty |
| `src/syncore/prompts.py` | the 60 Appendix-A prompts |
| `src/syncore/model.py` | `load()`, `Heads` geometry, `eos_ids()`, `DEVICE` |
| `src/syncore/record.py` | `HeadNormRecorder` context manager (o_proj-input norms per forward pass) |
| `src/syncore/capture.py` | batched chat generation; `capture()` → acts + tokens; `natural_lengths()` |
| `src/syncore/phiid.py` | pairwise ΦID → per-prompt S/R; `syn_red_rank()`; `layer_profile()` |
| `src/syncore/ablate.py` | `zero_heads()` and `noise_heads()` context managers |
| `src/syncore/divergence.py` | teacher-forced log-probs; `divergence_curve()` |
| `src/syncore/math_eval.py` | MATH-500 subset, `\boxed{}` extraction, normalisation, `evaluate()` |
| `src/syncore/plots.py` | Fig 2a, 2b, 2c, 4a, 4b in matplotlib |
| `scripts/01_capture.py` | run capture → `acts.npy`, `tokens.npy`, `natural_len.json`, `meta.json` |
| `scripts/02_phiid.py` | ΦID → `S_p.npy`, `R_p.npy`, `rank.npy`, `robustness.json` |
| `scripts/03_plots_step1.py` | Fig 2a/2b/2c |
| `scripts/04_divergence.py` | Fig 4a data + plot |
| `scripts/05_math.py` | α calibration and Fig 4b data + plot |
| `tests/conftest.py` | tiny-model fixture |
| `tests/test_*.py` | one test file per module |

---

### Task 1: Project scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `src/syncore/__init__.py`, `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Add build system and pytest config to `pyproject.toml`**

Append to the existing file (keep the `[project]` table and dependency list that `uv add` wrote):

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/syncore"]

[tool.hatch.metadata]
allow-direct-references = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package and tiny-model fixture**

`src/syncore/__init__.py`: empty file. `tests/__init__.py`: empty file.

`tests/conftest.py`:

```python
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
```

- [ ] **Step 3: Sync and verify import**

Run: `uv sync && uv run python -c "import syncore; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/syncore/__init__.py tests/__init__.py tests/conftest.py
git commit -m "chore: scaffold syncore package and tiny-model test fixture"
```

---

### Task 2: Prompts

**Files:**
- Create: `src/syncore/prompts.py`
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing test**

`tests/test_prompts.py`:

```python
from syncore.prompts import PROMPTS, all_prompts


def test_six_categories_of_ten():
    assert list(PROMPTS) == ["syntax", "pos", "numerical", "commonsense", "creative", "social"]
    assert all(len(v) == 10 for v in PROMPTS.values())


def test_all_prompts_order_and_content():
    ps = all_prompts()
    assert len(ps) == 60
    assert ps[0] == ("syntax", "Correct the error: He go to school every day.")
    assert ps[-1] == ("social", "Imagine a scenario where a character has to forgive someone who wronged them.")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'syncore.prompts'`

- [ ] **Step 3: Write implementation**

`src/syncore/prompts.py`:

```python
"""The 60 prompts from Appendix A of Urbina-Rodriguez et al. (2026), verbatim."""

PROMPTS = {
    "syntax": [f"Correct the error: {s}" for s in [
        "He go to school every day.", "She have two cats and a dogs.", "I eats breakfast at 8:00 in the morning.",
        "Every students in the classroom has their own laptop.", "She don't like going to the park on weekends.",
        "We was happy to see the rainbow after the storm.", "There is many reasons to celebrate today.",
        "Him and I went to the market yesterday.", "The books is on the table.",
        "They walks to school together every morning."]],
    "pos": [f"Identify the parts of speech in the sentence: {s}" for s in [
        "Quickly, the agile cat climbed the tall tree.",
        "She whispered a secret to her friend during the boring lecture.",
        "The sun sets in the west.", "Can you believe this amazing view?", "He quickly finished his homework.",
        "The beautifully decorated cake was a sight to behold.", "They will travel to Japan next month.",
        "My favorite book was lost.", "The loud music could be heard from miles away.",
        "She sold all of her paintings at the art show."]],
    "numerical": [
        "If you have 15 apples and you give away 5, how many do you have left?",
        "A rectangle's length is twice its width. If the rectangle's perimeter is 36 meters, what are its length and width?",
        "You read 45 pages of a book each day. How many pages will you have read after 7 days?",
        "If a train travels 60 miles in 1 hour, how far will it travel in 3 hours?",
        "There are 8 slices in a pizza. If you eat 2 slices, what fraction of the pizza is left?",
        "If one pencil costs 50 cents, how much do 12 pencils cost?",
        "You have a 2-liter bottle of soda. If you pour out 500 milliliters, how much soda is left?",
        "A marathon is 42 kilometers long. If you have run 10 kilometers, how much further do you have to run?",
        "If you divide 24 by 3, then multiply by 2, what is the result?",
        "A car travels 150 miles on 10 gallons of gas. How many miles per gallon does the car get?"],
    "commonsense": [
        "If it starts raining while the sun is shining, what weather phenomenon might you expect to see?",
        "Why do people wear sunglasses?", "What might you use to write on a chalkboard?",
        "Why would you put a letter in an envelope?", "If you're cold, what might you do to get warm?",
        "What is the purpose of a refrigerator?", "Why might someone plant a tree?",
        "What happens to ice when it's left out in the sun?", "Why do people shake hands when they meet?",
        "What can you use to measure the length of a desk?"],
    "creative": [
        "Imagine a future where humans have evolved to live underwater. Describe the adaptations they might develop.",
        "Invent a sport that could be played on Mars considering its lower gravity compared to Earth. Describe the rules.",
        "Describe a world where water is scarce, and every drop counts.",
        "Write a story about a child who discovers they can speak to animals.",
        "Imagine a city that floats in the sky. What does it look like, and how do people live?",
        "Create a dialogue between a human and an alien meeting for the first time.",
        "Design a vehicle that can travel on land, water, and air. Describe its features.",
        "Imagine a new holiday and explain how people celebrate it.",
        "Write a poem about a journey through a desert.",
        "Describe a device that allows you to experience other people's dreams."],
    "social": [
        "Write a dialogue between two characters where one comforts the other after a loss, demonstrating empathy.",
        "Describe a situation where someone misinterprets a friend's actions as hostile, and how they resolve the misunderstanding.",
        "Compose a letter from a character apologising for a mistake they made.",
        "Describe a scene where a character realizes they are in love.",
        "Write a conversation between two old friends who haven't seen each other in years.",
        "Imagine a character facing a moral dilemma. What do they choose and why?",
        "Describe a character who is trying to make amends for past actions.",
        "Write about a character who overcomes a fear with the help of a friend.",
        "Create a story about a misunderstanding between characters from different cultures.",
        "Imagine a scenario where a character has to forgive someone who wronged them."],
}


def all_prompts() -> list[tuple[str, str]]:
    """(category, prompt) pairs in fixed order: category by category, 10 each."""
    return [(cat, p) for cat, ps in PROMPTS.items() for p in ps]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/syncore/prompts.py tests/test_prompts.py
git commit -m "feat: add Appendix A prompts"
```

---

### Task 3: Model loading and head geometry

**Files:**
- Create: `src/syncore/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

`tests/test_model.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'syncore.model'`

- [ ] **Step 3: Write implementation**

`src/syncore/model.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/syncore/model.py tests/test_model.py
git commit -m "feat: model loading and head geometry"
```

---

### Task 4: Head-norm recorder

**Files:**
- Create: `src/syncore/record.py`
- Test: `tests/test_record.py`

- [ ] **Step 1: Write the failing test**

`tests/test_record.py`:

```python
import torch

from syncore.model import head_geometry
from syncore.record import HeadNormRecorder


def test_one_forward_matches_manual_norms(tiny):
    model, _ = tiny
    h = head_geometry(model)
    ids = torch.randint(0, 1000, (2, 7))
    raw = []
    hook = h.o_projs[1].register_forward_pre_hook(lambda m, a: raw.append(a[0].detach().clone()))
    with torch.no_grad(), HeadNormRecorder(model, h) as rec:
        model(input_ids=ids)
    hook.remove()
    out = rec.result()  # (B, N, T)
    assert out.shape == (2, h.n_total, 1)
    x_last = raw[0][:, -1]  # (B, H*D)
    manual = torch.stack([c.norm(dim=-1) for c in x_last.chunk(h.n_heads, dim=-1)], dim=1)  # (B, H)
    layer1 = out[:, h.n_heads:2 * h.n_heads, 0]
    torch.testing.assert_close(layer1, manual)


def test_one_column_per_forward_pass(tiny):
    model, _ = tiny
    h = head_geometry(model)
    with torch.no_grad(), HeadNormRecorder(model, h) as rec:
        for _ in range(3):
            model(input_ids=torch.randint(0, 1000, (1, 5)))
    assert rec.result().shape == (1, h.n_total, 3)


def test_hooks_removed_after_exit(tiny):
    model, _ = tiny
    h = head_geometry(model)
    with HeadNormRecorder(model, h):
        pass
    assert all(len(o._forward_pre_hooks) == 0 for o in h.o_projs)
    assert len(model._forward_pre_hooks) == 0 and len(model._forward_hooks) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_record.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'syncore.record'`

- [ ] **Step 3: Write implementation**

`src/syncore/record.py`:

```python
"""Record per-head L2 norm of attention output (o_proj input) at the last position of every forward pass."""
import torch


class HeadNormRecorder:
    """Context manager. Uses model-level pre/post hooks to group per-layer captures into one
    timestep per forward pass (don't patch model.forward: generate() inspects its signature)."""

    def __init__(self, model, heads):
        self.model, self.h = model, heads

    def __enter__(self):
        self.steps, self._cur, self._handles = [], [], []
        self._handles.append(self.model.register_forward_pre_hook(self._start))
        self._handles += [o.register_forward_pre_hook(self._record) for o in self.h.o_projs]
        self._handles.append(self.model.register_forward_hook(self._end))
        return self

    def __exit__(self, *exc):
        for hd in self._handles:
            hd.remove()

    def _start(self, module, args):
        self._cur = []

    def _record(self, module, args):
        x = args[0][:, -1]  # (B, H*D)
        self._cur.append(x.view(x.shape[0], self.h.n_heads, self.h.head_dim).float().norm(dim=-1).cpu())

    def _end(self, module, args, output):
        self.steps.append(torch.cat(self._cur, dim=1))  # (B, L*H), layer-major

    def result(self) -> torch.Tensor:
        """(B, N, T) with N = n_layers * n_heads, T = number of forward passes."""
        return torch.stack(self.steps, dim=-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_record.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/syncore/record.py tests/test_record.py
git commit -m "feat: per-head attention-output norm recorder"
```

---

### Task 5: Capture (batched generation)

**Files:**
- Create: `src/syncore/capture.py`
- Test: `tests/test_capture.py`

- [ ] **Step 1: Write the failing test**

`tests/test_capture.py`:

```python
import numpy as np

from syncore.capture import capture, natural_lengths
from syncore.model import eos_ids

PROMPTS = ["Why do people wear sunglasses?", "Correct the error: The books is on the table.", "Hi"]


def test_capture_shapes_and_no_eos(tiny):
    model, tok = tiny
    out = capture(model, tok, PROMPTS, n_tokens=12, batch_size=2)
    assert out["acts"].shape == (3, 12, 12)  # (P, N=3*4, T)
    assert out["tokens"].shape == (3, 12)
    assert np.isfinite(out["acts"]).all()
    assert not np.isin(out["tokens"], eos_ids(model)).any()


def test_natural_lengths_in_range(tiny):
    model, tok = tiny
    lens = natural_lengths(model, tok, PROMPTS, n_tokens=12, batch_size=2)
    assert len(lens) == 3 and all(1 <= n <= 12 for n in lens)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_capture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'syncore.capture'`

- [ ] **Step 3: Write implementation**

`src/syncore/capture.py`:

```python
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
```

Note: `or 1` guards the degenerate "EOS as first token" case so lengths stay ≥ 1.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_capture.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/syncore/capture.py tests/test_capture.py
git commit -m "feat: batched capture with EOS ban and natural lengths"
```

---

### Task 6: Capture script + real runs

**Files:**
- Create: `scripts/01_capture.py`

- [ ] **Step 1: Write the script**

`scripts/01_capture.py`:

```python
"""Generate 100 tokens per prompt and save per-head activation time series.

Usage: uv run python scripts/01_capture.py --model google/gemma-3-4b-it --out results/gemma [--random-init]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from syncore.capture import capture, natural_lengths
from syncore.model import head_geometry, load
from syncore.prompts import all_prompts

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--random-init", action="store_true")
ap.add_argument("--n-tokens", type=int, default=100)
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
model, tok = load(args.model, random_init=args.random_init)
h = head_geometry(model)
cats, prompts = zip(*all_prompts())

t0 = time.time()
res = capture(model, tok, list(prompts), n_tokens=args.n_tokens)
np.save(out / "acts.npy", res["acts"])
np.save(out / "tokens.npy", res["tokens"])
if not args.random_init:
    lens = natural_lengths(model, tok, list(prompts), n_tokens=args.n_tokens)
    json.dump(lens, open(out / "natural_len.json", "w"))
    print(f"natural length < {args.n_tokens}: {sum(n < args.n_tokens for n in lens)}/{len(lens)}")
json.dump({"model": args.model, "random_init": args.random_init,
           "n_layers": h.n_layers, "n_heads": h.n_heads, "head_dim": h.head_dim,
           "categories": list(cats), "prompts": list(prompts)}, open(out / "meta.json", "w"), indent=1)

a = res["acts"]
print(f"acts {a.shape} finite={np.isfinite(a).all()} min={a.min():.3f} max={a.max():.1f} "
      f"zero-var series={(a.std(-1) < 1e-6).sum()}  ({time.time() - t0:.0f}s)")
print("sample:", repr(tok.decode(res["tokens"][0][:60])))
```

- [ ] **Step 2: Run on Gemma (≈10 min incl. natural-length pass)**

Run: `uv run python scripts/01_capture.py --model google/gemma-3-4b-it --out results/gemma`
Expected: `acts (60, 272, 100) finite=True ... zero-var series=0`, `natural length < 100: 13/60` (±2).

- [ ] **Step 3: Run Gemma random-init (≈5 min)**

Run: `uv run python scripts/01_capture.py --model google/gemma-3-4b-it --out results/gemma_random --random-init`
Expected: `(60, 272, 100)`, finite. If non-finite in bf16, fall back to `google/gemma-3-1b-it` random init (fp32 4B = 17 GB does not fit) and note it.

- [ ] **Step 4: Run Qwen (≈5 min)**

Run: `uv run python scripts/01_capture.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --out results/qwen`
Expected: `acts (60, 336, 100) finite=True`.

- [ ] **Step 5: Commit (script + small JSON only; .npy is gitignored)**

```bash
git add scripts/01_capture.py results/*/meta.json results/*/natural_len.json
git commit -m "feat: capture script; Gemma, random-init and Qwen captures"
```

🛑 **CHECK-IN 1:** report per-run shapes, finiteness, zero-variance counts, natural-length stats, runtimes, and a sample decoded output from Gemma and random-init Gemma. Wait for go-ahead.

---

### Task 7: ΦID matrices and ranking

**Files:**
- Create: `src/syncore/phiid.py`
- Test: `tests/test_phiid.py`

- [ ] **Step 1: Write the failing test**

`tests/test_phiid.py`:

```python
import numpy as np

from syncore.phiid import layer_profile, pair_syn_red, prompt_matrices, syn_red_matrices, syn_red_rank

T = 2000


def _common_driver(rng):
    z = np.zeros(T)
    for t in range(1, T):
        z[t] = 0.9 * z[t - 1] + rng.standard_normal()
    return z + 0.3 * rng.standard_normal(T), z + 0.3 * rng.standard_normal(T)


def _coupled_sum(rng):
    x, y = np.zeros(T), np.zeros(T)
    for t in range(1, T):
        s = 0.6 * (x[t - 1] + y[t - 1])
        x[t], y[t] = s + rng.standard_normal(), -s + rng.standard_normal()
    return x, y


def test_redundant_system():
    syn, red = pair_syn_red(*_common_driver(np.random.default_rng(0)))
    assert red > 0.5 and red > 10 * abs(syn)


def test_synergistic_system():
    syn, red = pair_syn_red(*_coupled_sum(np.random.default_rng(0)))
    assert syn > 0.1 and syn > red


def test_independent_system():
    rng = np.random.default_rng(0)
    syn, red = pair_syn_red(rng.standard_normal(T), rng.standard_normal(T))
    assert abs(syn) < 0.01 and abs(red) < 0.01


def test_constant_series_gives_nan():
    syn, red = pair_syn_red(np.ones(50), np.random.default_rng(0).standard_normal(50))
    assert np.isnan(syn) and np.isnan(red)


def test_prompt_matrices_symmetric_zero_diag():
    acts = np.random.default_rng(0).standard_normal((4, 60))
    S, R = prompt_matrices(acts)
    assert S.shape == (4, 4)
    np.testing.assert_allclose(S, S.T)
    np.testing.assert_allclose(R, R.T)
    assert (np.diag(S) == 0).all()


def test_syn_red_matrices_shapes():
    acts = np.random.default_rng(0).standard_normal((3, 5, 40))
    S_p, R_p = syn_red_matrices(acts, n_jobs=2)
    assert S_p.shape == R_p.shape == (3, 5, 5)


def test_rank_orders_synergistic_first():
    S = np.array([[0, 3, 3], [3, 0, 1], [3, 1, 0]], float)  # head 0 most synergistic
    R = np.array([[0, 1, 1], [1, 0, 3], [1, 3, 0]], float)  # heads 1, 2 most redundant
    r = syn_red_rank(S, R)
    assert r.argmax() == 0 and r[0] > r[1]


def test_layer_profile_minmax():
    rank = np.array([0, 0, 5, 5, 1, 1], float)  # 3 layers x 2 heads
    np.testing.assert_allclose(layer_profile(rank, n_layers=3), [0, 1, 0.2])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_phiid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'syncore.phiid'`

- [ ] **Step 3: Write implementation**

`src/syncore/phiid.py`:

```python
"""Pairwise ΦID (Gaussian, MMI) between head activation series → synergy/redundancy matrices and ranking."""
import numpy as np
from joblib import Parallel, delayed
from phyid.calculate import calc_PhiID
from scipy.stats import rankdata


def pair_syn_red(x, y, tau=1, redundancy="MMI") -> tuple[float, float]:
    """Time-averaged Syn→Syn and Red→Red atoms. NaN if either series is constant."""
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan, np.nan
    atoms, _ = calc_PhiID(np.asarray(x, float), np.asarray(y, float), tau, kind="gaussian", redundancy=redundancy)
    return float(np.mean(atoms["sts"])), float(np.mean(atoms["rtr"]))


def prompt_matrices(acts_p, tau=1, redundancy="MMI"):
    """acts_p: (N, T) → symmetric S, R of shape (N, N), zero diagonal."""
    n = acts_p.shape[0]
    S, R = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s, r = pair_syn_red(acts_p[i], acts_p[j], tau, redundancy)
            S[i, j] = S[j, i] = s
            R[i, j] = R[j, i] = r
    return S, R


def syn_red_matrices(acts, tau=1, redundancy="MMI", n_jobs=8):
    """acts: (P, N, T) → per-prompt S_p, R_p of shape (P, N, N)."""
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(prompt_matrices)(a, tau, redundancy) for a in acts)
    return np.stack([r[0] for r in res]), np.stack([r[1] for r in res])


def syn_red_rank(S, R):
    """S, R: (N, N) prompt-averaged. Per head: mean over its pairs; rank(syn) − rank(red). Higher = more synergistic."""
    n = S.shape[0]
    off = ~np.eye(n, dtype=bool)
    syn = np.array([np.nanmean(S[i, off[i]]) for i in range(n)])
    red = np.array([np.nanmean(R[i, off[i]]) for i in range(n)])
    return rankdata(syn) - rankdata(red)


def layer_profile(rank, n_layers):
    """Mean rank per layer, min-max normalised to [0, 1] (paper Fig 2c)."""
    m = np.asarray(rank, float).reshape(n_layers, -1).mean(1)
    return (m - m.min()) / (m.max() - m.min())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_phiid.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/syncore/phiid.py tests/test_phiid.py
git commit -m "feat: pairwise PhiID synergy/redundancy matrices and ranking"
```

---

### Task 8: ΦID script + runs

**Files:**
- Create: `scripts/02_phiid.py`

- [ ] **Step 1: Write the script**

`scripts/02_phiid.py`:

```python
"""Compute per-prompt S/R matrices, the synergy-redundancy rank, and robustness checks.

Usage: uv run python scripts/02_phiid.py --run results/gemma [--n-jobs 8] [--bench]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from syncore.phiid import layer_profile, pair_syn_red, syn_red_matrices, syn_red_rank

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--n-jobs", type=int, default=8)
ap.add_argument("--bench", action="store_true", help="time 200 pairs and extrapolate, then exit")
args = ap.parse_args()

run = Path(args.run)
acts = np.load(run / "acts.npy").astype(np.float64)
meta = json.load(open(run / "meta.json"))
P, N, T = acts.shape
n_pairs = N * (N - 1) // 2

if args.bench:
    t0 = time.time()
    for k in range(200):
        pair_syn_red(acts[0, k % N], acts[0, (k + 1) % N])
    per = (time.time() - t0) / 200
    print(f"{per * 1e3:.2f} ms/pair → est. {per * n_pairs * P / args.n_jobs / 60:.1f} min on {args.n_jobs} jobs")
    raise SystemExit

t0 = time.time()
S_p, R_p = syn_red_matrices(acts, n_jobs=args.n_jobs)
np.save(run / "S_p.npy", S_p)
np.save(run / "R_p.npy", R_p)
S, R = np.nanmean(S_p, 0), np.nanmean(R_p, 0)
rank = syn_red_rank(S, R)
np.save(run / "rank.npy", rank)
prof = layer_profile(rank, meta["n_layers"])
print(f"done in {(time.time() - t0) / 60:.1f} min; NaN pairs: {np.isnan(S_p).sum() // 2}")
print("layer profile:", np.round(prof, 2).tolist())

robust = {}
lens_f = run / "natural_len.json"
if lens_f.exists():
    keep = np.array(json.load(open(lens_f))) >= T
    r_sub = syn_red_rank(np.nanmean(S_p[keep], 0), np.nanmean(R_p[keep], 0))
    robust["natural_long_prompts"] = int(keep.sum())
    robust["spearman_rank_vs_long_only"] = float(spearmanr(rank, r_sub)[0])
half = np.arange(P) % 2 == 0
r_a = syn_red_rank(np.nanmean(S_p[half], 0), np.nanmean(R_p[half], 0))
r_b = syn_red_rank(np.nanmean(S_p[~half], 0), np.nanmean(R_p[~half], 0))
robust["spearman_split_half"] = float(spearmanr(r_a, r_b)[0])
json.dump(robust, open(run / "robustness.json", "w"), indent=1)
print("robustness:", robust)
```

- [ ] **Step 2: Benchmark, then run Gemma**

Run: `uv run python scripts/02_phiid.py --run results/gemma --bench`
Expected: ≈ `0.9 ms/pair → est. ~4-5 min on 8 jobs`. If estimate > 60 min, stop and report.

Run: `uv run python scripts/02_phiid.py --run results/gemma`
Expected: `layer profile:` 34 values, low at both ends, high in the middle; `robustness:` split-half and long-only Spearman values.

- [ ] **Step 3: Run the other two runs**

Run: `uv run python scripts/02_phiid.py --run results/gemma_random`
Run: `uv run python scripts/02_phiid.py --run results/qwen`

- [ ] **Step 4: Commit**

```bash
git add scripts/02_phiid.py results/*/robustness.json
git commit -m "feat: PhiID script; rankings for gemma, random-init, qwen"
```

---

### Task 9: Step 1 plots

**Files:**
- Create: `src/syncore/plots.py`, `scripts/03_plots_step1.py`
- Test: `tests/test_plots.py`

- [ ] **Step 1: Write the failing test**

`tests/test_plots.py`:

```python
import numpy as np

from syncore import plots


def test_step1_figures_write_files(tmp_path):
    rng = np.random.default_rng(0)
    S, R = rng.random((6, 6)), rng.random((6, 6))
    rank = rng.permutation(6).astype(float)
    plots.fig2a(S, R, tmp_path / "a.png")
    plots.fig2b(rank, n_layers=3, path=tmp_path / "b.png")
    plots.fig2c({"m1": (rank, 3), "m2": (rank[::-1], 3)}, tmp_path / "c.png")
    assert all((tmp_path / f).stat().st_size > 1000 for f in ["a.png", "b.png", "c.png"])


def test_step2_figures_write_files(tmp_path):
    fr = np.linspace(0, 0.4, 5)
    plots.fig4a(fr, np.linspace(0, 2, 5), np.vstack([np.linspace(0, 1, 5)] * 3), tmp_path / "d.png")
    plots.fig4b({"Baseline": [0.5], "Redundant core": [0.45], "Random": [0.4, 0.42], "Synergistic core": [0.3]},
                tmp_path / "e.png")
    assert (tmp_path / "d.png").stat().st_size > 1000 and (tmp_path / "e.png").stat().st_size > 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_plots.py -v`
Expected: FAIL with `ImportError: cannot import name 'plots'`

- [ ] **Step 3: Write implementation**

`src/syncore/plots.py`:

```python
"""Static matplotlib figures mirroring paper Figs 2a/2b/2c/4a/4b.
Palette (validated, light surface): blue #2a78d6, orange #eb6834, aqua #1baf7a, yellow #eda100;
diverging blue↔gray↔red for synergy-redundancy rank (red = synergistic, as in the paper)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from syncore.phiid import layer_profile  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "font.size": 10, "lines.linewidth": 2,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig2a(S, R, path):
    """Synergy and redundancy matrices between head pairs."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, M, title in zip(axes, [S, R], ["Synergy (Syn→Syn)", "Redundancy (Red→Red)"]):
        im = ax.imshow(M, cmap=SEQ, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Target head")
        ax.set_ylabel("Source head")
        fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, path)


def fig2b(rank, n_layers, path):
    """Heads (rows) × layers (columns) grid of synergy-redundancy rank."""
    grid = np.asarray(rank, float).reshape(n_layers, -1).T
    fig, ax = plt.subplots(figsize=(max(6, n_layers * 0.28), 3.2))
    im = ax.imshow(grid, cmap=DIV, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Attention head")
    fig.colorbar(im, ax=ax, label="Synergy − redundancy rank")
    _save(fig, path)


def fig2c(runs: dict, path):
    """runs: name -> (rank, n_layers). Normalised mean rank per layer vs normalised depth."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for color, (name, (rank, n_layers)) in zip(SERIES, runs.items()):
        prof = layer_profile(rank, n_layers)
        x = np.linspace(0, 1, n_layers)
        ax.plot(x, prof, color=color, marker="o", markersize=4, label=name)
        ax.annotate(name, (x[-1], prof[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK, fontsize=8, va="center")
    ax.set_xlabel("Normalised layer depth")
    ax.set_ylabel("Normalised synergy−redundancy rank")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def fig4a(fractions, syn_curve, random_curves, path):
    """syn_curve: (F,), random_curves: (n_orders, F). Mean ± std band for random."""
    mu, sd = random_curves.mean(0), random_curves.std(0)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(fractions, syn_curve, color=SERIES[1], label="Synergistic order")
    ax.plot(fractions, mu, color=SERIES[0], linestyle="--", label="Random order (mean ± sd)")
    ax.fill_between(fractions, mu - sd, mu + sd, color=SERIES[0], alpha=0.2, linewidth=0)
    ax.set_xlabel("Fraction of heads deactivated")
    ax.set_ylabel("Behaviour divergence (KL)")
    ax.legend(frameon=False)
    _save(fig, path)


def fig4b(acc: dict, path):
    """acc: condition -> list of accuracies (one per seed). Bars = mean, whisker = sd, value labels on top."""
    names = list(acc)
    means = [np.mean(acc[n]) * 100 for n in names]
    sds = [np.std(acc[n]) * 100 for n in names]
    colors = [MUTED, SERIES[0], SERIES[3], SERIES[1]][: len(names)]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bars = ax.bar(names, means, yerr=sds, color=colors, capsize=4, width=0.6, edgecolor=SURFACE, linewidth=2)
    for b, m in zip(bars, means):
        ax.annotate(f"{m:.0f}%", (b.get_x() + b.get_width() / 2, m), xytext=(0, 4),
                    textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.set_ylabel("Accuracy on MATH subset (%)")
    _save(fig, path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_plots.py -v`
Expected: 2 passed

- [ ] **Step 5: Write the Step 1 plot script**

`scripts/03_plots_step1.py`:

```python
"""Fig 2a/2b for Gemma and Fig 2c overlay of all runs. Writes to results/figures/."""
import json
from pathlib import Path

import numpy as np

from syncore import plots

out = Path("results/figures")
out.mkdir(parents=True, exist_ok=True)
g = Path("results/gemma")
meta = json.load(open(g / "meta.json"))
plots.fig2a(np.nanmean(np.load(g / "S_p.npy"), 0), np.nanmean(np.load(g / "R_p.npy"), 0), out / "fig2a_gemma.png")
plots.fig2b(np.load(g / "rank.npy"), meta["n_layers"], out / "fig2b_gemma.png")

runs = {}
for name, d in [("Gemma-3-4B-it", "gemma"), ("Gemma-3-4B random init", "gemma_random"),
                ("Qwen2.5-Math-1.5B", "qwen")]:
    p = Path("results") / d
    if (p / "rank.npy").exists():
        runs[name] = (np.load(p / "rank.npy"), json.load(open(p / "meta.json"))["n_layers"])
plots.fig2c(runs, out / "fig2c_profiles.png")
print("wrote", sorted(f.name for f in out.iterdir()))
```

Run: `uv run python scripts/03_plots_step1.py`
Expected: `wrote ['fig2a_gemma.png', 'fig2b_gemma.png', 'fig2c_profiles.png']`

- [ ] **Step 6: Look at the figures**

Open each PNG with the Read tool and check: labels readable, no overlaps, colorbars present, Fig 2b is 8 rows × 34 columns.

- [ ] **Step 7: Commit**

```bash
git add src/syncore/plots.py tests/test_plots.py scripts/03_plots_step1.py results/figures/*.png
git commit -m "feat: Step 1 figures (2a, 2b, 2c)"
```

🛑 **CHECK-IN 2 (Step 1 result):** show the three figures; report Gemma layer profile, whether the inverted U is present and absent in random init, Qwen profile, split-half ρ, long-only ρ, ΦID runtime. **Go/no-go decision for Step 2 with the user.**

---

### Task 10: Ablation context managers

**Files:**
- Create: `src/syncore/ablate.py`
- Test: `tests/test_ablate.py`

- [ ] **Step 1: Write the failing test**

`tests/test_ablate.py`:

```python
import torch

from syncore.ablate import noise_heads, zero_heads
from syncore.model import head_geometry

IDS = torch.randint(0, 1000, (2, 9), generator=torch.Generator().manual_seed(0))


def _logits(model):
    with torch.no_grad():
        return model(input_ids=IDS).logits


def test_zero_nothing_is_identity(tiny):
    model, _ = tiny
    h = head_geometry(model)
    base = _logits(model)
    with zero_heads(h, []):
        torch.testing.assert_close(_logits(model), base, rtol=0, atol=0)


def test_zero_heads_equals_zeroing_o_proj_columns(tiny):
    model, _ = tiny
    h = head_geometry(model)
    idx = [1, 6]  # layer 0 head 1, layer 1 head 2
    with zero_heads(h, idx):
        masked = _logits(model)
    saved = [h.o_projs[0].weight[:, 16:32].clone(), h.o_projs[1].weight[:, 32:48].clone()]
    with torch.no_grad():
        h.o_projs[0].weight[:, 16:32] = 0
        h.o_projs[1].weight[:, 32:48] = 0
    try:
        ref = _logits(model)
    finally:
        with torch.no_grad():
            h.o_projs[0].weight[:, 16:32] = saved[0]
            h.o_projs[1].weight[:, 32:48] = saved[1]
    torch.testing.assert_close(masked, ref)
    assert not torch.allclose(masked, _logits(model))


def test_noise_restores_and_touches_only_targets(tiny):
    model, _ = tiny
    h = head_geometry(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    with noise_heads(h, [5], alpha=1.0, seed=0):  # layer 1, head 1 → rows/cols 16:32
        q, o = h.q_projs[1].weight, h.o_projs[1].weight
        q0 = before["model.layers.1.self_attn.q_proj.weight"]
        o0 = before["model.layers.1.self_attn.o_proj.weight"]
        assert not torch.equal(q[16:32], q0[16:32]) and torch.equal(q[:16], q0[:16]) and torch.equal(q[32:], q0[32:])
        assert not torch.equal(o[:, 16:32], o0[:, 16:32]) and torch.equal(o[:, :16], o0[:, :16])
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]), n
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ablate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'syncore.ablate'`

- [ ] **Step 3: Write implementation**

`src/syncore/ablate.py`:

```python
"""Head interventions. Head index convention: g = layer * n_heads + head."""
from contextlib import contextmanager

import torch


@contextmanager
def zero_heads(heads, idx):
    """Zero the selected heads' attention outputs by masking their slice of the o_proj input."""
    mask = torch.ones(heads.n_layers, heads.n_heads)
    for g in idx:
        mask[g // heads.n_heads, g % heads.n_heads] = 0
    handles = []
    for layer, o in enumerate(heads.o_projs):
        if bool(mask[layer].all()):
            continue
        m = mask[layer].repeat_interleave(heads.head_dim).to(o.weight.device, o.weight.dtype)
        handles.append(o.register_forward_pre_hook(lambda mod, args, m=m: (args[0] * m,) + tuple(args[1:])))
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


@contextmanager
def noise_heads(heads, idx, alpha, seed=0):
    """Add N(0, (alpha * std(W))^2) to each selected head's q_proj rows and o_proj columns; restore on exit.
    std is computed once per weight matrix before any perturbation."""
    gen = torch.Generator().manual_seed(seed)
    D = heads.head_dim
    saved = []
    with torch.no_grad():
        stds = {}
        for g in sorted(idx):
            layer, h = divmod(g, heads.n_heads)
            q, o = heads.q_projs[layer].weight, heads.o_projs[layer].weight
            if layer not in stds:
                stds[layer] = (q.float().std().item(), o.float().std().item())
            sl = slice(h * D, (h + 1) * D)
            saved.append((q, o, sl, q[sl].clone(), o[:, sl].clone()))
            q[sl] += (alpha * stds[layer][0] * torch.randn(q[sl].shape, generator=gen)).to(q.device, q.dtype)
            o[:, sl] += (alpha * stds[layer][1] * torch.randn(o[:, sl].shape, generator=gen)).to(o.device, o.dtype)
    try:
        yield
    finally:
        with torch.no_grad():
            for q, o, sl, q_old, o_old in reversed(saved):
                q[sl] = q_old
                o[:, sl] = o_old
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ablate.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/syncore/ablate.py tests/test_ablate.py
git commit -m "feat: zero-head and noise-head ablation context managers"
```

---

### Task 11: Behaviour divergence

**Files:**
- Create: `src/syncore/divergence.py`
- Test: `tests/test_divergence.py`

- [ ] **Step 1: Write the failing test**

`tests/test_divergence.py`:

```python
import numpy as np
import torch

from syncore.capture import capture
from syncore.divergence import divergence_curve, teacher_forced_logprobs
from syncore.model import head_geometry

PROMPTS = ["Why do people wear sunglasses?", "Correct the error: The books is on the table.", "Hi there"]


def test_teacher_forcing_reproduces_greedy(tiny):
    model, tok = tiny
    gen = capture(model, tok, PROMPTS, n_tokens=8, batch_size=3)["tokens"]
    lp = teacher_forced_logprobs(model, tok, PROMPTS, torch.tensor(gen))
    assert lp.shape[:2] == (3, 8)
    agree = (lp.argmax(-1).numpy() == gen).mean()
    assert agree > 0.9


def test_curve_zero_at_zero_and_positive_after(tiny):
    model, tok = tiny
    h = head_geometry(model)
    gen = capture(model, tok, PROMPTS, n_tokens=8, batch_size=3)["tokens"]
    orders = {"syn": list(range(h.n_total)), "rand0": list(np.random.default_rng(0).permutation(h.n_total))}
    curves = divergence_curve(model, tok, PROMPTS, gen, orders, fractions=[0.0, 0.25, 0.5], batch_size=2)
    assert set(curves) == {"syn", "rand0"}
    for c in curves.values():
        assert len(c) == 3 and abs(c[0]) < 1e-5 and c[2] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_divergence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'syncore.divergence'`

- [ ] **Step 3: Write implementation**

`src/syncore/divergence.py`:

```python
"""Teacher-forced behaviour divergence: mean over prompts and tokens of KL(p_clean || p_ablated)."""
import numpy as np
import torch

from syncore.ablate import zero_heads
from syncore.capture import encode
from syncore.model import head_geometry


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_divergence.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/syncore/divergence.py tests/test_divergence.py
git commit -m "feat: teacher-forced behaviour divergence curve"
```

---

### Task 12: Divergence script + Gemma run (Fig 4a)

**Files:**
- Create: `scripts/04_divergence.py`

- [ ] **Step 1: Write the script**

`scripts/04_divergence.py`:

```python
"""Fig 4a: KL behaviour divergence vs fraction of heads zeroed, synergistic vs 5 random orders.

Usage: uv run python scripts/04_divergence.py --run results/gemma [--batch-size 5]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from syncore import plots
from syncore.divergence import divergence_curve
from syncore.model import load

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--batch-size", type=int, default=5)
args = ap.parse_args()

run = Path(args.run)
meta = json.load(open(run / "meta.json"))
rank = np.load(run / "rank.npy")
tokens = np.load(run / "tokens.npy")
N = len(rank)
fractions = np.round(np.arange(0, 0.401, 0.025), 3).tolist()
orders = {"synergistic": np.argsort(-rank).tolist()}
for s in range(5):
    orders[f"random{s}"] = np.random.default_rng(s).permutation(N).tolist()

model, tok = load(meta["model"])
t0 = time.time()
curves = divergence_curve(model, tok, meta["prompts"], tokens, orders, fractions, batch_size=args.batch_size)
json.dump({"fractions": fractions, "curves": curves}, open(run / "divergence.json", "w"), indent=1)
rand = np.array([curves[f"random{s}"] for s in range(5)])
plots.fig4a(np.array(fractions), np.array(curves["synergistic"]), rand, Path("results/figures/fig4a_gemma.png"))
print(f"done in {(time.time() - t0) / 60:.1f} min")
for f, sv, rm, rs in zip(fractions, curves["synergistic"], rand.mean(0), rand.std(0)):
    print(f"  f={f:.3f}  syn={sv:.3f}  rand={rm:.3f}±{rs:.3f}")
```

- [ ] **Step 2: Smoke-run on 1 batch worth of timing, then full run (≈15-40 min)**

Run: `uv run python scripts/04_divergence.py --run results/gemma`
Expected: prints a table with `syn` ≥ `rand` mean for most fractions > 0; KL at f=0 is 0. If memory pressure (swap) appears, rerun with `--batch-size 2`.

- [ ] **Step 3: Look at `results/figures/fig4a_gemma.png`** with the Read tool (labels, band visible).

- [ ] **Step 4: Commit**

```bash
git add scripts/04_divergence.py results/gemma/divergence.json results/figures/fig4a_gemma.png
git commit -m "feat: Fig 4a behaviour divergence for Gemma"
```

🛑 **CHECK-IN 3:** show Fig 4a and the table; does the synergistic curve sit above the random band? Wait for go-ahead.

---

### Task 13: MATH evaluation

**Files:**
- Create: `src/syncore/math_eval.py`
- Test: `tests/test_math_eval.py`

- [ ] **Step 1: Write the failing test**

`tests/test_math_eval.py`:

```python
from syncore.math_eval import evaluate, is_correct, last_boxed, load_subset, normalize


def test_last_boxed_nested_and_last():
    assert last_boxed(r"so \boxed{1} then \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert last_boxed(r"\boxed{\left( 3, \frac{\pi}{2} \right)}") == r"\left( 3, \frac{\pi}{2} \right)"
    assert last_boxed("no answer here") is None
    assert last_boxed(r"\boxed{unclosed") is None


def test_normalize_equivalences():
    assert normalize(r"\left( 3, \frac{\pi}{2} \right)") == normalize(r"(3,\frac{\pi}{2})")
    assert normalize(r"\dfrac{1}{2}") == normalize(r"\frac{1}{2}")
    assert normalize(r"10\%") == normalize("10")
    assert normalize(r"5^\circ") == normalize("5")
    assert normalize(r"\text{(C)}") == normalize("(C)")
    assert normalize("7.") == "7"


def test_is_correct():
    assert is_correct(r"The answer is \boxed{\dfrac{1}{2}}.", r"\frac{1}{2}")
    assert not is_correct(r"\boxed{3}", "4")
    assert not is_correct("no box", "4")


def test_load_subset_stratified():
    sub = load_subset(per_level=2, seed=0)
    assert len(sub) == 10
    assert sorted(p["level"] for p in sub) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert load_subset(per_level=2, seed=0) == sub


def test_evaluate_runs_on_tiny(tiny):
    model, tok = tiny
    probs = [{"problem": "What is 1+1?", "answer": "2", "level": 1}]
    res = evaluate(model, tok, probs, max_new_tokens=5, batch_size=1)
    assert set(res) == {"accuracy", "correct", "outputs"} and 0.0 <= res["accuracy"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_math_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'syncore.math_eval'`

- [ ] **Step 3: Write implementation**

`src/syncore/math_eval.py`:

```python
"""MATH-500 subset evaluation with \\boxed{} answer extraction."""
import re

import numpy as np
from datasets import load_dataset

from syncore.capture import generate

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def load_subset(per_level=30, seed=0) -> list[dict]:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rng = np.random.default_rng(seed)
    out = []
    for level in range(1, 6):
        idx = [i for i, lv in enumerate(ds["level"]) if lv == level]
        for i in sorted(rng.choice(idx, size=per_level, replace=False).tolist()):
            r = ds[i]
            out.append({"problem": r["problem"], "answer": r["answer"], "level": r["level"]})
    return out


def last_boxed(text: str):
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j]
    return None


def normalize(s: str) -> str:
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s.strip())
    for a, b in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""),
                 ("dfrac", "frac"), ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""),
                 ("\\$", ""), ("$", ""), ("\\%", ""), ("%", "")]:
        s = s.replace(a, b)
    return s.replace(" ", "").rstrip(".")


def is_correct(output: str, gold: str) -> bool:
    pred = last_boxed(output)
    return pred is not None and normalize(pred) == normalize(gold)


def evaluate(model, tok, problems, max_new_tokens=512, batch_size=10) -> dict:
    outputs = []
    for i in range(0, len(problems), batch_size):
        batch = [p["problem"] + SUFFIX for p in problems[i:i + batch_size]]
        _, gen = generate(model, tok, batch, max_new_tokens, ban_eos=False)
        outputs += tok.batch_decode(gen, skip_special_tokens=True)
        print(f"  math {min(i + batch_size, len(problems))}/{len(problems)}", flush=True)
    correct = [is_correct(o, p["answer"]) for o, p in zip(outputs, problems)]
    return {"accuracy": float(np.mean(correct)), "correct": correct, "outputs": outputs}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_math_eval.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/syncore/math_eval.py tests/test_math_eval.py
git commit -m "feat: MATH-500 subset evaluation"
```

---

### Task 14: MATH script — calibration (α)

**Files:**
- Create: `scripts/05_math.py`

- [ ] **Step 1: Write the script**

`scripts/05_math.py`:

```python
"""Fig 4b: MATH accuracy under Gaussian weight noise on 25% of heads.

Calibrate: uv run python scripts/05_math.py --run results/gemma --calibrate
Evaluate:  uv run python scripts/05_math.py --run results/gemma --alpha 1.0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from syncore import plots
from syncore.ablate import noise_heads
from syncore.math_eval import evaluate, load_subset
from syncore.model import head_geometry, load

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--calibrate", action="store_true")
ap.add_argument("--alpha", type=float)
ap.add_argument("--frac", type=float, default=0.25)
ap.add_argument("--batch-size", type=int, default=10)
args = ap.parse_args()

run = Path(args.run)
meta = json.load(open(run / "meta.json"))
rank = np.load(run / "rank.npy")
model, tok = load(meta["model"])
h = head_geometry(model)
k = round(args.frac * h.n_total)
order = np.argsort(-rank)
conds = {"Synergistic core": order[:k].tolist(), "Redundant core": order[-k:].tolist()}
for s in range(3):
    conds[f"Random{s}"] = np.random.default_rng(100 + s).choice(h.n_total, k, replace=False).tolist()


def run_cond(problems, idx, alpha, seed=0):
    if idx is None:
        return evaluate(model, tok, problems, batch_size=args.batch_size)
    with noise_heads(h, idx, alpha, seed=seed):
        return evaluate(model, tok, problems, batch_size=args.batch_size)


t0 = time.time()
if args.calibrate:
    probs = load_subset(per_level=10, seed=1)  # 50 problems, disjoint seed from eval set
    res = {"baseline": run_cond(probs, None, 0)["accuracy"]}
    for alpha in [0.5, 1.0, 2.0]:
        res[f"random_a{alpha}"] = run_cond(probs, conds["Random0"], alpha)["accuracy"]
        print(res, f"({(time.time() - t0) / 60:.0f} min)", flush=True)
    json.dump(res, open(run / "math_calibration.json", "w"), indent=1)
    raise SystemExit

assert args.alpha is not None, "pass --alpha (from calibration) or --calibrate"
probs = load_subset(per_level=30, seed=0)
results = {"alpha": args.alpha, "baseline": run_cond(probs, None, 0)}
print(f"baseline {results['baseline']['accuracy']:.3f}", flush=True)
for name, idx in conds.items():
    results[name] = run_cond(probs, idx, args.alpha)
    print(f"{name} {results[name]['accuracy']:.3f}  ({(time.time() - t0) / 60:.0f} min)", flush=True)
    json.dump(results, open(run / "math.json", "w"), indent=1)  # checkpoint after each condition

acc = {"Baseline": [results["baseline"]["accuracy"]],
       "Redundant core": [results["Redundant core"]["accuracy"]],
       "Random": [results[f"Random{s}"]["accuracy"] for s in range(3)],
       "Synergistic core": [results["Synergistic core"]["accuracy"]]}
plots.fig4b(acc, Path("results/figures/fig4b_gemma.png"))
print({k: round(float(np.mean(v)), 3) for k, v in acc.items()})
```

- [ ] **Step 2: Run calibration (≈1-1.5 h)**

Run: `uv run python scripts/05_math.py --run results/gemma --calibrate`
Expected: `results/gemma/math_calibration.json` with baseline (≈0.5-0.6) and random-noise accuracies at α ∈ {0.5, 1, 2}.

Choose α = smallest value where `random_aα` is clearly below baseline (≥ 5 points) but above ~0.1 (not total collapse).

- [ ] **Step 3: Commit**

```bash
git add scripts/05_math.py results/gemma/math_calibration.json
git commit -m "feat: MATH ablation script and noise calibration"
```

🛑 **CHECK-IN 4:** report calibration table and proposed α; confirm before the long (multi-hour) eval run.

---

### Task 15: MATH evaluation run (Fig 4b)

- [ ] **Step 1: Run overnight (≈4-6 h; checkpoints after each condition)**

Run in background: `uv run python scripts/05_math.py --run results/gemma --alpha <chosen α> > results/gemma/math.log 2>&1`
Expected final line: dict with accuracies for Baseline, Redundant core, Random, Synergistic core.

- [ ] **Step 2: Look at `results/figures/fig4b_gemma.png`** with the Read tool.

- [ ] **Step 3: Spot-check scoring** — print 5 outputs marked wrong and confirm they are genuinely wrong (not extraction failures):

```bash
uv run python -c "
import json; r=json.load(open('results/gemma/math.json'))['baseline']
from syncore.math_eval import load_subset, last_boxed
ps=load_subset(30,0); bad=[i for i,c in enumerate(r['correct']) if not c][:5]
for i in bad: print('GOLD', ps[i]['answer'], '| PRED', last_boxed(r['outputs'][i]))"
```

If extraction failures dominate, fix `normalize()` (add a test case first), then re-score from saved outputs without regenerating.

- [ ] **Step 4: Commit**

```bash
git add results/figures/fig4b_gemma.png
git commit -m "feat: Fig 4b MATH ablation for Gemma"
```

(`results/gemma/math.json` contains full outputs; commit only if < 5 MB.)

🛑 **CHECK-IN 5 (Step 2 result / go-no-go for GRPO):** Fig 4a + 4b vs paper values (baseline ≈58%, synergistic ≈28%, random ≈44%, redundant ≈52%); success criteria from the spec; recommendation on spending GPU budget.
