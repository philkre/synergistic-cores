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
