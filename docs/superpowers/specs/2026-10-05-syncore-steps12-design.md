# Synergistic core reproduction — Steps 1 & 2 (M1 Pro, local)

**Date:** 2026-10-05
**Paper:** Urbina-Rodriguez et al., "A Brain-like Synergistic Core in LLMs Drives Behaviour and Learning", arXiv:2601.06851 (`Urbina-Rodriguez26.pdf`)
**Purpose:** Feasibility check. Decide whether the synergistic-core effect holds before spending GPU money on the GRPO experiment (Step 3, out of scope here). Speed over polish.

## Scope

In scope:
- **Step 1 — Synergy map:** reproduce Fig 2a, Fig 2b and the Gemma curve of Fig 2c for Gemma-3-4B-it; random-init control.
- **Step 2 — Ablations:** reproduce the Gemma curve of Fig 4a (behaviour divergence) and the Gemma bars of Fig 4b (MATH accuracy under perturbation).
- **Secondary:** Step 1 ranking only for Qwen2.5-Math-1.5B (input for a later GRPO run).

Out of scope: GRPO/SFT (Fig 5), Pythia training dynamics (Fig 3a), graph metrics (Fig 3c), other model families (Qwen3-8B, Llama-3.1-8B, DeepSeek-V2-Lite do not fit in 16 GB unquantised).

## Hardware and environment (verified 2026-10-05)

- Apple M1 Pro, 16 GB unified memory, macOS 27, PyTorch MPS.
- `uv` project, Python 3.13, torch 2.14.1, transformers 5.18.
- Smoke test results:
  - `google/gemma-3-4b-it`, bf16, MPS: 8.75 GB peak, 4.2 tok/s unbatched, ~30 tok/s effective at batch 10. Loads as `Gemma3ForConditionalGeneration`; 34 layers × 8 heads, head_dim 256.
  - `Qwen/Qwen2.5-Math-1.5B-Instruct`, bf16: 3.3 GB, 8.6 tok/s unbatched. 28 layers × 12 heads.
- No quantisation. bf16 fits; quantisation would complicate weight-noise ablations.
- Gemma is gated; HF login done on this machine.

## Method decisions

| Item | Decision | Source / rationale |
|---|---|---|
| Unit | Attention head | Paper |
| Activation | L2 norm of per-head attention output `softmax(qKᵀ/√d)V`, i.e. input to `o_proj` split into `(n_heads, head_dim)` | Paper Methods eq. for a(h_i, t) |
| Capture point | `register_forward_pre_hook` on every `self_attn.o_proj`, last position only; per-forward-pass flush via model-level pre/post hooks (not by patching `model.forward` — breaks `generate()` kwarg validation) | Smoke test |
| Timesteps | 100 generated tokens; include the prefill pass's last position (it produces token 1) → exactly 100 samples | Paper |
| Decoding | Greedy, chat template, batch by category (10 prompts), left padding | Speed |
| Early EOS | **Ban EOS** (`suppress_tokens` / `bad_words_ids` for all EOS ids) so every prompt yields 100 real tokens. Robustness check: forced `min_new_tokens=100` without banning. | 13/60 prompts end before 100 tokens naturally (syntax, numerical) |
| Prompts | 60 prompts, 6 categories × 10, verbatim from Appendix A | Paper |
| ΦID | `phyid.calculate.calc_PhiID(src, trg, tau=1, kind="gaussian", redundancy="MMI")`, from `git+https://github.com/Imperial-MIND-lab/integrated-info-decomp` | Paper cites this implementation |
| Redundancy function | MMI (default). Paper text says I_min, but Luppi et al. and phyid use MMI for Gaussian data. CCS available as a flag. | Ambiguity in paper |
| Synergy / redundancy | Mean over time of `sts` (Syn→Syn) and `rtr` (Red→Red) atoms, per prompt, then mean over prompts → `S[N,N]`, `R[N,N]` | Paper |
| Pairs | All unordered head pairs (272 heads → 36,856 pairs); run per prompt; parallelised with joblib | — |
| Ranking | Per head: mean of S and of R over all pairs involving it; `syn_red_rank = rank(S_mean) − rank(R_mean)` | Paper |

## Components

```
synergistic-cores/
  pyproject.toml
  src/syncore/
    model.py       load model (id, dtype) on MPS; head geometry; o_proj list
    prompts.py     60 prompts from Appendix A
    capture.py     batched generation with EOS ban + head-norm hooks → acts[P, N, T] (+ generated token ids)
    phiid.py       pairwise ΦID → S, R matrices; ranking
    ablate.py      context managers: zero heads (activation mask at o_proj input); noise heads (Q rows + O cols)
    divergence.py  teacher-forced KL behaviour divergence
    math_eval.py   MATH subset loading, batched generation, \boxed{} extraction + answer matching
  scripts/
    01_capture.py  02_phiid.py  03_divergence.py  04_math.py  plots.py
  tests/           sanity tests (see below)
  results/<model>/ acts.npy, tokens.json, S.npy, R.npy, rank.json, divergence.json, math.json, figures/*.png
```

Each script reads from and writes to `results/<model>/`; steps are cached and can be rerun independently. Model id is a CLI arg everywhere. Large arrays are gitignored; figures and small JSON summaries are committed.

## Step 1 — Synergy map

1. `01_capture.py`: generate 100 tokens for all 60 prompts, save `acts[60, N, 100]` and token ids.
2. `02_phiid.py`: compute S, R; save matrices and ranking. Benchmark one pair first to confirm the runtime estimate (~1 h on 8 cores for Gemma).
3. `plots.py`: Fig 2a (S and R heatmaps), Fig 2b (layer × head rank heatmap), Fig 2c (mean rank per layer, min-max normalised, x = normalised depth).
4. **Random-init control:** same pipeline on `from_config` Gemma-3-4B weights (random init, same seed).
5. Qwen2.5-Math-1.5B: steps 1–3 only.

## Step 2 — Ablations (Gemma-3-4B-it)

**Behaviour divergence (Fig 4a):**
- Reference: clean model's 100-token generations from Step 1 (EOS-banned).
- For fraction f ∈ {0, 0.025, …, 0.40}: zero the top-f·N heads in synergy-rank order (and, separately, in each of 5 random orders) by masking their slice of the `o_proj` input.
- Teacher-force the clean tokens through the ablated model; KL(p_clean ‖ p_ablated) per token, averaged over 100 tokens and 60 prompts.
- Output: curve + random band (mean ± std over 5 orders).

**MATH accuracy (Fig 4b):**
- Conditions: baseline; top-25% synergistic heads; top-25% redundant heads (bottom of rank); 3 random 25% subsets.
- Perturbation: add Gaussian noise `σ = α·std(W)` to the head's rows of `q_proj` and columns of `o_proj` (fixed seed). Weights restored after each condition.
- Calibrate α ∈ {0.5, 1, 2} on 50 problems; pick the smallest α where the random condition drops visibly below baseline.
- Eval set: 150 MATH test problems stratified across levels 1–5 (fixed seed), greedy, max 512 new tokens, batched; score by `\boxed{}` extraction + normalised string match.
- Paper reference values (Gemma-3-4B-IT): baseline ≈ 58%, synergistic ≈ 28%, random ≈ 44%, redundant ≈ 52%.

## Success criteria (go / no-go for GRPO)

- **Step 1:** mean rank per layer shows an inverted U (middle layers > early and late), absent in random init.
- **Step 2:** synergistic-order KL curve lies above the random band; MATH accuracy drop ordering synergistic > random > redundant.
- Both hold → worth spending GPU budget on GRPO with Qwen2.5-Math-1.5B ranking.

## Sanity tests

- Hook norms equal a manual recompute of `o_proj` input for one layer.
- Zeroing all heads in one layer changes the logits; zeroing none leaves them bit-identical.
- Noise context manager restores weights exactly.
- `phyid` on synthetic data: XOR-like Gaussian system gives syn > red; copy system gives red > syn.
- EOS ban: all 60 captures have exactly 100 tokens and no EOS id.

## Risks

- MPS numeric quirks → fallback: run capture on CPU (slower, still feasible for 60 × 100 tokens).
- ΦID runtime higher than estimated → subsample pairs within layers or reduce prompts per category; benchmark first.
- Unknown paper details (noise σ, exact ΦID settings, EOS handling) → documented choices above; robustness checks where cheap.
