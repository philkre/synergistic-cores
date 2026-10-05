# synergistic-cores

Feasibility reproduction of arXiv:2601.06851 (synergistic core in LLMs) on an M1 Pro 16 GB.
Spec + plan: `docs/superpowers/`. Branch `steps12`.

## Done
- **Pipeline** (`src/syncore/`, 30 tests): per-head attention-output norms during 100-token greedy generation
  (60 Appendix-A prompts) → pairwise ΦID (phyid, Gaussian, MMI; Syn→Syn, Red→Red) → synergy–redundancy rank
  (re-ranked 1..N) → per-layer profile. Ablation + KL-divergence code built, not yet run.
- **Runs:** Gemma-3-4B-it (bf16), Qwen2.5-Math-1.5B, Gemma random-init (2 controls), Gemma 5× sampled.
- **Step 1 result:** Gemma shows the broad inverted U (smoothed r≈0.7 vs paper), but layer detail differs
  (r≈0.4); Qwen shows none. Split-half reliability r≈0.75.
- **Ruled out:** per-layer averaging, pooled vs per-prompt ΦID, greedy vs sampled decoding.
  Root fact: syn and red correlate ~0.8 across heads → their rank difference is a noisy residual.

## Next
Test alternative explanations for the Fig 2c gap (ΦID variants, raw prompts, post-W_O norms), then Step 2 ablations.

## Run
`PYTHONPATH=src uv run python scripts/0X_*.py ...` (PYTHONPATH: macOS hides the venv `.pth` files, Python 3.13 skips them).
