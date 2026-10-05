# Exploration (pre-plan)

- `smoke.py MODEL [bf16|fp32]`: load on MPS, generate 100 tokens, check per-head o_proj-input norms. Gemma-3-4B-it bf16: 8.75 GB, 4.2 tok/s.
- `lengths.py MODEL OUT.json`: natural response length (EOS allowed, cap 150) for the 60 prompts. Result in `lengths_gemma.json`: 13/60 end before 100 tokens.
- `diag_profile.py RUN`: per-layer synergy/redundancy decomposition. Finding: syn and red correlate ~0.8 across heads (both track temporal persistence), so their rank difference is a noisy residual.
- `diag_pooled.py RUN`: pooled-over-prompts PhiID estimator. Same profile shape as per-prompt averaging.
- `compare_sampling.py`: greedy vs 5 sampled runs vs paper Fig 2c (eyeballed). Output in `compare_sampling_output.txt`: sampling/averaging does not explain the paper curve; greedy vs 5-run average r=0.88.
