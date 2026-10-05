# Exploration (pre-plan)

- `smoke.py MODEL [bf16|fp32]`: load on MPS, generate 100 tokens, check per-head o_proj-input norms. Gemma-3-4B-it bf16: 8.75 GB, 4.2 tok/s.
- `lengths.py MODEL OUT.json`: natural response length (EOS allowed, cap 150) for the 60 prompts. Result in `lengths_gemma.json`: 13/60 end before 100 tokens.
