# synergistic-cores

Feasibility reproduction of arXiv:2601.06851 (steps 1-2) on an M1 Pro. Spec and plan in `docs/superpowers/`.

## Troubleshooting

`ModuleNotFoundError: No module named syncore`: macOS flags the venv's `.pth` files hidden (again after each `uv run` reinstall), and Python 3.13 skips hidden `.pth` files. Run scripts with `PYTHONPATH=src` (tests set this via pytest config):

```bash
PYTHONPATH=src uv run python scripts/01_capture.py --model google/gemma-3-4b-it --out results/gemma
```
