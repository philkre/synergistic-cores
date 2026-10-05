# synergistic-cores

Feasibility reproduction of arXiv:2601.06851 (steps 1-2) on an M1 Pro. Spec and plan in `docs/superpowers/`.

## Troubleshooting

`ModuleNotFoundError: No module named syncore` when running scripts: macOS sometimes flags the venv `.pth` files as hidden, and Python 3.13 skips hidden `.pth` files. Fix:

```bash
chflags nohidden .venv/lib/python3.13/site-packages/*.pth
```
