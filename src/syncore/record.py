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
