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
SEQ_RED = LinearSegmentedColormap.from_list("seq_red", ["#fbdcdb", "#ef8f8e", "#e34948", "#8c1d1d"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "font.size": 10, "lines.linewidth": 2,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig2a(S, R, path):
    """Synergy (red) and redundancy (blue) matrices between head pairs, as in the paper."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, M, title, cmap in zip(axes, [S, R], ["Synergy (Syn→Syn)", "Redundancy (Red→Red)"], [SEQ_RED, SEQ]):
        im = ax.imshow(M, cmap=cmap, interpolation="nearest")
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


def fig2ab(S, R, rank, n_layers, title, path):
    """Left third: synergy (red) above redundancy (blue) head-pair matrices. Right two thirds: heads × layers rank map."""
    fig = plt.figure(figsize=(14, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], wspace=0.18, hspace=0.32)
    for row, (M, name, cmap) in enumerate([(S, "Synergy (Syn→Syn)", SEQ_RED), (R, "Redundancy (Red→Red)", SEQ)]):
        ax = fig.add_subplot(gs[row, 0])
        im = ax.imshow(M, cmap=cmap, interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("Source head")
        if row == 1:
            ax.set_xlabel("Target head")
        fig.colorbar(im, ax=ax, shrink=0.85)
    ax = fig.add_subplot(gs[:, 1])
    grid = np.asarray(rank, float).reshape(n_layers, -1).T
    im = ax.imshow(grid, cmap=DIV, aspect="auto", interpolation="nearest")
    ax.set_title("Synergy−redundancy rank per head", fontsize=10)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Attention head")
    fig.colorbar(im, ax=ax, label="Rank (red = synergistic, blue = redundant)", shrink=0.9)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2c(runs: dict, path):
    """runs: name -> (rank, n_layers). Normalised mean rank per layer vs normalised depth."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for color, (name, (rank, n_layers)) in zip(SERIES, runs.items()):
        ax.plot(np.linspace(0, 1, n_layers), layer_profile(rank, n_layers), color=color, marker="o", markersize=4,
                label=name)
    ax.set_xlabel("Normalised layer depth")
    ax.set_ylabel("Normalised synergy−redundancy rank")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2)
    _save(fig, path)


def fig4a(fractions, syn_curve, random_curves, path, red_curve=None):
    """syn_curve, red_curve: (F,); random_curves: (n_orders, F) shown as mean ± sd band.
    Colours match the head maps: red = synergistic, blue = redundant, grey = random."""
    mu, sd = random_curves.mean(0), random_curves.std(0)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.fill_between(fractions, mu - sd, mu + sd, color=MUTED, alpha=0.25, linewidth=0)
    ax.plot(fractions, mu, color=MUTED, linestyle="--", label=f"Random order (mean ± sd, n={len(random_curves)})")
    if red_curve is not None:
        ax.plot(fractions, red_curve, color="#2a78d6", marker="o", markersize=3, label="Redundant heads first")
    ax.plot(fractions, syn_curve, color="#e34948", marker="o", markersize=3, label="Synergistic heads first")
    ax.set_xlabel("Fraction of heads deactivated")
    ax.set_ylabel("Behaviour divergence (KL)")
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2)
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
