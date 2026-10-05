"""Vectorised MMI ΦID (Gaussian or discrete): time-averaged Syn→Syn and Red→Red for all head pairs at once.

Exact equivalent of averaging phyid.calc_PhiID(kind="gaussian", redundancy="MMI") over time:
- with MMI, every quantity is linear in the local MIs, and the min-selections use time-averaged MIs,
  so averaged atoms = atom equations applied to averaged MIs;
- for a Gaussian, the time-averaged local MI is 0.5 * (logdet C_A + logdet C_B - logdet C_AB) on the
  correlation matrix (the sample-size constants cancel);
- for binary data (discrete), the time-averaged local entropy is the plug-in entropy of the empirical distribution.
Verified against phyid in tests/test_phiid_fast.py.
"""
import numpy as np

# knowns -> atoms matrix, copied from phyid.calculate._get_atoms_four_vec.
# Rows: [rtr, Rxyta, Rxytb, Rxytab, Rabtx, Rabty, Rabtxy, Ixta, Ixtb, Iyta, Iytb, Ixyta, Ixytb, Ixtab, Iytab, Ixytab]
# Columns: atoms in phyid order [rtr, rtx, rty, rts, xtr, xtx, xty, xts, ytr, ytx, yty, yts, str, stx, sty, sts]
_KNOWNS_TO_ATOMS = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
], dtype=float)
_STS_FROM_KNOWNS = np.linalg.inv(_KNOWNS_TO_ATOMS)[15]  # Syn→Syn is the last atom


def _standardise(z):
    """Rows demeaned and scaled to unit std (ddof=1); constant rows become NaN."""
    z = z - z.mean(-1, keepdims=True)
    sd = z.std(-1, ddof=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 1e-12, z / sd, np.nan)


def _logdet(C, idx):
    """log det of the idx-submatrix of batched (..., 4, 4) C; NaN where not positive definite."""
    sub = C[..., idx, :][..., :, idx]
    with np.errstate(invalid="ignore"):  # NaN entries (constant series) are expected
        sign, ld = np.linalg.slogdet(sub)
    return np.where(sign > 0, ld, np.nan)


_SUBSETS = [(0,), (1,), (2,), (3,), (0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3),
            (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (0, 1, 2, 3)]  # 0=x_past 1=y_past 2=x_future 3=y_future


def _gaussian_halflogdets(acts_p, tau):
    """Per pair: {subset: 0.5 * log det of its correlation submatrix} (entropy up to constants that cancel in MIs)."""
    Zp, Zf = _standardise(acts_p[:, :-tau]), _standardise(acts_p[:, tau:])
    n = Zp.shape[1]
    Cpp, Cff, Cpf = Zp @ Zp.T / (n - 1), Zf @ Zf.T / (n - 1), Zp @ Zf.T / (n - 1)  # Cpf[i, j] = corr(i_past, j_future)

    # Per pair (i = x, j = y): correlation matrix of [x_past, y_past, x_future, y_future]
    N = acts_p.shape[0]
    C = np.empty((N, N, 4, 4))
    dp, df, auto = np.diag(Cpp), np.diag(Cff), np.diag(Cpf)
    C[..., 0, 0], C[..., 1, 1] = dp[:, None], dp[None, :]
    C[..., 2, 2], C[..., 3, 3] = df[:, None], df[None, :]
    C[..., 0, 1] = C[..., 1, 0] = Cpp
    C[..., 0, 2] = C[..., 2, 0] = auto[:, None]
    C[..., 0, 3] = C[..., 3, 0] = Cpf
    C[..., 1, 2] = C[..., 2, 1] = Cpf.T
    C[..., 1, 3] = C[..., 3, 1] = auto[None, :]
    C[..., 2, 3] = C[..., 3, 2] = Cff
    return {k: 0.5 * _logdet(C, list(k)) for k in _SUBSETS}


def _binary_entropies(acts_p, tau):
    """Per pair: {subset: plug-in entropy (bits) of the binarised variables}. Each window is binarised at its own
    mean, as phyid does; the time-averaged local binary entropy equals the plug-in entropy."""
    past, fut = acts_p[:, :-tau], acts_p[:, tau:]
    Bp, Bf = past > past.mean(-1, keepdims=True), fut > fut.mean(-1, keepdims=True)
    N, n = Bp.shape
    U = {(a, c): ((Bp == a) & (Bf == c)).astype(float) for a in (0, 1) for c in (0, 1)}  # head's (past, future) state
    P = np.empty((N, N, 2, 2, 2, 2))  # joint distribution over axes [x_past, y_past, x_future, y_future]
    for a in (0, 1):
        for c in (0, 1):
            for b in (0, 1):
                for d in (0, 1):
                    P[:, :, a, b, c, d] = U[a, c] @ U[b, d].T / n
    const = (past.std(-1) < 1e-12) | (fut.std(-1) < 1e-12)
    nan_pairs = const[:, None] | const[None, :]  # match pair_syn_red: constant series -> NaN

    def H(keep):
        m = P.sum(axis=tuple(2 + v for v in range(4) if v not in keep))
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.where(m > 0, m * np.log2(m), 0.0).reshape(N, N, -1).sum(-1)
        return np.where(nan_pairs, np.nan, h)
    return {k: H(k) for k in _SUBSETS}


def prompt_matrices_fast(acts_p, tau=1, kind="gaussian"):
    """acts_p: (N, T) → S, R of shape (N, N): time-averaged Syn→Syn and Red→Red (MMI), zero diagonal."""
    h = _gaussian_halflogdets(acts_p, tau) if kind == "gaussian" else _binary_entropies(acts_p, tau)
    mi = lambda a, b: h[a] + h[b] - h[tuple(sorted(a + b))]
    I_xta, I_xtb, I_yta, I_ytb = mi((0,), (2,)), mi((0,), (3,)), mi((1,), (2,)), mi((1,), (3,))
    I_xyta, I_xytb = mi((0, 1), (2,)), mi((0, 1), (3,))
    I_xtab, I_ytab, I_xytab = mi((0,), (2, 3)), mi((1,), (2, 3)), mi((0, 1), (2, 3))

    rtr = np.minimum(np.minimum(I_xta, I_xtb), np.minimum(I_yta, I_ytb))
    knowns = np.stack([
        rtr,
        np.minimum(I_xta, I_yta), np.minimum(I_xtb, I_ytb), np.minimum(I_xtab, I_ytab),
        np.minimum(I_xta, I_xtb), np.minimum(I_yta, I_ytb), np.minimum(I_xyta, I_xytb),
        I_xta, I_xtb, I_yta, I_ytb, I_xyta, I_xytb, I_xtab, I_ytab, I_xytab,
    ], axis=-1)
    S, R = knowns @ _STS_FROM_KNOWNS, rtr.copy()
    np.fill_diagonal(S, 0.0)
    np.fill_diagonal(R, 0.0)
    return S, R


def syn_red_matrices_fast(acts, tau=1, kind="gaussian"):
    """acts: (P, N, T) → per-prompt S_p, R_p of shape (P, N, N). MMI redundancy; kind 'gaussian' or 'discrete'."""
    res = [prompt_matrices_fast(np.asarray(a, float), tau, kind) for a in acts]
    return np.stack([r[0] for r in res]), np.stack([r[1] for r in res])
