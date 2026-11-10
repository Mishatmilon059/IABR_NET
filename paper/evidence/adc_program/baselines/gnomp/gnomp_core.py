"""GNOMP (Zhu, Zhang, Zhang, Fang, Qu, IEEE TAES 2026; arXiv:2307.00491) for few-bit observations.

Generic over the atom model: Atoms1D (the paper's 1-D line spectrum) and Atoms2D (our 16x16 beamspace, two
angles per path). The likelihood is the exact cell likelihood of finetune/adversarial/mlq.py (nll_terms), i.e. the
paper's eq. (43). Algorithm steps follow the paper's Algorithm 1 (Rao test on the N-bin grid, identification on a
gamma_os = 4 oversampled grid, Newton amplitude/frequency refinement, cyclic refinement, joint amplitude refit,
spurious-component suppression, CFAR stop). Every deviation from the paper is listed in REPORT-style text in
run_gnomp.py and in the README-like header of selftest_repro.py.

Real parameter block per path k: z_k = [theta_k (P entries), Re x_k, Im x_k]; slot layout (B, Kmax, P+2).
"""
import os, sys
import numpy as np
from scipy.special import ndtr

sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/finetune/adversarial')
import mlq  # exact cell NLL nll_terms(mu, lo, hi, s) -> (-logP, dNLL/dmu, curvature); shared with quantiser-ML

RIDGE = 0.02   # Tikhonov ridge on the 2x2 Fisher matrix (deviation D9: keeps the Rao denominator away from 0)
BIG = 1e6  # finite stand-in for an open outer cell edge (only used in the observed cells, never in h)


def tau_th(N, pfa=0.01):
    """paper eq. (39): threshold for a spectrum-wide false-alarm probability pfa over N independent bins."""
    return -2.0 * np.log(1.0 - (1.0 - pfa) ** (1.0 / N))


class Atoms1D:
    """a(w)_n = exp(j n w), n = 0..N-1 (paper eq. (1)). theta = (w,)."""
    P = 1

    def __init__(self, N):
        self.n = np.arange(N, dtype=float); self.M = N

    def atoms(self, th):
        a = np.exp(1j * self.n * th[..., :1])
        return a, (1j * self.n * a)[..., None]

    def grid(self, os_):
        G = self.M * os_
        th = (2 * np.pi * np.arange(G) / G)[:, None]
        a, _ = self.atoms(th)
        return th, a, np.arange(0, G, os_)            # theta grid, atoms on the grid, indices of the N-bin grid


class Atoms2D:
    """Beamspace atom of one path (our model, same convention as finetune/adversarial/mlq.py Model):
    a(u,v)[q,p] = rho_q(u) tau_p(v), rho = W^H r(u), tau = F^T t(v), r_n = exp(-j n u), t_m = exp(j m v).
    Flattened q-major (index q*16+p), matching acommon.adc_cells / mlq.cells_ri ordering."""
    P = 2

    def __init__(self, W, F):
        self.WH = np.conj(W).T; self.FT = F.T; self.n = np.arange(16.); self.M = 256

    def _vec(self, th):
        u, v = th[..., 0], th[..., 1]
        r = np.exp(-1j * self.n * u[..., None]); t = np.exp(1j * self.n * v[..., None])
        rho = r @ self.WH.T; drho = (-1j * self.n * r) @ self.WH.T
        tau = t @ self.FT.T; dtau = (1j * self.n * t) @ self.FT.T
        return rho, drho, tau, dtau

    def atoms(self, th):
        rho, drho, tau, dtau = self._vec(th)
        sh = th.shape[:-1] + (256,)
        a = (rho[..., :, None] * tau[..., None, :]).reshape(sh)
        du = (drho[..., :, None] * tau[..., None, :]).reshape(sh)
        dv = (rho[..., :, None] * dtau[..., None, :]).reshape(sh)
        return a, np.stack([du, dv], -1)

    def grid(self, os_):
        G = 16 * os_
        g = np.arange(G)
        U = -2 * np.pi * g / G; V = 2 * np.pi * g / G            # same index -> (u, v) map as acommon.fft_heat
        uu, vv = np.meshgrid(U, V, indexing='ij')
        th = np.stack([uu.ravel(), vv.ravel()], 1)
        a, _ = self.atoms(th)
        idxN = (np.arange(0, G, os_)[:, None] * G + np.arange(0, G, os_)[None, :]).ravel()   # the 16x16 DFT bins
        return th, a, idxN


def expected_info(m, thr, s):
    """Per real sample Fisher information E[(d log P_d / d mu)^2] summed over the quantiser cells (paper eq. (10)),
    divided by s^2 so that it is in the units of the score of nll_terms. m (B,2M) real means, thr (B,b+1) thresholds
    with -inf / +inf at the ends (open outer cells, as in the paper), s (B,1) per-component noise std."""
    with np.errstate(all='ignore'):
        mm = m[:, :, None]; ss = s[:, :, None]
        a = (thr[:, None, :-1] - mm) / ss; b = (thr[:, None, 1:] - mm) / ss
        pa = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi); pb = np.exp(-0.5 * b * b) / np.sqrt(2 * np.pi)
        P = ndtr(b) - ndtr(a)
        val = np.where(P > 1e-300, (pb - pa) ** 2 / np.maximum(P, 1e-300), 0.0)
    return val.sum(2) / (s ** 2)


def score_info(mu, lo, hi, s, thr):
    """pseudo-score psi = dlogP/dmu_R + j dlogP/dmu_I at the observed cells (paper eqs. (17)-(18)), per-sample Fisher
    information for Re and Im (paper eq. (9) through hR, hI), and the NLL at mu."""
    M = mu.shape[1]
    m = np.concatenate([mu.real, mu.imag], 1)
    f, g, _ = mlq.nll_terms(m, lo, hi, s)
    psi = -(g[:, :M] + 1j * g[:, M:])
    h = expected_info(m, thr, s)
    return psi, h[:, :M], h[:, M:], f.sum(1)


def rao(psi, hR, hI, Ag):
    """Rao-test pieces for a set of atoms Ag (G,M): c = a^H psi, S+ = sum h+ |a|^2, D = sum h- a^2, with
    h+ = (hR+hI)/2, h- = (hR-hI)/2. Real 2x2 Fisher matrix of (Re x, Im x) is [[S+ + ReD, -ImD], [-ImD, S+ - ReD]]."""
    w = 0.5 * (hR + hI); d = 0.5 * (hR - hI)
    c = psi @ np.conj(Ag).T
    Sp = (1.0 + RIDGE) * (w @ (Ag.real ** 2 + Ag.imag ** 2).T)     # ridge: I + RIDGE * S+ * I2 (deviation D9)
    D = d @ (Ag ** 2).T
    return c, Sp, D


def rao_T(c, Sp, D):
    """Rao statistic T = s^T I^{-1} s with s = (Re c, Im c); chi-square with 2 dof under H0 (paper eq. (19)/(32))."""
    den = np.maximum(Sp ** 2 - np.abs(D) ** 2, 1e-12 * Sp ** 2 + 1e-300)
    return (Sp * np.abs(c) ** 2 - np.real(D * np.conj(c) ** 2)) / den


def fisher_amp(c, Sp, D):
    """one Fisher-scoring step for the complex amplitude from x = 0: x = I^{-1} s (paper Sec. IV, line 7 start)."""
    den = np.maximum(Sp ** 2 - np.abs(D) ** 2, 1e-12 * Sp ** 2 + 1e-300)
    sR, sI = c.real, c.imag; Dr, Di = D.real, D.imag
    return (((Sp - Dr) * sR + Di * sI) + 1j * (Di * sR + (Sp + Dr) * sI)) / den


def state(model, Z, valid, lo, hi, s):
    P = model.P
    x = (Z[..., P] + 1j * Z[..., P + 1]) * valid
    a, Dm = model.atoms(Z[..., :P])
    mu = np.einsum('bk,bkm->bm', x, a)
    m = np.concatenate([mu.real, mu.imag], 1)
    f, g, h = mlq.nll_terms(m, lo, hi, s)
    return dict(x=x, a=a, Dm=Dm, mu=mu, g=g, h=h, nll=f.sum(1))


def jacobian(model, st):
    """real Jacobian (B, 2M, K(P+2)) of [Re mu, Im mu] wrt the real parameters [theta, Re x, Im x] of every slot."""
    x, a, Dm = st['x'], st['a'], st['Dm']
    B, K, M = a.shape; P = model.P
    th = x[..., None, None] * np.moveaxis(Dm, -1, 2)                        # (B,K,P,M)
    Jc = np.concatenate([th, a[:, :, None], 1j * a[:, :, None]], 2).reshape(B, K * (P + 2), M)
    return np.concatenate([Jc.real, Jc.imag], 2).transpose(0, 2, 1)


def cols_of(k, P, part):
    base = k * (P + 2)
    return list(range(base, base + P)) if part == 'th' else [base + P, base + P + 1]


def block_step(model, Z, valid, cols, upd, lo, hi, s, iters, clip=0.1):
    """Newton (Fisher-scoring) steps on a block of real columns of the paths, with a backtracking line search that only
    accepts a decrease of the exact NLL (paper Sec. IV-B: 'accepted only if the likelihood increases'). upd (B,) selects
    the scenes that are updated; invalid slots are frozen. Frequency steps are clipped to +-clip radians."""
    P = model.P; B = Z.shape[0]
    cols = np.asarray(cols); pk = cols // (P + 2); isth = (cols % (P + 2)) < P
    for _ in range(iters):
        st = state(model, Z, valid, lo, hi, s)
        J = jacobian(model, st)[:, :, cols]
        cv = (valid[:, pk] & upd[:, None]).astype(float)
        A = np.einsum('bik,bi,bil->bkl', J, st['h'], J)
        gr = np.einsum('bik,bi->bk', J, st['g'])
        lam = 1e-8 * (1.0 + np.abs(np.einsum('bkk->bk', A)).max(1))
        I = np.eye(len(cols))[None]
        Am = A * (cv[:, :, None] * cv[:, None, :]) + I * (lam[:, None, None] * cv[:, :, None] + (1 - cv)[:, :, None])
        d = -np.linalg.solve(Am, (gr * cv)[..., None])[..., 0]
        d[:, isth] = np.clip(d[:, isth], -clip, clip)
        cur = st['nll']; Zn = Z.copy(); done = ~upd
        for al in (1.0, 0.5, 0.25, 0.125):
            Zt = Z.reshape(B, -1).copy(); Zt[:, cols] += al * d; Zt = Zt.reshape(Z.shape)
            nt = state(model, Zt, valid, lo, hi, s)['nll']
            ok = (~done) & (nt < cur)
            Zn[ok] = Zt[ok]; done = done | ok
        Z = Zn
    return Z


def refine_cyclic(model, Z, valid, upd, lo, hi, s, m, Rc, iters_freq, iters_amp):
    """cyclic refinement over slots 0..m (paper Algorithm 1 lines 13-17), then joint amplitude refit (line 19)."""
    P = model.P
    for _ in range(Rc):
        for k in range(m + 1):
            Z = block_step(model, Z, valid, cols_of(k, P, 'th'), upd, lo, hi, s, iters_freq)
            Z = block_step(model, Z, valid, cols_of(k, P, 'amp'), upd, lo, hi, s, iters_amp)
    amp_all = [c for k in range(m + 1) for c in cols_of(k, P, 'amp')]
    return block_step(model, Z, valid, amp_all, upd, lo, hi, s, iters_amp)


def amp_start_guard(model, Z, valid, m, xs, lo, hi, s, active):
    """Fisher start of the new path's amplitude, accepted only if it lowers the exact NLL (backtracking over 1, 1/2, ..., 1/64;
    otherwise the amplitude stays 0, which leaves the NLL unchanged). Deviation D10: the paper's start is not guarded."""
    P = model.P; B = Z.shape[0]
    valid_m = valid.copy(); valid_m[:, m] = True
    base = state(model, Z, valid, lo, hi, s)['nll']
    newZ = Z.copy(); done = ~active
    for al in (1.0, 0.5, 0.25, 0.125, 1 / 16, 1 / 32, 1 / 64):
        Zt = Z.copy(); Zt[:, m, P] = al * xs.real; Zt[:, m, P + 1] = al * xs.imag
        nt = state(model, Zt, valid_m, lo, hi, s)['nll']
        ok = (~done) & (nt < base)
        newZ[ok] = Zt[ok]; done = done | ok
    return newZ


def gnomp(model, grid, lo, hi, s, thr, N, mode='cfar', L=3, Kmax=8, pfa=0.01,
          iters_amp=3, iters_freq=2, Rs=1, Rc=3, spurious=True):
    """GNOMP for a batch of scenes. mode 'cfar': stop when the Rao statistic on the N-bin grid falls below
    tau_th(N, pfa) (paper's CFAR rule, at most Kmax paths); mode 'L': exactly L paths, no test (the threshold is
    ignored, the same knowledge as every other decoder here). Returns Z (B,Kmax,P+2), valid (B,Kmax), stop-count."""
    th_g, Ag, idxN = grid
    Ag_N = Ag[idxN]
    B = lo.shape[0]; P = model.P; r = np.arange(B)
    Z = np.zeros((B, Kmax, P + 2)); valid = np.zeros((B, Kmax), bool)
    active = np.ones(B, bool); tau = tau_th(N, pfa)
    nmax = L if mode == 'L' else Kmax
    spur_removed = np.zeros(B, int)
    for m in range(nmax):
        st = state(model, Z, valid, lo, hi, s)
        psi, hR, hI, _ = score_info(st['mu'], lo, hi, s, thr)
        if mode == 'cfar':
            c, Sp, D = rao(psi, hR, hI, Ag_N)
            active &= rao_T(c, Sp, D).max(1) > tau
            if not active.any():
                break
        c, Sp, D = rao(psi, hR, hI, Ag)
        T = rao_T(c, Sp, D); bi = T.argmax(1)                       # identification on the oversampled grid (line 6)
        Z[:, m, :P] = th_g[bi]
        xs = fisher_amp(c[r, bi], Sp[r, bi], D[r, bi])
        Z[:, m, P] = 0.0; Z[:, m, P + 1] = 0.0
        Z = amp_start_guard(model, Z, valid, m, xs, lo, hi, s, active)
        valid[:, m] = active
        up = active.copy()
        Z = block_step(model, Z, valid, cols_of(m, P, 'amp'), up, lo, hi, s, iters_amp)          # line 7
        for _ in range(Rs):                                                                     # line 9
            Z = block_step(model, Z, valid, cols_of(m, P, 'th'), up, lo, hi, s, iters_freq)
            Z = block_step(model, Z, valid, cols_of(m, P, 'amp'), up, lo, hi, s, iters_amp)
        Z = refine_cyclic(model, Z, valid, up, lo, hi, s, m, Rc, iters_freq, iters_amp)         # lines 13-19
        if mode == 'cfar' and spurious and m >= 2:                                              # lines 21-29
            x = (Z[:, :, P] + 1j * Z[:, :, P + 1]) * valid
            ax = np.abs(x)
            cand = active & (ax[:, m - 1] < 0.5 * ax[:, m])
            if cand.any():
                valid_r = valid.copy(); valid_r[:, m - 1] = False
                st_r = state(model, Z, valid_r, lo, hi, s)
                psi_r, hR_r, hI_r, _ = score_info(st_r['mu'], lo, hi, s, thr)
                a1, _ = model.atoms(Z[:, m - 1, :P])
                w = 0.5 * (hR_r + hI_r); dd = 0.5 * (hR_r - hI_r)
                c1 = np.sum(psi_r * np.conj(a1), 1); Sp1 = np.sum(w * np.abs(a1) ** 2, 1); D1 = np.sum(dd * a1 ** 2, 1)
                rm = cand & (rao_T(c1, Sp1, D1) <= tau)
                if rm.any():
                    valid[rm, m - 1] = False; Z[rm, m - 1, P:] = 0.0; spur_removed += rm
                    Z = refine_cyclic(model, Z, valid, rm, lo, hi, s, m, Rc, iters_freq, iters_amp)
    return Z, valid, dict(spur_removed=spur_removed, steps=m + 1)


def to_pred(model, Z, valid, L):
    """(psi, phi) arrays (B,L): the L strongest valid paths (largest |x|), NaN-padded when fewer are found. 2-D: psi =
    arccos(u/pi), phi = arccos(v/pi) (u = pi cos psi, the convention of acommon.angles_to_uv)."""
    P = model.P; B = Z.shape[0]
    x = (Z[:, :, P] + 1j * Z[:, :, P + 1])
    key = np.where(valid, np.abs(x), -1.0)
    order = np.argsort(-key, 1)[:, :L]
    psi = np.full((B, L), np.nan); phi = np.full((B, L), np.nan)
    for b in range(B):
        sel = [k for k in order[b] if valid[b, k]]
        if len(sel) < L:
            continue
        th = Z[b, sel, :P]
        if P == 2:
            psi[b, :len(sel)] = np.arccos(np.clip(th[:, 0] / np.pi, -1, 1)); phi[b, :len(sel)] = np.arccos(np.clip(th[:, 1] / np.pi, -1, 1))
        else:
            psi[b, :len(sel)] = np.mod(th[:, 0], 2 * np.pi)
    return psi, phi


def adc_cells(Yq, bits):
    """the receiver's known cells of the peak-AGC mid-rise quantiser (same as finetune/adc/acommon.adc_cells, copied
    so that this module does not import torch). Returns rl, rh, il, ih (B,16,16) and A (B,)."""
    top = np.maximum(np.abs(Yq.real), np.abs(Yq.imag)).max((1, 2))[:, None, None]
    A = top / (1 - 2.0 ** -bits); D = 2 * A / 2 ** bits

    def cell(x):
        k = np.round(x / D - 0.5)
        return D * k, D * (k + 1)
    rl, rh = cell(Yq.real); il, ih = cell(Yq.imag)
    return rl, rh, il, ih, A[:, 0, 0]


def thresholds_peak(A, bits):
    """quantiser thresholds for the open-outer-cell model of the paper: -inf, -A + d D (d = 1..b-1), +inf."""
    b = 2 ** bits; D = 2 * A / b
    inner = -A[:, None] + D[:, None] * np.arange(1, b)[None, :]
    return np.concatenate([np.full((len(A), 1), -np.inf), inner, np.full((len(A), 1), np.inf)], 1)


def cells_ri(rl, rh, il, ih):
    B = len(rl)
    lo = np.concatenate([rl.reshape(B, -1), il.reshape(B, -1)], 1); hi = np.concatenate([rh.reshape(B, -1), ih.reshape(B, -1)], 1)
    return lo, hi
