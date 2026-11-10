"""Quantisation-aware classical estimators for the low-resolution-ADC cells (finetune/adversarial). No learning.
Model (exact generator of stress_sim.generate(.., 'M5', b) on ideal hardware):
   z = W^H Hm F + n,  Hm[n,m] = sum_k x_k exp(-j n u_k + j m v_k)   (u = pi cos psi, v = pi cos phi; same atoms as notebook decode)
   n: iid real/imag Gaussian, std sigma = sqrt(10^(-SNR/10)/2) (known to the receiver), y = ADC_b(z) with peak AGC.
The receiver knows its ADC (b, peak AGC) so every Re/Im sample lies in a known cell [lo, hi] (acommon.adc_cells; checked in s1_audit).
Exact log-likelihood:  sum_i log( Phi((hi_i - mu_i)/s) - Phi((lo_i - mu_i)/s) ),  mu = Re/Im of the model, s = kappa * sigma.
  'ML'     : Levenberg-Marquardt (exact per-component curvature, >= 0 since the interval-censored Gaussian likelihood is log-concave
             in mu) on all 4K real parameters (u, v, Re x, Im x), from several starts; per scene the start with the lowest final NLL
             is kept (no ground truth used).
  'EM'     : exact EM for the censored model: E-step z_hat = E[z | cell, mu] (truncated-normal mean), M-step = cyclic Newton refinement
             and a fresh greedy NOMP on to_ant(z_hat) (keep whichever has lower NLL), then the ML polish.
kappa (noise-std inflation) is the only tuned constant (tuned on seeds 8730 / 8745 only)."""
import numpy as np
from scipy.special import log_ndtr
import scipy.ndimage

LOG2PI = 0.5 * np.log(2 * np.pi)
_n = np.arange(16.)


class Model:
    def __init__(self, W, F, C):
        self.WH = np.conj(W).T          # rho = W^H r
        self.FT = F.T                   # tau = F^T t
        self.C = C                      # new_results/common (decoder pieces)
        self.W, self.F = W, F

    def parts(self, U, V):
        r = np.exp(-1j * _n[None, None, :] * U[..., None])        # (B,K,16)
        t = np.exp(1j * _n[None, None, :] * V[..., None])
        rho = np.einsum('qn,bkn->bkq', self.WH, r); tau = np.einsum('pm,bkm->bkp', self.FT, t)
        drho = np.einsum('qn,bkn->bkq', self.WH, -1j * _n * r); dtau = np.einsum('pm,bkm->bkp', self.FT, 1j * _n * t)
        return rho, tau, drho, dtau

    def mu(self, U, V, X):
        rho, tau, _, _ = self.parts(U, V)
        return np.einsum('bk,bkq,bkp->bqp', X, rho, tau)

    def jac(self, U, V, X):
        """mu (B,512) real and J (B,512,4K) real, columns [u_1..u_K, v_1..v_K, Re x_1..K, Im x_1..K]"""
        rho, tau, drho, dtau = self.parts(U, V)
        At = np.einsum('bkq,bkp->bkqp', rho, tau)                       # (B,K,16,16)
        mu = np.einsum('bk,bkqp->bqp', X, At)
        Ju = X[:, :, None, None] * np.einsum('bkq,bkp->bkqp', drho, tau)
        Jv = X[:, :, None, None] * np.einsum('bkq,bkp->bkqp', rho, dtau)
        Jc = np.concatenate([Ju, Jv, At, 1j * At], 1)                   # (B,4K,16,16) complex
        B = len(U); Jc = Jc.reshape(B, Jc.shape[1], 256)
        J = np.concatenate([Jc.real, Jc.imag], 2).transpose(0, 2, 1)   # (B,512,4K)
        m = mu.reshape(B, 256)
        return np.concatenate([m.real, m.imag], 1), J


def cells_ri(Yq, bits, adc_cells):
    rl, rh, il, ih = adc_cells(Yq, bits)
    B = len(Yq)
    lo = np.concatenate([rl.reshape(B, -1), il.reshape(B, -1)], 1); hi = np.concatenate([rh.reshape(B, -1), ih.reshape(B, -1)], 1)
    return lo, hi


def nll_terms(mu, lo, hi, s):
    """per-component NLL and its first / second derivative wrt mu (s: (B,1) std)."""
    a = (lo - mu) / s; b = (hi - mu) / s
    flip = (a + b) > 0
    a2 = np.where(flip, -b, a); b2 = np.where(flip, -a, b)
    la = log_ndtr(a2); lb = log_ndtr(b2)
    logZ = lb + np.log1p(-np.exp(np.minimum(la - lb, -1e-16)))
    ra = np.exp(-0.5 * a * a - LOG2PI - logZ); rb = np.exp(-0.5 * b * b - LOG2PI - logZ)
    g = (rb - ra) / s
    h = (b * rb - a * ra) / s ** 2 + g * g
    return -logZ, g, np.maximum(h, 1e-9 / s ** 2)


def mu_ri(Mo, U, V, X):
    m = Mo.mu(U, V, X).reshape(len(U), 256)
    return np.concatenate([m.real, m.imag], 1)


def nll(Mo, U, V, X, lo, hi, s):
    return nll_terms(mu_ri(Mo, U, V, X), lo, hi, s)[0].sum(1)


def lm(Mo, U, V, X, lo, hi, s, iters=40, which='all', step_clip=0.1):
    """Levenberg-Marquardt on the exact NLL. which='all' (u, v, x) or 'gain' (x only). Returns U, V, X, NLL."""
    U, V, X = U.copy(), V.copy(), X.copy(); B, K = U.shape
    lam = np.full(B, 1e-2)
    mu, J = Mo.jac(U, V, X); f, g, h = nll_terms(mu, lo, hi, s); F0 = f.sum(1)
    sel = slice(None) if which == 'all' else slice(2 * K, 4 * K)
    eye = np.eye(4 * K if which == 'all' else 2 * K)[None]
    for _ in range(iters):
        Js = J[:, :, sel]
        A = np.einsum('bik,bi,bil->bkl', Js, h, Js); gr = np.einsum('bik,bi->bk', Js, g)
        D = np.einsum('bkk->bk', A)
        Ad = A + (lam[:, None] * (D + 1e-9))[:, :, None] * eye
        d = -np.linalg.solve(Ad, gr[..., None])[..., 0]
        if which == 'all':
            du = np.clip(d[:, :K], -step_clip, step_clip); dv = np.clip(d[:, K:2 * K], -step_clip, step_clip)
            Un, Vn, Xn = U + du, V + dv, X + d[:, 2 * K:3 * K] + 1j * d[:, 3 * K:]
        else:
            Un, Vn, Xn = U, V, X + d[:, :K] + 1j * d[:, K:]
        mun, Jn = Mo.jac(Un, Vn, Xn); fn, gn, hn = nll_terms(mun, lo, hi, s); Fn = fn.sum(1)
        ok = Fn < F0
        U = np.where(ok[:, None], Un, U); V = np.where(ok[:, None], Vn, V); X = np.where(ok[:, None], Xn, X)
        J = np.where(ok[:, None, None], Jn, J); g = np.where(ok[:, None], gn, g); h = np.where(ok[:, None], hn, h)
        F0 = np.where(ok, Fn, F0); lam = np.clip(np.where(ok, lam / 3, lam * 4), 1e-7, 1e7)
    return U, V, X, F0


def to_ant(Mo, Yb):
    return np.einsum('nq,bqp,mp->bnm', Mo.W, Yb, np.conj(Mo.F))


def init_gains(Mo, U, V, Yq, lo, hi, s, iters=15):
    """LS gains on the cell-centre data (antenna domain), then the convex ML gain fit with the angles fixed."""
    X, _ = Mo.C.ls_gains(to_ant(Mo, Yq), U, V)
    return lm(Mo, U, V, X, lo, hi, s, iters=iters, which='gain')[2]


def fft_cands(H, K=5, os_=4):
    """top-K local maxima (3x3 wrap) of |FFT2(H)| on the os_ x grid (same index -> (u, v) map as nomp_greedy / heat_peaks)."""
    G = 16 * os_
    Xf = np.abs(np.fft.fft2(H, s=(G, G)))
    mx = scipy.ndimage.maximum_filter(Xf, size=(1, 3, 3), mode='wrap')
    score = np.where(Xf >= mx, Xf + 10 * Xf.max(), Xf).reshape(len(H), -1)
    idx = np.argsort(-score, 1)[:, :K]
    return -2 * np.pi * (idx // G) / G, 2 * np.pi * (idx % G) / G


def em(Mo, U, V, X, lo, hi, s, L, iters=6):
    """exact EM for the censored-Gaussian model. M-step candidates: cyclic (rc 2) of the current paths and fresh greedy NOMP (os 4, rc 3)
    on the E-step antenna channel; per scene keep the candidate with the lowest exact NLL (after the convex gain fit)."""
    C = Mo.C; B = len(U)
    best = nll(Mo, U, V, X, lo, hi, s)
    for _ in range(iters):
        mu = mu_ri(Mo, U, V, X)
        _, g, _ = nll_terms(mu, lo, hi, s)
        zh = mu - s ** 2 * g                                   # truncated-normal mean E[z | cell, mu]
        Hd = to_ant(Mo, (zh[:, :256] + 1j * zh[:, 256:]).reshape(B, 16, 16))
        for Ux, Vx in [C.cyclic(Hd, U, V, 2), C.nomp_greedy(Hd, L, 4, 3)]:
            Xx = lm(Mo, Ux, Vx, C.ls_gains(Hd, Ux, Vx)[0], lo, hi, s, iters=10, which='gain')[2]
            e = nll(Mo, Ux, Vx, Xx, lo, hi, s); bt = e < best
            U = np.where(bt[:, None], Ux, U); V = np.where(bt[:, None], Vx, V); X = np.where(bt[:, None], Xx, X); best = np.where(bt, e, best)
    return U, V, X, best


def pick(cands):
    """cands: list of (U, V, X, nll) -> per scene the lowest-NLL candidate (+ index chosen)."""
    Fm = np.stack([c[3] for c in cands], 1); k = Fm.argmin(1); i = np.arange(len(k))
    out = [np.stack([c[j] for c in cands], 1)[i, k] for j in range(3)]
    return out[0], out[1], out[2], Fm[i, k], k
