"""Stronger CLASSICAL impairment estimator (no learning, no tuning on test data) used to attack the claim that IABC-v2b closes a gap
classical methods cannot. Design fixed before any 9300-9307 test run; sanity/timing prototype on seeds 9340-9347 (a0_proto.log).
return_s2=True also returns the per-scene noise-variance estimate of the last LM round (used by the blind-SNR gate in a_lib.py).

EB-ML (empirical-Bayes joint maximum likelihood):
  1. NOMP(Ht) -> angles; gate-open ALS (selfcal_ret lam 1, tau 0, 4 it, cyclic angle refinement) -> e_ALS, angles.
  2. Joint ML over all parameters (identifiable impairment coefficients, complex path gains, u, v) by batched Levenberg-Marquardt
     (analytic Jacobian d2_crb.jac, verified against an independent finite-difference model in a1_crb_check.log), init from step 1.
  3. Per-scene noise variance sigma^2 = residual / (256 - P/2); covariance of the ML estimate = (sigma^2 / 2) (Re J^H J)^-1 (the CRB at
     the estimate; a1_crb_check shows ML attains it). Positive-part James-Stein / empirical-Bayes shrinkage per block (Rx phase,
     Rx gain, Tx phase, Tx gain): w = P/(P+N), N = mean diag of the covariance block, P = max(0, |x|^2/dim - N). No tuned constant.
  4. Re-detection: NOMP on Ht corrected by the shrunk estimate, then steps 2-3 again from the new angles (ROUNDS rounds).
Returns the identifiable estimate (B,32,2) in the IABC convention and the final NOMP angles on the corrected channel."""
import numpy as np
import sys
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/iabc_v2')
from v2common import C, selfcal_ret, apply_corr, ident_vec
import d2_crb as D2

Bp, Bg = D2.Bp, D2.Bg
n16 = np.arange(16.)


def theta_from(e, U, V, alpha):
    return np.concatenate([e[:, :16, 1] @ Bg, e[:, :16, 0] @ Bp, e[:, 16:, 1] @ Bg, e[:, 16:, 0] @ Bp,
                           alpha.real, alpha.imag, U, V], 1)


def unpack(th, L):
    a, b, c, d = th[:, :15] @ Bg.T, th[:, 15:29] @ Bp.T, th[:, 29:44] @ Bg.T, th[:, 44:58] @ Bp.T
    al = th[:, 58:58 + L] + 1j * th[:, 58 + L:58 + 2 * L]; U = th[:, 58 + 2 * L:58 + 3 * L]; V = th[:, 58 + 3 * L:58 + 4 * L]
    return a, b, c, d, al, U, V


def mu_b(th, L):
    a, b, c, d, al, U, V = unpack(th, L)
    G = np.exp(a + 1j * b)[:, :, None] * np.exp(c + 1j * d)[:, None, :]
    E = np.exp(-1j * n16[None, :, None, None] * U[:, None, None, :] + 1j * n16[None, None, :, None] * V[:, None, None, :])
    return G * (al[:, None, None, :] * E).sum(-1)


def lm(Ht, th, L, iters=10):
    B = len(Ht); lam = np.full(B, 1e-3)
    cost = lambda t: (np.abs(Ht - mu_b(t, L)) ** 2).sum((1, 2))
    c0 = cost(th)
    for _ in range(iters):
        a, b, c, d, al, U, V = unpack(th, L)
        J = D2.jac(a, b, c, d, al, U, V)                          # (B,256,P)
        r = (Ht - mu_b(th, L)).reshape(B, -1)
        A = np.real(np.conj(np.transpose(J, (0, 2, 1))) @ J); g = np.real(np.einsum('bkp,bk->bp', np.conj(J), r))
        dA = np.diagonal(A, axis1=1, axis2=2)
        for _t in range(4):
            M = A + lam[:, None, None] * (dA[:, :, None] * np.eye(A.shape[1])[None]) + 1e-9 * np.eye(A.shape[1])[None]
            st = np.linalg.solve(M, g[..., None])[..., 0]
            st = np.where(np.isfinite(st), st, 0.0)
            c1 = cost(th + st); ok = np.isfinite(c1) & (c1 < c0)
            th = np.where(ok[:, None], th + st, th); c0 = np.where(ok, c1, c0)
            lam = np.where(ok, np.maximum(lam / 10, 1e-9), lam * 10)
            if ok.all():
                break
    a, b, c, d, al, U, V = unpack(th, L)
    J = D2.jac(a, b, c, d, al, U, V); A = np.real(np.conj(np.transpose(J, (0, 2, 1))) @ J)
    P = A.shape[1]; s2 = c0 / (256 - P / 2)
    Cov = (s2 / 2)[:, None, None] * np.linalg.inv(A + 1e-9 * np.eye(P)[None])
    return th, Cov, s2


def eb_shrink(th, Cov):
    """positive-part James-Stein per block; returns shrunk identifiable estimate (B,32,2) and the weights (B,4)."""
    blocks = [slice(0, 15), slice(15, 29), slice(29, 44), slice(44, 58)]
    th = th.copy(); W = []
    dg = np.diagonal(Cov, axis1=1, axis2=2)
    for bl in blocks:
        x = th[:, bl]; N = dg[:, bl].mean(1); Pp = np.maximum(0.0, (x ** 2).mean(1) - N)
        w = Pp / (Pp + N + 1e-30); th[:, bl] = x * w[:, None]; W.append(w)
    e = np.stack([np.concatenate([th[:, 15:29] @ Bp.T, th[:, 44:58] @ Bp.T], 1), np.concatenate([th[:, :15] @ Bg.T, th[:, 29:44] @ Bg.T], 1)], -1)
    return ident_vec(e[..., 0], e[..., 1]), np.stack(W, 1)


def eb_ml(Ht, L, rounds=2, shrink=True, lm_iters=10, return_s2=False):
    """full classical chain; returns (estimate (B,32,2), final NOMP angles on the corrected channel, unshrunk ML estimate)."""
    U, V = C.nomp_greedy(Ht, L, 4, 3)
    U, V, e, _, _ = selfcal_ret(Ht, U, V, 1.0, 0.0, iters=4)
    e_ml = e
    for r in range(rounds):
        x, _ = C.ls_gains(apply_corr(Ht, e), U, V)
        th, Cov, s2 = lm(Ht, theta_from(e, U, V, x), L, lm_iters)
        a, b, c, d, al, Uu, Vv = unpack(th, L)
        e_ml = ident_vec(np.concatenate([b, d], 1), np.concatenate([a, c], 1))
        e = eb_shrink(th, Cov)[0] if shrink else e_ml
        U, V = C.nomp_greedy(apply_corr(Ht, e), L, 4, 3)
    if return_s2:
        return e, (U, V), e_ml, s2
    return e, (U, V), e_ml
