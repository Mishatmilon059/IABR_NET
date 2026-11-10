"""Independent NOMP implementation (Mamandipoor, Ramasamy, Madhow 2016, 2-D, known model order L).
Written from the algorithm description only; the notebook decoder was read only to confirm conventions:
  antenna-domain atom a(u,v)[n,m] = exp(-j n u + j m v), u = pi cos(psi) (AoA), v = pi cos(phi) (AoD), n,m = 0..15.
Nothing is imported or exec-ed from notebook_src.py.

Per detection step l = 1..L:
  1. 4x zero-padded 2-D FFT of the residual; the max |.| cell is the coarse (u,v) seed.
     FFT bin (k1,k2) of r equals <a(u,v), r> at u = -2 pi k1/K, v = +2 pi k2/K (K = 64).
  2. single-path Newton on f(u,v) = |<a(u,v), r>|^2 (4 steps). A step is taken only where the 2x2 Hessian is
     negative definite (h_uu < 0 and det > 0); each component of the step is clipped to +-2 pi/64.
  3. cyclic re-estimation of all detected paths: 3 rounds; in each round every path j in turn is refined with 2 Newton
     steps against y minus the LS-fitted contribution of the other paths (LS gains recomputed before each path).
  4. LS gains of all paths, residual = y - A x.
The correlation and its derivatives use the separable form S(u,v) = p(u)^T R q(v), p_n = e^{j n u}, q_m = e^{-j m v}
(a different computation from the notebook's full 256-term sums)."""
import numpy as np

N = 16
_n = np.arange(N, dtype=np.float64)
STEP_CLIP = 2 * np.pi / 64


def steering(U, V):
    """(B,K) -> atom matrix (B, 256, K), column k = vec(a(u_k, v_k)) in row-major (n, m) order."""
    ar = np.exp(-1j * _n[None, :, None] * U[:, None, :])       # (B,16,K)  rows n
    at = np.exp(+1j * _n[None, :, None] * V[:, None, :])       # (B,16,K)  cols m
    return (ar[:, :, None, :] * at[:, None, :, :]).reshape(len(U), N * N, U.shape[1])


def ls_fit(y, U, V):
    """Least-squares complex gains x = argmin ||vec(y) - A x||; returns x (B,K) and A."""
    A = steering(U, V)
    AH = np.conj(np.swapaxes(A, 1, 2))
    G = AH @ A + 1e-9 * np.eye(U.shape[1])[None]
    x = np.linalg.solve(G, AH @ y.reshape(len(y), N * N, 1))[..., 0]
    return x, A


def newton_refine(R, u, v, steps):
    """Maximise |<a(u,v), R>|^2 over (u,v) for each scene of the batch R (B,16,16)."""
    u = u.astype(np.float64).copy(); v = v.astype(np.float64).copy()
    jn = 1j * _n; n2 = _n ** 2
    for _ in range(steps):
        p = np.exp(1j * _n[None] * u[:, None])                 # (B,16)
        q = np.exp(-1j * _n[None] * v[:, None])                # (B,16)
        Rq = np.einsum('bnm,bm->bn', R, q)
        Rq1 = np.einsum('bnm,bm->bn', R, q * (-jn))            # d/dv of q
        Rq2 = np.einsum('bnm,bm->bn', R, q * (-n2))            # d2/dv2 of q
        S = (p * Rq).sum(1)
        Su = (p * jn * Rq).sum(1)
        Suu = (p * (-n2) * Rq).sum(1)
        Sv = (p * Rq1).sum(1)
        Svv = (p * Rq2).sum(1)
        Suv = (p * jn * Rq1).sum(1)
        cS = np.conj(S)
        gu = 2 * (cS * Su).real
        gv = 2 * (cS * Sv).real
        huu = 2 * (np.abs(Su) ** 2 + (cS * Suu).real)
        hvv = 2 * (np.abs(Sv) ** 2 + (cS * Svv).real)
        huv = 2 * ((np.conj(Su) * Sv).real + (cS * Suv).real)
        det = huu * hvv - huv * huv
        nd = (huu < 0) & (det > 0)
        safe = np.where(nd, det, 1.0)
        du = -(hvv * gu - huv * gv) / safe                    # -H^{-1} g
        dv = -(huu * gv - huv * gu) / safe
        u = u + np.where(nd, np.clip(du, -STEP_CLIP, STEP_CLIP), 0.0)
        v = v + np.where(nd, np.clip(dv, -STEP_CLIP, STEP_CLIP), 0.0)
    return u, v


def cyclic_refine(y, U, V, rounds=3, steps=2):
    U = U.copy(); V = V.copy(); B, K = U.shape
    for _ in range(rounds):
        for j in range(K):
            x, A = ls_fit(y, U, V)
            full = np.einsum('bik,bk->bi', A, x)
            others = (full - A[:, :, j] * x[:, j:j + 1]).reshape(B, N, N)
            U[:, j], V[:, j] = newton_refine(y - others, U[:, j], V[:, j], steps)
    return U, V


def nomp(y, L, os_=4, rounds=3, seed_steps=4, cyc_steps=2):
    """y (B,16,16) complex antenna-domain channel; returns (U, V) each (B, L) in detection order."""
    y = np.asarray(y, dtype=np.complex128); B = len(y); K = N * os_
    U = np.zeros((B, 0)); V = np.zeros((B, 0)); res = y
    for _ in range(L):
        Fr = np.fft.fft2(res, s=(K, K))
        k = np.argmax(np.abs(Fr).reshape(B, -1), axis=1)
        u0 = -2 * np.pi * (k // K) / K; v0 = 2 * np.pi * (k % K) / K
        u1, v1 = newton_refine(res, u0, v0, seed_steps)
        U = np.concatenate([U, u1[:, None]], 1); V = np.concatenate([V, v1[:, None]], 1)
        U, V = cyclic_refine(y, U, V, rounds, cyc_steps)
        x, A = ls_fit(y, U, V)
        res = y - np.einsum('bik,bk->bi', A, x).reshape(B, N, N)
    return U, V


def oracle_seeded(y, U0, V0, rounds=3, cyc_steps=2):
    """Same objective, seeded at the true (u,v) of every path (known L): cyclic re-estimation only."""
    y = np.asarray(y, dtype=np.complex128)
    return cyclic_refine(y, np.asarray(U0, float), np.asarray(V0, float), rounds, cyc_steps)


def to_angles(U, V):
    """(u,v) -> (psi, phi) in [0, pi]: u, v wrapped to (-pi, pi], psi = arccos(u/pi), phi = arccos(v/pi)."""
    u = np.angle(np.exp(1j * U)); v = np.angle(np.exp(1j * V))
    return np.arccos(np.clip(u / np.pi, -1, 1)), np.arccos(np.clip(v / np.pi, -1, 1))
