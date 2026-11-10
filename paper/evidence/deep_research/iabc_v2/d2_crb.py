"""Cramer-Rao bound for the identifiable per-antenna phase / log-gain.

Exact model used (antenna domain; equivalent to the beamspace model because the DFT codebooks W, F are unitary, t0_timing.log):
    Ht[n,m] = exp(a_n + j b_n) exp(c_m + j d_m) sum_{l=1..L} alpha_l exp(-j n u_l + j m v_l) + w[n,m],
    w ~ CN(0, sigma^2), sigma^2 = 10^(-SNR/10), n,m = 0..15, u = pi cos(psi), v = pi cos(phi).
  rows: a = log g_r, b = -eps_r ; cols: c = log g_t, d = +eps_t   (H~ = conj(D_r) H D_t, iabr2_sim.impairment_targets).
Identifiable parametrisation (no redundancy): b = Bp x_b, d = Bp x_d (Bp: 16x14 orthonormal basis orthogonal to [1, n]);
a = Bg x_a, c = Bg x_c (Bg: 16x15 orthonormal basis orthogonal to 1). The mean / ramp parts are absorbed by alpha (complex) and by
(u, v). Nuisance: Re/Im alpha_l, u_l, v_l (unknown-angle case) or Re/Im alpha_l only (known-angle case).
FIM = (2 / sigma^2) Re(J^H J), J = d mu / d theta (256 x P). CRB = FIM^-1; per-antenna identifiable MSE = trace(CRB block)/16 per array
(Bp, Bg orthonormal). Reported: RMS over the 32 antennas = sqrt(mean over scenes of (tr C_b + tr C_d)/32) [deg] and same for gain [dB].
The FIM is evaluated at the true identifiable impairment, true angles and true alpha of each scene. Because FIM is proportional to
1/sigma^2, CRB(SNR) = CRB(0 dB) * 10^(-SNR/10) per scene exactly.
Approximate Bayesian bound (labelled approximate): (FIM + Lambda^-1)^-1 with Lambda the prior covariance of the identifiable
coefficients for a severity cell (phase U(-delta,delta) -> delta^2/3, gain U(-g,g) dB -> (g/8.686)^2/3), a Gaussian-prior (Van Trees
style) approximation of what an SNR-aware (shrinking) estimator could reach; the uniform prior has no finite Fisher information,
so this is not a strict bound.
Scenes: the evaluation scenes (L=3) of e1_eval.py: seed 8900+k (k = SNR index), first 250 scenes per seed; only the angles / alpha
are used (the bound is per-scene and rescaled for SNR). Finite-difference check of J on 5 scenes."""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
os.environ['IABR2_SUITE'] = 'D:/ai_ml_project'
import sys, json
import numpy as np
sys.path.insert(0, 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2')
import iabr2_sim as S

HERE = os.path.dirname(os.path.abspath(__file__))
DB = 20 / np.log(10)
n16 = np.arange(16.)


def basis(cols):
    M = np.eye(16) - cols @ np.linalg.pinv(cols)
    U, s, _ = np.linalg.svd(M)
    return U[:, s > 0.5]


Bp = basis(np.stack([np.ones(16), n16 - 7.5], 1))     # 16x14
Bg = basis(np.ones((16, 1)))                          # 16x15
assert Bp.shape == (16, 14) and Bg.shape == (16, 15)
NI = 14 + 15 + 14 + 15


def mu(theta, L):
    xa, xb, xc, xd = theta[:15], theta[15:29], theta[29:44], theta[44:58]
    al = theta[58:58 + L] + 1j * theta[58 + L:58 + 2 * L]; u = theta[58 + 2 * L:58 + 3 * L]; v = theta[58 + 3 * L:58 + 4 * L]
    a, b, c, d = Bg @ xa, Bp @ xb, Bg @ xc, Bp @ xd
    s = (al[None, None] * np.exp(-1j * n16[:, None, None] * u[None, None] + 1j * n16[None, :, None] * v[None, None])).sum(-1)
    return np.exp(a + 1j * b)[:, None] * np.exp(c + 1j * d)[None] * s


def jac(a, b, c, d, alpha, u, v):
    """vectorised analytic Jacobian. a..d (B,16); alpha,u,v (B,L). returns J (B,256,58+4L)."""
    B, L = alpha.shape
    G = np.exp(a + 1j * b)[:, :, None] * np.exp(c + 1j * d)[:, None, :]                       # (B,16,16)
    E = np.exp(-1j * n16[None, :, None, None] * u[:, None, None, :] + 1j * n16[None, None, :, None] * v[:, None, None, :])  # (B,16,16,L)
    s = (alpha[:, None, None, :] * E).sum(-1); m = G * s
    cols = []
    cols.append(m[:, :, :, None] * Bg[None, :, None, :])                 # x_a  (row gain)
    cols.append(1j * m[:, :, :, None] * Bp[None, :, None, :])            # x_b  (row phase)
    cols.append(m[:, :, :, None] * Bg[None, None, :, :])                 # x_c
    cols.append(1j * m[:, :, :, None] * Bp[None, None, :, :])            # x_d
    GE = G[..., None] * E
    cols.append(GE); cols.append(1j * GE)                                # Re alpha, Im alpha
    cols.append(-1j * n16[None, :, None, None] * alpha[:, None, None, :] * GE)   # u
    cols.append(1j * n16[None, None, :, None] * alpha[:, None, None, :] * GE)    # v
    return np.concatenate(cols, -1).reshape(B, 256, -1)


def theta_vec(a, b, c, d, alpha, u, v):
    return np.concatenate([Bg.T @ a, Bp.T @ b, Bg.T @ c, Bp.T @ d, alpha.real, alpha.imag, u, v])


def crb_blocks(J, known_angles, L, prior=None):
    """J (B,256,P) at sigma^2 = 1 -> per-scene (phase MSE per antenna rad^2, gain MSE per antenna neper^2)."""
    if known_angles:
        J = J[..., :58 + 2 * L]
    Fi = 2 * np.real(np.conj(np.transpose(J, (0, 2, 1))) @ J)
    if prior is not None:
        Fi = Fi.copy(); Fi[:, :NI, :NI] += np.diag(prior)[None]
    Cm = np.linalg.inv(Fi)
    dg = np.diagonal(Cm, axis1=1, axis2=2)
    ph = (dg[:, 15:29].sum(1) + dg[:, 44:58].sum(1)) / 32
    g = (dg[:, 0:15].sum(1) + dg[:, 29:44].sum(1)) / 32
    return ph, g


def main():
    SNRS = [-10, -5, 0, 5, 10, 15, 20, 25]; N = 250; L = 3
    psi, phi, alpha = [], [], []
    for k in range(8):
        p_, f_, a_ = S.sample_paths(np.full(N, L), np.random.default_rng(8900 + k))   # identical to e1_eval scenes (angles/alpha)
        psi.append(p_); phi.append(f_); alpha.append(a_)
    psi = np.concatenate(psi); phi = np.concatenate(phi); alpha = np.concatenate(alpha)
    u, v = np.pi * np.cos(psi), np.pi * np.cos(phi)
    B = len(u)
    # ---- finite-difference check (5 scenes, impaired point c30_6 drawn with seed 8999)
    r = np.random.default_rng(8999)
    imp = S.impairment_targets(*[10 ** (r.uniform(-1, 1, (5, 16)) * 6 / 20) * np.exp(1j * r.uniform(-1, 1, (5, 16)) * np.deg2rad(30)) for _ in range(2)])
    a5, b5, c5, d5 = imp[:, :16, 1], imp[:, :16, 0], imp[:, 16:, 1], imp[:, 16:, 0]
    J5 = jac(a5, b5, c5, d5, alpha[:5], u[:5], v[:5]); worst = 0
    for i in range(5):
        th = theta_vec(a5[i], b5[i], c5[i], d5[i], alpha[i], u[i], v[i]); h = 1e-6
        Jn = np.stack([(mu(th + h * e, L) - mu(th - h * e, L)).ravel() / (2 * h) for e in np.eye(len(th))], 1)
        rel = np.abs(Jn - J5[i]).max() / np.abs(J5[i]).max(); worst = max(worst, rel)
        # also: projection consistency -- mu with theta built from the identifiable parts reproduces the target-impaired channel
    print(f'FD check of the analytic Jacobian (5 scenes, P={J5.shape[-1]}): max |J_fd - J| / max|J| = {worst:.2e}', flush=True)
    th = theta_vec(a5[0], b5[0], c5[0], d5[0], alpha[0], u[0], v[0])
    G = np.exp(a5[0] + 1j * b5[0])[:, None] * np.exp(c5[0] + 1j * d5[0])[None]
    H0 = (alpha[0][None, None] * np.exp(-1j * n16[:, None, None] * u[0] + 1j * n16[None, :, None] * v[0])).sum(-1)
    print(f'mu(theta) == exp(identifiable impairment) * H check: {np.abs(mu(th, L) - G * H0).max():.1e}', flush=True)
    Fi = 2 * np.real(np.conj(J5.transpose(0, 2, 1)) @ J5); cond = np.linalg.cond(Fi)
    print('FIM condition numbers (unknown angles, 5 scenes):', ' '.join(f'{c_:.1e}' for c_ in cond), flush=True)

    out = dict(model=__doc__.split('\n\n')[1] if False else 'Ht = exp(a+jb)_n exp(c+jd)_m sum alpha_l exp(-j n u_l + j m v_l) + CN(0,sigma^2)',
               scenes='seeds 8900..8907 (first 250 each, L=3)', snr=SNRS, cells={})
    zero = np.zeros((B, 16))
    pts = {'clean point': (0, 0), 'c30_6 point': (30, 6)}
    base = {}
    for pname, (pd, gd) in pts.items():
        rr = np.random.default_rng(8998)
        if pd or gd:
            Dr = 10 ** (rr.uniform(-1, 1, (B, 16)) * gd / 20) * np.exp(1j * rr.uniform(-1, 1, (B, 16)) * np.deg2rad(pd))
            Dt = 10 ** (rr.uniform(-1, 1, (B, 16)) * gd / 20) * np.exp(1j * rr.uniform(-1, 1, (B, 16)) * np.deg2rad(pd))
            t = S.impairment_targets(Dr, Dt).astype(float); A_, B_, C_, D_ = t[:, :16, 1], t[:, :16, 0], t[:, 16:, 1], t[:, 16:, 0]
        else:
            A_ = B_ = C_ = D_ = zero
        res = {}
        for ka in [False, True]:
            ph, g = [], []
            for s0 in range(0, B, 200):
                sl = slice(s0, s0 + 200)
                J = jac(A_[sl], B_[sl], C_[sl], D_[sl], alpha[sl, :L], u[sl, :L], v[sl, :L])
                p_, g_ = crb_blocks(J, ka, L); ph.append(p_); g.append(g_)
            ph = np.concatenate(ph); g = np.concatenate(g)
            res['known' if ka else 'unknown'] = (ph, g)
        base[pname] = res
        lines = []
        for nm, (ph, g) in res.items():
            for k, snr in enumerate(SNRS):
                q = slice(k * N, (k + 1) * N); sc = 10 ** (-snr / 10)
                lines.append((nm, snr, np.rad2deg(np.sqrt(ph[q].mean() * sc)), DB * np.sqrt(g[q].mean() * sc),
                              np.rad2deg(np.sqrt(np.median(ph[q]) * sc)), DB * np.sqrt(np.median(g[q]) * sc)))
        print(f'\n## CRB at the {pname} (RMS over 32 antennas; mean-of-CRB, and median scene in brackets)', flush=True)
        print('| angles | ' + ' | '.join(f'{s} dB' for s in SNRS) + ' |')
        print('|---|' + '---|' * len(SNRS))
        for nm in ['unknown', 'known']:
            ll = [l for l in lines if l[0] == nm]
            print(f'| phase deg, {nm} | ' + ' | '.join(f'{l[2]:.3g} [{l[4]:.3g}]' for l in ll) + ' |')
        for nm in ['unknown', 'known']:
            ll = [l for l in lines if l[0] == nm]
            print(f'| gain dB, {nm} | ' + ' | '.join(f'{l[3]:.3g} [{l[5]:.3g}]' for l in ll) + ' |')
        out[pname] = {nm: dict(phase_deg=[l[2] for l in lines if l[0] == nm], gain_db=[l[3] for l in lines if l[0] == nm],
                                phase_deg_median=[l[4] for l in lines if l[0] == nm], gain_db_median=[l[5] for l in lines if l[0] == nm]) for nm in ['unknown', 'known']}
    # ---- approximate Bayesian bound per severity cell (unknown angles, clean point FIM)
    print('\n## Approximate Bayesian (Gaussian-prior) bound vs prior RMS, unknown angles; % = share of identifiable MSE removable', flush=True)
    CELLS = [('ph5', 5, 0), ('ph8', 8, 0), ('ph20', 20, 0), ('ph30', 30, 0), ('ph45', 45, 0), ('g3', 0, 3), ('g6', 0, 6), ('c8_3', 8, 3), ('c20_4', 20, 4), ('c30_6', 30, 6)]
    Jall = [jac(zero[s0:s0 + 200], zero[s0:s0 + 200], zero[s0:s0 + 200], zero[s0:s0 + 200], alpha[s0:s0 + 200, :L], u[s0:s0 + 200, :L], v[s0:s0 + 200, :L]) for s0 in range(0, B, 200)]
    for name, pd, gd in CELLS:
        vp = np.deg2rad(max(pd, 1e-3)) ** 2 / 3; vg = (max(gd, 1e-3) / DB) ** 2 / 3
        prior_ph = vp * 14 / 16 if pd else 0.0; prior_g = vg * 15 / 16 if gd else 0.0
        row = dict(prior_phase_deg=np.rad2deg(np.sqrt(prior_ph)), prior_gain_db=DB * np.sqrt(prior_g), phase_deg=[], gain_db=[], phase_rem=[], gain_rem=[])
        for k, snr in enumerate(SNRS):
            sc = 10 ** (-snr / 10); ph, g = [], []
            pr = np.concatenate([np.full(15, 1 / vg), np.full(14, 1 / vp), np.full(15, 1 / vg), np.full(14, 1 / vp)])
            for J in Jall:
                p_, g_ = crb_blocks(J / np.sqrt(sc), False, L, prior=pr); ph.append(p_); g.append(g_)
            ph = np.concatenate(ph)[k * N:(k + 1) * N]; g = np.concatenate(g)[k * N:(k + 1) * N]
            row['phase_deg'].append(float(np.rad2deg(np.sqrt(ph.mean())))); row['gain_db'].append(float(DB * np.sqrt(g.mean())))
            row['phase_rem'].append(float(100 * (1 - ph.mean() / prior_ph)) if pd else None)
            row['gain_rem'].append(float(100 * (1 - g.mean() / prior_g)) if gd else None)
        out['cells'][name] = row
        if pd:
            print(f'{name:6s} phase: prior {row["prior_phase_deg"]:.2f} deg | bound ' + ' '.join(f'{s}:{x:.2f}({r:.0f}%)' for s, x, r in zip(SNRS, row['phase_deg'], row['phase_rem'])))
        if gd:
            print(f'{name:6s} gain : prior {row["prior_gain_db"]:.2f} dB  | bound ' + ' '.join(f'{s}:{x:.2f}({r:.0f}%)' for s, x, r in zip(SNRS, row['gain_db'], row['gain_rem'])))
    json.dump(out, open(f'{HERE}/crb.json', 'w'), indent=1, default=float)
    print('\nwrote crb.json')


if __name__ == '__main__':
    main()
