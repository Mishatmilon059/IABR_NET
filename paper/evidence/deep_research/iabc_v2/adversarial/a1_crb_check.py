"""Adversarial re-check of the CRB in iabc_v2/d2_crb.py (independent implementation).
1) Noise model: sigma^2 of to_ant(noise) per entry vs 10^(-SNR/10) (simulator noise, seed 9330).
2) Independent model mu(theta): parametrised with ANGLES (psi, phi) instead of (u, v) and built with the simulator's steering vectors
   (iabr2_sim.simulate formula) and an explicit D_r / D_t: Ht = conj(D_r) H D_t. Jacobian by central finite differences only
   (no analytic derivative). CRB (unknown angles / known angles) vs d2_crb.jac/crb_blocks on the same scenes (fresh seed 9331, L=3).
3) Monte-Carlo efficiency check: Gauss-Newton ML (FD Jacobian, init at truth, 8 iterations) at SNR 15/25/35 dB, 300 scenes, seed 9332,
   impairment cell 30 deg / 6 dB. Empirical identifiable MSE (projection identical to ident_err) / mean CRB should be ~1 at high SNR.
   Also: oracle ALS (v2common selfcal_ret with true angles fixed, lam 1, tau 0, 4 and 20 iterations) vs the known-angle CRB.
stdout -> a1_crb_check.log"""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '6'
import sys
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/iabc_v2')
from v2common import *            # noqa
import d2_crb as D2               # analytic jac / crb_blocks under test (main() not executed)

n16 = np.arange(16.)
Bp, Bg = D2.Bp, D2.Bg


def steer(ang, N=16):
    return np.exp(-1j * np.pi * np.cos(ang)[..., None] * np.arange(N)) / np.sqrt(N)


def mu2(th, L):
    """independent model: th = [xa(15), xb(14), xc(15), xd(14), Re a(L), Im a(L), psi(L), phi(L)]"""
    xa, xb, xc, xd = th[:15], th[15:29], th[29:44], th[44:58]
    al = th[58:58 + L] + 1j * th[58 + L:58 + 2 * L]; psi = th[58 + 2 * L:58 + 3 * L]; phi = th[58 + 3 * L:58 + 4 * L]
    H = 16.0 * np.einsum('l,ln,lm->nm', al, steer(psi), np.conj(steer(phi)))       # simulate(): sqrt(NT NR) sum alpha a_r a_t^H
    D_r = np.exp(Bg @ xa) * np.exp(-1j * (Bp @ xb))      # conj(D_r) row phase = +Bp xb  (target = -angle(D_r))
    D_t = np.exp(Bg @ xc) * np.exp(1j * (Bp @ xd))
    return np.conj(D_r)[:, None] * H * D_t[None, :]


def fd_jac(th, L, h=1e-6):
    return np.stack([(mu2(th + h * e, L) - mu2(th - h * e, L)).ravel() / (2 * h) for e in np.eye(len(th))], 1)


def crb_from_J(J, L, known):
    if known:
        J = J[:, :58 + 2 * L]
    Fi = 2 * np.real(J.conj().T @ J); Cm = np.linalg.inv(Fi); d = np.diag(Cm)
    return (d[15:29].sum() + d[44:58].sum()) / 32, (d[0:15].sum() + d[29:44].sum()) / 32


def ident_from_theta(th):
    """identifiable (32,2) [phase, log-gain] in the IABC convention (rows target = -angle(D_r) = Bp xb, cols = Bp xd)."""
    return np.stack([np.concatenate([Bp @ th[15:29], Bp @ th[44:58]]), np.concatenate([Bg @ th[0:15], Bg @ th[29:44]])], -1)


def main():
    L = 3
    # ---------------- 1) noise model
    o = S.simulate(np.full((2000, 1), 1.0), np.full((2000, 1), 1.0), np.zeros((2000, 1), complex), 7.0, np.random.default_rng(9330))
    Zt = to_ant(o['Y'])
    print(f'1) noise per entry: var(to_ant(noise)) at 7 dB = {np.mean(np.abs(Zt) ** 2):.5f}  vs 10^(-0.7) = {10 ** -0.7:.5f}  '
          f'(signal power E|H|^2 for L=3 scenes checked below)', flush=True)
    # ---------------- 2) independent FD CRB vs analytic d2_crb
    rng = np.random.default_rng(9331); n = 40
    psi, phi, alpha = S.sample_paths(np.full(n, L), rng)
    o = S.simulate(psi, phi, alpha, 300.0, np.random.default_rng([9331, 3]), phase_deg_max=30, gain_db_max=6)
    print(f'   signal power mean |H|^2 (40 scenes) = {np.mean(np.abs(o["H"]) ** 2):.3f} (CRB assumes alpha ~ CN(0,1/L): E|H|^2 = 1)', flush=True)
    tgt = S.impairment_targets(o['D_r'], o['D_t']).astype(float)
    rat = {k: [] for k in ['ph_unk', 'g_unk', 'ph_kn', 'g_kn']}; jmax = 0
    for i in range(n):
        lg_r = np.log(np.abs(o['D_r'][i])); lg_t = np.log(np.abs(o['D_t'][i]))
        th = np.concatenate([Bg.T @ lg_r, Bp.T @ tgt[i, :16, 0], Bg.T @ lg_t, Bp.T @ tgt[i, 16:, 0], alpha[i].real, alpha[i].imag, psi[i], phi[i]])
        # the identifiable-only model must reproduce the simulated channel up to the unidentifiable mean/ramp (absorbed): check fit of
        # |Ht| row/col structure is not needed; use the model at the identifiable point (same as d2_crb)
        J = fd_jac(th, L)
        u, v = np.pi * np.cos(psi[i]), np.pi * np.cos(phi[i])
        Ja = D2.jac(tgt[i:i + 1, :16, 1], tgt[i:i + 1, :16, 0], tgt[i:i + 1, 16:, 1], tgt[i:i + 1, 16:, 0], alpha[i:i + 1], u[None], v[None])
        for known, tag in [(False, 'unk'), (True, 'kn')]:
            p1, g1 = crb_from_J(J, L, known)
            p2, g2 = D2.crb_blocks(Ja, known, L); rat[f'ph_{tag}'].append(p1 / p2[0]); rat[f'g_{tag}'].append(g1 / g2[0])
        # impairment columns must agree exactly between the two models (same parametrisation of the impairment)
        jmax = max(jmax, np.abs(J[:, :58] - Ja[0][:, :58]).max() / np.abs(Ja[0]).max())
    print(f'2) independent FD-Jacobian CRB / d2_crb analytic CRB over {n} fresh scenes (seed 9331, L=3, 30 deg/6 dB point):', flush=True)
    for k, x in rat.items():
        x = np.array(x); print(f'   {k}: ratio mean {x.mean():.6f} min {x.min():.6f} max {x.max():.6f}', flush=True)
    print(f'   impairment-block Jacobian max rel diff (independent FD vs analytic): {jmax:.2e}', flush=True)
    # ---------------- 3) Monte-Carlo efficiency (Gauss-Newton ML, init at truth)
    n = 300; rng = np.random.default_rng(9332)
    psi, phi, alpha = S.sample_paths(np.full(n, L), rng)
    o = S.simulate(psi, phi, alpha, 300.0, np.random.default_rng([9332, 3]), phase_deg_max=30, gain_db_max=6)
    tgt = S.impairment_targets(o['D_r'], o['D_t']).astype(float)
    TH = []; MU = []; CR = []; CRk = []
    for i in range(n):
        lg_r = np.log(np.abs(o['D_r'][i])); lg_t = np.log(np.abs(o['D_t'][i]))
        th = np.concatenate([Bg.T @ lg_r, Bp.T @ tgt[i, :16, 0], Bg.T @ lg_t, Bp.T @ tgt[i, 16:, 0], alpha[i].real, alpha[i].imag, psi[i], phi[i]])
        TH.append(th); MU.append(mu2(th, L)); J = fd_jac(th, L); CR.append(crb_from_J(J, L, False)); CRk.append(crb_from_J(J, L, True))
    TH = np.array(TH); MU = np.array(MU); CR = np.array(CR); CRk = np.array(CRk)
    print(f'3) Monte-Carlo ML efficiency, {n} scenes (seed 9332), 30 deg / 6 dB, L=3, Gauss-Newton from truth (8 it, FD Jacobian):', flush=True)
    print('| SNR | ML phase MSE / CRB | ML gain MSE / CRB | ML RMS phase deg (CRB) | ML RMS gain dB (CRB) | oracle ALS 4 it phase/gain MSE / known-angle CRB | oracle ALS 20 it |', flush=True)
    print('|---|---|---|---|---|---|---|', flush=True)
    for snr in [15.0, 25.0, 35.0]:
        s2 = 10 ** (-snr / 10); g = np.random.default_rng([9332, int(snr)])
        Z = np.sqrt(s2 / 2) * (g.standard_normal((n, 16, 16)) + 1j * g.standard_normal((n, 16, 16)))
        Y = MU + Z; ep = []; eg = []
        for i in range(n):
            th = TH[i].copy(); lam = 1e-6
            cost = lambda t: np.sum(np.abs(Y[i] - mu2(t, L)) ** 2)
            c0 = cost(th)
            for it in range(12):         # Levenberg-Marquardt (accept only cost decreases)
                r = (Y[i] - mu2(th, L)).ravel(); J = fd_jac(th, L)
                Jr = np.concatenate([J.real, J.imag]); rr = np.concatenate([r.real, r.imag])
                A = Jr.T @ Jr; gvec = Jr.T @ rr
                for _ in range(10):
                    step = np.linalg.solve(A + lam * np.diag(np.diag(A)), gvec); c1 = cost(th + step)
                    if np.isfinite(c1) and c1 <= c0:
                        th = th + step; c0 = c1; lam = max(lam / 10, 1e-9); break
                    lam *= 10
            e = ident_err(ident_from_theta(TH[i])[None], ident_from_theta(th)[None])[0]
            ep.append((e[:, 0] ** 2).mean()); eg.append((e[:, 1] ** 2).mean())
        ep = np.array(ep); eg = np.array(eg); cp = CR[:, 0] * s2; cg = CR[:, 1] * s2
        # oracle ALS (the e1_eval 'oracle ALS' estimator) on the same noisy channel, true angles fixed
        U0, V0 = np.pi * np.cos(psi), np.pi * np.cos(phi)
        tg_id = np.array([ident_from_theta(t) for t in TH])
        outs = []
        for its in [4, 20]:
            est = selfcal_ret(Y, U0, V0, 1.0, 0.0, iters=its, fix_angles=True)[2]
            e = ident_err(tg_id, est); outs.append(((e[..., 0] ** 2).mean(1).mean() / (CRk[:, 0] * s2).mean(), (e[..., 1] ** 2).mean(1).mean() / (CRk[:, 1] * s2).mean()))
        # SE of the ratio by bootstrap over scenes
        bt = np.random.default_rng(0).integers(0, n, (2000, n))
        rp = ep[bt].mean(1) / cp[bt].mean(1); rg = eg[bt].mean(1) / cg[bt].mean(1)
        print(f'| {snr:.0f} | {ep.mean() / cp.mean():.3f} [{np.percentile(rp, 2.5):.3f},{np.percentile(rp, 97.5):.3f}] | {eg.mean() / cg.mean():.3f} '
              f'[{np.percentile(rg, 2.5):.3f},{np.percentile(rg, 97.5):.3f}] | {np.rad2deg(np.sqrt(ep.mean())):.3f} ({np.rad2deg(np.sqrt(cp.mean())):.3f}) | '
              f'{DB * np.sqrt(eg.mean()):.4f} ({DB * np.sqrt(cg.mean()):.4f}) | {outs[0][0]:.3f}/{outs[0][1]:.3f} | {outs[1][0]:.3f}/{outs[1][1]:.3f} |', flush=True)
    # scaled reference: CRB at 0 dB from these scenes (compare to d2_crb clean-point table: phase 12.7 deg, gain 2.02 dB at 0 dB)
    print(f'   CRB RMS at 0 dB on these fresh scenes (30/6 point, unknown angles): phase {np.rad2deg(np.sqrt(CR[:, 0].mean())):.2f} deg, gain {DB * np.sqrt(CR[:, 1].mean()):.3f} dB; '
          f'known angles: {np.rad2deg(np.sqrt(CRk[:, 0].mean())):.2f} deg / {DB * np.sqrt(CRk[:, 1].mean()):.3f} dB', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
