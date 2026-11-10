"""Step 1: sign/convention checks on NOISELESS single scenes (seed 7700, L=3).
 (a) to_ant(Y) == conj(D_r) H D_t exactly (noise 0).
 (b) applying the TRUE identifiable correction (impairment_targets) the way IABRv2.front does (Hc = Ht * c_r * c_t, c = exp(-lg - j ph))
     leaves a residual distortion whose identifiable part is ~0 (and the opposite sign doubles it).
 (c) dB conversion: dB = 20/ln(10) * natural-log gain, checked against 20*log10|g|.
 (d) self-calibration estimate convention: selfcal_ret from TRUE angles, gate open, noiseless -> its estimate matches the target
     (same representation as IABC's imp output); selfcal_ret reproduces the notebook selfcal angles exactly.
 (e) IABC's imp output has the same sign as the target (error after < before on noiseless impaired scenes).
usage: python s1_signcheck.py > s1_signcheck.log"""
from icommon import *

rng = np.random.default_rng(7700); N = 64
psi, phi, alpha = S.sample_paths(np.full(N, 3), rng)
zero = np.zeros((N, 16, 16), complex)
for pd, gd in [(8.0, 3.0), (30.0, 6.0)]:
    out = S.simulate(psi, phi, alpha, 100.0, np.random.default_rng([7700, 3]), phase_deg_max=pd, gain_db_max=gd, noise=zero)
    Y, D_r, D_t, H = out['Y'], out['D_r'], out['D_t'], out['H']
    Ht = to_ant(Y)
    expect = np.conj(D_r)[:, :, None] * H * D_t[:, None, :]
    print(f'--- phase {pd} deg, gain {gd} dB, N={N} noiseless scenes')
    print(f'(a) max |to_ant(Y) - conj(D_r) H D_t| / max|H| = {np.abs(Ht - expect).max() / np.abs(H).max():.2e}')
    tgt = S.impairment_targets(D_r, D_t).astype(np.float64)
    for sgn, name in [(1, 'front() sign  c = exp(-lg - j ph)'), (-1, 'OPPOSITE sign c = exp(+lg + j ph)')]:
        c = np.exp(sgn * (-tgt[..., 1] - 1j * tgt[..., 0]))
        res_r = np.conj(D_r) * c[:, :16]; res_t = D_t * c[:, 16:]      # residual per-antenna distortion after correction
        res = np.stack([np.concatenate([np.angle(res_r), np.angle(res_t)], 1), np.concatenate([np.log(np.abs(res_r)), np.log(np.abs(res_t))], 1)], -1)
        e = ident_err(res, np.zeros_like(res))                         # identifiable part of the residual
        ph, g = rms_deg_db(e); ph0, g0 = rms_deg_db(tgt)
        print(f'(b) {name}: identifiable residual RMS phase {ph.mean():.2e} deg, gain {g.mean():.2e} dB  (before: {ph0.mean():.3f} deg, {g0.mean():.3f} dB)')
    # check Hc built that way equals H times only the unidentifiable distortion
    c = np.exp(-tgt[..., 1] - 1j * tgt[..., 0]); Hc = Ht * c[:, :16, None] * c[:, None, 16:]
    U0, V0 = true_uv(psi, phi)
    x, A = C.ls_gains(Hc, U0, V0); fit = np.einsum('bik,bk->bi', A, x).reshape(Hc.shape)
    x2, A2 = C.ls_gains(Ht, U0, V0); fit2 = np.einsum('bik,bk->bi', A2, x2).reshape(Ht.shape)
    # with only mean+ramp left, Hc is (scaled) an exact L-path channel at slightly shifted angles; with the true angles a ramp shows up as misfit
    print(f'    relative LS misfit of the L-path model at true angles: raw Ht {np.mean(np.linalg.norm((Ht - fit2).reshape(N, -1), axis=1) / np.linalg.norm(Ht.reshape(N, -1), axis=1)):.3f}'
          f' | true-corrected Hc {np.mean(np.linalg.norm((Hc - fit).reshape(N, -1), axis=1) / np.linalg.norm(Hc.reshape(N, -1), axis=1)):.3f}'
          f' (non-zero only because the unidentifiable ramp shifts the angles)')
    Ug, Vg = C.nomp_greedy(Hc, 3, 4, 3); x, A = C.ls_gains(Hc, Ug, Vg); fit = np.einsum('bik,bk->bi', A, x).reshape(Hc.shape)
    print(f'    relative LS misfit after NOMP on the true-corrected Hc (free angles): {np.mean(np.linalg.norm((Hc - fit).reshape(N, -1), axis=1) / np.linalg.norm(Hc.reshape(N, -1), axis=1)):.2e}')
    # (c) dB
    g_lin = np.abs(np.concatenate([D_r, D_t], 1))
    print(f'(c) max |20 log10|g| - (20/ln10) ln|g|| = {np.abs(20 * np.log10(g_lin) - DB * np.log(g_lin)).max():.2e} dB; 20/ln10 = {DB:.6f};'
          f' max |raw gain| drawn = {np.abs(20 * np.log10(g_lin)).max():.3f} dB (spec {gd})')
    # (d) self-cal convention
    _, _, est, gate, _ = selfcal_ret(Ht, U0, V0, 1.0, 0.0, iters=4, fix_angles=True)
    e = ident_err(tgt, est); ph, g = rms_deg_db(e)
    eneg = ident_err(tgt, -est); phn, gn = rms_deg_db(eneg)
    print(f'(d) oracle ALS (true angles, noiseless) target-est RMS: phase {ph.mean():.3e} deg, gain {g.mean():.3e} dB | target+est (wrong sign): {phn.mean():.3f} deg, {gn.mean():.3f} dB | gate open {gate.mean():.2f}')
    Ug, Vg = C.nomp_greedy(Ht, 3, 4, 3)
    Ua, Va = C.selfcal(Ht, Ug, Vg, 1.0, 5.0)
    Ub, Vb, est2, _, _ = selfcal_ret(Ht, Ug, Vg, 1.0, 5.0)
    print(f'    selfcal_ret vs notebook selfcal (NOMP start, (1,5), iters 2): max |dU|,|dV| = {max(np.abs(Ua - Ub).max(), np.abs(Va - Vb).max()):.1e}')
    e = ident_err(tgt, est2); ph, g = rms_deg_db(e)
    print(f'    self-cal from NOMP angles (noiseless): target-est RMS phase {ph.mean():.3f} deg, gain {g.mean():.3f} dB')
    # (e) IABC sign
    _, _, imp = net(Y)
    e = ident_err(tgt, imp); ph, g = rms_deg_db(e); en = ident_err(tgt, -imp); phn, gn = rms_deg_db(en); ph0, g0 = rms_deg_db(tgt)
    print(f'(e) IABC (noiseless): before {ph0.mean():.3f} deg / {g0.mean():.3f} dB | target-imp {ph.mean():.3f} deg / {g.mean():.3f} dB | target+imp {phn.mean():.3f} deg / {gn.mean():.3f} dB')
    # check that IABC output == the correction front() applies
    c = np.exp(-imp[..., 1] - 1j * imp[..., 0]); _, Hc_net, _ = net(Y)
    print(f'    front() Hc == Ht * exp(-imp_lg - j imp_ph): max rel diff {np.abs(Hc_net - Ht * c[:, :16, None] * c[:, None, 16:]).max() / np.abs(Ht).max():.1e}')
