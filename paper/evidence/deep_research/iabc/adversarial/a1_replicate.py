"""Adversarial replication of the IABC claims on FRESH seeds (independent error metric, notebook decode() called directly).
Seeds (all disjoint from 5000/5015, 7700-7707, 9900/9915):
  sweep        scenes default_rng(8800+k), k = SNR index (SNR -10..25 step 5), noise default_rng([seed,1]), impairments default_rng([seed,3])
  L-robustness scenes default_rng(8830+j), j over (L, SNR) pairs, same noise/impairment sub-seeds
Per scene: identifiable per-antenna error, computed HERE with an independent least-squares projection (mean + linear ramp for phase,
mean for log-gain), NOT with iabr2_sim.impairment_targets / icommon.ident_err.
Estimators: before | IABC (CNN imp output) | self-cal tuned (lam 1, tau 2, 4 it, NOMP start; 0 where gate shut) |
  self-cal known-SNR gate (lam 1, tau 0, 4 it, applied only when nominal SNR >= 10 dB; threshold fixed a priori from the 7700-seed sweep) |
  IABC + self-cal tuned on Hc | oracle ALS (true angles, lam 1, tau 0, 4 it).
Pd methods (true L): NOMP | IABC+NOMP | IABR-Net heat = C.decode(Hc,L,heat,'heat',(1,5)) | IABR-v2 dual = C.decode(...,'dual',(1,5)) |
  NOMP+SC tuned | NOMP+SC snrgate | IABC+NOMP+SC tuned | IABC+NOMP+SC snrgate.
Part 2: official ideal-hardware bank (8000 scenes): Y rebuilt from cached Ht (Y = W^H Ht F, codebooks unitary), CNN re-run here,
fresh Hc compared with cached Hc, NOMP re-run on Ht and on fresh Hc (no cached predictions used).
Checkpoints: cache/a1__<cell>.npz, cache/a1_official.npz (resume: existing files are skipped).
usage: python -u a1_replicate.py > a1_replicate.log   (cells g1, g2 were added after the first run; that rerun skipped the cached cells and logged to a1_replicate_part2.log)"""
import os, sys, time
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/iabc')
from icommon import *          # sets threads 6, loads CNN (final.pt, read only), C = new_results/common, S = iabr2_sim

ADV = os.path.dirname(os.path.abspath(__file__)); AC = f'{ADV}/cache'; os.makedirs(AC, exist_ok=True)
SNRS = [-10, -5, 0, 5, 10, 15, 20, 25]; NPER = 200; CH = 400; GATE_DB = 10.0
CELLS = [('clean', 0, 0), ('ph8', 8, 0), ('g3', 0, 3), ('g6', 0, 6), ('g1', 0, 1), ('g2', 0, 2), ('c8_3', 8, 3), ('c20_4', 20, 4), ('c30_6', 30, 6), ('ph45', 45, 0)]
n_ = np.arange(16.) - 7.5
A_PH = np.stack([np.ones(16), n_], 1); P_PH = np.eye(16) - A_PH @ np.linalg.pinv(A_PH)   # removes mean + ramp
P_G = np.eye(16) - np.ones((16, 16)) / 16                                                 # removes mean


def ident(ph, lg):
    """ph, lg (B,32): identifiable part, per array (rows 0-15 Rx, 16-31 Tx)."""
    ph = np.concatenate([ph[:, :16] @ P_PH.T, ph[:, 16:] @ P_PH.T], 1); lg = np.concatenate([lg[:, :16] @ P_G.T, lg[:, 16:] @ P_G.T], 1)
    return ph, lg


def target(D_r, D_t):
    # antenna-domain distortion conj(D_r) H D_t: rows phase -eps_r, cols +eps_t; amplitude g_r, g_t
    return ident(np.concatenate([-np.angle(D_r), np.angle(D_t)], 1), np.concatenate([np.log(np.abs(D_r)), np.log(np.abs(D_t))], 1))


def err_mse(tph, tlg, est):
    """per-scene MSE over 32 antennas of the identifiable residual: phase deg^2, gain dB^2."""
    rph, rlg = ident(np.angle(np.exp(1j * (tph - est[..., 0]))), tlg - est[..., 1])
    return (np.rad2deg(rph) ** 2).mean(1), (((20 / np.log(10)) * rlg) ** 2).mean(1)


def make(seed, snr, n, L, pdeg, gdb):
    rng = np.random.default_rng(seed); psi, phi, alpha = S.sample_paths(np.full(n, L), rng)
    g = np.random.default_rng([seed, 1]); s = np.sqrt(10 ** (-snr / 10) / 2)
    noise = s * (g.standard_normal((n, 16, 16)) + 1j * g.standard_normal((n, 16, 16)))
    o = S.simulate(psi, phi, alpha, float(snr), np.random.default_rng([seed, 3]), phase_deg_max=pdeg, gain_db_max=gdb, noise=noise)
    return psi, phi, o


def process(Y, D_r, D_t, psi, phi, L, snr):
    Ht = np.matmul(np.matmul(S.W_IDEAL[None], Y), S.F_IDEAL.conj().T[None])
    heat, Hc, imp = net(Y)
    c = np.exp(-imp[..., 1] - 1j * imp[..., 0]); chk = np.abs(Hc - Ht * c[:, :16, None] * c[:, None, 16:]).max() / np.abs(Ht).max()
    tph, tlg = target(D_r, D_t)
    Ug, Vg = C.nomp_greedy(Ht, L, 4, 3); Uc, Vc = C.nomp_greedy(Hc, L, 4, 3)
    hi = snr >= GATE_DB
    Ut, Vt, est_t, gate_t, _ = selfcal_ret(Ht, Ug, Vg, 1.0, 2.0, iters=4)
    Uo, Vo, est_o, _, _ = selfcal_ret(Ht, Ug, Vg, 1.0, 0.0, iters=4)
    Ugt, Vgt = np.where(hi[:, None], Uo, Ug), np.where(hi[:, None], Vo, Vg); est_g = np.where(hi[:, None, None], est_o, 0.0)
    Uct, Vct, est_ct, _, _ = selfcal_ret(Hc, Uc, Vc, 1.0, 2.0, iters=4)
    Uco, Vco, _, _, _ = selfcal_ret(Hc, Uc, Vc, 1.0, 0.0, iters=4)
    Ucg, Vcg = np.where(hi[:, None], Uco, Uc), np.where(hi[:, None], Vco, Vc)
    _, _, est_or, _, _ = selfcal_ret(Ht, *true_uv(psi, phi), 1.0, 0.0, iters=4, fix_angles=True)
    # cross-check selfcal_ret angles vs the notebook selfcal
    Un, Vn = C.selfcal(Ht, Ug, Vg, 1.0, 2.0, rc=2, iters=4); chk2 = max(np.abs(Un - Ut).max(), np.abs(Vn - Vt).max())
    ang = {'NOMP': C.uv_to_angles(Ug, Vg), 'IABC+NOMP': C.uv_to_angles(Uc, Vc),
           'IABR-Net heat': C.decode(Hc, L, heat, 'heat', (1.0, 5.0)), 'IABR-v2 dual': C.decode(Hc, L, heat, 'dual', (1.0, 5.0)),
           'NOMP+SC tuned': C.uv_to_angles(Ut, Vt), 'NOMP+SC snrgate': C.uv_to_angles(Ugt, Vgt),
           'IABC+NOMP+SC tuned': C.uv_to_angles(Uct, Vct), 'IABC+NOMP+SC snrgate': C.uv_to_angles(Ucg, Vcg)}
    rec = {}
    for k, e in {'before': np.zeros_like(imp), 'iabc': imp, 'sc_tun': est_t, 'sc_snrgate': est_g, 'iabc_sct': imp + est_ct, 'oracle': est_or}.items():
        rec[f'ph_{k}'], rec[f'g_{k}'] = err_mse(tph, tlg, e)
    b = dict(psi=psi, phi=phi, L=np.full(len(psi), L))
    for m, a in ang.items():
        rec[f'S_{m}'] = C.scene_stats([(a[0][i], a[1][i]) for i in range(len(psi))], b)
    rec['gate_tun'] = gate_t.astype(float)
    return rec, chk, chk2


def run_cell(tag, pdeg, gdb, plan):
    fn = f'{AC}/a1__{tag}.npz'
    if os.path.exists(fn):
        print(f'{tag}: cached, skip', flush=True); return
    t0 = time.time(); parts = []; snr_all = []; mx = [0.0, 0.0]
    for seed, snr, n, L in plan:
        psi, phi, o = make(seed, snr, n, L, pdeg, gdb)
        for c0 in range(0, n, CH):
            sl = slice(c0, c0 + CH)
            r, a, b2 = process(o['Y'][sl], o['D_r'][sl], o['D_t'][sl], psi[sl], phi[sl], L, np.full(len(psi[sl]), float(snr)))
            parts.append(r); mx = [max(mx[0], a), max(mx[1], b2)]; snr_all.append(np.full(len(psi[sl]), float(snr)))
        del o
    rec = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(fn, snr=np.concatenate(snr_all), phase_deg=pdeg, gain_db=gdb, **rec)
    print(f'{tag}: done {time.time() - t0:.0f}s | check Hc==Ht*exp(-imp) {mx[0]:.1e} | selfcal_ret vs notebook selfcal max|dU| {mx[1]:.1e}', flush=True)


t00 = time.time()
for name, pdeg, gdb in CELLS:
    run_cell(name, pdeg, gdb, [(8800 + k, s, NPER, 3) for k, s in enumerate(SNRS)])
# L-robustness of the gain-removal number (g3, in training range): L = 1 and 6 at SNR 0 and 15
j = 0
for L in [1, 6]:
    run_cell(f'g3_L{L}', 0, 3, [(8830 + 2 * (L == 6) + i, s, 300, L) for i, s in enumerate([0, 15])])

# ---------------------------------------------------------------- part 2: official bank, re-run from cached Ht
fn = f'{AC}/a1_official.npz'
if not os.path.exists(fn):
    t0 = time.time(); b = C.load_bank('official'); Ls = b['L'].astype(int); z = np.load(f'{C.CACHE}/net__official.npz')
    N = len(Ls); assert len(z['Ht']) == N
    Wh = S.W_IDEAL.conj().T; F = S.F_IDEAL
    Sn = np.zeros((N, 7)); Sc = np.zeros((N, 7)); dHc = 0.0
    for s0 in range(0, N, CH):
        sl = slice(s0, s0 + CH); Ht = z['Ht'][sl].astype(np.complex128)
        Y = np.matmul(np.matmul(Wh[None], Ht), F[None])
        heat, Hc, imp = net(Y)
        dHc = max(dHc, float(np.abs(Hc - z['Hc'][sl]).max() / np.abs(z['Hc'][sl]).max()))
        bb = dict(psi=b['psi'][sl], phi=b['phi'][sl], L=Ls[sl])
        for L in sorted(set(Ls[sl].tolist())):
            ii = np.where(Ls[sl] == L)[0]
            for H_, St in [(Ht, Sn), (Hc, Sc)]:
                U, V = C.nomp_greedy(H_[ii], int(L), 4, 3); a = C.uv_to_angles(U, V)
                St[s0 + ii] = C.scene_stats([(a[0][i], a[1][i]) for i in range(len(ii))], dict(psi=bb['psi'][ii], phi=bb['phi'][ii], L=bb['L'][ii]))
        del Ht, Y, heat, Hc, imp
    np.savez_compressed(fn, S_NOMP=Sn, S_IABC_NOMP=Sc, snr=b['snr'], dHc=dHc)
    print(f'official: done {time.time() - t0:.0f}s | max rel |fresh Hc - cached Hc| = {dHc:.1e}', flush=True)
else:
    print('official: cached, skip', flush=True)
print(f'all done {time.time() - t00:.0f}s', flush=True)
