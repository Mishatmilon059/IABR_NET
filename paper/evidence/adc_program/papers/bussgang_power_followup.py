"""bussgang_power_followup.py - two follow-up checks for bussgang_power.md (after bussgang_power_check.log).
F1  Li et al. Fig. 2 setting (M = 16, K = 4, tau = 20): does the BLMMSE NMSE at high SNR depend on WHICH K columns of the 20x20 DFT
    matrix are used as pilots? (The paper says 'K columns of the tau x tau DFT matrix' without naming them; our first
    re-implementation used columns 0..3 and gave about -7.9 dB at 10-20 dB, while the plotted curve sits near -7.2 dB by eye.)
F2  Why does linear (LMMSE) de-quantisation hurt NOMP on our benchmark? 1-bit and 2-bit at 15 dB (test seed 12615, 1000 scenes, same as
    bussgang_power_check.py): (a) diagonal-only part of the empirical LMMSE matrix (pure per-beam re-weighting), (b) where the estimated
    paths land: mean |u| = |cos(angle)| of all estimated AoA/AoD vs the truth (endfire = |u| near 1, where the prior beam power is largest).
Memory-light: 2 threads, decoding chunks of 100 scenes.
"""
import os, sys, time, json
for _k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[_k] = '2'
os.environ.setdefault('IABR2_SUITE', 'D:/ai_ml_project')
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results'); sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/stress')
sys.path.insert(0, 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2')
import numpy as np
import stress_sim as SS
import torch
torch.set_num_threads(2)
import common as C

OUT = os.path.dirname(os.path.abspath(__file__))
T0 = time.time(); RES = {}


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def db(x):
    return 10 * np.log10(x)


def q1(y):
    return (np.sign(y.real) + 1j * np.sign(y.imag)) / np.sqrt(2)


def arcsine_Cr(Cy):
    s = 1 / np.sqrt(np.real(np.diag(Cy)))
    X = np.clip(s[:, None] * Cy.real * s[None], -1, 1); Yi = np.clip(s[:, None] * Cy.imag * s[None], -1, 1)
    return (2 / np.pi) * (np.arcsin(X) + 1j * np.arcsin(Yi))


# ============================================================== F1 pilot-column choice in the Fig. 2 setting
M, K, tau = 16, 4, 20
D20 = np.exp(-2j * np.pi * np.outer(np.arange(tau), np.arange(tau)) / tau)
SETS = {'adjacent 0,1,2,3': [0, 1, 2, 3], 'equispaced 0,5,10,15': [0, 5, 10, 15], 'mixed 0,3,7,12': [0, 3, 7, 12], 'pairs 0,1,10,11': [0, 1, 10, 11]}
f1 = {}
log('F1 Fig. 2 setting (M 16, K 4, tau 20), BLMMSE NMSE (dB) vs pilot columns, 2000 channels per SNR')
for name, cols in SETS.items():
    Phi = D20[:, cols]
    assert np.allclose(Phi.T @ Phi.conj(), tau * np.eye(K))
    row = {}
    for snr in [-10, 0, 10, 20]:
        rho = 10 ** (snr / 10); rng = np.random.default_rng(300 + snr)
        n = 2000
        H = (rng.standard_normal((n, M, K)) + 1j * rng.standard_normal((n, M, K))) / np.sqrt(2)
        N = (rng.standard_normal((n, M, tau)) + 1j * rng.standard_normal((n, M, tau))) / np.sqrt(2)
        Yp = np.sqrt(rho) * H @ Phi.T + N
        h = H.transpose(0, 2, 1).reshape(n, K * M); yp = Yp.transpose(0, 2, 1).reshape(n, tau * M)
        Phibar = np.kron(Phi, np.sqrt(rho) * np.eye(M))
        Cy = Phibar @ Phibar.conj().T + np.eye(M * tau)
        alpha = np.sqrt(2 / np.pi) / np.sqrt(K * rho + 1); Pt = alpha * Phibar
        Wt = np.linalg.solve(arcsine_Cr(Cy), Pt).conj()
        row[snr] = float(db(np.mean(np.abs(q1(yp) @ Wt - h) ** 2)))
    f1[name] = row
    log(f'   {name:22s} ' + '  '.join(f'{s:3d} dB: {v:6.2f}' for s, v in row.items()))
RES['F1'] = f1


# ============================================================== F2 why LMMSE de-quantisation hurts NOMP
def clean_and_noisy(nb, snr, seed):
    b = SS.base_scenes(nb, 3, float(snr), seed)
    Yn = SS.generate(b, 'clean')
    return b, Yn - b['noise'], Yn


def rq(Yq, bits):
    return q1(Yq) if bits == 1 else Yq


def as_list(p):
    return [(p[0][i], p[1][i]) for i in range(len(p[0]))]


def nomp_angles(Gh):
    return as_list(C.uv_to_angles(*C.nomp_greedy(SS.to_ant(Gh), 3, 4, 3)))


def boot_diff(Sa, Sb, snr_arr):
    Bt = C.boot_metrics({'a': Sa, 'b': Sb}, snr_arr, B=2000, seed=0)
    d = Bt['a'][:, 0] - Bt['b'][:, 0]
    pa = C.metric_by_snr(Sa, snr_arr)['mean']['pd_paper']; pb = C.metric_by_snr(Sb, snr_arr)['mean']['pd_paper']
    return [float(pa - pb), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]


snr, NPRIOR, CH, NTEST, DCH = 15, 8000, 500, 1000, 100
Sgr = {b: np.zeros((256, 256), complex) for b in [1, 2]}; Srr = {b: np.zeros((256, 256), complex) for b in [1, 2]}
for c in range(NPRIOR // CH):                                    # identical prior banks to bussgang_power_check.py
    _, G, Yn = clean_and_noisy(CH, snr, 80_600_000 + c)
    g = G.reshape(CH, 256)
    for bits in [1, 2]:
        rr = rq(SS.adc(Yn, bits, 'peak'), bits).reshape(CH, 256)
        Sgr[bits] += g.T @ rr.conj(); Srr[bits] += rr.T @ rr.conj()
b, G, Yn = clean_and_noisy(NTEST, snr, 12615)
bank = dict(psi=b['psi'], phi=b['phi'], L=b['L']); snr_arr = np.full(NTEST, float(snr))
true_u = np.abs(np.cos(np.concatenate([b['psi'][:, :3].ravel(), b['phi'][:, :3].ravel()])))
log(f'F2 {snr} dB, seed 12615, {NTEST} scenes: truth mean |u| = {true_u.mean():.3f}, fraction |u| > 0.9 = {(true_u > 0.9).mean():.3f}')
f2 = {}
for bits in [1, 2]:
    Wm = np.linalg.solve((Srr[bits] / NPRIOR).T, (Sgr[bits] / NPRIOR).T).T
    Yq = SS.adc(Yn, bits, 'peak'); rr = rq(Yq, bits).reshape(NTEST, 256)
    variants = {'NOMP on raw quantised data': Yq, 'empirical LMMSE + NOMP': (rr @ Wm.T).reshape(NTEST, 16, 16),
                'diagonal part of empirical LMMSE + NOMP': (rr * np.diag(Wm)[None]).reshape(NTEST, 16, 16)}
    cell = {}
    S = {}
    for name, Z in variants.items():
        pr = []
        for s in range(0, NTEST, DCH):
            pr += nomp_angles(Z[s:s + DCH])
        S[name] = C.scene_stats(pr, bank)
        est_u = np.abs(np.cos(np.concatenate([np.concatenate([p[0] for p in pr]), np.concatenate([p[1] for p in pr])])))
        cell[name] = dict(pd=float(C.metric_from_stats(S[name], np.arange(NTEST))['pd_paper']), mean_abs_u=float(np.nanmean(est_u)),
                          frac_abs_u_gt_0p9=float(np.nanmean(est_u > 0.9)))
    for name in variants:
        if name != 'NOMP on raw quantised data':
            cell[name]['diff_vs_nomp'] = boot_diff(S[name], S['NOMP on raw quantised data'], snr_arr)
    wd = np.abs(np.diag(Wm)); cell['diag_weight_spread_max_over_min'] = float(wd.max() / wd.min())
    cell['offdiag_energy_fraction_of_Wm'] = float(1 - np.sum(np.abs(np.diag(Wm)) ** 2) / np.sum(np.abs(Wm) ** 2))
    f2[f'{bits}bit'] = cell
    log(f'   {bits}-bit: LMMSE matrix diagonal |weight| max/min = {cell["diag_weight_spread_max_over_min"]:.2f}, '
        f'off-diagonal energy fraction of the matrix = {cell["offdiag_energy_fraction_of_Wm"]:.3f}')
    for name in variants:
        e = cell[name]; extra = '' if 'diff_vs_nomp' not in e else f'   diff vs NOMP {e["diff_vs_nomp"][0]:+.4f} [{e["diff_vs_nomp"][1]:+.4f}, {e["diff_vs_nomp"][2]:+.4f}]'
        log(f'      Pd {e["pd"]:.4f}  mean |u| of estimates {e["mean_abs_u"]:.3f}  fraction |u| > 0.9 {e["frac_abs_u_gt_0p9"]:.3f}  {name}{extra}')
RES['F2'] = f2
json.dump(RES, open(f'{OUT}/bussgang_power_followup.json', 'w'), indent=1, default=float)
log('done')
