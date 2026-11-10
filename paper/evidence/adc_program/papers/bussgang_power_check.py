"""bussgang_power_check.py - numerical checks for bussgang_power.md (Li, Tao, Seco-Granados, Mezghani, Swindlehurst, Liu,
IEEE TSP 2017, arXiv 1609.07427).  CPU, memory-light (2 threads, chunks of <= 100 scenes for decoding).

Part A (the paper's own setting, synthetic):
  A1  scalar Bussgang facts for the 1-bit quantiser Q(y) = (sgn Re y + j sgn Im y)/sqrt(2), y ~ CN(0,1):
      gain A = sqrt(2/pi) (eq. 11 with K rho_p + 1 = 1), q = r - A y uncorrelated with y, E|q|^2 = 1 - 2/pi (eq. 32), but q is NOT
      independent of y (its conditional power depends on |y|).
  A2  eq. (17)/(18): tau = K, M = 16, K = 4, DFT pilots, BLMMSE = Phi~^H r (eq. 16); Monte-Carlo NMSE vs 1 - 2K rho/(pi(K rho+1)).
  A3  Fig. 2 setting (M = 16, K = 4, tau = 20): BLMMSE with the arcsine law (eqs. 13, 14) and the additive-quantiser-noise variant
      of [28] (C_q = (1 - 2/pi) I, eqs. 27 + 32), NMSE at -20..20 dB.
Part B (our benchmark: 16x16 DFT beamspace, L = 3, stress_sim.adc peak AGC):
  B1  prior second-order statistics of g = vec(W^H H F) (clean beamspace) from prior banks (seeds 80,600,000 + c).
  B2  'Bussgang de-quantisation then NOMP' baselines on fresh test banks (seeds 12600 = 0 dB, 12615 = 15 dB, 1000 scenes):
        NOMP on raw quantised data | analytic Bussgang-LMMSE (Li et al. eq. 60 recipe with C_h -> prior C_g; 1 bit only) + NOMP |
        empirical LMMSE (Monte-Carlo C_gr C_r^-1 from the prior banks, any bit depth) + NOMP;
      NMSE of the de-quantised beamspace and paper-style Pd; paired bootstrap 2000 (seed 0).
  B3  NOMP at 40 dB (noise negligible) for inf / 4 / 2 / 1 bits (seed 12640, 500 scenes): an error floor that is caused by the
      quantiser distortion, not by noise.
All decoders are given the true L = 3 (as in finetune/adversarial/s3_adc_test.py). NOMP = common.nomp_greedy(H, 3, 4, 3).
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
T0 = time.time()
RES = {}


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def db(x):
    return 10 * np.log10(x)


def q1(y):
    """Li et al. eq. (2): one-bit quantiser, unit-power QPSK output"""
    return (np.sign(y.real) + 1j * np.sign(y.imag)) / np.sqrt(2)


# ============================================================== A1 scalar Bussgang facts
rng = np.random.default_rng(0)
n = 400000
y = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
r = q1(y)
A = np.vdot(y, r) / np.vdot(y, y)                    # E[r y*] / E|y|^2
q = r - A * y
small, big = np.abs(y) < 0.3, np.abs(y) > 1.5
a1 = dict(A=float(A.real), A_imag=float(A.imag), sqrt_2_over_pi=float(np.sqrt(2 / np.pi)),
          corr_q_y=float(np.abs(np.vdot(y, q) / n)), Eq2=float(np.mean(np.abs(q) ** 2)), one_minus_2_over_pi=float(1 - 2 / np.pi),
          Eq2_given_abs_y_lt_0p3=float(np.mean(np.abs(q[small]) ** 2)), Eq2_given_abs_y_gt_1p5=float(np.mean(np.abs(q[big]) ** 2)),
          SDR_inf_snr=float(np.abs(A) ** 2 / np.mean(np.abs(q) ** 2)), SDR_formula=float((2 / np.pi) / (1 - 2 / np.pi)))
log('A1 scalar 1-bit Bussgang (y ~ CN(0,1), n = 400000):')
for k, v in a1.items():
    log(f'   {k:28s} {v:.4f}')
log(f'   -> signal-to-distortion ratio of the linearised 1-bit output at infinite SNR = {a1["SDR_inf_snr"]:.3f} '
    f'({db(a1["SDR_inf_snr"]):.2f} dB); formula (2/pi)/(1-2/pi) = {a1["SDR_formula"]:.3f} ({db(a1["SDR_formula"]):.2f} dB)')
RES['A1'] = a1
del y, r, q


# ============================================================== A2 eq. (17)/(18), tau = K
def dft_pilots(tau, K):
    return np.exp(-2j * np.pi * np.outer(np.arange(tau), np.arange(K)) / tau)      # unit-modulus, Phi^T Phi* = tau I


def pilot_bank(n, M, K, tau, rho, rng):
    Phi = dft_pilots(tau, K)
    H = (rng.standard_normal((n, M, K)) + 1j * rng.standard_normal((n, M, K))) / np.sqrt(2)
    N = (rng.standard_normal((n, M, tau)) + 1j * rng.standard_normal((n, M, tau))) / np.sqrt(2)
    Yp = np.sqrt(rho) * H @ Phi.T + N                                              # eq. (3)
    h = H.transpose(0, 2, 1).reshape(n, K * M)                                       # vec (column stacking)
    yp = Yp.transpose(0, 2, 1).reshape(n, tau * M)
    Phibar = np.kron(Phi, np.sqrt(rho) * np.eye(M))                                 # eq. (4)
    assert np.allclose(yp - N.transpose(0, 2, 1).reshape(n, tau * M), h @ Phibar.T)
    return h, yp, Phibar


M, K = 16, 4
a2 = {}
log('A2 eq. (17): tau = K = 4, M = 16, BLMMSE = Phi~^H r (eq. 16), 2000 channels per SNR')
for snr in [-20, -10, 0, 10, 20, 40]:
    rho = 10 ** (snr / 10)
    h, yp, Phibar = pilot_bank(2000, M, K, K, rho, np.random.default_rng(100 + snr))
    alpha = np.sqrt(2 / np.pi) / np.sqrt(K * rho + 1)                              # eq. (11)
    hh = q1(yp) @ np.conj(alpha * Phibar)                                          # Phi~^H r, row form
    nmse = np.mean(np.abs(hh - h) ** 2)
    form = 1 - 2 * K * rho / (np.pi * (K * rho + 1))
    a2[snr] = dict(mc_db=float(db(nmse)), formula_db=float(db(form)))
    log(f'   SNR {snr:4d} dB: Monte-Carlo NMSE {db(nmse):6.2f} dB   eq.(17) {db(form):6.2f} dB')
log(f'   eq. (18) limit 1 - 2/pi = {db(1 - 2 / np.pi):.2f} dB')
RES['A2'] = a2


# ============================================================== A3 Fig. 2 setting, tau = 20
def arcsine_Cr(Cy):
    s = 1 / np.sqrt(np.real(np.diag(Cy)))
    X = np.clip(s[:, None] * Cy.real * s[None], -1, 1); Yi = np.clip(s[:, None] * Cy.imag * s[None], -1, 1)
    return (2 / np.pi) * (np.arcsin(X) + 1j * np.arcsin(Yi))                        # eq. (13)


def zscore_offdiag(emp, theory, N):
    """RMS of the off-diagonal deviation (real and imaginary parts) of an empirical covariance of unit-power sign outputs from the
    theoretical one, in units of the Monte-Carlo standard error sqrt(0.5/N) of one entry. ~1 (or below) = agreement up to sampling noise;
    >> 1 = systematic mismatch.  (Revision after run 1: a plain relative Frobenius error is dominated by sampling noise.)"""
    D = emp - theory; m = ~np.eye(len(D), dtype=bool)
    return float(np.sqrt(np.mean(np.concatenate([D.real[m], D.imag[m]]) ** 2)) / np.sqrt(0.5 / N))


tau = 20
a3 = {}
log('A3 Fig. 2 setting: M = 16, K = 4, tau = 20 (DFT pilots), 1000 channels per SNR')
for snr in [-20, -10, 0, 10, 20]:
    rho = 10 ** (snr / 10)
    h, yp, Phibar = pilot_bank(1000, M, K, tau, rho, np.random.default_rng(200 + snr))
    Cy = Phibar @ Phibar.conj().T + np.eye(M * tau)
    alpha = np.sqrt(2 / np.pi) / np.sqrt(K * rho + 1)
    assert np.allclose(np.diag(Cy).real, K * rho + 1)                             # basis of eq. (11)
    Pt = alpha * Phibar                                                             # Phi~ = A_p Phibar
    Cr = arcsine_Cr(Cy)
    Cr_aqn = Pt @ Pt.conj().T + (alpha ** 2 + 1 - 2 / np.pi) * np.eye(M * tau)      # eqs. (27) + (32), model of [28]
    rp = q1(yp)
    out = {}
    for name, R in [('BLMMSE (arcsine law)', Cr), ('additive quantiser noise [28]', Cr_aqn)]:
        Wt = np.linalg.solve(R, Pt).conj()                                         # rows: h^ = Phi~^H R^-1 r  (R Hermitian)
        hh = rp @ Wt
        out[name] = float(db(np.mean(np.abs(hh - h) ** 2)))
    emp = np.einsum('ni,nj->ij', rp, rp.conj()) / len(rp)
    out['z_arcsine_vs_empirical_Cr'] = zscore_offdiag(emp, Cr, len(rp))
    a3[snr] = out
    log(f'   SNR {snr:4d} dB: BLMMSE {out["BLMMSE (arcsine law)"]:6.2f} dB   AQN[28] {out["additive quantiser noise [28]"]:6.2f} dB'
        f'   (arcsine-law C_r vs empirical C_r: off-diagonal RMS deviation = {out["z_arcsine_vs_empirical_Cr"]:.2f} Monte-Carlo s.e.)')
RES['A3'] = a3


# ============================================================== B1 prior statistics of our beamspace
def clean_and_noisy(nb, snr, seed):
    b = SS.base_scenes(nb, 3, float(snr), seed)
    Yn = SS.generate(b, 'clean')
    return b, Yn - b['noise'], Yn


def rq(Yq, bits):
    """receiver-side representation of the quantised beamspace: 1 bit -> Li et al.'s unit-power signs; b >= 2 -> the ADC levels on
    the original scale (stress_sim.adc undoes the AGC gain)."""
    return q1(Yq) if bits == 1 else Yq


SNRS, BITS = [0, 15], [1, 2]
NPRIOR, CH = 8000, 500
stats = {}
for snr in SNRS:
    Sgg = np.zeros((256, 256), complex); Sgr = {b: np.zeros((256, 256), complex) for b in BITS}; Srr = {b: np.zeros((256, 256), complex) for b in BITS}
    mg = np.zeros(256, complex)
    for c in range(NPRIOR // CH):
        _, G, Yn = clean_and_noisy(CH, snr, 80_600_000 + c)
        g = G.reshape(CH, 256); Sgg += g.T @ g.conj(); mg += g.sum(0)
        for bits in BITS:
            rr = rq(SS.adc(Yn, bits, 'peak'), bits).reshape(CH, 256)
            Sgr[bits] += g.T @ rr.conj(); Srr[bits] += rr.T @ rr.conj()
    Cg = Sgg / NPRIOR
    stats[snr] = dict(Cg=Cg, Cgr={b: Sgr[b] / NPRIOR for b in BITS}, Cr={b: Srr[b] / NPRIOR for b in BITS})
    # Gaussian surrogate with the SAME second-order statistics (g_s ~ CN(0, C_g)): control for the Bussgang/arcsine formulas
    ev, U = np.linalg.eigh(Cg); Lc = U * np.sqrt(np.clip(ev, 0, None))[None]
    gs_rng = np.random.default_rng(80_690_000 + snr); s2 = 10 ** (-snr / 10)
    Sgr_s = np.zeros((256, 256), complex); Srr_s = np.zeros((256, 256), complex)
    for c in range(NPRIOR // CH):
        w = (gs_rng.standard_normal((CH, 256)) + 1j * gs_rng.standard_normal((CH, 256))) / np.sqrt(2)
        gsur = w @ Lc.T
        z = np.sqrt(s2 / 2) * (gs_rng.standard_normal((CH, 256)) + 1j * gs_rng.standard_normal((CH, 256)))
        rs = q1(gsur + z)
        Sgr_s += gsur.T @ rs.conj(); Srr_s += rs.T @ rs.conj()
    stats[snr]['Cgr_sur'] = Sgr_s / NPRIOR; stats[snr]['Cr_sur'] = Srr_s / NPRIOR
    d = np.real(np.diag(Cg)); off = Cg - np.diag(np.diag(Cg))
    b1 = dict(diag_mean=float(d.mean()), diag_min=float(d.min()), diag_max=float(d.max()),
              offdiag_energy_fraction=float(np.linalg.norm(off) ** 2 / np.linalg.norm(Cg) ** 2),
              mean_abs_over_rms=float(np.abs(mg / NPRIOR).max() / np.sqrt(d.mean())))
    RES.setdefault('B1', {})[snr] = b1
    log(f'B1 prior C_g ({NPRIOR} scenes, SNR {snr} dB): per-beam power mean {d.mean():.3f}, min {d.min():.3f}, max {d.max():.3f}; '
        f'off-diagonal energy fraction {b1["offdiag_energy_fraction"]:.3f}; max |mean|/rms {b1["mean_abs_over_rms"]:.3f}')


def estimators(snr, bits):
    """returns dict name -> 256x256 matrix Wm with g^ = Wm r (column-vector convention)"""
    st = stats[snr]; Cg = st['Cg']; s2 = 10 ** (-snr / 10)
    E = {}
    E['empirical LMMSE'] = np.linalg.solve(st['Cr'][bits].T, st['Cgr'][bits].T).T          # C_gr C_r^-1
    if bits == 1:
        Cy = Cg + s2 * np.eye(256)
        Ab = np.sqrt(2 / np.pi) * np.diag(1 / np.sqrt(np.real(np.diag(Cy))))                 # eq. (23)
        Cr = arcsine_Cr(Cy)                                                                 # eq. (25)
        E['Bussgang LMMSE (analytic)'] = np.linalg.solve(Cr.T, (Cg @ Ab.conj().T).T).T     # eq. (60) with Phibar = I, C_h = C_g
        Cr_aqn = Ab @ Cy @ Ab.conj().T + (1 - 2 / np.pi) * np.eye(256)                     # [28]-style diagonal distortion
        E['Bussgang LMMSE, diagonal distortion'] = np.linalg.solve(Cr_aqn.T, (Cg @ Ab.conj().T).T).T
        Cg_w = (np.trace(Cg).real / 256) * np.eye(256); Cy_w = Cg_w + s2 * np.eye(256)        # control: white prior
        Ab_w = np.sqrt(2 / np.pi) * np.diag(1 / np.sqrt(np.real(np.diag(Cy_w))))
        E['Bussgang LMMSE, white prior (control)'] = np.linalg.solve(arcsine_Cr(Cy_w).T, (Cg_w @ Ab_w.conj().T).T).T
        an = np.real(np.diag(Cg @ Ab.conj().T))
        ratio_real = np.real(np.diag(st['Cgr'][1])) / an; ratio_sur = np.real(np.diag(st['Cgr_sur'])) / an
        pw = np.real(np.diag(Cg))
        chk = dict(z_arcsine_Cr_real_beamspace=zscore_offdiag(st['Cr'][1], Cr, NPRIOR),
                   z_arcsine_Cr_gaussian_surrogate=zscore_offdiag(st['Cr_sur'], Cr, NPRIOR),
                   bussgang_gain_ratio_real_median=float(np.median(ratio_real)), bussgang_gain_ratio_real_p5=float(np.percentile(ratio_real, 5)),
                   bussgang_gain_ratio_real_p95=float(np.percentile(ratio_real, 95)),
                   bussgang_gain_ratio_real_strongest10pct_beams=float(np.median(ratio_real[pw >= np.percentile(pw, 90)])),
                   bussgang_gain_ratio_surrogate_median=float(np.median(ratio_sur)), bussgang_gain_ratio_surrogate_p5=float(np.percentile(ratio_sur, 5)),
                   bussgang_gain_ratio_surrogate_p95=float(np.percentile(ratio_sur, 95)))
        RES.setdefault('B1_checks', {})[f'{snr}dB'] = chk
        log(f'   1-bit, {snr} dB, Gaussian-input check of the Bussgang formulas ({NPRIOR} scenes):')
        log(f'      arcsine-law C_r vs empirical C_r, off-diagonal RMS deviation in Monte-Carlo s.e.: real beamspace '
            f'{chk["z_arcsine_Cr_real_beamspace"]:.2f}, Gaussian surrogate (same C_g) {chk["z_arcsine_Cr_gaussian_surrogate"]:.2f}')
        log(f'      per-beam ratio empirical E[g_i r_i*] / Bussgang C_g A^H: real beamspace median {chk["bussgang_gain_ratio_real_median"]:.3f} '
            f'(5-95%: {chk["bussgang_gain_ratio_real_p5"]:.3f}-{chk["bussgang_gain_ratio_real_p95"]:.3f}; strongest 10% beams '
            f'{chk["bussgang_gain_ratio_real_strongest10pct_beams"]:.3f}); Gaussian surrogate median {chk["bussgang_gain_ratio_surrogate_median"]:.3f} '
            f'(5-95%: {chk["bussgang_gain_ratio_surrogate_p5"]:.3f}-{chk["bussgang_gain_ratio_surrogate_p95"]:.3f})')
    # best single scalar (raw data, rescaled): c = tr(C_gr)/tr(C_r)
    E['raw (scalar LMMSE)'] = (np.trace(st['Cgr'][bits]) / np.trace(st['Cr'][bits])) * np.eye(256)
    return E


def as_list(p):
    return [(p[0][i], p[1][i]) for i in range(len(p[0]))]


def nomp_angles(Gh):
    H = SS.to_ant(Gh)
    return as_list(C.uv_to_angles(*C.nomp_greedy(H, 3, 4, 3)))


def boot_diff(Sa, Sb, snr_arr):
    Bt = C.boot_metrics({'a': Sa, 'b': Sb}, snr_arr, B=2000, seed=0)
    d = Bt['a'][:, 0] - Bt['b'][:, 0]
    pa = C.metric_by_snr(Sa, snr_arr)['mean']['pd_paper']; pb = C.metric_by_snr(Sb, snr_arr)['mean']['pd_paper']
    return [float(pa - pb), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]


# ============================================================== B2 test banks
NTEST, DCH = 1000, 100
for snr in SNRS:
    seed = 12600 + snr
    b, G, Yn = clean_and_noisy(NTEST, snr, seed)
    bank = dict(psi=b['psi'], phi=b['phi'], L=b['L'])
    snr_arr = np.full(NTEST, float(snr))
    for bits in BITS:
        E = estimators(snr, bits)
        Yq = SS.adc(Yn, bits, 'peak'); rr = rq(Yq, bits).reshape(NTEST, 256); g = G.reshape(NTEST, 256)
        preds = {'NOMP on raw quantised data': []}; nm = {}
        t = time.time()
        for s in range(0, NTEST, DCH):
            preds['NOMP on raw quantised data'] += nomp_angles(Yq[s:s + DCH])
        tn = (time.time() - t) / NTEST * 1e3
        for name, Wm in E.items():
            gh = rr @ Wm.T
            nm[name] = float(db(np.sum(np.abs(gh - g) ** 2) / np.sum(np.abs(g) ** 2)))
            if name == 'raw (scalar LMMSE)':
                continue
            key = f'{name} + NOMP'; preds[key] = []
            Gh = gh.reshape(NTEST, 16, 16)
            for s in range(0, NTEST, DCH):
                preds[key] += nomp_angles(Gh[s:s + DCH])
        S = {k: C.scene_stats(v, bank) for k, v in preds.items()}
        cell = dict(nmse_db=nm, nomp_ms_per_scene=tn, pd={}, pd_strict={}, diff_vs_nomp={}, identical_predictions={})
        for k in S:
            m = C.metric_from_stats(S[k], np.arange(NTEST)); cell['pd'][k] = float(m['pd_paper']); cell['pd_strict'][k] = float(m['pd_strict'])
            if k != 'NOMP on raw quantised data':
                cell['diff_vs_nomp'][k] = boot_diff(S[k], S['NOMP on raw quantised data'], snr_arr)
                same = [np.allclose(np.sort(a[0]), np.sort(c[0])) and np.allclose(np.sort(a[1]), np.sort(c[1]))
                        for a, c in zip(preds[k], preds['NOMP on raw quantised data'])]
                cell['identical_predictions'][k] = float(np.mean(same))
        RES.setdefault('B2', {})[f'{bits}bit_{snr}dB'] = cell
        log(f'B2 {bits}-bit / {snr} dB (seed {seed}, {NTEST} scenes): NMSE of de-quantised beamspace (dB): ' +
            ', '.join(f'{k} {v:.2f}' for k, v in nm.items()))
        for k in S:
            extra = '' if k not in cell['diff_vs_nomp'] else (
                f'   diff vs NOMP {cell["diff_vs_nomp"][k][0]:+.4f} [{cell["diff_vs_nomp"][k][1]:+.4f}, {cell["diff_vs_nomp"][k][2]:+.4f}]'
                f'   identical scene predictions {cell["identical_predictions"][k]:.3f}')
            log(f'   Pd {cell["pd"][k]:.4f} (strict {cell["pd_strict"][k]:.4f})  {k}{extra}')
        json.dump(RES, open(f'{OUT}/bussgang_power_check.json', 'w'), indent=1, default=float)

# ============================================================== B3 high-SNR floor
b, G, Yn = clean_and_noisy(500, 40, 12640)
bank = dict(psi=b['psi'], phi=b['phi'], L=b['L'])
b3 = {}
for bits in [None, 4, 2, 1]:
    Yx = Yn if bits is None else SS.adc(Yn, bits, 'peak')
    pr = []
    for s in range(0, 500, DCH):
        pr += nomp_angles(Yx[s:s + DCH])
    m = C.metric_from_stats(C.scene_stats(pr, bank), np.arange(500))
    b3['inf' if bits is None else bits] = dict(pd=float(m['pd_paper']), pd_strict=float(m['pd_strict']), rmse=float(m['rmse']))
    log(f'B3 40 dB, bits {"inf" if bits is None else bits}: NOMP Pd {m["pd_paper"]:.4f} (strict {m["pd_strict"]:.4f}, RMSE {m["rmse"]:.3f} deg)')
RES['B3'] = b3

# ============================================================== C Walden (1999) figure of merit: arithmetic only
Fmean = 7.9e10                                       # Walden Sec. V: most ADCs have F <= 7.9e10 (the mean), F = 2^SNRbits f_samp / P_diss (eq. 4)
c = dict(energy_per_conversion_step_pJ=1e12 / Fmean,
         walden_example1_F=2 ** 13 * 10e6 / 1.1, walden_example2_F=2 ** 7 * 1e9 / 1.75,
         implied_Pdiss_W_at_1GSps={b: 2 ** b * 1e9 / Fmean for b in [1, 2, 3, 4, 6, 8, 10, 12]})
log(f'C  Walden FOM arithmetic: 1/F(mean) = {c["energy_per_conversion_step_pJ"]:.1f} pJ per conversion step; '
    f'F of Walden example 1 (13 SNR-bits, 10 Ms/s, 1.1 W) = {c["walden_example1_F"]:.3g}; example 2 (7 SNR-bits, 1 Gs/s, 1.75 W) = {c["walden_example2_F"]:.3g}')
log('   implied P_diss = 2^b f_s / F at f_s = 1 Gs/s and F = 7.9e10 (constant-FOM extrapolation, NOT a statement of Walden): ' +
    ', '.join(f'{b} b: {p:.3g} W' for b, p in c['implied_Pdiss_W_at_1GSps'].items()))
RES['C'] = c
json.dump(RES, open(f'{OUT}/bussgang_power_check.json', 'w'), indent=1, default=float)
log('done')
