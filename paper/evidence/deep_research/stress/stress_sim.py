"""Mismatch simulator for the stress tests (extends iabr2_sim.simulate; iabr2_sim.py itself is NOT modified).

Base scene (identical draw order to iabr2_sim.simulate, so an impairment-free bank built from seed s is bit-identical
to make_standard_bank(seed=s) of the v1 suite):
    rng = default_rng(seed); psi, phi, alpha = sample_paths(L, rng, lmax)          (exact original angle sampler)
    eps_r, eps_t, g_r, g_t = rng.uniform(-1, 1, (B,16)) x4                         (M3 per-antenna phase / gain draws)
    noise = s * (rng.standard_normal((B,16,16)) + 1j rng.standard_normal(...)),  s = sqrt(10^(-SNR/10) / 2)
All other impairment randomness comes from a SEPARATE generator default_rng([seed, code]) (code fixed per impairment), and the
same standard draws are reused across severities (common random numbers): severities differ only by a scale factor.

Forward model, applied in this order:
    H     = sqrt(256) sum_l alpha_l a_r(psi_l) a_t(phi_l)^H      a_r[n] = exp(-j pi cos(psi) x_n)/4, x_n = n (+ M2 errors) (or M8 spherical)
    H     <- C_r H C_t^T                                          (M1 mutual coupling)
    W, F  = diag(D_r) Wq, diag(D_t) Fq                            (M3 per-antenna gain/phase, M4 quantised phase shifters Wq, Fq)
    Y     = W^H H F + noise
    Y     <- Y * exp(j Phi),  Phi_qp ~ N(0, sigma^2) iid           (M6 per-measurement phase noise)
    Y     <- ADC_b(Y)                                             (M5 / M5c, per-scene AGC, mid-rise quantiser on Re and Im)
The decoder side always assumes the ideal model: Hant = to_ant(Y) = W_IDEAL Y F_IDEAL^H.
"""
import os, sys
os.environ.setdefault('IABR2_SUITE', 'D:/ai_ml_project')
sys.path.insert(0, 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2')
import numpy as np
import iabr2_sim as S

NR = NT = 16
W_IDEAL, F_IDEAL = S.W_IDEAL, S.F_IDEAL
_n = np.arange(16.)
LAMBDA = 299792458.0 / 28e9          # 28 GHz
DSPACE = LAMBDA / 2                  # half-wavelength spacing
KWAVE = 2 * np.pi / LAMBDA
IMP_CODE = dict(M1=11, M2=12, M4=14, M5=15, M6=16, M8=18)


def to_ant(Y):
    """identical to notebook_src.py line 274"""
    return np.einsum('nq,bqp,mp->bnm', W_IDEAL, Y, np.conj(F_IDEAL))


def quantise_codebook(Wc, bits):
    """b-bit phase shifters: the ideal 16-point DFT codebook has phases k*2pi/16 (k integer, checked); each phase index is rounded
    to the nearest multiple of 2^(4-b) (ties rounded up = counter-clockwise). Constant modulus kept. bits=4 returns Wc exactly
    (it is multiplied by exp(0j) = 1)."""
    k = np.angle(Wc * 4) / (2 * np.pi / 16)
    assert np.abs(k - np.round(k)).max() < 1e-9 and np.allclose(np.abs(Wc), 0.25)
    k = np.round(k).astype(int) % 16
    s = 2 ** (4 - bits)
    kq = (s * np.floor((k + s / 2) / s).astype(int)) if s > 1 else k
    return Wc * np.exp(1j * (2 * np.pi / 16) * (kq - k))


def coupling_matrix(rho, th1, th2):
    """banded symmetric Toeplitz: C[i,i]=1, C[i,i+-1]=rho e^{j th1}, C[i,i+-2]=(rho/2) e^{j th2}; th1/th2 (B,) -> (B,16,16)"""
    B = len(th1); C = np.zeros((B, 16, 16), complex); i = np.arange(16)
    C[:, i, i] = 1.0
    c1 = rho * np.exp(1j * th1); c2 = (rho / 2) * np.exp(1j * th2)
    for d, c in [(1, c1), (2, c2)]:
        C[:, i[:-d], i[d:]] = c[:, None]; C[:, i[d:], i[:-d]] = c[:, None]
    return C


def nearfield_steer(ang, r):
    """spherical-wavefront ULA response for a point at range r (m) and angle ang from the array axis (array centre = reference),
    element n at x_n = (n - 7.5) lambda/2. a[n] = (r/r_n) exp(-j k (r - r_n)) exp(-j pi cos(ang) 7.5) / 4; the last factor makes the
    far-field limit r -> inf equal the ideal exp(-j pi cos(ang) n)/4 exactly. ang (B,L) -> (B,L,16)."""
    x = (_n - 7.5) * DSPACE
    c = np.cos(ang)[..., None]
    rn = np.sqrt(r ** 2 - 2 * r * x * c + x ** 2)
    dr = (2 * r * x * c - x ** 2) / (r + rn)                     # r - r_n, numerically stable
    return (r / rn) * np.exp(-1j * KWAVE * dr - 1j * np.pi * c * 7.5) / 4


def base_scenes(n, L, snr_db, seed, lmax=None):
    """L: int or array (per scene). Returns dict of the base randomness (draw order identical to make_standard_bank + simulate)."""
    rng = np.random.default_rng(seed)
    L_arr = np.full(n, L) if np.isscalar(L) else np.asarray(L)
    psi, phi, alpha = S.sample_paths(L_arr, rng, lmax=lmax)
    B = n
    Ur = rng.uniform(-1, 1, (B, NR)); Ut = rng.uniform(-1, 1, (B, NT))
    Gr = rng.uniform(-1, 1, (B, NR)); Gt = rng.uniform(-1, 1, (B, NT))
    s = np.sqrt(10 ** (-np.broadcast_to(np.asarray(snr_db, float), (B,)) / 10) / 2)[:, None, None]
    noise = s * (rng.standard_normal((B, 16, 16)) + 1j * rng.standard_normal((B, 16, 16)))
    return dict(psi=psi, phi=phi, alpha=alpha, L=L_arr.astype(int), snr=np.full(n, float(snr_db)), Ur=Ur, Ut=Ut, Gr=Gr, Gt=Gt,
                noise=noise, seed=seed)


def imp_rng(seed, kind):
    return np.random.default_rng([seed, IMP_CODE[kind]])


def generate(base, kind='clean', sev=None):
    """kind in clean, M1 (rho), M2 (sigma, half-wavelengths), M3 ((phase_deg_max, gain_db_max)), M4 (bits), M5 / M5c (bits), M6 (sigma deg),
    M8 (range m). Returns Y (B,16,16) complex128."""
    psi, phi, alpha = base['psi'], base['phi'], base['alpha']
    B, Lm = alpha.shape
    ps = np.nan_to_num(psi); ph = np.nan_to_num(phi)
    xr = np.broadcast_to(_n, (B, NR)); xt = np.broadcast_to(_n, (B, NT))
    if kind == 'M2':
        g = imp_rng(base['seed'], 'M2')
        xr = _n[None] + sev * g.standard_normal((B, NR)); xt = _n[None] + sev * g.standard_normal((B, NT))
    if kind == 'M8':
        a_r = nearfield_steer(ps, sev); a_t = nearfield_steer(ph, sev)
    elif kind == 'M2':
        a_r = np.exp(-1j * np.pi * np.cos(ps)[..., None] * xr[:, None, :]) / np.sqrt(NR)
        a_t = np.exp(-1j * np.pi * np.cos(ph)[..., None] * xt[:, None, :]) / np.sqrt(NT)
    else:   # exact iabr2_sim.simulate expressions
        a_r = np.exp(-1j * np.pi * np.cos(ps)[..., None] * np.arange(NR)) / np.sqrt(NR)
        a_t = np.exp(-1j * np.pi * np.cos(ph)[..., None] * np.arange(NT)) / np.sqrt(NT)
    H = np.sqrt(NT * NR) * np.einsum('bl,bln,blm->bnm', alpha, a_r, np.conj(a_t))
    if kind == 'M1':
        g = imp_rng(base['seed'], 'M1'); th = g.uniform(0, 2 * np.pi, (4, B))
        Cr = coupling_matrix(sev, th[0], th[1]); Ct = coupling_matrix(sev, th[2], th[3])
        H = Cr @ H @ np.transpose(Ct, (0, 2, 1))
    pdm, gdm = (sev if kind == 'M3' else (0.0, 0.0))
    pdm = np.broadcast_to(np.asarray(pdm, float), (B,)); gdm = np.broadcast_to(np.asarray(gdm, float), (B,))
    eps_r = base['Ur'] * np.deg2rad(pdm)[:, None]; eps_t = base['Ut'] * np.deg2rad(pdm)[:, None]
    g_r = 10 ** (base['Gr'] * gdm[:, None] / 20); g_t = 10 ** (base['Gt'] * gdm[:, None] / 20)
    D_r = g_r * np.exp(1j * eps_r); D_t = g_t * np.exp(1j * eps_t)
    Wq, Fq = (quantise_codebook(W_IDEAL, sev), quantise_codebook(F_IDEAL, sev)) if kind == 'M4' else (W_IDEAL, F_IDEAL)
    W = D_r[:, :, None] * Wq[None]; F = D_t[:, :, None] * Fq[None]
    G = np.einsum('bnq,bnm,bmp->bqp', np.conj(W), H, F)
    Y = G + base['noise']
    if kind == 'M6':
        g = imp_rng(base['seed'], 'M6')
        Y = Y * np.exp(1j * np.deg2rad(sev) * g.standard_normal((B, 16, 16)))
    if kind == 'M5':
        Y = adc(Y, sev, 'peak')
    if kind == 'M5c':
        Y = adc(Y, sev, 'rms3')
    return Y


def adc(Y, bits, agc='peak'):
    """per-scene AGC + b-bit uniform mid-rise quantiser applied separately to Re and Im of the 256 beam measurements.
    Full scale A per scene:  agc='peak' -> A = max over the scene of max(|Re|, |Im|)  (no clipping; primary M5 setting)
                             agc='rms3' -> A = 3 x RMS of one real component, rms = sqrt(mean(|Y|^2)/2)  (task's suggestion; the
                                           beamspace peak/RMS ratio is ~12 (median), so the strongest beams are clipped: variant M5c)
    step D = 2A/2^b; output levels D(k + 1/2), k = -2^(b-1) .. 2^(b-1)-1 (clipping outside +-A); bits=1 -> sign(x) A/2.
    The output stays on the original scale (ideal AGC gain undone); the decoders are scale-invariant."""
    if agc == 'peak':
        A = np.maximum(np.abs(Y.real), np.abs(Y.imag)).max((1, 2))[:, None, None]
    else:
        A = 3.0 * np.sqrt((np.abs(Y) ** 2).mean((1, 2)) / 2)[:, None, None]
    D = 2 * A / 2 ** bits; M = 2 ** (bits - 1)

    def q(x):
        k = np.clip(np.floor(x / D), -M, M - 1)
        return D * (k + 0.5)
    return q(Y.real) + 1j * q(Y.imag)


SEVERITIES = {
    'M1': [0.05, 0.1, 0.2, 0.3],
    'M2': [0.02, 0.05, 0.1, 0.2],
    'M3': [(20.0, 0.0), (30.0, 0.0), (45.0, 0.0), (0.0, 6.0), (30.0, 4.0)],
    'M4': [4, 3, 2, 1],
    'M5': [4, 3, 2, 1],
    'M5c': [4, 3, 2, 1],
    'M6': [5.0, 10.0, 20.0, 30.0],
    'M8': [2.0, 1.0, 0.5],
}
NAMES = {'M1': 'mutual coupling (rho)', 'M2': 'antenna position error (sigma, lambda/2 units)', 'M3': 'large per-antenna impairment (phase deg max, gain dB max)',
         'M4': 'phase-shifter resolution (bits)', 'M5': 'ADC resolution (bits), peak AGC', 'M5c': 'ADC resolution (bits), AGC full scale 3 RMS (clipping)', 'M6': 'per-measurement phase noise (sigma deg)',
         'M8': 'near field (range m, 28 GHz)'}


def sev_label(kind, sev):
    if kind == 'M3':
        return f'ph{sev[0]:g}_g{sev[1]:g}'
    return f'{sev:g}'
