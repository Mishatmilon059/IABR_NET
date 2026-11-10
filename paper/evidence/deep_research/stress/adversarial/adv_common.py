"""Adversarial re-check of the stress tests: an INDEPENDENT forward model written from the spec (not importing stress_sim for generation),
the source-of-truth decoder (notebook_src.py slice [527:647] via new_results/common.py), the CNN (slice [419:522]) and the evaluator
(TVT_Blob_Inference.prepare_for_metric / get_ang_difference via common.scene_stats).
Draw scheme for FRESH banks (different from stress_sim on purpose):
    rng = default_rng(seed): psi, phi, alpha = iabr2_sim.sample_paths(L, rng)  (original sampler)
    noise   from default_rng([seed, 901])
    M3 eps/gain from the original iabr2_sim.simulate (rng = default_rng([seed, 903])), noise passed in
    other impairment draws from default_rng([seed, 910 + k])"""
import os, sys, json, time
os.environ.setdefault('IABR2_SUITE', 'D:/ai_ml_project')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results'); sys.path.insert(0, 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2')
import numpy as np, scipy, scipy.ndimage, scipy.linalg, torch, torch.nn as nn, torch.nn.functional as TF
import common as C
import iabr2_sim as S

CACHE = f'{HERE}/cache'; os.makedirs(CACHE, exist_ok=True)
_src = open(f'{C.ROOT}/notebook_src.py', encoding='utf-8').read().split('\n')
_gm = dict(np=np, torch=torch, nn=nn, TF=TF, W_IDEAL=S.W_IDEAL, F_IDEAL=S.F_IDEAL, P=16, Q=16); exec('\n'.join(_src[419:522]), _gm)
torch.set_num_threads(8)
cnn = _gm['IABRv2'](32, 10, 32); cnn.load_state_dict(torch.load(f'{C.ROOT}/outputs/checkpoints/IABRv2/final.pt', map_location='cpu')); cnn.eval()
LAM = 299792458.0 / 28e9


def to_ant(Y):
    # Hant = W_IDEAL Y F_IDEAL^H  (notebook line 274)
    return np.matmul(np.matmul(S.W_IDEAL[None], Y), S.F_IDEAL.conj().T[None])


def net(Y, bs=500):
    X = np.ascontiguousarray(S.to_ri(Y).transpose(0, 3, 1, 2)); hs, hc = [], []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            o = cnn(torch.from_numpy(X[s:s + bs])); hs.append(torch.sigmoid(o['heat']).double().numpy()); hc.append(o['Hc'].numpy().astype(complex))
    return np.concatenate(hs), np.concatenate(hc)


# ---------------------------------------------------------------- independent forward model
def steer(cosang, pos):
    """cosang (B,L), pos (B,16) element positions in half-wavelengths -> (B,L,16) = exp(-j pi cos * pos)/4"""
    return np.exp(-1j * np.pi * cosang[:, :, None] * pos[:, None, :]) / 4.0


def steer_nf(ang, r):
    """spherical wave from a point at range r (m), angle ang from the array axis; elements at (n-7.5) lambda/2 on the x axis.
    Same far-field convention as the ideal model (phase reference at the array centre, then shifted to element 0)."""
    x = (np.arange(16) - 7.5) * LAM / 2
    px, py = r * np.cos(ang)[..., None], r * np.sin(ang)[..., None]
    rn = np.hypot(px - x, py)
    return (r / rn) * np.exp(1j * 2 * np.pi / LAM * (rn - r)) * np.exp(-1j * np.pi * np.cos(ang)[..., None] * 7.5) / 4.0


def coupling(rho, t1, t2):
    out = []
    for a, b in zip(t1, t2):
        c = np.zeros(16, complex); c[0] = 1; c[1] = rho * np.exp(1j * a); c[2] = rho / 2 * np.exp(1j * b)
        out.append(scipy.linalg.toeplitz(c, c))          # complex SYMMETRIC (reciprocal) banded Toeplitz
    return np.array(out)


def qphase(Wc, bits):
    """b-bit phase shifter: phase rounded to nearest multiple of 2pi/2^b, ties towards +inf (counter-clockwise)"""
    st = 2 * np.pi / 2 ** bits
    return np.abs(Wc) * np.exp(1j * st * np.floor(np.angle(Wc) / st + 0.5 + 1e-12))


def adc_peak(Y, bits):
    A = np.max(np.maximum(np.abs(Y.real), np.abs(Y.imag)), axis=(1, 2), keepdims=True)
    D = 2 * A / 2 ** bits

    def q(x):
        idx = np.clip(np.floor((x + A) / D), 0, 2 ** bits - 1)
        return -A + D * (idx + 0.5)
    return q(Y.real) + 1j * q(Y.imag)


def forward(psi, phi, alpha, noise, kind='clean', sev=None, draws=None):
    """Y = W^H C_r H C_t F + Z (+ M6 phase noise, M5 ADC). draws: dict of standard random draws for the impairment."""
    B = len(alpha); cps, cph = np.cos(np.nan_to_num(psi)), np.cos(np.nan_to_num(phi)); n = np.tile(np.arange(16.), (B, 1))
    if kind == 'M2':
        ar, at = steer(cps, n + sev * draws['dr']), steer(cph, n + sev * draws['dt'])
    elif kind == 'M8':
        ar, at = steer_nf(np.nan_to_num(psi), sev), steer_nf(np.nan_to_num(phi), sev)
    else:
        ar, at = steer(cps, n), steer(cph, n)
    H = 16.0 * np.einsum('bl,bln,blm->bnm', alpha, ar, at.conj())
    if kind == 'M1':
        H = coupling(sev, draws['t'][0], draws['t'][1]) @ H @ coupling(sev, draws['t'][2], draws['t'][3])
    Wm, Fm = (qphase(S.W_IDEAL, sev), qphase(S.F_IDEAL, sev)) if kind == 'M4' else (S.W_IDEAL, S.F_IDEAL)
    Y = Wm.conj().T[None] @ H @ Fm[None] + noise
    if kind == 'M6':
        Y = Y * np.exp(1j * np.deg2rad(sev) * draws['pn'])
    if kind == 'M5':
        Y = adc_peak(Y, sev)
    return Y


def fresh_bank(n, snr, seed, L=3):
    rng = np.random.default_rng(seed)
    psi, phi, alpha = S.sample_paths(np.full(n, L), rng)
    g = np.random.default_rng([seed, 901]); s = np.sqrt(10 ** (-snr / 10) / 2)
    noise = s * (g.standard_normal((n, 16, 16)) + 1j * g.standard_normal((n, 16, 16)))
    return dict(psi=psi, phi=phi, alpha=alpha, noise=noise, L=np.full(n, L), snr=np.full(n, float(snr)), seed=seed, n=n)


CODES = dict(M1=911, M2=912, M4=914, M5=915, M6=916, M8=918)


def fresh_Y(b, kind, sev):
    n = b['n']
    if kind in ('clean', 'M3'):
        pd, gd = sev if kind == 'M3' else (0.0, 0.0)
        return S.simulate(b['psi'], b['phi'], b['alpha'], b['snr'], np.random.default_rng([b['seed'], 903]), phase_deg_max=pd, gain_db_max=gd,
                          noise=b['noise'])['Y']
    g = np.random.default_rng([b['seed'], CODES[kind]]); dr = {}
    if kind == 'M1':
        dr['t'] = g.uniform(0, 2 * np.pi, (4, n))
    if kind == 'M2':
        dr['dr'] = g.standard_normal((n, 16)); dr['dt'] = g.standard_normal((n, 16))
    if kind == 'M6':
        dr['pn'] = g.standard_normal((n, 16, 16))
    return forward(b['psi'], b['phi'], b['alpha'], b['noise'], kind, sev, dr)


def stats(pred, b):
    if not isinstance(pred, list):
        pred = [(pred[0][i], pred[1][i]) for i in range(len(pred[0]))]
    return C.scene_stats(pred, dict(psi=b['psi'], phi=b['phi'], L=b['L']))


def pd(Sx):
    return C.metric_from_stats(Sx, np.arange(len(Sx)))['pd_paper']
