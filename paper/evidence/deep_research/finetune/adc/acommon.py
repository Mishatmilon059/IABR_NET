"""Shared helpers for the ADC fine-tuning study (finetune/adc). Read-only use of existing code:
  decoder      = notebook_src.py slice [527:647] via new_results/common.py (nomp_os 4, nomp_rc 3, SC = (1.0, 5.0))
  network-only = notebook_src.py slice [649:668] (parabolic_heat_decode)
  CNN          = notebook_src.py slice [419:522], IABRv2(32, 10, 32)
  simulator    = stress/stress_sim.py (base_scenes, generate, adc)
  metric       = common.scene_stats / metric_from_stats / boot_metrics (paper-style Pd, |err| <= 1 deg)
Seeds: stress M5 banks 5000 (0 dB) / 5015 (15 dB) (same as stress/run_stress.py); fresh replicate banks 8300 / 8315 (new, unused elsewhere);
tuned self-calibration (lam 1, tau 2, iters 4) taken frozen from stress/adversarial/replicate.log (tuned on seeds 9900/9915)."""
import os, sys, json, time
os.environ.setdefault('IABR2_SUITE', 'D:/ai_ml_project')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results'); sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/stress')
sys.path.insert(0, 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2')
import numpy as np, scipy, scipy.ndimage, torch, torch.nn as nn, torch.nn.functional as TF
import common as C
import stress_sim as SS

CACHE = f'{HERE}/cache'; os.makedirs(CACHE, exist_ok=True)
ROOT = C.ROOT
ORIG_CKPT = f'{ROOT}/outputs/checkpoints/IABRv2/final.pt'
FT_CKPT = f'{CACHE}/adc_ft.pt'
_src = open(f'{ROOT}/notebook_src.py', encoding='utf-8').read().split('\n')
_g = dict(np=np, scipy=scipy, uv_to_angles=C.uv_to_angles); exec('\n'.join(_src[649:668]), _g)
parabolic_heat_decode = _g['parabolic_heat_decode']
_gm = dict(np=np, torch=torch, nn=nn, TF=TF, W_IDEAL=SS.W_IDEAL, F_IDEAL=SS.F_IDEAL, P=16, Q=16); exec('\n'.join(_src[419:522]), _gm)
IABRv2 = _gm['IABRv2']; n_params = _gm['n_params']
# training-loss functions copied verbatim from notebook_src.py lines 686-691 (focal_loss)
_gl = dict(torch=torch); exec('\n'.join(_src[685:691]), _gl)
focal_loss = _gl['focal_loss']

SC = C.SC                       # default self-calibration (lam 1, tau 5, iters 2)
SC_TUNED = (1.0, 2.0, 4)        # lam, tau, iters  (stress/adversarial/replicate.log line 17)
STRESS_SEEDS = {0: 5000, 15: 5015}
FRESH_SEEDS = {0: 8300, 15: 8315}
BITS = [4, 3, 2, 1]


def load_cnn(path):
    m = IABRv2(32, 10, 32)
    sd = torch.load(path, map_location='cpu')
    m.load_state_dict(sd['model'] if 'model' in sd else sd); m.eval()
    return m


def to_ant(Y):
    return SS.to_ant(Y)


def cnn_forward(cnn, Y, bs=250):
    heat, Hc = [], []
    X = np.ascontiguousarray(SS.S.to_ri(Y).transpose(0, 3, 1, 2))
    with torch.no_grad():
        for s in range(0, len(X), bs):
            o = cnn(torch.from_numpy(X[s:s + bs]))
            heat.append(torch.sigmoid(o['heat']).numpy().astype(np.float64)); Hc.append(o['Hc'].numpy().astype(np.complex128))
    return np.concatenate(heat), np.concatenate(Hc)


def m5_bank(n, snr, seed, bits):
    """stress M5 bank: base scenes from seed, Y = ADC_bits(peak AGC)(W^H H F + Z). bits=None -> clean."""
    b = SS.base_scenes(n, 3, float(snr), seed)
    Y = SS.generate(b, 'clean') if bits is None else SS.generate(b, 'M5', bits)
    return b, Y


# ---------------------------------------------------------------- classical alternatives
def fft_heat(H, os_=4):
    """|FFT2(Hant)| on a (16 os)^2 grid, normalised to max 1 per scene. Same index->(u,v) map as nomp_greedy / heat_peaks:
    u = -2 pi q / G, v = 2 pi p / G."""
    G = 16 * os_
    X = np.abs(np.fft.fft2(H, s=(G, G)))
    return X / X.max((1, 2), keepdims=True)


def selfcal_tuned(H, U, V):
    lam, tau, it = SC_TUNED
    return C.selfcal(H, U, V, lam, tau, rc=2, iters=it)


def cyc(H, U, V, rc=3):
    return C.cyclic(H, U, V, rc)


def seeds_refine(H, U0, V0, rc=3):
    """the decode(...,'heat') refinement of given seeds: single-path Newton on the full channel, then cyclic."""
    B, L = U0.shape
    U, V = C.newton(H[:, None].repeat(L, 1).reshape(-1, 16, 16), U0.ravel(), V0.ravel())
    return C.cyclic(H, U.reshape(B, L), V.reshape(B, L), rc)


def pick_lowest(H, cands):
    """cands: list of (U, V); per scene keep the candidate with the lowest LS residual energy on H."""
    R = np.stack([C.residual_energy(H, U, V) for U, V in cands], 1)
    k = R.argmin(1); idx = np.arange(len(H))
    U = np.stack([c[0] for c in cands], 1)[idx, k]; V = np.stack([c[1] for c in cands], 1)[idx, k]
    return U, V, k


def adc_cells(Yq, bits):
    """quantisation cell [lo, hi] of every Re / Im sample, known to the receiver from its own ADC (b bits, peak AGC: no clipping).
    Output levels are D(k + 1/2); D is recovered per scene from the levels: A = max |level| * 2^b / (2^b - 1)... computed exactly as
    A = (max|level|) / (1 - 1/2^b) since the top level is A - D/2 = A (1 - 2^-b)."""
    top = np.maximum(np.abs(Yq.real), np.abs(Yq.imag)).max((1, 2))[:, None, None]
    A = top / (1 - 2.0 ** -bits); D = 2 * A / 2 ** bits

    def cell(x):
        k = np.round(x / D - 0.5)
        return D * k, D * (k + 1)
    rl, rh = cell(Yq.real); il, ih = cell(Yq.imag)
    return rl, rh, il, ih


def model_Y(H, U, V):
    """LS L-path model of the antenna-domain channel H, mapped back to beamspace (Y = W^H Hm F, W/F unitary DFT)."""
    x, A = C.ls_gains(H, U, V)
    Hm = np.einsum('bik,bk->bi', A, x).reshape(H.shape)
    return np.einsum('nq,bnm,mp->bqp', np.conj(SS.W_IDEAL), Hm, SS.F_IDEAL)


def consistent_nomp(Yq, bits, L=3, iters=8, U=None, V=None):
    """Quantisation-consistent re-estimation (POCS-style 'dequantisation', classical, no learning):
    start from NOMP on the quantised data; repeat: (1) fit the L-path LS model, (2) replace each Re/Im sample by the model value
    clipped into its known quantisation cell (the closest quantisation-consistent measurement), (3) re-run NOMP on the de-quantised
    channel and keep, per scene, whichever of {current, new NOMP} has the lower inconsistency. Selection uses only the observed
    cells (inconsistency = squared distance of the model's beamspace values from their cells, after an LS scale fit).
    Returns U, V and the final inconsistency."""
    rl, rh, il, ih = adc_cells(Yq, bits)
    H0 = to_ant(Yq)
    if U is None:
        U, V = C.nomp_greedy(H0, L, 4, 3)

    def incons(U, V, Yd):
        Ym = model_Y(to_ant(Yd), U, V)
        # optimal real positive scale of the model against the cell centres (scale-free AGC), then distance to the cells
        c = (Yq * np.conj(Ym)).real.sum((1, 2)) / ((np.abs(Ym) ** 2).sum((1, 2)) + 1e-12)
        Ym = Ym * np.maximum(c, 1e-12)[:, None, None]
        er = np.maximum(rl - Ym.real, 0) + np.maximum(Ym.real - rh, 0); ei = np.maximum(il - Ym.imag, 0) + np.maximum(Ym.imag - ih, 0)
        return (er ** 2 + ei ** 2).sum((1, 2)) / (np.abs(Yq) ** 2).sum((1, 2)), Ym

    Yd = Yq.copy()
    best, Ym = incons(U, V, Yd)
    for _ in range(iters):
        Yd = np.clip(Ym.real, rl, rh) + 1j * np.clip(Ym.imag, il, ih)
        Hd = to_ant(Yd)
        Uc, Vc = C.cyclic(Hd, U, V, 2)                     # refine current paths on the de-quantised channel
        Un, Vn = C.nomp_greedy(Hd, L, 4, 3)               # fresh greedy search on the de-quantised channel
        for Ux, Vx in [(Uc, Vc), (Un, Vn)]:
            e, Ymx = incons(Ux, Vx, Yd)
            better = e < best
            U = np.where(better[:, None], Ux, U); V = np.where(better[:, None], Vx, V); best = np.where(better, e, best)
        _, Ym = incons(U, V, Yd)
    return U, V, best


def multistart(H, L=3, K=6, rc=3):
    """multi-start search: top-K local maxima of |FFT2(H)| (4x) -> every L-subset as seeds -> Newton + cyclic -> lowest residual,
    plus greedy NOMP itself as one more candidate."""
    import itertools
    heat = fft_heat(H, 4)
    G = heat.shape[-1]
    mx = scipy.ndimage.maximum_filter(heat, size=(1, 3, 3), mode='wrap')
    score = np.where(heat >= mx, heat + 1.0, heat).reshape(len(heat), -1)
    idx = np.argsort(-score, 1)[:, :K]
    Uk = -2 * np.pi * (idx // G) / G; Vk = 2 * np.pi * (idx % G) / G
    Uk, Vk = C.newton(H[:, None].repeat(K, 1).reshape(-1, 16, 16), Uk.ravel(), Vk.ravel()); Uk = Uk.reshape(-1, K); Vk = Vk.reshape(-1, K)
    cands = [C.nomp_greedy(H, L, 4, rc)]
    for sub in itertools.combinations(range(K), L):
        cands.append(C.cyclic(H, Uk[:, list(sub)], Vk[:, list(sub)], rc))
    U, V, k = pick_lowest(H, cands)
    return U, V, k


# ---------------------------------------------------------------- metrics
def as_list(p):
    if isinstance(p, list):
        return p
    return [(p[0][i], p[1][i]) for i in range(len(p[0]))]


def stats(pred, b):
    return C.scene_stats(as_list(pred), dict(psi=b['psi'], phi=b['phi'], L=b['L']))


def pd(Sx, idx=None):
    return C.metric_from_stats(Sx, np.arange(len(Sx)) if idx is None else idx)['pd_paper']


def pack(pred, Lm=3):
    pl = as_list(pred); n = np.array([len(p[0]) for p in pl]); a = np.full((len(pl), Lm), np.nan); c = a.copy()
    for i, p in enumerate(pl):
        a[i, :n[i]] = p[0][:Lm]; c[i, :n[i]] = p[1][:Lm]
    return dict(psi=a, phi=c, n=n)


def boot_diff(Sa, Sb, snr, B=2000, seed=0):
    """paired bootstrap (within SNR groups, seed 0) of mean-over-SNR paper Pd difference a - b: (point, lo, hi)"""
    Bt = C.boot_metrics({'a': Sa, 'b': Sb}, snr, B=B, seed=seed)
    d = Bt['a'][:, 0] - Bt['b'][:, 0]
    pa = C.metric_by_snr(Sa, snr)['mean']['pd_paper']; pb = C.metric_by_snr(Sb, snr)['mean']['pd_paper']
    return [pa - pb, float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]


def save_json(o, path):
    json.dump(o, open(path, 'w'), indent=1, default=float)
