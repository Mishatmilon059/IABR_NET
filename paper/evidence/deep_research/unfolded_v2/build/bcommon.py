"""Shared helpers for unfolded_v2/build (implementation, training and evaluation of design A 'CA-UNOMP' and design B 'HU-NOMP',
both UNFOLDED FROM NOMP (Mamandipoor, Ramasamy, Madhow, IEEE TSP 2016); quantised-data parts unfolded from quantisation-aware NOMP/ML
(GNOMP-style EM de-quantisation + exact interval-censored likelihood, as in finetune/adversarial/mlq.py).
Read-only use of existing code: notebook_src.py decoder slice via new_results/common.py (module C), stress_sim (SS), finetune/adc
acommon (A: CNN class, adc_cells, consistent NOMP), finetune/adversarial mlq/advc (classical ML baselines).
Threads: THREADS env (default 6) set before numpy/torch import. Large arrays -> BIG (C:)."""
import os, sys, json, time, math, itertools
TH = os.environ.get('THREADS', '6')
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_k] = TH
os.environ.setdefault('IABR2_SUITE', 'D:/ai_ml_project')
HERE = os.path.dirname(os.path.abspath(__file__))
DR = 'D:/ai_ml_project/paper/evidence/deep_research'
sys.path.insert(0, f'{DR}/finetune/adc'); sys.path.insert(0, f'{DR}/finetune/adversarial')
import acommon as A                       # noqa: E402  (also puts new_results, stress, IABR_v2 on sys.path)
np, torch, C, SS = A.np, A.torch, A.C, A.SS
nn, TF = A.nn, A.TF
import mlq                                # noqa: E402
import iabr2_sim as SIM                   # noqa: E402
torch.set_num_threads(int(TH))
BIG = 'C:/Users/Milon/iabr_deep_research_cache/dr_cache/unfolded_v2/build'
os.makedirs(BIG, exist_ok=True)
MO = mlq.Model(SS.W_IDEAL, SS.F_IDEAL, C)
BITS_LIST = [1, 2, 3, 4, 0]               # 0 = no ADC (b = inf); one-hot index order
IMP_TYPES = ['none', 'coupling', 'position', 'perant', 'phasenoise']


def sigma_of(snr):
    """per-real-component noise std of the benchmark (s = sqrt(10^(-SNR/10) / 2))"""
    return np.sqrt(10 ** (-np.asarray(snr, float) / 10) / 2)


def to_beam(H):
    return np.einsum('nq,bnm,mp->bqp', np.conj(SS.W_IDEAL), H, SS.F_IDEAL)


def cells(Yq, bits):
    """(B,512) lo, hi for quantised scenes (acommon.adc_cells, peak AGC); bits array (B,), 0 = unquantised -> lo = hi = y."""
    B = len(Yq); lo = np.concatenate([Yq.real.reshape(B, -1), Yq.imag.reshape(B, -1)], 1); hi = lo.copy()
    for b in [1, 2, 3, 4]:
        s = bits == b
        if s.any():
            l, h = mlq.cells_ri(Yq[s], b, A.adc_cells); lo[s], hi[s] = l, h
    return lo, hi


# ---------------------------------------------------------------- scene generation (training / validation / fresh test banks)
def gen_scenes(rng, n, L, snr, sep=np.pi / 6, seed_int=0):
    """like stress_sim.base_scenes but with per-scene SNR array and an explicit separation (same draw order otherwise).
    L int, snr (n,) array. seed_int: integer used by stress_sim.imp_rng for impairment draws."""
    L_arr = np.full(n, L)
    psi, phi, alpha = SIM.sample_paths(L_arr, rng, sep=sep)
    Ur = rng.uniform(-1, 1, (n, 16)); Ut = rng.uniform(-1, 1, (n, 16)); Gr = rng.uniform(-1, 1, (n, 16)); Gt = rng.uniform(-1, 1, (n, 16))
    s = sigma_of(snr)[:, None, None]
    noise = s * (rng.standard_normal((n, 16, 16)) + 1j * rng.standard_normal((n, 16, 16)))
    return dict(psi=psi, phi=phi, alpha=alpha, L=L_arr.astype(int), snr=np.asarray(snr, float), Ur=Ur, Ut=Ut, Gr=Gr, Gt=Gt,
                noise=noise, seed=int(seed_int))


def sub(b, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(b['snr']) else v) for k, v in b.items()}


def imp_target(b, kind, sev):
    """IABC target (identifiable per-antenna phase / log-gain) for M3 scenes, zeros for ideal scenes; mask False otherwise."""
    n = len(b['snr'])
    if kind == 'M3':
        pdm, gdm = sev
        pdm = np.broadcast_to(np.asarray(pdm, float), (n,)); gdm = np.broadcast_to(np.asarray(gdm, float), (n,))
        D_r = 10 ** (b['Gr'] * gdm[:, None] / 20) * np.exp(1j * b['Ur'] * np.deg2rad(pdm)[:, None])
        D_t = 10 ** (b['Gt'] * gdm[:, None] / 20) * np.exp(1j * b['Ut'] * np.deg2rad(pdm)[:, None])
        return SIM.impairment_targets(D_r, D_t), np.ones(n, bool)
    return np.zeros((n, 32, 2), np.float32), np.full(n, kind in ('clean', 'M5'))


def make_mix(rng, n, seed_int, comp=None, L_fixed=None):
    """One training/validation batch, all scenes with the same L (batching constraint of the unrolled core).
    Components (design section 2): 'ideal' 30%, 'adc' 35%, 'imp' 20%, 'geom' 15% (L 1..6, sep 0..30 deg, b in {inf, 2}).
    Returns dict(Y, bits, snr, psi, phi, L, imp, imp_mask, imptype, comp)."""
    if comp is None:
        comp = rng.choice(['ideal', 'adc', 'imp', 'geom'], p=[0.30, 0.35, 0.20, 0.15])
    snr = rng.uniform(-12, 27, n)
    L = 3
    sep = np.pi / 6
    if comp == 'geom':
        L = int(L_fixed or rng.integers(1, 7)); sep = np.deg2rad(rng.uniform(0, 30))
    b = gen_scenes(rng, n, L, snr, sep=sep, seed_int=seed_int)
    bits = np.zeros(n, int); imptype = np.zeros(n, int)
    imp = np.zeros((n, 32, 2), np.float32); imask = np.zeros(n, bool)
    if comp == 'ideal':
        Y = SS.generate(b, 'clean'); imask[:] = True
    elif comp == 'adc':
        Y = SS.generate(b, 'clean'); imask[:] = True
        bits = rng.integers(1, 5, n); clip = rng.uniform(size=n) < 0.10
        for bb in [1, 2, 3, 4]:
            for c, agc in [(False, 'peak'), (True, 'rms3')]:
                s = (bits == bb) & (clip == c)
                if s.any():
                    Y[s] = SS.adc(Y[s], int(bb), agc)
        imask[clip] = False
    elif comp == 'imp':
        kind = rng.choice(['M1', 'M2', 'M3', 'M6'])
        if kind == 'M1':
            sev = rng.uniform(0, 0.3, n); Y = SS.generate(b, 'M1', sev); imptype[:] = 1
        elif kind == 'M2':
            sev = rng.uniform(0, 0.2, (n, 1)); Y = SS.generate(b, 'M2', sev); imptype[:] = 2
        elif kind == 'M3':
            sev = (rng.uniform(0, 45, n) * (rng.uniform(size=n) < 0.7), rng.uniform(0, 6, n) * (rng.uniform(size=n) < 0.7))
            Y = SS.generate(b, 'M3', sev); imptype[:] = 3
            imp, imask = imp_target(b, 'M3', sev)
        else:
            sev = rng.uniform(0, 30, (n, 1, 1)); Y = SS.generate(b, 'M6', sev); imptype[:] = 4
        q = rng.uniform(size=n) < 0.2
        bq = rng.integers(2, 5, n)
        for bb in [2, 3, 4]:
            s = q & (bq == bb)
            if s.any():
                Y[s] = SS.adc(Y[s], int(bb), 'peak'); bits[s] = bb
    else:
        Y = SS.generate(b, 'clean'); imask[:] = True
        q = rng.uniform(size=n) < 0.5
        if q.any():
            Y[q] = SS.adc(Y[q], 2, 'peak'); bits[q] = 2
    return dict(Y=Y, bits=bits, snr=snr, psi=b['psi'], phi=b['phi'], L=L, imp=imp, imp_mask=imask, imptype=imptype, comp=str(comp))


# ---------------------------------------------------------------- metrics (paper Pd) and bootstrap
def preds_from_uv(U, V, n=None):
    """(B,K) u/v (nan = no path) -> list of (psi, phi) arrays (first n[i] entries)"""
    ps, ph = C.uv_to_angles(np.nan_to_num(U), np.nan_to_num(V))
    if n is None:
        n = np.full(len(U), U.shape[1])
    return [(ps[i, :n[i]], ph[i, :n[i]]) for i in range(len(U))]


def stats(preds, bank):
    return C.scene_stats(A.as_list(preds), dict(psi=bank['psi'], phi=bank['phi'], L=bank['L']))


def pd(S, snr=None):
    if snr is None:
        return C.metric_from_stats(S, np.arange(len(S)))['pd_paper']
    return C.metric_by_snr(S, snr)['mean']['pd_paper']


def boot_pairs(S, snr, pairs, B=2000):
    Bt = C.boot_metrics(S, snr, B=B, seed=0)
    pdm = {m: C.metric_by_snr(S[m], snr)['mean']['pd_paper'] for m in S}
    out = {}
    for a_, b_ in pairs:
        d = Bt[a_][:, 0] - Bt[b_][:, 0]
        out[f'{a_} - {b_}'] = [pdm[a_] - pdm[b_], float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
    return pdm, {m: list(C.ci(Bt[m][:, 0])[1:]) for m in S}, out


def fmt(d):
    return f'{d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]' + ('*' if d[1] > 0 or d[2] < 0 else '')


def save_json(o, path):
    json.dump(o, open(path, 'w'), indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else float(x))
