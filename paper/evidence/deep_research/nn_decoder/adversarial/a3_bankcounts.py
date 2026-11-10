"""Independent recount of the 'bank counts' table: for every test bank with truth (s0_inventory.json, excluding tune_*),
recompute all-neural SC on (StagedSC + lurd.pt from the net__ cache) and the classical heat decoder (decode(Hc, L, heat, 'heat', SC=(1,5)))
here; NOMP-only and U-Net from the cached pred__ files. Own metric (Hungarian as permute_pairs, |wrapped err| <= 1 deg) and own paired
bootstrap (2000 resamples within SNR groups, seed 0). Counts: 95% CI of the paired difference above 0 / below 0 / containing 0.
usage: python -u a3_bankcounts.py > a3_bankcounts.log"""
import os, sys, json, time
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '6'
LD = 'D:/ai_ml_project/paper/evidence/learned_decoder'
sys.path.insert(0, LD)
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
import exp5 as E5
torch.set_num_threads(6)
HERE = os.path.dirname(os.path.abspath(__file__))
ND = os.path.dirname(HERE)
ROOT = 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2'
CACHE = f'{ROOT}/outputs/cache'
GEN = 'D:/ai_ml_project/IABR_Net_TestSuite/data/generated_banks'
ADC = 'C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/nn_decoder_adv'
g = E5.g
model = E5.StagedSC(E5.RefinerV3(), pre=6, rounds=3, inner=2)
model.load_state_dict(torch.load(f'{LD}/lurd.pt', map_location='cpu')); model.eval(); model.use_sc = True
inv = json.load(open(f'{ND}/s0_inventory.json'))
banks = [n for n, v in inv.items() if v is not None and not n.startswith('tune_')]


def truth(name):
    if name == 'official':
        b = np.load('D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz')
        return dict(psi=b['feat'][:, 0, :].astype(np.float64), phi=b['feat'][:, 1, :].astype(np.float64), L=b['meta'][:, 0].astype(int), snr=b['meta'][:, 1].astype(int))
    p = f'{ROOT}/data/banks/{name}.npz'
    z = np.load(p if os.path.exists(p) else f'{GEN}/{name}.npz')
    return dict(psi=z['psi'].astype(np.float64), phi=z['phi'].astype(np.float64), L=z['L'].astype(int), snr=z['snr'].astype(int))


def by_L(Ls, fn):
    psi = np.full((len(Ls), Ls.max()), np.nan); phi = psi.copy()
    for L in sorted(set(Ls.tolist())):
        ii = np.where(Ls == L)[0]; a, b = fn(ii, L); psi[ii, :L] = a; phi[ii, :L] = b
    return dict(psi=psi, phi=phi, n=Ls.copy())


def an(Hc, heat, Ls):
    def fn(ii, L):
        oa, ob = [], []
        for s in range(0, len(ii), 250):
            jj = ii[s:s + 250]; U0, V0 = g['heat_peaks'](heat[jj].astype(np.float64), L)
            with torch.no_grad():
                U, V = model(torch.from_numpy(np.ascontiguousarray(Hc[jj]).astype(np.complex64)), torch.tensor(U0, dtype=torch.float32), torch.tensor(V0, dtype=torch.float32))
            a, b = g['uv_to_angles'](U.numpy().astype(np.float64), V.numpy().astype(np.float64)); oa.append(a); ob.append(b)
        return np.concatenate(oa), np.concatenate(ob)
    return by_L(Ls, fn)


def heatdec(Hc, heat, Ls):
    def fn(ii, L):
        oa, ob = [], []
        for s in range(0, len(ii), 400):
            jj = ii[s:s + 400]
            a, b = g['decode'](Hc[jj].astype(np.complex128), L, heat[jj].astype(np.float64), 'heat', (1.0, 5.0)); oa.append(a); ob.append(b)
        return np.concatenate(oa), np.concatenate(ob)
    return by_L(Ls, fn)


def get(name, tag, fn):
    p = f'{ADC}/{name}__{tag}.npz'
    if os.path.exists(p):
        z = np.load(p); return dict(psi=z['psi'], phi=z['phi'], n=z['n'])
    r = fn(); np.savez_compressed(p, **r); return r


def stats(pred, T):
    N = len(T['L']); S = np.zeros((N, 4))
    for i in range(N):
        L = T['L'][i]; ps, ph = pred['psi'][i, :pred['n'][i]], pred['phi'][i, :pred['n'][i]]
        if len(ps) < L or np.isnan(ps[:L]).any():
            S[i] = [0, 0, 1, L]; continue
        G = np.stack([T['psi'][i, :L], T['phi'][i, :L]], 1); P = np.stack([ps[:L], ph[:L]], 1)
        r, c = linear_sum_assignment(np.linalg.norm(G[:, None] - P[None], axis=2))
        ok = np.abs(np.degrees(np.angle(np.exp(1j * (G[r] - P[c]))))) <= 1.0; S[i] = [ok.sum(), (~ok).sum(), 0, L]
    return S


def boot(Sd, snr, B=2000, seed=0):
    rng = np.random.default_rng(seed); grp = [np.where(snr == s)[0] for s in sorted(set(snr.tolist()))]
    acc = {m: np.zeros((B, 2)) for m in Sd}; pt = {m: np.zeros(2) for m in Sd}
    for ix in grp:
        R = rng.integers(0, len(ix), (B, len(ix)))
        for m, S in Sd.items():
            gd = S[ix, 0][R].sum(1); bd = S[ix, 1][R].sum(1); dr = (2 * S[ix, 2] * S[ix, 3])[R].sum(1)
            acc[m] += np.stack([gd / (gd + bd), gd / (gd + bd + dr)], 1)
            s = S[ix].sum(0); pt[m] += [s[0] / (s[0] + s[1]), s[0] / (s[0] + s[1] + (2 * S[ix, 2] * S[ix, 3]).sum())]
    return {m: a / len(grp) for m, a in acc.items()}, {m: p / len(grp) for m, p in pt.items()}


counts = {}; rows = {}
t0 = time.time()
for n in banks:
    T = truth(n); z = np.load(f'{CACHE}/net__{n}.npz'); heat, Hc = z['heat'], z['Hc']
    P = dict(AN=get(n, 'an_sc', lambda: an(Hc, heat, T['L'])), HEAT=get(n, 'heat', lambda: heatdec(Hc, heat, T['L'])))
    for k, f in [('NOMP', 'NOMP_no_network'), ('UNET', 'U-Net_paper')]:
        q = np.load(f'{CACHE}/pred__{f}__{n}.npz'); P[k] = dict(psi=q['psi'], phi=q['phi'], n=q['n'])
    S = {k: stats(v, T) for k, v in P.items()}
    Bt, pt = boot(S, T['snr'])
    grp = 'L=3' if set(T['L'].tolist()) == {3} else 'L!=3'
    rows[n] = dict(group=grp, pd={k: float(v[0]) for k, v in pt.items()})
    line = f'{n:20s} {grp:5s} AN {pt["AN"][0]:.4f}'
    for ref in ['UNET', 'NOMP', 'HEAT']:
        for j, met in enumerate(['paper', 'strict']):
            d = Bt['AN'][:, j] - Bt[ref][:, j]; lo, hi = np.percentile(d, [2.5, 97.5]); m = pt['AN'][j] - pt[ref][j]
            cls = 'above' if lo > 0 else 'below' if hi < 0 else 'contains0'
            counts.setdefault(grp, {}).setdefault(f'AN-{ref} {met}', dict(above=[], below=[], contains0=[]))[cls].append(n)
            rows[n][f'AN-{ref} {met}'] = [float(m), float(lo), float(hi)]
            if met == 'paper':
                line += f' | d{ref} {m:+.4f} [{lo:+.4f},{hi:+.4f}]'
    print(line, flush=True)
    del z, heat, Hc
print(f'\nrecount ({time.time() - t0:.0f}s): 95% CI above 0 / below 0 / contains 0', flush=True)
for grp, dd in counts.items():
    for k, c in dd.items():
        print(f'  {grp:5s} {k:16s} {len(c["above"]):2d} / {len(c["below"]):2d} / {len(c["contains0"]):2d}   above={c["above"]} below={c["below"] if len(c["below"]) < 6 else "..."}', flush=True)
json.dump(dict(rows=rows, counts=counts), open(f'{HERE}/a3_bankcounts.json', 'w'), indent=1)
print('done', flush=True)
