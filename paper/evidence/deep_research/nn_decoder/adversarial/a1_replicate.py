"""Adversarial replication of the nn_decoder claims (independent code path; reads final.pt / lurd.pt only).
Banks: official (all 8000), T2_d5, gain_g4_snr15, sep_3deg_snr15.
Per bank: (a) cache alignment: re-run the CNN from Y16 on every 16th scene and compare with net__<bank>.npz;
(b) all-neural SC on/off recomputed here (own loop: notebook heat_peaks -> StagedSC(lurd.pt), eval mode);
(c) NOMP-only recomputed here (notebook decode(Ht, L, 'greedy', sc=None)) and compared with the cached pred__NOMP_no_network;
(d) classical heat decoder recomputed here (decode(Hc, L, heat, 'heat', SC=(1,5)));
(e) U-Net: cached pred__U-Net_paper (not re-run; alignment shown by its Pd);
(f) own metric (Hungarian on unwrapped Euclidean distance as permute_pairs, |wrapped err| <= 1 deg) cross-checked with common.scene_stats;
    own paired bootstrap (2000 resamples within SNR groups, seed 0). No new random seeds (deterministic decoders; bootstrap seed 0).
Official only: the 2000 scenes (first 250 per SNR) used by exp/exp3/exp5 for design selection vs the other 6000 scenes; per-SNR diffs.
usage: python -u a1_replicate.py > a1_replicate.log"""
import os, sys, json, time
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '6'
LD = 'D:/ai_ml_project/paper/evidence/learned_decoder'
sys.path.insert(0, LD)
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
import exp5 as E5                      # loads the CNN (final.pt, read only)
torch.set_num_threads(6)
import iabr2_sim as sim
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2'
CACHE = f'{ROOT}/outputs/cache'
GEN = 'D:/ai_ml_project/IABR_Net_TestSuite/data/generated_banks'
NEWR = 'D:/ai_ml_project/paper/evidence/new_results'
SCR = 'C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache'
DRC_OLD = f'{SCR}/nn_decoder'
ADC = f'{SCR}/nn_decoder_adv'
os.makedirs(ADC, exist_ok=True)
g = E5.g                                # notebook decoder namespace (notebook_src lines 528-647)
SC = (1.0, 5.0)


def truth(name):
    if name == 'official':
        b = np.load('D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz')
        return dict(psi=b['feat'][:, 0, :].astype(np.float64), phi=b['feat'][:, 1, :].astype(np.float64), L=b['meta'][:, 0].astype(int),
                    snr=b['meta'][:, 1].astype(int), Y16=sim.downsample16(b['data']))
    p = f'{ROOT}/data/banks/{name}.npz'
    z = np.load(p if os.path.exists(p) else f'{GEN}/{name}.npz')
    return dict(psi=z['psi'].astype(np.float64), phi=z['phi'].astype(np.float64), L=z['L'].astype(int), snr=z['snr'].astype(int), Y16=z['Y16'])


model = E5.StagedSC(E5.RefinerV3(), pre=6, rounds=3, inner=2).to(E5.DEV)
model.load_state_dict(torch.load(f'{LD}/lurd.pt', map_location=E5.DEV))
model.eval()
print('lurd params', sum(p.numel() for p in model.parameters()), 'lam', round(float(model.lam), 4), 'training flag', model.training, flush=True)


def by_L(Ls, fn):
    psi = np.full((len(Ls), Ls.max()), np.nan); phi = psi.copy()
    for L in sorted(set(Ls.tolist())):
        ii = np.where(Ls == L)[0]; a, b = fn(ii, L); psi[ii, :L] = a; phi[ii, :L] = b
    return dict(psi=psi, phi=phi, n=Ls.copy())


def allneural(Hc, heat, Ls, use_sc):
    model.use_sc = use_sc

    def fn(ii, L):
        oa, ob = [], []
        for s in range(0, len(ii), 250):
            jj = ii[s:s + 250]; U0, V0 = g['heat_peaks'](heat[jj].astype(np.float64), L)
            with torch.no_grad():
                U, V = model(torch.from_numpy(np.ascontiguousarray(Hc[jj]).astype(np.complex64)), torch.tensor(U0, dtype=torch.float32), torch.tensor(V0, dtype=torch.float32))
            a, b = g['uv_to_angles'](U.numpy().astype(np.float64), V.numpy().astype(np.float64)); oa.append(a); ob.append(b)
        return np.concatenate(oa), np.concatenate(ob)
    return by_L(Ls, fn)


def classical(Hsrc, Ls, mode, heat=None, sc=None):
    def fn(ii, L):
        oa, ob = [], []
        for s in range(0, len(ii), 400):
            jj = ii[s:s + 400]
            a, b = g['decode'](Hsrc[jj].astype(np.complex128), L, None if heat is None else heat[jj].astype(np.float64), mode, sc); oa.append(a); ob.append(b)
        return np.concatenate(oa), np.concatenate(ob)
    return by_L(Ls, fn)


def get(name, tag, fn):
    p = f'{ADC}/{name}__{tag}.npz'
    if os.path.exists(p):
        z = np.load(p); return dict(psi=z['psi'], phi=z['phi'], n=z['n'])
    t0 = time.time(); r = fn(); np.savez_compressed(p, **r); print(f'   computed {tag:8s} {time.time() - t0:.1f}s', flush=True); return r


def load_pred(p):
    z = np.load(p); return dict(psi=z['psi'], phi=z['phi'], n=z['n'])


def own_stats(pred, T):
    """per scene: good, bad, dropped (fewer than L finite predictions), L"""
    N = len(T['L']); S = np.zeros((N, 4))
    for i in range(N):
        L = T['L'][i]; ps, ph = pred['psi'][i, :pred['n'][i]], pred['phi'][i, :pred['n'][i]]
        if len(ps) < L or np.isnan(ps[:L]).any():
            S[i] = [0, 0, 1, L]; continue
        G = np.stack([T['psi'][i, :L], T['phi'][i, :L]], 1); P = np.stack([ps[:L], ph[:L]], 1)
        r, c = linear_sum_assignment(np.linalg.norm(G[:, None] - P[None], axis=2))
        err = np.degrees(np.angle(np.exp(1j * (G[r] - P[c]))))
        ok = np.abs(err) <= 1.0; S[i] = [ok.sum(), (~ok).sum(), 0, L]
    return S


def pd_mean(S, snr, idx=None):
    idx = np.arange(len(snr)) if idx is None else idx
    v = []
    for s in sorted(set(snr[idx].tolist())):
        j = idx[snr[idx] == s]; v.append(S[j, 0].sum() / (S[j, 0].sum() + S[j, 1].sum()))
    return float(np.mean(v))


def pd_strict(S, snr):
    v = []
    for s in sorted(set(snr.tolist())):
        j = snr == s; v.append(S[j, 0].sum() / (S[j, 0].sum() + S[j, 1].sum() + 2 * (S[j, 2] * S[j, 3]).sum()))
    return float(np.mean(v))


def boot(Sd, snr, idx=None, B=2000, seed=0):
    idx = np.arange(len(snr)) if idx is None else idx
    rng = np.random.default_rng(seed); grp = [idx[snr[idx] == s] for s in sorted(set(snr[idx].tolist()))]
    acc = {m: np.zeros(B) for m in Sd}
    for ix in grp:
        R = rng.integers(0, len(ix), (B, len(ix)))
        for m, S in Sd.items():
            gd = S[ix, 0][R].sum(1); bd = S[ix, 1][R].sum(1); acc[m] += gd / (gd + bd)
    return {m: a / len(grp) for m, a in acc.items()}


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def maxdiff_deg(p, q):
    """order-free max |difference| (deg) per scene"""
    out = np.zeros(len(p['n']))
    for i in range(len(out)):
        L = p['n'][i]; P = np.stack([p['psi'][i, :L], p['phi'][i, :L]], 1); Q = np.stack([q['psi'][i, :L], q['phi'][i, :L]], 1)
        if np.isnan(Q).any() or np.isnan(P).any():
            out[i] = np.inf; continue
        r, c = linear_sum_assignment(np.abs(P[:, None] - Q[None]).sum(2)); out[i] = np.degrees(np.abs(P[r] - Q[c]).max())
    return out


sys.path.insert(0, NEWR)
import common as C
RES = {}
for name in ['official', 'T2_d5', 'gain_g4_snr15', 'sep_3deg_snr15']:
    print(f'\n===== {name} =====', flush=True)
    T = truth(name); Ls = T['L']; snr = T['snr']
    z = np.load(f'{CACHE}/net__{name}.npz'); heat, Hc, Ht = z['heat'], z['Hc'], z['Ht']
    assert len(heat) == len(Ls)
    # (a) cache alignment: re-run the CNN on every 16th scene
    sub = np.arange(0, len(Ls), 16)
    with torch.no_grad():
        y = torch.from_numpy(np.ascontiguousarray(T['Y16'][sub].transpose(0, 3, 1, 2)).astype(np.float32))
        o = [E5.cnn(y[s:s + 250]) for s in range(0, len(y), 250)]
    h2 = torch.sigmoid(torch.cat([q['heat'] for q in o])).numpy(); c2 = torch.cat([q['Hc'] for q in o]).numpy(); t2 = torch.cat([q['Ht'] for q in o]).numpy()
    al = dict(n=len(sub), max_abs_heat=float(np.abs(h2 - heat[sub]).max()), max_rel_Hc=float(np.abs(c2 - Hc[sub]).max() / np.abs(Hc[sub]).max()),
              max_rel_Ht=float(np.abs(t2 - Ht[sub]).max() / np.abs(Ht[sub]).max()),
              control_shift1_max_abs_heat=float(np.abs(h2[1:] - heat[sub[:-1]]).max()))
    print('cache alignment (CNN re-run from Y16 vs net__ cache):', json.dumps(al), flush=True)
    del o, y, h2, c2, t2
    P = {}
    P['AN_sc'] = get(name, 'an_sc', lambda: allneural(Hc, heat, Ls, True))
    P['AN_nosc'] = get(name, 'an_nosc', lambda: allneural(Hc, heat, Ls, False))
    P['NOMP_re'] = get(name, 'nomp', lambda: classical(Ht, Ls, 'greedy'))
    P['HEAT_re'] = get(name, 'heat', lambda: classical(Hc, Ls, 'heat', heat, SC))
    P['NOMP_cache'] = load_pred(f'{CACHE}/pred__NOMP_no_network__{name}.npz')
    P['UNET_cache'] = load_pred(f'{CACHE}/pred__U-Net_paper__{name}.npz')
    P['AN_sc_s2'] = load_pred(f'{DRC_OLD}/{name}__allneural_sc.npz')
    for a, b in [('AN_sc', 'AN_sc_s2'), ('NOMP_re', 'NOMP_cache')]:
        d = maxdiff_deg(P[a], P[b])
        print(f'   {a} vs {b}: scenes identical (<1e-6 deg) {np.mean(d < 1e-6):.4f}, < 0.01 deg {np.mean(d < 0.01):.4f}, max {d.max():.3g} deg', flush=True)
    d = maxdiff_deg(P['AN_sc'], P['HEAT_re'])
    agree = dict(lt_0p01=float(np.mean(d < 0.01)), lt_0p1=float(np.mean(d < 0.1)), lt_1=float(np.mean(d < 1)), median=float(np.median(d)))
    print('   all-neural SC on vs classical heat decoder (recomputed): frac scenes max diff <0.01/<0.1/<1 deg', agree, flush=True)
    S = {k: own_stats(v, T) for k, v in P.items() if k != 'AN_sc_s2'}
    bk = dict(psi=T['psi'], phi=T['phi'], L=Ls, snr=snr.astype(np.float32))
    for k in ['AN_sc', 'UNET_cache']:
        Sc = C.scene_stats([(P[k]['psi'][i, :P[k]['n'][i]], P[k]['phi'][i, :P[k]['n'][i]]) for i in range(len(Ls))], bk)
        print(f'   metric cross-check {k}: own good/bad/drop = {S[k][:, :3].sum(0).tolist()}  common = {Sc[:, :3].sum(0).tolist()}', flush=True)
    sok = {k: (S[k][:, 0] == 2 * S[k][:, 3]) for k in ['AN_sc', 'HEAT_re']}
    xt = dict(both=int((sok['AN_sc'] & sok['HEAT_re']).sum()), only_allneural=int((sok['AN_sc'] & ~sok['HEAT_re']).sum()),
              only_classical=int((~sok['AN_sc'] & sok['HEAT_re']).sum()))
    r = dict(N=len(Ls), L=sorted(set(Ls.tolist())), alignment=al, agree_vs_heat=agree, all_angles_ok_crosstab=xt,
             pd={k: pd_mean(S[k], snr) for k in S}, pd_strict={k: pd_strict(S[k], snr) for k in S}, dropped={k: int(S[k][:, 2].sum()) for k in S})
    Bt = boot(S, snr)
    r['diff'] = {}
    for a, b in [('AN_sc', 'UNET_cache'), ('AN_sc', 'NOMP_re'), ('AN_sc', 'NOMP_cache'), ('AN_sc', 'HEAT_re'), ('AN_sc', 'AN_nosc'), ('AN_nosc', 'HEAT_re'), ('NOMP_re', 'NOMP_cache')]:
        r['diff'][f'{a}-{b}'] = [r['pd'][a] - r['pd'][b], *ci(Bt[a] - Bt[b])]
    print('   Pd paper-style:', {k: round(v, 4) for k, v in r['pd'].items()}, flush=True)
    print('   Pd strict    :', {k: round(v, 4) for k, v in r['pd_strict'].items()}, ' dropped', r['dropped'], flush=True)
    for k, v in r['diff'].items():
        print(f'   d {k:22s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]', flush=True)
    print('   all-angles-ok cross-tab (AN_sc vs recomputed classical heat):', xt, flush=True)
    if name == 'official':
        sel = np.concatenate([np.where(snr == s)[0][:250] for s in sorted(set(snr.tolist()))]); rest = np.setdiff1d(np.arange(len(snr)), sel)
        r['split'] = {}
        for tag, idx in [('exp5_design_subset_2000', sel), ('held_out_6000', rest)]:
            Bs = boot(S, snr, idx)
            r['split'][tag] = {f'{a}-{b}': [pd_mean(S[a], snr, idx) - pd_mean(S[b], snr, idx), *ci(Bs[a] - Bs[b])] for a, b in [('AN_sc', 'UNET_cache'), ('AN_sc', 'NOMP_re'), ('AN_sc', 'HEAT_re')]}
            r['split'][tag]['pd_AN_sc'] = pd_mean(S['AN_sc'], snr, idx)
            print(f'   [{tag}] AN_sc Pd {r["split"][tag]["pd_AN_sc"]:.4f} ' + ' '.join(f'd {k} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]' for k, v in r['split'][tag].items() if k != 'pd_AN_sc'), flush=True)
        r['per_snr'] = {}
        for s in sorted(set(snr.tolist())):
            idx = np.where(snr == s)[0]; Bs = boot(S, snr, idx)
            q = {k: pd_mean(S[k], snr, idx) for k in ['AN_sc', 'NOMP_re', 'UNET_cache', 'HEAT_re']}
            q['d_NOMP'] = [q['AN_sc'] - q['NOMP_re'], *ci(Bs['AN_sc'] - Bs['NOMP_re'])]
            q['d_UNET'] = [q['AN_sc'] - q['UNET_cache'], *ci(Bs['AN_sc'] - Bs['UNET_cache'])]
            r['per_snr'][s] = q
            print(f'   SNR {s:4d}: AN {q["AN_sc"]:.4f} NOMP {q["NOMP_re"]:.4f} UNet {q["UNET_cache"]:.4f} | d NOMP {q["d_NOMP"][0]:+.4f} [{q["d_NOMP"][1]:+.4f}, {q["d_NOMP"][2]:+.4f}]  d UNet {q["d_UNET"][0]:+.4f} [{q["d_UNET"][1]:+.4f}, {q["d_UNET"][2]:+.4f}]', flush=True)
    RES[name] = r
    del heat, Hc, Ht, z, T, P, S
    json.dump(RES, open(f'{HERE}/a1_replicate.json', 'w'), indent=1, default=float)
print('done', flush=True)
