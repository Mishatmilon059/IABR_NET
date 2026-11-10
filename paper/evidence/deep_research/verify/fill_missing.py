"""Fill cached-prediction gaps for the side-by-side table (sep_2deg_snr15, phase_d0_snr0, phase_d0_snr15 have no
'network only' cache and no IABR-Net heat-seeded predictions in new_results/preds_heat).
  * network only = notebook parabolic_heat_decode (notebook_src.py lines 650-668) on the cached heatmap net__<bank>.npz['heat'];
    the recipe is first checked to reproduce the cached pred__v2_network_only_no_physics__official exactly.
  * IABR-Net heat-seeded = new_results/heat_main.py recipe: common.decode(Hc, L, heat, 'heat', SC=(1.0, 5.0));
    the recipe is first checked to reproduce new_results/preds_heat/sep_5deg_snr15.npz and gain_g4_snr15.npz exactly.
Saves cache/netonly__<bank>.npz and cache/heat__<bank>.npz.  usage: python fill_missing.py  (stdout -> fill_missing.log)"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcommon import *
sys.path.insert(0, NEWR)
import common as C
import scipy.ndimage

src = open(f'{ROOT}/notebook_src.py', encoding='utf-8').read().split('\n')
g = dict(np=np, scipy=scipy, uv_to_angles=C.uv_to_angles); exec('\n'.join(src[649:668]), g)
parabolic = g['parabolic_heat_decode']


def pack(preds, Lm):
    n = np.array([len(p[0]) for p in preds]); ps = np.full((len(preds), Lm), np.nan); ph = ps.copy()
    for i, p in enumerate(preds):
        ps[i, :n[i]] = p[0][:Lm]; ph[i, :n[i]] = p[1][:Lm]
    return dict(psi=ps, phi=ph, n=n)


def netonly(name):
    b = load_truth(name); heat = np.load(f'{CACHE}/net__{name}.npz')['heat']; Ls = b['L']; preds = [None] * len(Ls)
    for L in sorted(set(Ls.tolist())):
        ii = np.where(Ls == L)[0]
        for j, p in zip(ii, parabolic(heat[ii], L)):
            preds[j] = p
    return pack(preds, int(Ls.max()))


def heat_seeded(name):
    b = load_truth(name); z = np.load(f'{CACHE}/net__{name}.npz'); Ls = b['L']
    ps = np.full((len(Ls), int(Ls.max())), np.nan); ph = ps.copy()
    for L in sorted(set(Ls.tolist())):
        ii = np.where(Ls == L)[0]
        for s in range(0, len(ii), 1000):
            jj = ii[s:s + 1000]
            ps[jj, :L], ph[jj, :L] = C.decode(z['Hc'][jj].astype(np.complex128), L, z['heat'][jj].astype(np.float64), 'heat', C.SC)
    return dict(psi=ps, phi=ph, n=np.array(Ls))


def same(a, path):
    z = np.load(path)
    return bool(np.array_equal(a['n'], z['n'])), float(np.nanmax(np.abs(np.concatenate([(a['psi'] - z['psi']).ravel(), (a['phi'] - z['phi']).ravel()]))))


t0 = time.time(); r = same(netonly('official'), f'{CACHE}/pred__v2_network_only_no_physics__official.npz')
print(f'check network-only recipe on official vs cache: n identical {r[0]}, max |diff| {r[1]:.2e} rad ({time.time()-t0:.0f}s)', flush=True)
for nm in ['sep_5deg_snr15', 'gain_g4_snr15']:
    r = same(heat_seeded(nm), f'{NEWR}/preds_heat/{nm}.npz')
    print(f'check heat-seeded recipe on {nm} vs new_results/preds_heat: n identical {r[0]}, max |diff| {r[1]:.2e} rad', flush=True)
for nm in ['sep_2deg_snr15', 'phase_d0_snr0', 'phase_d0_snr15']:
    np.savez_compressed(f'{MYCACHE}/netonly__{nm}.npz', **netonly(nm))
    np.savez_compressed(f'{MYCACHE}/heat__{nm}.npz', **heat_seeded(nm))
    print(f'{nm}: network-only and heat-seeded predictions written', flush=True)
