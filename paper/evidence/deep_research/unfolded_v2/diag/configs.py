"""Task 5 (adaptivity): run a grid of NOMP decoder settings on a bank and store per-scene correct-component counts.
Grid: oversampling os in {2,4,8,16} x cyclic rounds rc in {1,3,6} x self-calibration in {off, (lam 1, tau 5) = notebook default}.
usage: python configs.py <bank> (official = 2000-scene subset 250/SNR, val, train); arrays -> C: cache; stdout -> configs.log"""
import sys, time
from dcommon import *
bank = sys.argv[1]
if bank == 'official':
    b = load_bank('official'); idx = subset(b, 250)
    H_all = np.load(f'{CACHE}/net__official.npz')['Ht'][idx].astype(np.complex128)
    pt_all, ft_all, snr_all = b['psi'][idx].astype(np.float64), b['phi'][idx].astype(np.float64), b['snr'][idx].astype(np.float64)
else:
    z = np.load(f'{BIG}/bank_{bank}.npz'); H_all = z['Ht']; pt_all, ft_all, snr_all = z['psi'], z['phi'], z['snr']
N = len(H_all)
CFGS = [(os_, rc, sc) for os_ in [2, 4, 8, 16] for rc in [1, 3, 6] for sc in [0, 1]]
fn = f'{BIG}/configs_{bank}.npz'
NC = np.zeros((N, len(CFGS)), int); RES = np.zeros((N, len(CFGS)))
t0 = time.time()
for s in range(0, N, 200):
    sl = slice(s, min(s + 200, N)); y = H_all[sl]; pt, ft = pt_all[sl], ft_all[sl]
    cache_g = {}
    for k, (os_, rc, sc) in enumerate(CFGS):
        if (os_, rc) not in cache_g:
            cache_g[(os_, rc)] = nomp_greedy(y, 3, os_, rc)
        U, V = cache_g[(os_, rc)]
        if sc:
            U, V = selfcal(y, U, V, 1.0, 5.0)
        nc, _ = score_uv(pt, ft, U[:, None], V[:, None]); NC[sl, k] = nc[:, 0]; RES[sl, k] = residual_energy(y, U, V)
    print(f'{bank} {s + 200}/{N} {time.time()-t0:.0f}s', flush=True)
np.savez(fn, NC=NC, RES=RES, snr=snr_all, cfgs=np.array(CFGS))
snr = snr_all.astype(int)
print(f'\n{bank}: Pd per config (mean over SNR groups) and per SNR')
for k, c in enumerate(CFGS):
    per = [NC[snr == s, k].mean() / 6 for s in sorted(set(snr.tolist()))]
    print(f'os{c[0]:2d} rc{c[1]} sc{c[2]} | mean {np.mean(per):.4f} | ' + ' '.join(f'{p:.4f}' for p in per))
