"""Step 4: the 56 cached banks. NOMP (notebook nomp_greedy, os 4, rc 3, true L, no self-cal) run on the CACHED IABC-corrected Hc,
compared per scene with the cached NOMP-on-Ht predictions (pred__NOMP_no_network__<bank>.npz; for the 4 tune_* banks, which have no
cached NOMP prediction, NOMP is run here on the cached Ht). Paired bootstrap (2000 resamples within SNR groups, seed 0) of the
mean-over-SNR paper Pd difference (IABC+NOMP) - (NOMP-only). Deterministic, no random draws except the bootstrap.
usage: python s4_banks.py > s4_banks.log   (per-bank predictions -> cache/nomphc__<bank>.npz)"""
from icommon import *
import glob

names = sorted(os.path.basename(f)[5:-4] for f in glob.glob(f'{C.CACHE}/net__*.npz'))
# official first (the headline), then the rest
names = ['official'] + [n for n in names if n != 'official']
missing = [n for n in names if not C.bank_exists(n)]
print(f'{len(names)} cached banks; skipped (truth angles not on this machine, cannot be scored): {missing}', flush=True)
names = [n for n in names if C.bank_exists(n)]


def run(H, Ls):
    Lm = int(Ls.max()); ps = np.full((len(Ls), Lm), np.nan); ph = ps.copy()
    for L in sorted(set(Ls.tolist())):
        ii = np.where(Ls == L)[0]
        for s in range(0, len(ii), 400):
            jj = ii[s:s + 400]; U, V = C.nomp_greedy(H[jj], int(L), 4, 3)
            ps[jj, :L], ph[jj, :L] = C.uv_to_angles(U, V)
    return dict(psi=ps, phi=ph, n=Ls.astype(np.int64))


out = {}; t00 = time.time()
for name in names:
    t0 = time.time(); b = C.load_bank(name); Ls = b['L'].astype(int)
    z = np.load(f'{C.CACHE}/net__{name}.npz'); assert len(z['Hc']) == len(Ls), name
    fn = f'{CACHE}/nomphc__{name}.npz'
    if os.path.exists(fn):
        pc = dict(np.load(fn))
    else:
        pc = run(z['Hc'].astype(np.complex128), Ls); np.savez_compressed(fn, **pc)
    ref = f'{C.CACHE}/pred__NOMP_no_network__{name}.npz'
    if os.path.exists(ref):
        src = 'cached pred__NOMP_no_network'; pr = C.load_preds(ref)
    else:
        fr = f'{CACHE}/nompht__{name}.npz'
        if not os.path.exists(fr):
            np.savez_compressed(fr, **run(z['Ht'].astype(np.complex128), Ls))
        src = 'NOMP on cached Ht (run here)'; pr = C.load_preds(fr)
    pcl = [(pc['psi'][i, :pc['n'][i]], pc['phi'][i, :pc['n'][i]]) for i in range(len(Ls))]
    Sd = {'IABC+NOMP': C.scene_stats(pcl, b), 'NOMP': C.scene_stats(pr, b)}
    snr = b['snr']; Bt = C.boot_metrics(Sd, snr, B=2000, seed=0)
    mb = {m: C.metric_by_snr(s_, snr)['mean'] for m, s_ in Sd.items()}
    d = {k: [mb['IABC+NOMP'][k] - mb['NOMP'][k], *C.ci(Bt['IABC+NOMP'][:, i] - Bt['NOMP'][:, i])[1:]] for i, k in enumerate(['pd_paper', 'pd_strict'])}
    lo, hi = d['pd_paper'][1], d['pd_paper'][2]
    verdict = 'above0' if lo > 0 else ('below0' if hi < 0 else 'contains0')
    out[name] = dict(n=len(Ls), L=sorted(set(Ls.tolist())), snrs=sorted(set(snr.astype(int).tolist())), ref=src,
                     pd_nomp=mb['NOMP']['pd_paper'], pd_iabc_nomp=mb['IABC+NOMP']['pd_paper'], strict_nomp=mb['NOMP']['pd_strict'],
                     strict_iabc_nomp=mb['IABC+NOMP']['pd_strict'], diff_pd_paper=d['pd_paper'], diff_pd_strict=d['pd_strict'], verdict=verdict)
    print(f'{name:20s} N={len(Ls):5d} L={out[name]["L"]} | NOMP {mb["NOMP"]["pd_paper"]:.4f} IABC+NOMP {mb["IABC+NOMP"]["pd_paper"]:.4f} | '
          f'diff {d["pd_paper"][0]:+.4f} [{lo:+.4f},{hi:+.4f}] {verdict} | strict diff {d["pd_strict"][0]:+.4f} [{d["pd_strict"][1]:+.4f},{d["pd_strict"][2]:+.4f}] | {src} | {time.time() - t0:.0f}s', flush=True)
    json.dump(out, open(f'{HERE}/s4_banks.json', 'w'), indent=1, default=float)
cnt = {v: sum(r['verdict'] == v for r in out.values()) for v in ['above0', 'contains0', 'below0']}
print(f'banks: {len(out)} | CI above 0: {cnt["above0"]} | contains 0: {cnt["contains0"]} | below 0: {cnt["below0"]} | {time.time() - t00:.0f}s')
