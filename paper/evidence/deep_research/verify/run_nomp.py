"""Steps 4 and 6: run the independent NOMP (nomp_mine.py) on the raw antenna-domain channel Ht of every bank (complex64 cache cast to
complex128), known L, os=4, 3 cyclic rounds. Compares per scene with the cached NOMP_no_network predictions. On the official bank also
runs the oracle-seeded decoder (true (u,v) as seeds -> cyclic re-estimation, 3 rounds x 2 Newton steps; and 10 rounds as a sensitivity check).
Deterministic (no random numbers). Saves cache/mynomp__<bank>.npz, cache/oracle3__official.npz, cache/oracle10__official.npz.
usage: python run_nomp.py   (stdout -> run_nomp.log)"""
import os, sys, json, time
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcommon import *
import nomp_mine as NM


def decode_bank(H, Ls, fn):
    Lm = int(Ls.max()); ps = np.full((len(Ls), Lm), np.nan); ph = ps.copy()
    for L in sorted(set(Ls.tolist())):
        ii = np.where(Ls == L)[0]
        for s in range(0, len(ii), 1000):
            jj = ii[s:s + 1000]
            U, V = fn(H[jj], L, jj)
            ps[jj, :L], ph[jj, :L] = NM.to_angles(U, V)
    return dict(psi=ps, phi=ph, n=Ls.astype(np.int64))


out = {}
for name in BANKS:
    t0 = time.time()
    b = load_truth(name); Ls = b['L']
    H = np.load(f'{CACHE}/net__{name}.npz')['Ht'].astype(np.complex128)
    assert len(H) == len(Ls)
    p = decode_bank(H, Ls, lambda h, L, jj: NM.nomp(h, L, 4, 3))
    np.savez_compressed(f'{MYCACHE}/mynomp__{name}.npz', **p)
    z = np.load(f'{CACHE}/pred__NOMP_no_network__{name}.npz')
    assert np.array_equal(z['n'], p['n'])
    d = np.fmax(np.nanmax(np.abs(p['psi'] - z['psi']), 1), np.nanmax(np.abs(p['phi'] - z['phi']), 1))
    r = dict(n=len(Ls), frac_lt_1e6rad=float((d < 1e-6).mean()), frac_lt_001deg=float((np.rad2deg(d) < 0.01).mean()),
             max_diff_rad=float(d.max()), n_differ_001deg=int((np.rad2deg(d) >= 0.01).sum()), sec=time.time() - t0)
    out[name] = r
    print(f'{name:20s} N={len(Ls):5d} L={sorted(set(Ls.tolist()))} | scenes with max|diff|<1e-6 rad: {r["frac_lt_1e6rad"]:.4f} | <0.01 deg: {r["frac_lt_001deg"]:.4f} '
          f'| max diff {r["max_diff_rad"]:.2e} rad | {r["sec"]:.1f}s', flush=True)

# oracle-seeded decoder on the official bank
b = load_truth('official'); H = np.load(f'{CACHE}/net__official.npz')['Ht'].astype(np.complex128)
U0 = np.pi * np.cos(b['psi'].astype(np.float64)); V0 = np.pi * np.cos(b['phi'].astype(np.float64))
for R in [3, 10]:
    t0 = time.time()
    p = decode_bank(H, b['L'], lambda h, L, jj: NM.oracle_seeded(h, U0[jj, :L], V0[jj, :L], R, 2))
    np.savez_compressed(f'{MYCACHE}/oracle{R}__official.npz', **p)
    print(f'oracle-seeded ({R} cyclic rounds) official done in {time.time()-t0:.1f}s', flush=True)
json.dump(out, open(f'{HERE}/run_nomp.json', 'w'), indent=1)
