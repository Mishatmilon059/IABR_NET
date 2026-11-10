"""E13/E6 step 1: NOMP+selfcal decode on the raw antenna-domain channel Ht (net__<bank>.npz), (lam,tau)=(1.0,5.0); also re-runs greedy-only and
checks it is bit-identical to the cached NOMP_no_network predictions. Saves selfcal_pred__<bank>.npz (psi,phi,n; same format as the cache).
usage: python e13_decode.py bank1,bank2,..."""
import sys; sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
from common import *
os.makedirs(f'{OUT}/preds', exist_ok=True)
for name in sys.argv[1].split(','):
    t0 = time.time(); b = load_bank(name); Ls = b['L']; Ht = np.load(f'{CACHE}/net__{name}.npz')['Ht']
    assert len(Ht) == len(Ls); Lu = sorted(set(Ls.tolist()))
    psi_g = np.full((len(Ls), max(Lu)), np.nan); phi_g = psi_g.copy(); psi_s = psi_g.copy(); phi_s = psi_g.copy()
    for L in Lu:
        ii = np.where(Ls == L)[0]
        for s in range(0, len(ii), 1000):
            jj = ii[s:s + 1000]; H = Ht[jj].astype(np.complex128)
            U, V = nomp_greedy(H, L, CFG['nomp_os'], CFG['nomp_rc']); a = uv_to_angles(U, V); psi_g[jj, :L], phi_g[jj, :L] = a
            U2, V2 = selfcal(H, U, V, *SC); a = uv_to_angles(U2, V2); psi_s[jj, :L], phi_s[jj, :L] = a
    z = np.load(f'{CACHE}/pred__NOMP_no_network__{name}.npz')
    same = bool(np.allclose(psi_g, z['psi'], equal_nan=True) and np.allclose(phi_g, z['phi'], equal_nan=True))
    maxdiff = float(np.nanmax(np.abs(np.concatenate([(psi_g - z['psi']).ravel(), (phi_g - z['phi']).ravel()]))))
    n = np.array(Ls)
    np.savez_compressed(f'{OUT}/preds/selfcal_pred__{name}.npz', psi=psi_s, phi=phi_s, n=n)
    print(f'{name:22s} N={len(Ls)} L={Lu} greedy recompute == cached NOMP: {same} (max abs diff {maxdiff:.2e} rad) | {time.time()-t0:.0f}s', flush=True)
