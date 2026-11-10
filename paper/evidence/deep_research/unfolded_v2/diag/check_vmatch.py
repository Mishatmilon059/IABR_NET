"""Check vectorised matching against common.scene_stats on the cached NOMP / U-Net predictions (official, 8000). stdout -> check_vmatch.log"""
from dcommon import *
b = load_bank('official')
for m in ['NOMP_no_network', 'IABR-v2']:
    preds = C.load_preds(f'{CACHE}/pred__{m}__official.npz')
    S = scene_stats(preds, b)
    z = np.load(f'{CACHE}/pred__{m}__official.npz')
    e, _ = vmatch(b['psi'].astype(np.float64), b['phi'].astype(np.float64), z['psi'][:, None].astype(np.float64), z['phi'][:, None].astype(np.float64))
    nc = (e[:, 0] <= 1).sum((1, 2))
    print(m, 'per-scene good-count identical:', bool(np.array_equal(nc, S[:, 0].astype(int))), 'mismatches', int((nc != S[:, 0]).sum()))
