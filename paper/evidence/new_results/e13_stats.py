"""E13+E6 step 2: metrics() quantities (pd_paper, pd_strict, pd_source, conditional rmse) per bank and method, mean over SNR groups,
and paired bootstrap (2000 resamples, scenes resampled within each SNR group, identical indices for all methods, seed 0) of four differences.
Methods: U-Net_paper, IABR-v2 (hybrid), NOMP_no_network, NOMP+selfcal (preds/selfcal_pred__<bank>.npz from e13_decode.py).
usage: python e13_stats.py bank1,bank2,..."""
import sys; sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
from common import *
os.makedirs(f'{OUT}/bank_json', exist_ok=True)
FILES = {'U-Net': lambda n: f'{CACHE}/pred__U-Net_paper__{n}.npz', 'hybrid': lambda n: f'{CACHE}/pred__IABR-v2__{n}.npz',
         'NOMP': lambda n: f'{CACHE}/pred__NOMP_no_network__{n}.npz', 'NOMP+selfcal': lambda n: f'{OUT}/preds/selfcal_pred__{n}.npz'}
DIFFS = [('NOMP - U-Net', 'NOMP', 'U-Net'), ('hybrid - NOMP', 'hybrid', 'NOMP'), ('NOMP+selfcal - hybrid', 'NOMP+selfcal', 'hybrid'), ('NOMP+selfcal - NOMP', 'NOMP+selfcal', 'NOMP')]
for name in sys.argv[1].split(','):
    t0 = time.time(); b = load_bank(name); N = len(b['L'])
    S = {}
    for m, f in FILES.items():
        z = np.load(f(name)); assert len(z['n']) == N, (name, m)
        S[m] = scene_stats([(z['psi'][i, :z['n'][i]], z['phi'][i, :z['n'][i]]) for i in range(N)], b)
    B = boot_metrics(S, b['snr'], B=2000, seed=0)
    snrs = sorted(set(b['snr'].astype(int).tolist()))
    res = dict(bank=name, n=N, L=sorted(set(b['L'].tolist())), snr_groups=snrs, bootstrap=dict(B=2000, seed=0, unit='scenes within SNR group'), methods={}, diffs={})
    for m in S:
        per = metric_by_snr(S[m], b['snr']); mean = per['mean']
        res['methods'][m] = dict(mean=mean, ci={k: ci(B[m][:, j])[1:] for j, k in enumerate(KEYS)},
                                 per_snr={str(s): per[s] for s in snrs} if len(snrs) > 1 else None, dropped=int(S[m][:, 2].sum()))
    for nm, a_, b_ in DIFFS:
        d = B[a_] - B[b_]; ma, mb = res['methods'][a_]['mean'], res['methods'][b_]['mean']
        res['diffs'][nm] = {k: [ma[k] - mb[k], *ci(d[:, j])[1:]] for j, k in enumerate(KEYS)}
    json.dump(res, open(f'{OUT}/bank_json/{name}.json', 'w'), indent=1, default=float)
    print(f'{name:22s} N={N} | ' + ' | '.join(f'{m} {res["methods"][m]["mean"]["pd_paper"]:.4f}' for m in S) + ' | ' +
          ' | '.join(f'{nm} {res["diffs"][nm]["pd_paper"][0]:+.4f} [{res["diffs"][nm]["pd_paper"][1]:+.4f},{res["diffs"][nm]["pd_paper"][2]:+.4f}]' for nm, _, _ in DIFFS) + f' | {time.time()-t0:.0f}s', flush=True)
