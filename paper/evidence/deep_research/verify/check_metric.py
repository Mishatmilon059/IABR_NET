"""Step 2: cross-check the independent metric (metric_mine.py) against new_results/common.py (scene_stats / metric_by_snr, which
calls the evaluator functions compiled from TVT_Blob_Inference.py) and the notebook's own metrics()/by_snr() (exec-ed verbatim by common.py),
on every cached prediction set of the official bank. Also checks the paired bootstrap against common.boot_metrics.
usage: python check_metric.py   (stdout -> check_metric.log)"""
import sys, glob, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcommon import *
import metric_mine as MM
sys.path.insert(0, NEWR)
import common as C

b = load_truth('official')
paths = sorted(glob.glob(f'{CACHE}/pred__*__official.npz')) + [f'{NEWR}/preds/selfcal_pred__official.npz', f'{NEWR}/preds_heat/official.npz']
res = {}; Smine = {}; Sref = {}
worst = 0.0
for p in paths:
    nm = os.path.basename(p).replace('.npz', '')
    ps, ph, n = load_pred(p)
    S, nn = MM.scene_table(ps, ph, n, b['psi'], b['phi'], b['L'])
    mine = MM.by_snr(S, b['snr'])
    preds = [(ps[i, :n[i]], ph[i, :n[i]]) for i in range(len(n))]
    R = C.scene_stats(preds, b)                       # cols good, bad, dropped, src_ok, L, sumsq, L
    ref = C.metric_by_snr(R, b['snr'])
    nb = C.nb_by_snr(preds, b)                        # notebook metrics()/by_snr() verbatim
    Rm = R[:, [0, 1, 2, 3, 4, 5]]
    scene_max = np.abs(S - Rm).max(0)
    same_counts = bool(np.array_equal(S[:, :5], Rm[:, :5]))
    d = {k: [mine['mean'][k], ref['mean'][k], nb['mean'][k]] for k in MM.KEYS}
    agree4 = all(f'{v[0]:.4f}' == f'{v[1]:.4f}' == f'{v[2]:.4f}' for v in d.values())
    maxabs = max(max(abs(v[0] - v[1]), abs(v[0] - v[2])) for v in d.values()); worst = max(worst, maxabs)
    per_snr_max = max(abs(mine[s][k] - ref[s][k]) for s in mine if s != 'mean' for k in MM.KEYS)
    print(f'{nm:45s} mine pd {d["pd_paper"][0]:.4f} strict {d["pd_strict"][0]:.4f} src {d["pd_source"][0]:.4f} rmse {d["rmse"][0]:.4f} '
          f'| common pd {d["pd_paper"][1]:.4f} strict {d["pd_strict"][1]:.4f} src {d["pd_source"][1]:.4f} rmse {d["rmse"][1]:.4f} '
          f'| notebook pd {d["pd_paper"][2]:.4f} | per-scene counts identical {same_counts}, max |sumsq diff| {scene_max[5]:.2e}, '
          f'max |mean diff| {maxabs:.2e}, max |per-SNR diff| {per_snr_max:.2e}, 4-digit agreement {agree4}, NaN-pred scenes {nn}, dropped {mine["mean"]["dropped"]}', flush=True)
    res[nm] = dict(mine=mine['mean'], common=ref['mean'], notebook=nb['mean'], per_scene_counts_identical=same_counts,
                   max_abs_mean_diff=maxabs, max_abs_per_snr_diff=per_snr_max, agree_4_digits=agree4)
    Smine[nm] = S; Sref[nm] = R
# bootstrap cross-check on two methods
k1, k2 = 'pred__NOMP_no_network__official', 'pred__U-Net_paper__official'
Bm = MM.paired_bootstrap({k1: Smine[k1], k2: Smine[k2]}, b['snr'])
Bc = C.boot_metrics({k1: Sref[k1], k2: Sref[k2]}, b['snr'], B=2000, seed=0)
bd = float(np.nanmax(np.abs(Bm[k1] - Bc[k1]))); bd2 = float(np.nanmax(np.abs((Bm[k1] - Bm[k2]) - (Bc[k1] - Bc[k2]))))
print(f'bootstrap (2000, seed 0) mine vs common.boot_metrics: max |diff| NOMP replicates {bd:.2e}, NOMP-U-Net replicates {bd2:.2e}')
print(f'worst |mean metric diff| over all methods: {worst:.2e}')
json.dump(dict(methods=res, bootstrap_max_diff=bd, bootstrap_diff_max_diff=bd2, worst=worst), open(f'{HERE}/check_metric.json', 'w'), indent=1, default=float)
