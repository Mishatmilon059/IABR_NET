"""Round-1 revision of Section VII-A: paired bootstrap of hybrid - U-Net on the official bank, four metrics, with the procedure of e13_stats.py
(2000 resamples, scenes resampled within each SNR group, seed 0; resample indices do not depend on the set of methods)."""
import sys; sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
from common import *
name = 'official'; b = load_bank(name); N = len(b['L']); S = {}
for m, f in {'U-Net': f'{CACHE}/pred__U-Net_paper__{name}.npz', 'hybrid': f'{CACHE}/pred__IABR-v2__{name}.npz'}.items():
    z = np.load(f); assert len(z['n']) == N
    S[m] = scene_stats([(z['psi'][i, :z['n'][i]], z['phi'][i, :z['n'][i]]) for i in range(N)], b)
B = boot_metrics(S, b['snr'], B=2000, seed=0)
mean = {m: metric_by_snr(S[m], b['snr'])['mean'] for m in S}
for j, k in enumerate(KEYS):
    d = B['hybrid'][:, j] - B['U-Net'][:, j]
    print('hybrid - U-Net %-9s %+.4f [%+.4f, %+.4f]' % (k, mean['hybrid'][k] - mean['U-Net'][k], np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)))
