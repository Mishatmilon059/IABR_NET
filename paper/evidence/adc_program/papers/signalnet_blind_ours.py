"""SignalNet-style 'worst-case learning threshold' for OUR metrics (paper-style Pd, strict Pd, RMSE on hits).

Blind = input-independent prediction: the same set of 3 (psi, phi) pairs for every scene, whatever Y is. Same scenes, seeds and
candidate sets as Part B of signalnet_thresholds.py (seeds 81000202 / 81000203, 4000 scenes, L = 3, DG.generate_points with min
separation pi/6, common.scene_stats / metric_from_stats). The constant is chosen on even scenes and evaluated on odd scenes.
Adds: strict Pd, RMSE on hits (common.py averages squared error over components with |err| <= 1 deg only), SE of blind Pd, and the
analytic RMSE-on-hits of a blind guess (errors ~ uniform on [-1, 1] deg given a hit -> 1/sqrt(3) deg).
Usage: python signalnet_blind_ours.py  (stdout -> signalnet_blind_ours.log)
"""
import os, sys, time
os.environ['OMP_NUM_THREADS'] = '2'; os.environ['MKL_NUM_THREADS'] = '2'; os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ.setdefault('IABR2_SUITE', 'D:/ai_ml_project')
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
import common as C
import dldoa_dataset_generation as DG
t0 = time.time(); NS = 4000
rng = np.random.default_rng(81000202)
psi = np.zeros((NS, 3)); phi = np.zeros((NS, 3))
for b in range(NS):
    pts = DG.generate_points(3, np.pi / 6, rng=rng)
    phi[b] = [p[0] for p in pts]; psi[b] = [p[1] for p in pts]
Lv = np.full(NS, 3)
sel, ev = np.arange(0, NS, 2), np.arange(1, NS, 2)


def stats(P, Q, idx):
    return C.scene_stats([(np.array(P), np.array(Q))] * len(idx), dict(psi=psi[idx], phi=phi[idx], L=Lv[idx]))


cands = {'3 x centre (90,90)': ([np.pi / 2] * 3, [np.pi / 2] * 3)}
r2 = np.random.default_rng(81000203)
for k in range(40):
    cands['random set %02d' % k] = (list(r2.uniform(0, np.pi, 3)), list(r2.uniform(0, np.pi, 3)))
cands['spread triangle'] = ([np.pi / 2, np.pi / 4, 3 * np.pi / 4], [3 * np.pi / 4, np.pi / 4, np.pi / 4])
sc = {k: C.metric_from_stats(stats(*v, sel), np.arange(len(sel)))['pd_paper'] for k, v in cands.items()}
best = max(sc, key=sc.get)
print('blind (input-independent) predictor on our geometry: %d scenes (L=3, U[0,pi]^2, min sep pi/6); selection on %d, evaluation on %d'
      % (NS, len(sel), len(ev)))
for name in [best, '3 x centre (90,90)', 'spread triangle']:
    S = stats(*cands[name], ev); M = C.metric_from_stats(S, np.arange(len(ev)))
    hits = S[:, 0] / 6.0   # 6 angle components per scene (psi and phi of 3 paths)
    print('   %-20s Pd_paper %.4f (SE %.4f) | Pd_strict %.4f | RMSE on hits %.3f deg | hits %d of %d angle components | selection-half Pd %.4f'
          % (name, M['pd_paper'], hits.std(ddof=1) / np.sqrt(len(ev)), M['pd_strict'], M['rmse'], int(S[:, 0].sum()), 6 * len(ev), sc[name]))
print('   analytic RMSE on hits of any blind guess (error ~ U[-1,1] deg given |err|<=1): %.3f deg' % (1 / np.sqrt(3)))
print('   note: Pd and RMSE are SNR-independent for a blind predictor, so the same floor applies to every SNR cell.')
print('time %.1f s' % (time.time() - t0))
