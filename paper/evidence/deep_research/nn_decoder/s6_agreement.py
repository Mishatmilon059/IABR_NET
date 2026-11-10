"""Step 6: how close are the all-neural (SC on) predictions to the classical heat-seeded decoder's, scene by scene (both start from the same
heatmap; official + T2 banks)? Also cross-tab of 'all L paths within 1 deg' (pd_source per scene). Light; reads cached predictions only.
usage: python -u s6_agreement.py > s6_agreement.log"""
import os, sys, json
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
import numpy as np
import common as C
HERE = os.path.dirname(os.path.abspath(__file__))
DRC = 'C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/nn_decoder'
res = {}
for n in ['official', 'T2_d1', 'T2_d2', 'T2_d5']:
    a = np.load(f'{DRC}/{n}__allneural_sc.npz'); h = np.load(f'{C.OUT}/preds_heat/{n}.npz'); q = np.load(f'{DRC}/{n}__allneural_nosc.npz')
    def dmax(x, y):    # order-free per-scene max angular difference (deg) after matching y to x by the min total distance
        import itertools
        out = np.zeros(len(x['psi']))
        for i in range(len(out)):
            P = np.stack([x['psi'][i], x['phi'][i]], 1); Q = np.stack([y['psi'][i], y['phi'][i]], 1)
            best = min(itertools.permutations(range(len(Q))), key=lambda p: np.abs(P - Q[list(p)]).sum())
            out[i] = np.degrees(np.abs(P - Q[list(best)]).max())
        return out
    d = dmax(a, h); d0 = dmax(q, h)
    b = C.load_bank(n)
    Sa = C.scene_stats(C.load_preds(f'{DRC}/{n}__allneural_sc.npz'), b); Sh = C.scene_stats(C.load_preds(f'{C.OUT}/preds_heat/{n}.npz'), b)
    ga, gh = Sa[:, 3] == Sa[:, 4], Sh[:, 3] == Sh[:, 4]
    r = dict(sc_on_vs_classical={f'frac_scenes_maxdiff_lt_{t}deg': float(np.mean(d < t)) for t in [0.01, 0.1, 1.0]},
             sc_off_vs_classical={f'frac_scenes_maxdiff_lt_{t}deg': float(np.mean(d0 < t)) for t in [0.01, 0.1, 1.0]},
             median_maxdiff_deg_sc_on=float(np.median(d)), all_paths_ok_crosstab=dict(both=int((ga & gh).sum()), only_allneural=int((ga & ~gh).sum()),
                                                                                         only_classical=int((~ga & gh).sum()), neither=int((~ga & ~gh).sum())))
    res[n] = r; print(n, json.dumps(r), flush=True)
json.dump(res, open(f'{HERE}/s6_agreement.json', 'w'), indent=1)
