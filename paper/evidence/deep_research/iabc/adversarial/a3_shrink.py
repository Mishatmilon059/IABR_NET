"""Is IABC's low-SNR gain removal (claim i.2) beyond what a SHRUNK classical ALS reaches?  The 'oracle ALS' of the original report
is the un-shrunk least-squares fit (lam 1), which is not an upper bound at low SNR. Here: per SNR, lam in {0.25,0.5,0.75,1.0} is
picked on TUNING scenes (seeds 8850 @0 dB, 8851 @15 dB) and evaluated on TEST scenes (seeds 8852 @0 dB, 8853 @15 dB), 400 scenes each,
L=3, cells g3 (0 deg, 3 dB) and ph8 (8 deg, 0 dB). Estimators: oracle ALS (true angles, angles fixed, tau 0, 4 it, lam) and
self-cal from NOMP angles (tau 0, 4 it, lam) -- the latter is a deployable classical estimator. Same independent error metric as a1.
usage: python -u a3_shrink.py > a3_shrink.log   (a3_shrink_attempt1.log = same run before the bootstrap CI line was added)"""
import os, sys
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/iabc')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icommon import *
import importlib.util
spec = importlib.util.spec_from_file_location('a1', os.path.dirname(os.path.abspath(__file__)) + '/a1_replicate.py')
src = open(spec.origin, encoding='utf-8').read().split('t00 = time.time()')[0]   # helper functions only (no runs)
g1 = dict(__file__=spec.origin, __name__='a1h'); exec(src, g1)
make, target, err_mse = g1['make'], g1['target'], g1['err_mse']
LAMS = [0.25, 0.5, 0.75, 1.0]; N = 400


def ests(seed, snr, pdeg, gdb):
    psi, phi, o = make(seed, snr, N, 3, pdeg, gdb); Y = o['Y']
    Ht = to_ant(Y); _, _, imp = net(Y); tph, tlg = target(o['D_r'], o['D_t'])
    Ug, Vg = C.nomp_greedy(Ht, 3, 4, 3); U0, V0 = true_uv(psi, phi)
    r = {'before': err_mse(tph, tlg, np.zeros_like(imp)), 'iabc': err_mse(tph, tlg, imp)}
    for lam in LAMS:
        r[f'oracle_lam{lam}'] = err_mse(tph, tlg, selfcal_ret(Ht, U0, V0, lam, 0.0, iters=4, fix_angles=True)[2])
        r[f'sc_lam{lam}'] = err_mse(tph, tlg, selfcal_ret(Ht, Ug, Vg, lam, 0.0, iters=4)[2])
    return r


res = {}
for cell, pdeg, gdb, q in [('g3', 0, 3, 1), ('ph8', 8, 0, 0)]:
    for k, snr in enumerate([0, 15]):
        tu = ests(8850 + k, snr, pdeg, gdb); te = ests(8852 + k, snr, pdeg, gdb)
        rm = lambda r, key: 1 - r[key][q].sum() / r['before'][q].sum()
        line = f'{cell} SNR {snr:2d} ({"gain" if q else "phase"}) TEST % MSE removed: IABC {100 * rm(te, "iabc"):.1f}%'
        for fam in ['oracle', 'sc']:
            best = max(LAMS, key=lambda l: rm(tu, f'{fam}_lam{l}'))
            line += f' | {fam} lam1 {100 * rm(te, f"{fam}_lam1.0"):.1f}% , {fam} lam*={best} (picked on tuning seeds) {100 * rm(te, f"{fam}_lam{best}"):.1f}%'
            line += ' [all lam on test: ' + ', '.join(f'{l}:{100 * rm(te, f"{fam}_lam{l}"):.1f}' for l in LAMS) + ']'
        bs = np.random.default_rng(0).integers(0, N, (2000, N)); bsc = max(LAMS, key=lambda l: rm(tu, f'sc_lam{l}'))
        dd = (te['iabc'][q][bs].sum(1) - te[f'sc_lam{bsc}'][q][bs].sum(1)) / te['before'][q][bs].sum(1)
        line += f' || (% removed by sc lam*) - (% removed by IABC) = {100 * (rm(te, f"sc_lam{bsc}") - rm(te, "iabc")):+.1f} pts [{100 * np.percentile(dd, 2.5):+.1f},{100 * np.percentile(dd, 97.5):+.1f}] (bootstrap 2000, seed 0)'
        if q == 1:
            line += f' | phase error ADDED by sc lam* in this gain-only cell: RMS {np.sqrt(te[f"sc_lam{bsc}"][0].mean()):.2f} deg (IABC {np.sqrt(te["iabc"][0].mean()):.2f} deg)'
        print(line, flush=True)
