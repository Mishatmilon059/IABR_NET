"""Fresh-seed spot check of the old-IABC diagnosis (d1_diagnose.log, part A at 25 dB and part C): cell (8 deg, 3 dB), L in {1,3,9},
SNR 25 dB, 300 scenes, seed 9360+L (noise [s,1], impairment [s,3]); severity sweep L=3, 25 dB, 600 scenes, seed 9370.
Reports old IABC % identifiable MSE removed + regression slope, and the raw-feature estimate (feature = estimate). stdout -> a6_d1_check.log"""
import os
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_k] = '6'
import sys
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/iabc_v2')
from v2common import *


def feat_est(Ht, Y):
    with torch.no_grad():
        f = cnn.iabc.features(torch.from_numpy(Ht.astype(np.complex64)), torch.from_numpy(Y.astype(np.complex64))).double().numpy()
    return f[..., [1, 0]]


def rem(tgt, est, j):
    e = ident_err(tgt, est)[..., j]; return 100 * (1 - (e ** 2).sum() / (tgt[..., j] ** 2).sum())


def slope(tgt, est, j):
    t = tgt[..., j].ravel(); e = est[..., j].ravel(); return float((t @ e) / (t @ t))


print('A (8 deg / 3 dB, 25 dB): | L | phase IABC % | phase slope | phase feature=est % | gain IABC % | gain slope | gain feature=est % |')
for L in [1, 3, 9]:
    d = make_scenes(9360 + L, 300, L, 25.0, 8, 3); Ht = to_ant(d['Y']); _, _, imp = net(d['Y']); fe = feat_est(Ht, d['Y']); t = d['tgt']
    print(f'| {L} | {rem(t, imp, 0):.1f} | {slope(t, imp, 0):.2f} | {rem(t, fe, 0):.1f} | {rem(t, imp, 1):.1f} | {slope(t, imp, 1):.2f} | {rem(t, fe, 1):.1f} |', flush=True)
print('C (L=3, 25 dB, 600 scenes, seed 9370): | cell | phase slope / % | gain slope / % |')
for nm, pd, gd in [('ph8', 8, 0), ('ph20', 20, 0), ('ph45', 45, 0), ('g3', 0, 3), ('g6', 0, 6)]:
    d = make_scenes(9370, 600, 3, 25.0, pd, gd); _, _, imp = net(d['Y']); t = d['tgt']
    s = f'{slope(t, imp, 0):.2f} / {rem(t, imp, 0):.1f}' if pd else '-'; g = f'{slope(t, imp, 1):.2f} / {rem(t, imp, 1):.1f}' if gd else '-'
    print(f'| {nm} | {s} | {g} |', flush=True)
print('DONE')
