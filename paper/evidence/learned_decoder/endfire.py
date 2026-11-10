import numpy as np, scipy.ndimage, itertools
src = open('D:/ai_ml_project/IABR_v2_extracted/IABR_v2/notebook_src.py', encoding='utf-8').read().split('\n')
g = dict(np=np, scipy=scipy, CFG=dict(nomp_os=4, nomp_rc=3)); exec('\n'.join(src[527:647]), g)
bank = np.load('D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz'); net = np.load('D:/ai_ml_project/IABR_v2_extracted/IABR_v2/outputs/cache/net__official.npz')
idx = np.where(bank['meta'][:, 1] == 20)[0][:400]
psi_p, phi_p = g['decode'](net['Hc'][idx], 3, net['heat'][idx], 'heat', (1.0, 5.0))
psi_p, phi_p = np.degrees(psi_p), np.degrees(phi_p)
T = np.degrees(bank['feat'][idx])                                   # (n,2,3)
res = {'psi': {True: [], False: []}, 'phi': {True: [], False: []}}
for k in range(len(idx)):
    best = min(itertools.permutations(range(3)), key=lambda p: sum(abs(psi_p[k, q] - T[k, 0, j]) + abs(phi_p[k, q] - T[k, 1, j]) for j, q in enumerate(p)))
    for j, q in enumerate(best):
        for name, pred, tru in (('psi', psi_p[k, q], T[k, 0, j]), ('phi', phi_p[k, q], T[k, 1, j])):
            ef = abs(np.cos(np.radians(tru))) > 0.995
            res[name][ef].append(abs(pred - tru))
for name in res:
    for ef in (True, False):
        a = np.array(res[name][ef])
        print(f'{name} end-fire={ef!s:5}: n={len(a):4d}  within 1 deg: {100*np.mean(a<=1):5.1f}%   error>90 deg (fold): {int((a>90).sum())}')
