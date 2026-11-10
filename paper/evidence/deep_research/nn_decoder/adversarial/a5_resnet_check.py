"""Own-metric check of the cached ResNet-port predictions (s5_resnet output on C:) on the official bank vs all-neural SC on (recomputed in a1).
Paper-style and strict Pd, own paired bootstrap (2000 within SNR, seed 0). usage: python -u a5_resnet_check.py > a5_resnet_check.log"""
import numpy as np
from scipy.optimize import linear_sum_assignment
SCR = 'C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache'
b = np.load('D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz'); tp, tf, snr = b['feat'][:, 0].astype(float), b['feat'][:, 1].astype(float), b['meta'][:, 1].astype(int)
def st(p):
    S = np.zeros((8000, 3))
    for i in range(8000):
        n = p['n'][i]; ps = p['psi'][i, :n]
        if n < 3 or np.isnan(ps[:3]).any(): S[i] = [0, 0, 1]; continue
        G = np.stack([tp[i], tf[i]], 1); P = np.stack([ps[:3], p['phi'][i, :3]], 1)
        r, c = linear_sum_assignment(np.linalg.norm(G[:, None] - P[None], axis=2)); ok = np.abs(np.degrees(np.angle(np.exp(1j * (G[r] - P[c]))))) <= 1
        S[i] = [ok.sum(), (~ok).sum(), 0]
    return S
P = {k: np.load(f) for k, f in [('resnet', f'{SCR}/nn_decoder/resnet__official.npz'), ('AN', f'{SCR}/nn_decoder_adv/official__an_sc.npz')]}
S = {k: st(v) for k, v in P.items()}
rng = np.random.default_rng(0); acc = {k: np.zeros((2000, 2)) for k in S}; pt = {k: np.zeros(2) for k in S}; G = sorted(set(snr.tolist()))
for s in G:
    ix = np.where(snr == s)[0]; R = rng.integers(0, len(ix), (2000, len(ix)))
    for k, X in S.items():
        gd, bd, dr = X[ix, 0][R].sum(1), X[ix, 1][R].sum(1), 6 * X[ix, 2][R].sum(1); acc[k] += np.stack([gd / (gd + bd), gd / (gd + bd + dr)], 1)
        t = X[ix].sum(0); pt[k] += [t[0] / (t[0] + t[1]), t[0] / (t[0] + t[1] + 6 * t[2])]
for k in S: print(k, 'Pd paper %.4f strict %.4f dropped %d' % (pt[k][0] / len(G), pt[k][1] / len(G), S[k][:, 2].sum()))
for j, m in enumerate(['paper', 'strict']):
    d = (acc['AN'][:, j] - acc['resnet'][:, j]) / len(G); print('AN - resnet %s %+.4f [%+.4f, %+.4f]' % (m, (pt['AN'][j] - pt['resnet'][j]) / len(G), *np.percentile(d, [2.5, 97.5])))
