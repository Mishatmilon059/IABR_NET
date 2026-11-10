"""Probe: noise scale in Ht, gain distribution at true angles, minimum separation in the official truth, NOMP separation
violations, decoder timing. stdout -> probe.log"""
import time, json
from dcommon import *
idx, tb, H, heat, U0, V0 = load_official(250)
Ut, Vt = true_uv(tb)
snr = tb['snr']
x, A = ls_gains(H, Ut, Vt)
res = residual_energy(H, Ut, Vt)
print('N subset', len(idx), 'L values', set(tb['L'].tolist()))
for s in sorted(set(snr.astype(int).tolist())):
    m = snr.astype(int) == s
    print(f'snr {s:4d}: residual/(256-6) at truth / sigma2 = {np.mean(res[m]) / 250 / sigma2(s):.3f}; mean sum|x|^2 at truth {np.mean((np.abs(x[m])**2).sum(1)):.3f}')
hi = snr >= 20
g = np.abs(x[hi]) ** 2
print('high-SNR |x_k|^2 mean per stored slot (order of feat columns):', g.mean(0).round(3), ' -> sorted strongest first?', (np.diff(g, axis=1) <= 0).all(1).mean().round(3))
print('|x|^2 overall mean (expect 1/L=0.333):', g.mean().round(4), ' var/mean^2 of |x|^2 (exp dist ->1):', (g.var() / g.mean() ** 2).round(3))
b = load_bank('official')
ps, ph = b['psi'].astype(np.float64), b['phi'].astype(np.float64)
d = np.full(len(ps), np.inf)
for i in range(3):
    for j in range(i + 1, 3):
        d = np.minimum(d, np.hypot(ps[:, i] - ps[:, j], ph[:, i] - ph[:, j]))
print(f'official truth (8000): min pairwise (psi,phi) distance min {np.rad2deg(d.min()):.3f} deg (rule pi/6 = 30 deg); frac < 30 deg: {(d < SEP - 1e-6).mean():.4f}')
print('truth psi range deg', np.rad2deg(ps.min()).round(2), np.rad2deg(ps.max()).round(2))
dn = sep_violation(U0, V0)
for s in sorted(set(snr.astype(int).tolist())):
    m = snr.astype(int) == s
    print(f'snr {s:4d}: NOMP solutions violating min-sep (<30 deg): {(dn[m] < SEP).mean():.3f}; <15 deg: {(dn[m] < SEP/2).mean():.3f}')
t = time.time(); Ug, Vg = nomp_greedy(H[:400], 3, 4, 3); print(f'nomp_greedy os4 rc3, 400 scenes: {time.time()-t:.1f}s; max |u diff| vs cache {np.abs(np.angle(np.exp(1j*(Ug-U0[:400])))).max():.2e}')
t = time.time(); Uc, Vc = cyclic(H[:400], Ut[:400], Vt[:400], 3); print(f'cyclic 3 rounds, 400 scenes: {time.time()-t:.1f}s')
t = time.time(); bt = bayes_terms(H[:400], Uc, Vc, snr[:400]); print(f'bayes terms 400: {time.time()-t:.2f}s')
