import numpy as np, scipy.ndimage, itertools, sys
src = open('D:/ai_ml_project/IABR_v2_extracted/IABR_v2/notebook_src.py', encoding='utf-8').read().split('\n')
CFG = dict(nomp_os=4, nomp_rc=3)
g = dict(np=np, scipy=scipy, CFG=CFG)
exec('\n'.join(src[527:647]), g)          # decoder section (atoms ... decode)
atoms, ls_gains, newton, cyclic, heat_peaks, selfcal, uv_to_angles, residual_energy = [g[k] for k in
    ['atoms', 'ls_gains', 'newton', 'cyclic', 'heat_peaks', 'selfcal', 'uv_to_angles', 'residual_energy']]
NN_, MM_ = g['NN_'], g['MM_']

bank = np.load('D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz')
net = np.load('D:/ai_ml_project/IABR_v2_extracted/IABR_v2/outputs/cache/net__official.npz')
meta = bank['meta']; feat = bank['feat']
heat_all, Hc_all = net['heat'], net['Hc']
print('cache dtypes', heat_all.dtype, Hc_all.dtype, heat_all.shape, Hc_all.shape)


def pick(snr=20):
    for i in np.where(meta[:, 1] == snr)[0]:
        psi, phi = np.degrees(feat[i, 0]), np.degrees(feat[i, 1])
        d = min(np.hypot(psi[a] - psi[b], phi[a] - phi[b]) for a, b in [(0, 1), (0, 2), (1, 2)])
        if d > 15 and np.all(np.abs(np.cos(feat[i])) < 0.9):
            return i
i = pick(20)
L = 3
Hc = Hc_all[i][None]; heat = heat_all[i][None]
psi_t, phi_t = np.degrees(feat[i, 0]), np.degrees(feat[i, 1])
print(f'sample {i}: SNR {meta[i,1]:.0f} dB, L={int(meta[i,0])}')
print('TRUE  AoA psi (deg):', np.round(psi_t, 2), ' AoD phi (deg):', np.round(phi_t, 2))


def match_err(U, V):
    psi, phi = uv_to_angles(U, V); psi, phi = np.degrees(psi[0]), np.degrees(phi[0])
    best = None
    for perm in itertools.permutations(range(L)):
        e = sum(abs(psi[p] - psi_t[k]) + abs(phi[p] - phi_t[k]) for k, p in enumerate(perm))
        if best is None or e < best[0]: best = (e, perm)
    perm = best[1]
    return psi, phi, [(psi[p] - psi_t[k], phi[p] - phi_t[k]) for k, p in enumerate(perm)], perm


def show(tag, U, V):
    psi, phi, errs, perm = match_err(U, V)
    print(f'\n--- {tag} ---')
    for k, p in enumerate(perm):
        print(f'  path {k+1}: u={U[0,p]:+.4f} v={V[0,p]:+.4f} | psi={psi[p]:7.2f} phi={phi[p]:7.2f} | err psi={errs[k][0]:+.2f} deg, phi={errs[k][1]:+.2f} deg')
    print('  worst abs error (deg): %.2f' % max(max(abs(a), abs(b)) for a, b in errs))
    return perm


# ---- step 1: peak picking
mx = scipy.ndimage.maximum_filter(heat, size=(1, 3, 3), mode='wrap')
is_max = (heat >= mx)[0]
score = np.where(heat >= mx, heat + 1.0, heat).reshape(1, -1)
idx = np.argsort(-score, 1)[0, :L]
print('\n=== STEP 1: PEAK PICKING ===')
print('heatmap max value %.3f, number of local maxima (>=3x3 max): %d' % (heat.max(), is_max.sum()))
for j, k in enumerate(idx):
    r, c = divmod(int(k), 32)
    print(f'  peak {j+1}: row {r:2d}, col {c:2d}, value {heat[0, r, c]:.3f} -> u = -2*pi*{r}/32 = {-2*np.pi*r/32:+.4f}, v = 2*pi*{c}/32 = {2*np.pi*c/32:+.4f}')
U0, V0 = heat_peaks(heat, L)
perm = show('after peak picking (heatmap only)', U0, V0)

# ---- step 2: Newton, verbose for the first peak
print('\n=== STEP 2: NEWTON PER PATH ===  (path shown: peak 1)')
r = Hc[0]
u, v = float(U0[0, 0]), float(V0[0, 0])
st = 2 * np.pi / 64
for it in range(4):
    E = np.exp(1j * NN_ * u - 1j * MM_ * v); rE = r * E
    S = rE.sum(); Su = (1j * NN_ * rE).sum(); Sv = (-1j * MM_ * rE).sum()
    Suu = (-NN_ ** 2 * rE).sum(); Svv = (-MM_ ** 2 * rE).sum(); Suv = (NN_ * MM_ * rE).sum()
    f = abs(S) ** 2
    gu = 2 * np.real(np.conj(S) * Su); gv = 2 * np.real(np.conj(S) * Sv)
    huu = 2 * np.real(np.conj(Su) * Su + np.conj(S) * Suu); hvv = 2 * np.real(np.conj(Sv) * Sv + np.conj(S) * Svv)
    huv = 2 * np.real(np.conj(Su) * Sv + np.conj(S) * Suv)
    det = huu * hvv - huv ** 2; ok = (huu < 0) and (det > 0)
    du = -(hvv * gu - huv * gv) / det if ok else 0.0
    dv = -(-huv * gu + huu * gv) / det if ok else 0.0
    cdu, cdv = np.clip(du, -st, st), np.clip(dv, -st, st)
    print(f'  iter {it+1}: score f={f:9.2f} | slope g=({gu:+9.2f},{gv:+9.2f}) | curvature H=[[{huu:+9.1f},{huv:+8.1f}],[{huv:+8.1f},{hvv:+9.1f}]] '
          f'hill? {ok} | raw step=({du:+.4f},{dv:+.4f}) clipped=({cdu:+.4f},{cdv:+.4f}) | (u,v)=({u+cdu:+.4f},{v+cdv:+.4f})')
    u, v = u + cdu, v + cdv
print(f'  clip limit = pi/32 = {st:.4f} rad')
Un, Vn = newton(Hc[:, None].repeat(L, 1).reshape(-1, 16, 16), U0.ravel(), V0.ravel())
Un, Vn = Un.reshape(1, L), Vn.reshape(1, L)
show('after Newton (all 3 paths, each alone)', Un, Vn)

# ---- step 3: cyclic
print('\n=== STEP 3: CYCLIC REFINEMENT ===')
x, A = ls_gains(Hc, Un, Vn)
print('  LS gains |x| for the 3 paths:', np.round(np.abs(x[0]), 3))
res_before = residual_energy(Hc, Un, Vn)[0]
tot = (np.abs(Hc) ** 2).sum()
print(f'  energy left after removing the 3 fitted paths: {res_before:.3f} of total {tot:.2f}  ({100*res_before/tot:.2f}%)')
Uc, Vc = cyclic(Hc, Un, Vn, 3)
show('after cyclic refinement (3 rounds)', Uc, Vc)
res_after = residual_energy(Hc, Uc, Vc)[0]
print(f'  residual energy: {res_before:.4f} -> {res_after:.4f}')

# ---- step 4: self-cal, verbose
print('\n=== STEP 4: SELF-CALIBRATION (lam=1.0, tau=5) ===')
lam, tau = 1.0, 5.0
yc = Hc.copy(); U, V = Uc.copy(), Vc.copy()
x, A = ls_gains(yc, U, V); Hm = np.einsum('bik,bk->bi', A, x).reshape(Hc.shape)
snr_est = (np.abs(Hm) ** 2).sum() / ((np.abs(yc - Hm) ** 2).sum() + 1e-12)
print(f'  SNR_est = |Hm|^2 / |Hc-Hm|^2 = {snr_est:.1f}  (>{tau}: correction is applied = {snr_est > tau})')
y = Hc
dt = np.ones((1, 16), complex)
for _ in range(3):
    M = Hm * dt[:, None, :]; dr = (y * np.conj(M)).sum(2) / ((np.abs(M) ** 2).sum(2) + 1e-12)
    M = Hm * dr[:, :, None]; dt = (y * np.conj(M)).sum(1) / ((np.abs(M) ** 2).sum(1) + 1e-12)
_nc = g['_nc']


def clean(d):
    lg = np.log(np.abs(d) + 1e-12); ph = np.unwrap(np.angle(d), axis=1)
    lg = lg - lg.mean(1, keepdims=True); ph = ph - ph.mean(1, keepdims=True)
    ph = ph - (ph @ _nc)[:, None] / (_nc @ _nc) * _nc
    return lam * lg, lam * ph


lr, pr = clean(dr); lt, pt = clean(dt)
print('  estimated Rx errors: gain spread %.2f dB, phase spread %.2f deg;  Tx: gain %.2f dB, phase %.2f deg' % (
    20 * np.log10(np.e) * lr.std(), np.degrees(pr.std()), 20 * np.log10(np.e) * lt.std(), np.degrees(pt.std())))
cr = np.exp(-lr - 1j * pr); ct = np.exp(-lt - 1j * pt)
yc = y * cr[:, :, None] * ct[:, None, :]
print('  energy of the mismatch |Hc-Hm|^2 before fix: %.3f, after fix: %.3f' % ((np.abs(y - Hm) ** 2).sum(), (np.abs(yc - Hm) ** 2).sum()))
Us, Vs = selfcal(Hc, Uc, Vc, lam, tau)
show('FINAL (after self-calibration + re-refinement)', Us, Vs)
pf, hf = g['decode'](Hc, L, heat, 'heat', (lam, tau))
psi_f, phi_f = np.degrees(pf), np.degrees(hf)
print('\ndecode(mode="heat") output psi:', np.round(psi_f[0], 2), ' phi:', np.round(phi_f[0], 2))
print('TRUE   psi:', np.round(psi_t, 2), ' phi:', np.round(phi_t, 2))
