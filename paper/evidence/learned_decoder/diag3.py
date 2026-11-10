import numpy as np, scipy.ndimage, itertools, json, time
ROOT = 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2'
src = open(f'{ROOT}/notebook_src.py', encoding='utf-8').read().split('\n')
g = dict(np=np, scipy=scipy, CFG=dict(nomp_os=4, nomp_rc=3)); exec('\n'.join(src[527:647]), g)
newton, cyclic, heat_peaks, uv_to_angles, residual_energy = [g[k] for k in ['newton', 'cyclic', 'heat_peaks', 'uv_to_angles', 'residual_energy']]
PERMS = list(itertools.permutations(range(3)))
NTEST = 250

b = np.load('D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz'); n = np.load(f'{ROOT}/outputs/cache/net__official.npz')
psi_t, phi_t, snr_all = b['feat'][:, 0], b['feat'][:, 1], b['meta'][:, 1]
idx = np.concatenate([np.where(snr_all == s)[0][:NTEST] for s in sorted(set(snr_all.tolist()))])
heat, Hc, PT, FT, snr = n['heat'][idx], n['Hc'][idx].astype(np.complex128), psi_t[idx][:, :3], phi_t[idx][:, :3], snr_all[idx]
UT, VT = np.pi * np.cos(PT), np.pi * np.cos(FT)
N = len(idx); print('samples', N, flush=True)


def flags(U, V):
    """per-sample (n,6) booleans: matched psi/phi errors within 1 deg; also abs errors (deg)"""
    ps, ph = uv_to_angles(U, V); ps, ph = np.degrees(ps), np.degrees(ph); tp, tf = np.degrees(PT), np.degrees(FT)
    ok = np.zeros((len(U), 6), bool); err = np.zeros((len(U), 6))
    for k in range(len(U)):
        best = min(PERMS, key=lambda p: sum(abs(ps[k, q] - tp[k, j]) + abs(ph[k, q] - tf[k, j]) for j, q in enumerate(p)))
        e = [abs(ps[k, q] - tp[k, j]) for j, q in enumerate(best)] + [abs(ph[k, q] - tf[k, j]) for j, q in enumerate(best)]
        err[k] = e; ok[k] = np.array(e) <= 1
    return ok, err


def decode(U0, V0, hc=Hc, pre=4, rc=3):
    Un, Vn = newton(hc[:, None].repeat(3, 1).reshape(-1, 16, 16), U0.ravel(), V0.ravel(), iters=pre); Un, Vn = Un.reshape(-1, 3), Vn.reshape(-1, 3)
    return cyclic(hc, Un, Vn, rc)


def pd(ok): return float(ok.mean())



from scipy.stats import norm
ls_gains = g['ls_gains']
U0, V0 = heat_peaks(heat, 3); UA, VA = decode(U0, V0); okA, errA = flags(UA, VA)
x, _ = ls_gains(Hc, UT, VT); amp2 = np.abs(x) ** 2                      # (n,3) true path power
RT = residual_energy(Hc, UT, VT); sig2 = RT / (256 - 3)                 # noise energy per data entry
sd_u = np.sqrt(sig2[:, None] / (2 * amp2 * 5440))                       # Cramer-Rao std of u (and v), single-path, rad
d1 = np.radians(1.0)


def p_within(theta, sd, clamp):
    """P(|theta_hat - theta| <= 1 deg) when u_hat ~ N(pi cos theta, sd^2); theta in rad (psi or phi)"""
    mu = np.pi * np.cos(theta)
    lo_t, hi_t = np.minimum(theta + d1, np.pi), np.maximum(theta - d1, 0.0)
    u_lo, u_hi = np.pi * np.cos(lo_t), np.pi * np.cos(hi_t)
    if clamp:
        u_lo = np.where(theta + d1 >= np.pi, -np.inf, u_lo); u_hi = np.where(theta - d1 <= 0, np.inf, u_hi)
    return norm.cdf((u_hi - mu) / sd) - norm.cdf((u_lo - mu) / sd)


P_wrap = np.concatenate([p_within(PT, sd_u, False), p_within(FT, sd_u, False)], 1)
P_clamp = np.concatenate([p_within(PT, sd_u, True), p_within(FT, sd_u, True)], 1)
obs = (errA <= 1).astype(float)                                          # (n,6) matched to truth path order
cosall = np.concatenate([np.abs(np.cos(PT)), np.abs(np.cos(FT))], 1); endf = cosall > 0.995
print('Component success (within 1 deg) vs the Cramer-Rao-limit prediction (ideal estimator at the noise limit)')
print('SNR   observed   CRB-limit(wrap at +-pi)   CRB-limit(clamp at +-pi)')
for s_ in sorted(set(snr.tolist())):
    m = snr == s_
    print('%4d   %.3f        %.3f                    %.3f' % (int(s_), obs[m].mean(), P_wrap[m].mean(), P_clamp[m].mean()))
print('all    %.3f        %.3f                    %.3f' % (obs.mean(), P_wrap.mean(), P_clamp.mean()))
print('\nSplit by whether the true angle is within ~6 deg of end-fire (0/180):')
for name, m in [('end-fire comps', endf), ('other comps   ', ~endf)]:
    print('  %s n=%5d  observed %.3f | CRB-wrap %.3f | CRB-clamp %.3f' % (name, m.sum(), obs[m].mean(), P_wrap[m].mean(), P_clamp[m].mean()))
print('\nWeak paths: component success vs the true path power rank (weakest path of the 3):')
rk = np.argsort(np.argsort(amp2, 1), 1)                                  # 0 = weakest
for r_, nm in [(0, 'weakest'), (1, 'middle '), (2, 'strongest')]:
    m6 = np.concatenate([rk == r_, rk == r_], 1)
    print('  %s path: observed %.3f | CRB-clamp %.3f | median path SNR %.1f dB' % (nm, obs[m6].mean(), P_clamp[m6].mean(), 10*np.log10(np.median((amp2/sig2[:, None])[rk == r_]))))
