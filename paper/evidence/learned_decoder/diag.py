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


# ---- A: classical decoder from the heatmap peaks
t0 = time.time(); U0, V0 = heat_peaks(heat, 3); UA, VA = decode(U0, V0); okA, errA = flags(UA, VA)
print(f'A  classical (heat init)           Pd {pd(okA):.3f}   [{time.time()-t0:.0f}s]', flush=True)

# ---- B: is the optimiser converged?  run far more iterations / rounds
UB, VB = decode(U0, V0, pre=30, rc=15); okB, _ = flags(UB, VB)
du = np.degrees(np.abs(np.angle(np.exp(1j * (UB - UA))))).max(1)
print(f'B  10x more iterations             Pd {pd(okB):.3f}   samples whose u moved >0.05 deg: {100*np.mean(du>0.05):.1f}%', flush=True)

# ---- C: oracle starts (what a perfect initialiser could give)
UC, VC = decode(UT.copy(), VT.copy()); okC, _ = flags(UC, VC)
print(f'C1 start AT the true angles        Pd {pd(okC):.3f}', flush=True)
rng = np.random.default_rng(0); pert = np.pi / 32
UD, VD = decode(UT + rng.uniform(-pert, pert, UT.shape), VT + rng.uniform(-pert, pert, VT.shape)); okD, _ = flags(UD, VD)
print(f'C2 true + random half-cell offset  Pd {pd(okD):.3f}', flush=True)

# ---- D: objective value (ML residual energy) at the estimate vs at the truth
RA = residual_energy(Hc, UA, VA); RT = residual_energy(Hc, UT, VT); RC = residual_energy(Hc, UC, VC)
fail = ~okA.all(1)
print(f'\nsamples with at least one of 6 components wrong (>1 deg): {fail.sum()} of {N} ({100*fail.mean():.1f}%)')
tol = 1e-3
cause_obj = fail & (RA <= RT * (1 + tol))        # the estimate fits the data at least as well as the truth does
cause_opt = fail & (RA > RT * (1 + tol))         # the truth fits better: optimiser/initialiser missed it
print(f'  objective-limited (estimate fits data as well/better than the TRUTH): {cause_obj.sum():4d}  ({100*cause_obj.sum()/max(1,fail.sum()):.0f}% of failures)')
print(f'  optimiser/init failure (the truth fits the data better):          {cause_opt.sum():4d}  ({100*cause_opt.sum()/max(1,fail.sum()):.0f}% of failures)')
rec = cause_opt & okC.all(1)
print(f'  of the optimiser/init failures, recovered by starting at the truth:{rec.sum():4d}')

print('\nper SNR:  Pd(A)  fail%  objective-limited%  optimiser/init%   mean residual E_est / E_true')
for s in sorted(set(snr.tolist())):
    m = snr == s; f = fail & m
    print(f'  {int(s):4d}   {pd(okA[m]):.3f}  {100*f.sum()/m.sum():5.1f}   {100*(cause_obj&m).sum()/m.sum():5.1f}              {100*(cause_opt&m).sum()/m.sum():5.1f}          {np.median(RA[m]/RT[m]):.3f}')

# ---- E: where the failing components are (end-fire?)
ef = np.concatenate([np.abs(np.cos(PT)) > 0.995, np.abs(np.cos(FT)) > 0.995], 1)
# flags are in matched order, not truth order: use error size only
big = errA > 90
print(f'\ncomponents with error >90 deg (fold errors): {big.sum()}   of all wrong components: {(~okA).sum()}')
json.dump(dict(pdA=pd(okA), pdB=pd(okB), pdC1=pd(okC), pdC2=pd(okD)), open('diag.json', 'w'))
