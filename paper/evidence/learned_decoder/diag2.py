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



U0, V0 = heat_peaks(heat, 3); UA, VA = decode(U0, V0); okA, errA = flags(UA, VA)
UC, VC = decode(UT.copy(), VT.copy()); okC, errC = flags(UC, VC)
RA = residual_energy(Hc, UA, VA); RT = residual_energy(Hc, UT, VT); RC = residual_energy(Hc, UC, VC)
sA, sC = okA.all(1), okC.all(1)
sig2 = RT / (256 - 9)                      # noise energy per data entry, estimated at the truth (3 complex gains)
dE = (RA - RT) / sig2                      # how much worse (in noise units) the estimate fits than the truth does
print('overall sample success (all 6 comps within 1 deg): heat-init %.3f | truth-init %.3f' % (sA.mean(), sC.mean()))
fail = ~sA
print('\nFAILED samples (heat-init): %d' % fail.sum())
c1 = fail & (dE <= 0); c2 = fail & (dE > 0) & (dE <= 16.8); c3 = fail & (dE > 16.8)
print('  estimate fits data BETTER than the truth (dE<=0):        %4d  %.0f%%  -> pure noise/objective error, no optimiser can fix' % (c1.sum(), 100*c1.sum()/fail.sum()))
print('  fits slightly worse, inside chi2(6) 99%% band (dE<=16.8): %4d  %.0f%%  -> statistically indistinguishable from truth' % (c2.sum(), 100*c2.sum()/fail.sum()))
print('  fits clearly worse than the truth (dE>16.8):             %4d  %.0f%%  -> true optimiser/initialiser miss' % (c3.sum(), 100*c3.sum()/fail.sum()))
print('  failed with heat-init but fine with truth-init: %d  (their median dE = %.1f)' % ((fail & sC).sum(), np.median(dE[fail & sC]) if (fail & sC).any() else float('nan')))

print('\nper SNR   sample-success: heat-init  truth-init   | failures that are true optimiser misses (dE>16.8)')
for s_ in sorted(set(snr.tolist())):
    m = snr == s_
    print('  %4d        %.3f       %.3f              %5.1f%% of all samples' % (int(s_), sA[m].mean(), sC[m].mean(), 100*(c3 & m).sum()/m.sum()))

# what do the high-SNR failures look like?
hi = snr >= 10
endf = (np.abs(np.cos(PT)) > 0.995).any(1) | (np.abs(np.cos(FT)) > 0.995).any(1)
tp, tf = np.degrees(PT), np.degrees(FT)
minsep = np.array([min(np.hypot(tp[k, a] - tp[k, b], tf[k, a] - tf[k, b]) for a, b in [(0, 1), (0, 2), (1, 2)]) for k in range(N)])
print('\nHIGH-SNR samples (>=10 dB): %d' % hi.sum())
for name, grp in [('has an end-fire path (|cos|>0.995)', endf), ('no end-fire path', ~endf)]:
    m = hi & grp; print('  %-38s n=%4d  success %.3f' % (name, m.sum(), sA[m].mean()))
for lo_, hi_ in [(0, 10), (10, 20), (20, 40), (40, 400)]:
    m = hi & (minsep >= lo_) & (minsep < hi_) & ~endf
    print('  no end-fire, closest pair %3d-%3d deg  n=%4d  success %.3f' % (lo_, hi_, m.sum(), sA[m].mean() if m.sum() else float('nan')))
m = hi & ~endf & (minsep >= 20)
print('  => clean case (no end-fire, pairs >=20 deg apart): success %.3f   (n=%d)' % (sA[m].mean(), m.sum()))
