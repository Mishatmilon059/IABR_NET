"""Unit checks of the ML machinery (seed 8745 tuning bank only, first 60 scenes):
 (1) the model mean mu(true u, v, alpha) equals the noiseless beamspace channel of stress_sim.generate (clean minus noise);
 (2) acommon.adc_cells: the unquantised z lies inside its reported cell for every sample, b = 1..4 (cell recovery from the output);
 (3) analytic NLL gradient / curvature vs finite differences;
 (4) smoke run of ml_quant at b = 1, 15 dB: Pd of consistent NOMP, ML variants, oracle-started ML, NLL(truth) vs NLL(estimate).
usage: python t0_unit.py > t0_unit.log"""
from advc import *

n = 60
b = SS.base_scenes(300, 3, 15.0, TUNE_SEEDS[15])
b = {k: (v[:n] if isinstance(v, np.ndarray) and v.ndim and len(v) == 300 else v) for k, v in b.items()}
Y0 = SS.generate(b, 'clean') - b['noise']
U = np.pi * np.cos(b['psi']); V = np.pi * np.cos(b['phi'])
mu = MO.mu(U, V, b['alpha'])
print('(1) max|mu(true) - noiseless Y| =', float(np.abs(mu - Y0).max()), ' max|Y0| =', float(np.abs(Y0).max()))
Z = SS.generate(b, 'clean')
for bits in [1, 2, 3, 4]:
    Yq = SS.adc(Z, bits, 'peak'); assert np.array_equal(Yq, SS.generate(b, 'M5', bits))
    rl, rh, il, ih = A.adc_cells(Yq, bits)
    inside = ((Z.real >= rl - 1e-9) & (Z.real <= rh + 1e-9) & (Z.imag >= il - 1e-9) & (Z.imag <= ih + 1e-9)).mean()
    print(f'(2) b={bits}: fraction of unquantised samples inside the recovered cell = {inside:.6f}; Yq == generate(M5) True')
# (3) derivative check
rng = np.random.default_rng(0)
lo = np.array([[-1.0, 0.0, -3.0, 0.5]]); hi = np.array([[0.0, 2.0, 3.0, 9.0]]); s = np.array([[0.3]])
m = rng.normal(size=(1, 4)) * 2
f, g, h = mlq.nll_terms(m, lo, hi, s); e = 1e-5
fp = mlq.nll_terms(m + e, lo, hi, s); fm = mlq.nll_terms(m - e, lo, hi, s)
print('(3) grad analytic', np.round(g, 5), 'fd', np.round((fp[0] - fm[0]) / (2 * e), 5))
print('    curv analytic', np.round(h, 5), 'fd', np.round((fp[1] - fm[1]) / (2 * e), 5))
# (4) smoke
for bits in [1, 2]:
    Yq = SS.generate(b, 'M5', bits); t0 = time.time()
    out = ml_quant(Yq, bits, b['snr'], extra={'truth': (U[:, :3], V[:, :3])})
    dt = time.time() - t0
    lo, hi = mlq.cells_ri(Yq, bits, A.adc_cells); s = sigma_of(b['snr'])[:, None]
    Xt = b['alpha'][:, :3]; Ft = mlq.nll(MO, U[:, :3], V[:, :3], Xt, lo, hi, s)
    Uc, Vc, _ = A.consistent_nomp(Yq, bits, 3, 8)
    print(f'(4) b={bits} 15 dB, {n} scenes, {dt:.1f}s: consistent NOMP Pd {A.pd(A.stats(C.uv_to_angles(Uc, Vc), b)):.3f}')
    for k, v in out.items():
        if k.startswith('_'):
            continue
        print(f'    {k:28s} Pd {A.pd(A.stats(C.uv_to_angles(v[0], v[1]), b)):.3f}   median NLL - NLL(truth, true gains) {np.median(v[2] - Ft):+.2f}')
    print('    start chosen counts:', dict(zip(out['_start_names'], np.bincount(out['ML (all starts)'][3], minlength=len(out['_start_names'])).tolist())))
