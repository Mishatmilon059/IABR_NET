"""Follow-up (pre-specified, no tuning): at b = 1, 15 dB the hybrid 'ML from FT network seeds' beat the classical 'ML (all starts)' (s3).
Is that a search gap a purely classical search closes?  'ML wide search' = ml_quant with K = 7 FFT local maxima (35 subsets) + the other
classical starts + EM, then 8 jittered restarts around the per-scene winner (each path moved by +-1/2 of a 16-grid cell in u and v,
random signs from default_rng([8715, 7])), lowest exact NLL kept. Same test bank as s3 (seed 8715, 1000 scenes, b = 1), paired
with the s3 per-scene stats (PCACHE/s3_*.npz).  usage: python s7_restarts.py  (stdout -> s7_restarts.log; writes s7_restarts.json)"""
from advc import *

N, CH = 1000, 200
out = {}
for bits in [1]:
    name = f'b{bits}_snr15'
    base = SS.base_scenes(N, 3, 15.0, TEST_SEEDS[15]); Y = SS.generate(base, 'M5', bits); t0 = time.time()
    rng = np.random.default_rng([8715, 7])
    Sw, S3 = [], {}
    for c0 in range(0, N, CH):
        f = f'{PCACHE}/s7_{name}_{c0}.npz'
        sl = slice(c0, c0 + CH); bc = {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim and len(v) == N else v) for k, v in base.items()}
        jit = rng.choice([-1.0, 1.0], size=(8, 2, CH, 3)) * (np.pi / 16)
        if os.path.exists(f):
            Sw.append(np.load(f)['S'])
        else:
            y = Y[sl]; snr = bc['snr']
            o = ml_quant(y, bits, snr, 3, kappa=1.0, K=7)
            U, V, F, _ = o['ML (all starts)']
            lo, hi = mlq.cells_ri(y, bits, A.adc_cells); s = sigma_of(snr)[:, None]
            X = mlq.init_gains(MO, U, V, y, lo, hi, s)
            cands = [mlq.lm(MO, U, V, X, lo, hi, s, iters=10)]
            for j in range(8):
                Uj, Vj = U + jit[j, 0], V + jit[j, 1]
                cands.append(mlq.lm(MO, Uj, Vj, mlq.init_gains(MO, Uj, Vj, y, lo, hi, s), lo, hi, s))
            Ub, Vb, _, _, _ = mlq.pick(cands)
            Sc = A.stats(C.uv_to_angles(Ub, Vb), bc); np.savez(f, S=Sc); Sw.append(Sc)
            print(f'  {name} chunk {c0} done [{time.time() - t0:.0f}s]', flush=True)
        z = np.load(f'{PCACHE}/s3_{name}_{c0}.npz', allow_pickle=True)
        for m in ['ML (all starts)', 'ML from FT network seeds', 'FT network only', 'ML from truth']:
            S3.setdefault(m, []).append(z[m])
    S = {'ML wide search': np.concatenate(Sw), **{m: np.concatenate(v) for m, v in S3.items()}}
    pdm, ci_, D = boot(S, base['snr'], [('ML wide search', 'ML (all starts)'), ('ML from FT network seeds', 'ML wide search'),
                                        ('ML wide search', 'FT network only')])
    out[name] = dict(pd=pdm, ci=ci_, diffs=D, sec_per_scene=(time.time() - t0) / N)
    print(f'=== {name}: ' + ', '.join(f'{m} {pdm[m]:.4f}' for m in S), flush=True)
    for k, d in D.items():
        print(f'   {k:50s} {fmt(d)}', flush=True)
    json.dump(out, open(f'{HERE}/s7_restarts.json', 'w'), indent=1, default=float)
print('done', flush=True)
