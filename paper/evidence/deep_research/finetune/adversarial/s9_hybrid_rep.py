"""Replication of the only network advantage left after s3/s7: the ADC-fine-tuned CNN's read-out used as an EXTRA START of the
quantiser-ML refinement ('ML from FT network seeds') beat the classical 'ML (all starts)' at b = 1, 15 dB (+0.032*, s3) and a wider
classical search (+0.018*, s7) on seed 8715. Here: a NEW bank, seed 9515 (15 dB; grep-checked unused anywhere under D:/ai_ml_project),
1000 scenes, b = 1 and 2, same code as s3 (advc.ml_quant, kappa 1, nothing tuned). Methods: FT network only, ML (all starts),
ML from FT network seeds (single start), ML all starts + FT seeds (per-scene lowest NLL over the classical starts and the FT start).
Paired bootstrap 2000 (one SNR), seed 0. Resumable per 200-scene chunk.  usage: python s9_hybrid_rep.py (stdout -> s9_hybrid_rep.log)"""
from advc import *
N, CH, SEED = 1000, 200, 9515
ft = A.load_cnn(A.FT_CKPT)
out = {}
jp = f'{HERE}/s9_hybrid_rep.json'
if os.path.exists(jp):
    out = json.load(open(jp))
base = SS.base_scenes(N, 3, 15.0, SEED)
M = ['FT network only', 'consistent NOMP', 'ML (all starts)', 'ML from FT network seeds', 'ML all starts + FT seeds']
for bits in [1, 2]:
    name = f'b{bits}_snr15'
    if name in out:
        print('skip', name, flush=True); continue
    Y = SS.generate(base, 'M5', bits); t0 = time.time(); S = {m: [] for m in M}
    for c0 in range(0, N, CH):
        f = f'{PCACHE}/s9_{name}_{c0}.npz'
        sl = slice(c0, c0 + CH); bc = {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim and len(v) == N else v) for k, v in base.items()}
        if os.path.exists(f):
            z = np.load(f); Sc = {m: z[m] for m in M}
        else:
            y = Y[sl]; P = {}
            hf, _ = A.cnn_forward(ft, y); pl = A.parabolic_heat_decode(hf, 3); P['FT network only'] = pl
            Un = np.array([np.pi * np.cos(p[0][:3]) for p in pl]); Vn = np.array([np.pi * np.cos(p[1][:3]) for p in pl])
            o = ml_quant(y, bits, bc['snr'], 3, kappa=1.0, extra={'FT network seeds': (Un, Vn)})
            P['ML (all starts)'] = C.uv_to_angles(o['ML (all starts)'][0], o['ML (all starts)'][1])
            P['ML from FT network seeds'] = C.uv_to_angles(o['ML from FT network seeds'][0], o['ML from FT network seeds'][1])
            P['consistent NOMP'] = C.uv_to_angles(*A.consistent_nomp(y, bits, 3, 8)[:2])
            # combined: lowest exact NLL over the classical-start winner and the FT-start result
            rf = o['ML from FT network seeds']; use = rf[2] < o['ML (all starts)'][2]
            Uc = np.where(use[:, None], rf[0], o['ML (all starts)'][0]); Vc = np.where(use[:, None], rf[1], o['ML (all starts)'][1])
            P['ML all starts + FT seeds'] = C.uv_to_angles(Uc, Vc)
            Sc = {m: A.stats(P[m], bc) for m in M}; np.savez(f, **Sc)
            print(f'  {name} chunk {c0} done [{time.time() - t0:.0f}s]', flush=True)
        for m in M:
            S[m].append(Sc[m])
    S = {m: np.concatenate(v) for m, v in S.items()}
    pdm, ci_, D = boot(S, base['snr'], [('ML (all starts)', 'FT network only'), ('ML from FT network seeds', 'ML (all starts)'),
                                         ('ML all starts + FT seeds', 'ML (all starts)'), ('FT network only', 'consistent NOMP')])
    out[name] = dict(pd=pdm, ci=ci_, diffs=D, seed=SEED, n=N)
    print(f'=== {name} (seed {SEED}, n={N})', flush=True)
    for m in M:
        print(f'   {m:28s} Pd {pdm[m]:.4f} [{ci_[m][0]:.4f},{ci_[m][1]:.4f}]', flush=True)
    for k, d in D.items():
        print(f'   {k:55s} {fmt(d)}', flush=True)
    json.dump(out, open(jp, 'w'), indent=1, default=float)
print('done', flush=True)
