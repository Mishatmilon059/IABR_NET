"""Tune the ONE constant of the quantisation-aware ML estimator (noise-std inflation kappa) on the TUNING banks only
(seeds 8730 (0 dB) / 8745 (15 dB), 300 scenes, b = 1, 2; never used for testing). kappa grid {0.7, 1, 1.5, 2.5}; criterion = mean Pd of
'ML (all starts)' over the four cells (ties -> kappa closest to 1). Resumable: one json per (kappa, cell) in tune_parts/.
usage: python s2_tune.py   (stdout -> s2_tune.log; writes tune.json)"""
from advc import *

KAPPAS = [1.0, 1.5, 2.5, 0.7]
N = 300
os.makedirs(f'{HERE}/tune_parts', exist_ok=True)
res = {}
for snr in [15, 0]:
    base = SS.base_scenes(N, 3, float(snr), TUNE_SEEDS[snr])
    for bits in [1, 2]:
        Y = SS.generate(base, 'M5', bits)
        for kap in KAPPAS:
            p = f'{HERE}/tune_parts/k{kap}_b{bits}_snr{snr}.json'
            if os.path.exists(p):
                res[(kap, bits, snr)] = json.load(open(p)); print('skip', p, flush=True); continue
            t0 = time.time(); P = {}
            for s0 in range(0, N, 100):
                sl = slice(s0, s0 + 100)
                o = ml_quant(Y[sl], bits, base['snr'][sl], kappa=kap)
                for k, v in o.items():
                    if not k.startswith('_'):
                        P.setdefault(k, []).extend(A.as_list(C.uv_to_angles(v[0], v[1])))
            r = {k: A.pd(A.stats(v, base)) for k, v in P.items()}
            r['sec'] = time.time() - t0
            json.dump(r, open(p, 'w'), indent=1); res[(kap, bits, snr)] = r
            print(f'kappa {kap} b{bits} {snr} dB: ' + ', '.join(f'{k} {v:.4f}' for k, v in r.items() if k != 'sec') + f'  [{r["sec"]:.0f}s]', flush=True)
score = {kap: float(np.mean([res[(kap, b_, s_)]['ML (all starts)'] for b_ in [1, 2] for s_ in [15, 0]])) for kap in KAPPAS}
best = max(KAPPAS, key=lambda k: (round(score[k], 4), -abs(np.log(k))))
print('mean Pd of ML (all starts) over the 4 tuning cells:', {k: round(v, 4) for k, v in score.items()}, '-> kappa =', best, flush=True)
json.dump(dict(score={str(k): v for k, v in score.items()}, kappa=best, cells={f'{k[0]}_b{k[1]}_snr{k[2]}': v for k, v in res.items()}),
          open(f'{HERE}/tune.json', 'w'), indent=1)
print('done', flush=True)
