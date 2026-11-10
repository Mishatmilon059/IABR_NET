"""Latency per scene, ONE thread (OMP/OPENBLAS/MKL = 1 before import, torch 1 thread), batch 1, L = 3, first 30 scenes of the test bank
b = 1, 15 dB (seed 8715); 2 warm-ups, median. Same protocol as finetune/adc/latency.py.  usage: python s6_latency.py > s6_latency.log"""
import os
os.environ['THREADS'] = '1'
from advc import *
torch.set_num_threads(1)
KAPPA = json.load(open(f'{HERE}/tune.json'))['kappa']
ft = A.load_cnn(A.FT_CKPT)
base = SS.base_scenes(30, 3, 15.0, TEST_SEEDS[15]); Y = SS.generate(base, 'M5', 1); snr = base['snr']


def ml_single(y, s):
    lo, hi = mlq.cells_ri(y, 1, A.adc_cells); sg = (KAPPA * sigma_of(s))[:, None]
    U, V, _ = A.consistent_nomp(y, 1, 3, 8); X = mlq.init_gains(MO, U, V, y, lo, hi, sg)
    return mlq.lm(MO, U, V, X, lo, hi, sg)


F = {
    'NOMP': lambda y, s: C.nomp_greedy(A.to_ant(y), 3, 4, 3),
    'consistent NOMP': lambda y, s: A.consistent_nomp(y, 1, 3, 8),
    'ML from consistent NOMP (1 start)': ml_single,
    'ML (all starts: 13 + EM)': lambda y, s: ml_quant(y, 1, s, 3, kappa=KAPPA),
    'FT network only (CNN + parabolic)': lambda y, s: A.parabolic_heat_decode(A.cnn_forward(ft, y, bs=1)[0], 3),
}
res = {}
for nm, f in F.items():
    for i in range(2):
        f(Y[i:i + 1], snr[i:i + 1])
    t = []
    for i in range(len(Y)):
        t0 = time.perf_counter(); f(Y[i:i + 1], snr[i:i + 1]); t.append(time.perf_counter() - t0)
    res[nm] = 1000 * float(np.median(t))
    print(f'{nm:36s} {res[nm]:9.1f} ms/scene (median of {len(Y)}, 1 thread, batch 1)', flush=True)
json.dump(res, open(f'{HERE}/s6_latency.json', 'w'), indent=1)
