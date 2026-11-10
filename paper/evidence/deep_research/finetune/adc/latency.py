"""Latency per scene, single thread (OMP/OPENBLAS/MKL = 1 before import, torch 1 thread), batch 1, L = 3, on the first 40 scenes of the
stress M5 b=1 SNR 15 bank; 3 warm-ups, median over the 40 scenes. CNN time = forward of IABRv2 (identical architecture for the original and
the fine-tuned weights). usage: python latency.py > latency.log  (writes latency.json)"""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
from acommon import *
import platform
torch.set_num_threads(1)
ft = load_cnn(FT_CKPT)
b, Y = m5_bank(500, 15, STRESS_SEEDS[15], 1)
Y = Y[:40]


def cnn1(y):
    return cnn_forward(ft, y, bs=1)


F = {
    'NOMP': lambda y: C.nomp_greedy(to_ant(y), 3, 4, 3),
    'NOMP+SC tuned': lambda y: selfcal_tuned(to_ant(y), *C.nomp_greedy(to_ant(y), 3, 4, 3)),
    'classical dual': lambda y: C.decode(to_ant(y), 3, fft_heat(to_ant(y)), 'dual', None),
    'NOMP os8 rc6': lambda y: C.nomp_greedy(to_ant(y), 3, 8, 6),
    'multi-start K6': lambda y: multistart(to_ant(y), 3, 6),
    'consistent NOMP': lambda y: consistent_nomp(y, 1, 3, 8),
    'CNN forward only': lambda y: cnn1(y),
    'FT heat+SC (CNN + decode)': lambda y: (lambda h, hc: C.decode(hc, 3, h, 'heat', SC))(*cnn1(y)),
    'FT dual+SC (CNN + decode)': lambda y: (lambda h, hc: C.decode(hc, 3, h, 'dual', SC))(*cnn1(y)),
    'FT network only (CNN + parabolic)': lambda y: parabolic_heat_decode(cnn1(y)[0], 3),
}
res = dict(platform=platform.platform(), numpy=np.__version__, torch=torch.__version__, n=len(Y), ms={})
for nm, f in F.items():
    for i in range(3):
        f(Y[i:i + 1])
    t = []
    for i in range(len(Y)):
        t0 = time.perf_counter(); f(Y[i:i + 1]); t.append(time.perf_counter() - t0)
    res['ms'][nm] = 1000 * float(np.median(t))
    print(f'{nm:36s} {res["ms"][nm]:8.2f} ms/scene (median of {len(Y)})', flush=True)
save_json(res, f'{HERE}/latency.json')
