"""Latency spot-check (CPU, 1 thread, batch 1, median of 40 runs after 5 warm-ups; official scenes 0..44, cached heat/Hc/Ht; decoders only, no CNN):
learned decoder SC on / SC off (StagedSC + lurd.pt, complex64 torch), the SAME unrolled torch structure with the MLP replaced by the clipped Newton
step (complex64; isolates implementation cost from the MLP), classical heat decoder (numpy, decode 'heat' SC), NOMP notebook (decode 'greedy').
usage: python -u a4_latency.py > a4_latency.log"""
import os, sys, time, math
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
LD = 'D:/ai_ml_project/paper/evidence/learned_decoder'
sys.path.insert(0, LD)
import numpy as np, torch, torch.nn as nn
import exp5 as E5
torch.set_num_threads(1)
g = E5.g
z = np.load('D:/ai_ml_project/IABR_v2_extracted/IABR_v2/outputs/cache/net__official.npz')
heat, Hc, Ht = z['heat'][:45].astype(np.float64), z['Hc'][:45], z['Ht'][:45]
m = E5.StagedSC(E5.RefinerV3(), pre=6, rounds=3, inner=2); m.load_state_dict(torch.load(f'{LD}/lurd.pt', map_location='cpu')); m.eval()


class NewtonRef(nn.Module):
    def forward(self, r, u, v):
        S, gu, gv, huu, hvv, huv = E5.quantities(r, u, v)
        du, dv, _ = E5.newton_step(gu, gv, huu, hvv, huv, math.pi / 32); return torch.stack([du, dv], 1)


nt = E5.StagedSC(NewtonRef(), pre=6, rounds=3, inner=2); nt.eval()


def torch_dec(model, sc):
    def f(i):
        model.use_sc = sc; U0, V0 = g['heat_peaks'](heat[i:i + 1], 3)
        with torch.no_grad():
            return model(torch.from_numpy(Hc[i:i + 1].astype(np.complex64)), torch.tensor(U0, dtype=torch.float32), torch.tensor(V0, dtype=torch.float32))
    return f


fns = [('learned decoder SC on', torch_dec(m, True)), ('learned decoder SC off', torch_dec(m, False)),
       ('Newton step in the same torch structure, SC on', torch_dec(nt, True)),
       ('classical heat decoder (numpy), SC on', lambda i: g['decode'](Hc[i:i + 1].astype(np.complex128), 3, heat[i:i + 1], 'heat', (1.0, 5.0))),
       ('NOMP notebook (numpy), no SC', lambda i: g['decode'](Ht[i:i + 1].astype(np.complex128), 3, None, 'greedy', None))]
for name, f in fns:
    for i in range(5):
        f(i)
    ts = []
    for i in range(5, 45):
        t0 = time.perf_counter(); f(i); ts.append((time.perf_counter() - t0) * 1e3)
    print(f'{name:50s} batch 1: median {np.median(ts):7.2f} ms (p10 {np.percentile(ts, 10):.2f}, p90 {np.percentile(ts, 90):.2f}, 40 runs)', flush=True)
print('done', flush=True)
