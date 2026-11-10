"""Step 7: CPU latency, single thread (OMP/OPENBLAS/MKL threads = 1 set before numpy/torch import, torch.set_num_threads(1)).
(a) NOMP greedy (nomp_mine.nomp, os=4, 3 rounds; also the notebook nomp_greedy for reference) on the raw channel Ht,
(b) CNN forward (IABRv2, final.pt) + heat-seeded decode (heat_main recipe: decode(Hc, L, heat, 'heat', SC=(1.0,5.0))),
(c) CNN forward alone.
Batch 1 and batch 256, official-bank scenes (L=3; scenes drawn with np.random.default_rng(0)); 5 warm-up calls, then the median of 60 timed runs
(batch 1: 60 distinct scenes; batch 256: 60 runs cycling over 4 distinct batches). Per-scene time for batch 256 = run time / 256.
Also checks that the CPU forward reproduces the cached heat / Hc (net__official.npz).
usage: python latency.py   (stdout -> latency.log)"""
import os, sys, json, time, platform
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
os.environ['IABR2_SUITE'] = 'D:/ai_ml_project'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2')
import numpy as np, scipy, torch, torch.nn as nn, torch.nn.functional as TF
torch.set_num_threads(1); torch.set_num_interop_threads(1)
from vcommon import *
import nomp_mine as NM
import iabr2_sim as sim
sys.path.insert(0, NEWR)
import common as C

src = open(f'{ROOT}/notebook_src.py', encoding='utf-8').read().split('\n')
gm = dict(np=np, torch=torch, nn=nn, TF=TF, W_IDEAL=sim.W_IDEAL, F_IDEAL=sim.F_IDEAL, P=16, Q=16); exec('\n'.join(src[419:522]), gm)
cnn = gm['IABRv2'](32, 10, 32); cnn.load_state_dict(torch.load(f'{ROOT}/outputs/checkpoints/IABRv2/final.pt', map_location='cpu')); cnn.eval()
nparam = sum(p.numel() for p in cnn.parameters())

fb = np.load(FROZEN)
rng = np.random.default_rng(0); order = rng.permutation(len(fb['data']))
NEED = 60 + 4 * 256
ids = np.sort(order[:NEED])
Y16 = sim.downsample16(fb['data'][ids]); Yc = Y16[..., 0].astype(np.float64) + 1j * Y16[..., 1]
X = np.ascontiguousarray(sim.to_ri(Yc).transpose(0, 3, 1, 2)).astype(np.float32)
Ht = np.load(f'{CACHE}/net__official.npz')
heat_c, Hc_c, Ht_c = Ht['heat'][ids], Ht['Hc'][ids], Ht['Ht'][ids]
L = 3

with torch.no_grad():
    o = cnn(torch.from_numpy(X[:256]))
dh = float(np.abs(torch.sigmoid(o['heat']).numpy() - heat_c[:256]).max()); dc = float(np.abs(o['Hc'].numpy() - Hc_c[:256]).max())
print(f'IABRv2 params {nparam:,}; CPU forward vs cached (256 scenes): max |heat diff| {dh:.2e}, max |Hc diff| {dc:.2e}, max|Hc| {np.abs(Hc_c[:256]).max():.2f}', flush=True)


def f_nomp(i0, B):
    U, V = NM.nomp(Ht_c[i0:i0 + B].astype(np.complex128), L, 4, 3); return NM.to_angles(U, V)


def f_nomp_nb(i0, B):
    U, V = C.nomp_greedy(Ht_c[i0:i0 + B].astype(np.complex128), L, 4, 3); return C.uv_to_angles(U, V)


def f_cnn(i0, B):
    with torch.no_grad():
        o = cnn(torch.from_numpy(X[i0:i0 + B]))
        return torch.sigmoid(o['heat']).numpy(), o['Hc'].numpy()


def f_cnn_heat(i0, B):
    heat, Hc = f_cnn(i0, B)
    return C.decode(Hc.astype(np.complex128), L, heat.astype(np.float64), 'heat', C.SC)


res = dict(platform=platform.platform(), processor=platform.processor(), python=platform.python_version(), numpy=np.__version__,
           torch=torch.__version__, torch_threads=torch.get_num_threads(), cnn_params=nparam, cpu_vs_cache=dict(heat=dh, Hc=dc), timings={})
for nm, f in [('(a) NOMP greedy, mine', f_nomp), ('(a) NOMP greedy, notebook nomp_greedy', f_nomp_nb), ('(c) CNN forward only', f_cnn),
              ('(b) CNN forward + heat-seeded decode', f_cnn_heat)]:
    for B in [1, 256]:
        starts = [i for i in range(60)] if B == 1 else [60 + 256 * (r % 4) for r in range(60)]
        for i in starts[:5]:
            f(i, B)
        ts = []
        for i in starts:
            t0 = time.perf_counter(); f(i, B); ts.append(time.perf_counter() - t0)
        ts = np.array(ts) * 1e3
        r = dict(batch=B, runs=len(ts), median_ms_per_call=float(np.median(ts)), p10_ms=float(np.percentile(ts, 10)), p90_ms=float(np.percentile(ts, 90)),
                 median_ms_per_scene=float(np.median(ts) / B))
        res['timings'][f'{nm} | batch {B}'] = r
        print(f'{nm:42s} batch {B:3d}: median {r["median_ms_per_call"]:9.2f} ms/call (p10 {r["p10_ms"]:.2f}, p90 {r["p90_ms"]:.2f}, {len(ts)} runs) '
              f'-> {r["median_ms_per_scene"]:.3f} ms/scene', flush=True)
json.dump(res, open(f'{HERE}/latency.json', 'w'), indent=1)
