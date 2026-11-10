"""Step 4: CPU latency, single thread (OMP/OPENBLAS/MKL/NUMEXPR = 1 set before numpy/torch import via NDTHREADS=1, torch.set_num_threads(1)).
Same protocol as verify/latency.py: official-bank scenes (L=3) drawn with np.random.default_rng(0); 5 warm-up calls then the median of
60 timed runs at batch 1 (60 distinct scenes) and 40 timed runs at batch 256 (cycling over 4 distinct batches); per-scene = call / batch.
Pipelines: NOMP greedy (verify/nomp_mine.nomp and notebook nomp_greedy) on Ht; CNN forward only; CNN + classical heat-seeded decoder
(decode(Hc, L, heat, 'heat', SC)); CNN + all-neural decoder (heat_peaks -> StagedSC lurd.pt) SC on / SC off; learned decoder alone (SC on, from cached heat/Hc).
Also checks that the CPU all-neural pipeline reproduces the s2 predictions on 256 scenes. Resumable: s4_latency.json is rewritten after each entry.
usage: python -u s4_latency.py > s4_latency.log"""
import os, sys, json, time, platform
os.environ['NDTHREADS'] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ndcommon import *
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError as e:
    print('interop threads not set:', e)
sys.path.insert(0, VER)
import nomp_mine as NM
from vcommon import FROZEN
sim = E5.sim; cnn = E5.cnn.cpu().eval()
DRC = 'C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/nn_decoder'
OUTJ = f'{HERE}/s4_latency.json'
model = load_learned()
fb = np.load(FROZEN); rng = np.random.default_rng(0); order = rng.permutation(len(fb['data']))
ids = np.sort(order[:60 + 4 * 256])
Y16 = sim.downsample16(fb['data'][ids]); Yc = Y16[..., 0].astype(np.float64) + 1j * Y16[..., 1]
X = np.ascontiguousarray(sim.to_ri(Yc).transpose(0, 3, 1, 2)).astype(np.float32)
z = np.load(f'{CACHE}/net__official.npz'); heat_c, Hc_c, Ht_c = z['heat'][ids], z['Hc'][ids], z['Ht'][ids]
L = 3


def f_cnn(i0, B):
    with torch.no_grad():
        o = cnn(torch.from_numpy(X[i0:i0 + B]))
        return torch.sigmoid(o['heat']).numpy(), o['Hc'].numpy()


def f_nomp(i0, B):
    U, V = NM.nomp(Ht_c[i0:i0 + B].astype(np.complex128), L, 4, 3); return NM.to_angles(U, V)


def f_nomp_nb(i0, B):
    U, V = C.nomp_greedy(Ht_c[i0:i0 + B].astype(np.complex128), L, 4, 3); return C.uv_to_angles(U, V)


def f_cnn_heat(i0, B):
    heat, Hc = f_cnn(i0, B); return C.decode(Hc.astype(np.complex128), L, heat.astype(np.float64), 'heat', C.SC)


def f_cnn_an(i0, B, sc=True):
    heat, Hc = f_cnn(i0, B); return learned_decode(model, Hc, heat, L, sc, bs=256)


def f_an_only(i0, B):
    return learned_decode(model, Hc_c[i0:i0 + B], heat_c[i0:i0 + B], L, True, bs=256)


# reproduction check vs s2 predictions (official bank, scenes ids[60:316])
a, b = f_cnn_an(60, 256); p = load_predfile(f'{DRC}/official__allneural_sc.npz'); jj = ids[60:316]
dd = np.maximum(np.abs(a - p['psi'][jj]), np.abs(b - p['phi'][jj]))
print(f'CPU single-thread all-neural (fresh CNN forward) vs s2 predictions, 256 scenes: median |d| {np.degrees(np.median(dd)):.2e} deg, '
      f'scenes with max |d| < 0.01 deg: {np.mean(np.degrees(dd.max(1)) < 0.01):.4f}, max {np.degrees(dd.max()):.3f} deg', flush=True)
res = json.load(open(OUTJ)) if os.path.exists(OUTJ) else dict(timings={})
res.update(platform=platform.platform(), processor=platform.processor(), python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
           torch_threads=torch.get_num_threads(), cnn_params=sum(p.numel() for p in cnn.parameters()), decoder_params=sum(p.numel() for p in model.parameters()))
for nm, f in [('NOMP greedy, verify nomp_mine', f_nomp), ('NOMP greedy, notebook nomp_greedy', f_nomp_nb), ('CNN forward only', f_cnn),
              ('CNN + classical heat-seeded decoder (IABR-Net classical)', f_cnn_heat), ('CNN + learned decoder SC on (all-neural)', f_cnn_an),
              ('CNN + learned decoder SC off (all-neural)', lambda i, B: f_cnn_an(i, B, False)), ('learned decoder alone, SC on', f_an_only)]:
    for B in [1, 256]:
        key = f'{nm} | batch {B}'
        if key in res['timings']:
            r = res['timings'][key]
        else:
            starts = list(range(60)) if B == 1 else [60 + 256 * (k % 4) for k in range(40)]
            for i in starts[:5]:
                f(i, B)
            ts = []
            for i in starts:
                t0 = time.perf_counter(); f(i, B); ts.append(time.perf_counter() - t0)
            ts = np.array(ts) * 1e3
            r = dict(batch=B, runs=len(ts), median_ms_per_call=float(np.median(ts)), p10_ms=float(np.percentile(ts, 10)), p90_ms=float(np.percentile(ts, 90)),
                     median_ms_per_scene=float(np.median(ts) / B))
            res['timings'][key] = r; json.dump(res, open(OUTJ, 'w'), indent=1)
        print(f'{nm:58s} batch {B:3d}: median {r["median_ms_per_call"]:9.2f} ms/call (p10 {r["p10_ms"]:.2f}, p90 {r["p90_ms"]:.2f}, {r["runs"]} runs) '
              f'-> {r["median_ms_per_scene"]:.3f} ms/scene', flush=True)
print('done', flush=True)
