"""E8: batch-1 CPU latency of the network-free decoder (L=3). usage: python e8_latency.py <threads>
Threads are limited through OMP/MKL/OPENBLAS_NUM_THREADS set before numpy is imported (numpy BLAS calls only; np.fft and np.einsum loops are single-threaded anyway).
Scenes: 10 different official-bank scenes at SNR 15 dB (raw channel Ht); each timed call decodes ONE scene (batch = 1); median over 100 runs (10 scenes x 10 repeats)."""
import sys, os
th = sys.argv[1]
for v in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[v] = th
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
from common import *
import platform
try:
    from threadpoolctl import threadpool_info; tp = threadpool_info()
except Exception as e:
    tp = str(e)
b = load_bank('official'); Ht = np.load(f'{CACHE}/net__official.npz')['Ht']
ii = np.where(b['snr'] == 15)[0][:10]; scenes = [Ht[i:i + 1].astype(np.complex128) for i in ii]
def t_greedy(H): U, V = nomp_greedy(H, 3, 4, 3); return uv_to_angles(U, V)
def t_sc(H): U, V = nomp_greedy(H, 3, 4, 3); U, V = selfcal(H, U, V, 1.0, 5.0); return uv_to_angles(U, V)
res = dict(threads=int(th), platform=platform.platform(), python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, threadpool=tp)
for nm, f in [('greedy (os=4, rc=3)', t_greedy), ('greedy + selfcal (1.0, 5.0)', t_sc)]:
    for H in scenes[:3]: f(H)
    ts = []
    for rep in range(10):
        for H in scenes:
            t0 = time.perf_counter(); f(H); ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1e3; res[nm] = dict(median_ms=float(np.median(ts)), p5_ms=float(np.percentile(ts, 5)), p95_ms=float(np.percentile(ts, 95)), n_runs=len(ts))
    print(f'threads={th}  {nm:30s} median {np.median(ts):.2f} ms  (p5 {np.percentile(ts,5):.2f}, p95 {np.percentile(ts,95):.2f}, {len(ts)} runs)', flush=True)
json.dump(res, open(f'{OUT}/e8_latency_threads{th}.json', 'w'), indent=1, default=str)
