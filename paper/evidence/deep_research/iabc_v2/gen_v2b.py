"""Generate stage-2 (refiner) training data for IABC-v2b: decoder-in-the-loop features from fresh synthetic scenes.
Scenes: iabc2_model.make_train_batch (same wide distribution as v2a), chunk k <- default_rng([63000000, k]), 400 scenes per chunk,
NCH chunks; validation chunks <- default_rng([63999999, k]), 5 x 400. Per chunk: v2a (cache/v2a_best.pt) estimate e_A, then
v2common.hybrid_inputs grouped by L (NOMP + gate-open ALS on the e_A-corrected channel). Resumable: one npz per chunk in cache/v2b_data/."""
import os
from v2common import *
from iabc2_model import make_train_batch, load_v2, predict

OUT = os.environ.get('V2B_DATA', f'{V2CACHE}/v2b_data'); os.makedirs(OUT, exist_ok=True)
NCH = int(os.environ.get('NCH', 150))
v2a = load_v2(f'{V2CACHE}/v2a_best.pt')
jobs = [(f'val_{k}', [63999999, k]) for k in range(5)] + [(f'tr_{k}', [63000000, k]) for k in range(NCH)]
t00 = time.time()
for name, sd in jobs:
    fn = f'{OUT}/{name}.npz'
    if os.path.exists(fn):
        continue
    t0 = time.time(); b = make_train_batch(np.random.default_rng(sd), 400)
    Ht = (b['Hr'] + 1j * b['Hi']).astype(complex); eA = predict(v2a, Ht)
    tok = np.zeros((400, 32, 6), np.float32); glob = np.zeros((400, 3), np.float32)
    for L in np.unique(b['L']):
        g = np.where(b['L'] == L)[0]
        tok[g], glob[g], _ = hybrid_inputs(Ht[g], int(L), eA[g])
    np.savez(fn, tok=tok, glob=glob, eA=eA.astype(np.float32), tgt=b['tgt'], snr=b['snr'], L=b['L'], pd=b['pd'], gd=b['gd'])
    print(f'{name}: {time.time() - t0:.1f}s [{time.time() - t00:.0f}s total]', flush=True)
print('DONE', flush=True)
