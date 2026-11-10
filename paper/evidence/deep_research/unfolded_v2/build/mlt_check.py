"""Check that mlt.py (torch) reproduces the numpy quantised ML of finetune/adversarial (advc.ml_quant, mlq) on validation scenes
(G0 cells b1/15 and b2/0, seeds 71_200_101 / 71_200_104, first N scenes). Also times both."""
import os
os.environ.setdefault('THREADS', '2')
from mlt import *
sys.path.insert(0, f'{DR}/finetune/adversarial')
import advc  # noqa
torch.set_num_threads(int(TH))
N = int(os.environ.get('N', 60))
for bits, snr, seed in [(1, 15, 71_200_101), (2, 0, 71_200_104)]:
    base = SS.base_scenes(N, 3, float(snr), seed); Y = SS.generate(base, 'M5', bits)
    t0 = time.time(); o = advc.ml_quant(Y, bits, base['snr'], 3, kappa=1.0); t1 = time.time()
    Un, Vn, Fn, _ = o['ML (all starts)']
    t2 = time.time(); r = ml_family(Y, bits, base['snr'], 3); t3 = time.time()
    Ut, Vt = r['ML']
    same = (np.abs(np.angle(np.exp(1j * (Un - Ut)))).max(1) < 1e-3) & (np.abs(np.angle(np.exp(1j * (Vn - Vt)))).max(1) < 1e-3)
    print(f'b{bits}_snr{snr} n={N}: numpy ML Pd {pd(stats(C.uv_to_angles(Un, Vn), base)):.4f} | torch ML Pd {pd(stats(C.uv_to_angles(Ut, Vt), base)):.4f} | '
          f'same solution (1e-3 rad) in {same.mean():.3f} of scenes | numpy {t1 - t0:.1f}s, torch family (6 variants) {t3 - t2:.1f}s', flush=True)
    for k_, (u, v) in r.items():
        print(f'   {k_:16s} Pd {pd(stats(C.uv_to_angles(u, v), base)):.4f}', flush=True)
