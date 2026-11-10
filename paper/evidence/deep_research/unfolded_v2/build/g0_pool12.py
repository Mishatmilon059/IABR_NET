"""Supplement to g0_gate.py (same validation scenes): classical hypothesis pool with the final polish (12 LM layers, classical adaptive
damping) instead of 4 fixed-damping layers. Pd only (compare with g0_gate.log ML numbers on the same scenes)."""
import os
os.environ.setdefault('THREADS', '6')
from ucore import *
for bits, snr, seed in [(1, 15, 71_200_101), (2, 15, 71_200_102), (1, 0, 71_200_103), (2, 0, 71_200_104)]:
    base = SS.base_scenes(200, 3, float(snr), seed); Y = SS.generate(base, 'M5', bits)
    m = UNOMP('A'); m.cfg.update(classical=True, polish=True, sc=False); m.eval(); t0 = time.time()
    with torch.no_grad():
        oo = m(prep_batch(Y, np.full(200, bits), base['snr']), 3)
    Sh = {h: stats(C.uv_to_angles(oo['res'][h]['U'].numpy(), oo['res'][h]['V'].numpy()), base) for h in oo['hyps']}
    hits = np.stack([Sh[h][:, 0] for h in oo['hyps']], 1); best = hits.argmax(1)
    Sor = np.stack([Sh[h] for h in oo['hyps']], 1)[np.arange(200), best]
    print(f'b{bits}_snr{snr}: pool(12 LM) lowest-NLL Pd {pd(stats(C.uv_to_angles(oo["U"].numpy(), oo["V"].numpy()), base)):.4f} oracle {pd(Sor):.4f} '
          + ' '.join(f'{h}:{pd(Sh[h]):.3f}' for h in oo['hyps']) + f' [{time.time() - t0:.1f}s]', flush=True)
