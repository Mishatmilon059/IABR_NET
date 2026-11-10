"""Step 1b: is the M5 'oracle-seeded' gap recoverable search loss, or is the LS objective biased by the quantiser?
On the stress M5 banks (seeds 5000/5015, 500 scenes):
  oracle rc=3 (the stress-test reference), oracle with 10 / 30 cyclic rounds (does the oracle drift away from the truth when the decoder
  is run to convergence?), median beam-cell distance of the final estimates from the truth;
  'ideal heatmap' seeds = true (u,v) rounded to the 32x32 heat grid (the best a perfect 32x32 heatmap could seed), and to the 64x64
  grid, refined exactly like decode(...,'heat') (Newton + 3 cyclic rounds); and the same seeds with no refinement.
  quick timing / Pd of the classical candidates (FFT-peak seeds, consistent NOMP, multi-start) on b = 1, 2.
usage: python diag2.py > diag2.log"""
from acommon import *

t00 = time.time()


def cdist(U, V, tu, tv):
    """per-path cell distance to its own truth (estimates matched by index of nearest truth): median over all paths"""
    du = np.angle(np.exp(1j * (U[:, :, None] - tu[:, None, :]))); dv = np.angle(np.exp(1j * (V[:, :, None] - tv[:, None, :])))
    return float(np.median((np.maximum(np.abs(du), np.abs(dv)) * 16 / (2 * np.pi)).min(1)))


def gridround(t, G, sign):
    return sign * 2 * np.pi * (np.round(sign * t * G / (2 * np.pi)) % G) / G


res = {}
for snr in [15, 0]:
    for bits in [None, 4, 2, 1]:
        b, Y = m5_bank(500, snr, STRESS_SEEDS[snr], bits); H = to_ant(Y); lab = 'clean' if bits is None else f'b{bits}'
        tu = np.pi * np.cos(b['psi'][:, :3]); tv = np.pi * np.cos(b['phi'][:, :3])
        rec = {}
        t0 = time.time(); U, V = C.nomp_greedy(H, 3, 4, 3); rec['NOMP'] = (pd(stats(C.uv_to_angles(U, V), b)), cdist(U, V, tu, tv), time.time() - t0)
        for rc in [3, 10, 30]:
            Uo, Vo = C.cyclic(H, tu, tv, rc); rec[f'oracle rc{rc}'] = (pd(stats(C.uv_to_angles(Uo, Vo), b)), cdist(Uo, Vo, tu, tv))
        rec['truth itself'] = (pd(stats(C.uv_to_angles(tu, tv), b)), 0.0)
        for G in [32, 64]:
            U0 = gridround(tu, G, -1); V0 = gridround(tv, G, 1)
            rec[f'ideal {G}-grid seeds, no refinement'] = (pd(stats(C.uv_to_angles(U0, V0), b)), cdist(U0, V0, tu, tv))
            Ur, Vr = seeds_refine(H, U0, V0, 3); rec[f'ideal {G}-grid seeds + Newton/cyclic'] = (pd(stats(C.uv_to_angles(Ur, Vr), b)), cdist(Ur, Vr, tu, tv))
        if bits in (1, 2):
            t0 = time.time(); Uh, Vh = C.decode(H, 3, fft_heat(H), 'heat', None) and (None, None)
            ps = C.decode(H, 3, fft_heat(H), 'heat', None); rec['FFT-peak seeds (heat)'] = (pd(stats(ps, b)), None, time.time() - t0)
            t0 = time.time(); ps = C.decode(H, 3, fft_heat(H), 'dual', None); rec['FFT-peak dual'] = (pd(stats(ps, b)), None, time.time() - t0)
            t0 = time.time(); Uc, Vc, inc = consistent_nomp(Y, bits, 3, 8); rec['consistent NOMP (8 it)'] = (pd(stats(C.uv_to_angles(Uc, Vc), b)), cdist(Uc, Vc, tu, tv), time.time() - t0)
            t0 = time.time(); Um, Vm, k = multistart(H, 3, 6); rec['multi-start K6'] = (pd(stats(C.uv_to_angles(Um, Vm), b)), cdist(Um, Vm, tu, tv), time.time() - t0)
        res[f'{lab}_snr{snr}'] = rec
        print(f'--- {lab} SNR {snr}', flush=True)
        for k, v in rec.items():
            print(f'    {k:42s} Pd {v[0]:.4f}  median cell dist {v[1] if v[1] is None else round(v[1], 4)}' + (f'  {v[2]:.1f}s' if len(v) > 2 else ''), flush=True)
save_json(res, f'{HERE}/diag2.json')
print(f'total {time.time() - t00:.0f}s')
