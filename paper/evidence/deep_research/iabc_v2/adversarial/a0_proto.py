"""PROTOTYPE / sanity run of the classical comparators (seeds 9340+k, k = SNR index, 40 scenes per SNR, L=3; cells clean, ph8, c30_6,
ph45). Purpose: timing, crash check, and that nothing diverges; NO constant is chosen here (all constants fixed in a_lib.py). Not test data.
stdout -> a0_proto.log"""
import time
from a_lib import *

N = 40
for name, pdeg, gdb in [('clean', 0, 0), ('ph8', 8, 0), ('c30_6', 30, 6), ('ph45', 45, 0)]:
    rows = {}
    for k, snr in enumerate(SNRS):
        d = make_scenes(9340 + k, N, 3, snr, pdeg, gdb); Ht = to_ant(d['Y']); T = {}
        t = time.time(); E = {'none': np.zeros((N, 32, 2))}
        E['SC tuned'] = sc_iter(Ht, 3, 4)[0]; T['SC tuned'] = time.time() - t; t = time.time()
        E['SC tuned 10it'] = sc_iter(Ht, 3, 10)[0]; T['SC tuned 10it'] = time.time() - t; t = time.time()
        E['SC redetect'] = sc_redetect(Ht, 3)[0]; T['SC redetect'] = time.time() - t; t = time.time()
        e, _, eml, _ = eb_ml(Ht, 3); E['EB-ML'] = e; E['ML unshrunk'] = eml; T['EB-ML'] = time.time() - t; t = time.time()
        E['v2b'] = v2b_est(Ht, 3); T['v2b'] = time.time() - t
        for m, e in E.items():
            ph, g, mp, mg = err_rms(d['tgt'], e); rows.setdefault(m, []).append((np.rad2deg(np.sqrt(mp.mean())), DB * np.sqrt(mg.mean())))
        if k == 7:
            print(f'{name}: ms/scene ' + ' '.join(f'{m} {1000 * v / N:.1f}' for m, v in T.items()), flush=True)
    print(f'{name} RMS phase deg / gain dB by SNR {SNRS}')
    for m, r in rows.items():
        print(f'  {m:14s} ' + ' '.join(f'{a:6.2f}/{b:4.2f}' for a, b in r), flush=True)
print('DONE')
