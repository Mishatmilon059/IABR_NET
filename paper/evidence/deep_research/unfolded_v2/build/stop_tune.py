"""Tune the threshold multiplier of the classical NOMP residual-threshold stopping baselines (true sigma / blindly estimated sigma)
on the VALIDATION L-mixed bank (seed 71_200_200, L 1..6, SNR -10..25, 800 scenes): grid of multipliers, maximise pooled hit-F1."""
import os
os.environ.setdefault('THREADS', '6')
from ev import *
d = load('valLmix'); H = SS.to_ant(d['Y']); out = {}
for nm, s2 in [('NOMP stop true sigma', 2 * sigma_of(d['snr']) ** 2), ('NOMP stop est sigma', sig2_est(H))]:
    best = None
    for mult in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
        P = []
        for c0 in range(0, len(H), 400):
            P += nomp_stop(H[c0:c0 + 400], s2[c0:c0 + 400], mult=mult)
        f = f1_stats(P, d).sum(0); f1 = 2 * f[0] / (f[1] + f[2]); fa = (f[2] - f[0]) / max(f[2], 1)
        print(f'{nm:22s} mult {mult:4.2f}: hit-F1 {f1:.4f} false-alarm rate {fa:.4f} mean n_est {f[2] / len(P):.2f} (true {f[1] / len(P):.2f})', flush=True)
        if best is None or f1 > best[1]:
            best = (mult, f1)
    out[nm] = best[0]
json.dump(out, open(f'{HERE}/stop_tune.json', 'w'), indent=1); print('chosen', out)
