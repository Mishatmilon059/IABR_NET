"""Re-run ONLY the selected (pilot-chosen) step of each nML/GD variant on the test set, and count non-finite iterates.
Purpose: the final nML check printed overflow/NaN RuntimeWarnings; NaN rows would be counted as errors by ber(). Same seeds as lordnet_nml_check.py."""
import os, math, json
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
from scipy.special import log_ndtr
m, n, B, T, ITERS = 128, 16, 512, 2048, 700
K = 6
HERE = os.path.dirname(os.path.abspath(__file__))
sel = json.load(open(os.path.join(HERE, 'lordnet_nml_check.json')))
def data(H, N, rng):
    x = rng.choice([-1.0, 1.0], size=(N, n))
    y = x @ H.T + rng.standard_normal((N, m))
    return x, np.where(y >= 0, 1.0, -1.0)
def mills(z):
    return np.exp(-0.5 * z * z - 0.5 * math.log(2 * math.pi) - log_ndtr(z))
def run(H, R, step, radius, init):
    N = R.shape[0]
    if init == 'mr':
        g = R @ H; x = math.sqrt(n) * g / np.linalg.norm(g, axis=1, keepdims=True)
    else:
        x = np.zeros((N, n))
    rad2 = radius ** 2 * n
    for _ in range(ITERS):
        z = R * (x @ H.T)
        with np.errstate(all='ignore'):
            x = x + step * ((R * mills(z)) @ H)
        if math.isfinite(radius):
            nr = np.sum(x * x, 1, keepdims=True)
            x = np.where(nr > rad2, x * np.sqrt(rad2 / np.maximum(nr, 1e-300)), x)
    return x
out = {}
for snr_db in [4.0, 8.0]:
    snr = 10 ** (snr_db / 10); key = f'{snr_db:g}dB'
    tot = {'nML_radius1': [0, 0], 'GD_from_0': [0, 0]}  # [errors counting nonfinite rows as wrong, nonfinite rows]
    tot_clean = {'nML_radius1': [0, 0], 'GD_from_0': [0, 0]}  # [errors over finite rows, finite rows]
    nonfin_rows = {'nML_radius1': 0, 'GD_from_0': 0}
    for k in range(K):
        rng = np.random.default_rng(80_000_000 + 100 * int(snr_db) + k)
        H = rng.standard_normal((m, n)) * math.sqrt(snr / n)
        Xp, Rp = data(H, B, rng)
        Xt, Rt = data(H, T, np.random.default_rng(12_900 + 100 * int(snr_db) + k))
        per = sel[key]['per_channel'][k]
        for name, radius, init in [('nML_radius1', 1.0, 'mr'), ('GD_from_0', float('inf'), 'zero')]:
            s = per[name + '_step']
            with np.errstate(all='ignore'):
                xt = run(H, Rt, s, radius, init)
            bad = ~np.all(np.isfinite(xt), axis=1)
            nonfin_rows[name] += int(bad.sum())
            sign = np.where(xt >= 0, 1.0, -1.0)
            wrong = (sign != Xt)
            tot[name][0] += int(wrong.sum()); tot[name][1] += int(bad.sum())
            good = ~bad
            tot_clean[name][0] += int(wrong[good].sum()); tot_clean[name][1] += int(good.sum() * n)
            print(f'{key} ch {k} {name} step {s:g}: recomputed BER {wrong.mean():.5f} (logged {per[name]:.5f}); nonfinite rows {int(bad.sum())}/{T}', flush=True)
    out[key] = {'nonfinite_rows_total': nonfin_rows,
                'BER_all_bits': {a: tot[a][0] / (K * T * n) for a in tot},
                'BER_finite_rows_only': {a: tot_clean[a][0] / max(tot_clean[a][1], 1) for a in tot_clean}}
    print(key, json.dumps(out[key]), flush=True)
json.dump(out, open(os.path.join(HERE, 'lordnet_nml_nan_check.json'), 'w'), indent=1)
print('done')
