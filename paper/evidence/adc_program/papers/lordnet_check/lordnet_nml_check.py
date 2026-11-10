"""Why does LoRD-Net's nML baseline look so weak? A training-free check on the SAME toy channels as lordnet_mini.py.
LoRD-Net (arXiv 2102.02993v1, Sec. IV-A) compares with the nML detector of Choi, Mo, Heath (IEEE TCOM 2016, arXiv 1507.04452,
Algorithm 1): projected gradient ASCENT on sum_i log Phi(r_i h_i^T x) (real BPSK, sigma = 1) with the norm-ball constraint
||x||^2 <= K (K = n real symbols here) and a maximum-ratio initial point x0 = sqrt(K) G^T 1 / ||G^T 1||, G = Diag(r) H
(Choi eqs. (28)-(29), Algorithm 1 steps 5-7); LoRD-Net runs it for 700 iterations with a grid-searched step.
Here we run, with the TRUE H and no learning at all:
  nML(radius 1)     : Choi Algorithm 1 with ball radius sqrt(n) (radii 2, 4, 8, inf were tried in a first run, see
                      lordnet_nml_check_run1_partial.log: identical BER on channel 0, so they were dropped)
  GD from 0         : unconstrained gradient ascent from x0 = 0 (this is LoRD-Net's 'basic policy' with the true H; = the
                      'relaxedML_trueH_700' row of lordnet_mini.log, recomputed as a consistency check)
Step sizes for every variant: grid search on the B = 512 pilot symbols (as in the paper: 'step-size chosen using a grid search').
Same seeds as lordnet_mini.py: channel + pilots rng 80,000,000 + 100*SNR + k, test symbols rng 12,900 + 100*SNR + k.
Scaling the ball radius by c with the true H is EXACTLY the same detector as radius 1 with a surrogate channel c*H
(the objective depends on x only through Hx), so this also measures how much a pure 'surrogate scale' can buy.
"""
import os, math, json, time
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np
from scipy.special import log_ndtr

m, n, B, T, ITERS = 128, 16, 512, 2048, 700
SNRS = [4.0, 8.0]
K = 6
STEPS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
RADII = [1.0]  # a first run with radii {1,2,4,8,inf} gave identical BERs on channel 0 (radial clipping hardly changes the signs), so only Choi's radius 1 is kept


def data(H, N, rng):  # identical calls / order to lordnet_mini.py
    x = rng.choice([-1.0, 1.0], size=(N, n))
    y = x @ H.T + rng.standard_normal((N, m))
    return x, np.where(y >= 0, 1.0, -1.0)


def mills(z):  # phi(z) / Phi(z), stable
    return np.exp(-0.5 * z * z - 0.5 * math.log(2 * math.pi) - log_ndtr(z))


def run(H, R, step, radius, init):
    N = R.shape[0]
    if init == 'mr':
        g = R @ H  # rows: (G^T 1)^T per sample, G = Diag(r) H
        x = math.sqrt(n) * g / np.linalg.norm(g, axis=1, keepdims=True)
    else:
        x = np.zeros((N, n))
    rad2 = radius ** 2 * n
    for _ in range(ITERS):
        z = R * (x @ H.T)
        x = x + step * ((R * mills(z)) @ H)
        if math.isfinite(radius):
            nr = np.sum(x * x, 1, keepdims=True)
            x = np.where(nr > rad2, x * np.sqrt(rad2 / np.maximum(nr, 1e-300)), x)
    return x


def ber(x, X):
    return float(np.mean(np.where(x >= 0, 1.0, -1.0) != X))


variants = [(f'nML_radius{c:g}', c, 'mr') for c in RADII] + [('GD_from_0', float('inf'), 'zero')]
t0 = time.time(); out = {}
for snr_db in SNRS:
    snr = 10 ** (snr_db / 10); rows = []
    for k in range(K):
        rng = np.random.default_rng(80_000_000 + 100 * int(snr_db) + k)
        H = rng.standard_normal((m, n)) * math.sqrt(snr / n)
        Xp, Rp = data(H, B, rng)
        Xt, Rt = data(H, T, np.random.default_rng(12_900 + 100 * int(snr_db) + k))
        r = {}
        for name, c, init in variants:
            pil = [(ber(run(H, Rp, s, c, init), Xp), s) for s in STEPS]
            bp, s = min(pil)
            xt = run(H, Rt, s, c, init)
            r[name] = ber(xt, Xt); r[name + '_step'] = s
            nrm = np.linalg.norm(xt, axis=1) / math.sqrt(n)
            r[name + '_median_norm_over_sqrt_n'] = float(np.median(nrm)); r[name + '_frac_on_ball'] = float(np.mean(nrm > 0.999))
        rows.append(r)
        print(f'SNR {snr_db:g} dB ch {k}: ' + ' | '.join(f'{nm} {r[nm]:.5f} (step {r[nm + "_step"]:g}, median |x|/sqrt(n) {r[nm + "_median_norm_over_sqrt_n"]:.3f}, frac on ball {r[nm + "_frac_on_ball"]:.3f})' for nm, _, _ in variants) + f'  [{time.time() - t0:.0f}s]', flush=True)
    summ = {nm: float(np.mean([rr[nm] for rr in rows])) for nm, _, _ in variants}
    summ['bit_errors_total_GD_from_0'] = int(round(sum(rr['GD_from_0'] for rr in rows) * T * n))
    summ['bit_errors_total_nML_radius1'] = int(round(sum(rr['nML_radius1'] for rr in rows) * T * n))
    summ['channels_GD_from_0_better_than_nML_radius1'] = int(sum(rr['GD_from_0'] < rr['nML_radius1'] for rr in rows))
    out[f'{snr_db:g}dB'] = dict(summary=summ, per_channel=rows)
    print(f'== SNR {snr_db:g} dB mean BER over {K} channels x {T * n} test bits: ' + json.dumps({a: (round(b, 5) if isinstance(b, float) else b) for a, b in summ.items()}), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lordnet_nml_check.json'), 'w'), indent=1)
print(f'done in {time.time() - t0:.0f}s')
