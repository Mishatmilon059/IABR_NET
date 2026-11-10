"""Amplitude 'learning threshold' sanity check for SignalNet (Dreifuerst & Heath, IEEE TSP 2022, arXiv:2106.05490).

Question: the paper draws ONE amplitude threshold, Var(U(0.1,1)) = 0.0675 (-11.71 dB), for every m, yet its 1-bit amplitude figure
(arXiv Fig. 10a = TSP Fig. 6a) shows the Periodogram (no learning) at about -18 dB for m = 4, 5 at SNR = -20 dB. Is that possible if the
labels really are a_i ~ U(0.1, 1)?  We re-simulate Algorithm 1 (N = 64) and measure how much amplitude information the Periodogram
amplitude estimate carries, by giving it the BEST affine calibration a_hat -> alpha + beta * a_hat fitted on the test data itself
(optimistic for the Periodogram). If even that stays near Var(a), no plain Periodogram can sit 6 dB below Var(a).

Also checks the scale-invariance argument at 1 bit, m = 1, high SNR (sign() ignores any positive scale).
SNR conventions (the paper does not define SNR): 'ref' noise var = 1/SNR (unit reference amplitude); 'sig' noise var = mean|signal|^2/SNR.
Normalisation: y * sqrt(N)/||y|| (unit power per complex sample), then b-bit uniform mid-rise quantiser on [-1, 1] per real part
(step 2/2^b, clipped) - our reading of eqs. (3)-(5); for b = 1 any positive scale gives the same output.
Periodogram: zero-padded FFT (N_FFT = 4096), m largest local maxima, amplitude |r|/N, peaks matched to labels in frequency order.
Seeds 81000301.. (validation range). numpy only, chunks of 200, 2 threads.
Usage: python signalnet_amp_floor.py  (stdout -> signalnet_amp_floor.log)
"""
import os, sys, time
os.environ['OMP_NUM_THREADS'] = '2'; os.environ['MKL_NUM_THREADS'] = '2'; os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
N, NFFT, T, CH = 64, 4096, 4000, 200
db = lambda x: 10 * np.log10(x)
n = np.arange(N)


def gen(m, snr_db, conv, rng, B):
    while True:  # Algorithm 1 with its do-while rejection (max f <= 0.5)
        w0 = rng.uniform(0, 0.25, B); f = np.empty((B, m)); f[:, 0] = w0
        for i in range(1, m):
            f[:, i] = w0 + i / N + np.abs(rng.normal(0, np.sqrt(2.5 / N), B))
        ok = f.max(1) <= 0.5
        if ok.sum() >= B // 2: f = f[ok]; break
    B2 = len(f)
    a = rng.uniform(0.1, 1.0, (B2, m)); ph = rng.uniform(0, 2 * np.pi, (B2, m))
    s = (a[:, :, None] * np.exp(1j * (2 * np.pi * f[:, :, None] * n + ph[:, :, None]))).sum(1)
    p = 1.0 if conv == 'ref' else (np.abs(s) ** 2).mean(1, keepdims=True)
    sig2 = p / 10 ** (snr_db / 10)
    y = s + np.sqrt(sig2 / 2) * (rng.standard_normal((B2, N)) + 1j * rng.standard_normal((B2, N)))
    return f, a, y * np.sqrt(N) / np.linalg.norm(y, axis=1, keepdims=True)


def quant(x, b):
    if b == 1: return np.sign(x.real) + 1j * np.sign(x.imag)
    st = 2.0 / 2 ** b
    q = lambda v: np.clip((np.floor(v / st) + 0.5) * st, -1 + st / 2, 1 - st / 2)
    return q(x.real) + 1j * q(x.imag)


def periodogram_amp(z, m):
    r = np.fft.fft(z, NFFT, axis=1); mag = np.abs(r) / N
    out_a = np.zeros((len(z), m)); out_f = np.zeros((len(z), m))
    for k in range(len(z)):
        g = mag[k]; loc = np.where((g > np.roll(g, 1)) & (g >= np.roll(g, -1)))[0]
        top = loc[np.argsort(g[loc])[::-1][:m]]
        if len(top) < m: top = np.concatenate([top, np.argsort(g)[::-1][:m - len(top)]])
        fr = (top / NFFT + 0.5) % 1 - 0.5; o = np.argsort(fr)
        out_a[k] = g[top][o]; out_f[k] = fr[o]
    return out_a


t0 = time.time(); seed = 81000301
print('Var(U(0.1,1)) = %.4f (%.2f dB) - the paper\'s single amplitude threshold' % (0.81 / 12, db(0.81 / 12)))
print('%-4s %-3s %-5s %-4s | %-26s | %-34s | %s' % ('m', 'b', 'SNR', 'conv', 'best constant (blind) MSE', 'best-affine Periodogram MSE (rho^2)', 'raw Periodogram MSE'))
for m, b, snr, conv in [(5, 1, -20, 'ref'), (5, 1, -20, 'sig'), (4, 1, -20, 'ref'), (5, 3, -20, 'ref'),
                        (1, 1, 19, 'ref'), (1, 1, 19, 'sig'), (1, 3, 19, 'ref'), (1, 3, 19, 'sig'), (5, 1, 19, 'ref')]:
    rng = np.random.default_rng(seed); seed += 1
    A, AH = [], []
    while sum(len(x) for x in A) < T:
        f, a, y = gen(m, snr, conv, rng, CH)
        lab = a[np.arange(len(a))[:, None], np.argsort(f, 1)]       # labels in frequency order
        A.append(lab); AH.append(periodogram_amp(quant(y, b), m))
    A = np.concatenate(A)[:T].ravel(); AH = np.concatenate(AH)[:T].ravel()
    blind = A.var()
    rho2 = np.corrcoef(A, AH)[0, 1] ** 2
    aff = A.var() * (1 - rho2)
    raw = ((A - AH) ** 2).mean()
    print('%-4d %-3d %-5d %-4s | %.4f (%6.2f dB)            | %.4f (%6.2f dB)  rho^2 = %.3f       | %.4f (%6.2f dB)'
          % (m, b, snr, conv, blind, db(blind), aff, db(aff), rho2, raw, db(raw)))
print('paper plotted (digitised, arXiv Fig. 10a, 1-bit, SNR -20 dB): Periodogram m=4 -18.40 dB, m=5 -17.85 dB; network m=4 -19.53, m=5 -21.88 dB')
print('time %.1f s' % (time.time() - t0))
