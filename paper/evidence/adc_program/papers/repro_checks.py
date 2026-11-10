"""Small arithmetic checks for gnomp.md (GNOMP paper, Zhu et al., IEEE TAES 2026; arXiv 2307.00491v4).
(1) CFAR threshold tau_th = -2 ln(1 - (1 - PFA)^(1/N)) (published eq. 39; v4 eq. 44) and its approximation -2 ln(PFA/N) (published eq. 41)
    for N = 128 (paper example, threshold reported as 25.5 dB), 256 (our 16x16 beam grid), 512, 1024.
(2) Unquantised, single sinusoid: integrated SNR needed for Pd = 0.5 at PFA = 0.01, N = 1024 (paper Sec. V-B reports 12.4 dB),
    for an on-grid target, a target with uniformly random grid offset (Fejer straddle loss beta, published eq. 35) and a half-bin target.
(3) Runtime ratios 'sigma^2 unknown' / 'sigma^2 known' from the paper's reported run times (published Sec. V-I and V-K)."""
import numpy as np
from scipy.stats import ncx2

print('(1) CFAR thresholds, PFA = 0.01')
for N in [128, 256, 512, 1024]:
    t = -2 * np.log(1 - (1 - 0.01) ** (1 / N))
    print(f'  N={N:5d}: tau_th={t:7.3f}  10log10={10 * np.log10(t):5.2f} dB  20log10={20 * np.log10(t):5.2f} dB  approx eq.(41)={-2 * np.log(0.01 / N):7.3f}')
print('  paper example statistics (dB read as 20log10): 41.4 dB ->', round(10 ** (41.4 / 20), 2), '; 28.8 dB ->', round(10 ** (28.8 / 20), 2),
      '; 24 dB ->', round(10 ** (24 / 20), 2))

print('(2) unquantised Pd = 0.5 point, N = 1024, PFA = 0.01')
N = 1024; tau = -2 * np.log(1 - 0.99 ** (1 / N)); snr = np.arange(8, 20, 0.01)
d = np.linspace(-0.5, 0.5, 2001)
with np.errstate(invalid='ignore', divide='ignore'):
    beta = np.where(np.abs(d) < 1e-12, 1.0, (np.sin(np.pi * d) / (N * np.sin(np.pi * d / N))) ** 2)
for name, b in [('on-grid (beta = 1)', np.array([1.0])), ('uniform random offset, mean Pd', beta), ('half-bin offset', np.array([beta[0]]))]:
    lam = 2 * 10 ** (snr / 10)                     # lambda = (2/sigma^2) N |x|^2 = 2 * integrated SNR (published eq. 25)
    pd = np.array([ncx2.sf(tau, 2, l * b).mean() for l in lam])
    print(f'  {name:32s}: Pd = 0.5 at integrated SNR {np.interp(0.5, pd, snr):5.2f} dB (paper: 12.4 dB, target frequency not stated)')

print('(3) run-time ratio sigma^2 unknown / known (paper numbers)')
print(f'  Sec. V-I (time-varying thresholds): {6.12 / 1.70:.2f}x (PFA 0.01), {6.18 / 1.74:.2f}x (PFA 0.05)')
print(f'  Sec. V-K (AWR1642 data): {0.85 / 0.15:.2f}x (B=2), {1.05 / 0.14:.2f}x (B=3)')
print(f'  integrated SNR of one of L=3 equal-power paths in our scenes = SNR + 10log10(256/3) = SNR + {10 * np.log10(256 / 3):.2f} dB')
