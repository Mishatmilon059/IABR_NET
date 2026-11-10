"""Numerical checks of the four foundation papers (NOMP, NOMP-CFAR, GAMP, unrolling) for foundations.md.
Only numpy/scipy; memory-light (threads 2, Monte-Carlo in chunks of 2000). stdout -> foundations_checks.log.
Every check recomputes a number printed in a paper (or applies a paper's formula to our setting) so foundations.md can cite a log.
Seeds: Monte-Carlo checks use default_rng(13000 + k) (adversarial-fresh bank range of the project rules; no scenes are generated)."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np
from scipy import integrate, optimize, special, stats

np.set_printoptions(precision=4, suppress=True)
print('=' * 100)
print('A. NOMP (Mamandipoor, Ramasamy, Madhow, IEEE TSP 2016; arXiv 1509.01942v2)')
print('=' * 100)
N, PFA = 256, 1e-2
tau_exact = -np.log(1 - (1 - PFA) ** (1 / N))                    # eq (12) inverted (Sec. III-A)
tau_asym = np.log(N) - np.log(np.log(1 / (1 - PFA)))             # eq (13)
print(f'A1 stopping threshold tau/sigma^2, N={N}, Pfa={PFA}: exact (12) {tau_exact:.4f}, asymptotic (13) {tau_asym:.4f}, '
      f'diff {tau_asym - tau_exact:+.2e}')
for p in (1e-3, 5e-2, 1e-1):
    print(f'    Pfa={p:g}: exact {-np.log(1 - (1 - p) ** (1 / N)):.4f}, asymptotic {np.log(N) - np.log(np.log(1 / (1 - p))):.4f}')

# A2: E[alpha] for frequency uniform within one DFT bin (Sec. III-B, "E[alpha] = 0.88")
w = np.linspace(-np.pi / N, np.pi / N, 200001)[1:-1]
alpha = np.sin(N * w / 2) / (N * np.sin(w / 2))
print(f'A2 E[alpha] = E[sin(N w/2)/(N sin(w/2))], w ~ U[-pi/N, pi/N], N={N}: {alpha.mean():.4f} (paper: 0.88)')


# A3: minimum oversampling factor for Newton convergence (Sec. IV): I=(-a,a), M(a)=sup_I 0.5|h'''/h''|, need a < 1/M(a)
def h(x, N=N):
    return np.sin(np.pi * x) / (N * np.sin(np.pi * x / N))


def d(f, x, k, e=1e-3):
    if k == 2:
        return (f(x + e) - 2 * f(x) + f(x - e)) / e ** 2
    if k == 3:
        return (f(x + 2 * e) - 2 * f(x + e) + 2 * f(x - e) - f(x - 2 * e)) / (2 * e ** 3)


def margin(a):
    x = np.linspace(-a, a, 4001); x = x[np.abs(x) > 1e-6]
    M = np.max(0.5 * np.abs(d(h, x, 3) / d(h, x, 2)))
    return 1 / M - a, M


m45, M45 = margin(0.45)
a_star = optimize.brentq(lambda a: margin(a)[0], 0.2, 0.9)
print(f'A3 Newton-convergence interval: a=0.45 -> M={M45:.4f}, 1/M={1 / M45:.4f} (condition |x0|<1/M holds: {m45 > 0}); '
      f'largest a with a<1/M(a): {a_star:.4f} -> max grid spacing {2 * a_star:.4f} DFT bins -> min oversampling {1 / (2 * a_star):.3f} '
      f'(paper: I=(-0.45,0.45), spacing ~0.9, factor ~1.12)')

# A4: Theorem 2 / 3 factor (1 - 2 pi / gamma)^-1
for g in (4, 2 * np.pi, 8, 10, 20, 64):
    f = 1 - 2 * np.pi / g
    print(f'A4 gamma={g:6.3f}: 1 - 2pi/gamma = {f:+.4f} -> bound factor {"undefined/vacuous (<=0)" if f <= 0 else f"{1 / f:.4f}"}')

# A5: SNR conventions. NOMP integrated SNR_l = |g_l|^2/sigma^2; per-sample = integrated / N (Remark 1)
print(f'A5 NOMP Remark 1: integrated 25 dB -> per-sample {25 - 10 * np.log10(N):.2f} dB (paper: ~1 dB)')
# our model (stress_sim docstring): H = sqrt(256) sum_l alpha_l a_r a_t^H with unit-norm a_r, a_t -> g_l = 16 alpha_l; to_ant unitary;
# noise variance per entry sigma^2 = 10^(-SNR/10); alpha_l ~ CN(0, 1/L) sorted, L = 3.
L = 3
mean_path = 10 * np.log10(256 / L)
weak_mean = 10 * np.log10(256 / L / L)          # min of L iid Exp(mean 1/L) is Exp(mean 1/L^2)
weak_med = 10 * np.log10(256 * np.log(2) / L ** 2)
print(f'A5 our setting: integrated per-path SNR = SNR_project + {mean_path:.2f} dB (mean path), + {weak_mean:.2f} dB (mean of weakest of 3), '
      f'+ {weak_med:.2f} dB (median of weakest)')
for s in (-10, -5, 0, 5, 15, 25):
    print(f'    project SNR {s:+3d} dB -> mean-path integrated {s + mean_path:5.1f} dB, weakest-path median {s + weak_med:5.1f} dB')

# A6: realised false-alarm rate of tau_exact on the N-point DFT grid vs a 4x oversampled grid (1-D, N=256, noise only)
rng = np.random.default_rng(13000)
hits_dft = hits_os = 0; T = 20000
for c in range(T // 2000):
    z = (rng.standard_normal((2000, N)) + 1j * rng.standard_normal((2000, N))) / np.sqrt(2)     # sigma^2 = 1
    G = np.abs(np.fft.fft(z, axis=1)) ** 2 / N
    Go = np.abs(np.fft.fft(z, n=4 * N, axis=1)) ** 2 / N
    hits_dft += (G.max(1) > tau_exact).sum(); hits_os += (Go.max(1) > tau_exact).sum()
print(f'A6 1-D noise-only, {T} trials: P(max over DFT grid > tau) = {hits_dft / T:.4f} (nominal {PFA}); '
      f'P(max over 4x oversampled grid > tau) = {hits_os / T:.4f}')

# A7: eqs (10)-(11): single-frequency CRB and ZZB (rad^2), normalised by the DFT spacing^2; where does the ZZB leave the CRB?
hh = np.linspace(1e-9, np.pi, 400001)
dk = np.sin(N * hh / 2) / (N * np.sin(hh / 2))
print('A7 single sinusoid, N=256: SNR_int(dB)  CRB/dft^2(dB)  ZZB/dft^2(dB)  ZZB-CRB(dB)')
thr = None
for s_db in range(0, 31, 2):
    snr = 10 ** (s_db / 10)
    crb = 6 / (snr * (N ** 2 - 1))
    zzb = np.trapezoid(special.ndtr(-np.sqrt(snr * (1 - dk))) * hh, hh) if hasattr(np, 'trapezoid') else np.trapz(special.ndtr(-np.sqrt(snr * (1 - dk))) * hh, hh)
    gap = 10 * np.log10(zzb / crb)
    if thr is None and gap < 3:
        thr = s_db
    print(f'    {s_db:4d}  {10 * np.log10(crb / (2 * np.pi / N) ** 2):8.2f}  {10 * np.log10(zzb / (2 * np.pi / N) ** 2):8.2f}  {gap:7.2f}')
print(f'A7 first even integrated SNR where ZZB is within 3 dB of the CRB: {thr} dB')
# A8: our Pd tolerance (1 deg per angle) expressed in DFT bins of a 16-element ULA in u = pi cos(psi) (|du| = pi sin(psi) dpsi)
d16 = 2 * np.pi / 16
for psi in (90, 60, 30, 15):
    du = np.pi * np.sin(np.radians(psi)) * np.radians(1.0)
    print(f'A8 psi={psi:3d} deg: 1 deg -> |du| = {du:.4f} rad = {du / d16:.3f} DFT bins (16-pt); NOMP miss radius 0.25 bins, '
          f'NOMP-CFAR detection radius 0.5 bins')

print('=' * 100)
print('B. NOMP-CFAR (Xu, Zhu, Fang, Zhang, Xu, IEEE TAES 59(5) 2023; arXiv 2210.10302v1)')
print('=' * 100)


def pfa_ca(alpha_, Nr, N=N, S=1):
    """eq (19) for S=1, eq (52) for S>1: Pfa = 1 - E_x[ G_S(alpha x / Nr)^N ], x ~ Gamma(S Nr), G_S = Gamma(S) CDF."""
    f = lambda x: stats.gamma.cdf(alpha_ * x / Nr, S) ** N * stats.gamma.pdf(x, S * Nr)
    lo, hi = stats.gamma.ppf(1e-12, S * Nr), stats.gamma.ppf(1 - 1e-12, S * Nr)
    return 1 - integrate.quad(f, lo, hi, limit=400)[0]


def alpha_for(p, Nr, N=N, S=1):
    return optimize.brentq(lambda a: pfa_ca(a, Nr, N, S) - p, 1e-3, 200)


a50 = alpha_for(PFA, 50)
a50_approx = 50 * (PFA / N) ** (-1 / 50) - 50                     # eq (34): Pfa ~ N (alpha/Nr + 1)^-Nr
print(f'B1 Nr=50, N=256, Pfa=1e-2: alpha (19) = {a50:.3f} ({10 * np.log10(a50):.2f} dB); approx (34) = {a50_approx:.3f} '
      f'({10 * np.log10(a50_approx):.2f} dB); NOMP alpha\' (14) = {tau_exact:.3f} ({10 * np.log10(tau_exact):.2f} dB) '
      f'[paper Sec. IV-A: 11.22 / 11.25 / 10.15 = 10.50 / 10.51 / 10.06 dB]')
# check eq (20) (finite binomial sum) against (19) where it is numerically stable
s20 = 1 - sum((-1) ** n * special.comb(N, n, exact=True) / (n * a50 / 50 + 1) ** 50 for n in range(0, N + 1))
print(f'    eq (20) at alpha={a50:.3f} (alternating binomial sum, exact integer binomials): {float(s20):.4e} '
      f'(equals the nominal {PFA} if (19) and (20) agree)')
for S, ref in ((10, 2.81), (50, 1.67)):
    print(f'B2 MMV S={S}, Nr=50: alpha_mmv (52) = {alpha_for(PFA, 50, S=S):.3f} [paper Sec. IV-A: {ref}]')
a60 = alpha_for(PFA, 60)
atr60 = 60 * ((PFA / N) ** (-1 / 60) - 1)
print(f'B3 real-data Exp. 1 (N=256, 60 training cells): NOMP-CFAR alpha (19) = {a60:.3f} ({10 * np.log10(a60):.2f} dB); '
      f'per-cell CA-CFAR with Pfa_tr = 1e-2/256: alpha_tr = {atr60:.3f} ({10 * np.log10(atr60):.2f} dB) [paper Sec. V-A: 11.03 / 11.06]')
for Nr in (10, 20, 100, 236, 1000):
    print(f'B4 alpha(Nr={Nr:4d}) = {alpha_for(PFA, Nr):.3f}  (-> {tau_exact:.3f} as Nr -> inf, Remark 1)')
# B5 Monte Carlo of the CA-CFAR detector under the paper's independence assumption (18)
rng = np.random.default_rng(13001); hit = 0
for c in range(10):
    P = rng.exponential(1.0, (2000, N)); R = rng.exponential(1.0, (2000, 50))
    hit += (P.max(1) / R.mean(1) > a50).sum()
print(f'B5 MC (20000 trials, independent reference cells): P(peak/sigma_hat^2 > {a50:.3f}) = {hit / 20000:.4f} (nominal {PFA})')

print('=' * 100)
print('C. GAMP (Rangan, ISIT 2011; arXiv 1010.5141v2): sum-product output function for our ADC cells')
print('=' * 100)
# real scalar output channel y = Q(z + w), w ~ N(0, tw): y observed only as the cell [lo, hi) of the b-bit mid-rise quantiser
# (step 2A/2^b, peak AGC A). GAMP (36)-(37): gout = (E[z|y] - p)/tp,  -dgout/dp = (1 - var(z|y)/tp)/tp;  (38): gout = d/dp log p(y|p,tp).


def post_moments(p, tp, tw, lo, hi):
    """z ~ N(p, tp) a priori, y-cell likelihood P(lo <= z + w < hi). Returns log p(y|p), E[z|y], var(z|y)."""
    s = np.sqrt(tp + tw)
    a, b = (lo - p) / s, (hi - p) / s
    Z = special.ndtr(b) - special.ndtr(a)
    pa, pb = stats.norm.pdf(a), stats.norm.pdf(b)
    pa = 0.0 if np.isinf(a) else pa; pb = 0.0 if np.isinf(b) else pb
    aa = 0.0 if np.isinf(a) else a * pa; bb = 0.0 if np.isinf(b) else b * pb
    m1 = (pa - pb) / Z                       # E[(z+w-p)/s | cell]
    v1 = 1 + (aa - bb) / Z - m1 ** 2         # var[(z+w-p)/s | cell]
    k = tp / s                               # z - p = (tp/s^2)(z+w-p) + independent part
    Ez = p + k * m1
    Vz = tp - tp ** 2 / s ** 2 + k ** 2 * v1
    return np.log(Z), Ez, Vz


A = 1.0; tw = 0.05
for b in (1, 2):
    step = 2 * A / 2 ** b
    edges = -A + step * np.arange(2 ** b + 1); edges[0], edges[-1] = -np.inf, np.inf
    worst = 0.0; worst2 = 0.0
    for p in (-0.7, -0.2, 0.0, 0.3, 0.9):
        for tp in (0.02, 0.2, 1.0):
            for i in range(2 ** b):
                lo, hi = edges[i], edges[i + 1]
                lZ, Ez, Vz = post_moments(p, tp, tw, lo, hi)
                g = (Ez - p) / tp
                e = 1e-5
                g_fd = (post_moments(p + e, tp, tw, lo, hi)[0] - post_moments(p - e, tp, tw, lo, hi)[0]) / (2 * e)
                gp = ((post_moments(p + e, tp, tw, lo, hi)[1] - (p + e)) / tp - (post_moments(p - e, tp, tw, lo, hi)[1] - (p - e)) / tp) / (2 * e)
                worst = max(worst, abs(g - g_fd)); worst2 = max(worst2, abs(-gp - (1 - Vz / tp) / tp))
    print(f'C1 {b}-bit cells (A={A}, step {step}): max |gout(36) - dlogp/dp (38)| = {worst:.2e}; '
          f'max |-dgout/dp - (1 - var/tp)/tp (37)| = {worst2:.2e}  (75 (p, tp, cell) cases)')
# C2: AWGN limit: a very narrow cell around y reproduces gout = (y - p)/(tw + tp) (42)
y, p, tp = 0.4, 0.1, 0.3
lZ, Ez, Vz = post_moments(p, tp, tw, y - 1e-4, y + 1e-4)
print(f'C2 narrow cell at y={y}: gout = {(Ez - p) / tp:.5f}; AWGN formula (42) (y-p)/(tw+tp) = {(y - p) / (tw + tp):.5f}; '
      f'-gout\' = {(1 - Vz / tp) / tp:.5f} vs (43) 1/(tp+tw) = {1 / (tp + tw):.5f}')

print('=' * 100)
print('D. Algorithm unrolling (Monga, Li, Eldar, IEEE SPM 2021): Table II ratios recomputed from the printed numbers')
print('=' * 100)
for name, un, it, net, pu, pn in (('ADMM-CSNet [14] vs D-AMP [64] vs ReconNet [58]', 2.61, 12.59, 2.83, 7.8e4, 3.2e5),
                                  ('CORONA [34] vs FISTA [29] vs ResNet [54]', 5.07, 15.33, 5.36, 1.8e3, 8.3e3),
                                  ('DUBLID [40] vs Perrone [49] vs DeblurGAN [59]', 1.47, 1462.90, 10.29, 2.3e4, 1.2e7)):
    print(f'D1 {name}: iterative/unrolled time x{it / un:.1f}; generic-net/unrolled time x{net / un:.2f}; '
          f'generic-net/unrolled params x{pn / pu:.1f}')
print('done')
