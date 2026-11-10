"""Numerical checks of the mathematics in Hu, Sun, Zhang, IEEE JSTSP 19(6):909-925, 2025 (arXiv 2502.17473),
written for the explainer onebit_unrolled.md. Pure numpy, memory-light (threads 2). Output: check_hu2025_math.log

A. eq.(41): f(s)=ln(1+a exp(-b s)); is the printed f'' formula right (finite differences)?
B. eq.(41): is f''(s) <= a b^2/(a+1)^2 for all s (the printed bound)? Compare with the true maximum b^2/4.
C. eq.(42): is the quadratic with curvature a b^2/(a+1)^2 a global upper bound (majoriser) of f?  (needed for MM)
D. eq.(33): does sig(y)^((1+yb)/2) (1-sig(y))^((1-yb)/2) = sig(yb*y) hold, and do the two class probabilities sum to 1?
E. logistic vs Gaussian CDF: which slope b (a=1) best matches Phi(x)? how far is the learned b~0.19 (Fig. 9) from it?
F. eq.(38)-(39): with no prior on the noise offsets eps, can the objective be driven to its lower bound by x=0?
G. our project's 1-bit ADC (stress_sim.adc, the real source code) equals (A/2)*csgn(Y) with sign(0)=+1 (paper's convention).
"""
import os
os.environ['OMP_NUM_THREADS'] = '2'; os.environ['OPENBLAS_NUM_THREADS'] = '2'; os.environ['MKL_NUM_THREADS'] = '2'
import numpy as np
from math import erf, sqrt

def Phi(x):
    return 0.5 * (1.0 + np.vectorize(erf)(np.asarray(x) / sqrt(2.0)))

def f(s, a, b):
    return np.log1p(a * np.exp(-b * s))

def fp(s, a, b):                       # analytic f'
    return -a * b * np.exp(-b * s) / (1.0 + a * np.exp(-b * s))

def fpp_paper(s, a, b):                # printed eq.(41), left part
    return a * b ** 2 / (np.exp(b * s) + a ** 2 * np.exp(-b * s) + 2 * a)

print('=== A. eq.(41) formula vs finite differences ===')
s = np.linspace(-8, 8, 3201); h = 1e-4
for a, b in [(1.0, 1.0), (0.5, 1.7), (3.0, 1.0), (1.0, 0.19)]:
    fd = (f(s + h, a, b) - 2 * f(s, a, b) + f(s - h, a, b)) / h ** 2
    print(f'a={a:<4} b={b:<4}  max|f\'\'_FD - f\'\'_eq41| = {np.abs(fd - fpp_paper(s, a, b)).max():.2e}')

print('\n=== B. printed bound a b^2/(a+1)^2 vs true max of f\'\' (= b^2/4 at s = ln(a)/b) ===')
s = np.linspace(-60, 60, 600001)
for b in [0.19, 1.0, 1.7]:
    for a in [0.5, 1.0, 1.7, 3.0, 7.0]:
        true_max = fpp_paper(s, a, b).max()
        bound = a * b ** 2 / (a + 1) ** 2
        flag = 'OK (bound valid)' if true_max <= bound * (1 + 1e-9) else 'VIOLATED'
        print(f'b={b:<4} a={a:<4}  max f\'\'={true_max:.5f}  b^2/4={b*b/4:.5f}  printed bound={bound:.5f}  '
              f'ratio max/bound={true_max/bound:.3f}  {flag}')

print('\n=== C. is the eq.(42) quadratic a majoriser of f?  min over s, s_k of Q(s;s_k) - f(s) (must be >= 0) ===')
sk = np.linspace(-10, 10, 401)[:, None]; ss = np.linspace(-15, 15, 3001)[None, :]
for a, b in [(1.0, 1.0), (1.0, 0.19), (0.5, 1.7), (0.5, 1.0), (3.0, 1.0), (7.0, 1.0)]:
    Lc = a * b ** 2 / (a + 1) ** 2
    Q = f(sk, a, b) + fp(sk, a, b) * (ss - sk) + 0.5 * Lc * (ss - sk) ** 2
    gap = Q - f(ss, a, b)
    i, j = np.unravel_index(np.argmin(gap), gap.shape)
    print(f'a={a:<4} b={b:<4}  min gap = {gap.min(): .4e}  at s_k={sk[i,0]: .2f}, s={ss[0,j]: .2f}  '
          f'-> {"majoriser" if gap.min() >= -1e-12 else "NOT a majoriser"}')

print('\n=== D. eq.(33) identity and normalisation ===')
y = np.linspace(-5, 5, 1001)
for a, b in [(1.0, 1.0), (0.5, 1.7), (3.0, 1.0)]:
    sig = lambda x: 1.0 / (1.0 + a * np.exp(-b * x))
    err_minus = np.abs((1 - sig(y)) - sig(-y)).max()          # yb = -1 case of eq.(33)
    tot = sig(y) + sig(-y)                                      # P(yb=+1) + P(yb=-1) under the sig(yb*y) form
    print(f'a={a:<4} b={b:<4}  max|1-sig(y) - sig(-y)| = {err_minus:.4f}   sig(y)+sig(-y) in [{tot.min():.4f}, {tot.max():.4f}]')

print('\n=== E. logistic slope that best fits the Gaussian CDF (a=1), sup-norm over x in [-10,10] ===')
x = np.linspace(-10, 10, 4001); P = Phi(x)
bs = np.linspace(0.05, 3.0, 5901)
errs = np.array([np.abs(P - 1 / (1 + np.exp(-bb * x))).max() for bb in bs])
print(f'best b = {bs[errs.argmin()]:.3f}, sup error = {errs.min():.4f}')
for bb in [0.19, 0.5, 1.0, 1.5, 1.7, 2.0]:
    print(f'  b={bb:<4}: sup|Phi - logistic| = {np.abs(P - 1/(1+np.exp(-bb*x))).max():.4f}')
print('  logistic(b x) equals Phi(x / c) best when c = 1.7/b, i.e. b=0.19 behaves like a Gaussian likelihood with noise std about'
      f' {1.7/0.19:.1f} x the nominal (noise-normalised) one')

print('\n=== F. eq.(38)-(39): objective with free noise offsets eps (no prior on eps in (38)) ===')
rng = np.random.default_rng(13001)        # adversarial fresh-bank seed range, illustration only
pos = np.array([0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])   # 18-element SLA of Sec. IV-B (units lambda/2)
grid = np.deg2rad(np.arange(-60, 61, 1.0)); M, N = len(pos), len(grid)
Adict = np.exp(1j * np.pi * pos[:, None] * np.sin(grid)[None, :])
th = np.deg2rad([-10.0, 20.0]); s_true = rng.uniform(0.5, 1, 2) + 1j * rng.uniform(0.5, 1, 2)
snr_db = 10.0; sigma = np.sqrt(np.sum(np.abs(s_true) ** 2) / 10 ** (snr_db / 10))   # illustrative SNR convention (paper gives none)
yclean = np.exp(1j * np.pi * pos[:, None] * np.sin(th)[None, :]) @ s_true
n = sigma / np.sqrt(2) * (rng.standard_normal(M) + 1j * rng.standard_normal(M))
ybar = np.where(np.real(yclean + n) >= 0, 1.0, -1.0) + 1j * np.where(np.imag(yclean + n) >= 0, 1.0, -1.0)
gam, alp, eta = 1.0, 0.5, 1e-6            # illustrative (paper gives no values for gamma0 / alpha used)
def objective(xt, eps, a, b):
    z = Adict @ xt + eps
    lik = np.sum(np.log1p(a * np.exp(-b * np.real(ybar) * np.real(z)))) + np.sum(np.log1p(a * np.exp(-b * np.imag(ybar) * np.imag(z))))
    pri = gam / alp * np.sum((np.abs(xt) ** 2 + eta) ** (alp / 2))
    return lik, pri
x_true = np.zeros(N, complex); x_true[np.argmin(np.abs(grid[:, None] - th[None, :]), 0)] = s_true / sigma
for a, b in [(1.0, 1.0), (0.5, 1.7)]:
    lik, pri = objective(x_true, np.zeros(M), a, b)
    print(f'a={a}, b={b}: at (x = true/sigma, eps = 0): likelihood term {lik:.3f}, prior term {pri:.3f}, total {lik+pri:.3f}')
    for t in [0.0, 1.0, 5.0, 20.0, 50.0]:
        lik, pri = objective(np.zeros(N, complex), t * ybar, a, b)
        print(f'      at (x = 0, eps = {t:>4} * ybar): likelihood term {lik:.3e}, prior term {pri:.4f}, total {lik+pri:.4f}')
print(f'  lower bound of the objective = prior at x=0 = (gamma/alpha) N eta^(alpha/2) = {gam/alp*N*eta**(alp/2):.4f}; '
      'the likelihood term is >= 0 and -> 0 as t grows, so x = 0 (no source) is the infimum of (39) as written.')

print('\n=== G. project 1-bit ADC (stress_sim.adc source) vs (A/2) csgn ===')
src = open('D:/ai_ml_project/paper/evidence/deep_research/stress/stress_sim.py', encoding='utf-8').read()
start = src.index('def adc('); end = src.index('\n\n\nSEVERITIES')
ns = {'np': np}; exec(src[start:end], ns)
Y = (rng.standard_normal((50, 16, 16)) + 1j * rng.standard_normal((50, 16, 16)))
Y[0, 0, 0] = 0.0 + 0.3j                                     # exact zero real part: sign(0) convention
A = np.maximum(np.abs(Y.real), np.abs(Y.imag)).max((1, 2))[:, None, None]
def sgn(v): return np.where(v >= 0, 1.0, -1.0)            # paper: sign(x)=1 for x>=0
Q1 = ns['adc'](Y, 1, 'peak')
print('max |adc(Y,1) - (A/2) csgn(Y)| =', np.abs(Q1 - (A / 2) * (sgn(Y.real) + 1j * sgn(Y.imag))).max())

print('\n=== H. FLOP formulas of Sec. IV-E / Table I evaluated at the paper\'s array sizes (N inferred from the stated grids) ===')
def flops(N, M):
    return dict(SBRI_on_iter=2/3*N**3 + 2*M*N**2, SBRI_off_iter=N**3 + 3*M*N**2,
                SBRINet_on_layer=8*N*M + 13088*N, SBRINet_off_layer=16*N*M + 13088*N + 12*N**2,
                SBRIXNet_on_layer=12*N*M + 13088*N, SBRIXNet_off_layer=20*N*M + 13088*N + 12*N**2)
for tag, N, M in [('on-grid, 1 deg grid on [-60,60] (N=121), M=18', 121, 18), ('off-grid, 2 deg grid (N=61), M=18', 61, 18),
                  ('off-grid, 2 deg grid (N=61), M=10', 61, 10), ('OUR 2-D case if gridded 64x64 (N=4096), M=256', 4096, 256)]:
    d = flops(N, M)
    print(tag); print('   ' + ', '.join(f'{k}={v:.3e}' for k, v in d.items()))
    print(f'   on-grid: one SBRI-Net layer / one SBRI iteration = {d["SBRINet_on_layer"]/d["SBRI_on_iter"]:.3g};  '
          f'off-grid: one SBRI-X-Net layer / one off-grid SBRI iteration = {d["SBRIXNet_off_layer"]/d["SBRI_off_iter"]:.3g}')

print('\n=== I. SBRI pseudo-measurement (eqs. 14, 17) = mean of a truncated normal (EM E-step for one-bit data)? ===')
from scipy.stats import truncnorm, norm
mu = np.linspace(-6, 6, 241); worst = 0.0
for yb in [1.0, -1.0]:
    d = yb * mu
    v_paper = yb * (d + norm.pdf(d) / norm.cdf(d))                  # ybar * (d - I'(d)),  I'(x) = -phi(x)/Phi(x)
    lo, hi = (0 - mu, np.inf) if yb > 0 else (-np.inf, 0 - mu)      # z ~ N(mu,1) restricted to the observed sign
    v_tn = truncnorm.mean(lo, hi, loc=mu, scale=1.0)
    worst = max(worst, np.abs(v_paper - v_tn).max())
print(f'max |v_eq17 - E[z | sign(z) = ybar]| over mu in [-6,6], ybar = +-1 : {worst:.2e}')

print('\n=== J. how far is the paper\'s DOA dictionary from the RIP assumption (Theorems 1-2 need delta_2s < 1)? ===')
for step in [1.0, 2.0]:
    g = np.deg2rad(np.arange(-60, 60 + step / 2, step))
    Ad = np.exp(1j * np.pi * pos[:, None] * np.sin(g)[None, :]) / np.sqrt(len(pos))
    coh = np.abs(np.sum(np.conj(Ad[:, :-1]) * Ad[:, 1:], 0)).max()
    dc = dr = 0.0                                                    # unit-norm columns = most favourable scaling
    for i in range(len(g) - 3):                                      # 2s = 4 adjacent atoms (s = 2 sources)
        G = np.conj(Ad[:, i:i + 4]).T @ Ad[:, i:i + 4]
        ec = np.linalg.eigvalsh(G); er = np.linalg.eigvalsh(np.real(G))   # complex x / real x (paper states x in R^N)
        dc = max(dc, ec.max() - 1, 1 - ec.min()); dr = max(dr, er.max() - 1, 1 - er.min())
    print(f'grid step {step} deg, 18-element SLA: max adjacent-atom coherence {coh:.4f}; delta_4 >= {dc:.3f} (complex x), >= {dr:.3f} (real x)'
          f' -> assumption delta_4 < 1 {"VIOLATED" if min(dc, dr) >= 1 else "not excluded by this check"}')
print('\n=== K. our beamspace model in the paper\'s form: vec(Y) = (F^T kron W^H) vec(H); matched filter A^H vec(Y) = a_r(u)^H (W Y F^H) a_t(v) ===')
Wr = np.linalg.qr(rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16)))[0]
Fr = np.linalg.qr(rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16)))[0]
Yb = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
uu, vv = 0.3, -0.55
ar = np.exp(-1j * np.pi * uu * np.arange(16)) / 4; at = np.exp(-1j * np.pi * vv * np.arange(16)) / 4   # project steering convention
col = np.kron(Fr.T, Wr.conj().T) @ np.kron(at.conj(), ar)            # dictionary atom for (u, v), column-major vec
lhs = np.vdot(col, Yb.flatten(order='F'))                             # A^H vec(Y) for this atom
rhs = ar.conj() @ (Wr @ Yb @ Fr.conj().T) @ at
print(f'|lhs - rhs| = {abs(lhs - rhs):.2e}  (atom equals vec(W^H a_r a_t^H F): '
      f'{np.abs(col - (Wr.conj().T @ np.outer(ar, at.conj()) @ Fr).flatten(order="F")).max():.2e})')
print('done')
