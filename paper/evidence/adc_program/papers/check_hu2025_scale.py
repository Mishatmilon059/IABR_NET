"""Check L for onebit_unrolled.md (Hu, Sun, Zhang, IEEE JSTSP 2025, eq. (39)): the SBRI-X data term
   sum_i ln(1 + a exp(-b ybar_i (a_i^T x + eps_i)))   (real-stacked, Re and Im parts)
depends on (b, x, eps) only through (b x, b eps). So the slope b is not identifiable from the likelihood alone:
(b, x, eps) and (1, b x, b eps) give the same data term; only the sparsity prior gamma sum (|x_i|^2 + eta)^(alpha/2) changes.
Instance: 18-element SLA of the paper (Sec. IV-B), 1-deg grid on [-60, 60], 2 sources at (-10, 20) deg, Re/Im gains ~ U(0.5, 1)."""
import numpy as np
rng = np.random.default_rng(0)
pos = np.array([0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], float)
grid = np.arange(-60, 61, 1.0)
A = np.exp(1j * np.pi * np.outer(pos, np.sin(np.deg2rad(grid))))
x = np.zeros(len(grid), complex)
for th in (-10, 20):
    x[grid == th] = rng.uniform(0.5, 1) + 1j * rng.uniform(0.5, 1)
sigma = 10 ** (-10 / 20)  # 10 dB if |x| ~ 1
y = A @ x + sigma / np.sqrt(2) * (rng.standard_normal(18) + 1j * rng.standard_normal(18))
yb = np.sign(y.real) + 1j * np.sign(y.imag)

def data_term(a, b, xt, eps):
    z = A @ xt + eps
    s = np.concatenate([yb.real * z.real, yb.imag * z.imag])
    return np.sum(np.log1p(a * np.exp(-b * s)))

def prior(xt, gamma=1.0, alpha=1.0, eta=1e-6):
    return gamma / alpha * np.sum((np.abs(xt) ** 2 + eta) ** (alpha / 2))

xt = x / sigma
eps = 0.3 * (rng.standard_normal(18) + 1j * rng.standard_normal(18))
print('L. scale confounding of the slope b in eq. (39) (a = 1)')
for b in [1.0, 0.5, 0.19]:
    d1 = data_term(1.0, b, xt, eps); d2 = data_term(1.0, 1.0, b * xt, b * eps)
    print(f'  b={b:4.2f}: data term (b, x, eps) = {d1:.10f}   (1, b x, b eps) = {d2:.10f}   |diff| = {abs(d1 - d2):.1e}'
          f'   prior(x) = {prior(xt):.4f}  prior(b x) = {prior(b * xt):.4f}')
print('  -> identical data terms: a learned b only fixes the scale of x (and eps) relative to the prior / the network weights;')
print('     reading b = 0.19 as a noise level requires x to stay in the noise-normalised units x/sigma of eq. (35).')
print('done')
