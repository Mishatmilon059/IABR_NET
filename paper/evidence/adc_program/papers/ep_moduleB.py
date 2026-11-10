"""Module B of VALSE-EP (Zhu, Zhang, Meng, China Commun. 2021, arXiv 1811.05680, Sec. V-A, eqs. (47)-(51)) for a
cell-quantised complex measurement, written for our ADC model (cells from acommon.adc_cells). Reference code, no learning.

Per real component (Re and Im separately, as in the authors' MATLAB GaussianMomentsComputation_MJH.m):
  prior from module A:  z ~ N(p, v)            (p = Re or Im of z_A_ext, v = v_A_ext / 2)
  observation:          r = z + n, n ~ N(0, s2) (s2 = sigma^2 / 2),  r in [lo, hi)
  c = sqrt(v + s2), a = (lo - p)/c, b = (hi - p)/c, Z = Phi(b) - Phi(a)
  E[z | cell]   = p + (v / c) (phi(a) - phi(b)) / Z
  Var[z | cell] = v + (v^2 / c^2) [ (a phi(a) - b phi(b)) / Z - ((phi(a) - phi(b)) / Z)^2 ]
Complex: z_B_post = E_re + j E_im, v_B_post = Var_re + Var_im (eqs. 47-48).
Extrinsic (eq. 51): v_B_ext = 1 / (1/v_B_post - 1/v_A_ext),  z_B_ext = v_B_ext (z_B_post/v_B_post - z_A_ext/v_A_ext)
  -> pseudo-unquantised data  y~ = z_B_ext  with per-sample noise variance  sigma~^2 = v_B_ext  (eqs. 60-61).
Run: python -u ep_moduleB.py  (Monte-Carlo check of the closed form; seed 0; log -> ep_moduleB_check.log)
"""
import numpy as np
from scipy.special import ndtr, log_ndtr


def _phi(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)


def trunc_moments(p, v, s2, lo, hi):
    """posterior mean / variance of z ~ N(p, v) given z + N(0, s2) in [lo, hi); lo may be -inf, hi may be +inf."""
    c = np.sqrt(v + s2)
    a = (lo - p) / c; b = (hi - p) / c
    # Z computed in a numerically safer way on the side where the mass is small
    Z = np.where(a > 0, ndtr(-a) - ndtr(-b), ndtr(b) - ndtr(a))
    Z = np.maximum(Z, 1e-300)
    fa, fb = np.isfinite(a), np.isfinite(b)
    a0 = np.where(fa, a, 0.0); b0 = np.where(fb, b, 0.0)          # phi(+-inf) = 0 and x phi(x) -> 0 at +-inf
    pa = np.where(fa, _phi(a0), 0.0); pb = np.where(fb, _phi(b0), 0.0)
    apa = a0 * pa; bpb = b0 * pb
    r1 = (pa - pb) / Z
    mean = p + (v / c) * r1
    var = v + (v * v / (c * c)) * ((apa - bpb) / Z - r1 * r1)
    return mean, np.maximum(var, 1e-12 * v)


def module_b(z_A_ext, v_A_ext, lo_re, hi_re, lo_im, hi_im, sigma2):
    """complex extrinsic message of module B; all arrays same shape (e.g. (B,16,16)). Returns z_B_post, v_B_post, y_tilde, s2_tilde."""
    mr, vr = trunc_moments(z_A_ext.real, v_A_ext / 2, sigma2 / 2, lo_re, hi_re)
    mi, vi = trunc_moments(z_A_ext.imag, v_A_ext / 2, sigma2 / 2, lo_im, hi_im)
    z_post = mr + 1j * mi; v_post = vr + vi
    v_ext = 1.0 / (1.0 / v_post - 1.0 / v_A_ext)
    v_ext = np.where(v_ext > 0, v_ext, 1e6 * v_A_ext)      # guard (authors' code uses a similar positivity guard on v_A_ext)
    z_ext = v_ext * (z_post / v_post - z_A_ext / v_A_ext)
    return z_post, v_post, z_ext, v_ext


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    n = 400000
    worst = 0.0
    cases = [(0.3, 1.0, 0.1, 0.0, np.inf), (0.3, 1.0, 0.1, -np.inf, 0.0), (-0.2, 0.5, 0.05, -0.25, 0.0),
             (1.5, 0.2, 0.3, 0.5, 1.0), (0.0, 10.0, 0.05, 0.0, np.inf), (2.0, 1.0, 0.5, -np.inf, -1.0)]
    for (p, v, s2, lo, hi) in cases:
        z = p + np.sqrt(v) * rng.standard_normal(n); r = z + np.sqrt(s2) * rng.standard_normal(n)
        sel = (r >= lo) & (r < hi)
        mc_m, mc_v = z[sel].mean(), z[sel].var()
        m, var = trunc_moments(np.array(p), np.array(v), s2, np.array(lo), np.array(hi))
        se_m = np.sqrt(mc_v / sel.sum())
        print(f'p={p:+.2f} v={v:.2f} s2={s2:.2f} cell=[{lo},{hi}) kept={sel.sum():6d}  mean closed {float(m):+.5f} MC {mc_m:+.5f} (se {se_m:.5f})'
              f'  var closed {float(var):.5f} MC {mc_v:.5f}')
        worst = max(worst, abs(float(m) - mc_m) / se_m)
    print(f'max |closed - MC| / se(mean) = {worst:.2f}  (expect < ~4)')
    # extrinsic sanity: with an uninformative prior (huge v_A_ext) and a narrow cell, y~ -> cell centre region, s2~ small
    zA = np.zeros(3, complex); vA = np.full(3, 1e4)
    lo = np.array([0.0, 0.25, -0.5]); hi = np.array([0.25, 0.5, -0.25])
    zp, vp, ye, se = module_b(zA, vA, lo, hi, lo, hi, sigma2=1e-3)
    print('narrow cells, flat prior: y~ =', np.round(ye, 4), ' sigma~^2 =', np.round(se, 5), ' (cell width^2/12*2 =', round(2 * 0.25 ** 2 / 12, 5), ')')
    lo1 = np.array([0.0, -np.inf, 0.0]); hi1 = np.array([np.inf, 0.0, np.inf])
    zA1 = np.array([0.1 + 0.1j, 0.1 - 0.1j, -0.5 + 0.5j]); vA1 = np.full(3, 0.2)
    zp, vp, ye, se = module_b(zA1, vA1, lo1, hi1, lo1, hi1, sigma2=0.1)
    print('1-bit cells, informative prior: y~ =', np.round(ye, 3), ' sigma~^2 =', np.round(se, 3),
          ' (per-sample variances differ = heteroscedastic pseudo-noise)')
