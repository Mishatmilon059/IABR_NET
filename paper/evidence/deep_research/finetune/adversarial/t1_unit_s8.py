"""Unit checks for s8 (seed 9435 smoke bank, 4 scenes): (1) POS model at the true position errors equals stress_sim.generate(M2) minus
noise; (2) analytic POS Jacobian vs finite differences; (3) PN-ML objective gradient direction sanity (finite-difference check of g at d=0).
usage: THREADS=1 python t1_unit_s8.py > t1_unit_s8.log"""
import os
os.environ['THREADS'] = '1'
import importlib.util, sys
from advc import *
sys.argv = ['x']
src = open(f'{HERE}/s8_physics_ml.py').read().split('\nout = {}')[0]
src = src.replace("ft = I.load_cnn(I.FT_CKPT)", "ft = None")
g = {'__name__': 's8'}; exec(src, g)
n = 4; b = SS.base_scenes(n, 3, 15.0, 9435)
Y0 = SS.generate(b, 'M2', 0.2) - b['noise']
gg = SS.imp_rng(9435, 'M2'); dr = 0.2 * gg.standard_normal((n, 16)); dt = 0.2 * gg.standard_normal((n, 16))
U = np.pi * np.cos(b['psi']); V = np.pi * np.cos(b['phi'])
mu, J = g['pos_jac'](U, V, b['alpha'], dr, dt)
m = mu[:, :256] + 1j * mu[:, 256:]
print('(1) max|POS model(true) - noiseless M2 Y| =', float(np.abs(m - Y0.reshape(n, 256)).max()), ' max|Y0| =', float(np.abs(Y0).max()))
e = 1e-6; K = 3
def pack(U, V, X, dr, dt): return np.concatenate([U, V, X.real, X.imag, dr, dt], 1)
def unpack(p): return p[:, :K], p[:, K:2*K], p[:, 2*K:3*K] + 1j * p[:, 3*K:4*K], p[:, 4*K:4*K+16], p[:, 4*K+16:]
p0 = pack(U, V, b['alpha'], dr, dt); err = 0
for j in range(p0.shape[1]):
    pp = p0.copy(); pp[:, j] += e; pm = p0.copy(); pm[:, j] -= e
    fd = (g['pos_jac'](*unpack(pp))[0] - g['pos_jac'](*unpack(pm))[0]) / (2 * e)
    err = max(err, float(np.abs(fd - J[:, :, j]).max() / (np.abs(J[:, :, j]).max() + 1e-12)))
print('(2) POS Jacobian max relative error vs finite differences:', err)
# (3) PN-ML: g from pn_eval equals -dF/dparam under fixed weights -> compare to FD of the weighted residual objective with frozen weights
y = SS.generate(b, 'M6', 30.0).reshape(n, 256); sn2 = (sigma_of(b['snr']) ** 2)[:, None]
q = np.exp(-np.deg2rad(30.) ** 2); cr = (1 - q) ** 2 / (2 * q); ct = (1 - q ** 2) / (2 * q)
X = b['alpha']; F, Am, gr = g['pn_eval'](U, V, X, y, sn2, cr, ct)
mu0 = MO.jac(U, V, X)[0]; m0 = mu0[:, :256] + 1j * mu0[:, 256:]; e0 = np.exp(-1j * np.angle(m0)); a2 = np.abs(m0) ** 2
vr = sn2 + cr * a2; vt = sn2 + ct * a2
def Ffix(U, V, X):
    mm = MO.mu(U, V, X).reshape(n, 256); r = (y - mm) * e0
    return 0.5 * (r.real ** 2 / vr + r.imag ** 2 / vt).sum(1)
p0 = np.concatenate([U, V, X.real, X.imag], 1); err = 0
for j in range(4 * K):
    pp = p0.copy(); pp[:, j] += e; pm = p0.copy(); pm[:, j] -= e
    f = lambda p: Ffix(p[:, :K], p[:, K:2*K], p[:, 2*K:3*K] + 1j * p[:, 3*K:])
    fd = -(f(pp) - f(pm)) / (2 * e)
    err = max(err, float(np.abs(fd - gr[:, j]).max() / (np.abs(gr[:, j]).max() + 1e-12)))
print('(3) PN-ML Gauss-Newton gradient (frozen weights) max relative error vs finite differences:', err)
