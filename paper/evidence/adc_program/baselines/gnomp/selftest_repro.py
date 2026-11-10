"""Unit checks of gnomp_core and the reproduction of the paper's 1-D example (Zhu et al. 2026, Sec. V-C, Fig. 7).
Setting taken from the paper (P) Sec. V-C: K = 2, omega = 2.2 and 2.4, x1 = -1.505-0.497j, x2 = -0.164-0.609j,
N = 128, B = 1 (sign quantiser, threshold 0), P_FA = 0.01 (threshold 25.5 dB = 20 log10 tau_th). Not stated in the
paper and set here: sigma^2 = 1 (complex; the example's time-domain SNRs are then +4.4 and -4.0 dB, matching the
quoted +4 / -4 dB). The paper's Fig. 7 is a single noise realisation (not given), so the reproduction is statistical:
300 independent noise draws, the same amplitudes and frequencies.
Run: python selftest_repro.py  (stdout -> selftest_repro.log). Numpy only, small memory."""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gnomp_core as G

rng = np.random.default_rng(20260401)
print('== unit checks', flush=True)

# (1) one-bit Fisher information at the threshold: paper (P) text after eq. (12): h_B = 2/pi at mu on the threshold
s0 = 0.5
thr1 = np.array([[-np.inf, 0.0, np.inf]])
h0 = G.expected_info(np.zeros((1, 1)), thr1, np.array([[s0]]))[0, 0]
print(f'  h_B(mu=0, 1 bit) * s^2 = {h0 * s0 ** 2:.6f} (expected 2/pi = {2 / np.pi:.6f})  {"OK" if abs(h0 * s0 ** 2 - 2 / np.pi) < 1e-9 else "FAIL"}')

# (2) 1-D Jacobian vs finite differences
M1 = G.Atoms1D(16); B = 2; K = 2
Z = np.zeros((B, K, 3)); Z[..., 0] = rng.uniform(0, 6, (B, K)); Z[..., 1:] = rng.normal(size=(B, K, 2))
valid = np.ones((B, K), bool)
lo = rng.normal(size=(B, 32)) - 1; hi = lo + 2.0; s = np.full((B, 1), 0.7)
st = G.state(M1, Z, valid, lo, hi, s); J = G.jacobian(M1, st)
eps = 1e-6; num = np.zeros_like(J)
for c in range(K * 3):
    Zp = Z.reshape(B, -1).copy(); Zp[:, c] += eps; Zm = Z.reshape(B, -1).copy(); Zm[:, c] -= eps
    mp = G.state(M1, Zp.reshape(Z.shape), valid, lo, hi, s)['mu']; mm = G.state(M1, Zm.reshape(Z.shape), valid, lo, hi, s)['mu']
    num[:, :, c] = (np.concatenate([(mp - mm).real, (mp - mm).imag], 1)) / (2 * eps)
print(f'  1-D Jacobian max |analytic - numeric| = {np.abs(J - num).max():.2e}  {"OK" if np.abs(J - num).max() < 1e-6 else "FAIL"}')

# (3) Rao statistic with unit-modulus atoms and equal information: T = |c|^2 / S+  (paper eq. (20))
psi = rng.normal(size=(1, 16)) + 1j * rng.normal(size=(1, 16)); hh = np.full((1, 16), 0.3)
Ag = M1.atoms(np.linspace(0, 2 * np.pi, 7)[1:-1, None])[0]
c, Sp, D = G.rao(psi, hh, hh, Ag)
T = G.rao_T(c, Sp, D)
Tp = np.abs(c) ** 2 / Sp
print(f'  Rao T vs |c|^2/S+ (equal h, zero D): max diff {np.abs(T - Tp).max():.2e}  {"OK" if np.abs(T - Tp).max() < 1e-9 else "FAIL"}')

# (4) 2-D atoms vs the mlq.Model used by quantiser ML (identical beamspace model)
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/stress')
sys.path.insert(0, 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2')
os.environ.setdefault('IABR2_SUITE', 'D:/ai_ml_project')
import stress_sim as SS
import mlq
A2 = G.Atoms2D(SS.W_IDEAL, SS.F_IDEAL)
th2 = rng.uniform(-3, 3, (2, 2, 2)); x2 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
a2, D2 = A2.atoms(th2)
mo = mlq.Model(SS.W_IDEAL, SS.F_IDEAL, None)
mu_ref = mo.mu(th2[..., 0], th2[..., 1], x2).reshape(2, 256)
mu_mine = np.einsum('bk,bkm->bm', x2, a2)
print(f'  2-D beamspace atoms vs mlq.Model.mu: max diff {np.abs(mu_ref - mu_mine).max():.2e}  {"OK" if np.abs(mu_ref - mu_mine).max() < 1e-9 else "FAIL"}')
# (5) 2-D Jacobian vs finite differences
Z2 = np.zeros((2, 2, 4)); Z2[..., :2] = th2; Z2[..., 2] = x2.real; Z2[..., 3] = x2.imag; v2 = np.ones((2, 2), bool)
lo2 = rng.normal(size=(2, 512)) - 1; hi2 = lo2 + 2; s2 = np.full((2, 1), 0.7)
st2 = G.state(A2, Z2, v2, lo2, hi2, s2); J2 = G.jacobian(A2, st2); num2 = np.zeros_like(J2); eps = 1e-6
for c in range(8):
    Zp = Z2.reshape(2, -1).copy(); Zp[:, c] += eps; Zm = Z2.reshape(2, -1).copy(); Zm[:, c] -= eps
    dm = G.state(A2, Zp.reshape(Z2.shape), v2, lo2, hi2, s2)['mu'] - G.state(A2, Zm.reshape(Z2.shape), v2, lo2, hi2, s2)['mu']
    num2[:, :, c] = np.concatenate([dm.real, dm.imag], 1) / (2 * eps)
print(f'  2-D Jacobian max |analytic - numeric| = {np.abs(J2 - num2).max():.2e}  {"OK" if np.abs(J2 - num2).max() < 1e-6 else "FAIL"}')

# (6) the paper's example: 1-D, N = 128, sign quantiser
print('== reproduction: paper Sec. V-C example (1-D, N = 128, B = 1, P_FA = 0.01)', flush=True)
N = 128; ws = np.array([2.2, 2.4]); xs = np.array([-1.505 - 0.497j, -0.164 - 0.609j]); sig2 = 1.0; sc = np.sqrt(sig2 / 2)
M = G.Atoms1D(N); grid = M.grid(4)
tau = G.tau_th(N, 0.01)
print(f'  tau_th(N=128, P_FA=0.01) = {tau:.4f}  -> 20 log10 = {20 * np.log10(tau):.2f} dB (paper: 25.5 dB)')
T_ = 300
n_ = np.arange(N)
a_true = np.exp(1j * n_[None, :] * ws[:, None])               # (2,N)
zc = (a_true.T @ xs)                                          # (N,)
noise = sc * (rng.standard_normal((T_, N)) + 1j * rng.standard_normal((T_, N)))
Yz = zc[None, :] + noise
lo = np.where(Yz.real > 0, 0.0, -G.BIG); hi = np.where(Yz.real > 0, G.BIG, 0.0)
loI = np.where(Yz.imag > 0, 0.0, -G.BIG); hiI = np.where(Yz.imag > 0, G.BIG, 0.0)
lo_all = np.concatenate([lo, loI], 1); hi_all = np.concatenate([hi, hiI], 1)
thr = np.tile(np.array([[-np.inf, 0.0, np.inf]]), (T_, 1)); s_all = np.full((T_, 1), sc)
t0 = time.time()
Z, valid, info = G.gnomp(M, grid, lo_all, hi_all, s_all, thr, N, mode='cfar', Kmax=6, spurious=True)
el = time.time() - t0
K_hat = valid.sum(1)
print(f'  trials {T_}, GNOMP-CFAR time {el:.1f} s total ({1e3 * el / T_:.1f} ms/trial, 2 threads, batched)')
vals, cnt = np.unique(K_hat, return_counts=True)
print('  K_hat distribution (paper: 2 detections then stop):', dict(zip(vals.tolist(), cnt.tolist())))
print(f'  fraction K_hat == 2: {np.mean(K_hat == 2):.3f}')
both = []; errs = []
for i in range(T_):
    f_est = np.mod(Z[i, valid[i], 0], 2 * np.pi)
    if len(f_est) < 2:
        continue
    # match each true frequency to the nearest estimate
    e = [np.min(np.abs(np.angle(np.exp(1j * (f_est - w))))) for w in ws]
    both.append(max(e) < 0.2); errs.append(e)
errs = np.array(errs)
print(f'  trials with 2+ detections: {len(both)}; both true tones found within 0.2 rad: {np.mean(both):.3f}')
print(f'  median |omega_hat - omega| for the 2.2 / 2.4 tones: {np.median(errs[:, 0]):.4f} / {np.median(errs[:, 1]):.4f} rad '
      f'(paper example: 0.002 / 0.001)')
# first Rao statistic (zeta = 0), N-bin grid, in the paper's dB convention 20 log10 T
psi0, h0R, h0I, _ = G.score_info(np.zeros((T_, N), complex), lo_all, hi_all, s_all, thr)
c0, Sp0, D0 = G.rao(psi0, h0R, h0I, grid[1][grid[2]])
T0 = G.rao_T(c0, Sp0, D0).max(1)
q = np.percentile(20 * np.log10(T0), [10, 50, 90])
print(f'  first Rao statistic 20 log10 T (zeta = 0): 10/50/90 % = {q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f} dB '
      f'(paper example, one realisation: 41.4 dB; its threshold 25.5 dB)')
print(f'  fraction of trials whose first test exceeds the threshold: {np.mean(T0 > tau):.3f}')
print('done', flush=True)
