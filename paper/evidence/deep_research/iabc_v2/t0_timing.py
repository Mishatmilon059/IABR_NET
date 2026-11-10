"""Timing / sanity probes used to size the experiments (no results used in claims except the codebook-unitarity check).
Seeds: 61000-61003 (timing only)."""
import time
from v2common import *

W, F = S.W_IDEAL, S.F_IDEAL
print('W W^H = c I ? max|W W^H - I*mean diag| =', np.abs(W @ W.conj().T - np.eye(16) * np.real(np.diag(W @ W.conj().T)).mean()).max(),
      ' diag mean', np.real(np.diag(W @ W.conj().T)).mean())
print('F F^H = c I ? max|F F^H - I*mean diag| =', np.abs(F @ F.conj().T - np.eye(16) * np.real(np.diag(F @ F.conj().T)).mean()).max(),
      ' diag mean', np.real(np.diag(F @ F.conj().T)).mean())
print('W^H W diag mean', np.real(np.diag(W.conj().T @ W)).mean(), 'max offdiag', np.abs(W.conj().T @ W - np.diag(np.diag(W.conj().T @ W))).max())
# check antenna-domain model: to_ant(Y_noiseless) == conj(D_r) H D_t
d = make_scenes(61000, 50, 3, 300.0, 20, 4)
Ht = to_ant(d['Y']); Hm = np.conj(d['D_r'])[:, :, None] * d['H'] * d['D_t'][:, None, :]
print('to_ant(Y) vs conj(D_r) H D_t (SNR 300 dB): max rel diff', np.abs(Ht - Hm).max() / np.abs(Hm).max())
n = np.arange(16)
u, v = true_uv(d['psi'][:1], d['phi'][:1])
Hs = (d['alpha'][0][None, None, :3] * np.exp(-1j * n[:, None, None] * u[0][None, None] + 1j * n[None, :, None] * v[0][None, None])).sum(-1)
print('H[n,m] == sum_l alpha_l exp(-j n u_l + j m v_l) (u = pi cos psi): max diff', np.abs(Hs - d['H'][0]).max())

rng = np.random.default_rng(61001); cfg = dict(train_L=(1, 9), train_snr=(-15, 24), clean_frac=0.2, train_phase_max=45, train_gain_max=6, grid=32, heat_sigma=1.0)
t = time.time(); b = S.make_batch(rng, 256, cfg); print(f'make_batch(256): {time.time() - t:.2f}s')
t = time.time(); Ls = rng.integers(1, 10, 2000); S.sample_paths(Ls, rng, lmax=9); print(f'sample_paths 2000 (L 1..9): {time.time() - t:.2f}s')
for L in [3, 9]:
    d = make_scenes(61002, 400, L, 10.0, 20, 4); Ht = to_ant(d['Y'])
    t = time.time(); U, V = C.nomp_greedy(Ht, L, 4, 3); tn = time.time() - t
    t = time.time(); selfcal_ret(Ht, U, V, 1.0, 0.0, iters=4); ts = time.time() - t
    print(f'L={L}: nomp_greedy 400 scenes {tn:.1f}s, selfcal 4 it {ts:.1f}s')
t = time.time(); net(d['Y']); print(f'CNN 400 scenes {time.time() - t:.2f}s')
