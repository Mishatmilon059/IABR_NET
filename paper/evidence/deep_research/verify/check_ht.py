"""Step 3: check that the cached raw antenna-domain channel Ht (net__official.npz) equals to_ant(Y) recomputed from the frozen bank
(Y16 = downsample16(data) as in notebook load_bank; to_ant(Y) = W_IDEAL Y F_IDEAL^H), first 200 official scenes.
usage: python check_ht.py   (stdout -> check_ht.log)"""
import os, sys, json
os.environ['IABR2_SUITE'] = 'D:/ai_ml_project'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2')
from vcommon import *
import iabr2_sim as sim

NS = 200
fb = np.load(FROZEN)
Y16 = sim.downsample16(fb['data'][:NS])
Y = Y16[..., 0].astype(np.float64) + 1j * Y16[..., 1].astype(np.float64)
H = sim.W_IDEAL[None] @ Y @ np.conj(sim.F_IDEAL.T)[None]          # (NS,16,16)
Ht = np.load(f'{CACHE}/net__official.npz')['Ht'][:NS].astype(np.complex128)
err = np.abs(H - Ht)
rel = np.linalg.norm((H - Ht).reshape(NS, -1), axis=1) / np.linalg.norm(H.reshape(NS, -1), axis=1)
unit = float(np.abs(sim.W_IDEAL.conj().T @ sim.W_IDEAL - np.eye(16)).max()), float(np.abs(sim.F_IDEAL.conj().T @ sim.F_IDEAL - np.eye(16)).max())
print(f'scenes {NS}: max |to_ant(Y) - Ht| = {err.max():.3e}, max |Ht| = {np.abs(H).max():.3f}, max per-scene relative Frobenius error = {rel.max():.3e}, median {np.median(rel):.3e}')
print(f'codebook unitarity: max |W^H W - I| = {unit[0]:.2e}, max |F^H F - I| = {unit[1]:.2e}')
print('Ht is stored as complex64 (float32 CNN front end); a relative error of order 1e-7 is float32 rounding.')
print('PASS' if rel.max() < 1e-5 else 'FAIL')
json.dump(dict(n=NS, max_abs=float(err.max()), max_rel=float(rel.max()), median_rel=float(np.median(rel)), unitarity=unit), open(f'{HERE}/check_ht.json', 'w'), indent=1)
