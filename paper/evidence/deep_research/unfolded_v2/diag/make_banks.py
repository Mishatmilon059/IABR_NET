"""Simulate fresh ideal-hardware L=3 banks with the benchmark generator (iabr2_sim.sample_paths -> DG.generate_points(L, pi/6),
alpha ~ CN(0,1/L) sorted, iabr2_sim.simulate, Ht = W_IDEAL Y F_IDEAL^H as the notebook's to_ant).
train: seed 70,000,000, 500 per SNR (4000); val: seed 71,000,000, 250 per SNR (2000). SNR -10..25 step 5.
Saved to the C: cache (BIG). stdout -> make_banks.log"""
import os
os.environ['IABR2_SUITE'] = 'D:/ai_ml_project'
import sys; sys.path.insert(0, 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2')
from dcommon import *
import iabr2_sim as S
SNRS = np.arange(-10, 26, 5)
for name, seed, per in [('train', 70_000_000, 500), ('val', 71_000_000, 250)]:
    rng = np.random.default_rng(seed)
    snr = np.repeat(SNRS, per).astype(np.float64); B = len(snr)
    psi, phi, alpha = S.sample_paths(np.full(B, 3), rng)
    Y = S.simulate(psi, phi, alpha, snr, rng)['Y']
    Ht = np.einsum('nq,bqp,mp->bnm', S.W_IDEAL, Y, np.conj(S.F_IDEAL))
    np.savez(f'{BIG}/bank_{name}.npz', Ht=Ht, psi=psi, phi=phi, alpha=alpha, snr=snr, L=np.full(B, 3))
    Ut, Vt = np.pi * np.cos(psi), np.pi * np.cos(phi)
    r = residual_energy(Ht, Ut, Vt) / 250 / sigma2(snr)
    print(name, B, 'scenes; residual at truth / ((256-6) sigma2) mean', r.mean().round(4))
