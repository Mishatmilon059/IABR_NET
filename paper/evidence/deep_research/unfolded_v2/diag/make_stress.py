"""Prior-violation test banks (task 3): L=3, ideal hardware, same generator as the benchmark except the minimum separation.
nosep : DG.generate_points(3, 0)        (independent uniform angles), seed 9500
sep10 : DG.generate_points(3, 10 deg)   (violates the benchmark's pi/6 = 30 deg rule), seed 9501
SNR -10, -5, 0, 5 dB (where the separation rule matters), 250 per SNR (1000 scenes each). Saved to the C: cache.
stdout -> make_stress.log"""
import os
os.environ['IABR2_SUITE'] = 'D:/ai_ml_project'
import sys; sys.path.insert(0, 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2')
from dcommon import *
import iabr2_sim as S
SNRS = np.array([-10, -5, 0, 5])
for name, seed, sep in [('nosep', 9500, 0.0), ('sep10', 9501, np.deg2rad(10))]:
    rng = np.random.default_rng(seed)
    snr = np.repeat(SNRS, 250).astype(np.float64); B = len(snr)
    psi, phi, alpha = S.sample_paths(np.full(B, 3), rng, sep=sep)
    Y = S.simulate(psi, phi, alpha, snr, rng)['Y']
    Ht = np.einsum('nq,bqp,mp->bnm', S.W_IDEAL, Y, np.conj(S.F_IDEAL))
    np.savez(f'{BIG}/bank_{name}.npz', Ht=Ht, psi=psi, phi=phi, alpha=alpha, snr=snr, L=np.full(B, 3))
    Ut, Vt = np.pi * np.cos(psi), np.pi * np.cos(phi)
    r = residual_energy(Ht, Ut, Vt) / 250 / sigma2(snr)
    d = sep_violation(Ut, Vt)
    print(name, B, 'scenes; residual at truth / ((256-6) sigma2) mean', r.mean().round(4),
          '| frac of scenes violating pi/6:', (d < SEP).mean().round(3), '| < 10 deg:', (d < np.deg2rad(10)).mean().round(3),
          '| < 2 deg:', (d < np.deg2rad(2)).mean().round(3))
