"""GNOMP oracle detection bound (Zhu et al., IEEE TAES 2026: published eq. (34), (35), (39), (40); arXiv v4 Sec. III-C eq. (64)-(66)
product bound) applied to OUR four ADC cells (+ 3-bit for reference).  OUR extension of the paper's 1-D theory to the 2-D beamspace atom.

Same scenes and seeds as snrloss_our_cells.py (validation range 81,000,000+, 1000 scenes per cell, L = 3, peak AGC, open outer cells).
For path k, the other two paths are perfectly known 'thresholds' zeta_k (the paper's oracle view), and
    lambda_k = 2 * beta_k * (a_k^H diag(h_{B,+}(zeta_k)) a_k) |x_k|^2 / sigma^2          (published eq. (34), h_- term dropped as in eq. (31))
    Pd_k     = Q1(sqrt(lambda_k), sqrt(tau_th)) = ncx2.sf(tau_th, 2, lambda_k)              (published eq. (40))
    tau_th   = -2 ln(1 - (1 - P_FA)^(1/N)), N = 256 beam bins, P_FA = 0.01                  (published eq. (39))
beta_k: straddle factor of the nearest DFT bin; for our unitary DFT codebooks the beam samples ARE the 16x16 DFT bins, so
    beta_k = max_qp |G_k,qp|^2 / sum_qp |G_k,qp|^2   (2-D analogue of published eq. (35)); 'beta = 1' is the on-grid best case.
Reported per cell: mean Pd_k per path rank, mean over the 3 paths, and the mean of prod_k Pd_k (v4 eq. (66): an upper bound on
P(all 3 paths detected) in the oracle setting).  These are detection bounds for a CFAR detector, NOT bounds on our 1-degree Pd metric."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
import sys
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/stress')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import ncx2
import stress_sim as SS
from snrloss_our_cells import hB


def main(n=1000, chunk=100, pfa=0.01):
    W, F = SS.W_IDEAL, SS.F_IDEAL
    N = 256
    tau_th = -2 * np.log(1 - (1 - pfa) ** (1 / N))
    print(f'tau_th (N = {N}, P_FA = {pfa}) = {tau_th:.3f}')
    nn = np.arange(16.)
    print('cell | beta | Pd path1 | Pd path2 | Pd path3 | mean Pd over paths | mean prod Pd (all 3 detected, upper bound) | median beta per path')
    for snr in [15.0, 0.0]:
        for bits in [1, 2, 3]:
            pd = {b: [[], [], []] for b in ('1', 'g')}; betas = [[], [], []]
            for c0, seed in zip(range(0, n, chunk), range(81000000 + int(snr) * 1000 + bits * 100, 10 ** 9)):
                m = min(chunk, n - c0)
                base = SS.base_scenes(m, 3, snr, seed)
                ps, ph, al = base['psi'], base['phi'], base['alpha']
                a_r = np.exp(-1j * np.pi * np.cos(ps)[..., None] * nn) / 4
                a_t = np.exp(-1j * np.pi * np.cos(ph)[..., None] * nn) / 4
                Gk = 16 * np.einsum('nq,bkn,bkm,mp->bkqp', W.conj(), a_r, a_t.conj(), F)
                Gp = al[:, :, None, None] * Gk
                Y = Gp.sum(1) + base['noise']
                assert np.allclose(Y, SS.generate(base, 'clean'))
                A = np.maximum(np.abs(Y.real), np.abs(Y.imag)).max((1, 2))
                b = 2 ** bits
                tau = np.concatenate([np.full((m, 1), -np.inf), np.stack([d * 2 * A / b - A for d in range(1, b)], 1), np.full((m, 1), np.inf)], 1)
                sigma2 = 10 ** (-snr / 10); s = np.sqrt(sigma2 / 2)
                for k in range(3):
                    zeta = Gp.sum(1) - Gp[:, k]
                    hp = 0.5 * (hB(zeta.real, s, tau[:, None, None, :]) + hB(zeta.imag, s, tau[:, None, None, :]))
                    ak2 = np.abs(Gk[:, k]) ** 2
                    Sq = (hp * ak2).sum((1, 2))
                    beta = ak2.max((1, 2)) / ak2.sum((1, 2))
                    lam1 = 2 * Sq * np.abs(al[:, k]) ** 2 / sigma2
                    pd['1'][k] += list(ncx2.sf(tau_th, 2, lam1)); pd['g'][k] += list(ncx2.sf(tau_th, 2, beta * lam1))
                    betas[k] += list(beta)
            for tag, lab in (('1', 'beta=1 (on-grid best case)'), ('g', 'beta=actual nearest-bin')):
                P = np.array(pd[tag])
                print(f'{bits}-bit/{snr:.0f} dB | {lab} | ' + ' | '.join(f'{P[k].mean():.3f}' for k in range(3))
                      + f' | {P.mean():.3f} | {P.prod(0).mean():.3f} | ' + ' '.join(f'{np.median(betas[k]):.3f}' for k in range(3)), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
