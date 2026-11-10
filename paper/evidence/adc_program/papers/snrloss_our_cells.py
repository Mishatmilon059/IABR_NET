"""GNOMP theory (Zhu et al., IEEE TAES 2026, arXiv 2307.00491v4, Sec. III: eq. (7a), (8), (29)-(30) of the v4 PDF) applied to OUR
ADC cells. For each path k of a scene, the other L-1 paths are treated as perfectly known 'thresholds' zeta_{\\k} (the paper's oracle
view) and the weak-signal Rao-test SNR loss of path k is
    SNR_loss_k = ||a_k||^2 / (a_k^H diag(h_{B,+}(zeta_{\\k})) a_k),
with a_k the path's beamspace atom (vec of W^H sqrt(256) a_r a_t^H F) and h_{B,+} from the paper's eq. (7a)/(8). This is the
paper's eq. (30) generalised from |a_n| = 1 to a general atom (the paper's own Sec. V-H replaces a by Phi a); this generalisation is OURS.
Quantiser: our peak AGC (stress_sim.adc): gamma = A = max over the scene's 256 noisy beam samples of max(|Re|, |Im|), uniform b-bit,
thresholds d*2A/2^b - A; extreme cells treated as open (paper's eq. 3 convention; with peak AGC no sample falls outside +-A).
Noise: complex variance sigma^2 = 10^(-SNR/10) (stress_sim), i.e. per-component std sigma/sqrt2 = the paper's convention.
Integrated SNR of path k = ||G_k||^2 / sigma^2 (paper Sec. V: N|x|^2/sigma^2); effective = integrated - loss (paper p.12, PDF v4).
Seeds: validation range 81,000,000+ (analysis only; no decoder is evaluated here). 1000 scenes per cell, L = 3."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
import sys
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/stress')
import numpy as np
from scipy.stats import norm
import stress_sim as SS


def hB(x, s, tau):
    out = np.zeros(np.shape(x), dtype=float)
    for d in range(tau.shape[-1] - 1):
        a = (tau[..., d] - x) / s; c = (tau[..., d + 1] - x) / s
        den = norm.cdf(c) - norm.cdf(a)
        out += np.where(den > 1e-300, (norm.pdf(c) - norm.pdf(a)) ** 2 / np.maximum(den, 1e-300), 0.0)
    return out


def main(n=1000, chunk=100):
    W, F = SS.W_IDEAL, SS.F_IDEAL
    print('unitarity check ||W^H W - I||, ||F^H F - I||:', np.abs(W.conj().T @ W - np.eye(16)).max(), np.abs(F.conj().T @ F - np.eye(16)).max())
    nn = np.arange(16.)
    print('cell | path rank (1 = strongest) | median integrated SNR (dB) | SNR loss (dB): median [10%, 90%] | median effective SNR (dB) | frac effective < 12 dB')
    for snr in [15.0, 0.0]:
        for bits in [1, 2, 3]:
            rows = {k: dict(isnr=[], loss=[]) for k in range(3)}
            for c0, seed in zip(range(0, n, chunk), range(81000000 + int(snr) * 1000 + bits * 100, 10 ** 9)):
                m = min(chunk, n - c0)
                base = SS.base_scenes(m, 3, snr, seed)
                ps, ph, al = base['psi'], base['phi'], base['alpha']
                a_r = np.exp(-1j * np.pi * np.cos(ps)[..., None] * nn) / 4
                a_t = np.exp(-1j * np.pi * np.cos(ph)[..., None] * nn) / 4
                Gk = 16 * np.einsum('nq,bkn,bkm,mp->bkqp', W.conj(), a_r, a_t.conj(), F)        # unit-gain beamspace atom of each path
                Gp = al[:, :, None, None] * Gk                                              # path contributions
                Y = Gp.sum(1) + base['noise']
                assert np.allclose(Y, SS.generate(base, 'clean'))
                A = np.maximum(np.abs(Y.real), np.abs(Y.imag)).max((1, 2))
                b = 2 ** bits
                tau = np.concatenate([np.full((m, 1), -np.inf), np.stack([d * 2 * A / b - A for d in range(1, b)], 1), np.full((m, 1), np.inf)], 1)
                sigma2 = 10 ** (-snr / 10); s = np.sqrt(sigma2 / 2)
                for k in range(3):
                    zeta = Gp.sum(1) - Gp[:, k]
                    t = tau[:, None, None, :]
                    hp = 0.5 * (hB(zeta.real, s, t) + hB(zeta.imag, s, t))
                    ak2 = np.abs(Gk[:, k]) ** 2
                    loss = 10 * np.log10(ak2.sum((1, 2)) / (hp * ak2).sum((1, 2)))
                    isnr = 10 * np.log10((np.abs(Gp[:, k]) ** 2).sum((1, 2)) / sigma2)
                    rows[k]['isnr'] += list(isnr); rows[k]['loss'] += list(loss)
            for k in range(3):
                i = np.array(rows[k]['isnr']); l = np.array(rows[k]['loss']); e = i - l
                print(f'{bits}-bit/{snr:.0f} dB | {k + 1} | {np.median(i):6.2f} | {np.median(l):5.2f} [{np.quantile(l, .1):5.2f}, {np.quantile(l, .9):5.2f}] '
                      f'| {np.median(e):6.2f} | {np.mean(e < 12):.3f}', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
