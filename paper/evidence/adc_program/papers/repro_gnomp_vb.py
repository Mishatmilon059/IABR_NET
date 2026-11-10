"""Reproduce GNOMP paper (Zhu et al., IEEE TAES 2026; arXiv 2307.00491v4) Sec. V-B: SNR loss (eq. 30 of the v4 PDF) and the
single-sinusoid Rao-test detection probability with a known non-zero threshold zeta = a(w1) x1 (Fig. 5).
Paper settings (v4 PDF p.24-25): N = 1024, x1 = 2, time-domain SNR of x1 = 6 dB, w1 = pi/2 (scenario 1) or pi/2 + 0.1 (scenario 2),
PFA = 0.01, 10^4 MC trials, quantiser full scale gamma = max(|x_k|, 3 sigma / sqrt 2) (Sec. V, p.24).
NOT stated in the paper (our assumptions): sigma^2 (main run: sigma^2 = 4 / 10^0.6 so that x1 has exactly 6 dB; sensitivity run
sigma^2 = 1), the target frequency w (we use an on-grid w = 2 pi * 300 / 1024, i.e. beta_g* = 1), the target phase (uniform random),
and that the 'measured' Pd is the Rao test (eq. 36 / 18) evaluated with the true zeta (oracle threshold, as in Sec. III).
Detection = argmax over the N-point DFT grid is the true bin AND T > tau_th (eq. 44 / 45).
Usage: python repro_gnomp_vb.py [n_trials]   (paper: 10000)"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
import sys
import numpy as np
from scipy.stats import norm, ncx2


def thresholds(B, gamma):
    """tau_0 = -inf, tau_d = d 2 gamma / b - gamma (d = 1..b-1), tau_b = +inf  (eq. 3, b = 2^B)"""
    b = 2 ** B
    t = np.array([d * 2 * gamma / b - gamma for d in range(1, b)])
    return np.concatenate([[-np.inf], t, [np.inf]])


def quant_cells(x, tau):
    """cell (lo, hi) of the uniform quantiser containing x"""
    d = np.clip(np.searchsorted(tau, x, side='left') - 1, 0, len(tau) - 2)
    return tau[d], tau[d + 1]


def hB(x, s, tau):
    """eq. (8) of the v4 PDF: sum_d [phi((t_{d+1}-x)/s) - phi((t_d-x)/s)]^2 / [Phi((t_{d+1}-x)/s) - Phi((t_d-x)/s)], s = sigma/sqrt2"""
    out = np.zeros(np.shape(x), dtype=float)
    for d in range(len(tau) - 1):
        a = (tau[d] - x) / s; c = (tau[d + 1] - x) / s
        num = (norm.pdf(c) - norm.pdf(a)) ** 2
        den = norm.cdf(c) - norm.cdf(a)
        out += np.where(den > 1e-300, num / np.maximum(den, 1e-300), 0.0)
    return out


def phi_B(lo, hi, mu, s):
    """one real component of eq. (16): -(phi(u') - phi(l')) / (Phi(u') - Phi(l')), u' = (u - mu) / s"""
    a = (lo - mu) / s; c = (hi - mu) / s
    den = np.maximum(norm.cdf(c) - norm.cdf(a), 1e-300)
    return -(norm.pdf(c) - norm.pdf(a)) / den


def snr_loss_db(N, zeta, s, tau):
    """eq. (30): SNR_loss = N / (a^H diag(h_{B,+}(zeta)) a) = N / sum_n h_{B,+}(zeta_n) for |a_n| = 1; h_+ from eq. (7a)"""
    hp = 0.5 * (hB(zeta.real, s, tau) + hB(zeta.imag, s, tau))
    return 10 * np.log10(N / hp.sum())


def run(w1, sigma2, label, ntr=10000, chunk=100, seed=12000):
    N = 1024; n = np.arange(N); x1 = 2.0
    s = np.sqrt(sigma2 / 2)
    zeta = x1 * np.exp(1j * w1 * n)
    gamma0 = max(abs(x1), 3 * np.sqrt(sigma2) / np.sqrt(2))
    PFA = 0.01
    tau_th = -2 * np.log(1 - (1 - PFA) ** (1 / N))                     # eq. (44)
    g_true = 300; w = 2 * np.pi * g_true / N; a = np.exp(1j * w * n)
    print(f'--- {label}: sigma^2={sigma2:.4f}, gamma={gamma0:.4f}, tau_th={tau_th:.3f} '
          f'(10log10 = {10 * np.log10(tau_th):.2f} dB, 20log10 = {20 * np.log10(tau_th):.2f} dB)', flush=True)
    res = {}
    for B in [1, 2, 3, None]:
        tag = 'inf' if B is None else str(B)
        snr_int = np.arange(8.0, 26.001, 0.01)
        amp = np.sqrt(10 ** (snr_int / 10) * sigma2 / N)
        if B is None:
            loss = 0.0
            lam = 2 / sigma2 * N * amp ** 2                                   # eq. (27)
        else:
            tau = thresholds(B, gamma0)
            loss = snr_loss_db(N, zeta, s, tau)
            Sp = 0.5 * (hB(zeta.real, s, tau) + hB(zeta.imag, s, tau)).sum()
            lam = 2 / sigma2 * Sp * amp ** 2                                  # eq. (38) with beta = 1; the h_- term averages to 0
        pd_th = ncx2.sf(tau_th, 2, lam)                                       # eq. (45): Q1(sqrt(lam), sqrt(tau_th))
        snr50_th = float(np.interp(0.5, pd_th, snr_int))
        rng = np.random.default_rng([seed, 0 if B is None else B, int(round(w1 * 1000)), int(round(sigma2 * 1000))])
        snr_mc = np.arange(10.0, 24.001, 1.0); pd_mc = []
        for si in snr_mc:
            A = np.sqrt(10 ** (si / 10) * sigma2 / N); det = 0
            if B is not None:
                gam = max(abs(x1), A, 3 * np.sqrt(sigma2) / np.sqrt(2))     # Sec. V quantiser design rule
                tq = thresholds(B, gam)
                hp_ = 0.5 * (hB(zeta.real, s, tq) + hB(zeta.imag, s, tq)); hm_ = 0.5 * (hB(zeta.real, s, tq) - hB(zeta.imag, s, tq))
                Sp_ = hp_.sum()
                Sm_all = np.conj(np.fft.fft(hm_))[(2 * np.arange(N)) % N]   # a^T(w_g) diag(h_-) a(w_g) = sum_n h_-,n e^{+j 2 w_g n}
            for c0 in range(0, ntr, chunk):
                m = min(chunk, ntr - c0)
                ph = rng.uniform(0, 2 * np.pi, m)
                z = zeta[None] + (A * np.exp(1j * ph))[:, None] * a[None] + s * (rng.standard_normal((m, N)) + 1j * rng.standard_normal((m, N)))
                if B is None:
                    T = (2 / sigma2) / N * np.abs(np.fft.fft(z - zeta[None], axis=1)) ** 2          # eq. (26)
                else:
                    lr, hr = quant_cells(z.real, tq); li, hi_ = quant_cells(z.imag, tq)
                    phi = phi_B(lr, hr, zeta.real[None], s) + 1j * phi_B(li, hi_, zeta.imag[None], s)   # eq. (16)
                    AH = np.fft.fft(phi, axis=1)                                 # a^H(w_g) phi
                    T = (Sp_ * np.abs(AH) ** 2 - np.real(Sm_all[None] * AH ** 2)) / (Sp_ ** 2 - np.abs(Sm_all[None]) ** 2)  # eq. (18)
                det += int(((T.argmax(1) == g_true) & (T.max(1) > tau_th)).sum())
            pd_mc.append(det / ntr)
        pd_mc = np.array(pd_mc)
        snr50_mc = float(np.interp(0.5, pd_mc, snr_mc)) if (pd_mc.max() >= 0.5 and pd_mc.min() <= 0.5) else float('nan')
        res[tag] = (loss, snr50_th, snr50_mc)
        print(f'B={tag:>3}: SNR loss eq.(30) = {loss:5.2f} dB | Pd=0.5 at integrated SNR: theory eq.(45) {snr50_th:5.2f} dB, '
              f'MC {snr50_mc:5.2f} dB | MC Pd at {snr_mc[0]:.0f}..{snr_mc[-1]:.0f} dB: ' + ' '.join(f'{p:.3f}' for p in pd_mc), flush=True)
    for t in ['1', '2', '3']:
        print(f'  B={t}: MC loss vs unquantised at Pd=0.5: {res[t][2] - res["inf"][2]:.2f} dB; theory-curve loss: {res[t][1] - res["inf"][1]:.2f} dB')
    return res


if __name__ == '__main__':
    ntr = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    print(f'n_trials per point = {ntr}')
    print('PAPER (v4 PDF p.24-25): scenario 1: eq.(30) losses 4.8 / 2.3 / 1 dB; measured Pd=0.5 at 17.2 / 14.7 / 13.3 / 12.4 dB '
          '(1/2/3/inf bits) -> 4.8 / 2.3 / 0.9 dB. Scenario 2: eq.(30) 6.4 / 2.2 / 0.9 dB; measured 18.7 / 14.6 / 13.2 / 12.4 -> 6.3 / 2.2 / 0.8 dB')
    s2 = 4 / 10 ** 0.6
    run(np.pi / 2, s2, 'scenario 1 (w1 = pi/2), x1 = 2 at exactly 6 dB', ntr)
    run(np.pi / 2 + 0.1, s2, 'scenario 2 (w1 = pi/2 + 0.1), x1 = 2 at exactly 6 dB', ntr)
    print('=== sensitivity: sigma^2 = 1 (x1 at 6.02 dB)')
    run(np.pi / 2, 1.0, 'scenario 1, sigma^2 = 1', ntr)
    run(np.pi / 2 + 0.1, 1.0, 'scenario 2, sigma^2 = 1', ntr)
    print('=== zero threshold (zeta = 0: first detection, weak target, gamma = 3 sigma / sqrt 2): eq.(30) loss per bit')
    for B in [1, 2, 3, 4]:
        tau = thresholds(B, 3 / np.sqrt(2))
        print(f'B={B}: {snr_loss_db(1, np.zeros(1, complex), np.sqrt(0.5), tau):.2f} dB'
              + (f'  (paper: pi/2 = {10 * np.log10(np.pi / 2):.2f} dB, eq. 35)' if B == 1 else ''))
