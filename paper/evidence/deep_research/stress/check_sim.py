"""Simulator checks (stdout -> check_sim.log).
 1. impairment-free bank from stress_sim with the v1 seeds reproduces the stored phase_d0_snr0 / phase_d0_snr15 banks (and M3 with
    phase 5 deg reproduces phase_d5_snr0, gain 4 dB reproduces gain_g4_snr15) -- Y16 is stored as float32, so agreement is up to float32 rounding;
 2. stress_sim.generate('clean') / ('M3') is bit-identical to iabr2_sim.simulate with the same rng;
 3. zero-severity limits: M4 b=4 == clean exactly; M1 rho=0, M2 sigma=0, M6 sigma=0, M8 r=1e9 m, M5 b=20 -> clean;
 4. codebook quantisation tables (phase index of row 1, rank of the quantised codebooks);
 5. CFAR threshold (rule (a) of M7) on noise-only scenes: realised false-alarm rate of max over the 16x16 grid and over the 4x oversampled grid;
 6. time of one 500-scene bank through all six methods."""
import time
from scommon import *

GEN = 'D:/ai_ml_project/IABR_Net_TestSuite/data/generated_banks'


def ld(name):
    z = np.load(f'{GEN}/{name}.npz'); return z['Y16'][..., 0].astype(np.float64) + 1j * z['Y16'][..., 1], z


print('1. reproduction of stored v1 banks (seed, L=3, n=500)')
for name, seed, snr, kind, sev in [('phase_d0_snr0', 1000, 0, 'clean', None), ('phase_d0_snr15', 1015, 15, 'clean', None),
                                   ('phase_d5_snr0', 1050, 0, 'M3', (5.0, 0.0)), ('gain_g4_snr15', 2055, 15, 'M3', (0.0, 4.0))]:
    Yb, z = ld(name)
    assert int(z['seed']) == seed, (name, int(z['seed']))
    b = SS.base_scenes(500, 3, snr, seed); Y = SS.generate(b, kind, sev)
    print(f'   {name:16s} seed {seed}: max|Y - stored| = {np.abs(Y - Yb).max():.2e} (max|Y| {np.abs(Yb).max():.1f}); '
          f'max|psi diff| {np.abs(b["psi"] - z["psi"]).max():.1e}, max|phi diff| {np.abs(b["phi"] - z["phi"]).max():.1e}')

print('2. bit-identity with iabr2_sim.simulate (same rng)')
for kind, sev in [('clean', None), ('M3', (30.0, 4.0))]:
    rng = np.random.default_rng(77); ps, ph, al = SS.S.sample_paths([3] * 64, rng)
    pdm, gdm = sev if sev else (0.0, 0.0)
    Ys = SS.S.simulate(ps, ph, al, np.full(64, 5.0), rng, phase_deg_max=pdm, gain_db_max=gdm)['Y']
    Ym = SS.generate(SS.base_scenes(64, 3, 5.0, 77), kind, sev)
    print(f'   {kind} {sev}: max|diff| = {np.abs(Ys - Ym).max():.1e}, identical: {np.array_equal(Ys, Ym)}')

print('3. zero-severity limits (base seed 5, n=64, SNR 15)')
b = SS.base_scenes(64, 3, 15.0, 5); Y0 = SS.generate(b, 'clean')
for kind, sev in [('M4', 4), ('M1', 0.0), ('M2', 0.0), ('M6', 0.0), ('M8', 1e9), ('M5', 20), ('M5c', 20)]:
    d = np.abs(SS.generate(b, kind, sev) - Y0).max()
    print(f'   {kind} sev={sev}: max|Y - Y_clean| = {d:.2e}' + ('  (exactly identical)' if d == 0 else ''))

print('4. quantised codebooks')
for bits in [4, 3, 2, 1]:
    Wq = SS.quantise_codebook(SS.W_IDEAL, bits); Fq = SS.quantise_codebook(SS.F_IDEAL, bits)
    kw = np.round(np.angle(Wq * 4) / (2 * np.pi / 16)).astype(int) % 16
    print(f'   b={bits}: W row n=1 phase index (x 2pi/16): {kw[1].tolist()}; rank W {np.linalg.matrix_rank(Wq)}, rank F {np.linalg.matrix_rank(Fq)}; '
          f'||Wq - W||_F/||W||_F = {np.linalg.norm(Wq - SS.W_IDEAL) / np.linalg.norm(SS.W_IDEAL):.3f}; max distinct-column count W {len({tuple(np.round(c, 9)) for c in Wq.T})}')

print('5. CFAR threshold on noise-only scenes (rule (a) of M7)')
N, Pfa = 256, 1e-2
kappa = -np.log(1 - (1 - Pfa) ** (1 / N))
print(f'   tau / sigma^2 = -ln(1 - (1-Pfa)^(1/N)) = {kappa:.4f}  (N=256, Pfa=1e-2)')
rng = np.random.default_rng(123); M = 20000
for snr in [0, 15]:
    s2 = 10 ** (-snr / 10)
    Z = np.sqrt(s2 / 2) * (rng.standard_normal((M, 16, 16)) + 1j * rng.standard_normal((M, 16, 16)))
    H = SS.to_ant(Z)
    g1 = (np.abs(np.fft.fft2(H)) ** 2).reshape(M, -1).max(1) / 256
    g4 = (np.abs(np.fft.fft2(H, s=(64, 64))) ** 2).reshape(M, -1).max(1) / 256
    print(f'   SNR {snr:2d}: noise var in Hant {np.mean(np.abs(H) ** 2):.4f} (sigma^2 {s2:.4f}); realised Pfa 16x16 grid {np.mean(g1 > kappa * s2):.4f}, '
          f'4x oversampled grid {np.mean(g4 > kappa * s2):.4f}  ({M} noise-only scenes)')

print('6. timing, one 500-scene bank (L=3, SNR 15, clean) through all six methods')
b = SS.base_scenes(500, 3, 15.0, 9); Y = SS.generate(b, 'clean')
t0 = time.time(); out, heat, Hc = run_methods(Y, b['psi'], b['phi'], 3); print(f'   {time.time() - t0:.1f} s')
for m in METHODS:
    S_ = stats(out[m], b['psi'], b['phi'], 3); r = C.metric_from_stats(S_, np.arange(500))
    print(f'   {m:15s} Pd {r["pd_paper"]:.4f} strict {r["pd_strict"]:.4f} rmse {r["rmse"]:.4f}')
