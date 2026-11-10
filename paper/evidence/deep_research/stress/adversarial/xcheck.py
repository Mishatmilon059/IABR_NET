"""(1) Independent forward model (adv_common.forward, written from the spec) vs stress_sim.generate on the same base scenes and the SAME
standard random draws (reproduced from stress_sim.imp_rng). Agreement => the stress simulator implements the spec as I read it.
(2) The 'peak-to-RMS ~12' statement in RESULTS.md / stress_sim.py:137 (not computed in any logged script): measured here.
(3) Hermitian vs symmetric coupling: how much does the convention matter (M1 rho 0.3) -- reported only as a size of difference in Y.
usage: python xcheck.py > xcheck.log"""
from adv_common import *
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/stress')
import stress_sim as SS

print('(1) independent forward model vs stress_sim.generate (64 scenes, seed 4242, SNR 15)')
b = SS.base_scenes(64, 3, 15.0, 4242); B = 64
print(f'   to_ant max|diff| {np.abs(to_ant(b["noise"]) - SS.to_ant(b["noise"])).max():.2e}')
for kind, sevs in [('clean', [None]), ('M1', [0.1, 0.3]), ('M2', [0.05, 0.2]), ('M4', [4, 3, 2, 1]), ('M5', [4, 3, 2, 1]), ('M6', [10.0, 30.0]), ('M8', [2.0, 0.5])]:
    for sev in sevs:
        dr = {}
        if kind in ('M1', 'M2', 'M6'):
            g = SS.imp_rng(b['seed'], kind)
            if kind == 'M1':
                dr['t'] = g.uniform(0, 2 * np.pi, (4, B))
            if kind == 'M2':
                dr['dr'] = g.standard_normal((B, 16)); dr['dt'] = g.standard_normal((B, 16))
            if kind == 'M6':
                dr['pn'] = g.standard_normal((B, 16, 16))
        Ym = forward(b['psi'], b['phi'], b['alpha'], b['noise'], kind, sev, dr)
        Ys = SS.generate(b, kind, sev)
        rel = np.abs(Ym - Ys).max() / np.abs(Ys).max()
        extra = ''
        if kind == 'M5':
            nlev = len(np.unique(np.round(Ym.real / np.abs(Ym.real).max(), 9)))
            extra = f'; distinct Re levels {nlev} (2^b = {2 ** sev})'
        print(f'   {kind:5s} sev {sev!s:5s}: max|Y_mine - Y_stress| / max|Y| = {rel:.2e}{extra}')

print('(1b) M4 quantised codebooks: my rounding vs stress_sim')
for bits in [3, 2, 1]:
    print(f'   b={bits}: W max|diff| {np.abs(qphase(S.W_IDEAL, bits) - SS.quantise_codebook(S.W_IDEAL, bits)).max():.2e}, '
          f'F max|diff| {np.abs(qphase(S.F_IDEAL, bits) - SS.quantise_codebook(S.F_IDEAL, bits)).max():.2e}')

print('(1c) M8 far-field limit of my near-field steering (r = 1e8 m): max|a_nf - a_ff| = '
      f'{np.abs(steer_nf(np.array([[0.3, 1.2, 2.5]]), 1e8) - steer(np.cos(np.array([[0.3, 1.2, 2.5]])), np.arange(16.)[None])).max():.2e}')

print('(2) beamspace peak / RMS, ratio = max(|Re|,|Im|) / sqrt(mean|Y|^2 / 2) per scene, impairment-free L=3, 2000 scenes (seed 4243)')
for snr in [0, 15]:
    bb = fresh_bank(2000, snr, 4243 + snr); Y = fresh_Y(bb, 'clean', None)
    r = np.maximum(np.abs(Y.real), np.abs(Y.imag)).max((1, 2)) / np.sqrt((np.abs(Y) ** 2).mean((1, 2)) / 2)
    print(f'   SNR {snr:2d}: median {np.median(r):.2f}, IQR [{np.percentile(r, 25):.2f}, {np.percentile(r, 75):.2f}]; '
          f'fraction of real components clipped by the 3-RMS AGC: {np.mean(np.r_[np.abs(Y.real).ravel(), np.abs(Y.imag).ravel()] > np.repeat(3 * np.sqrt((np.abs(Y) ** 2).mean((1, 2)) / 2), 512)):.4f}')

print('(3) coupling convention: Hermitian Toeplitz instead of complex-symmetric (reciprocal), rho 0.3: relative change of Y')
g = SS.imp_rng(b['seed'], 'M1'); t = g.uniform(0, 2 * np.pi, (4, B))
Ysym = SS.generate(b, 'M1', 0.3)
CrH = np.array([scipy.linalg.toeplitz(np.r_[1, 0.3 * np.exp(1j * a), 0.15 * np.exp(1j * c), np.zeros(13)]) for a, c in zip(t[0], t[1])])
CtH = np.array([scipy.linalg.toeplitz(np.r_[1, 0.3 * np.exp(1j * a), 0.15 * np.exp(1j * c), np.zeros(13)]) for a, c in zip(t[2], t[3])])
cps, cph = np.cos(b['psi']), np.cos(b['phi'])
H = 16.0 * np.einsum('bl,bln,blm->bnm', b['alpha'], steer(cps, np.tile(np.arange(16.), (B, 1))), steer(cph, np.tile(np.arange(16.), (B, 1))).conj())
YH = S.W_IDEAL.conj().T[None] @ (CrH @ H @ np.transpose(CtH, (0, 2, 1))) @ S.F_IDEAL[None] + b['noise']
print(f'   ||Y_herm - Y_sym|| / ||Y_sym - noise|| = {np.linalg.norm(YH - Ysym) / np.linalg.norm(Ysym - b["noise"]):.3f}')
